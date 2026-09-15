import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "prepare_classifier_crops", Path(__file__).parents[1] / "prepare_classifier_crops.py"
)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


class PrepareClassifierCropsTests(unittest.TestCase):
    def test_split_is_preserved(self):
        self.assertEqual(preparer.split_for(Path("train/a.jpg")), "train")
        self.assertEqual(preparer.split_for(Path("nested/a.jpg")), "unsplit")
        self.assertEqual(
            preparer.split_for(Path("bbox_1/a.jpg"), {"bbox_1": "train"}), "train"
        )

    def test_mapping_parser(self):
        self.assertEqual(
            preparer.parse_mapping(["R_Signal=red", "G_Signal=green"]),
            {"R_Signal": "red", "G_Signal": "green"},
        )
        with self.assertRaises(ValueError):
            preparer.parse_mapping(["bbox_1=invalid"], preparer.SPLIT_NAMES)

    def test_yolo_box_adds_padding_and_clips(self):
        annotation = (0, 0.5, 0.5, 0.2, 0.4)
        self.assertEqual(preparer.yolo_box(annotation, 100, 50, 0.1), (38, 13, 62, 37))
        edge = (0, 0.0, 0.0, 0.2, 0.2)
        self.assertEqual(preparer.yolo_box(edge, 100, 50, 0.1), (0, 0, 12, 6))

    def test_label_parser_maps_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "image.txt"
            label.write_text("0 0.5 0.5 0.2 0.4\n2 0.1 0.2 0.1 0.1\n", encoding="utf-8")
            annotations = preparer.parse_yolo_labels(label, ["red", "green", "unknown"])
            self.assertEqual([item[0] for item in annotations], [0, 2])

    def test_invalid_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "image.txt"
            label.write_text("3 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                preparer.parse_yolo_labels(label, ["red", "green", "unknown"])

    def test_labelme_rectangles_are_mapped_and_other_labels_ignored(self):
        payload = {
            "shapes": [
                {"label": "R_Signal", "shape_type": "rectangle", "points": [[10, 20], [30, 60]]},
                {"label": "G_Signal", "shape_type": "rectangle", "points": [[80, 90], [60, 50]]},
                {"label": "Zebra_Cross", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "image.json"
            label.write_text(json.dumps(payload), encoding="utf-8")
            annotations, ignored = preparer.parse_labelme_labels(
                label,
                ["red", "green"],
                {"R_Signal": "red", "G_Signal": "green"},
                image_width=100,
                image_height=100,
            )
        self.assertEqual([annotation[0] for annotation in annotations], [0, 1])
        self.assertEqual(annotations[0][1:], (0.2, 0.4, 0.2, 0.4))
        self.assertEqual(annotations[1][1:], (0.7, 0.7, 0.2, 0.4))
        self.assertEqual(ignored, ["Zebra_Cross"])


if __name__ == "__main__":
    unittest.main()
