import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "prepare_yolo_signal_dataset",
    Path(__file__).parents[1] / "tools" / "prepare_yolo_signal_dataset.py",
)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


class PrepareYoloSignalDatasetTests(unittest.TestCase):
    def test_split_mapping_is_validated(self):
        self.assertEqual(
            preparer.parse_mapping(["bbox_1=train", "bbox_3=val"]),
            {"bbox_1": "train", "bbox_3": "val"},
        )
        with self.assertRaises(ValueError):
            preparer.parse_mapping(["bbox_1=unknown"])

    def test_signal_labels_are_merged_to_boxes(self):
        payload = {
            "imageWidth": 200,
            "imageHeight": 100,
            "shapes": [
                {"label": "R_Signal", "shape_type": "rectangle", "points": [[20, 10], [60, 50]]},
                {"label": "G_Signal", "shape_type": "rectangle", "points": [[100, 20], [140, 60]]},
                {"label": "Zebra_Cross", "shape_type": "rectangle", "points": [[0, 0], [10, 10]]},
            ],
        }
        boxes, ignored = preparer.signal_boxes(
            payload, {"R_Signal", "G_Signal"}, Path("sample.json")
        )
        self.assertEqual(boxes[0], (0, 0.2, 0.3, 0.2, 0.4))
        self.assertEqual(boxes[1], (0, 0.6, 0.4, 0.2, 0.4))
        self.assertEqual(ignored, ["Zebra_Cross"])

    def test_crosswalk_is_a_separate_class(self):
        payload = {
            "imageWidth": 200,
            "imageHeight": 100,
            "shapes": [
                {"label": "R_Signal", "shape_type": "rectangle", "points": [[20, 10], [60, 50]]},
                {"label": "Zebra_Cross", "shape_type": "rectangle", "points": [[0, 50], [200, 100]]},
            ],
        }
        boxes, ignored = preparer.signal_boxes(
            payload, {"R_Signal", "G_Signal"}, Path("sample.json"), include_crosswalk=True
        )
        self.assertEqual([box[0] for box in boxes], [0, 1])
        self.assertEqual(ignored, [])

    def test_negative_sampling_uses_positive_ratio_per_split(self):
        records = []
        for index in range(2):
            records.append({"split": "train", "boxes": [(0.5, 0.5, 0.1, 0.1)], "relative": Path(f"p{index}.jpg")})
        for index in range(5):
            records.append({"split": "train", "boxes": [], "relative": Path(f"n{index}.jpg")})
        selected = preparer.select_records(records, negative_ratio=0.5, seed=42)
        self.assertEqual(sum(bool(record["boxes"]) for record in selected), 2)
        self.assertEqual(sum(not record["boxes"] for record in selected), 1)

    def test_small_dataset_is_written_with_empty_negative_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            (source / "bbox_train").mkdir(parents=True)
            (source / "bbox_val").mkdir(parents=True)
            payload = {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [{
                    "label": "R_Signal", "shape_type": "rectangle",
                    "points": [[10, 10], [30, 50]],
                }],
            }
            for folder in ("bbox_train", "bbox_val"):
                (source / folder / "signal.jpg").write_bytes(b"not-decoded-by-converter")
                (source / folder / "signal.json").write_text(json.dumps(payload), encoding="utf-8")
            preparer.main([
                "--source", str(source),
                "--output", str(output),
                "--split-map", "bbox_train=train", "bbox_val=val",
                "--negative-ratio", "0",
            ])
            self.assertTrue((output / "data.yaml").is_file())
            self.assertEqual(len(list((output / "labels" / "train").glob("*.txt"))), 1)
            self.assertTrue(next((output / "images" / "train").iterdir()).is_symlink())

    def test_two_class_dataset_preserves_both_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            payload = {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {"label": "G_Signal", "shape_type": "rectangle", "points": [[10, 10], [30, 50]]},
                    {"label": "Zebra_Cross", "shape_type": "rectangle", "points": [[0, 60], [90, 90]]},
                ],
            }
            for folder in ("bbox_train", "bbox_val"):
                (source / folder).mkdir(parents=True)
                (source / folder / "scene.jpg").write_bytes(b"not-decoded-by-converter")
                (source / folder / "scene.json").write_text(json.dumps(payload), encoding="utf-8")
            preparer.main([
                "--source", str(source), "--output", str(output),
                "--split-map", "bbox_train=train", "bbox_val=val",
                "--include-crosswalk", "--negative-ratio", "0",
            ])
            label = next((output / "labels" / "train").glob("*.txt"))
            self.assertEqual([line[0] for line in label.read_text().splitlines()], ["0", "1"])
            self.assertIn("1: crosswalk", (output / "data.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
