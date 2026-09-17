import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "runtime"))


spec = importlib.util.spec_from_file_location(
    "traffic_pipeline", Path(__file__).parents[1] / "runtime" / "pipeline.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Boxes:
    xyxy = Array([[10, 20, 30, 50], [0, 0, 9, 9]])
    conf = Array([0.85, 0.9])
    cls = Array([9, 5])

    def cpu(self):
        return self


class PipelineTests(unittest.TestCase):
    def test_argument_defaults(self):
        args = runner.parse_args(["--source", "x.jpg"])
        self.assertEqual(args.detector, "yolo26s.pt")
        self.assertIn("traffic light", args.signal_classes)
        self.assertEqual(args.classifier_model, "efficientnet_b0")
        self.assertFalse(args.imagenet_pretrained)
        self.assertFalse(args.allow_untrained_predictions)
        self.assertEqual(args.detector_imgsz, 960)
        self.assertEqual(args.classifier_imgsz, 224)
        self.assertEqual(args.min_crop_size, 6)
        self.assertEqual(args.classifier_min_confidence, 0.60)
        self.assertEqual(args.class_names, ["red", "green", "unknown"])

    def test_probability_rejects_nonfinite(self):
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.assertRaises(runner.argparse.ArgumentTypeError):
                runner.probability(value)

    def test_device_validation(self):
        self.assertEqual(runner.torch_device("0"), "cuda:0")
        self.assertEqual(runner.torch_device("cuda:1"), "cuda:1")
        self.assertEqual(runner.torch_device("cpu"), "cpu")
        with self.assertRaises(ValueError):
            runner.torch_device("gpu")

    def test_expanded_box_is_clipped(self):
        self.assertEqual(runner.expanded_box([10, 10, 20, 20], 100, 80, 0.1), (9, 9, 21, 21))
        self.assertEqual(runner.expanded_box([-5, -5, 110, 90], 100, 80, 0.1), (0, 0, 100, 80))

    def test_detector_keeps_only_signal_classes(self):
        result = SimpleNamespace(boxes=Boxes(), names={9: "traffic light", 5: "bus"})
        detections = runner.extract_signal_detections(result)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "traffic light")

    def test_detector_class_filter_is_configurable(self):
        result = SimpleNamespace(boxes=Boxes(), names={9: "custom_signal", 5: "bus"})
        detections = runner.extract_signal_detections(result, ["custom_signal"])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "custom_signal")

    def test_crosswalk_is_extracted_separately(self):
        result = SimpleNamespace(boxes=Boxes(), names={9: "pedestrian_signal", 5: "crosswalk"})
        self.assertEqual(len(runner.extract_signal_detections(result)), 1)
        self.assertEqual(len(runner.extract_crosswalk_detections(result)), 1)

    def test_association_uses_direction_before_size(self):
        import numpy as np
        from association import associate

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        crossing = [{"xyxy": [300, 350, 700, 1000]}]
        signals = [
            {"xyxy": [490, 180, 510, 230]},
            {"xyxy": [640, 100, 760, 300]},
        ]
        with patch("association.estimate_vanishing_point", return_value=[500, 300]):
            decision = associate(frame, signals, crossing, None)
        self.assertEqual(decision["status"], "candidate")
        self.assertEqual(decision["signal_index"], 0)

    def test_one_signal_is_selected_without_crosswalk_geometry(self):
        import numpy as np
        from association import TemporalSelector, associate

        signals = [{"xyxy": [490, 180, 510, 230]}]
        with patch("association.estimate_vanishing_point") as estimate:
            decision = associate(np.zeros((1000, 1000, 3), dtype=np.uint8), signals, [], None)
        estimate.assert_not_called()
        self.assertEqual(decision["status"], "single_signal")
        self.assertEqual(decision["signal_index"], 0)
        self.assertEqual(decision["reason"], "crosswalk_relation_unverified")
        self.assertEqual(TemporalSelector(3).update(decision, signals, [])["status"],
                         "single_signal")

    def test_ambiguous_or_missing_geometry_is_unknown(self):
        import numpy as np
        from association import associate

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        crossing = [{"xyxy": [300, 350, 700, 1000]}]
        signals = [{"xyxy": [470, 180, 490, 230]}, {"xyxy": [510, 180, 530, 230]}]
        with patch("association.estimate_vanishing_point", return_value=None):
            decision = associate(frame, signals, crossing, None)
        self.assertEqual(decision["reason"], "vanishing_point_unavailable")
        with patch("association.estimate_vanishing_point", return_value=[500, 300]):
            decision = associate(frame, signals, crossing, None)
        self.assertEqual(decision["reason"], "ambiguous_signals")

    def test_larger_signal_resolves_close_geometry(self):
        import numpy as np
        from association import associate

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        crossing = [{"xyxy": [300, 350, 700, 1000]}]
        signals = [{"xyxy": [484, 180, 496, 210]}, {"xyxy": [500, 150, 540, 230]}]
        with patch("association.estimate_vanishing_point", return_value=[500, 300]):
            decision = associate(frame, signals, crossing, None)
        self.assertEqual(decision["status"], "candidate")
        self.assertEqual(decision["signal_index"], 1)

    def test_temporal_selector_requires_same_box_for_three_frames(self):
        from association import TemporalSelector

        selector = TemporalSelector(3)
        signals = [{"xyxy": [490, 180, 510, 230]}]
        crosswalks = [{"xyxy": [300, 350, 700, 1000]}]
        def candidate():
            return {"status": "candidate", "reason": None, "signal_index": 0,
                    "crosswalk_index": 0}
        first = selector.update(candidate(), signals, crosswalks)
        self.assertEqual(first["status"], "unknown")
        self.assertIsNone(first["signal_index"])
        self.assertEqual(selector.update(candidate(), signals, crosswalks)["status"], "unknown")
        self.assertEqual(selector.update(candidate(), signals, crosswalks)["status"], "matched")
        shifted = [{"xyxy": [0, 350, 200, 1000]}]
        self.assertEqual(selector.update(candidate(), signals, shifted)["stable_frames"], 1)
        selector.update({"status": "unknown", "signal_index": None}, signals, crosswalks)
        self.assertEqual(selector.update(candidate(), signals, crosswalks)["stable_frames"], 1)

    def test_vanishing_point_from_converging_edges(self):
        import cv2
        import numpy as np
        from association import estimate_vanishing_point

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        for bottom_x in (180, 220, 260, 740, 780, 820):
            cv2.line(frame, (500, 300), (bottom_x, 900), (255, 255, 255), 4)
        point = estimate_vanishing_point(frame, [100, 200, 900, 950], cv2)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 500, delta=20)
        self.assertAlmostEqual(point[1], 300, delta=20)

    def test_association_mode_returns_only_confirmed_signal(self):
        import numpy as np
        import torch
        from association import TemporalSelector

        args = runner.parse_args([
            "--source", "x.jpg", "--device", "cpu", "--associate-crosswalk",
            "--association-stable-frames", "1",
        ])
        signals = [
            {"class_name": "pedestrian_signal", "xyxy": [100, 100, 120, 150]},
            {"class_name": "pedestrian_signal", "xyxy": [500, 100, 520, 150]},
        ]
        crosswalks = [{"class_name": "crosswalk", "confidence": 0.9,
                       "xyxy": [100, 400, 900, 1000]}]
        captured = {}

        def fake_crops(frame, selected, args, cv2, torch_module):
            captured["selected"] = selected
            return torch.zeros((len(selected), 3, 8, 8)), selected

        timing = {"detector_preprocess": 0.0, "detector_inference": 0.0,
                  "detector_postprocess": 0.0, "detector_pipeline_wall": 0.0,
                  "crop_preprocess": 0.0}
        with patch.object(runner, "run_detector", return_value=(object(), timing)), \
             patch.object(runner, "extract_signal_detections", return_value=signals), \
             patch.object(runner, "extract_crosswalk_detections", return_value=crosswalks), \
             patch.object(runner, "associate", return_value={
                 "status": "candidate", "reason": None, "crosswalk_index": 0,
                 "signal_index": 1, "candidates": [], "vanishing_point": [500, 300],
             }), \
             patch.object(runner, "prepare_crops", side_effect=fake_crops), \
             patch.object(runner, "run_classifier", return_value=([{
                 "class_name": "green", "raw_class_name": "green",
                 "confidence": 0.99, "trained": True,
             }], {"classifier_input_transfer": 0.0, "classifier_inference": 0.0,
                 "classifier_postprocess": 0.0})):
            visible, found_crosswalks, association, _ = runner.run_pipeline(
                object(), object(), ["green", "red"], True,
                np.zeros((1000, 1000, 3), dtype=np.uint8), args, torch, object(),
                TemporalSelector(1),
            )
        self.assertEqual(len(captured["selected"]), 1)
        self.assertEqual(captured["selected"][0]["xyxy"], signals[1]["xyxy"])
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["xyxy"], signals[1]["xyxy"])
        self.assertEqual(association["status"], "matched")

    def test_single_signal_pipeline_classifies_without_crosswalk(self):
        import numpy as np
        import torch
        from association import TemporalSelector

        args = runner.parse_args(["--source", "x.jpg", "--device", "cpu", "--associate-crosswalk"])
        signal = {"class_name": "pedestrian_signal", "xyxy": [490, 180, 510, 230]}
        timing = {"detector_pipeline_wall": 0.0}
        with patch.object(runner, "run_detector", return_value=(object(), timing)), \
             patch.object(runner, "extract_signal_detections", return_value=[signal]), \
             patch.object(runner, "extract_crosswalk_detections", return_value=[]), \
             patch.object(runner, "prepare_crops", return_value=(torch.zeros(1, 3, 8, 8), [signal])), \
             patch.object(runner, "run_classifier", return_value=([{
                 "class_name": "green", "confidence": 0.99, "trained": True,
             }], {"classifier_input_transfer": 0.0, "classifier_inference": 0.0,
                 "classifier_postprocess": 0.0})):
            visible, crosswalks, decision, _ = runner.run_pipeline(
                object(), object(), ["green", "red"], True,
                np.zeros((1000, 1000, 3), dtype=np.uint8), args, torch, object(),
                TemporalSelector(3),
            )
        self.assertEqual(len(visible), 1)
        self.assertEqual(crosswalks, [])
        self.assertEqual(decision["status"], "single_signal")
        self.assertEqual(decision["color"], "green")

    def test_summary_separates_empty_classifier_frames(self):
        timing = {
            "detector_preprocess": 1.0, "detector_inference": 2.0,
            "detector_postprocess": 1.0, "detector_pipeline_wall": 4.0,
            "crop_preprocess": 0.5, "classifier_input_transfer": 0.1,
            "classifier_inference": 3.0,
            "classifier_postprocess": 0.2, "total_pipeline_wall": 7.7,
        }
        records = [
            {"timing_ms": timing, "detections": [{"classification": {}}]},
            {"timing_ms": {**timing, "classifier_inference": 0.0}, "detections": []},
        ]
        summary = runner.summarize(records)
        self.assertEqual(summary["processed_frames"], 2)
        self.assertEqual(summary["frames_with_classification"], 1)
        self.assertEqual(summary["mean_classifier_inference_ms_when_run"], 3.0)

    def test_low_neural_confidence_becomes_unknown(self):
        import torch

        class FixedModel:
            def __call__(self, batch):
                return torch.tensor([[1.0, 0.9, 0.8]])

        args = runner.parse_args([
            "--source", "x.jpg", "--device", "cpu", "--classifier-min-confidence", "0.6"
        ])
        predictions, _ = runner.run_classifier(
            FixedModel(), torch.zeros(1, 3, 8, 8), ["red", "green", "unknown"],
            True, args, torch,
        )
        self.assertEqual(predictions[0]["class_name"], "unknown")
        self.assertEqual(predictions[0]["raw_class_name"], "red")

    def test_untrained_predictions_can_be_shown_for_pipeline_check(self):
        import torch

        class FixedModel:
            def __call__(self, batch):
                return torch.tensor([[5.0, 1.0, 0.0]])

        args = runner.parse_args([
            "--source", "x.jpg", "--device", "cpu", "--allow-untrained-predictions"
        ])
        predictions, _ = runner.run_classifier(
            FixedModel(), torch.zeros(1, 3, 8, 8), ["red", "green", "unknown"],
            False, args, torch,
        )
        self.assertEqual(predictions[0]["class_name"], "red")
        self.assertFalse(predictions[0]["trained"])


if __name__ == "__main__":
    unittest.main()
