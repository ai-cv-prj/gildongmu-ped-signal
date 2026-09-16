import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "benchmark_pipeline", Path(__file__).parents[1] / "benchmark_yolo_classifier.py"
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
