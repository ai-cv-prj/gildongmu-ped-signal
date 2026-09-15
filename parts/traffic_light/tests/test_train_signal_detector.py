import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "train_signal_detector", Path(__file__).parents[1] / "train_signal_detector.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainSignalDetectorTests(unittest.TestCase):
    def test_training_defaults(self):
        args = trainer.parse_args(["--data", "dataset/data.yaml"])
        self.assertEqual(args.model, "yolo26s.pt")
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.imgsz, 960)
        self.assertEqual(args.batch, 16)

    def test_invalid_numbers_are_rejected(self):
        with self.assertRaises(SystemExit):
            trainer.parse_args(["--data", "dataset/data.yaml", "--epochs", "0"])
        with self.assertRaises(SystemExit):
            trainer.parse_args(["--data", "dataset/data.yaml", "--learning-rate", "nan"])

    def test_main_uses_trainer_save_directory(self):
        class FakeYOLO:
            def __init__(self, model, task):
                self.model = model
                self.task = task

            def train(self, **kwargs):
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True)
                self.trainer = types.SimpleNamespace(save_dir=save_dir)
                return types.SimpleNamespace()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.yaml"
            data.write_text("names: {0: pedestrian_signal}\n", encoding="utf-8")
            fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                trainer.main([
                    "--data", str(data), "--output", str(root / "runs"),
                    "--epochs", "1", "--device", "cpu",
                ])
            summaries = list((root / "runs").glob("*/training_summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertTrue(summary["best_checkpoint"].endswith("weights/best.pt"))


if __name__ == "__main__":
    unittest.main()
