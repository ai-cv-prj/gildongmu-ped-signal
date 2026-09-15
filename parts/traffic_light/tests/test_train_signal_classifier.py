import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "train_signal_classifier", Path(__file__).parents[1] / "train_signal_classifier.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainSignalClassifierTests(unittest.TestCase):
    def test_defaults_compare_both_models(self):
        args = trainer.parse_args(["--data", "dataset"])
        self.assertEqual(args.model, "both")
        self.assertEqual(args.imgsz, 224)
        self.assertEqual(trainer.model_names(args.model), list(trainer.MODEL_NAMES))

    def test_device_validation(self):
        self.assertEqual(trainer.torch_device("0"), "cuda:0")
        self.assertEqual(trainer.torch_device("cpu"), "cpu")
        with self.assertRaises(ValueError):
            trainer.torch_device("gpu")


if __name__ == "__main__":
    unittest.main()
