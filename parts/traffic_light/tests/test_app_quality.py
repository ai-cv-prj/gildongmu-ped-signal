"""Small synthetic checks only: no production data, model inference or training."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from parts.traffic_light.tools import app_quality_common as common
from parts.traffic_light.tools import app_quality_metrics as metrics
from parts.traffic_light.tools import evaluate_app_quality as evaluator
from parts.traffic_light.tools import prepare_app_quality as preparer
from parts.traffic_light.tools import prepare_app_holdout as holdout_preparer
from parts.traffic_light.tools import train_app_quality as trainer


def signal(box, **kwargs):
    return {"class_name": "pedestrian_signal", "xyxy": box, **kwargs}


class GeometryAndMetricTests(unittest.TestCase):
    def test_resize_never_upscales_and_uses_js_rounding(self):
        self.assertEqual(common.dimensions(320, 240), (320, 240))
        self.assertEqual(common.dimensions(1920, 1081), (960, 541))
        self.assertEqual(common.dimensions(1081, 1920), (541, 960))

    def test_rounded_dimensions_scale_each_axis(self):
        self.assertEqual(common.scale_crop([10, 10, 100, 100], (1920, 1081), (960, 541)), (5, 5, 50, 51))

    def test_one_to_one_matching_counts_duplicate_as_fp(self):
        frames = [{"ground_truth": [signal([0, 0, 10, 10])], "predictions": [
            signal([0, 0, 10, 10], confidence=.9), signal([0, 0, 10, 10], confidence=.8)]}]
        result = metrics.detection_metrics(frames, "pedestrian_signal", .4, .5, 1024)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["small_recall"], 1)
        self.assertEqual(result["precision"], .5)

    def test_matching_uses_best_unmatched_gt(self):
        gt = [signal([0, 0, 10, 10]), signal([2, 0, 12, 10])]
        pred = [signal([0, 0, 10, 10], confidence=.9), signal([0, 0, 10, 10], confidence=.8)]
        self.assertEqual(len(metrics.match(gt, pred, .5)), 2)

    def test_small_recall_keeps_missed_gt_and_excludes_large_gt(self):
        frame = {"ground_truth": [signal([0, 0, 5, 5]), signal([30, 30, 90, 90])],
                 "predictions": [signal([30, 30, 90, 90], confidence=.9)]}
        result = metrics.detection_metrics([frame], "pedestrian_signal", .4, .5, 1024)
        self.assertEqual(result["recall"], .5)
        self.assertEqual((result["small_gt"], result["small_recall"]), (1, 0))

    def test_ap_confidence_order_and_absent_class(self):
        frame = {"ground_truth": [signal([0, 0, 5, 5])], "predictions": [
            signal([20, 20, 25, 25], confidence=.9), signal([0, 0, 5, 5], confidence=.8)]}
        self.assertAlmostEqual(metrics.average_precision([frame], "pedestrian_signal", .5), .5)
        self.assertIsNone(metrics.average_precision([frame], "crosswalk", .5))
        frame["predictions"] = []
        self.assertEqual(metrics.average_precision([frame], "pedestrian_signal", .5), 0)

    def test_color_misses_do_not_disappear_from_denominator(self):
        frame = {"ground_truth": [signal([0, 0, 5, 5], color="red"), signal([20, 20, 25, 25], color="red")],
                 "predictions": [signal([0, 0, 5, 5], confidence=.9, color="green", raw_color="green")]}
        result = metrics.paired_colors([frame], .4, .5)
        self.assertEqual(result["confusion"]["red"]["missed"], 1)
        self.assertEqual(result["red_to_green_per_gt"], .5)
        self.assertEqual(result["red_to_green_per_classified"], 1)
        self.assertIsNone(result["green_to_red_per_gt"])

    def test_unknown_raw_prediction_and_skipped_crop_are_distinct(self):
        frame = {"ground_truth": [signal([0, 0, 5, 5], color="red")],
                 "predictions": [signal([0, 0, 5, 5], confidence=.9, color="unknown", raw_color="green")]}
        self.assertEqual(metrics.paired_colors([frame], .4, .5)["red_to_green_count"], 0)
        self.assertEqual(metrics.paired_colors([frame], .4, .5, raw=True)["red_to_green_count"], 1)

    def test_only_app_training_uses_original_pretrained_weights(self):
        config = common.read_json(common.DEFAULT_CONFIG)
        app = trainer.detector_arguments(config)
        self.assertEqual(Path(app["model"]).name, "yolo26s.pt")
        self.assertTrue(app["data"].endswith("detector_app/data.yaml"))
        self.assertEqual(app["name"], "app")
        self.assertFalse(app["resume"])
        self.assertEqual(app["imgsz"], 960)
        with self.assertRaises(ValueError):
            trainer.detector_arguments(config, "original")
        # Reject before environment checks, file writes or any training imports.
        with self.assertRaises(ValueError):
            trainer.train({}, "classifier", "original")

    def test_evaluation_needs_only_historical_and_app_models(self):
        config = common.read_json(common.DEFAULT_CONFIG)
        paths = evaluator.model_paths(config)
        self.assertEqual(set(paths), {"historical", "app"})
        self.assertEqual(paths["historical"][0], common.resolve(config["detector"]["historical_weights"]))
        self.assertTrue(str(paths["app"][1]).endswith("classifier/app/mobilenet_v3_small/best.pt"))
        report = {"pedestrian_signal": {"small_recall": .5},
                  "crosswalk": dict.fromkeys(["precision", "recall", "AP50", "AP50_95"], .5),
                  "color": {"red_to_green_per_gt": None, "green_to_red_per_gt": .1},
                  "latency": {"total": dict.fromkeys(["mean_ms", "p50_ms", "p95_ms"], 20)}}
        app = copy.deepcopy(report)
        app["pedestrian_signal"]["small_recall"] = .75
        app["latency"]["total"]["mean_ms"] = 19
        comparisons = evaluator.compare_reports({"historical": report, "app": app})
        self.assertEqual(set(comparisons), {"app_minus_historical"})
        delta = comparisons["app_minus_historical"]
        self.assertEqual(delta["pedestrian_signal.small_recall"]["delta"], .25)
        self.assertEqual(delta["latency.total.mean_ms"]["delta"], -1)
        self.assertIsNone(delta["color.red_to_green_per_gt"]["delta"])

    def test_reject_output_reuse_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                common.new_directory(directory)
            with self.assertRaises(ValueError):
                common.child(directory, "../source.jpg")

    def test_initial_hash_mismatch_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.pt"
            path.write_bytes(b"synthetic")
            with self.assertRaises(ValueError):
                trainer.verify_initial({"initial_weights": str(path), "initial_sha256": "wrong",
                                        "historical_weights": str(path)})


class FakeEncoder:
    """Fixture only. Deliberately changes all full-frame pixels before crops."""
    calls = []

    def __init__(self, settings):
        self.settings = settings
        self.metadata = {"engine": "synthetic-test-only"}

    def __enter__(self):
        return self

    def encode(self, source):
        import cv2
        import numpy as np
        self.calls.append(source)
        frame = cv2.imread(str(source))
        old = (frame.shape[1], frame.shape[0])
        new = common.dimensions(*old)
        transformed = np.full((new[1], new[0], 3), 77, dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", transformed)
        assert ok
        return buffer.tobytes(), old, new

    def __exit__(self, *exc):
        pass


class PreparationTests(unittest.TestCase):
    def test_restart_archives_partial_run_and_protects_completed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (output / "full_frames").mkdir(parents=True)
            (output / "full_frames/frame.jpg").write_bytes(b"preserve partial bytes")
            (output / "sources.jsonl").write_text("partial manifest")
            preparer.preparation_output(output, restart=True)
            backup, = root.glob("output.interrupted_*")
            self.assertEqual((backup / "full_frames/frame.jpg").read_bytes(), b"preserve partial bytes")
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            (output / "summary.json").write_text('{"complete": true}')
            with self.assertRaises(ValueError):
                preparer.preparation_output(output, restart=True)
            with self.assertRaises(ValueError):
                preparer.preparation_output(root, restart=True)

    def make_fixture(self, root):
        import cv2
        import numpy as np
        dataset, crops = root / "detector", root / "crops"
        crops.mkdir()
        detector_rows, crop_rows = [], []
        for index, split in enumerate(("train", "val")):
            source = root / f"source_{split}.png"
            cv2.imwrite(str(source), np.full((1001, 1920, 3), index * 100, dtype=np.uint8))
            image = dataset / f"images/{split}/a.png"
            label = dataset / f"labels/{split}/a.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.symlink_to(source)
            label.write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.8 0.2 0.2\n" if index == 0 else "")
            detector_rows.append({"image": str(image.relative_to(dataset)), "label": str(label.relative_to(dataset)),
                                  "source_image": str(source), "split": split})
            crop_rows.append({"crop": f"{split}/red/a.jpg", "source_image": str(source), "split": split,
                              "class_name": "red", "xyxy": [40, 40, 44, 44]})
        (dataset / "data.yaml").write_text("path: ignored\ntrain: images/train\nval: images/val\nnames:\n  0: pedestrian_signal\n  1: crosswalk\n")
        for path, data in ((dataset / "manifest.jsonl", detector_rows), (crops / "manifest.jsonl", crop_rows)):
            path.write_text("".join(json.dumps(r) + "\n" for r in data))
        return {"detector": {"source_dataset": str(dataset)}, "classifier": {"source_dataset": str(crops)},
                "prepared_root": str(root / "output"), "transport": {"max_side": 960, "quality": .8}}

    def test_full_frame_transform_then_lossless_crop_and_preserved_labels(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_fixture(root)
            before = common.sha256(root / "source_train.png")
            FakeEncoder.calls = []
            preparer.prepare(config, encoder_factory=FakeEncoder)
            self.assertEqual(len(FakeEncoder.calls), 2)  # deduplicate detector/classifier sources
            self.assertEqual(common.sha256(root / "source_train.png"), before)
            output = root / "output"
            self.assertFalse((output / "detector_original").exists())
            self.assertFalse((output / "classifier_original").exists())
            self.assertEqual((output / "detector_app/labels/train/a.txt").read_text(),
                             (root / "detector/labels/train/a.txt").read_text())
            self.assertEqual((output / "detector_app/labels/val/a.txt").read_text(), "")
            crop = cv2.imread(str(output / "classifier_app/train/red/a.png"))
            self.assertTrue((crop == 77).all())
            self.assertEqual(crop.shape[:2], (3, 2))
            self.assertEqual(common.read_json(output / "summary.json")["tiny_crops_retained"]["app"], 2)
            self.assertEqual(len(list(common.rows(output / "classifier_app/manifest.jsonl"))), 2)
            with self.assertRaises(FileExistsError):
                preparer.prepare(config, encoder_factory=FakeEncoder)

    def test_conflicting_splits_stop_before_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_fixture(root)
            path = root / "crops/manifest.jsonl"
            data = list(common.rows(path))
            data[0]["split"] = "test"
            path.write_text("".join(json.dumps(r) + "\n" for r in data))
            with self.assertRaises(ValueError):
                preparer.prepare(config, encoder_factory=FakeEncoder)
            self.assertFalse((root / "output").exists())


class HoldoutTests(unittest.TestCase):
    def test_labelme_manifest_preserves_upload_bytes_and_rejects_missing_labels(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session1"
            session.mkdir()
            path = session / "frame.jpg"
            cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))
            before = common.sha256(path)
            with self.assertRaises(FileNotFoundError):
                holdout_preparer.build_manifest(root, ["session1"], root / "missing.jsonl")
            common.write_json(path.with_suffix(".json"), {"imageWidth": 60, "imageHeight": 40, "shapes": [
                {"label": "R_Signal", "shape_type": "rectangle", "points": [[10, 20], [2, 3]]},
                {"label": "Zebra_Cross", "shape_type": "rectangle", "points": [[20, 20], [50, 39]]}]})
            output = root / "manifest.jsonl"
            holdout_preparer.build_manifest(root, ["session1"], output)
            row = next(common.rows(output))
            self.assertEqual(row["annotations"][0], signal([2, 3, 10, 20], color="red"))
            self.assertEqual(row["annotations"][1]["class_name"], "crosswalk")
            self.assertEqual(common.sha256(path), before)
            with self.assertRaises(FileExistsError):
                holdout_preparer.build_manifest(root, ["session1"], output)

    def test_missing_labels_invalid_bounds_and_training_duplicates_rejected(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "frame.jpg"
            cv2.imwrite(str(path), np.zeros((40, 60, 3), dtype=np.uint8))
            row = {"image": "frame.jpg", "width": 60, "height": 40, "split": "test", "session_id": "heldout1",
                   "origin": "app_upload", "unused_for_training": True, "annotations": []}
            evaluator.validate_holdout([copy.deepcopy(row)], root / "manifest.jsonl", set(), cv2)
            with self.assertRaises(ValueError):
                evaluator.validate_holdout([copy.deepcopy(row)], root / "manifest.jsonl", {common.sha256(path)}, cv2)
            broken = copy.deepcopy(row)
            del broken["annotations"]
            with self.assertRaises(ValueError):
                evaluator.validate_holdout([broken], root / "manifest.jsonl", set(), cv2)
            broken = {**row, "annotations": [signal([0, 0, 61, 30], color="red")]}
            with self.assertRaises(ValueError):
                evaluator.validate_holdout([broken], root / "manifest.jsonl", set(), cv2)
            with self.assertRaises(ValueError):
                evaluator.validate_holdout([copy.deepcopy(row), copy.deepcopy(row)], root / "manifest.jsonl", set(), cv2)


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Playwright 미설치: 실제 browser smoke test 생략")
class BrowserSmokeTests(unittest.TestCase):
    def test_file_input_recycles_browser_without_reading_source_into_python(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, common.BrowserJPEG(
                {"max_side": 960, "quality": .8}, restart_every=1) as encoder:
            source = Path(directory) / "한글 image.png"
            pixels = np.random.default_rng(42).integers(0, 256, size=(121, 160, 3), dtype=np.uint8)
            cv2.imwrite(str(source), pixels)
            first_browser = encoder.browser
            with patch.object(Path, "read_bytes", side_effect=AssertionError("Must not buffer full source")):
                first, _, _ = encoder.encode(source)
                self.assertEqual(encoder.page.locator('#source').input_value(), "")
                second, _, _ = encoder.encode(source)
            self.assertFalse(first_browser.is_connected())
            self.assertTrue(encoder.browser.is_connected())
            self.assertEqual(first, second)
            self.assertEqual(encoder.metadata["input_transfer"], "local_file")

    def test_browser_jpeg_dimensions_and_small_image(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, common.BrowserJPEG({"max_side": 960, "quality": .8}) as encoder:
            for width, height in ((1920, 1081), (80, 60)):
                source = Path(directory) / f"{width}.png"
                cv2.imwrite(str(source), np.full((height, width, 3), (0, 0, 255), dtype=np.uint8))
                payload, old, new = encoder.encode(source)
                decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(old, (width, height))
                self.assertEqual((decoded.shape[1], decoded.shape[0]), common.dimensions(width, height))
                self.assertEqual(new, common.dimensions(width, height))
                self.assertGreater(float(decoded[:, :, 2].mean()), 240)


if __name__ == "__main__":
    unittest.main()
