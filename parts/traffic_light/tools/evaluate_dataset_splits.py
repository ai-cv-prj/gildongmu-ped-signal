"""Evaluate existing detectors on full prepared train/val/test splits; never train."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

from .app_quality_common import DEFAULT_CONFIG, new_directory, read_json, resolve, rows, sha256, write_json
from .app_quality_metrics import detection_metrics, ratio


def evaluate(config, output, splits, arms, batch, workers):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
    from .train_app_quality import verify_environment

    environment = verify_environment(config)
    settings = config["evaluation"]
    prepared = resolve(config["prepared_root"])
    root = prepared / "detector_app"
    summary = read_json(prepared / "summary.json")
    source_manifest = resolve(config["detector"]["source_dataset"]) / "manifest.jsonl"
    if not summary["complete"] or sha256(source_manifest) != summary["detector_manifest_sha256"]:
        raise ValueError("전처리 완료 상태/manifest 해시 불일치")
    counts = Counter()
    for row in rows(root / "manifest.jsonl"):
        counts[row["split"]] += 1
        if row["split"] in splits and sha256(root / row["label"]) != row["label_sha256"]:
            raise ValueError(f"전처리 후 라벨이 변경되었습니다: {row['label']}")
    paths = {"historical": resolve(config["detector"]["historical_weights"]),
             "app": resolve(config["runs_root"]) / "detector/app/weights/best.pt"}
    audit = read_json(resolve(config["runs_root"]) / "audit/detector_app/complete.json")
    expected = {"historical": config["detector"]["historical_sha256"], "app": audit["sha256"]}
    for arm in arms:
        if sha256(paths[arm]) != expected[arm]:
            raise ValueError(f"가중치 해시 불일치: {paths[arm]}")
    output = new_directory(output)
    validation = dict(data=str(root / "data.yaml"), imgsz=settings["imgsz"], batch=batch,
                      workers=workers, device=settings["device"], conf=settings["ap_conf"],
                      iou=settings["nms_iou"], max_det=settings["max_det"], quantize=32,
                      augment=False, rect=True, fraction=1.0, plots=False, save_json=False,
                      save_txt=False, verbose=False, project=str(output), exist_ok=False)
    report = {"scope": "full_prepared_dataset_splits_detection_only", "environment": environment,
              "config": config, "validation_args": validation, "split_image_counts": dict(counts),
              "source_manifest_sha256": summary["detector_manifest_sha256"],
              "prepared_manifest_sha256": sha256(root / "manifest.jsonl"),
              "weights": {arm: {"path": str(paths[arm]), "sha256": expected[arm]} for arm in arms},
              "notes": ["train measures fitting, val was used for model selection, test is held out",
                        "AP: Ultralytics official metric at conf=.001; operational: score-ordered matching",
                        "small: GT area <1024 pixels in decoded prepared image, before model letterbox",
                        "speed is batched detector only; excludes decoding, I/O, color classifier and app network",
                        "No color classifier evaluation or training in this run"], "results": {}}
    write_json(output / "report.json", report)
    for split in splits:
        report["results"][split] = {}
        for arm in arms:
            counters = {name: Counter() for name in ("pedestrian_signal", "crosswalk")}
            seen = 0
            with (output / f"{split}_{arm}_frames.jsonl").open("w", encoding="utf-8") as stream:
                class SplitValidator(DetectionValidator):
                    def update_metrics(self, preds, batch_data):
                        nonlocal seen
                        super().update_metrics(preds, batch_data)
                        for si, pred in enumerate(preds):
                            pbatch = self._prepare_batch(si, batch_data)
                            gt_boxes = ops.scale_boxes(pbatch["imgsz"], pbatch["bboxes"].clone(),
                                                       pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
                            keep = pred["conf"] >= min(settings["conf"], settings["crosswalk_conf"])
                            selected = {k: v[keep] for k, v in pred.items()}
                            selected = self.scale_preds(selected, pbatch)
                            gt = [{"class_name": self.names[int(cid)], "xyxy": box}
                                  for cid, box in zip(pbatch["cls"].cpu().tolist(), gt_boxes.cpu().tolist())]
                            predictions = [{"class_name": self.names[int(cid)], "xyxy": box, "confidence": conf}
                                           for cid, box, conf in zip(selected["cls"].cpu().tolist(),
                                               selected["bboxes"].cpu().tolist(), selected["conf"].cpu().tolist())]
                            frame = {"image": pbatch["im_file"], "ground_truth": gt, "predictions": predictions}
                            for name, counter in counters.items():
                                conf = settings["conf"] if name == "pedestrian_signal" else settings["crosswalk_conf"]
                                metric = detection_metrics([frame], name, conf, settings["match_iou"],
                                                           settings["small_area_px"], calculate_ap=False)
                                counter.update({k: metric[k] for k in ("tp", "fp", "fn", "gt", "small_gt", "small_tp")})
                            stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
                            seen += 1

                print(f"평가 시작: {split}/{arm}, {counts[split]} images", flush=True)
                start = time.perf_counter()
                model = YOLO(str(paths[arm]), task="detect")
                if model.names != {0: "pedestrian_signal", 1: "crosswalk"}:
                    raise ValueError(f"클래스 매핑 불일치: {model.names}")
                metrics = model.val(validator=SplitValidator, split=split, name=f"{split}_{arm}", **validation)
                if seen != counts[split]:
                    raise ValueError(f"평가 이미지 수 불일치: {seen} != {counts[split]}")
                operational = {}
                for name, counter in counters.items():
                    operational[name] = {**counter, "precision": ratio(counter["tp"], counter["tp"] + counter["fp"]),
                        "recall": ratio(counter["tp"], counter["gt"]),
                        "small_recall": ratio(counter["small_tp"], counter["small_gt"])}
                result = {"images": seen, "elapsed_seconds": time.perf_counter() - start,
                          "official": [{k: v.item() if hasattr(v, "item") else v for k, v in row.items()}
                                       for row in metrics.summary(decimals=8)],
                          "overall": {k: float(v) for k, v in metrics.results_dict.items()},
                          "operational": operational, "speed_ms_per_image": metrics.speed}
                report["results"][split][arm] = result
                write_json(output / f"{split}_{arm}.json", result)
                write_json(output / "report.json", report)
                print(json.dumps({"completed": f"{split}/{arm}", **result}, ensure_ascii=False), flush=True)
                del model, metrics
                gc.collect()
                torch.cuda.empty_cache()
    print(f"완료: {output / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["val", "test", "train"])
    parser.add_argument("--models", nargs="+", choices=("historical", "app"), default=["historical", "app"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluate(read_json(args.config), args.output.resolve(), args.splits, args.models, args.batch, args.workers)


if __name__ == "__main__":
    main()
