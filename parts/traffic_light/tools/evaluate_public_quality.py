"""Provisional public-test comparison using the SAME historical color classifier.

This is not a real-app holdout evaluation. It needs only the newly trained
detector; it does not train or substitute a new color classifier.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .app_quality_common import DEFAULT_CONFIG, new_directory, read_json, resolve, rows, sha256, write_json
from .app_quality_metrics import color_metrics, detection_metrics, latency_summary, paired_colors
from .evaluate_app_quality import classify, compare_reports, predict, runtime_args
from .prepare_classifier_crops import parse_labelme_labels, parse_yolo_labels


def test_records(config, cv2):
    """Use the frozen detector labels; attach color only after matching source boxes."""
    from .app_quality_metrics import match
    root = resolve(config["prepared_root"]) / "detector_app"
    result = []
    for row in rows(root / "manifest.jsonl"):
        if row["split"] != "test":
            continue
        image = root / row["image"]
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"이미지 디코딩 실패: {image}")
        height, width = frame.shape[:2]
        label = root / row["label"]
        if sha256(label) != row["label_sha256"]:
            raise ValueError(f"전처리 후 라벨이 변경되었습니다: {label}")
        names = ["pedestrian_signal", "crosswalk"]
        annotations = []
        for cid, cx, cy, bw, bh in parse_yolo_labels(label, names):
            annotations.append({"class_name": names[cid],
                "xyxy": [(cx - bw / 2) * width, (cy - bh / 2) * height,
                         (cx + bw / 2) * width, (cy + bh / 2) * height]})
        colors, _ = parse_labelme_labels(Path(row["source_label"]), ["red", "green"],
                                         {"R_Signal": "red", "G_Signal": "green"})
        source_boxes = [{"color": ["red", "green"][cid], "confidence": 1.0,
                         "xyxy": [(cx - bw / 2) * width, (cy - bh / 2) * height,
                                  (cx + bw / 2) * width, (cy + bh / 2) * height]}
                        for cid, cx, cy, bw, bh in colors]
        signals = [g for g in annotations if g["class_name"] == "pedestrian_signal"]
        matches = match(signals, source_boxes, .999)
        if len(matches) != len(signals) or len(source_boxes) != len(signals):
            raise ValueError(f"색상 정답과 보존된 검출 라벨이 다릅니다: {image}")
        for pi, gi in matches.items():
            signals[gi]["color"] = source_boxes[pi]["color"]
        result.append({"image": str(image), "image_sha256": sha256(image),
                       "source_label_sha256": sha256(row["source_label"]),
                       "ground_truth": annotations})
    if not result:
        raise ValueError("공개 test 이미지가 없습니다.")
    return result


def evaluate(config, output):
    import cv2
    import torch
    from torchvision import models
    from ultralytics import YOLO
    from ..runtime import pipeline
    from .train_app_quality import verify_environment

    environment = verify_environment(config)
    settings = config["evaluation"]
    device = pipeline.torch_device(settings["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 접근 권한을 확인하세요.")
    records = test_records(config, cv2)
    paths = {"historical": resolve(config["detector"]["historical_weights"]),
             "app": resolve(config["runs_root"]) / "detector/app/weights/best.pt"}
    app_audit = read_json(resolve(config["runs_root"]) / "audit/detector_app/complete.json")
    for arm, path in paths.items():
        expected = config["detector"]["historical_sha256"] if arm == "historical" else app_audit["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"검출기 해시 불일치: {path}")
    classifier_path = resolve(config["classifier"]["historical_weights"])
    if sha256(classifier_path) != config["classifier"]["historical_sha256"]:
        raise ValueError("기존 분류기 해시 불일치")
    args = runtime_args(settings, classifier_path)
    classifier, names, _ = pipeline.load_classifier(args, torch, models)
    detectors = {arm: YOLO(str(path), task="detect") for arm, path in paths.items()}
    output = new_directory(output)
    write_json(output / "provenance.json", {
        "scope": "public_dataset_test_not_real_app", "config": config, "environment": environment,
        "classifier_policy": "same historical classifier for both detectors; app classifier not trained",
        "classifier": {"path": str(classifier_path), "sha256": sha256(classifier_path)},
        "detectors": {a: {"path": str(p), "sha256": sha256(p)} for a, p in paths.items()},
        "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
        "timing_scope": "synchronized batch=1; decode excluded; detection + all detected signal crops + classifier",
        "data": records,
    })
    first = cv2.imread(records[0]["image"])
    for detector in detectors.values():
        if detector.names != {0: "pedestrian_signal", 1: "crosswalk"}:
            raise ValueError(f"클래스 매핑 불일치: {detector.names}")
        for _ in range(settings["warmup"]):
            predictions = predict(detector, first, settings, settings["conf"])
            signals = [p for p in predictions if p["class_name"] == "pedestrian_signal"]
            classify(first, signals or [{"xyxy": [0, 0, first.shape[1], first.shape[0]]}],
                     classifier, names, args, cv2, torch)
    collected = {a: [] for a in detectors}
    timings = {a: {"detector": [], "color_pipeline": [], "total": []} for a in detectors}
    oracle = []
    with (output / "frames.jsonl").open("w") as stream:
        for index, row in enumerate(records):
            frame = cv2.imread(row["image"])
            order = list(detectors) if index % 2 == 0 else list(reversed(detectors))
            for arm in order:
                samples = []
                for _ in range(settings["repeats"]):
                    pipeline.synchronize(torch, device)
                    start = time.perf_counter()
                    predictions = predict(detectors[arm], frame, settings, min(settings["conf"], settings["crosswalk_conf"]))
                    pipeline.synchronize(torch, device)
                    detected = time.perf_counter()
                    classify(frame, [p for p in predictions if p["class_name"] == "pedestrian_signal" and p["confidence"] >= settings["conf"]],
                             classifier, names, args, cv2, torch)
                    pipeline.synchronize(torch, device)
                    end = time.perf_counter()
                    sample = {"detector": (detected-start)*1000, "color_pipeline": (end-detected)*1000,
                              "total": (end-start)*1000}
                    samples.append(sample)
                    for key, value in sample.items():
                        timings[arm][key].append(value)
                ap_predictions = predict(detectors[arm], frame, settings, settings["ap_conf"])
                entry = {**row, "arm": arm, "predictions": predictions,
                         "ap_predictions": ap_predictions, "timings_ms": samples}
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                collected[arm].append(entry)
            gt_signals = [dict(g) for g in row["ground_truth"] if g["class_name"] == "pedestrian_signal"]
            actual = [g["color"] for g in gt_signals]
            classify(frame, gt_signals, classifier, names, args, cv2, torch)
            oracle.extend(zip(actual, [g["color"] for g in gt_signals]))
            if (index + 1) % 10 == 0:
                print(f"공개 test 평가: {index+1}/{len(records)}", flush=True)
    reports = {}
    for arm, frames in collected.items():
        report = {}
        for cls, conf in (("pedestrian_signal", settings["conf"]), ("crosswalk", settings["crosswalk_conf"])):
            operational = detection_metrics(frames, cls, conf, settings["match_iou"], settings["small_area_px"], calculate_ap=False)
            ap = detection_metrics([{**f, "predictions": f["ap_predictions"]} for f in frames], cls,
                                   conf, settings["match_iou"], settings["small_area_px"])
            operational.update(AP50=ap["AP50"], AP50_95=ap["AP50_95"])
            report[cls] = operational
        report["color"] = paired_colors(frames, settings["conf"], settings["match_iou"])
        report["raw_color"] = paired_colors(frames, settings["conf"], settings["match_iou"], raw=True)
        report["latency"] = {k: latency_summary(v) for k, v in timings[arm].items()}
        reports[arm] = report
    report = {"scope": "public_dataset_test_not_real_app", "frames": len(records),
              "gt_counts": dict(Counter(g["class_name"] for r in records for g in r["ground_truth"])),
              "classifier_policy": "fixed_historical", "oracle_color": color_metrics(oracle),
              "models": reports, "comparisons": compare_reports(reports)}
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"완료: {output / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(read_json(args.config), args.output.resolve())


if __name__ == "__main__":
    main()
