"""Compare historical and app-quality models on labeled, held-out app uploads."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from .app_quality_common import DEFAULT_CONFIG, new_directory, read_json, resolve, rows, sha256, write_json
from .app_quality_metrics import color_metrics, detection_metrics, latency_summary, paired_colors


def validate_holdout(records, manifest, excluded_hashes, cv2):
    if not records:
        raise ValueError("평가 manifest가 비어 있습니다.")
    seen = set()
    for row in records:
        if (row.get("split") != "test" or row.get("origin") != "app_upload"
                or row.get("unused_for_training") is not True or not row.get("session_id")):
            raise ValueError("test/app_upload/session_id/unused_for_training=true가 필요합니다.")
        path = Path(row["image"])
        path = path if path.is_absolute() else manifest.parent / path
        digest = sha256(path)
        if digest in excluded_hashes or digest in seen:
            raise ValueError(f"학습 데이터와 겹치거나 중복된 평가 이미지: {path}")
        seen.add(digest)
        if path.read_bytes()[:2] != b"\xff\xd8":
            raise ValueError(f"앱이 실제 전송한 JPEG가 필요합니다: {path}")
        frame = cv2.imread(str(path))
        if frame is None or max(frame.shape[:2]) > 960:
            raise ValueError(f"유효한 앱 전송 이미지(긴 변 <=960)가 아닙니다: {path}")
        height, width = frame.shape[:2]
        if row.get("width") != width or row.get("height") != height:
            raise ValueError(f"평가 라벨의 이미지 크기가 다릅니다: {path}")
        if not isinstance(row.get("annotations"), list):
            raise ValueError("누락 라벨은 허용하지 않습니다. 음성 이미지는 annotations=[]로 표시하세요.")
        for gt in row["annotations"]:
            if gt.get("class_name") not in {"pedestrian_signal", "crosswalk"}:
                raise ValueError(f"잘못된 평가 클래스: {gt}")
            box = gt.get("xyxy", [])
            if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                raise ValueError(f"잘못된 평가 bbox: {gt}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"이미지 범위 밖 평가 bbox: {gt}")
            if gt["class_name"] == "pedestrian_signal" and gt.get("color") not in {"red", "green", "unknown"}:
                raise ValueError(f"신호등 색상 정답이 필요합니다: {gt}")
        row["resolved_image"] = str(path.resolve())
        row["sha256"] = digest


def model_paths(config):
    root = resolve(config["runs_root"])
    result = {"historical": (resolve(config["detector"]["historical_weights"]),
                             resolve(config["classifier"]["historical_weights"]))}
    result["app"] = (root / "detector/app/weights/best.pt",
                     root / "classifier/app" / config["classifier"]["args"]["model"] / "best.pt")
    return result


def runtime_args(settings, weights):
    return argparse.Namespace(device=settings["device"], half=settings["half"],
        classifier_model="mobilenet_v3_small", classifier_weights=weights, class_names=["green", "red"],
        imagenet_pretrained=False, classifier_imgsz=settings["classifier_imgsz"],
        classifier_min_confidence=settings["classifier_conf"], crop_padding=settings["crop_padding"],
        min_crop_size=settings["min_crop_size"], allow_untrained_predictions=False)


def classify(frame, detections, classifier, names, args, cv2, torch):
    from ..runtime import pipeline
    batch, selected = pipeline.prepare_crops(frame, detections, args, cv2, torch)
    predictions, _ = pipeline.run_classifier(classifier, batch, names, True, args, torch)
    for detection in detections:
        detection["color"] = detection["raw_color"] = "crop_skipped"
    for detection, prediction in zip(selected, predictions):
        detection["color"] = prediction["class_name"]
        detection["raw_color"] = prediction["raw_class_name"]


def predict(detector, frame, settings, confidence):
    result = detector.predict(frame, imgsz=settings["imgsz"], conf=confidence,
        iou=settings["nms_iou"], max_det=settings["max_det"], device=settings["device"],
        quantize=16 if settings["half"] else 32, verbose=False, save=False)[0]
    return [{"class_name": result.names[int(cls)], "confidence": float(score), "xyxy": box}
            for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(),
                                      result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]


def evaluate(config):
    import cv2
    import torch
    from torchvision import models
    from ultralytics import YOLO
    from ..runtime import pipeline
    from .train_app_quality import verify_environment

    environment = verify_environment(config)
    settings = config["evaluation"]
    if settings["repeats"] < 1 or settings["warmup"] < 1:
        raise ValueError("repeats와 warmup은 1 이상이어야 합니다.")
    manifest = resolve(settings["manifest"])
    records = list(rows(manifest))
    prepared = resolve(config["prepared_root"])
    summary = read_json(prepared / "summary.json")
    if summary.get("complete") is not True or sha256(prepared / "sources.jsonl") != summary["sources_sha256"]:
        raise ValueError("완료된 전처리 source manifest가 필요합니다.")
    excluded = {r[k] for r in rows(prepared / "sources.jsonl") for k in ("source_sha256", "transport_sha256")}
    validate_holdout(records, manifest, excluded, cv2)
    paths = model_paths(config)
    provenance = {}
    for arm, pair in paths.items():
        provenance[arm] = {"detector": {"path": str(pair[0]), "sha256": sha256(pair[0])},
                           "classifier": {"path": str(pair[1]), "sha256": sha256(pair[1])}}
        for kind in ("detector", "classifier"):
            expected = (config[kind]["historical_sha256"] if arm == "historical" else
                        read_json(resolve(config["runs_root"]) / "audit" / f"{kind}_{arm}" / "complete.json")["sha256"])
            if provenance[arm][kind]["sha256"] != expected:
                raise ValueError(f"모델 SHA256 불일치: {arm}/{kind}")
    output = new_directory(resolve(settings["output"]))
    device = pipeline.torch_device(settings["device"])
    write_json(output / "provenance.json", {"config": config, "environment": environment,
        "manifest_sha256": sha256(manifest), "models": provenance,
        "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
        "torch_threads": torch.get_num_threads(), "browser": summary["browser"],
        "timing_scope": "batch=1; decode excluded; detector + all detected signal crops + classifier; CUDA synchronized; no association/network/UI"})
    loaded = {}
    first = cv2.imread(records[0]["resolved_image"])
    for arm, (detector_path, classifier_path) in paths.items():
        args = runtime_args(settings, classifier_path)
        classifier, names, _ = pipeline.load_classifier(args, torch, models)
        detector = YOLO(str(detector_path), task="detect")
        if detector.names != {0: "pedestrian_signal", 1: "crosswalk"}:
            raise ValueError(f"기대와 다른 검출 클래스: {detector.names}")
        loaded[arm] = detector, classifier, names, args
        for _ in range(settings["warmup"]):
            predictions = predict(detector, first, settings, min(settings["conf"], settings["crosswalk_conf"]))
            # Warm up the classifier even if the first frame has no detections.
            signals = [p for p in predictions if p["class_name"] == "pedestrian_signal" and p["confidence"] >= settings["conf"]]
            if not signals:
                signals = [{"xyxy": [0, 0, first.shape[1], first.shape[0]]}]
            classify(first, signals, classifier, names, args, cv2, torch)
    collected = {arm: [] for arm in paths}
    oracle = {arm: [] for arm in paths}
    timings = {arm: {"detector": [], "color_pipeline": [], "total": []} for arm in paths}
    with (output / "frames.jsonl").open("w") as stream:
        for index, row in enumerate(records):
            frame = cv2.imread(row["resolved_image"])
            order = list(paths)
            order = order[index % len(order):] + order[:index % len(order)]
            for arm in order:
                detector, classifier, names, args = loaded[arm]
                frame_timings = []
                for _ in range(settings["repeats"]):
                    pipeline.synchronize(torch, device)
                    started = time.perf_counter()
                    predicted = predict(detector, frame, settings, min(settings["conf"], settings["crosswalk_conf"]))
                    pipeline.synchronize(torch, device)
                    detected = time.perf_counter()
                    classify(frame, [p for p in predicted if p["class_name"] == "pedestrian_signal" and p["confidence"] >= settings["conf"]],
                             classifier, names, args, cv2, torch)
                    pipeline.synchronize(torch, device)
                    finished = time.perf_counter()
                    sample = {"detector": (detected - started) * 1000,
                              "color_pipeline": (finished - detected) * 1000, "total": (finished - started) * 1000}
                    frame_timings.append(sample)
                    for key, value in sample.items():
                        timings[arm][key].append(value)
                operating = predicted
                # A separate low-confidence pass measures AP. Never mix that
                # pass into operational latency, or classify its many weak boxes.
                ap_predictions = predict(detector, frame, settings, settings["ap_conf"])
                oracle_signals = [dict(g) for g in row["annotations"] if g["class_name"] == "pedestrian_signal"]
                expected = [g["color"] for g in oracle_signals]
                classify(frame, oracle_signals, classifier, names, args, cv2, torch)
                oracle[arm].extend(zip(expected, [p["color"] for p in oracle_signals]))
                entry = {"arm": arm, "image": row["resolved_image"], "sha256": row["sha256"],
                         "session_id": row["session_id"], "ground_truth": row["annotations"],
                         "predictions": operating, "ap_predictions": ap_predictions,
                         "oracle_predictions": oracle_signals, "timings_ms": frame_timings}
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                collected[arm].append(entry)
            if (index + 1) % 50 == 0:
                print(f"평가: {index + 1}/{len(records)}", flush=True)
    reports = {}
    for arm, frames in collected.items():
        report = {}
        for class_name, conf in (("pedestrian_signal", settings["conf"]), ("crosswalk", settings["crosswalk_conf"])):
            operational = detection_metrics(frames, class_name, conf, settings["match_iou"], settings["small_area_px"], calculate_ap=False)
            ap_frames = [{**f, "predictions": f["ap_predictions"]} for f in frames]
            ap = detection_metrics(ap_frames, class_name, conf, settings["match_iou"], settings["small_area_px"])
            operational.update(AP50=ap["AP50"], AP50_95=ap["AP50_95"])
            report[class_name] = operational
        report["color"] = paired_colors(frames, settings["conf"], settings["match_iou"])
        report["raw_color"] = paired_colors(frames, settings["conf"], settings["match_iou"], raw=True)
        report["oracle_color"] = color_metrics(oracle[arm])
        report["latency"] = {key: latency_summary(values) for key, values in timings[arm].items()}
        reports[arm] = report
    comparisons = compare_reports(reports)
    write_json(output / "report.json", {"frames": len(records), "sessions": dict(Counter(r["session_id"] for r in records)),
                                       "models": reports, "comparisons": comparisons})
    print(f"평가 완료: {output / 'report.json'}")


def compare_reports(reports):
    """Compute deltas using only the preserved historical and newly trained app models."""
    fields = [("pedestrian_signal", "small_recall"), ("crosswalk", "precision"), ("crosswalk", "recall"),
              ("crosswalk", "AP50"), ("crosswalk", "AP50_95"), ("color", "red_to_green_per_gt"),
              ("color", "green_to_red_per_gt")]
    differences = {}
    for group, metric in fields:
        old, new = reports["historical"][group][metric], reports["app"][group][metric]
        differences[f"{group}.{metric}"] = {"baseline": old, "app": new,
                                              "delta": new - old if old is not None and new is not None else None}
    for metric in ("mean_ms", "p50_ms", "p95_ms"):
        old, new = reports["historical"]["latency"]["total"][metric], reports["app"]["latency"]["total"][metric]
        differences[f"latency.total.{metric}"] = {"baseline": old, "app": new, "delta": new - old}
    return {"app_minus_historical": differences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    evaluate(read_json(args.config))


if __name__ == "__main__":
    main()
