"""Detect traffic lights with YOLO and classify their active color."""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
SIGNAL_NAMES = {"traffic light", "pedestrian_signal", "pedestrian_red", "pedestrian_green"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수가 필요합니다.")
    return value


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("0~1 사이의 유한한 값을 입력하세요.")
    return value


def positive_probability(value):
    value = probability(value)
    if value == 0:
        raise argparse.ArgumentTypeError("0보다 크고 1 이하인 값을 입력하세요.")
    return value


def uint8_value(value):
    value = int(value)
    if not 0 <= value <= 255:
        raise argparse.ArgumentTypeError("0~255 사이의 정수가 필요합니다.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="사진/영상/폴더 경로 또는 웹캠 번호(0)")
    parser.add_argument("--detector", default="yolo26s.pt", help="YOLO 공식 모델명 또는 로컬 .pt 경로")
    parser.add_argument(
        "--signal-classes", nargs="+", default=sorted(SIGNAL_NAMES),
        help="YOLO 결과 중 신호등으로 받아들일 class 이름",
    )
    parser.add_argument(
        "--color-method",
        choices=["hsv", "neural"],
        default="hsv",
        help="색 판별 방식: 학습 없이 바로 쓰는 hsv 또는 체크포인트가 필요한 neural",
    )
    parser.add_argument(
        "--classifier-model",
        choices=["efficientnet_b0", "mobilenet_v3_small"],
        default="efficientnet_b0",
        help="색상 분류기 구조 (기본값: EfficientNet-B0)",
    )
    parser.add_argument("--classifier-weights", type=Path, help="선택한 분류기 구조로 학습한 체크포인트")
    parser.add_argument(
        "--imagenet-pretrained",
        action="store_true",
        help="커스텀 체크포인트 대신 torchvision ImageNet 기본 백본 가중치 사용",
    )
    parser.add_argument("--class-names", nargs="+", default=["red", "green", "unknown"])
    parser.add_argument("--device", default="0", help="GPU: 0 또는 cuda:0, CPU: cpu")
    parser.add_argument("--detector-imgsz", type=positive_int, default=960)
    parser.add_argument("--classifier-imgsz", type=positive_int, default=224)
    parser.add_argument("--conf", type=probability, default=0.25, help="YOLO 신호등 검출 임계값")
    parser.add_argument("--crop-padding", type=probability, default=0.10, help="박스 각 변의 추가 여백 비율")
    parser.add_argument(
        "--min-crop-size", type=positive_int, default=6,
        help="이보다 가로/세로가 작은 검출 crop은 색 판별에서 제외",
    )
    parser.add_argument(
        "--classifier-min-confidence", type=probability, default=0.60,
        help="신경망 최고 확률이 이 값보다 낮으면 unknown 처리",
    )
    parser.add_argument("--hsv-min-saturation", type=uint8_value, default=55)
    parser.add_argument("--hsv-min-value", type=uint8_value, default=80)
    parser.add_argument(
        "--hsv-min-color-ratio",
        type=positive_probability,
        default=0.01,
        help="crop 전체에서 유효 색 픽셀이 차지해야 하는 최소 비율",
    )
    parser.add_argument("--vid-stride", type=positive_int, default=1)
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--half", action="store_true", help="CUDA 모델을 FP16으로 실행")
    parser.add_argument("--show", action="store_true", help="결과 화면 표시 (GUI 필요)")
    parser.add_argument("--no-save-media", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/yolo_classifier_benchmark")
    return parser.parse_args(argv)


def resolve_source(value):
    if value.isdecimal():
        return int(value)
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"입력 경로가 없습니다: {path}")
    return path


def torch_device(value):
    if value == "cpu":
        return "cpu"
    if value.isdecimal():
        return f"cuda:{value}"
    if value.startswith("cuda:") and value[5:].isdecimal():
        return value
    raise ValueError("--device는 cpu, 0 또는 cuda:0 형식이어야 합니다.")


def synchronize(torch, device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)#cpu가 gpu의 모든 연산이 끝날때까지 대기


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def normalized_labels(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(index): str(name).strip().lower() for index, name in items}


def expanded_box(xyxy, width, height, padding):
    x1, y1, x2, y2 = map(float, xyxy)
    pad_x = max(0.0, x2 - x1) * padding
    pad_y = max(0.0, y2 - y1) * padding
    return (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(width, int(math.ceil(x2 + pad_x))),
        min(height, int(math.ceil(y2 + pad_y))),
    )


def extract_signal_detections(result, signal_names=SIGNAL_NAMES):
    labels = normalized_labels(result.names)
    signal_names = {str(name).strip().lower() for name in signal_names}
    detections = []
    if result.boxes is None:
        return detections
    boxes = result.boxes.cpu()
    for xyxy, score, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        class_id = int(class_id)
        name = labels[class_id]
        if name in signal_names:
            detections.append({
                "class_id": class_id,
                "class_name": name,
                "confidence": float(score),
                "xyxy": [float(value) for value in xyxy],
            })
    return detections


class FrameReader:
    def __init__(self, source, stride, cv2):
        self.source = source
        self.stride = stride
        self.cv2 = cv2
        self.capture = None
        self.fps = None
        if isinstance(source, int):
            self.kind = "webcam"
            self.capture = cv2.VideoCapture(source)
        elif source.is_dir():
            self.kind = "directory"
            self.files = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            if not self.files:
                raise ValueError(f"지원하는 이미지가 없는 폴더입니다: {source}")
        elif source.suffix.lower() in IMAGE_SUFFIXES:
            self.kind = "image"
            self.files = [source]
        else:
            self.kind = "video"
            self.capture = cv2.VideoCapture(str(source))
        if self.capture is not None:
            if not self.capture.isOpened():
                raise ValueError(f"영상 소스를 열 수 없습니다: {source}")
            fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            self.fps = fps if math.isfinite(fps) and fps > 0 else 10.0

    @property
    def output_fps(self):
        return max((self.fps or 1.0) / self.stride, 1.0)

    def __iter__(self):
        if self.kind in {"image", "directory"}:
            for index, path in enumerate(self.files, start=1):
                if (index - 1) % self.stride:
                    continue
                frame = self.cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
                yield index, None, path.name, frame
            return
        source_index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            source_index += 1
            if (source_index - 1) % self.stride:
                continue
            timestamp = None if self.kind == "webcam" else float(self.capture.get(self.cv2.CAP_PROP_POS_MSEC))
            yield source_index, timestamp, str(self.source), frame

    def close(self):
        if self.capture is not None:
            self.capture.release()


def load_classifier(args, torch, models):
    class_names = list(args.class_names)
    payload = None
    trained = args.classifier_weights is not None
    if trained:
        checkpoint = args.classifier_weights.expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"분류기 체크포인트가 없습니다: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and payload.get("model_name") not in (None, args.classifier_model):
            raise ValueError(
                f"체크포인트 모델은 {payload['model_name']}이지만 "
                f"--classifier-model은 {args.classifier_model}입니다."
            )
        if isinstance(payload, dict) and payload.get("classifier_imgsz") not in (
            None, args.classifier_imgsz
        ):
            raise ValueError(
                f"체크포인트 입력 크기는 {payload['classifier_imgsz']}이지만 "
                f"--classifier-imgsz는 {args.classifier_imgsz}입니다."
            )
        if isinstance(payload, dict) and "class_names" in payload:
            class_names = [str(name) for name in payload["class_names"]]
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("분류기 클래스명은 중복 없이 2개 이상이어야 합니다.")

    if trained and args.imagenet_pretrained:
        raise ValueError(
            "--classifier-weights와 --imagenet-pretrained는 동시에 사용하지 마세요."
        )

    if args.classifier_model == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if args.imagenet_pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if args.imagenet_pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(class_names))
    if trained:
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=True)
    device = torch_device(args.device)
    model = model.to(device).eval()
    if args.half:
        model = model.half()
    return model, class_names, trained


def run_detector(model, frame, args, torch):
    device = torch_device(args.device)
    synchronize(torch, device)
    started = time.perf_counter()
    result = model.predict(
        source=frame,
        imgsz=args.detector_imgsz,
        conf=args.conf,
        device=args.device,
        half=args.half,
        verbose=False,
        save=False,
        stream=False,
    )[0]
    synchronize(torch, device)
    wall_ms = (time.perf_counter() - started) * 1000
    return result, {
        "detector_preprocess": float(result.speed.get("preprocess", 0.0)),
        "detector_inference": float(result.speed.get("inference", 0.0)),
        "detector_postprocess": float(result.speed.get("postprocess", 0.0)),
        "detector_pipeline_wall": wall_ms,
    }


def prepare_crops(frame, detections, args, cv2, torch):
    height, width = frame.shape[:2]
    canvases, selected = [], []
    for detection in detections:
        x1, y1, x2, y2 = expanded_box(detection["xyxy"], width, height, args.crop_padding)
        if x2 - x1 < args.min_crop_size or y2 - y1 < args.min_crop_size:
            continue
        crop = frame[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape[:2]
        scale = min(args.classifier_imgsz / crop_width, args.classifier_imgsz / crop_height)
        resized_width = max(1, int(round(crop_width * scale)))
        resized_height = max(1, int(round(crop_height * scale)))
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = torch.zeros((3, args.classifier_imgsz, args.classifier_imgsz), dtype=torch.float32)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        top = (args.classifier_imgsz - resized_height) // 2
        left = (args.classifier_imgsz - resized_width) // 2
        canvas[:, top:top + resized_height, left:left + resized_width] = tensor
        canvases.append(canvas)
        selected.append(detection)
    if not canvases:
        return None, selected
    batch = torch.stack(canvases)
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std, selected


def classify_hsv_crop(crop, args, cv2):
    """Classify a cropped signal using bright, saturated HSV pixels.

    The returned confidence is a heuristic dominance score, not a calibrated
    neural-network probability. Extra ratios are kept for threshold tuning.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    valid = (saturation >= args.hsv_min_saturation) & (value >= args.hsv_min_value)
    candidate_masks = {
        "red": valid & ((hue <= 12) | (hue >= 165)),
        "green": valid & (hue >= 39) & (hue <= 95),
    }
    rejected_mask = valid & (hue >= 13) & (hue <= 38)
    pixel_count = max(int(hue.size), 1)
    ratios = {
        name: float(mask.sum()) / pixel_count
        for name, mask in candidate_masks.items()
    }
    rejected_ratio = float(rejected_mask.sum()) / pixel_count
    dominant_name = max(ratios, key=ratios.get)
    dominant_ratio = ratios[dominant_name]
    color_ratio = sum(ratios.values()) + rejected_ratio

    if dominant_ratio < args.hsv_min_color_ratio or rejected_ratio >= dominant_ratio:
        class_name = "unknown"
        if rejected_ratio >= dominant_ratio and rejected_ratio > 0:
            confidence = rejected_ratio / max(rejected_ratio + dominant_ratio, 1e-12)
        else:
            confidence = max(
                0.0,
                1.0 - dominant_ratio / max(args.hsv_min_color_ratio, 1e-12),
            )
    else:
        class_name = dominant_name
        confidence = dominant_ratio / max(color_ratio, 1e-12)
    return {
        "class_name": class_name,
        "confidence": float(confidence),
        "color_pixel_ratio": float(color_ratio),
        "color_ratios": ratios,
        "rejected_color_ratio": rejected_ratio,
        "method": "hsv",
    }


def run_hsv_classifier(frame, detections, args, cv2):
    height, width = frame.shape[:2]
    predictions, selected = [], []
    started = time.perf_counter()
    crops = []
    for detection in detections:
        x1, y1, x2, y2 = expanded_box(detection["xyxy"], width, height, args.crop_padding)
        if x2 - x1 < args.min_crop_size or y2 - y1 < args.min_crop_size:
            continue
        crops.append(frame[y1:y2, x1:x2])
        selected.append(detection)
    crop_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    for crop in crops:
        predictions.append(classify_hsv_crop(crop, args, cv2))
    inference_ms = (time.perf_counter() - started) * 1000
    return predictions, selected, {
        "crop_preprocess": crop_ms,
        "classifier_input_transfer": 0.0,
        "classifier_inference": inference_ms,
        "classifier_postprocess": 0.0,
    }


def run_classifier(model, batch, class_names, trained, args, torch):
    if batch is None:
        return [], {
            "classifier_input_transfer": 0.0,
            "classifier_inference": 0.0,
            "classifier_postprocess": 0.0,
        }
    device = torch_device(args.device)
    started = time.perf_counter()
    batch = batch.to(device, non_blocking=device.startswith("cuda"))
    if args.half:
        batch = batch.half()
    synchronize(torch, device)
    input_transfer_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
    synchronize(torch, device)
    inference_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    probabilities = logits.float().softmax(dim=1).cpu()
    scores, indices = probabilities.max(dim=1)
    predictions = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        raw_class_name = class_names[index] if trained else "untrained_timing_only"
        class_name = (
            raw_class_name
            if not trained or score >= args.classifier_min_confidence
            else "unknown"
        )
        predictions.append({
            "class_name": class_name,
            "raw_class_name": raw_class_name,
            "confidence": float(score) if trained else None,
        })
    postprocess_ms = (time.perf_counter() - started) * 1000
    return predictions, {
        "classifier_input_transfer": input_transfer_ms,
        "classifier_inference": inference_ms,
        "classifier_postprocess": postprocess_ms,
    }


def run_pipeline(detector, classifier, class_names, trained, frame, args, torch, cv2):
    pipeline_started = time.perf_counter()
    result, timing = run_detector(detector, frame, args, torch)
    detections = extract_signal_detections(result, args.signal_classes)

    if args.color_method == "hsv":
        predictions, classified_detections, classifier_timing = run_hsv_classifier(
            frame, detections, args, cv2
        )
        timing.update(classifier_timing)
    else:
        started = time.perf_counter()
        batch, classified_detections = prepare_crops(frame, detections, args, cv2, torch)
        timing["crop_preprocess"] = (time.perf_counter() - started) * 1000
        predictions, classifier_timing = run_classifier(
            classifier, batch, class_names, trained, args, torch
        )
        timing.update(classifier_timing)
    for detection, prediction in zip(classified_detections, predictions):
        detection["classification"] = prediction
    synchronize(torch, torch_device(args.device))
    timing["total_pipeline_wall"] = (time.perf_counter() - pipeline_started) * 1000
    return detections, timing


def draw_result(frame, detections, timing, cv2):
    canvas = frame.copy()
    title = (
        f"YOLO {timing['detector_pipeline_wall']:.1f}ms + "
        f"classifier {timing['classifier_inference']:.1f}ms = "
        f"total {timing['total_pipeline_wall']:.1f}ms"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (25, 25, 25), -1)
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["xyxy"])
        classification = detection.get("classification")
        if classification and classification["confidence"] is not None:
            label = f'{classification["class_name"]} {classification["confidence"]:.2f}'
        elif classification:
            label = "classifier: timing only"
        else:
            label = "signal: crop skipped"
        class_name = classification["class_name"] if classification else "unknown"
        color = {
            "red": (0, 0, 255),
            "green": (0, 200, 0),
            "unknown": (180, 180, 180),
        }.get(class_name, (0, 215, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, label, (max(0, x1), max(50, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return canvas


def summarize(records):
    keys = (
        "detector_preprocess", "detector_inference", "detector_postprocess",
        "detector_pipeline_wall", "crop_preprocess", "classifier_input_transfer",
        "classifier_inference", "classifier_postprocess", "total_pipeline_wall",
    )
    summary = {"processed_frames": len(records)}
    for key in keys:
        values = [record["timing_ms"][key] for record in records]
        summary[f"mean_{key}_ms"] = sum(values) / len(values) if values else None
    summary["detected_signals"] = sum(len(record["detections"]) for record in records)
    classified_frames = [record for record in records if any("classification" in item for item in record["detections"])]
    summary["frames_with_classification"] = len(classified_frames)
    if classified_frames:
        summary["mean_classifier_inference_ms_when_run"] = sum(
            record["timing_ms"]["classifier_inference"] for record in classified_frames
        ) / len(classified_frames)
    else:
        summary["mean_classifier_inference_ms_when_run"] = None
    total = summary["mean_total_pipeline_wall_ms"]
    summary["estimated_pipeline_fps"] = 1000 / total if total else None
    return summary


def main(argv=None):
    args = parse_args(argv)
    args.signal_classes = sorted({name.strip().lower() for name in args.signal_classes if name.strip()})
    if not args.signal_classes:
        raise ValueError("--signal-classes를 하나 이상 입력하세요.")
    source = resolve_source(args.source)
    device = torch_device(args.device)
    if args.color_method == "hsv" and (
        args.classifier_weights is not None or args.imagenet_pretrained
    ):
        raise ValueError(
            "분류기 가중치를 사용할 때는 --color-method neural을 함께 지정하세요."
        )
    if args.half and device == "cpu":
        raise ValueError("--half는 CUDA 장치에서만 사용하세요.")
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Python 가상환경을 활성화하고 requirements를 설치하세요.") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. scripts/check_gpu.py를 확인하세요.")

    print(f"YOLO 로딩: {args.detector}")
    detector = YOLO(args.detector, task="detect")
    if detector.task != "detect":
        raise ValueError("--detector에는 YOLO 객체검출 모델을 지정하세요.")
    detector_label_names = set(normalized_labels(detector.names).values())
    matched_signal_classes = detector_label_names.intersection(args.signal_classes)
    if not matched_signal_classes:
        raise ValueError(
            "YOLO 클래스와 --signal-classes가 하나도 일치하지 않습니다. "
            f"YOLO 클래스: {sorted(detector_label_names)}"
        )
    if matched_signal_classes == {"traffic light"}:
        print(
            "주의: COCO traffic light는 차량 신호등도 포함합니다. 최종 보행자 신호 판단에는 "
            "pedestrian_signal 단일 클래스로 파인튜닝한 검출기를 권장합니다."
        )
    if matched_signal_classes.intersection({"pedestrian_red", "pedestrian_green"}):
        print(
            "주의: YOLO가 이미 색 클래스를 출력하고 있습니다. 위치 검출은 pedestrian_signal "
            "단일 클래스로 합치고 색은 분류기에 맡기면 역할 중복을 줄일 수 있습니다."
        )
    if args.color_method == "neural":
        from torchvision import models
        classifier, class_names, trained = load_classifier(args, torch, models)
        if trained:
            print(f"분류기 가중치: {args.classifier_weights} / 클래스: {class_names}")
        else:
            weight_status = (
                "ImageNet 사전학습 백본 사용"
                if args.imagenet_pretrained
                else "가중치 없음"
            )
            print(
                f"분류기 {weight_status}: {args.classifier_model} 속도만 측정하며 "
                "3클래스 출력층은 미학습 상태라 색 예측은 사용하지 않습니다."
            )
    else:
        classifier, class_names, trained = None, ["red", "green", "unknown"], False
        print("색 판별: HSV 규칙 기반 (학습 가중치 불필요)")

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output.expanduser().resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": sys.version,
        "packages": {name: package_version(name) for name in (
            "torch", "torchvision", "ultralytics", "opencv-python"
        )},
        "torch_cuda": torch.version.cuda,
        "device": device,
        "detector_labels": normalized_labels(detector.names),
        "classifier_class_names": class_names,
        "classifier_trained": trained,
        "classifier_imagenet_pretrained": args.imagenet_pretrained,
        "color_method": args.color_method,
    }
    (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    reader = FrameReader(source, args.vid_stride, cv2)
    writer = None
    records = []
    status = "complete"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as log:
            for result_index, (source_index, timestamp_ms, source_name, frame) in enumerate(reader, start=1):
                if result_index == 1 and args.warmup:
                    print(f"검출기 워밍업 {args.warmup}회 (측정 제외)")
                    for _ in range(args.warmup):
                        run_detector(detector, frame, args, torch)
                    if args.color_method == "neural":
                        print(f"분류기 워밍업 {args.warmup}회 (측정 제외)")
                        dummy_batch = torch.zeros(
                            (1, 3, args.classifier_imgsz, args.classifier_imgsz), dtype=torch.float32
                        )
                        for _ in range(args.warmup):
                            run_classifier(classifier, dummy_batch, class_names, trained, args, torch)
                detections, timing = run_pipeline(
                    detector, classifier, class_names, trained, frame, args, torch, cv2
                )
                record = {
                    "result_index": result_index,
                    "source_frame_index": source_index,
                    "source_timestamp_ms": timestamp_ms,
                    "source": source_name,
                    "original_shape_hw": list(frame.shape[:2]),
                    "crop_batch_size": sum("classification" in item for item in detections),
                    "timing_ms": timing,
                    "detections": detections,
                }
                records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(
                    f"[{result_index}] signals={len(detections)} | "
                    f"YOLO={timing['detector_pipeline_wall']:.1f}ms "
                    f"crop={timing['crop_preprocess']:.1f}ms "
                    f"classifier={timing['classifier_inference']:.1f}ms "
                    f"total={timing['total_pipeline_wall']:.1f}ms"
                )

                if not args.no_save_media or args.show:
                    annotated = draw_result(frame, detections, timing, cv2)
                    if not args.no_save_media:
                        if reader.kind in {"image", "directory"}:
                            cv2.imwrite(str(output / f"{result_index:06d}_result.jpg"), annotated)
                        else:
                            if writer is None:
                                height, width = annotated.shape[:2]
                                writer = cv2.VideoWriter(
                                    str(output / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    reader.output_fps, (width, height),
                                )
                                if not writer.isOpened():
                                    raise RuntimeError("결과 영상 VideoWriter를 열 수 없습니다.")
                            writer.write(annotated)
                    if args.show:
                        cv2.imshow(f"YOLO -> {args.classifier_model}", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            status = "user_stopped"
                            break
                if args.max_frames and result_index >= args.max_frames:
                    status = "frame_limit"
                    break
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        reader.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        summary = summarize(records)
        summary["status"] = status
        summary["classifier_trained"] = trained
        summary["classifier_imagenet_pretrained"] = args.imagenet_pretrained
        summary["color_method"] = args.color_method
        summary["classifier_timing_scope"] = (
            "all HSV crop classifications in a frame"
            if args.color_method == "hsv"
            else "one batched forward pass for all detected signals in a frame"
        )
        summary["accuracy_available"] = False
        summary["accuracy_note"] = "정답 라벨 평가가 아니며 처리시간 측정 결과입니다."
        summary["peak_cuda_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.startswith("cuda") else None
        )
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"결과 폴더: {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"오류: {error}", file=sys.stderr)
        sys.exit(1)
