"""Fine-tune an Ultralytics YOLO detector for one pedestrian_signal class."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 유한한 값이 필요합니다.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="생성된 data.yaml 경로")
    parser.add_argument("--model", default="yolo26s.pt", help="사전학습 YOLO 가중치")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/traffic_light_detector")
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--imgsz", type=positive_int, default=960)
    parser.add_argument("--batch", type=positive_int, default=16)
    parser.add_argument("--device", default="0", help="GPU 번호(0), cuda:0 또는 cpu")
    parser.add_argument("--workers", type=nonnegative_int, default=4)
    parser.add_argument("--patience", type=nonnegative_int, default=7)
    parser.add_argument("--learning-rate", type=positive_float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="이미지를 RAM/디스크 캐시에 저장")
    return parser.parse_args(argv)


def run_name(now=None):
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def main(argv=None):
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise ValueError(f"data.yaml이 없습니다: {data}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("ultralytics를 설치하세요: pip install -r parts/traffic_light/requirements.txt") from error

    name = run_name()
    model = YOLO(args.model, task="detect")
    model.train(
        data=str(data),
        project=str(output),
        name=name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.learning_rate,
        seed=args.seed,
        cache=args.cache,
        exist_ok=False,
    )
    run_directory = Path(model.trainer.save_dir).resolve()
    best = run_directory / "weights" / "best.pt"
    summary = {
        "run_directory": str(run_directory),
        "best_checkpoint": str(best),
        "last_checkpoint": str(run_directory / "weights" / "last.pt"),
        "data": str(data),
        "base_model": args.model,
    }
    (run_directory / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"오류: {error}")
