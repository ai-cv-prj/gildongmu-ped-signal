"""Train on app-quality images from original pretrained weights, without resume."""

import argparse
from importlib.metadata import version
from pathlib import Path

from .app_quality_common import (DEFAULT_CONFIG, new_directory, read_json, resolve,
                                 sha256, write_json)


def verify_environment(config):
    actual = {name: version(name) for name in config["environment"]}
    if actual != config["environment"]:
        raise ValueError(f"환경 버전이 고정 설정과 다릅니다. 비교군 모두 같은 환경이 필요합니다: {actual}")
    return actual


def verify_initial(section):
    path = resolve(section["initial_weights"])
    if sha256(path) != section["initial_sha256"]:
        raise ValueError(f"원본 사전학습 가중치 SHA256 불일치: {path}")
    if path.resolve() == resolve(section["historical_weights"]).resolve():
        raise ValueError("기존 파인튜닝 체크포인트로 시작할 수 없습니다.")
    return path


def require_app_arm(arm):
    if arm != "app":
        raise ValueError("원본 화질 재학습은 제외되었습니다. --arm app만 사용할 수 있습니다.")


def detector_arguments(config, arm="app"):
    require_app_arm(arm)
    reference = resolve(config["detector"]["reference_args"])
    if sha256(reference) != config["detector"]["reference_snapshot_sha256"]:
        raise ValueError("기존 검출기 설정 사본의 SHA256이 다릅니다.")
    args = read_json(reference)
    # Replay the full recorded settings instead of inheriting today's library defaults.
    # save_dir is an internal, stale output path, not an accepted train override.
    args.pop("save_dir", None)
    data = resolve(config["prepared_root"]) / f"detector_{arm}/data.yaml"
    args.update(model=str(resolve(config["detector"]["initial_weights"])), data=str(data),
                project=str(resolve(config["runs_root"]) / "detector"), name=arm,
                resume=False, exist_ok=False, pretrained=True)
    return args


def train(config, task, arm="app"):
    require_app_arm(arm)
    environment = verify_environment(config)
    prepared = resolve(config["prepared_root"])
    summary = read_json(prepared / "summary.json")
    if not summary["complete"] or summary["config"] != config:
        raise ValueError("전처리 완료 당시 설정과 현재 설정이 다릅니다. 새 출력 경로로 준비하세요.")
    for kind in ("detector", "classifier"):
        source = resolve(config[kind]["source_dataset"]) / "manifest.jsonl"
        if sha256(source) != summary[f"{kind}_manifest_sha256"]:
            raise ValueError(f"기준 {kind} manifest가 전처리 후 변경되었습니다.")
    initial = verify_initial(config[task])
    output = resolve(config["runs_root"]) / task / arm
    if output.exists():
        raise FileExistsError(f"기존 실험 보호: {output}")
    # Audit is written before training, in its own non-reusable directory.
    audit = new_directory(resolve(config["runs_root"]) / "audit" / f"{task}_{arm}")
    write_json(audit / "provenance.json", {
        "task": task, "arm": arm, "initial_weights": str(initial),
        "initial_sha256": sha256(initial), "environment": environment,
        "config": config, "preparation_summary_sha256": sha256(prepared / "summary.json"),
    })
    if task == "detector":
        from ultralytics import YOLO
        args = detector_arguments(config, arm)
        write_json(audit / "args.json", args)
        model = YOLO(str(initial), task="detect")
        model.train(**args)
        best = Path(model.trainer.save_dir) / "weights/best.pt"
    else:
        import torch
        from torchvision import datasets, models, transforms
        from torchvision.transforms import functional
        from . import train_signal_classifier as trainer
        output = new_directory(output)
        args = argparse.Namespace(**config["classifier"]["args"], output=output,
                                  data=prepared / f"classifier_{arm}", initial_weights=initial)
        write_json(audit / "args.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        if args.amp and not torch.cuda.is_available():
            raise RuntimeError("기존 설정은 CUDA AMP를 사용합니다. CUDA 환경이 필요합니다.")
        train_set, val_set, test_set = trainer.make_datasets(args.data, args.imgsz, datasets, transforms, functional)
        result = trainer.train_one(args.model, train_set, val_set, test_set, args, torch, models)
        write_json(output / "comparison.json", {"models": [result]})
        best = Path(result["checkpoint"])
    write_json(audit / "complete.json", {"best": str(best), "sha256": sha256(best)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", choices=["detector", "classifier"], required=True)
    parser.add_argument("--arm", choices=["app"], default="app", help="앱 전송 화질만 학습 (기본값)")
    parser.add_argument("--print-only", action="store_true", help="학습 없이 설정 JSON만 출력")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if args.print_only:
        import json
        settings = detector_arguments(config, args.arm) if args.task == "detector" else config["classifier"]["args"]
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return
    train(config, args.task, args.arm)


if __name__ == "__main__":
    main()
