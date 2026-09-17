"""Train and compare traffic-signal color classifiers from crop folders."""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("mobilenet_v3_small", "efficientnet_b0")


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
    parser.add_argument(
        "--data", type=Path, required=True,
        help="train/<class>, val/<class>, 선택적 test/<class> 구조의 crop 데이터셋",
    )
    parser.add_argument(
        "--model", choices=[*MODEL_NAMES, "both"], default="both",
        help="학습할 백본. both는 같은 데이터로 두 후보를 순서대로 비교",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/traffic_light_classifier")
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--imgsz", type=positive_int, default=224)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=positive_float, default=1e-4)
    parser.add_argument("--patience", type=nonnegative_int, default=7)
    parser.add_argument("--workers", type=nonnegative_int, default=4)
    parser.add_argument("--device", default="0", help="GPU 번호(0), cuda:0 또는 cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="ImageNet 사전학습 가중치 미사용")
    parser.add_argument("--no-class-weights", action="store_true", help="클래스 불균형 보정 미사용")
    parser.add_argument("--amp", action="store_true", help="CUDA mixed precision 학습")
    return parser.parse_args(argv)


def torch_device(value):
    if value == "cpu":
        return "cpu"
    if value.isdecimal():
        return f"cuda:{value}"
    if value.startswith("cuda:") and value[5:].isdecimal():
        return value
    raise ValueError("--device는 cpu, 0 또는 cuda:0 형식이어야 합니다.")


def model_names(value):
    return list(MODEL_NAMES) if value == "both" else [value]


def set_seed(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SquarePad:
    """Pad a PIL image to a square without stretching the signal shape."""

    def __init__(self, functional):
        self.functional = functional

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return self.functional.pad(
            image, [left, top, side - width - left, side - height - top], fill=0
        )


def build_transforms(imgsz, transforms, functional):
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    common = [SquarePad(functional), transforms.Resize((imgsz, imgsz))]
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
        transforms.RandomHorizontalFlip(),
        *common,
        transforms.ToTensor(),
        normalization,
    ])
    evaluation_transform = transforms.Compose([
        *common,
        transforms.ToTensor(),
        normalization,
    ])
    return train_transform, evaluation_transform


def build_model(name, class_count, pretrained, torch, models):
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {name}")
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(input_features, class_count)
    return model


def make_datasets(data_root, imgsz, datasets, transforms, functional):
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise ValueError("--data 아래에 train/<class>와 val/<class> 폴더가 필요합니다.")
    train_transform, evaluation_transform = build_transforms(imgsz, transforms, functional)
    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_root, transform=evaluation_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            f"train/val 클래스 폴더가 다릅니다: {train_dataset.classes} / {val_dataset.classes}"
        )
    test_root = data_root / "test"
    test_dataset = datasets.ImageFolder(test_root, transform=evaluation_transform) if test_root.is_dir() else None
    if test_dataset is not None and test_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError(f"test 클래스 폴더가 다릅니다: {test_dataset.classes}")
    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, args, torch, shuffle):
    generator = torch.Generator().manual_seed(args.seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch_device(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
        generator=generator,
    )


def balanced_weights(dataset, torch):
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes)).float()
    if (counts == 0).any():
        raise ValueError(f"학습 이미지가 없는 클래스가 있습니다: {dataset.classes}")
    return counts.sum() / (len(counts) * counts)


def run_epoch(model, loader, criterion, device, torch, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    confusion = None
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.startswith("cuda"))
        targets = targets.to(device, non_blocking=device.startswith("cuda"))
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        predictions = logits.argmax(dim=1)
        if confusion is None:
            class_count = logits.shape[1]
            confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
        encoded = targets.detach().cpu() * confusion.shape[0] + predictions.detach().cpu()
        confusion += torch.bincount(encoded, minlength=confusion.numel()).reshape(confusion.shape)
        total_items += targets.size(0)
        total_loss += loss.detach().item() * targets.size(0)
        total_correct += (predictions == targets).sum().item()
    confusion = confusion if confusion is not None else torch.zeros((0, 0), dtype=torch.int64)
    recalls = []
    for class_index in range(confusion.shape[0]):
        actual_count = confusion[class_index].sum().item()
        recalls.append(
            confusion[class_index, class_index].item() / actual_count if actual_count else None
        )
    available_recalls = [value for value in recalls if value is not None]
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
        "macro_recall": sum(available_recalls) / len(available_recalls) if available_recalls else 0.0,
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
        "samples": total_items,
    }


def train_one(name, train_dataset, val_dataset, test_dataset, args, torch, models):
    device = torch_device(args.device)
    set_seed(args.seed, torch)
    model = build_model(name, len(train_dataset.classes), not args.no_pretrained, torch, models)
    model = model.to(device)
    weights = None if args.no_class_weights else balanced_weights(train_dataset, torch).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 3)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    train_loader = make_loader(train_dataset, args, torch, shuffle=True)
    val_loader = make_loader(val_dataset, args, torch, shuffle=False)
    model_output = args.output / name
    model_output.mkdir(parents=True, exist_ok=False)

    best_score = -1.0
    best_epoch = 0
    best_val_metrics = None
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, torch,
            optimizer=optimizer, scaler=scaler, amp=args.amp,
        )
        with torch.inference_mode():
            val_metrics = run_epoch(model, val_loader, criterion, device, torch, amp=args.amp)
        scheduler.step(val_metrics["macro_recall"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"{name} [{epoch:03d}/{args.epochs}] "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_recall={val_metrics['macro_recall']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["macro_recall"] > best_score:
            best_score = val_metrics["macro_recall"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": name,
                "class_names": train_dataset.classes,
                "classifier_imgsz": args.imgsz,
                "epoch": epoch,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_recall": val_metrics["macro_recall"],
                "imagenet_pretrained": not args.no_pretrained,
            }, model_output / "best.pt")
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(f"{name}: validation macro recall이 {args.patience} epoch 동안 개선되지 않아 종료")
            break

    (model_output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "model": name,
        "best_epoch": best_epoch,
        "selection_metric": "val_macro_recall",
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_macro_recall": best_val_metrics["macro_recall"],
        "best_val_per_class_recall": dict(zip(train_dataset.classes, best_val_metrics["per_class_recall"])),
        "best_val_confusion_matrix": best_val_metrics["confusion_matrix"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str((model_output / "best.pt").resolve()),
    }
    if test_dataset is not None:
        checkpoint = torch.load(model_output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loader = make_loader(test_dataset, args, torch, shuffle=False)
        with torch.inference_mode():
            result["test"] = run_epoch(model, test_loader, criterion, device, torch, amp=args.amp)
    (model_output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    args.data = args.data.expanduser().resolve()
    output_base = args.output.expanduser().resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    args.output = output_base / run_name
    device = torch_device(args.device)
    if args.amp and not device.startswith("cuda"):
        raise ValueError("--amp는 CUDA에서만 사용할 수 있습니다.")
    args.output.mkdir(parents=True, exist_ok=False)

    try:
        import torch
        from torchvision import datasets, models, transforms
        from torchvision.transforms import functional
    except ImportError as error:
        raise RuntimeError("torch와 torchvision을 설치하세요.") from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. --device cpu로 확인하거나 GPU 설정을 점검하세요.")

    train_dataset, val_dataset, test_dataset = make_datasets(
        args.data, args.imgsz, datasets, transforms, functional
    )
    dataset_summary = {
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "test_images": len(test_dataset) if test_dataset is not None else 0,
    }
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    results = [
        train_one(name, train_dataset, val_dataset, test_dataset, args, torch, models)
        for name in model_names(args.model)
    ]
    comparison = {"dataset": dataset_summary, "models": results}
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: {args.output / 'comparison.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"오류: {error}")
