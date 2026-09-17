"""Build an Ultralytics YOLO pedestrian-signal dataset from LabelMe JSON."""

import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "test"}


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("0 이상의 유한한 값이 필요합니다.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="LabelMe 이미지/JSON 루트")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "datasets/pedestrian_signal_yolo",
        help="생성할 Ultralytics 데이터셋(빈 폴더)",
    )
    parser.add_argument(
        "--signal-labels", nargs="+", default=["R_Signal", "G_Signal"],
        help="pedestrian_signal 하나로 합칠 원본 LabelMe 라벨",
    )
    parser.add_argument(
        "--include-crosswalk", action="store_true",
        help="Zebra_Cross를 class 1 crosswalk로 추가",
    )
    parser.add_argument(
        "--split-map", nargs="+", required=True,
        help="최상위폴더=train|val|test 매핑",
    )
    parser.add_argument(
        "--negative-ratio", type=nonnegative_float, default=0.20,
        help="신호등 이미지 수 대비 background 이미지 비율",
    )
    parser.add_argument(
        "--image-mode", choices=["symlink", "copy"], default="symlink",
        help="원본 이미지를 링크하거나 복사하는 방식",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def parse_mapping(values):
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"매핑은 폴더=train|val|test 형식이어야 합니다: {value}")
        folder, split = (item.strip() for item in value.rsplit("=", 1))
        if not folder or split not in SPLIT_NAMES:
            raise ValueError(f"잘못된 split 매핑입니다: {value}")
        if folder in mapping:
            raise ValueError(f"중복된 폴더 매핑입니다: {folder}")
        mapping[folder] = split
    return mapping


def ensure_empty_output(path):
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"출력 폴더가 비어 있지 않습니다: {path}\n"
            "원본 보호를 위해 중단했습니다. 새 --output 경로를 사용하세요."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_labelme(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"잘못된 LabelMe JSON입니다: {path}") from error


def signal_boxes(payload, signal_labels, label_path, include_crosswalk=False):
    try:
        width = float(payload["imageWidth"])
        height = float(payload["imageHeight"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"imageWidth/imageHeight가 잘못되었습니다: {label_path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"이미지 크기는 양수여야 합니다: {label_path}")

    boxes = []
    ignored = []
    for shape_index, shape in enumerate(payload.get("shapes", []), start=1):
        label = str(shape.get("label", "")).strip()
        if label not in signal_labels and not (include_crosswalk and label == "Zebra_Cross"):
            if label:
                ignored.append(label)
            continue
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"대상 라벨이 rectangle이 아닙니다: {label_path} shape {shape_index}")
        points = shape.get("points", [])
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError(f"잘못된 rectangle 좌표입니다: {label_path} shape {shape_index}")
        try:
            (ax, ay), (bx, by) = points
            x1, x2 = sorted((max(0.0, float(ax)), min(width, float(bx))))
            y1, y2 = sorted((max(0.0, float(ay)), min(height, float(by))))
        except (TypeError, ValueError) as error:
            raise ValueError(f"잘못된 좌표입니다: {label_path} shape {shape_index}") from error
        x1, x2 = max(0.0, x1), min(width, x2)
        y1, y2 = max(0.0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((
            1 if label == "Zebra_Cross" else 0,
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ))
    return boxes, ignored


def destination_stem(relative_path):
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{relative_path.stem}_{digest}"


def discover_records(source, split_map, signal_labels, include_crosswalk=False):
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"원본 폴더가 없습니다: {source}")
    records = []
    ignored_counts = {}
    for image_path in sorted(
        path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(source)
        if not relative.parts or relative.parts[0] not in split_map:
            continue
        label_path = image_path.with_suffix(".json")
        if not label_path.is_file():
            continue
        boxes, ignored = signal_boxes(
            read_labelme(label_path), signal_labels, label_path, include_crosswalk
        )
        for label in ignored:
            ignored_counts[label] = ignored_counts.get(label, 0) + 1
        records.append({
            "source_image": image_path,
            "source_label": label_path,
            "relative": relative,
            "split": split_map[relative.parts[0]],
            "boxes": boxes,
        })
    if not records:
        raise ValueError("매핑된 폴더에서 이미지와 LabelMe JSON 쌍을 찾지 못했습니다.")
    return records, ignored_counts


def select_records(records, negative_ratio, seed):
    selected = []
    randomizer = random.Random(seed)
    for split in sorted(SPLIT_NAMES):
        split_records = [record for record in records if record["split"] == split]
        positives = [record for record in split_records if record["boxes"]]
        negatives = [record for record in split_records if not record["boxes"]]
        randomizer.shuffle(negatives)
        negative_count = min(len(negatives), round(len(positives) * negative_ratio))
        selected.extend(positives)
        selected.extend(negatives[:negative_count])
    return sorted(selected, key=lambda record: record["relative"].as_posix())


def write_image(source, destination, mode):
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def write_dataset(records, output, image_mode):
    manifest = []
    split_counts = {}
    box_counts = {}
    class_box_counts = {}
    negative_counts = {}
    for record in records:
        split = record["split"]
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        stem = destination_stem(record["relative"])
        image_destination = image_dir / f"{stem}{record['source_image'].suffix.lower()}"
        label_destination = label_dir / f"{stem}.txt"
        write_image(record["source_image"], image_destination, image_mode)
        label_destination.write_text(
            "".join(
                f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n"
                for class_id, center_x, center_y, width, height in record["boxes"]
            ),
            encoding="utf-8",
        )
        split_counts[split] = split_counts.get(split, 0) + 1
        box_counts[split] = box_counts.get(split, 0) + len(record["boxes"])
        counts = class_box_counts.setdefault(split, {})
        for class_id, *_ in record["boxes"]:
            counts[class_id] = counts.get(class_id, 0) + 1
        if not record["boxes"]:
            negative_counts[split] = negative_counts.get(split, 0) + 1
        manifest.append({
            "image": str(image_destination.relative_to(output)),
            "label": str(label_destination.relative_to(output)),
            "source_image": str(record["source_image"]),
            "source_label": str(record["source_label"]),
            "split": split,
            "box_count": len(record["boxes"]),
        })
    return manifest, split_counts, box_counts, class_box_counts, negative_counts


def main(argv=None):
    args = parse_args(argv)
    if len(set(args.signal_labels)) != len(args.signal_labels):
        raise ValueError("--signal-labels에 중복이 있습니다.")
    split_map = parse_mapping(args.split_map)
    records, ignored_counts = discover_records(
        args.source, split_map, set(args.signal_labels), args.include_crosswalk
    )
    selected = select_records(records, args.negative_ratio, args.seed)
    if not any(record["split"] == "train" and record["boxes"] for record in selected):
        raise ValueError("train split에 신호등 정답 박스가 없습니다.")
    if not any(record["split"] == "val" and record["boxes"] for record in selected):
        raise ValueError("val split에 신호등 정답 박스가 없습니다.")
    if args.include_crosswalk:
        for split in ("train", "val"):
            present_classes = {
                box[0] for record in selected if record["split"] == split
                for box in record["boxes"]
            }
            if present_classes != {0, 1}:
                raise ValueError(f"{split} split에 두 클래스 정답 박스가 모두 필요합니다.")
    output = ensure_empty_output(args.output)
    manifest, split_counts, box_counts, class_box_counts, negative_counts = write_dataset(
        selected, output, args.image_mode
    )
    names = ["pedestrian_signal", "crosswalk"] if args.include_crosswalk else ["pedestrian_signal"]
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(names)),
        encoding="utf-8",
    )
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for item in manifest:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "source_pairs": len(records),
        "selected_images": len(selected),
        "split_image_counts": split_counts,
        "split_box_counts": box_counts,
        "split_class_box_counts": {
            split: {names[class_id]: count for class_id, count in counts.items()}
            for split, counts in class_box_counts.items()
        },
        "split_negative_counts": negative_counts,
        "signal_labels": args.signal_labels,
        "class_names": names,
        "ignored_label_counts": ignored_counts,
        "negative_ratio": args.negative_ratio,
        "image_mode": args.image_mode,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"YOLO 설정: {output / 'data.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"오류: {error}")
