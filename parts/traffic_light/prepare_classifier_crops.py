"""Convert YOLO or LabelMe annotations into an ImageFolder crop dataset.

Expected input layout::

    dataset/images/train/example.jpg
    dataset/labels/train/example.txt

For YOLO labels, class ids follow ``--class-names``. For LabelMe JSON,
source labels are mapped with ``--label-map``. The output can be read
directly by ``torchvision.datasets.ImageFolder``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "test"}


def nonnegative_probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("0~1 사이의 유한한 값이 필요합니다.")
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="이미지 루트 폴더")
    parser.add_argument(
        "--labels", type=Path,
        help="YOLO labels 폴더. LabelMe에서 생략하면 JSON이 이미지 옆에 있다고 가정",
    )
    parser.add_argument(
        "--label-format", choices=["yolo", "labelme"], default="yolo",
        help="입력 라벨 형식",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/traffic_light_classifier",
        help="생성할 ImageFolder 데이터셋(빈 폴더)",
    )
    parser.add_argument(
        "--class-names", nargs="+", default=["red", "green", "unknown"],
        help="YOLO class id 순서",
    )
    parser.add_argument(
        "--label-map", nargs="+", default=["R_Signal=red", "G_Signal=green"],
        help="LabelMe 원본라벨=출력클래스 매핑",
    )
    parser.add_argument(
        "--split-map", nargs="*", default=[],
        help="최상위폴더=train|val|test 매핑. 미지정 시 train/val/test 폴더명을 사용",
    )
    parser.add_argument("--padding", type=nonnegative_probability, default=0.10)
    parser.add_argument("--min-size", type=positive_int, default=6, help="최소 crop 가로/세로 픽셀")
    parser.add_argument("--jpeg-quality", type=positive_int, default=95)
    parser.add_argument("--workers", type=positive_int, default=8, help="동시 이미지 처리 수")
    return parser.parse_args(argv)


def ensure_empty_output(path):
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"출력 폴더가 비어 있지 않습니다: {path}\n"
            "원본 보호를 위해 중단했습니다. 새 --output 경로를 사용하세요."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_images(root):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"images 폴더가 없습니다: {root}")
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"지원하는 이미지가 없습니다: {root}")
    return root, images


def parse_mapping(values, allowed_values=None):
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"매핑은 원본=대상 형식이어야 합니다: {value}")
        source, target = (item.strip() for item in value.rsplit("=", 1))
        if not source or not target:
            raise ValueError(f"빈 매핑 값은 사용할 수 없습니다: {value}")
        if allowed_values is not None and target not in allowed_values:
            raise ValueError(f"매핑 대상은 {sorted(allowed_values)} 중 하나여야 합니다: {value}")
        if source in mapping:
            raise ValueError(f"중복된 매핑 원본입니다: {source}")
        mapping[source] = target
    return mapping


def split_for(relative_path, split_map=None):
    top_folder = relative_path.parts[0]
    if split_map and top_folder in split_map:
        return split_map[top_folder]
    return top_folder.lower() if top_folder.lower() in SPLIT_NAMES else "unsplit"


def parse_yolo_labels(path, class_names):
    if not path.exists():
        return []
    annotations = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"잘못된 YOLO bbox 라벨({path}:{line_number}): 정확히 5개 열이 필요합니다."
            )
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = map(float, fields[1:5])
        except ValueError as error:
            raise ValueError(f"잘못된 YOLO 라벨({path}:{line_number}): 숫자가 아닙니다.") from error
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"class id {class_id}가 --class-names 범위를 벗어났습니다: {path}:{line_number}")
        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"좌표는 0~1 사이여야 합니다: {path}:{line_number}")
        if width == 0 or height == 0:
            continue
        annotations.append((class_id, center_x, center_y, width, height))
    return annotations


def parse_labelme_labels(path, class_names, label_map, image_width=None, image_height=None):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"잘못된 LabelMe JSON입니다: {path}") from error
    if image_width is None:
        image_width = payload.get("imageWidth")
    if image_height is None:
        image_height = payload.get("imageHeight")
    try:
        image_width = float(image_width)
        image_height = float(image_height)
    except (TypeError, ValueError) as error:
        raise ValueError(f"LabelMe 이미지 크기가 잘못되었습니다: {path}") from error
    if not all(math.isfinite(value) and value > 0 for value in (image_width, image_height)):
        raise ValueError(f"LabelMe 이미지 크기는 양수여야 합니다: {path}")
    annotations = []
    ignored_labels = []
    for shape_index, shape in enumerate(payload.get("shapes", []), start=1):
        source_label = str(shape.get("label", "")).strip()
        if source_label not in label_map:
            if source_label:
                ignored_labels.append(source_label)
            continue
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"rectangle이 아닌 LabelMe shape입니다: {path} shape {shape_index}")
        points = shape.get("points", [])
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError(f"잘못된 LabelMe rectangle 좌표입니다: {path} shape {shape_index}")
        try:
            (first_x, first_y), (second_x, second_y) = points
            x1, x2 = sorted((float(first_x), float(second_x)))
            y1, y2 = sorted((float(first_y), float(second_y)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"잘못된 LabelMe 좌표입니다: {path} shape {shape_index}") from error
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f"유한하지 않은 LabelMe 좌표입니다: {path} shape {shape_index}")
        x1, x2 = max(0.0, x1), min(float(image_width), x2)
        y1, y2 = max(0.0, y1), min(float(image_height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        target_name = label_map[source_label]
        class_id = class_names.index(target_name)
        annotations.append((
            class_id,
            ((x1 + x2) / 2) / image_width,
            ((y1 + y2) / 2) / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        ))
    return annotations, ignored_labels


def yolo_box(annotation, image_width, image_height, padding):
    _, center_x, center_y, width, height = annotation
    half_width = width * image_width * (0.5 + padding)
    half_height = height * image_height * (0.5 + padding)
    pixel_x = center_x * image_width
    pixel_y = center_y * image_height
    return (
        max(0, int(math.floor(pixel_x - half_width))),
        max(0, int(math.floor(pixel_y - half_height))),
        min(image_width, int(math.ceil(pixel_x + half_width))),
        min(image_height, int(math.ceil(pixel_y + half_height))),
    )


def label_path_for(image_path, images_root, labels_root, label_format="yolo"):
    suffix = ".json" if label_format == "labelme" else ".txt"
    return labels_root / image_path.relative_to(images_root).with_suffix(suffix)


def crop_name(relative_path, annotation_index):
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{relative_path.stem}_{digest}_{annotation_index:03d}.jpg"


def crop_annotated_image(job, output, class_names, padding, min_size, jpeg_quality, cv2):
    image_path, label_path, relative, split, annotations = job
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    image_height, image_width = image.shape[:2]
    records = []
    skipped_small = 0
    for annotation_index, annotation in enumerate(annotations, start=1):
        class_id = annotation[0]
        class_name = class_names[class_id]
        x1, y1, x2, y2 = yolo_box(annotation, image_width, image_height, padding)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            skipped_small += 1
            continue
        destination_dir = output / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / crop_name(relative, annotation_index)
        if not cv2.imwrite(
            str(destination), image[y1:y2, x1:x2],
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        ):
            raise OSError(f"crop 저장 실패: {destination}")
        records.append({
            "crop": str(destination.relative_to(output)),
            "source_image": str(image_path),
            "source_label": str(label_path),
            "annotation_index": annotation_index,
            "split": split,
            "class_id": class_id,
            "class_name": class_name,
            "xyxy": [x1, y1, x2, y2],
        })
    return records, skipped_small


def main(argv=None):
    args = parse_args(argv)
    if args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality는 1~100이어야 합니다.")
    if len(args.class_names) < 2 or len(set(args.class_names)) != len(args.class_names):
        raise ValueError("--class-names는 중복 없이 2개 이상이어야 합니다.")
    label_map = parse_mapping(args.label_map) if args.label_format == "labelme" else {}
    unknown_targets = set(label_map.values()).difference(args.class_names)
    if unknown_targets:
        raise ValueError(f"--label-map 대상이 --class-names에 없습니다: {sorted(unknown_targets)}")
    split_map = parse_mapping(args.split_map, SPLIT_NAMES)
    images_root, image_paths = find_images(args.images)
    if args.label_format == "yolo" and args.labels is None:
        raise ValueError("YOLO 형식에서는 --labels 폴더가 필요합니다.")
    labels_root = (
        args.labels.expanduser().resolve() if args.labels is not None else images_root
    )
    if not labels_root.is_dir():
        raise ValueError(f"labels 폴더가 없습니다: {labels_root}")
    output = ensure_empty_output(args.output)

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python을 설치하세요.") from error

    jobs = []
    missing_label_images = 0
    ignored_label_counts = {}
    for image_path in image_paths:
        relative = image_path.relative_to(images_root)
        label_path = label_path_for(
            image_path, images_root, labels_root, args.label_format
        )
        if not label_path.exists():
            missing_label_images += 1
            continue
        if args.label_format == "labelme":
            annotations, ignored_labels = parse_labelme_labels(
                label_path, args.class_names, label_map
            )
            for label in ignored_labels:
                ignored_label_counts[label] = ignored_label_counts.get(label, 0) + 1
        else:
            annotations = parse_yolo_labels(label_path, args.class_names)
        if not annotations:
            continue
        split = split_for(relative, split_map)
        jobs.append((image_path, label_path, relative, split, annotations))

    records = []
    skipped_small = 0
    class_counts = {name: 0 for name in args.class_names}
    split_counts = {}

    def process(job):
        return crop_annotated_image(
            job, output, args.class_names, args.padding, args.min_size,
            args.jpeg_quality, cv2,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for processed_index, (image_records, image_skipped) in enumerate(
            executor.map(process, jobs), start=1
        ):
            skipped_small += image_skipped
            for record in image_records:
                class_counts[record["class_name"]] += 1
                split = record["split"]
                split_counts[split] = split_counts.get(split, 0) + 1
                records.append(record)
            if processed_index % 1000 == 0:
                print(f"crop 진행: {processed_index}/{len(jobs)} 이미지")

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source_images": len(image_paths),
        "missing_label_images": missing_label_images,
        "written_crops": len(records),
        "skipped_small_crops": skipped_small,
        "class_counts": class_counts,
        "split_counts": split_counts,
        "class_names": args.class_names,
        "label_format": args.label_format,
        "label_map": label_map if args.label_format == "labelme" else None,
        "split_map": split_map,
        "ignored_label_counts": ignored_label_counts,
        "padding": args.padding,
        "min_size": args.min_size,
        "workers": args.workers,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if "train" not in split_counts or "val" not in split_counts:
        print("주의: train/val 폴더가 모두 필요합니다. 현재 split이 없으면 unsplit로 저장됩니다.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"오류: {error}")
