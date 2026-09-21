"""Replay browser upload processing while preserving existing dataset membership."""

import argparse
from collections import Counter
from datetime import datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .app_quality_common import (BrowserJPEG, DEFAULT_CONFIG, child, new_directory,
                                 read_json, resolve, rows, scale_crop, sha256, write_json)


def preparation_output(path, restart=False):
    """Archive an interrupted run only on explicit request; never overwrite a completed one."""
    path = Path(path)
    if restart and path.exists():
        if (path.is_symlink() or (path / "summary.json").exists()
                or not (path / "sources.jsonl").is_file() or not (path / "full_frames").is_dir()):
            raise ValueError(f"확인된 미완료 전처리 폴더만 보관 후 재시작할 수 있습니다: {path}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.name}.interrupted_{stamp}_{uuid4().hex[:8]}")
        path.rename(backup)
        print(f"중단된 결과 보관: {backup}\n새 전처리는 처음부터 시작합니다.", flush=True)
    return new_directory(path)


def prepare(config, encoder_factory=BrowserJPEG, restart=False):
    import cv2
    import yaml

    if "environment" in config:
        from .train_app_quality import verify_environment
        verify_environment(config)

    dataset = resolve(config["detector"]["source_dataset"])
    crops = resolve(config["classifier"]["source_dataset"])
    detector_rows = list(rows(dataset / "manifest.jsonl"))
    crop_rows = list(rows(crops / "manifest.jsonl"))
    data_yaml = yaml.safe_load((dataset / "data.yaml").read_text())
    if data_yaml["names"] != {0: "pedestrian_signal", 1: "crosswalk"}:
        raise ValueError("2클래스 pedestrian_signal/crosswalk 데이터가 필요합니다.")
    # Validate membership before writing. Do not resample negatives or recreate splits.
    membership = {}
    for record in detector_rows + crop_rows:
        source = str(Path(record["source_image"]).resolve())
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"알 수 없는 split: {split}")
        if source in membership and membership[source] != split:
            raise ValueError(f"원본 이미지의 split이 충돌합니다: {source}")
        membership[source] = split
    if not detector_rows or not crop_rows:
        raise ValueError("기존 manifest가 비어 있습니다.")
    output = preparation_output(resolve(config["prepared_root"]), restart=restart)
    for split in sorted({r["split"] for r in crop_rows}):
        for name in sorted({r["class_name"] for r in crop_rows}):
            (output / "classifier_app" / split / name).mkdir(parents=True)

    transformed = {}
    content_splits = {}
    with encoder_factory(config["transport"]) as encoder, (output / "sources.jsonl").open("w") as audit:
        for index, (source, split) in enumerate(membership.items(), 1):
            digest = sha256(source)
            if digest in content_splits and content_splits[digest] != split:
                raise ValueError(f"동일 이미지 내용이 서로 다른 split에 있습니다: {source}")
            content_splits[digest] = split
            try:
                payload, original_size, size = encoder.encode(source)
            except (MemoryError, OSError) as error:
                if isinstance(error, OSError) and error.errno != 12:
                    raise
                raise RuntimeError(
                    f"{index}번째 이미지 처리 중 메모리 부족: {source}\n"
                    "다른 메모리 사용 작업을 종료한 뒤 --restart --browser-restart-every 10으로 재시작하세요."
                ) from error
            key = hashlib.sha256(source.encode()).hexdigest()
            target = output / "full_frames" / f"{key}.jpg"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(payload)
            # Decode now, so a corrupt browser output fails before training.
            decoded = cv2.imread(str(target))
            if decoded is None or (decoded.shape[1], decoded.shape[0]) != size:
                raise ValueError(f"JPEG 디코딩 실패: {target}")
            record = {"source_image": source, "split": split, "source_sha256": digest,
                      "transport_sha256": sha256(target), "image": str(target.relative_to(output)),
                      "original_size": original_size, "size": size,
                      "scale_xy": [size[0] / original_size[0], size[1] / original_size[1]]}
            audit.write(json.dumps(record, ensure_ascii=False) + "\n")
            audit.flush()
            del payload, decoded
            transformed[source] = record
            if index % 50 == 0:
                print(f"canvas JPEG: {index}/{len(membership)}", flush=True)
        browser_metadata = encoder.metadata

    detector_output = output / "detector_app"
    detector_output.mkdir()
    write_detector(dataset, detector_output, detector_rows, transformed, output)
    variant_yaml = {**data_yaml, "path": str(detector_output)}
    (detector_output / "data.yaml").write_text(yaml.safe_dump(variant_yaml, allow_unicode=True, sort_keys=False))

    tiny = Counter()
    with (output / "classifier_app" / "manifest.jsonl").open("w") as manifest:
        # Group by full frame to avoid repeated decoding when an image has many signals.
        grouped = {}
        for row in crop_rows:
            grouped.setdefault(str(Path(row["source_image"]).resolve()), []).append(row)
        destinations = set()
        for source, group in grouped.items():
            info = transformed[source]
            frame = cv2.imread(str(output / info["image"]))
            if frame is None:
                raise ValueError(f"전송 이미지 디코딩 실패: {source}")
            for row in group:
                relative = Path(row["crop"]).with_suffix(".png")
                if str(relative) in destinations:
                    raise ValueError(f"중복 crop: {relative}")
                destinations.add(str(relative))
                new_size = (frame.shape[1], frame.shape[0])
                # Existing xyxy already includes padding. Preserve the exact ROI,
                # class assignment and sample membership (also corrected val labels).
                box = scale_crop(row["xyxy"], info["original_size"], new_size)
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"비어 있는 crop: {row['crop']}")
                if min(x2 - x1, y2 - y1) < 6:
                    tiny["app"] += 1  # Keep tiny samples; do not change train membership.
                target = child(output / "classifier_app", relative)
                if not cv2.imwrite(str(target), frame[y1:y2, x1:x2]):
                    raise OSError(f"crop 저장 실패: {target}")
                manifest.write(json.dumps({**row, "crop": str(relative), "xyxy": box,
                                           "source_sha256": info["source_sha256"]},
                                          ensure_ascii=False) + "\n")
    summary = {"complete": True, "browser": browser_metadata,
               "detector_images": len(detector_rows), "classifier_crops_per_variant": len(crop_rows),
               "tiny_crops_retained": dict(tiny), "config": config,
               "detector_manifest_sha256": sha256(dataset / "manifest.jsonl"),
               "classifier_manifest_sha256": sha256(crops / "manifest.jsonl"),
               "sources_sha256": sha256(output / "sources.jsonl")}
    write_json(output / "summary.json", summary)
    print(f"전처리 완료: {output}")


def write_detector(dataset, detector_output, detector_rows, transformed, output):
    with (detector_output / "manifest.jsonl").open("w") as manifest:
        destinations = set()
        for row in detector_rows:
            record = transformed[str(Path(row["source_image"]).resolve())]
            relative = Path(row["image"]).with_suffix(".jpg")
            if str(relative) in destinations:
                raise ValueError(f"중복 이미지 경로: {relative}")
            destinations.add(str(relative))
            target = child(detector_output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            # All links point to browser-encoded images within this new preparation run.
            source = output / record["image"]
            if sha256(child(dataset, row["image"])) != record["source_sha256"]:
                raise ValueError(f"검출 manifest 원본과 실제 학습 이미지가 다릅니다: {row['image']}")
            target.symlink_to(source)
            label = child(detector_output, row["label"])
            label.parent.mkdir(parents=True, exist_ok=True)
            source_label = child(dataset, row["label"])
            validate_yolo(source_label)
            shutil.copyfile(source_label, label)
            # YOLO normalized xywh is invariant under independent x/y scaling,
            # including the integer rounding used for the resized canvas dimensions.
            manifest.write(json.dumps({**row, "image": str(relative),
                                       "source_sha256": record["source_sha256"],
                                       "label_sha256": sha256(label)}, ensure_ascii=False) + "\n")


def validate_yolo(path):
    import math
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5 or values[0] not in {"0", "1"}:
            raise ValueError(f"잘못된 YOLO 라벨: {path}")
        x, y, w, h = map(float, values[1:])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError(f"잘못된 YOLO 좌표: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--restart", action="store_true", help="미완료 출력 폴더를 보관하고 처음부터 다시 전처리")
    parser.add_argument("--browser-restart-every", type=int, default=50,
                        help="브라우저/드라이버를 재시작할 이미지 수 (기본 50)")
    args = parser.parse_args(argv)
    if args.browser_restart_every < 1:
        parser.error("--browser-restart-every는 1 이상이어야 합니다.")
    prepare(read_json(args.config), encoder_factory=partial(BrowserJPEG, restart_every=args.browser_restart_every),
            restart=args.restart)


if __name__ == "__main__":
    main()
