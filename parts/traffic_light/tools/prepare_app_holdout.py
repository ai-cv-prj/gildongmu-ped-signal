"""Build an evaluation manifest from fully labeled, unused real app sessions.

Input: source/<session_id>/*.jpg with adjacent LabelMe JSON, drawn directly
on the JPEG bytes saved by the app server. No resizing or re-encoding occurs.
"""

import argparse
import json
from pathlib import Path

from .app_quality_common import read_json
from .evaluate_app_quality import validate_holdout

LABELS = {"R_Signal": ("pedestrian_signal", "red"), "G_Signal": ("pedestrian_signal", "green"),
          "Unknown_Signal": ("pedestrian_signal", "unknown"),
          "Zebra_Cross": ("crosswalk", None), "crosswalk": ("crosswalk", None)}


def build_manifest(source, sessions, output):
    import cv2
    records = []
    if output.exists():
        raise FileExistsError(f"기존 manifest 보호: {output}")
    for session in sessions:
        if Path(session).name != session or session in {".", ".."}:
            raise ValueError(f"세션 ID는 폴더 이름이어야 합니다: {session}")
        images = sorted(p for p in (source / session).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"})
        if not images:
            raise ValueError(f"이미지가 없는 세션: {session}")
        for image in images:
            annotation = read_json(image.with_suffix(".json"))  # Missing labels are errors, not negatives.
            if not isinstance(annotation.get("shapes"), list):
                raise ValueError(f"LabelMe shapes 목록이 필요합니다: {image}")
            objects = []
            for shape in annotation["shapes"]:
                if shape.get("label") not in LABELS or shape.get("shape_type") != "rectangle":
                    raise ValueError(f"지원하지 않는 평가 라벨/shape: {image}: {shape}")
                points = shape["points"]
                if len(points) != 2 or any(len(p) != 2 for p in points):
                    raise ValueError(f"rectangle 좌표 오류: {image}")
                (ax, ay), (bx, by) = points
                class_name, color = LABELS[shape["label"]]
                obj = {"class_name": class_name, "xyxy": [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]}
                if color is not None:
                    obj["color"] = color
                objects.append(obj)
            records.append({"image": str(image.resolve()), "width": annotation["imageWidth"],
                            "height": annotation["imageHeight"], "session_id": session,
                            "origin": "app_upload", "split": "test", "unused_for_training": True,
                            "annotations": objects})
    validate_holdout(records, output, set(), cv2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in records:
            row.pop("resolved_image")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"평가 manifest: {output} ({len(records)} images)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="학습·모델 선택·임계값 튜닝에 사용하지 않은 세션 폴더 이름")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_manifest(args.source.resolve(), args.sessions, args.output.resolve())


if __name__ == "__main__":
    main()
