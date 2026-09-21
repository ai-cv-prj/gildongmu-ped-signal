"""YOLO와 분류기 모델이 학습하기 위한 이미지를 생성하기 위해
동영상을 프레임단위로  이미지 여러장으로 분해"""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 유한한 수가 필요합니다.")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("0 이상의 유한한 수가 필요합니다.")
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return value


def jpeg_quality(value):
    value = int(value)
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError("JPEG 품질은 1~100이어야 합니다.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="영상 파일 또는 영상 폴더")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/traffic_light_frames",
        help="추출 이미지와 manifest.jsonl을 저장할 빈 폴더",
    )
    parser.add_argument("--sample-fps", type=positive_float, default=3.0, help="초당 추출 이미지 수")
    parser.add_argument("--start-sec", type=nonnegative_float, default=0.0)
    parser.add_argument("--end-sec", type=positive_float)
    parser.add_argument("--jpeg-quality", type=jpeg_quality, default=95)
    parser.add_argument("--max-frames-per-video", type=positive_int, help="영상별 최대 추출 수")
    return parser.parse_args(argv)


def find_videos(source):
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"지원하는 영상 확장자가 아닙니다: {source}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"입력 경로가 없습니다: {source}")
    videos = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"지원하는 영상이 없는 폴더입니다: {source}")
    return videos


def ensure_empty_output(output):
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"출력 폴더가 비어 있지 않습니다: {output}\n"
            "기존 데이터를 보호하기 위해 중단했습니다. 다른 --output 경로를 사용하세요."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def unique_output_names(videos):
    counts = {}
    names = []
    for video in videos:
        stem = video.stem
        counts[stem] = counts.get(stem, 0) + 1
        suffix = "" if counts[stem] == 1 else f"_{counts[stem]}"
        names.append(stem + suffix)
    return names


def extract_video(video, video_output, args, cv2, manifest):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError(f"영상 FPS를 확인할 수 없습니다: {video}")
    if args.end_sec is not None and args.end_sec <= args.start_sec:
        capture.release()
        raise ValueError("--end-sec는 --start-sec보다 커야 합니다.")

    video_output.mkdir(parents=True, exist_ok=False)
    first_frame_index = max(0, int(math.ceil(args.start_sec * source_fps - 1e-9)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)
    frame_index = first_frame_index
    next_sample_sec = args.start_sec
    sample_period = 1.0 / args.sample_fps
    extracted = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_sec = frame_index / source_fps
            if args.end_sec is not None and timestamp_sec >= args.end_sec:
                break
            if timestamp_sec + 1e-9 >= next_sample_sec:
                timestamp_ms = round(timestamp_sec * 1000)
                filename = f"frame_{frame_index + 1:08d}_t{timestamp_ms:010d}ms.jpg"
                image_path = video_output / filename
                written = cv2.imwrite(
                    str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if not written:
                    raise OSError(f"이미지 저장에 실패했습니다: {image_path}")
                height, width = frame.shape[:2]
                manifest.write(json.dumps({
                    "image": str(image_path.relative_to(args.output)),
                    "source_video": str(video),
                    "source_frame_index": frame_index + 1,
                    "timestamp_ms": timestamp_ms,
                    "width": width,
                    "height": height,
                    "source_fps": source_fps,
                }, ensure_ascii=False) + "\n")
                extracted += 1
                while next_sample_sec <= timestamp_sec + 1e-9:
                    next_sample_sec += sample_period
                if args.max_frames_per_video and extracted >= args.max_frames_per_video:
                    break
            frame_index += 1
    finally:
        capture.release()
    return extracted, source_fps


def main(argv=None):
    args = parse_args(argv)
    videos = find_videos(args.source)
    args.output = ensure_empty_output(args.output)

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("가상환경을 활성화하고 opencv-python을 설치하세요.") from error

    total = 0
    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for video, output_name in zip(videos, unique_output_names(videos)):
            count, source_fps = extract_video(
                video, args.output / output_name, args, cv2, manifest
            )
            total += count
            print(f"{video.name}: {count}장 추출 (원본 {source_fps:.3f} FPS)")

    config = {
        "source": str(args.source.expanduser().resolve()),
        "output": str(args.output),
        "sample_fps": args.sample_fps,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "jpeg_quality": args.jpeg_quality,
        "max_frames_per_video": args.max_frames_per_video,
        "video_count": len(videos),
        "extracted_frames": total,
    }
    (args.output / "extraction_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: 영상 {len(videos)}개에서 총 {total}장")
    print(f"출력 폴더: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"오류: {error}")
