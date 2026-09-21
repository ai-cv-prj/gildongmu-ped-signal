"""Create a grouped, stratified split of existing app-quality images; train separately."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import shutil

from .app_quality_common import child, new_directory, read_json, resolve, rows, sha256, write_json
from .train_app_quality import detector_arguments, verify_environment, verify_initial

DEFAULT_CONFIG = resolve("parts/traffic_light/configs/app_quality_resplit.json")
SPLITS = ("train", "val", "test")
CLASSES = {0: "pedestrian_signal", 1: "crosswalk"}


def sequence_key(source, block_size):
    """A filename heuristic, NOT verified location/session metadata."""
    stem = Path(source).stem
    matched = re.fullmatch(r"(.*?\D)(\d+)", stem)
    if not matched:
        return f"unparsed:{stem}"
    return f"{matched[1]}:{int(matched[2]) // block_size}"


def grouped_indices(records, block_size):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("sequence_block_size는 양의 정수여야 합니다.")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        keys = (f"sha:{row['source_sha256']}", f"seq:{sequence_key(row['source_image'], block_size)}")
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return list(groups.values())


def label_features(text, size):
    counts = Counter()
    size_counts = Counter()
    for line in text.splitlines():
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError("잘못된 YOLO 라벨")
        cid, x, y, w, h = values
        if cid not in CLASSES or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError("클래스/정규화 좌표가 잘못되었습니다.")
        counts[CLASSES[int(cid)]] += 1
        if cid == 0:
            edge = math.sqrt(w * size[0] * h * size[1])
            size_counts["tiny" if edge < 16 else "small" if edge < 32 else "large"] += 1
    return counts, size_counts


def allocate(records, groups, ratios, seed):
    """Greedy weighted feature balance with deterministic group-level refinement."""
    if len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("train/val/test 비율은 양수이고 합이 1이어야 합니다.")
    if len(groups) < 3:
        raise ValueError("세 분할에 배치할 독립 그룹이 부족합니다.")
    vectors = []
    for group in groups:
        vector = Counter()
        for i in group:
            row = records[i]
            counts, sizes = row["class_counts"], row["signal_size_counts"]
            vector["images"] += 1
            vector[f"signals_{min(counts.get('pedestrian_signal', 0), 3)}"] += 1
            vector[f"crosswalks_{min(counts.get('crosswalk', 0), 3)}"] += 1
            vector[f"previous_{row['split']}"] += 1
            for key, value in sizes.items():
                vector[f"signal_boxes_{key}"] += value
        vectors.append(vector)
    total = sum(vectors, Counter())
    weights = {k: (8 if k == "images" else 1) / max(v, 20) for k, v in total.items()}
    current = [Counter() for _ in ratios]
    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda i: -sum(weights[k] * v * v for k, v in vectors[i].items()))
    assigned = {}

    def delta(split, vector, sign):
        return sum(weights[k] * (2 * (current[split][k] - ratios[split] * total[k]) * sign * v + v * v)
                   for k, v in vector.items())

    for i in order:
        split = min(range(3), key=lambda s: (delta(s, vectors[i], 1), s))
        assigned[i] = split
        current[split].update(vectors[i])
    group_counts = Counter(assigned.values())
    for _ in range(4):
        changed = False
        for i in order:
            old = assigned[i]
            if group_counts[old] <= 1:
                continue
            remove = delta(old, vectors[i], -1)
            choices = [(remove + delta(s, vectors[i], 1), s) for s in range(3) if s != old]
            score, new = min(choices)
            if score < -1e-9:
                current[old].subtract(vectors[i])
                current[new].update(vectors[i])
                assigned[i] = new
                group_counts[old] -= 1
                group_counts[new] += 1
                changed = True
        if not changed:
            break
    if any(not current[s]["images"] for s in range(3)):
        raise ValueError("빈 분할이 있습니다. 그룹 규칙/비율을 확인하세요.")
    membership = {}
    for gi, indices in enumerate(groups):
        for i in indices:
            membership[i] = (SPLITS[assigned[gi]], f"group_{gi:06d}")
    return membership


def load_base(config):
    base = read_json(resolve(config["base_config"]))
    if sha256(resolve(config["base_config"])) != config["base_config_sha256"]:
        raise ValueError("기준 설정이 변경됐습니다.")
    return base


def prepare(config):
    import yaml
    base = load_base(config)
    prepared = resolve(base["prepared_root"])
    original = prepared / "detector_app"
    previous_summary = read_json(prepared / "summary.json")
    if not previous_summary["complete"]:
        raise ValueError("앱 화질 전처리가 완료되지 않았습니다.")
    if sha256(original / "manifest.jsonl") != config["prepared_manifest_sha256"]:
        raise ValueError("기준 검출 manifest가 변경됐습니다.")
    if sha256(prepared / "sources.jsonl") != previous_summary["sources_sha256"]:
        raise ValueError("전처리 이미지 정보가 변경됐습니다.")
    sources = {r["source_image"]: r for r in rows(prepared / "sources.jsonl")}
    records = list(rows(original / "manifest.jsonl"))
    for row in records:
        label = child(original, row["label"])
        info = sources[row["source_image"]]
        if sha256(label) != row["label_sha256"] or row["source_sha256"] != info["source_sha256"]:
            raise ValueError(f"원본 이미지/라벨 해시 불일치: {label}")
        row["class_counts"], row["signal_size_counts"] = label_features(label.read_text(), info["size"])
    groups = grouped_indices(records, config["sequence_block_size"])
    membership = allocate(records, groups, config["ratios"], config["seed"])
    source_membership = {}
    digest_membership = {}
    for i, row in enumerate(records):
        split, group = membership[i]
        source_membership[row["source_image"]] = (split, group)
        if row["source_sha256"] in digest_membership and digest_membership[row["source_sha256"]] != split:
            raise ValueError("동일 이미지가 여러 분할에 배치됐습니다.")
        digest_membership[row["source_sha256"]] = split
    crop_root = prepared / "classifier_app"
    crops = list(rows(crop_root / "manifest.jsonl"))
    if any(r["source_image"] not in source_membership for r in crops):
        raise ValueError("검출기에 없는 분류 crop 원본이 있습니다.")
    output = new_directory(resolve(config["output"]))
    detector = output / "detector_app"
    classifier = output / "classifier_app"
    for split in SPLITS:
        (detector / "images" / split).mkdir(parents=True)
        (detector / "labels" / split).mkdir(parents=True)
        for name in sorted({c["class_name"] for c in crops}):
            (classifier / split / name).mkdir(parents=True)
    statistics = {s: Counter() for s in SPLITS}
    transitions = {s: Counter() for s in SPLITS}
    crop_counts = {s: Counter() for s in SPLITS}
    with (detector / "manifest.jsonl").open("w") as stream:
        for i, row in enumerate(records):
            split, group = membership[i]
            info = sources[row["source_image"]]
            src = child(original, row["image"])
            if sha256(src) != info["transport_sha256"]:
                raise ValueError(f"압축 이미지 해시 불일치: {src}")
            # Include old split to prevent rare filename collisions after merging.
            stem = f"{row['split']}_{Path(row['image']).stem}"
            image_path = Path("images") / split / f"{stem}.jpg"
            label_path = Path("labels") / split / f"{stem}.txt"
            (detector / image_path).symlink_to(src.resolve())
            shutil.copyfile(child(original, row["label"]), detector / label_path)
            entry = {**row, "previous_split": row["split"], "split": split, "group": group,
                     "image": str(image_path), "label": str(label_path), "transport_sha256": info["transport_sha256"]}
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            st = statistics[split]
            st["images"] += 1
            st.update(row["class_counts"])
            st.update({f"signal_{k}": v for k, v in row["signal_size_counts"].items()})
            st["signal_images"] += row["class_counts"].get("pedestrian_signal", 0) > 0
            st["multi_signal_images"] += row["class_counts"].get("pedestrian_signal", 0) >= 2
            st["multi_crosswalk_images"] += row["class_counts"].get("crosswalk", 0) >= 2
            transitions[row["split"]][split] += 1
            if (i + 1) % 5000 == 0:
                print(f"재분할 파일 준비: {i+1}/{len(records)}", flush=True)
    with (classifier / "manifest.jsonl").open("w") as stream:
        for row in crops:
            split, group = source_membership[row["source_image"]]
            relative = Path(split) / row["class_name"] / f"{row['split']}_{Path(row['crop']).name}"
            source = child(crop_root, row["crop"])
            if not source.is_file():
                raise FileNotFoundError(source)
            (classifier / relative).symlink_to(source.resolve())
            stream.write(json.dumps({**row, "previous_split": row["split"], "split": split,
                                     "group": group, "crop": str(relative)}, ensure_ascii=False) + "\n")
            crop_counts[split][row["class_name"]] += 1
    data = {"path": str(detector), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASSES}
    (detector / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    summary = {"complete": True, "config": config, "statistics": statistics, "previous_to_new": transitions,
               "classifier_counts": crop_counts, "groups": len(groups), "largest_group": max(map(len, groups)),
               "source_image_count": len(records), "classifier_crop_count": len(crops),
               "detector_manifest_sha256": sha256(detector / "manifest.jsonl"),
               "classifier_manifest_sha256": sha256(classifier / "manifest.jsonl"),
               "data_yaml_sha256": sha256(detector / "data.yaml"),
               "limitations": ["sequence groups are filename blocks, not verified capture sessions/locations",
                               "new test contains old train samples; historical checkpoints are not independent baselines",
                               "existing image JPEGs/crops reused without re-encoding; coordinates unchanged"]}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def training_args(config):
    base = load_base(config)
    args = detector_arguments(base)
    args.update(data=str(resolve(config["output"]) / "detector_app/data.yaml"),
                project=str(resolve(config["runs_root"]) / "detector"), name="app", epochs=config["epochs"],
                patience=config["patience"], seed=config["seed"], resume=False, exist_ok=False)
    return args


def train(config, print_only=False):
    args = training_args(config)
    if print_only:
        print(json.dumps(args, ensure_ascii=False, indent=2))
        return
    base = load_base(config)
    environment = verify_environment(base)
    initial = verify_initial(base["detector"])
    output = resolve(config["output"])
    summary = read_json(output / "summary.json")
    if not summary["complete"] or summary["config"] != config:
        raise ValueError("재분할 완료 당시 설정과 현재 설정이 다릅니다.")
    detector = output / "detector_app"
    if sha256(detector / "manifest.jsonl") != summary["detector_manifest_sha256"] or sha256(detector / "data.yaml") != summary["data_yaml_sha256"]:
        raise ValueError("재분할 manifest/data.yaml이 변경됐습니다.")
    for row in rows(detector / "manifest.jsonl"):
        if sha256(child(detector, row["label"])) != row["label_sha256"] or sha256(child(detector, row["image"])) != row["transport_sha256"]:
            raise ValueError(f"학습 이미지/라벨이 변경됐습니다: {row['image']}")
    run_root = resolve(config["runs_root"])
    if (run_root / "detector/app").exists():
        raise FileExistsError("기존 학습 결과 보호: 새 runs_root를 지정하세요.")
    audit = new_directory(run_root / "audit/detector_app")
    write_json(audit / "provenance.json", {"config": config, "environment": environment,
                                          "initial_sha256": sha256(initial), "summary_sha256": sha256(output / "summary.json")})
    write_json(audit / "args.json", args)
    from ultralytics import YOLO
    model = YOLO(str(initial), task="detect")
    model.train(**args)
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    write_json(audit / "complete.json", {"best": str(best), "sha256": sha256(best)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare" and args.print_only:
        parser.error("--print-only는 train 설정 확인에만 사용할 수 있습니다.")
    config = read_json(args.config)
    if args.action == "prepare":
        prepare(config)
    else:
        train(config, args.print_only)


if __name__ == "__main__":
    main()
