"""Detection and color metrics with explicit denominators and one-to-one matching."""


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return ratio(intersection, area_a + area_b - intersection) or 0.0


def match(ground_truth, predictions, threshold):
    """Greedy confidence order, best available GT; inputs must be one class/frame."""
    used = set()
    matches = {}
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i]["confidence"]):
        candidates = [(iou(predictions[index]["xyxy"], gt["xyxy"]), gi)
                      for gi, gt in enumerate(ground_truth) if gi not in used]
        overlap, gi = max(candidates, default=(0.0, -1))
        if gi >= 0 and overlap >= threshold:
            used.add(gi)
            matches[index] = gi
    return matches


def average_precision(frames, class_name, threshold):
    scored = []
    total = 0
    for frame in frames:
        gt = [g for g in frame["ground_truth"] if g["class_name"] == class_name]
        pred = [p for p in frame["predictions"] if p["class_name"] == class_name]
        total += len(gt)
        matched = match(gt, pred, threshold)
        scored.extend((p["confidence"], int(i in matched)) for i, p in enumerate(pred))
    if not total:
        return None
    scored.sort(key=lambda entry: -entry[0])
    tp = 0
    curve = []
    for count, (_, is_tp) in enumerate(scored, 1):
        tp += is_tp
        curve.append((tp / total, tp / count))
    # COCO-style 101 recall points, with a precision envelope. This is a bbox
    # metric for fully annotated frames; it does not implement crowd/ignore areas.
    return sum(max((p for r, p in curve if r >= point / 100), default=0.0)
               for point in range(101)) / 101


def detection_metrics(frames, class_name, confidence, match_iou, small_area, calculate_ap=True):
    tp = fp = fn = small_gt = small_tp = 0
    for frame in frames:
        gt = [g for g in frame["ground_truth"] if g["class_name"] == class_name]
        pred = [p for p in frame["predictions"] if p["class_name"] == class_name and p["confidence"] >= confidence]
        matches = match(gt, pred, match_iou)
        found = set(matches.values())
        tp += len(matches)
        fp += len(pred) - len(matches)
        fn += len(gt) - len(matches)
        for index, g in enumerate(gt):
            x1, y1, x2, y2 = g["xyxy"]
            if (x2 - x1) * (y2 - y1) < small_area:
                small_gt += 1
                small_tp += index in found
    ap = [average_precision(frames, class_name, .5 + .05 * i) for i in range(10)] if calculate_ap else [None]
    return {"tp": tp, "fp": fp, "fn": fn, "gt": tp + fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "small_gt": small_gt, "small_tp": small_tp, "small_recall": ratio(small_tp, small_gt),
            "AP50": ap[0], "AP50_95": sum(ap) / len(ap) if ap[0] is not None else None}


def color_metrics(pairs):
    columns = ("red", "green", "unknown", "missed", "crop_skipped")
    confusion = {color: dict.fromkeys(columns, 0) for color in ("red", "green", "unknown")}
    for actual, predicted in pairs:
        confusion[actual][predicted] += 1
    result = {"confusion": confusion}
    for actual, wrong in (("red", "green"), ("green", "red")):
        total = sum(confusion[actual].values())
        classified = sum(confusion[actual][c] for c in ("red", "green", "unknown"))
        result[f"{actual}_gt"] = total
        result[f"{actual}_to_{wrong}_count"] = confusion[actual][wrong]
        result[f"{actual}_to_{wrong}_per_gt"] = ratio(confusion[actual][wrong], total)
        result[f"{actual}_to_{wrong}_per_classified"] = ratio(confusion[actual][wrong], classified)
        result[f"{actual}_correct_per_gt"] = ratio(confusion[actual][actual], total)
    return result


def paired_colors(frames, confidence, match_iou, raw=False):
    pairs = []
    for frame in frames:
        gt = [g for g in frame["ground_truth"] if g["class_name"] == "pedestrian_signal"]
        pred = [p for p in frame["predictions"] if p["class_name"] == "pedestrian_signal" and p["confidence"] >= confidence]
        found = {gi: pi for pi, gi in match(gt, pred, match_iou).items()}
        for gi, g in enumerate(gt):
            predicted = pred[found[gi]].get("raw_color" if raw else "color", "crop_skipped") if gi in found else "missed"
            pairs.append((g["color"], predicted))
    return color_metrics(pairs)


def latency_summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    def percentile(q):
        at = (len(ordered) - 1) * q
        lo = int(at)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (1 - (at - lo)) + ordered[hi] * (at - lo)
    return {"samples": len(values), "mean_ms": sum(values) / len(values),
            "p50_ms": percentile(.5), "p95_ms": percentile(.95)}
