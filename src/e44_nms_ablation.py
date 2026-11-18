"""E44: paired NMS ablation using the frozen E42 augmented checkpoint; VAL only.

Soft-NMS decays *positive* score weights and converts them back to original signed
decision-score units, preserving the frozen E42 hard-NMS scoring domain.
The log-sigmoid inverse is intentionally used rather than a raw logistic score:
a nonlinear per-box mapping changes global COCO AP ranking across images.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np

from cloud_audit import data, iou, nms
from e42_full_rbf_augmented import prep
from metrics_map import evaluate_map50
from nonlinear_appearance_study import rbf_logits
from rebuild_detector import draw_box

VARIANTS = ("hard035", "hard050", "linear", "gaussian")
FOCUS = {"beach_bird_1379.jpg", "beach_bird_1389.jpg",
         "beach_bird_1399.jpg", "beach_bird_1419.jpg",
         "beach_bird_1435.jpg", "beach_bird_1452.jpg",
         "beach_bird_1560.jpg", "beach_bird_1606.jpg"}


def _softplus(x):
    return np.logaddexp(0.0, x)


def soft_nms(boxes, scores, method, sigma=0.5, threshold=0.35,
             limit=300, min_score=-np.inf):
    """Greedy Soft-NMS with descending adjusted-score output, no input mutation.

    Original signed score s is represented by exp(s), modified by standard
    linear or Gaussian multiplicative decay, and mapped back by log. This is
    mathematically equivalent to adding log(decay) to s and requires no
    per-image min/max normalization or sigmoid saturation.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1).copy()
    if len(boxes) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid Soft-NMS input")
    if method not in ("linear", "gaussian"):
        raise ValueError(method)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    remaining = np.arange(len(scores))
    kept = []
    while len(remaining) and len(kept) < limit:
        # Canonical tie-breaking by original index.
        winner = int(np.argmax(scores[remaining]))
        idx = int(remaining[winner])
        remaining = np.delete(remaining, winner)
        if scores[idx] < min_score:
            break
        kept.append(idx)
        if len(remaining):
            overlaps = np.clip(iou(boxes[idx:idx + 1], boxes[remaining])[0], 0, 1)
            if method == "linear":
                decay = np.where(overlaps > threshold, 1.0 - overlaps, 1.0)
            else:
                decay = np.exp(-(overlaps ** 2) / sigma)
            # A perfect overlap is completely suppressed in linear Soft-NMS.
            scores[remaining] += np.log(np.maximum(decay, np.finfo(float).tiny))
    return np.asarray(kept, dtype=int), scores


def apply_nms(boxes, scores, mode):
    if mode == "hard035":
        return nms(boxes, scores, 0.35, 300), scores
    if mode == "hard050":
        return nms(boxes, scores, 0.50, 300), scores
    if mode in ("linear", "gaussian"):
        return soft_nms(boxes, scores, mode)
    raise ValueError(mode)


def match_topk(boxes, gt, k=20):
    """One-to-one matching in confidence order, same as E42 reporting."""
    ov = iou(boxes[:k], gt)
    matched = set()
    tp = 0
    assigned = []
    for j in range(min(k, len(boxes))):
        best = -1
        if len(gt):
            for g in np.argsort(-ov[j], kind="stable"):
                if ov[j, g] >= .5 and int(g) not in matched:
                    best = int(g)
                    break
        assigned.append(best)
        if best >= 0:
            matched.add(best)
            tp += 1
    return tp, assigned


def size_bin(gt_box):
    area = float(np.prod(gt_box[2:] - gt_box[:2]))
    return "small" if area < 1024 else ("medium" if area < 10000 else "large")


def self_test():
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], float)
    scores = np.array([2.0, 1.0, -2.0])
    before = scores.copy()
    h, hs = apply_nms(b, scores, "hard035")
    assert h.tolist() == [0, 2] and np.array_equal(hs, before)
    for method in ("linear", "gaussian"):
        kept, adj = soft_nms(b, scores, method)
        assert np.array_equal(scores, before)
        assert len(kept) == 3 and kept[0] == 0
        assert np.all(np.diff(adj[kept]) <= 1e-12)
        assert adj[1] < scores[1] and adj[2] == scores[2]
    lin, adjusted = soft_nms(b, scores, "linear")
    assert adjusted[1] < -700  # exact duplicate should decay effectively to zero
    # Non-overlapping predictions retain exactly the ORIGINAL signed confidence.
    far = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    ids, adj = soft_nms(far, np.array([-6., 4.]), "gaussian")
    assert ids.tolist() == [1, 0] and np.array_equal(adj, [-6., 4.])
    # The official metric and output order must agree with E42 hard-NMS.
    gt = np.array([[0, 0, 10, 10]], float)
    report = evaluate_map50([(b[h], scores[h], gt)])
    assert np.isclose(report["map50_coco101"], 1.0)
    assert match_topk(b[h], gt)[0] == 1
    print("E44 PASS: signed scores, NMS ranking, duplicate decay, IoU, COCO AP", flush=True)


def run(root, checkpoint, out, val_limit=0, expected_baseline=0.20015650733804727):
    out.mkdir(parents=True, exist_ok=True)
    visual = out / "visual"
    visual.mkdir(exist_ok=True)
    models = joblib.load(checkpoint)
    stage1 = models["stage1"]
    model = models["augmented_model"]
    val = data(root, "val", val_limit)
    preds = {k: [] for k in VARIANTS}
    rows = []
    stages = {"ground_truth": 0, "proposal_covered50": 0}
    by_size = {k: {"gt": 0, "raw_covered50": 0} for k in ("small", "medium", "large")}
    for index, (path, gt) in enumerate(val, 1):
        rec, im = prep(path, gt, stage1)
        boxes = rec["boxes"]
        scores = (.5 * rec["margin"] + .5 * rbf_logits(model, rec)
                  if len(boxes) else np.empty(0))
        assert boxes.shape == (len(scores), 4)
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        overlap = iou(boxes, gt)
        best = overlap.max(axis=0) if len(boxes) and len(gt) else np.zeros(len(gt))
        covered = int(np.sum(best >= .5))
        stages["ground_truth"] += len(gt)
        stages["proposal_covered50"] += covered
        for box, o in zip(gt, best):
            item = by_size[size_bin(box)]
            item["gt"] += 1
            item["raw_covered50"] += int(o >= .5)
        record = {"image": path.name, "gt": len(gt), "proposals": len(boxes),
                  "raw_covered50": covered, "raw_recall50": covered / max(1, len(gt))}
        for mode in VARIANTS:
            ids, adjusted = apply_nms(boxes, scores, mode)
            pred, conf = boxes[ids], adjusted[ids]
            assert len(pred) == len(conf) and np.isfinite(conf).all()
            assert np.all(np.diff(conf) <= 1e-10), "NMS output is not confidence-ranked"
            preds[mode].append((pred, conf, gt))
            tp, matched = match_topk(pred, gt)
            rows.append(dict(**record, mode=mode, kept=len(ids), top20_tp=tp,
                             top20_fp=min(20, len(pred)) - tp,
                             top20_recall=tp / max(1, len(gt)),
                             post_nms_covered50=int(np.sum(
                                 iou(pred, gt).max(axis=0) >= .5))
                                 if len(pred) and len(gt) else 0))
            if mode in ("hard035", "gaussian") and path.name in FOCUS:
                canvas = im.copy()
                for box in gt:
                    draw_box(canvas, box, (0, 255, 0), 2)
                for j, box in enumerate(pred[:20]):
                    draw_box(canvas, box, (255, 0, 0) if matched[j] >= 0
                             else (0, 0, 255), 2)
                cv2.imwrite(str(visual / (path.stem + "_" + mode + ".jpg")), canvas)
        if index % 20 == 0:
            print("E44 VAL", index, len(val), flush=True)
    summary = {}
    for mode in VARIANTS:
        metric = evaluate_map50(preds[mode])
        subset = [row for row in rows if row["mode"] == mode]
        metric.update(top20_tp=sum(r["top20_tp"] for r in subset),
                      top20_fp=sum(r["top20_fp"] for r in subset),
                      post_nms_covered50=sum(r["post_nms_covered50"] for r in subset))
        summary[mode] = metric
        print("E44 RESULT", mode, json.dumps(metric), flush=True)
    with (out / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = sorted((r for r in rows if r["mode"] == "hard035"),
                    key=lambda r: (r["raw_recall50"], r["top20_recall"], -r["gt"]))
    (out / "worst_images.json").write_text(json.dumps(failed[:40], indent=2))
    report = dict(validation_images=len(val), gt=stages["ground_truth"],
                  model_checkpoint=str(checkpoint), split="VAL only; TEST untouched",
                  nms=summary, proposal=stages, by_size=by_size,
                  baseline_reference=expected_baseline,
                  score_definition="raw fused E42 margin+RBF logit; Soft-NMS log-weight decay",
                  baseline_verified=None)
    if val_limit == 0:
        difference = summary["hard035"]["map50_coco101"] - expected_baseline
        report["baseline_verified"] = abs(difference) < 1e-6
        report["baseline_difference"] = difference
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    if val_limit == 0 and not report["baseline_verified"]:
        raise RuntimeError("E44 hard035 FAILED frozen E42 baseline reproduction: "
                           + str(report["baseline_difference"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("e44_results"))
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        if not args.dataset or not args.checkpoint:
            parser.error("--dataset and --checkpoint required")
        run(args.dataset, args.checkpoint, args.output, args.val_limit)
