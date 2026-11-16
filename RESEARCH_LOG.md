# Research log — Beach Bird Detection (16 November 2026)

## Objective
Audit why bounding-box mAP is extremely low, validate each stage independently, and attempt justified corrections before any full training. Required dataset: public Kaggle Beach Bird Detection Dataset v1.0. Train/validation/test must remain separated.

## Stage A — Reproducible CPU cloud environment
GitHub Actions run [#1](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43785783873) successfully installed dependencies, downloaded the Kaggle dataset and passed synthetic IoU, AP50 and NMS tests.

The cloud quick baseline uses a different seeded random subset and patch sampling procedure from the original Windows notebook, so numerical values are **not directly paired** with the historical notebook.

### Cloud quick baseline, 80 train images / 20 val images / 109 val birds

| Method | Proposal recall @ IoU 0.5 | Candidate boxes per image |
|---|---:|---:|
| Canny capped at 200 | 7.34% | 191.3 |
| Canny uncapped | 32.11% | 1,310.95 |
| Canny uncapped + 25% padding | 42.20% | 1,310.95 |

The **capped Canny + HOG + linear SVM** achieved 0.000861235 *continuous AP50* (0.0861%), with mean inference 0.1453 s/image. This is a measured exploratory val-subset AP, **not a final test score**.

## Stage B — Tiny-image memorisation gate

GitHub Actions run [#2](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43785992697) tested the same 2 positive + 1 negative train images (6 annotated birds).

| Proposal setting | Proposal recall | Patch accuracy | Raw-image train AP50 |
|---|---:|---:|---:|
| Canny capped | 0% | N/A | N/A (no positive patches) |
| Canny uncapped | 16.67% | 100% | 16.67% |
| Canny uncapped + padding | 33.33% | 100% | 33.33% |

**Gate failed:** the classifier can memorise available patches but cannot detect birds for which the fixed candidate generator lacks suitable boxes. Training longer cannot restore a missing proposal.

## Stage C — Multi-scale geometric anchors, image-only ranking

GitHub Actions run [#3](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43786604315) compared candidates on the *same* 20 val images and 109 birds. Anchors use 9 sizes (8–144 pixels) and 3 aspect ratios with roughly quarter-window strides; edge-ranked modes select at most 450 or 900 candidates per size/ratio using only original image edges. Ground truth is read strictly after proposals are made to measure recall.

| Generator | Proposal recall @0.5 | Proposal recall @0.75 | Small-bird recall @0.5 | Mean candidate boxes |
|---|---:|---:|---:|---:|
| Canny capped | 7.34% | 2.75% | 5.88% | 191 |
| Dense anchors | **93.58%** | 38.53% | **100%** | 296,584 |
| Edge anchors / 450 | 82.57% | 32.11% | 58.82% | 11,245 |
| Edge anchors / 900 | **85.32%** | 34.86% | 76.47% | 20,702 |

This is a **proposal recall** result, not classifier detection AP/mAP. The dense grid provides a geometric upper bound and is expensive to classify. Edge-ranked anchors improve candidate coverage substantially but remain computationally demanding and can miss tiny birds.

## Stage D — Tiny anchor-based classifier
Code: `src/tiny_anchor_train.py`. The cloud workflow trains HOG+SVM on positive and uncontaminated negative anchors from the same three tiny training images, scores candidate boxes from image pixels, and evaluates raw-image training AP50. A >=0.90 training AP50 gate is used only as an overfitting sanity test; it does **not** imply generalisation.

Results should only be added after reading the actual GitHub Actions run logs or exported artifacts.

## Scientific safeguards
- No test-set metrics have been claimed.
- No ground-truth boxes are inserted into proposal inference; training labels use GT as appropriate.
- Raw geometric candidate recall is not the same as end-to-end detection recall or AP.
- Validation metrics are exploratory until a frozen model and untouched test evaluation.
- Full-scale training remains blocked by tiny-set gate; cloud execution success is not model quality.


## Stage E — Pruning redundant boxes without discarding birds

A naive global top-scoring/NMS candidate budget was **rejected** after independent measurements: 400 and 800 candidates yielded 0% proposal recall; 1,600 yielded 0.92%; 3,200 yielded 28.44%. Ranking all box shapes together over-concentrated candidates in highly textured regions and destroyed useful proposal diversity. These are experimental failure results, not adopted changes.

A replacement selects candidates per scale/aspect group and distributes them across spatial tiles using only image edges. Benchmark [Spatial Pruning Benchmark](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43787913098), same 20 validation images and 109 birds:

| Spatial proposals | Mean boxes/image | Recall@0.5 | Small-bird recall@0.5 |
|---|---:|---:|---:|
| 50 per scale | 1,350 | 71.56% | 41.18% |
| 100 per scale | 2,700 | 78.90% | 52.94% |
| **200 per scale (selected)** | **5,334** | **83.49%** | **58.82%** |
| 300 per scale | 7,734 | 85.32% | 64.71% |
| 450 per scale | 11,245 | 88.99% | 76.47% |

200-per-scale reduces the candidate count by about 53% versus 450-per-scale with a 5.50 percentage-point reduction in proposal recall. This is a compute/coverage trade-off: **not** proof of better detector AP. It is an exploratory validation-based choice; full evaluation must use an untouched test split once frozen.

## Stage F — Hard negatives and final detections

Earlier tiny-set hard-negative mining reduced mean false positives from 298.3 to 22.0 per image at its best AP iteration, increasing training AP@0.5 from 13.24% to 37.31%. The tiny set consists of 2 positive images plus 1 negative image, not the held-out validation set.

`src/detection_postprocess.py` strictly applies a score threshold, final NMS and an upper bound of 150 returned boxes. SVM margin is **not a probability**; a chosen threshold must be calibrated on validation and then frozen. The strict final limit is only a safeguard, not an expectation that images contain 150 birds.

`src/spatial_detector_tiny.py` integrates the selected spatial budget, hard-negative mining and final output filtering on the tiny training set. Its end-to-end results must be reported from its cloud-run logs, not predicted from proposal recall.
