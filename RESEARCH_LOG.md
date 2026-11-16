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
