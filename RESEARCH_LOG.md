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


## Stage G — Integrated spatial proposal, classifier and final-box gate (measured)

Run [Spatial Pruning Benchmark / integrated tiny detector](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43788148237) completed successfully, including unit tests for the final detection filter.

For exactly the same **three training images** (six annotated birds), `spatial_200` generated **5,334 candidates/image**. Their proposal recall on these images was **83.33%** (different from the 83.49% validation-subset coverage), so even a perfect classifier cannot detect the missing sixth bird with these fixed candidates.

Hard-negative mining and a 150-output cap were used with final NMS. Selected diagnostic rounds:

| Mining round | Mean final detections/image (SVM margin>=0) | Mean final FP/image | Tiny-train AP50 |
|---|---:|---:|---:|
| 0 | 150.0 | 148.67 | 18.75% |
| 1 | 92.67 | 90.67 | **37.10%** |
| 2 | 17.0 | 14.67 | 33.33% |
| 6 | **14.0** | **11.33** | 36.51% |

For the stricter **margin >= 1.0** diagnostic:
- Round 1: 5.0 final detections/image, 4.0 false positives/image, 29.17% tiny-train AP50.
- Round 6: 2.0 final detections/image, 1.33 false positives/image, 13.89% tiny-train AP50.

All figures are from **training images only**; they are not held-out validation mAP. They demonstrate that reducing output count requires a precision/recall trade-off. Margin 1.0 was **not** selected as a calibrated deployment threshold. Hard-negative mining helps but the tiny overfitting gate remains **FAILED**: no setting reaches 90% raw-image train AP50.

**Current decision:** use spatially balanced 200-per-scale proposals as the efficient *research candidate generator*, keep GT-free inference, final NMS, score threshold and max-detection safeguards, but DO NOT claim the current HOG/SVM classifier is deployment-ready or ready for full training. Next improvements should focus on a more discriminative classifier, better small-object representation and box regression or a suitable learned detector. A final threshold must be chosen using validation and frozen before one-time test evaluation.


## Stage H — Fair HOG vs HOG+LBP validation ablation

[GitHub Actions run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43788975227) completed successfully. The same seeded **40 train / 12 validation images** (77 annotated validation birds), same spatial-50 generator (1,350 candidate boxes/image), IoU, NMS, and SVM hard-negative mining protocol were used. These are **single-class continuous AP@0.5** on a small exploratory validation subset, **not test mAP**:

| Method | Validation AP@0.5 | Mean boxes after NMS | Mean candidate count |
|---|---:|---:|---:|
| HOG + linear SVM | **3.73%** | 272.33 | 1,350 |
| HOG + 4×4 spatial uniform LBP + linear SVM | **7.82%** | 272.58 | 1,350 |

LBP helps but remains unsatisfactory. Thresholding, localization, feature separation and NMS require further failure analysis. The 300-NMS-box cap is a computational reporting cap; many boxes still survive.

## Classical-only constraint from the team

Only **classical computer vision and traditional ML** methods are permitted in this workstream. No pretrained neural models, CNNs, SAM, DINO or learned foundation-model visual embeddings. The previously tested OpenCV **EdgeBoxes + structured-edge pretrained model** may incorporate learned (non-neural) structured forests, so describe its training provenance clearly; prefer purely handcrafted edge/texture/morphology/contour methods when strict training-free proposals are required. HOG, LBP, SVM and classical part-based detection remain in scope. Existing experiment results are historical, not approval of neural components in a final model.


## Stage I — Diversified hard-negative + hard-positive ablation (16 November 2026)

Measured [GitHub Actions 43792817123](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43792817123), with independent COCO API mAP50 calculation and per-image FP breakdowns in the experiment artifact. Reused one feature bank from 35 TRAIN /12 VAL images (77 validation birds); LoG+Otsu and HOG+LBP were fixed. Four methods shared the same initial random 20 negative proposals/image: global mining; per-image+scale+IoU-balanced negative mining; the latter with low-margin positive training patch reweighting; and an additional bounded 1400-negative cache. Evaluated after rounds 0–3, trained only on train images. 

**Best results (round 3, held-out validation subset):** global **11.73% COCO mAP@.5, 305 FP, 44 FN**; diverse-only **10.83%, 316 FP, 43 FN**; diverse+hard positives **11.64%, 333 FP, 43 FN**; bounded cache **9.82%, 345 FP, 45 FN**. FP/FN counts use final NMS detections at margin≥0, whereas mAP uses all ranked candidate scores capped by COCO maxDets=100. Performance was nonmonotonic: round 1 dropped below initial 7.59% in every variant, before recovering in later rounds. 

**Decision:** global mining remains best on this limited validation sample. A third mining round improves on the previously tested two-round global result (8.82%), but neither candidate diversity, hard-positive weights, nor bounded-cache selection outperformed the global strategy. Keep the failed ablations in the report, and avoid overfitting to a 12-image development split. Plan larger split and localisation-focused diagnostics before final test evaluation. More mining may have diminishing returns, and none of the models currently meets 50% mAP.


## Stage J — Learning curve versus training data, same validation set (16 Nov 2026)

Completed [GitHub Actions run 43793628053](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43793628053), with [figure and result bundle](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43793628053/artifacts/14193349564). Nested TRAIN samples 35/100/300/700/full 1,369, fixed VAL 60 images containing 376 bird annotations, 3 hard-negative mining rounds for every N, no test usage, primary metric official COCO 101-point bbox mAP@.50 (maxDets=100). Bounded positive and safe-negative training feature selection makes this **different from** earlier 35-image E14.

| TRAIN N | Validation mAP@.5 | False boxes @margin0 | Missed birds @margin0 |
|---:|---:|---:|---:|
| 35 | 7.49% | 3,630 | 220 |
| 100 | 6.54% | 2,970 | 224 |
| 300 | 9.78% | 2,697 | 221 |
| 700 | 11.25% | 2,997 | 203 |
| 1,369 | 11.29% | 3,004 | 204 |

This experiment tests training-data sufficiency: mAP rises mostly between 100 and 700 but plateaus after 700 (+0.043 percentage points to full 1369). A larger data set alone **does not repair the current detector**. Data contains scenes with dense flocks and small birds; the existing LoG/Otsu candidates have limited localisation accuracy and HOG/LBP linear SVM scores still rank many textured backgrounds above birds. The full-train model produces 2,078 background FP, 920 localisation FP, and 6 duplicate FP at margin 0 on the 60 VAL images. Keep failure cases and runtime trade-offs in the discussion; future work should be localisation- and feature-based, not uncontrolled hard-mining rounds. The output bundle includes actual PNG mAP/error curves and CSV of each round.
