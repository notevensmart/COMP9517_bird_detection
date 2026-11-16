# E33 Full Test — Final Classical Bird Detector Evaluation

**Date:** 16 November 2026. **Status:** verified completed evaluation. No deep-learning models or pretrained visual encoders.

## Frozen method and primary source

Selected architecture: OpenCV Selective Search, 64x64 HOG+LBP patches, Linear SVM stage 1, TRAIN-derived PCA64 plus 32D handcrafted colour/texture/geometry context, RBF SVM stage 2, 50/50 score fusion and NMS IoU 0.35.

- **Primary run:** [43828726227](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43828726227), commit `b6bba718bf2436b5f1cbce05218c90550c18f09a`, [raw report, per-image CSV and original prediction overlays](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43828726227/artifacts/14211079111).
- **Duplicate run:** [43830283187](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43830283187), commit `3a0889e2ce81c9703742e82f7fcfba1b4e84fdfc`, [artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43830283187/artifacts/14214421518).
- Both snapshots used the same evaluation code blob `2e411930f18262fa12569a9e77c8d3c905951ab2`. The first successfully completed frozen run is the primary result, not whichever scored higher.

## Training and independent validation

- **TRAIN:** 200 of 1,369 official training images (14.6%), seeded annotated-positive subset. PCA and classifiers fit on TRAIN only.
- **Complementary VAL:** 231 images with 1,467 official GT birds, disjoint from the 60-image E32/E35 development VAL sample. It is not guaranteed to be disjoint from *all* prior exploratory studies.
- **TEST:** all 240 official test images, containing 1,518 GT birds. Model choice was written to `model_freeze.json` before any TEST data was accessed.

| Method on complementary VAL | COCO 101-point mAP@IoU .5 | Correct among top20 per image | Small-bird top100 recall | Crowded top100 recall |
|---|---:|---:|---:|---:|
| Pure Selective Search + RBF SVM | **13.4112%** | **609 / 4,620** | 33.20% | 36.17% |
| Bounded small-object hybrid + RBF SVM | 13.9390% | 582 / 4,620 | **40.39%** | **40.19%** |

The small-object hybrid gained only **+0.5279 percentage points** in overall validation mAP, lost 27 top20 true positives, and therefore **failed the predeclared confirmation requirement** of >=+1pp mAP, no loss in top20 TP, and limited small/crowding regressions. The **pure SS/RBF baseline** was frozen before TEST.

## Official full-TEST metrics (not subset TEST)

| Metric | Primary frozen run | Duplicate consistency run |
|---|---:|---:|
| COCO 101-point bird bbox mAP@.5 (maxDets100) | **11.8888%** | 11.7971% |
| Test images | 240 | 240 |
| Official GT birds | 1,518 | 1,518 |
| Top20 TP / total top20 predicted | **564 / 4,799** | 561 / 4,799 |
| Top20 FP | 4,235 | 4,238 |
| Top100 official GT recall | 56.39% | 56.39% |
| Small-bird top100 recall | 38.35% | 38.53% |
| Medium-bird top100 recall | 64.20% | 63.92% |
| Large-bird top100 recall | 73.36% | 73.77% |
| Crowded-scene top100 GT recall | 42.80% | 42.42% |
| End-to-end inference mean seconds/image | **3.519** | 4.762 |

Timing includes image read, candidate generation, HOG/LBP+context features, SVM scoring, and NMS; **no GT matching** in timing. GitHub-hosted CPU variance makes between-run speed comparisons imperfect. Do not label 11.89% as training, VAL or subset TEST mAP: it covers **all 240 TEST images**.

## Model output observations and limitations

Top-ranked detections remain false-positive heavy. Original post-hoc test overlays and per-image CSV are archived with the primary run. The quantitative GT-size gap is large: small birds 38.35% top100 recall vs large 73.36%; crowded scenes 42.80%. Previous VAL photo audits identified false seaweed/reed/drone boxes, bird-part regions instead of entire large birds, and official zero-GT images containing visible birds. Labels must remain unmodified for official metrics; manual semantic QA must be reported separately. No test output was used to refit the model.

## Reproducibility caveat — E36 / E37 / E38

[E36](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43830191020) established that OpenCV returns the same *uncapped* Selective Search proposal coordinate set across repeated calls but with variable output order. The old first-1200 cap can therefore select a different proposal subset on repeated calls. The duplicate frozen runs yielded 11.8888% vs 11.7971% mAP50, a 0.0917pp difference consistent with that bug; **do not claim bit-exact reproducibility** for the final frozen method.

On separate `research/deterministic-selective-proposals` branch, [E37](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43831293647) canonicalized box order before a fixed-seed cap and demonstrated identical repeated/interleaved proposal hashes on the two audited images. The corrected cap was **not substituted into this full TEST**.

[E38](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43831970676), on only TRAIN45/VAL30, measured deterministic pure SS **14.2707%** mAP50 and bounded hybrid **12.4978%** mAP50 (top20 TPs 83 vs 73). Extra proposal coverage did not improve final detection. This is VAL-only analysis, *not* another TEST result.

## Interpretation

The 11.89% full-TEST mAP was achieved using only **200/1369 (14.6%) of the training images**, so data scaling is a legitimate future TRAIN+VAL research question. **It is not evidence that full training will necessarily raise test mAP.** Prior training-size and negative-mining experiments sometimes plateaued or regressed. Given the TEST split has now been evaluated, any further tuning or scaling experiment must be reported transparently as post-test development, assessed on TRAIN/VAL (or a new genuinely untouched held-out set), and must not be presented as a fresh independent test confirmation.

The target 30–50% test mAP was **not achieved**. Primary remaining bottlenecks: whole-bird box geometry, severe natural-background false positives, tiny and crowded birds, incomplete annotations, and nondeterministic proposal capping.
