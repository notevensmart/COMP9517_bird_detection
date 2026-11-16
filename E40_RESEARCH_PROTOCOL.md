# E40 — Three scale-specific boosted bird experts, calibrated and spatially refined

**Strict classical ML only. No CNNs, neural encoders, pretrained visual embeddings, or new TEST evaluation.** Branch `research/e40-three-expert-detector`. Workflow: [E40 GitHub Actions](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e40-three-expert.yml).

## Predeclared design

The complete user-requested architecture is now implemented in `src/e40_three_expert_detector.py`:

1. **Deterministic proposals** from E37: unique canonicalized OpenCV Selective Search candidates before a fixed-seed 1200-region cap, no GT required at inference.
2. **Object and context preprocessing** using E39's audited 176-D handcrafted spatial gradients, local LBP, HSV/LAB colour, native geometry, contrast, texture, and surrounding-context differences. Native bird crop aspect ratio is preserved by neutral letterboxing; CLAHE modifies only the grayscale gradient channel, not original colour data.
3. **Three independent gradient-boosted tree experts:** small candidate area <1024px²; medium [1024,10000); large >=10000. Size routing is solely from proposal dimensions (image-observable inference), never target/GT size. Each LightGBM has fixed 210 rounds, LR=.045, 15 leaves, regularization=4, seed 9440+scale index. No VAL hyperparameter search.
4. **Independent confidence calibration:** image-disjoint *TRAIN* calibration proposals; one unweighted sigmoid logistic model per expert. Expert tree models do not see calibration images. Calibration uses their GT solely for calibration fitting; validation labels are not used for model fitting.
5. **Four-dimensional bounding-box regression:** supervised ExtraTrees head (80 trees, depth12, leaf>=4) trained on *TRAIN-core* region/GT matches with IoU in [.35,.95), features are 176D learned object/context descriptors + 5D original geometry + calibrated confidence. Offsets constrained as in E32; no GT used for prediction.
6. **Crowd-aware Soft NMS:** Gaussian confidence decay under image-only candidate overlaps; smaller penalty when *both* candidates are small, retaining plausible neighbouring bird detections. No GT-dependent crowd flags, threshold search, or use of official test split.

## Fair evaluation protocol

Random seed 9440: **60 eligible TRAIN images**, divided into 45 TRAIN-core classifier/regressor images and **15 disjoint TRAIN-calibration images**. Fixed **30 VAL images** with official COCO single-class bbox AP50 (maxDets100). Identical GT-blind proposal/feature banks and identical fitted RBF stage1+2 comparator on the 45 TRAIN-core images. Exact staged ablations:

| Arm | Three trees | Sigmoid calibration | TRAIN-only box refiner | Crowd Soft NMS |
|---|---|---|---|---|
| RBF baseline | No | No | No | No |
| Three experts raw | Yes | No | No | No |
| Three experts calibrated | Yes | Yes | No | No |
| Calibrated + refined | Yes | Yes | Yes | No |
| Calibrated + refined + crowd Soft NMS | Yes | Yes | Yes | Yes |

The score is official COCO bbox mAP@IoU .5; also top20 correctly ranked birds, GT-small and crowded top100 recall, raw proposal recall, preprocessing and original prediction-image panels for known failure scenes. Original colours/labels are preserved. If a stage fails, retain its measured negative result and avoid promoting it based on patch accuracy or raw recall.

**Predeclared gate:** +2.0 absolute percentage points COCO mAP50 compared to the paired RBF baseline, no loss of top20 TP, no more than 3pp small-bird or crowd recall decline. Even a passing 30-VAL result remains exploratory until a larger complementary VAL study. The official TEST 240-image partition has *already been used* during E33, so this experiment must not evaluate on TEST or treat any after-test development as a new independent full test.

## Relevant prior lessons

E34/E39 boosting alone reduced mAP; E32 regression improved small/crowded scenes but worsened large swans; E35/E38 small-bird extra proposals could lower AP, and E36 showed cap-order instability. The three-tree E40 design addresses **specialization, calibration, box geometry and crowd suppression as separate testable stages**, not an indiscriminate parameter sweep.

All image preprocessing and augmentation inspection from E39 is retained; E40 intentionally avoids E39 augmentation as the boosted augmentation arms had worse ranking mAP than their nonaugmented counterparts. Reintroduce augmentation only if a separate TRAIN/VAL ablation supports it.
