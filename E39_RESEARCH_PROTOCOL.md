# E39 — Scale-aware classical boosting, preprocessing and augmentation audit

**Status:** First exploratory experiment running, results not yet verified. **Branch:** `research/e39-scale-aware-boosting`. **Workflow:** [E39 GitHub Actions](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e39-scale-boosting.yml).

## Research question

The RBF two-stage classifier trained with 200 of 1,369 images reached 11.89% on the official full 240-image TEST. Its errors include partial-bird boxes, water/reeds/seaweed/drone backgrounds, small/crowded birds, and poor classification despite strong training-patch accuracy. The E34 attempted to replace its RBF stage2 with LightGBM using the same old HOG/LBP representation and worsened mAP. **E39 does not repeat E34**: the key innovation is a more spatially structured, geometry-preserving handcrafted representation with separate boosted models for small/other bird proposal scales, not a bare learner swap. It is strictly classical ML, and **only TRAIN and VAL are accessed**.

## Preprocessing specification and validation gates

- Read image in native 8-bit BGR; assert readable, nondegenerate, and finite pixel coordinates.
- Preserve every proposal's native aspect ratio; letterbox **64×64** with neutral median-colour border and a binary mask. **Never distort a wide bird into a square.** Keep native width, height, aspect, position and scaled context.
- CLAHE grayscale contrast applies to the **gradient descriptor only**, not to the native RGB/HSV colour channel nor to proposal segmentation. This avoids corrupting chromatic information; mild lighting variation is used only for augmentation.
- Extract masked 4×4 ×6 orientation histograms, 4×4 HSV/LAB colour/micro-contrast channels, masked local binary pattern histograms, native geometry, Canny edge density and comparison with the surrounding scene.
- Self-tests require aspect-preserving letterboxing, valid mask, deterministic feature reproduction, fixed feature dimension, no NaNs/infs, correct tiny/border images and rejection of invalid zero-area boxes.
- Emit **real source versus letterboxed** crop contact sheets and preprocessing JSON with tiny/native crop counts. FAIL pipeline before training if any mandatory condition fails.

## Augmentation — train only, audited separately

For balanced capped TRAIN candidate-positive and safe-negative proposal samples, apply (i) a horizontal flip to the **entire image**, mapping `x1' = W-x2`, `x2' = W-x1`, and (ii) conservative photometric variation on whole-image LAB luminance and HSV saturation without altering boxes. Reextract descriptors from the actual augmented image and mapped proposal so outer context and spatial features remain self-consistent. No rotation-based bounding-box errors, vertical flip, synthetic foreground masks, extreme colour shifts, upscaling beyond the existing 64×64 presentation, or validation/test augmentation. Preview **raw source crop beside the augmented crop** and assert exact mirror crop equivalence and box validity. Include both positive and safe negative proposals to avoid trivial brightness/augmentation label leakage.

## Fixed controlled arms and acceptance

Fixed **45 annotated-positive TRAIN /30 VAL** image IDs and the *same E37 deterministic Selective Search GT-blind candidate bank*:

1. Incumbent TRAIN-only HOG/LBP→linear SVM + PCA64+aux RBF, NMS .35.
2. One global LightGBM on handcrafted 176-D letterbox/appearance/context features, no augmentation.
3. Two boosted experts gated by *candidate box area* (<1024 vs ≥1024 px²) with 50/50 global/expert score combination, no augmentation.
4. Global LightGBM with fixed TRAIN-only flips and modest LAB/HSV changes.
5. Two boosted experts with the identical TRAIN-only augmentations.

Each boosted learner uses frozen 210 trees, LR=.045, 15 leaves, min_child14, reg_lambda4, balanced class weights, seed 9444. Global and experts see the same underlying TRAIN candidate examples, subject to expert area bin. No hyperparameter sweep or TEST score reuse.

Compare COCO 101-point bbox AP50, top20 TP/FP, small/crowded top100 recall and actual original-resolution prediction overlays. A proposed arm is exploratory-only unless it improves the paired incumbent by >=2 absolute mAP points with no top20 TP loss and <=3pp small/crowded recall loss; independent validation is needed before promotion. **The official TEST split has already been used in E33, so it is not available as a fresh independent test of E39.** Never report E39 as a new full TEST result.

## Prior failures explicitly avoided

E19 and E38 showed extra proposals can reduce mAP even when raw recall rises. E34 showed tree classifier replacement alone reduced mAP. E23/E27 showed indiscriminate extra negatives can hurt. E36 exposed stochastic first-1200 candidate selection; E39 derives from E37's deterministic cap. Train patch accuracy is diagnostic only; class prediction is evaluated with final ranked detection AP, not training accuracy. Every successful or failed result should be written to a research log and include real image review.


## E39 first cloud run failure and correction — 16 Nov 2026

[Failed run 43852165545](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43852165545) ran all synthetic preprocessing and geometry tests, generated 42,079 TRAIN and 33,612 VAL feature vectors (176-D, zero nonfinite), and generated 2,118 TRAIN-only augmented patches (1,038 positive, 1,080 negative). It then failed **before validation scoring** with `ValueError: not enough values to unpack (expected 4, got 3)` at `fit_boosters`: the caller had intentionally passed `augmented[:3]`, while the fitting function incorrectly requested four values. This is a programmer interface-contract mistake, not evidence that preprocessing or LightGBM performs poorly. No mAP for E39 is available from that failed run.

**Fixed:** `fit_boosters` now explicitly accepts exactly three arrays (features, class labels, native candidate areas) with length, shape and finite checks before model fitting; added self-test contract check. The [E39 rerun](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e39-scale-boosting.yml) uses identical predeclared TRAIN/VAL partitions, features, augmentation modes and classifier hyperparameters. Do not tune the model based on this integration error. Keep failed-run evidence as an audit trail.


## E39 completed rerun — actual measured evaluation + reviewed original photos (16 Nov 2026)

**Successful GitHub Actions run:** [43853411880](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43853411880), commit `52e0133bdf1cda158121ca33a9b930fd485a8eec`. [Measured JSON, per-image CSV, original annotated photos and preprocessing/augmentation preview artifact 14219782538](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43853411880/artifacts/14219782538). Dataset TRAIN **45 annotated-positive**, VAL **30** with **156 official bird GT boxes**. **TEST not accessed.**

### Preprocessing verification

- TRAIN: 42,079 extracted proposals, **176 finite features per candidate**, **2,804 crops with native minimum side <8 pixels**, zero reported nonfinite descriptors or unreadable source images.
- VAL: 33,612 proposals, **1,680 native minimum side <8 pixels**, 176 finite features per candidate.
- TRAIN-only augmented: 2,118 patches (**1,038 positives, 1,080 negatives**) using complete-image mirrored boxes with checked x-transform and mild LAB-luma / HSV-saturation variation.
- **Actual preview inspection:** `preprocessing/raw_vs_letterboxed.jpg` has correctly retained aspect ratio with neutral border; however images contain very tiny birds/water patches where native pixels remain impoverished. `preprocessing/train_augmentation_examples.jpg` shows horizontally flipped birds with aligned crop geometry and mild photometric change without spatial warping. This establishes correct *geometric* augmentation, **not** that augmentation helps recognition.

### Equal-candidate held-out VAL outcomes (COCO 101pt bbox AP50, maxDets100; each top20 count out of 600)

| Method | VAL mAP@.5 | top20 TP | top20 official FP | small top100 recall | crowded top100 recall |
|---|---:|---:|---:|---:|---:|
| **TRAIN-only RBF PCA64 incumbent** | **14.2705%** | **83** | **517** | 54.35% | 42.31% |
| global LightGBM, spatial 176-D | 12.9847% | 69 | 531 | 56.52% | 50.00% |
| two LightGBM scale experts, 176-D | 12.7272% | 65 | 535 | 56.52% | 46.15% |
| global LightGBM, with TRAIN-only augmentation | 10.6975% | 71 | 529 | **65.22%** | **61.54%** |
| two LightGBM scale experts, with augmentation | 11.0765% | 62 | 538 | 58.70% | 50.00% |

**Gate result:** all four boosted-model arms FAILED >=+2pp mAP/no top20 loss predeclared selection requirement. Keep the RBF incumbent on this identical deterministic SS bank. **Augmentation improves small/crowded top100 recall but worsens ranked AP and/or top20 TPs**. Do not promote augmentation or boosted experts based only on increased recall. E34 had likewise shown LightGBM could underperform RBF on the old descriptor; E39 confirms richer spatial handcrafted features + fixed boosting hyperparams do not solve overall precision.

### Specific actual annotated scene comparisons, manually reviewed

- `beach_bird_1452.jpg`, GT26, dense near-water birds: RBF **4** top20 TP, global boosted **5**, two experts **4**, global augmented **7**, two augmented experts **4**. The improved augmented crop model finds more shorebirds but still draws large red surf/sand false regions.
- `beach_bird_1389.jpg`, GT16: RBF **12**, global boosted **6**, two experts **4**, global augmented **8**, two augmented experts **5**. A major counterexample to universal augmentation benefit.
- `beach_bird_1399.jpg`, GT2 large swans: RBF **1** top20 TP, ALL boosted arms **0**, repeatedly ranking partial-object/texture boxes instead.
- `beach_bird_1419.jpg`, GT5 reed/vegetation clutter: RBF **0**, nonaugmented boosted global/experts **1**, augmented boosted variants **0**. Poor general background discrimination persists.
- `beach_bird_1560.jpg`, GT7: RBF/global/experts **4**; global augmented **2** and expert augmented **3**. Nested partial-bird boxes continue.
- `beach_bird_1606.jpg`, official GT0 with a drone/background confuser: every arm produces **20 official FP among top20**.

### Decision / next work

**No architecture change or TEST retest.** The main learned insight is that added boosted capacity and geometric-safe augmentation partly improve small/crowded recall, but candidate confidence ordering/precision becomes worse. Retain the classical RBF model as primary; future changes should investigate more discriminative context and calibrated rescoring, training on a wider TRAIN subset, foreground/background quality and the *native information limit* for <8px bird patches. Compare on frozen larger VAL; since E33 already evaluated the official TEST, any later development is post-test exploratory, not a new independent final test.
