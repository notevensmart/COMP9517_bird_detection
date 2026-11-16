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
