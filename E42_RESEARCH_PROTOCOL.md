# E42 full-training classical RBF SVM + augmentation ablation

Run on **all official TRAIN images with at least one annotated bird** (up to 1,369 images; exact count printed), and **all official VAL images**. Do not touch TEST. This is an independent full-scale training experiment alongside E40/E41, not a replacement of those branches.

## Frozen paired design
- Deterministic E37 Selective Search (1200 candidate cap per image), same proposals and fixed 0.35 NMS for both arms.
- First-stage classical linear HOG/LBP SVM with GT+jitter positives and safe negatives, followed by one TRAIN-only hard-negative mining pass. Image-wise streaming avoids retaining millions of descriptors in RAM.
- Second-stage baseline: HOG+LBP PCA64, 32D colour/context and stage-one margin; TRAIN-only scaling, RBF SVC(C3,gamma scale,balanced, probability True).
- Augmented second-stage: **exact same TRAIN-fitted PCA and scaler** as baseline, identical original examples, plus capped TRAIN-only flip and mild LAB/HSV colour/brightness augmented positive and negative crops; re-extract features and stage-one margin from transformed whole images.
- Compare all VAL images on COCO 101-point bbox AP50, top20 TP/FP, raw proposal coverage and original-photo GT/TP/FP overlays. Save a real before/after augmentation preview.
- Augmentation must improve actual detection AP without material loss of top20 TP to be promoted. If it fails, retain full TRAIN unaugmented RBF.
- The full TRAIN subset excludes zero-annotation scenes because annotations are known incomplete. This is **all usable annotated-positive TRAIN**, not a claim of using every image regardless of label quality.
- Full TRAIN training and RBF calibration may take substantial compute. Workflow timeout 360 minutes. A failed run must not be represented as a measured model result.

## Geometric correctness
Horizontal flip maps native x coordinates `[x1,x2] -> [W-x2,W-x1]` and flips the whole image. LAB luminance/HSV saturation adjustment keeps boxes unchanged. Both classes augmented. No TEST data or VAL images enter model training.
