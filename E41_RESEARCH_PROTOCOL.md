# E41 — Three size-specialized RBF SVMs (pre-registered experiment)

**Objective:** Determine whether three classical RBF SVM proposal verifiers, specialized by *candidate* size, outperform the single global RBF detector before deciding which model deserves full TRAIN1369 training.

**Strict classical CV/ML:** No neural nets, pretrained encoders, external training images or TEST usage. Branch `research/e41-three-rbf-experts`, based on E37's reproducible Selective Search candidate cap.

## Controlled comparison

- TRAIN: 100 seeded annotated-positive official training images; VAL: fixed 60 official validation images. Same Selective Search candidate boxes, stage-one SVM, TRAIN-only PCA64 HOG+LBP+32D context+stage-one-margin features, training selection and NMS protocol across arms.
- Global RBF SVM is trained by the incumbent `fit_models` pipeline, preserving the original baseline and score `0.5*stage1_margin+0.5*global_RBF_logit`.
- Three expert SVC(RBF,C=3,gamma=scale,class_weight=balanced,probability=True) models fit only on TRAIN proposal samples, routed by **proposal area**: small <1024, medium 1024–9999, large >=10000 pixels squared. Ground-truth size is never used to route a prediction.
- All experts share the global TRAIN-fitted PCA and standardization; they fit their own nonlinear classification boundary and TRAIN-only Platt probability conversion. An expert with <15 samples in either class falls back to global RBF.
- Pre-registered arms: (1) global RBF baseline, (2) three expert-only verifiers, (3) fixed 50/50 global+expert RBF logits, (4) fixed blend with TRAIN-only ExtraTrees bounding-box refinement, (5) fixed blend with relaxed IoU .55 NMS as a separate crowd-oriented ablation.
- All score variants use same image-only features; no GT boxes enter inference. The refinement and NMS arms are exploratory and cannot be credited to expert specialization.
- Validate on full COCO 101-point bbox AP50 (maxDets100), top20 TP/FP, size/crowded top100 recall, raw proposal coverage and actual 512×512 annotated output images. Emit side-by-side raw-vs-letterboxed TRAIN crop QA. The *actual classifier* continues to use the incumbent HOG+LBP preprocessing for fair comparison; the letterboxing check is an independent crop-audit, not a silent change to the baseline descriptor.
- Predeclared E41 promotion gate for 50/50 blend: >=+2 percentage points absolute mAP50 versus paired RBF, no loss of top20 TP, no >3 percentage-point loss of small or crowded top100 recall. Any promising but nonpassing change needs separate independent VAL confirmation; do not promote by selecting the best TEST score.
- If no expert configuration is promising, proceed with full TRAIN1369 global RBF baseline. If a specialist passes and replicates, train that architecture fully. Official TEST was already evaluated in E33; later official TEST reruns are *post-test*, not a fresh independent holdout.

## Important limitations

Size-specific training subsets may be sparse or class-imbalanced; fallback is explicit. SVC probability=True performs internal TRAIN-only probability calibration, but cross-expert probability comparability is imperfect. Crop visual audit is not evidence of improved detector mAP. ExtraTrees box regression previously did not reliably improve the incumbent; it is evaluated as an isolated ablation. Both larger training and genuinely new independent evaluation would be necessary for strong generalization claims.
