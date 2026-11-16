# E31-C: Controlled Classical Classifier Comparison — Measured Results and Visual Review

**Date:** 2026-11-16. **Experiment:** E31-C (classifier-only; distinguish from E31-SS Selective Search experiment). **Successful visual-audit run:** [43820800202](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43820800202). **Download actual visual overlays, full per-image CSV and metrics:** [artifact 14204626929](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43820800202/artifacts/14204626929). All eight real-image 4-model panels are under `visual_review/` in that artifact. The prior numerical-only run [43819441516](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43819441516) produced the same AP values.

## Exact paired protocol

- **45 TRAIN** images (only annotated-positive images, seed 61; excludes all zero-GT TRAIN images because some contain unlabelled birds); **30 official VAL** images (seed 43), **156 annotated VAL birds**; **TEST untouched**. TRAIN positives are GT crop + three jitters per box with IoU>=.55; negatives are the same 20 per TRAIN image, GT-safe by IoU<.1 and containment<.5. However, TRAIN images with GT can also contain missing annotations, so negatives are not perfectly verified.
- Each of the four classifiers receives the **same** 2,078 HOG+LBP (1924D) train patches (1232 positive, 846 negative), same fixed LoG/Otsu candidate geometry (~959 candidates/image), same 64x64 resize, NMS IoU=.35 with maximum 300 post-NMS candidates.
- **No hard-negative mining or validation hyperparameter search.** Linear SVM C=.01; RBF SVM C=1; Random Forest 250 trees/min_leaf=2; LightGBM 250 rounds/rate=.05/num_leaves=15.
- Primary comparable metric: independent COCO 101-point **single-class bbox mAP@IoU=.5**, maxDets=100. All `top20_TP` counts are one-to-one GT matches, IoU>=.5, among **20 highest-ranked detections per image after NMS**. SVM decision margins and tree class probabilities have incompatible scales, so fixed raw-score threshold FP counts should NOT be compared.
- No neural models, pretrained embeddings, or test-set use.

## Measured quantitative results

| Model | VAL COCO mAP@.5 | TP among top600 | FP among top600 | Train patch accuracy | Mean score+NMS seconds/image |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 4.1925% | 39 | 561 | 100.00% | 0.0194 |
| RBF SVM | 5.4572% | **46** | **554** | 98.75% | 0.7890 |
| Random Forest | 5.8840% | 45 | 555 | 99.95% | 0.0657 |
| LightGBM | **6.5644%** | 44 | 556 | 100.00% | 0.0218 |

Same-run empirical increase LightGBM minus Linear SVM = **+2.3718 percentage points**, not a universal performance claim. LightGBM has the highest **area under detection precision-recall**, but RBF yields the most true detections within the fixed top-20 operating region. Across the 26 GT birds in the only sampled crowded (>=20 GT) image, Linear SVM obtains 6/26 top20 matches while RBF SVM, Random Forest and LightGBM obtain 0/26. Tiny/medium/large recall must be interpreted together with class distributions, not as a stand-alone selection metric.

**Training patch accuracy of 98.7–100% is not meaningful evidence of end-to-end detection success.** The region proposal bank is inaccurate for whole large objects, and high-scoring textured backgrounds overwhelm all models.

## Real source-photo visual audit (personally inspected)

Four side-by-side panels were inspected for each of **eight** fixed VAL scenes, with blue=one-to-one top20 true positives, red=official-ground-truth top20 false positives, green=other GT boxes, yellow=missed GT boxes.

| Image | Observed output failure or difference |
|---|---|
| `beach_bird_1379.jpg`, GT 0 | All four classifiers assign 20 high-ranked boxes to piles of seaweed/washed-up debris. These are background false positives against plausible visible scene content. |
| `beach_bird_1399.jpg`, GT 2 | Both enormous swans are missed by all four top20 lists. Numerous red boxes isolate beaks/necks/parts rather than enclosing the actual GT bird extents. This is a **proposal localization failure plus ranking issue**, not simply a classifier type issue. |
| `beach_bird_1419.jpg`, GT 5 | All algorithms rank reed and vegetation texture highly: Linear 0/5, RBF 1/5, Random Forest 2/5, LightGBM 1/5 top20; several real waterbirds remain yellow. |
| `beach_bird_1435.jpg`, official GT 0 | **Visible real shorebirds appear throughout the image despite no official GT annotations.** Each model's top20 output is marked red by the official label evaluator; many such boxes encompass genuine birds. Do not count these as verified semantic false positives in qualitative discussion; preserve official VAL annotations for comparable mAP and flag separately for label QA. |
| `beach_bird_1560.jpg`, GT 7 | Correct detections remain surrounded by overlapping, nested and part-bird false boxes; Linear 2, RBF 2, RF 3, LightGBM 2 TP in top20. Enlarging rank quality alone has not fixed duplicate/box alignment. |
| `beach_bird_1606.jpg`, GT 0 | All four rank 20 red boxes on a drone/aircraft and its components, an unmistakable hard background confuser. |
| `beach_bird_1452.jpg`, GT 26 | In one crowded distant-bird photo, **Linear SVM recovers 6** birds in the top20, whereas RBF SVM, RF and LightGBM recover **zero**, focusing on shoreline buildings, water patterns and wrong-scale background regions. More flexible classifier does **not** guarantee crowd robustness. |
| `beach_bird_1509.jpg`, GT 1 | Large close-up bird's box remains yellow/unmatched for all four; classifiers rank multiple small red boxes on the bird head/body/legs rather than the whole object. |

**Interpretation:** Three distinct causes remain:
1. **Proposal geometry**: LoG/Otsu produces part-object boxes rather than full large-bird candidate boxes; no patch classifier can repair absent GT-aligned proposals.
2. **Appearance/ranking**: repetitive natural (seaweed/reeds/waves) and manmade (drone) backgrounds score as birds, even for flexible tree/kernel learners.
3. **Annotation integrity**: at least one fixed VAL image `1435` visually contains genuine birds with zero official labels. Standard benchmark metrics count them as FPs; do not alter labels and do not pretend observed red boxes are all visually wrong.

## Context and research decision (learned from prior log)

- **E14:** More TRAIN data plateaued; don't just increase training sample or call 100% patch accuracy successful.
- **E19:** Bigger candidate pool raised proposal recall but **decreased** mAP; ranking must be re-audited after geometry changes.
- **E23/E27:** Additional or diverse negatives failed to improve top-ranked results; don't repeat indiscriminate negative expansion.
- **E28/E29:** Nonlinear RBF on PCA HOG+LBP and auxiliary shape features worked better **as a stage-2 verifier** (10.13% mAP on 100 TRAIN/60 VAL, not paired with E31-C).
- **E31-SS:** Paired **Selective Search + staged RBF appearance verification** scored **17.18% mAP50** versus **6.56%** in its paired LoG control on 45 TRAIN/30 VAL. It substantially improves large-bird localization; the unconstrained LoG+SS union reduced mAP to 11.35% despite higher proposal recall. This is a *different full pipeline*, not directly comparable to E31-C's standalone LightGBM and must not be presented as a classifier head-to-head.
- **Decision:** Record LightGBM as **best of four standalone classifiers by AP on this fixed small VAL subset**, not the new full-pipeline winner. Do **not** promote it to full training or claim solved detection. Keep Selective Search + stage-2 RBF as the leading **classical** architecture and prioritize localization, verified hard-scene negatives, scale-conditioned handling of crowded small birds, and evaluation on a larger predeclared VAL subset. No final TEST evaluation until selection is frozen.

## Reproducibility

- Experiment code: `src/classifier_benchmark.py`.
- Image audit: `src/e31_visual_audit.py`, full `visual_review/per_image.csv`, `visual_review/visual_audit.json`, and 8 full-resolution 4-model overlays in artifact above.
- Run artifacts are downloadable from GitHub Actions; retention is 14 days. These are actual model predictions on original source images, not illustrative drawings or generated fake examples.
