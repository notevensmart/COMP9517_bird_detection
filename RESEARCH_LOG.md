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


## Stage K — Ground-truth crop correction, actual bird-box visuals and multiscale ablation (16 Nov 2026)

**Controlled paired 100 TRAIN/30 VAL, 156 GT birds:** E17 measured old LoG/Otsu proposal-positive HOG+LBP+SVM **8.03% COCO mAP50**. Replacing proposal positives with GT+jitter train-only positive crops increased to **9.54%**, and mining TRAIN full-image post-NMS false alarms for 2 rounds reached **11.76%** mAP50 (12.65% under exploratory larger NMS=.65), with 1,196 final FP and 83 FN at margin0. A training-set classifier score ≈99% did not imply viable detection. Numerical pipeline completed successfully and visual artifact was uploaded in [run 43797610003](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797610003); the workflow status failed **only at optional post-experiment GitHub push due a non-fast-forward conflict**, not in training/metrics/visual generation. Raw output [artifact 14194939019](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797610003/artifacts/14194939019). Fixes to the previous visual script's `GT`/ `gt` key mismatch were tested; real annotated JPEG overlays were independently inspected.

**Visual diagnosis:** Existing LoG/Otsu uses sigma<=18, factor<=2.4, ratios<=1.5. Max width for symmetric blobs ~106 px; the original image `beach_bird_1399.jpg` contains two very large swans with bounding boxes spanning hundreds of pixels. The model produces 30 false small boxes on their textures and misses both actual GTs. Other visual examples show large false-box clusters on beached debris, breaking waves, sand and bird parts. Some empty-GT scenes, especially `beach_bird_1435.jpg`, visually contain genuine birds; 33 of 291 VAL images have zero labels overall. Manual annotation quality check required; don't edit official labels.

**Scale intervention E18:** add image-edge-ranked 96..512 large object windows with five aspect ratios and spatial quota, unioned with LoG/Otsu small-bird candidates. On same 30 VAL images, proposal recall@.5 rises **73.08%→90.38%** (large-bird 54.24%→93.22%); mean proposals 959→1920/image; recall@.75 19.23%→30.13%. Actual overlay of swan now shows whole-object candidate overlap. [Successful proposal run 43798486226](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798486226), [artifacts](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798486226/artifacts/14195192223). **Recall is not mAP**.

**End-to-end E19:** retrain same GT+jitter + SVM + full-image mining on hybrid proposal pool. 100 TRAIN/30 VAL (same images). Real COCO mAP50 at NMS=.35 for mining rounds 0/1/2: **4.77%, 7.52%, 6.18%**, respectively; mAP50 at NMS=.65: 5.26%, **9.22%**, 7.54%. FP @margin0 rounds 0/1/2 = 3500/2842/2199; FN=65/76/78. [Successful run 43798829411](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411), [visual artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411/artifacts/14196090985). Visually, expanding large proposals creates many high-scoring red FP rectangles around rocks, waves, water texture and vegetation; mAP is **worse** than smaller candidate pipeline despite 90.38% proposal recall. Don't deploy hybrid as is.

**Scientific interpretation:** Two separable bottlenecks. (1) **Geometry**: the original proposal generator misses medium/large birds, verified with real overlay and corrected proposal-only test. (2) **Classification/ranking**: when fixing geometry, the current single HOG+LBP SVM cannot distinguish expanded large bird proposals from varied large background patches; adding a thousand new proposals increases false detections and depresses mAP. Fully resolving this would require *scale-conditioned classifiers and properly calibrated cross-scale confidence*, plus training-only hard negatives specific to each size, and possibly stronger colour/shape/boundary descriptors. Neither a better box oracle nor 99% training-patch accuracy implies strong final mAP. A dataset annotation-quality check is also needed. Report both the successful geometric correction and failed end-to-end integration.


## E23 — Quality-checked small-set retrain, no full-scale promotion (16 Nov 2026)

User correctly identified risk of false negative patches from missing TRAIN bird labels. Manual inspection found at least two zero-GT TRAIN files containing bird-like silhouettes (`beach_bird_0690.jpg` and `beach_bird_1343.jpg`), in addition to two earlier flagged. Quarantined those exact files via `src/label_quality.py`. For the fast E23 gate, *all* zero-GT TRAIN images were excluded, including unreviewed images, leaving 45 randomly sampled **bird-annotated** TRAIN images with 308 GT birds. VAL remained the fixed independently sampled 12 images (77 GT birds), same COCO bbox mAP@.5, same 1,232 GT+jitter positive HOG+LBP features, same LoG/Otsu candidate bank.

A = random 20 safe negatives/image + two post-NMS mining rounds. B = spatial/scale-diverse 20 safe negatives/image + two rounds with image cap and overlap suppression. Measured A round0/1/2 COCO mAP50 = **4.07/6.30/5.24%**; B = **2.69/4.49/5.39%**. At predeclared round2, B gains **0.146 percentage points**, with 640 vs 646 FP at margin0 but **unchanged 216 false positives among top-240 predicted boxes** and same 42 missed GT birds. **Failed the acceptance gate**; no full-set run recommended. The original baseline's round1 is better than either method at round2, demonstrating non-monotonic mining.

Real visual examination of both model contact sheets: `beach_bird_1379.jpg` background seaweed generates many false boxes, `beach_bird_1419.jpg` shoreline vegetation produces dense false alarms and missed waterbirds, `beach_bird_1560.jpg` produces many wrong nested sub-bird boxes, and `beach_bird_1606.jpg` textured structures produce ~80 detections despite GT=0. The two contact sheets look qualitatively almost identical, consistent with negligible mAP difference. No visual claim of successful negative correction is warranted.

[Completed Github Actions E23](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43804273592); [JPEGs + numeric CSV/JSON artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43804273592/artifacts/14196769350). Code: `src/quality_subset_train.py`. **Key distinction:** Quarantine is applied, but E23 does not causally isolate *the quarantine* from the old experiment. The immediate obstacle is background-versus-bird *ranking*, not merely negative-sample diversity. Next work should examine high-scoring false patches and true positives as frozen crops, with a background rejection gate, before wider scaling. No TEST used.


## E24/E25 focused false-positive reduction and strict confirmation (16 Nov 2026)

After E23's spatially diverse negatives failed the improvement gate, trained a second-stage strictly **classical RandomForest** on real TRAIN proposals using stage1 HOG+LBP SVM margin plus 32 custom HSV/texture/edge/geometry attributes. Stage2 positive proposals have IoU>=.5 to TRAIN GT; negative proposals have IoU<.1 and low containment, including high-scoring post-NMS false alarms. TRAIN has 45 annotated-bird images, excludes all zero-annotation TRAIN (avoids known missing-label negatives); VAL/TESТ never used to fit stage1 or stage2. Stage2: 633 proposal positives, 706 mined hard negatives, 512 random safe negatives, covering 43 TRAIN images. Score blending with original SVM reorders candidate predictions.

**Exploratory E24 12 VAL / 77 birds:** Single SVM = 6.304% COCO 101pt mAP50, 21/240 true in top20/image; blend alpha .25 = 9.373%, 28/240; **alpha .50 = 9.602%, 29/240**; alpha .75 = 6.263%, 23/240. Chose alpha=.50 with validation acknowledgement; increased alpha hurt, showing overly aggressive rejection loses useful ranking. Raw FP @score>=0 drops 909→471 but score thresholds differ, so the real metrics to compare are COCO mAP and top20 rank. [Cloud run E24](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805068857), [images and CSV](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805068857/artifacts/14198260647).

**Fixed E25 verification (without retuning) 30 VAL / 156 birds:** stage1 mAP50 **3.375%** / top20 TP 38 of 600; fixed alpha=.50 cascade **6.499%** / top20 TP 56 of 600. +3.124 pp absolute mAP, +18 top-ranked true boxes, and top-ranked FP down 562→544. Score>=0 FP 2069→1056 but scores have different ranges, not a fixed operating-point comparison. Per-image top20 bird detections improved on 13 images, equal on 16, worse on 1. [Successful run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805517540) and [artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805517540/artifacts/14197594526). Original images personally inspected; major unresolved scenes: zero-GT seaweed 1379, shoreline vegetation/reeds 1419, nested partial-bird proposals 1560, and drone-like negative 1606. They still contain many false red boxes in top20, even after cascade. This is an actual meaningful, confirmed improvement in ranking under two validation subsets, **not** close to 50% mAP nor proof of full-dataset generalisation. Need targeted, trustworthy background negatives and better shape/part/box geometry before any claim of resolved FP issues.


## E31 — Controlled classical classifier comparison (16 November 2026; launched, results pending verification)

**Objective:** Diagnose whether the weak HOG+LBP detection ranking is primarily a classifier limitation rather than another proposal/negative-mining problem. Four strictly classical alternatives are evaluated on exactly the same precomputed proposal/feature banks: Linear SVM, RBF SVM, Random Forest and LightGBM. See `src/classifier_benchmark.py` and [GitHub Actions E31](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e31-classifiers.yml).

**Predeclared protocol:** 45 TRAIN images with at least one GT box (seeded `training_subset`), 30 fixed VAL images (`data(...,"val",30)`), same LoG/Otsu proposals, unchanged 64x64 HOG+LBP features, positives from TRAIN GT+jitter with IoU>=.55, 20 safe random negatives per TRAIN image (IoU<.1, containment<.5). All four methods see identical X/y and VAL candidate boxes. Stage-specific changes, hard-negative mining and validation hyperparameter searches are deliberately **disabled** to isolate the classifier variable. All zero-GT TRAIN images are excluded because earlier manual QA discovered missing bird annotations. Official VAL labels are preserved, and TEST remains untouched.

**Fixed models:** LinearSVC(C=.01), RBF SVC(C=1, gamma=scale), RF(250 trees, min_samples_leaf=2), LightGBM(250 trees, learning rate .05, 15 leaves, min_child_samples=15). Class balancing and fixed seeds applied where supported. Score ordering uses SVM decision values or tree positive-class probabilities, respectively; these ranges are not interchangeable. **Primary metric:** COCO 101-point single-class bbox AP50, maxDets=100, NMS=.35. Secondary: top-20 one-to-one matching, scale-stratified top-20 recall, crowded-image recall, training time and per-image scoring+NMS latency. These are exploratory VAL-only metrics; no test claims.

**Critical lessons incorporated from prior experiments:**
- E14: training-data increases plateaued, so this is a paired diagnostic rather than simply expanding to all TRAIN.
- E17/E19: high proposal recall can reduce actual AP due to false-alarm ranking; preserve same proposal bank for all models and evaluate **detection** AP, not geometric candidate coverage.
- E23: more/diverse negatives hurt or barely helped; do not rerun uncontrolled mining or use suspicious unannotated TRAIN scenes as trustworthy negatives.
- E24/E25: RandomForest *second-stage* cascade improved top-ranked results. A standalone RF here is **not the same intervention** and must not be described as a reproduction.
- E30: separate small/medium/large and dense-scene diagnostics; note poor IoU=.75 localization cannot be fixed by classifier changes alone.
- Rank-based AP is comparable across methods; counts using raw SVM margin >=0 versus tree probability >=0 are **not** comparable. Avoid claiming smaller FP from thresholds on incompatible scores.
- Prior GitHub push conflicts were unrelated to model quality: upload generated artifacts through Actions instead of writing back to the branch.
- Historical experiments' training populations and feature samplers differ. Do not compare raw scores across E31 and E25 as a causal paired test.

**Status:** Experiment script and cloud workflow committed; **numerical results must only be added after verifying successful execution and reviewing the artifact**. If a method fails, preserve its traceback and partial results. Do not choose a model for full training solely from one small validation subset; confirm on larger frozen VAL sample and visually audit its characteristic failures first.


## E31-C completed — four classical classifier algorithms, fixed 45 TRAIN / 30 VAL, real-image QA (16 Nov 2026)

[Successful E31-C run 43820800202](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43820800202), [actual full-resolution model prediction artifact 14204626929](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43820800202/artifacts/14204626929), [review with reproducibility details](CLASSIFIER_COMPARISON_REPORT.md). This is classifier-only **E31-C**, distinct from the existing **E31-SS** Selective Search architecture study.

Four algorithms used **exactly the same** LoG/Otsu candidate bank, 1924-D resized 64x64 HOG+LBP descriptors, 1232 GT+jitter positives, 846 safe TRAIN negative patches, IoU=.35 NMS and 30 VAL images/156 official GT birds, with no hard mining or hyperparameter sweep and no TEST access. Official COCO 101-point single-class bbox mAP@.5 (maxDets100):

| Standalone model | mAP@.5 | TP/600 top20 | FP/600 top20 | mean scoring + NMS sec/image |
|---|---:|---:|---:|---:|
| Linear SVM | 4.1925% | 39 | 561 | .0194 |
| RBF SVM | 5.4572% | **46** | **554** | .7890 |
| Random Forest | 5.8840% | 45 | 555 | .0657 |
| LightGBM | **6.5644%** | 44 | 556 | .0218 |

**LightGBM improves AP +2.37pp vs Linear SVM in this paired small-subset comparison, but no variant is good enough.** RBF has slightly better top20 ranking TP count, while much slower at inference. All models achieve ~99–100% training-patch accuracy and yet rank >550/600 top20 boxes as FPs by official GT. On the one sampled crowded image `beach_bird_1452` (26 GT), only Linear SVM finds 6 birds in top20; **RBF/RF/LightGBM find zero**.

**Actual visual inspection of eight original-image side-by-side comparisons:** `1379` gets 20 seaweed/debris false alarms per model; `1399` both large swans are missed while all models score beak/neck/body subparts; `1419` reed/vegetation FPs dominate and many waterbirds are missed; `1560` bird parts produce nested red detections; `1509` lone close-up bird misses GT while head/body subpart boxes score highly; `1606` drone parts are hard background confusers; `1452` shows serious loss of small crowded birds under all nonlinear models; `1435` **official GT=0 but visually contains many shorebirds**, so predictions marked official FP in that image are not all real semantic mistakes. Do not change official annotations; track QA separately.

**Decision:** Classifier-only experiments are not enough. Do NOT select LightGBM as a full-detector winner or automatically upscale to 1369 TRAIN. Learn from prior E14/E19/E23/E27 failures: more data/negatives or more geometric candidates did not solve appearance ranking. The *separate* E31-SS pipeline scored **17.18%** mAP on the same 45/30 size using Selective Search + nonlinear stage-2 handcrafted verifier, compared to its own matched LoG control **6.56%**. That proposal/alignment gain greatly exceeds the classifier-only improvement. **Prioritize the E31-SS branch**, verify with larger held-out VAL and study explicit geometry/scale/background errors, using TEST only after all choices are frozen. This is an evidence-based choice, not a claim of reaching 30%+ mAP.


## E32 — Failure-driven classical detector corrections (pre-registered 16 Nov 2026)

Based on E30/E31/visual audits, the strongest observed classical pipeline was **Selective Search fast + stage1 GT-jitter HOG/LBP linear SVM + stage2 PCA64 HOG/LBP + 32D context RBF verifier**, with **17.18% mAP50 on 45 TRAIN/30 VAL**. Main unresolved problems: false seaweed/reeds/drone rankings, entire swan/large-bird localization, small crowded birds, weak IoU75, labels with omitted birds, and small-sample overfitting. Prior E19 showed that extra raw proposals can hurt final AP, E23/E27 that extra indiscriminate negatives hurt, E14 that simply adding data can plateau. We will not repeat these as unmotivated standalone changes.

**Experiment E32 code:** `src/e32_failure_driven.py`, [cloud workflow](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e32-failure-driven.yml). Prespecified **100 annotated-positive TRAIN, 60 fixed VAL** images, TEST untouched. Exactly one Selective Search candidate/feature bank and one identical frozen pair of stage1+2 models for four detector variants: existing NMS=.35, relaxed NMS=.60, TRAIN-only ExtraTrees bounding-box regression + NMS=.35, and TRAIN-only regression + NMS=.60. Regression inputs 32 handcrafted context features + 5 normalized geometry + stage1 margin + RBF verifier score. Targets are relative GT center/size offsets from TRAIN IoU>=.35 and <.95 matched proposals only, with capped prediction shifts and clipping. VAL labels are not used in fitting or proposal generation.

**Primary gate:** at least +2 absolute percentage points COCO 101-point bbox mAP50 over paired baseline, no loss of total top20 TP, no more than 3 percentage points small-bird top100-recall loss, and no more than 3 percentage points crowded-scene top100-recall loss. If none passes, keep baseline. Record all branches even if they fail. Also record raw proposal recall at IoU50/75, per-size end-to-end top100 recall, TP/FP top20, actual original-resolution annotated prediction photos (`1379,1399,1419,1435,1452,1560,1606`) and per-image CSV.

**Important scientific caveats:** The selected variant is exploratory and **must be confirmed on a larger independent VAL partition before TEST**. VAL score selection requires a subsequent frozen check; there is no automatic launch of test evaluation. This only addresses NMS/box alignment, not fundamental semantic false positives; failed correction should be documented and the stronger baseline retained. Test metric must be reported only if the selected architecture is frozen and a once-only test evaluation is actually executed.


## E33 — Independent confirmation, frozen-model selection and full test protocol (pre-registration, 16 Nov 2026)

**Source:** `src/e33_frozen_full_test.py`. **Strict process:**
1. Use E32's fixed 60 VAL sample to choose at most one predeclared correction, only if E32's multi-metric acceptance gate passes. Otherwise candidate remains unmodified Selective Search baseline NMS=.35.
2. Fit the frozen Selective Search + two-stage RBF classical model on **200 seeded annotated-positive TRAIN images** (resource-aware larger training subset; not the full 1,369). If selected, also fit TRAIN-only ExtraTrees bbox refiner. No model parameters or thresholds are optimized using TEST.
3. Independently confirm the candidate on the **231 VAL images disjoint from E32's selected 60** using paired COCO mAP50, top20 TP, small-object and crowded recall. Because previous E31 studies repeatedly used VAL, this is disjoint from E32's development subset but **not guaranteed independent of all prior experimentation**. Reject candidate if mAP or top20 TP decrease or if per-size/crowding losses exceed predeclared tolerance.
4. Write a `model_freeze.json` decision **before loading any TEST** image/label. Only then run once on **all 240 official TEST images**, report COCO 101-point single-class bbox mAP@.5, training time, per-image inference cost, top20 TP/FP, per-size/crowding recall, and save original-image prediction overlays. This final TEST pass is descriptive, never used to tune anything.

**Caveat:** Increasing TRAIN from 100 (E32) to 200 is a predeclared training-size change; not a perfectly identical E32 fitted model. It is validated on the complementary VAL before TEST. The ability to run the full process in the GitHub Actions compute/memory budget must be verified. If any integrity gate or job fails, do not invent final test metrics. Report exact training subset and known label omissions in final research results.
