# Experiment registry — COMP9517 bird detection

This is a **paper-ready ledger** of all meaningful experiments and failures. Every claim must link to a reproducible GitHub Actions run or stored artifact. **Do not replace candidate recall with detection AP**, do not mix training and validation metrics, and do not call the continuous single-class AP50 implementation COCO AP50:95.

| ID | Experimental question | Method and conditions | Source | Observation | Status/decision |
|---|---|---|---|---|---|
| E01 | Can Canny yield enough boxes? | 20 seeded validation images, 109 birds | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43785783873) | 200-cap proposal recall@.5 7.34%; uncapped 32.11%; padding25 42.20%; quick Canny HOG/SVM val continuous AP50 0.086% | Reject capped Canny |
| E02 | Can Canny+HOG overfit three images? | 2 positive + 1 negative TRAIN images, 6 birds | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43785992697) | 0% / 16.67% / 33.33% train AP across capped/uncapped/padded; 100% classification accuracy on available patches | **Failed** detector overfit: missing boxes |
| E03 | Can multiscale anchors fix proposal coverage? | Same seeded 20 VAL images | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43786604315) | Dense 93.58% recall/296584 candidates; edge450 82.57%/11245; edge900 85.32%/20702 | Supports improved coverage; too many boxes |
| E04 | Can anchor+HOG overfit three TRAIN images? | Same 6 training birds | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43786808910) | Proposal coverage 100%; patch acc 100%; training AP50 12.63% edge450 or 9.42% edge900 | **Failed:** many false detections |
| E05 | Can hard-negative mining suppress backgrounds? | 3 TRAIN images, edge450 + HOG SVM | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43787231085) | Tiny train AP50 best 37.31%; mean positive-margin false detections 298.3 → 22.0/image by round 2 | Helps but overfit gate fails |
| E06 | Does global proposal pruning preserve birds? | Same 20 VAL images | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43787383739) | 400 or 800 boxes 0% proposal recall; 1600: 0.92%; 3200: 28.44% | **Reject global edge-score top-k** |
| E07 | Can scale+spatial quotas prune safely? | Same 20 VAL images, spatial proposals | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43787913098) | 1350/2700/5334 proposals: 71.56%/78.90%/83.49% recall@.5 | 5334 candidate budget selected for follow-up |
| E08 | Does spatial-pruned HOG+SVM overfit? | Same 3 TRAIN images, 6 birds, SVM hard negatives | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43788148237) | Proposal 83.33% recall, tiny TRAIN AP50 best 37.10%; stricter margin 1.0 lowers boxes but also AP | Failed overfit gate |
| E09 | Does LBP improve HOG classification? | Same 40 TRAIN /12 VAL images (77 VAL birds), spatial 1350, same hard-negative schedule | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43788975227) | HOG 3.73%, HOG+LBP 7.82% **exploratory validation** AP50 continuous | Improvement, but poor detection |
| E10 | Can published EdgeBoxes reduce proposals? | 8 VAL images, 58 birds; OpenCV structured edge forest (non-neural trained component) | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43789080416) | 1000 boxes: 84.48% proposal recall@.5; 2000: 93.10%; **not detection AP** | Useful reference, provenance must be disclosed |
| E11 | Can LoG blobs + local Otsu segmentation improve detection? | Same 35 TRAIN /12 VAL images, 77 validation birds; GT-blind proposals | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43790916222) | LoG HOG AP50 2.65%; LoG+Otsu HOG **9.26%**; Otsu proposal recall@.5 74.03%, mean 959 boxes/image | Improved localisation but many FP |
| E12 | Does rotation-pooled HOG/SVM improve classical detection? | Same 35 TRAIN /12 VAL subset, LoG and LoG+Otsu candidates | [Run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43790916222) | LoG+Otsu HOG 9.26%, rotation pooling 7.40%; LoG HOG 2.65%, rotation 1.45%. **Not Fourier HOG** | Reject simplistic rotation pooling |

## Metric definitions and reporting protocol

- **Candidates**: boxes from a GT-blind proposal method, before any supervised classification; most unmatched candidates are *not yet detector false positives*.
- **Candidate recall@IoU T**: fraction of annotated birds with >=1 candidate IoU≥T.
- **Candidate background boxes**: proposals whose maximum GT IoU<0.1; near-miss boxes: 0.1≤max IoU<0.5. This decomposition is descriptive, not detection FP.
- **Duplicate candidates**: proposal boxes with GT IoU≥0.5 beyond the first qualifying candidate for each bird. This is an approximate multiplicity count, **not** the standard precision/recall assignment.
- **Final detection TP/FP**: confidence-ranked greedy 1-to-1 matching per image at IoU≥0.5; unmatched predictions are false positives. Record background vs localisation near-miss vs duplicate overlap separately; no double-counting.
- **AP@0.5**: global ranking across all chosen images using the same implementation for all methods. Distinguish this continuous interpolated AP from COCO 101-point AP@0.5.
- **Empty scenes**: record candidate count AND number of final predicted boxes after threshold/NMS (the latter are genuine FP).
- **Guardrails**: Never use test GT to tune parameters. Freeze thresholds on validation before final test. Include images/birds sampled, seeds, code commit and artifact link.
- **Computational cost**: record proposal time, feature extraction time, classification time, and total inference time where available. Penalise methods that produce thousands of boxes even if proposal recall is high.

## Literature context for discussion

- Hodgson et al. (2018), *Drones count wildlife more accurately and precisely than humans*: rotation-invariant Fourier HOG with SVM and structured spatial suppression in wildlife counting. **Fourier HOG is a specific descriptor; generic rotation pooling is NOT a faithful reproduction.**
- Mader & Grenzdörffer (2016), *Automatic Sea Bird Detection from High Resolution Aerial Imagery*: local candidate regions followed by graph-cut segmentation.
- Felzenszwalb et al., *Object Detection with Discriminatively Trained Part-Based Models*: latent SVM and deformable HOG-based part filters.
- Cheng et al. (2014), *BING: Binarized Normed Gradients for Objectness Estimation*: compact gradient-based region proposal ranking. Reported proposal recall on VOC is not bird mAP.
- Zitnick & Dollár (2014), *Edge Boxes: Locating Object Proposals from Edges*: objectness scoring using edges. Our OpenCV implementation depends on a pretrained structured-edge model, so disclose learned edge-component provenance.
- LoG blob detection is a classical multiscale scale-space approach; local Otsu-based refinement is a **new test design here**, not a direct reproduction of a specific bird-detection paper.

## Paper discussion points

1. Why candidate recall is an upper bound on detection recall for a fixed proposal set, **not** on unrelated proposal methods.
2. Why perfect training-patch classification can coexist with very poor raw-image AP due to extreme class imbalance and redundant candidates.
3. Why naive top-k proposal pruning loses almost all birds; why per-scale spatial diversity matters.
4. Why hard-negative mining reduces false detections but can sacrifice recall when thresholds are raised.
5. Why different historical subset results are not paired statistical comparisons; use E09 and E11/E12 matched-split experiments for claims.
6. Why crowded scenes and tiny birds require box-level diagnostics rather than only image-level classification accuracy.

## E11/E12 false-box accounting — first classical methods run

Run [43790916222](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43790916222) used 35 TRAIN /12 VAL (77 VAL birds). All values below describe final detections with **SVM decision margin >=0** after NMS, rather than candidate boxes or AP scoring's full ranked list.

| Method | Validation continuous AP@.5 | Mean final boxes/image | Background FP | Localisation FP | Duplicate FP | Missed birds |
|---|---:|---:|---:|---:|---:|---:|
| LoG + HOG | 2.65% | 41.33 | 382 | 101 | 0 | 64 |
| LoG + pooled rotation HOG | 1.45% | 42.42 | 394 | 104 | 0 | 66 |
| LoG + Otsu + HOG | **9.26%** | 71.67 | 629 | 196 | 0 | 42 |
| LoG + Otsu + pooled rotation HOG | 7.40% | 76.58 | 695 | 186 | 2 | 41 |

Candidate-only results, before classification: LoG 720 boxes/image with recall50 44.16%, 7,559 background candidates and 965 near-misses; LoG+Otsu 959 boxes/image with recall50 74.03%, 9,582 background candidates and 1,663 near-misses (counts across all 12 images). **Do not call candidate-background counts detector FP.**

Research decision: Otsu region refinement improves proposal coverage and AP, but produces many false positives. Simple 90-degree rotation-pooled HOG worsens AP on this test; it is not an implementation of published Fourier HOG. Next: hold the Otsu candidate set fixed, compare HOG vs HOG+LBP and hard-negative mining; report final FP types, misses, AP50, compute time. This is an exploratory matched-subset result and does not imply test performance.


## E13 — Official mAP@0.5 validation and incorrect-box ablation (measured)

**Source:** [GitHub Actions run 43792119328](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43792119328) with archived `otsu_hog_lbp_map50.json` and per-image CSV. **Reproducible code:** `src/otsu_hog_lbp_map.py`, `src/metrics_map.py`. All methods trained with the *same fixed seed 35 TRAIN images* and were evaluated on the *same 12 VAL images (77 annotated birds)*; no test samples used. Proposal stage: **LoG multi-scale peaks + local Otsu segmentation**. Mean ~959.08 proposals/image for all arms; identical proposals used for all three descriptors/classifiers. All metrics here use the *official COCO 101-point bbox mAP@IoU 0.50* (maxDets=100 per image), single bird class; COCO API synthetic tests passed.

| Model | Primary VAL mAP@0.5 | Continuous AP@0.5 (secondary) | Output boxes / image (margin≥0) | Total FP (12 images) | Background FP | Localisation FP | Duplicate FP | FN birds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoG+Otsu → HOG+SVM | 4.60% | 4.46% | 85.42 | 994 | 785 | 209 | 0 | 46 |
| LoG+Otsu → HOG+LBP+SVM | 7.59% | 7.37% | 81.58 | 941 | 741 | 200 | 0 | 39 |
| **LoG+Otsu → HOG+LBP+SVM + 2 mining rounds** | **8.82%** | **8.41%** | **37.83** | **425** | **296** | **128** | **1** | **48** |

**Important:** FP/FN counts correspond to a **fixed decision margin of 0 after NMS**, while AP is computed from confidence-ranked detections with the COCO max-100 limit; don't treat these as identical operating points. There are also 163/161/59 false detections on blank images across the three arms. Average preliminary end-to-end inference time approximately 1.40 sec/image (Python/OpenCV/scikit-image CPU environment); observed training fit times 0.44/0.40/3.05 sec **excluding feature extraction time**. Feature extraction computed once per image and shared across comparison methods.

**Interpretation:** LBP improves background/foreground separation at a given region candidate set. Hard-negative mining further suppresses background FP (741→296) but also increases FN (39→48); this is a precision–recall trade-off requiring thoughtful threshold selection. Despite improvement, **8.82% mAP@0.5 is below target** and the small validation subset is exploratory rather than a final test result. No claim of >50% mAP is made.

**Methodology caveat:** Earlier E11's LoG+Otsu HOG *continuous* AP50 9.26% used a **different training sampler**; E13 HOG continuous AP50 4.46% is not a contradiction or matched comparison. Compare only the E13 rows against each other.

**Next question:** Can better localisation, colour/shape features, or actual rotation-invariant Fourier-HOG improve mAP without reverting to thousands of candidates? Prioritise improving the fraction of birds whose generated boxes achieve IoU≥0.5, and reducing high-scoring near-miss predictions. Validate with official mAP and exact FP taxonomy again.

## E14 — Controlled hard-example mining ablation (registered before results)

**Hypothesis:** Global top-500 hard-negative selection over-samples duplicate beach backgrounds; per-image/per-scale diversity, training-only hard-positive weights and a bounded balanced negative cache may improve precision without suppressing birds.

**Code:** `src/mining_ablation.py`; **cloud run:** [43792817123](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43792817123). Four variants under a shared LoG+Otsu proposal and HOG+4×4 uniform-LBP descriptor bank: (A) global top-scoring mining; (B) diversity-filtered hard negatives (per-image, area-bin and IoU controls); (C) B plus upweighting up to 120 lowest-margin training positives; (D) C plus bounded 1,400-negative cache (70% hard, 30% distributed reserve). Same 35 TRAIN /12 VAL image selection as E13; all receive three mining rounds. COCO 101-point bbox mAP50 with maxDets=100 is the PRIMARY metric, measured after **each** round. Additional diagnostics: unmatched final boxes (background, localisation, duplicate), missed birds, false detections on empty images, inference seconds and training sample mix. All training labels/weighting/mining use TRAIN only; any selection by VAL is exploratory and requires untouched TEST confirmation.

**Status:** Cloud execution in progress when this entry was created. Do **not** invent or prefill numerical results. Update after the completed workflow and logs are inspected.


### E14 measured results — matched classical mining study

**Run:** [43792817123](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43792817123) (workflow success). Code: `src/mining_ablation.py`. SAME 35 TRAIN /12 VAL images (77 VAL birds), SAME LoG+local Otsu proposals and HOG+LBP feature bank, fixed seeds, same COCO 101-point bird mAP@IoU=0.50 maxDets=100. Neither VAL nor TEST annotations were used for training or hard-sample selection. All four configurations were evaluated at rounds 0,1,2,3; no cherry-picking from different splits.

| Variant | Round 0 mAP | Round 1 | Round 2 | Round 3 (best) | Round-3 FP (margin≥0) | Round-3 missed birds |
|---|---:|---:|---:|---:|---:|---:|
| A Global top-500 mining | 7.59% | 3.74% | 8.82% | **11.73%** | **305** | 44 |
| B Diverse negative mining | 7.59% | 4.56% | 9.92% | 10.83% | 316 | **43** |
| C B + hard-positive reweighting | 7.59% | 4.26% | 9.97% | 11.64% | 333 | **43** |
| D C + bounded 1400-negative cache | 7.59% | 4.26% | 9.04% | 9.82% | 345 | 45 |

All FP counts above are **total final false-positive boxes across 12 VAL images at fixed SVM decision margin 0 after NMS**, and FN are missed GT birds at that threshold. Primary mAP uses globally score-ranked detections, not that fixed margin. The FP taxonomy / all intermediate rounds are in the run artifact `mining_ablation.json` and `mining_by_image.csv`.

**Important interpretation:** 
1. The *simpler global strategy with another training round* is the best validation mAP: from 8.82% after two mining rounds (E13) to **11.73% after three**, +2.91 percentage points. This is exploratory small-subset VAL and not final TEST performance.
2. Diversity-control was plausible but did **not** improve the primary metric here. Hard-positive weights nearly recovered the global mAP (11.64% versus 11.73%) but did not exceed it; the bounded cache degraded it.
3. Round 1 substantially reduced mAP compared with round 0 across variants, showing mining trajectories are nonmonotonic; record every round and never assume more mining always improves performance.
4. Hard-positive weighting in this code is **training-only low-margin positive reweighting**, not newly generating ground-truth boxes at inference. Its benefits may depend on C, weight, and dataset size; further tuning would require controlled confirmation.
5. The error counts remain high and Otsu proposal recall is a bottleneck. **Do not report 11.73% as full validation-set or test mAP**, and never imply the method meets 50%.

**Current decision:** retain A as the empirical small-subset winner; use more images / stable validation to test whether it generalises before spending time on bounded-cache sophistication. Classification and localisation are still limiting factors, and validation reuse means these exploratory choices need a frozen final test.

## E15 — Classical detector learning curve: TRAIN size versus VAL mAP@0.5

**Research question:** Was the previous 11.73% mAP@0.5 largely due to training on only 35 images, or does the classical proposal/classifier pipeline saturate?

**Cloud execution:** [GitHub Actions 43793628053](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43793628053), implemented by `src/learning_curve.py` and `.github/workflows/learning-curve.yml`. **At registration, the workflow was running.** Numerical results should only be appended after inspecting actual output.

**Protocol:** Strictly classical LoG+local Otsu proposals, HOG+4×4 uniform LBP features, balanced LinearSVC (C=0.01), global top-500 safe hard-negative mining per round for 3 rounds. Nested fixed-seed random TRAIN subsets of **35, 100, 300, 700, and entire available train split**; common **60 validation images** selected with seed 43. Both positive and safe-negative training descriptors are subsampled in a bounded, fixed fashion for CPU/RAM (max 24 matched positives, 55 safe negative candidates/image). This **changes train patch subsampling versus E14**, so the 35-image point will not necessarily equal previous 11.73%; comparisons within E15 are paired, E14-to-E15 are not.

**Primary dependent variable:** single-class COCO 101-point bbox **mAP@IoU=0.5** (maxDets=100), computed from NMS-scored detections. At each training size, use the same predeclared 3 mining rounds, with all-round metrics separately retained. **Secondary:** false boxes at fixed SVM decision margin ≥0 (background, near-miss localisation, duplicates), missed birds, total returned boxes, training sizes and fit time. All validation images stay fixed; TEST remains untouched.

**Artifacts to record:** `learning_curve_map50.png`, `learning_curve_errors.png`, `learning_curve.csv`, `learning_curve_all_rounds.csv`, `learning_curve_results.json` and `progress.json`. If runtime/memory fails, label partial results and the failure; do not extrapolate the graph.


### E15 measured learning-curve results — completed

**Run:** [GitHub Actions 43793628053](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43793628053), successful, with archived [PNG plots and raw CSV/JSON output](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43793628053/artifacts/14193349564). Dataset: 1,369 available TRAIN images and a *fixed* 60-image VAL subset containing 376 annotated birds. Nested TRAIN subsets sampled using seed 42, VAL seed 43, using the LoG + local Otsu candidates, HOG + spatial uniform LBP, balanced linear SVM (C=.01), three rounds of top-500 hard-negative mining. Bounded training-patch extraction (up to 24 positives and 55 safe negatives per image) makes this protocol **distinct from E14**. Every row is an actual COCO API 101-point **single-class bbox mAP@0.5, maxDets 100** measured on the same VAL images.

| TRAIN images | VAL mAP@0.5 | False-positive boxes across 60 VAL images (margin≥0) | Missed birds (margin≥0) | Final boxes per VAL image (margin≥0) | Training positive patches | Training negative patches in round 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 35 | 7.49% | 3,630 | 220 | 63.10 | 381 | 1,761 |
| 100 | 6.54% | 2,970 | 224 | 52.03 | 1,007 | 3,356 |
| 300 | 9.78% | 2,697 | 221 | 47.53 | 2,993 | 7,167 |
| 700 | 11.25% | 2,997 | 203 | 52.83 | 6,747 | 14,704 |
| **1,369 (full TRAIN)** | **11.29%** | **3,004** | **204** | **52.93** | **12,824** | **27,199** |

**Interpretation:** The learning curve is *not monotone* at small N: 35→100 decreases from 7.49% to 6.54%. Adding images beyond 300 increases mAP, but **700→1369 improves only ~0.04 percentage points** (11.24995%→11.29259%). This is evidence of a plateau for the current fixed descriptor/proposal/SVM/mining setup on this 60-image validation sample, **not proof that all classical methods cannot improve**. False-positive counts remain in the thousands at the margin-0 threshold; a threshold adjustment may reduce FP but can increase FN and must be validated separately. At the largest size: 2,078 background FP, 920 localisation FP, 6 duplicate FP, and 204 FN birds. Proposal box geometry and background discrimination remain dominant challenges.

**Figure citations in paper:** `learning_curve_map50.png` is the training-size vs official mAP50 curve; `learning_curve_errors.png` plots final false-positive boxes vs missed birds, both over the **same** VAL set. Exact underlying values are in `learning_curve.csv` and `learning_curve_all_rounds.csv` (every intermediate mining round). Neither held-out TEST GT nor other external pretrained models were used. This is exploratory VAL evaluation; freeze the final model and use the untouched TEST split for the final reported unbiased estimate.

**Decision:** Do not invest in simply adding more training examples to this unchanged pipeline. Prioritise improving the IoU of classical candidates (local segmentation, box refinement) and reducing hard background mistakes with more discriminative handcrafted features. Document the saturation as a scientifically relevant limitation of this particular traditional detector.


## E16 — Independent oracle-based debugging of low classical mAP (registered pre-results)

**Motivation:** User challenges implausibly low mAP despite 1,369 training images. Treat the issue as a falsifiable engineering/research hypothesis, **not** as an inherent classical-model limit.

**Independent cloud run:** [GitHub Actions 43795502852](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43795502852); code `src/diagnostic_oracles.py`. Audit 60 fixed VAL images using LoG+Otsu proposal generation and 100 TRAIN images for oracle-crop classification (no held-out labels used in fitting). Diagnostics include:
1. Original annotation geometry, bird-box aspect ratios, density and small-object counts; detect mismatch or dense scenes exceeding the COCO 100-detection per-image cap.
2. Proposal recall@0.5/.75, image-wise candidate coverage, IoU distribution; visual GT green boxes and matching blue candidate boxes.
3. **Perfect ground-truth-box prediction oracle** (expected COCO mAP50 ~1 if <=100 GT per image). If this fails, fix evaluation before using any prior results.
4. **Oracle proposal ranking** (GT-derived labels only for diagnostic upper-bound): raw candidates versus NMS IoU=.35 versus .65. These are NOT deployed validation mAP and must never be included in final performance numbers.
5. Classifier trained on TRAIN **ground-truth bird crops**, with TRAIN background negatives, then evaluated on exact VAL ground-truth bird crops, using both naive distorted 64x64 resize and aspect-preserving letterbox. This is a **recognition diagnostic**, not a detector metric.

**Known possible issues identified by static code review:** fixed maximum 120 LoG peaks/image, resize distorts aspect ratios, training samples few backgrounds compared with hundreds scored at inference, NMS IoU=.35 may remove adjacent birds, and COCO maxDets=100 can constrain crowded images. These are hypotheses until measured.

**Status at registration:** cloud run ongoing. Numerical results must only be copied from completed logs and archived artifacts. Preserve negative findings and any evaluation bugs separately from algorithmic limitations.


### E16 actual audit findings — VALIDATED after repairing diagnostic boundary clipping

Initial [failed audit](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43795502852) exposed a specific **diagnostic-only** crop bug: normalized YOLO labels close to x=0 may convert to x=-0.000256, and blindly applying `floor` then NumPy slicing produces an empty crop. Fixed clipping to [0,width]/[0,height] in oracle descriptor extraction and reran [successful audit 43795691581](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43795691581). Dataset: fixed 60 VAL images (376 birds); GT crop classifier used 100 independent TRAIN images.

| Independent diagnostic | Measured result | What this proves |
|---|---:|---|
| COCO mAP50 on perfect GT boxes | **100.0%** | Basic bbox conversion/evaluator correct for this sample; no scene has >100 annotated birds |
| Raw LoG+Otsu proposals R50 | **72.07%** (271/376) | A significant fraction of real birds has no matching IoU≥.5 proposal |
| Raw proposals with GT-assisted optimal scoring | **71.33%** oracle COCO mAP50 | Proposal set contains high-recall detections when scored with **validation ground truth**; NOT detector performance |
| Same proposal oracle with NMS IoU=.35 | **69.31%** oracle mAP50 | Current NMS loses ~2.03 points under GT-assisted ranking |
| Same proposal oracle with NMS IoU=.65 | **70.42%** oracle mAP50 | Less aggressive NMS recovers ~1.11 points versus .35 under oracle scores |
| GT crop HOG+LBP classifier on true VAL bird boxes (TRAIN GT crops / TRAIN negatives) | **280/376 = 74.47%** true birds margin≥0 | The feature/classifier can recognise many **perfectly cropped birds**. NOT detection recall or mAP |
| Aspect-preserving letterbox variant | **279/376 = 74.20%** | Simple letterboxing alone did not improve positive-only GT crop classification in this audit |
| Training-patch accuracy (GT crop test) | **100%** on 454 positive/1400 negative TRAIN patches | Training memorisation does not guarantee full-image precision, rank or localisation |
| Size and aspect properties | 136/376 GT birds <1024 px²; 25 wide (>2:1), 11 tall (<0.5:1) | Most are small, some have unusual aspect ratios |
| Existing E15 full-detector VAL mAP | **11.29%** | Very large gap between oracle capacity and actual ranked detections |

**Critical nuance:** 74.47% is *positive-only recall on GT crops* with all localization made perfect; it says nothing about false positive rate on full images. 71.33% is an *oracle* that illegally uses validation labels to assign scores, so it must **never** be called actual model mAP. The 11.29% is the authentic deployable model mAP under the same 60-image validation seed. No inference leakage was introduced into E15.

**Implications:** mAP formula itself is not the central failure. The likely bottleneck is poor *relative classification score ranking* on candidate crops amid ~957 candidates/image, coupled with missing candidates for 28% of birds and insufficient overlap at IoU .75 (only 17.29%). More hard-negative mining cannot repair localisation or crop-shift alone. Systematically test training from GT positives with jittered realistic positive crops, training against mined *post-NMS* false alarms, and score distribution / precision-recall diagnostics, before declaring classical methods intrinsically weak. NMS=.65 is a testable, small secondary gain. Negative-mining diversity and letterboxing alone did not deliver evidence of a major fix.

**Artifacts:** [run output with JSON diagnostics and visual overlays](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43795691581/artifacts/14193588457). Green=GT bird boxes, blue=candidate boxes with IoU≥.5. The oracle method is evaluation/diagnostic only and MUST NOT be included among deployed model scores in the paper.


## E17 — Full classical integration repair with visual diagnosis (registered before results)

**Research hypothesis:** The severe loss between ~71% ground-truth-assisted proposal oracle mAP and ~11% real detector mAP arises partly from poorly aligned proposal-positive training and weak full-image hard negative examples. Fix GT-border crop clipping, use **true TRAIN GT boxes with IoU>=.55 jitter**, and mine false alarms from full TRAIN images after NMS. All methods must use identical LoG+local-Otsu candidates, HOG+LBP, linear SVM, fixed VAL set and official COCO 101-point mAP@.5.

**Method and preplanned arms:** A historical-style baseline trained on positive candidate regions (IoU>=.5); B GT+jitter positive crops; C GT+jitter with **two rounds of TRAIN full-image post-NMS hard negatives** (up to 600 new sampled/image-diverse safe negatives each round). All use same initial TRAIN safe negatives; features are 64x64 gray HOG+uniform-LBP; no neural methods. Primary NMS IoU=.35 plus separate NMS=.65 ablation. Initial quick gate: 100 TRAIN /30 VAL seeded images, no TEST.

**Primary metric:** official COCO single-class bbox mAP@IoU=.50, 101 recall points, maxDets=100. Secondary: background/localisation/duplicate FP and FN on VAL at fixed decision margin 0, empty-image false alarms and inference/runtime. Visual evidence: overlays of *blue TP predicted boxes, red FP predicted boxes, green matched GT and yellow missed GT*; GT+proposal overlays; generated real JPEGs as workflow artifact and compact contact-sheet embed file committed for remote review. No VAL GT enters train fitting. The visual selection uses errors solely to choose informative images AFTER evaluation and never for training.

**Source:** `src/rebuild_detector.py`, `.github/workflows/classical-repair.yml`. [Github Actions experiment](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43796575319).

**Status when registered:** experiment queued. Do not publish any success or mAP improvement unless measured from completed logs. Compare A/B/C within E17; E15's 1,369-image result is not a matched score because E17 uses 100 TRAIN and 30 VAL.


## E17 — Paired pipeline repair + visual QA (pre-registered)

**Research question:** Is poor 11.29% validation mAP caused primarily by training positive-crop mismatch and insufficient full-image hard false-alarm mining rather than inherent classical detector limits?

**Source:** `src/repair_and_visual_audit.py` and [run 43797974207](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797974207). Protocol: paired legacy vs repaired model on same fixed-seed **100 TRAIN /30 VAL** images, identical image-only LoG+Otsu candidate generator (up to 120 blob peaks), identical HOG+4x4 uniform LBP descriptors (1924 dimensions), linear SVM C=.01, three hard-negative mining rounds. Primary metric **official COCO 101-point single-class bbox mAP@IoU 0.50 (maxDets 100)**; per-image FP classification after margin>=0 NMS=.35 is supplemental.

**Repairs**: clipping pixel coordinates to valid image dimensions after converting YOLO floating GT; train-only GT boxes and light IoU>=.55 centre/scale jitter as positive examples (plus positive detector-like proposals); train-only mining from **real NMS-filtered full-image false alarms** with margin>=0, per-image limit 12, per-round global max 500. Legacy arm uses sampled positive detector proposals and global top-500 background negatives.

**Visual QA:** selected crowded, empty and sparse scenes receive four-panel JPEGs: GT (green), pixel-only proposals (blue), legacy detections (green where IoU>=.5, red otherwise), repaired detections (same colours), with detection margins printed on top-ranked boxes. These visual labels are diagnostic and never used to train models. Files archived under `repair_results/visuals/` plus `repaired_detector_results.json` and `repaired_detector_per_image.csv`.

**Scientific limitations:** test split untouched; paired 30-image validation only, not E15's 60-image dataset; results must NOT be conflated with E15. The visual green/rule uses 'overlaps some GT at >=0.5', not one-to-one matching (duplicate boxes may appear green). Inspect against FP taxonomy as well. At registration cloud run in progress, no metrics available.

### E17 numerical and visual audit results (measured; artifact saved despite workflow push failure)

[GitHub Actions run 43797610003](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797610003) completed its entire data preparation, training, validation and JPEG visual export. **GitHub job conclusion=FAILURE only because the optional git push of visual JSON was non-fast-forward** (research registry commits had advanced main); all metrics and visual outputs were successfully uploaded as [artifact 14194939019](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797610003/artifacts/14194939019) and independently inspected. This is not a failed model experiment. Source code has reproducible tests for GT boundary clipping, >0.55 jitter IoU, safe negative selection and COCO AP synthetic cases.

Matched comparison on **100 TRAIN images; 30 VAL images, 156 birds**; GT-blind LoG+Otsu proposals (~960/image), HOG+LBP + LinearSVC, same seed and feature extraction, COCO 101-point bbox mAP@0.5 maxDets100:

| Method | Real validation mAP50 | Final FP @ margin0 | Missed GT @ margin0 | mAP50 if NMS raised to 0.65 (exploratory) |
|---|---:|---:|---:|---:|
| A: candidate-only positive crops + background SVM | 8.03% | 1,999 | 78 | 8.15% |
| B: TRAIN GT crops + valid jitter positives | 9.54% | 2,072 | 74 | 9.87% |
| C: B plus full-image post-NMS hard-negative mining, round1 | 7.09% | 1,732 | 90 | 8.42% |
| **C, round2** | **11.76%** | **1,196** | **83** | **12.65%** |

**Actual image inspection on six validation scenes reveals major errors:** 
- `beach_bird_1399.jpg`: image contains a very large pale swan covering hundreds of pixels; its two GT bird objects are missed (2 FN, 0 TP), while 30 small red false-detection boxes fire on bird parts and textures. The LoG proposals use sigma<=18 with max factor 2.4 and ratio<=1.5, so the largest generated region is roughly <=106 pixels wide. A full swan cannot be covered by this proposal generator at IoU .5. This is a structural scale-coverage defect and cannot be solved by hard-negative mining.
- `beach_bird_1560.jpg`: 7 GT birds, 71 post-NMS margin0 detections; 3 TP, 68 FP, 4 FN. Many redundant red rectangles on birds and brown beach features, indicating ranking and possible incomplete annotations.
- `beach_bird_1435.jpg`: GT=0 yet 48 final detections; visual image *appears* to contain shorebirds. **Potential annotation completeness issue: require manual validation of source labels**, and do not silently relabel official test images.
- `beach_bird_1379.jpg`: GT=0 and numerous red boxes on washed-up dark seaweed, revealing texture confusions.

These visuals are actual JPEGs archived in the artifact. **Legend:** blue boxes = TP final predictions, red = final FP, green = matched GT, yellow = missed GT. The contact sheet proves the method continues to overdetect background objects and incorrectly localise large birds; a 99%-accurate training patch SVM is not sufficient evidence for a working detector.

**Decision:** The *next highest priority* is multi-scale localisation rescue for medium/large birds, not a further hard-negative mining round. Explore classical edge-ranked large-window anchors combined with LoG/Otsu small-object proposals; report recall by bird scale before any retraining, then confirm true mAP gain. Annotation integrity must be manually reviewed for suspicious GT=0 scenes. The 11.76% is exploratory 30-image VAL, not full validation or TEST mAP.

## E18 — Test fixed classical medium/large candidate generator (registered)

**Hypothesis:** The maximum LoG+Otsu width is too small for many medium/large bird GT boxes. Supplement existing small-scale proposals with deterministic image-only edge-ranked large anchors of sizes 96,144,192,256,384,512 pixels and aspect ratios 0.45,0.7,1,1.5,2.2, selected using per-shape spatial balance (50 candidates/shape maximum). Report LoG-only, large-only and hybrid-union candidate recall@.5/.75, split into small (<1024 px²), medium (1024–9999 px²) and large (>=10000 px²) GT objects, candidate count/image, and example overlays. All method proposals are GT-blind and all methods evaluate the exact same fixed 30 VAL images.

**Source:** `src/multiscale_coverage.py`, `.github/workflows/multiscale-coverage.yml`; run to be recorded after completion. Proposal recall is not detection mAP; subsequent SVM training must be measured separately.


### E18 actual scale-coverage validation and visual verification

[Successful GitHub Actions run 43798486226](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798486226), [ZIP result artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798486226/artifacts/14195192223). Tested identical 30 VAL images (156 GT birds) and GT-blind proposals. The original sigma<=18 LoG+Otsu spans too few large-object sizes. Supplement with edge-ranked, spatially balanced 96..512-pixel scales and five ratios; then union.

| Proposal method | Boxes/image | Recall@IoU.5 | Recall@IoU.75 | Small R50 (46 birds) | Medium R50 (51) | Large R50 (59) |
|---|---:|---:|---:|---:|---:|---:|
| Original LoG/Otsu | 959 | 73.08% | 19.23% | 91.30% | 78.43% | 54.24% |
| Large-window grid alone | 961 | 39.10% | 11.54% | 0% | 17.65% | 88.14% |
| **Hybrid small+large union** | **1,920** | **90.38%** | **30.13%** | **91.30%** | **86.27%** | **93.22%** |

The union raises recall50 by **17.31 percentage points** and large-bird recall by **38.98 points**; stronger evidence of a geometry defect than previous generic feature speculation. Real overlay `beach_bird_1399_hybrid_gt_green_candidates_blue.jpg` now shows medium/large blue proposals overlapping the full green swan bird GT rectangles (earlier small LoG could only select part crops). Additional crowded beach overlays are archived. These are PROPOSAL metrics, not deployable mAP. The extra ~961 candidates/image are a legitimate additional classification and latency cost.

### E19 planned end-to-end hybrid-classical detector test

**Run:** [GitHub Actions 43798829411](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411), code `src/hybrid_detector_eval.py`, workflow `hybrid-detector.yml`. Same 100 TRAIN/30 VAL seeded image subset as E17, same HOG+LBP LinearSVC, GT+jitter positives, safe negatives and two full-image mined-negative rounds; swap **ONLY the GT-blind proposal strategy** to hybrid LoG+Otsu+balanced large-window edge proposals. Evaluate COCO 101-point single-class bbox mAP@.5, NMS .35/.65, full FP/FN taxonomy and 6 real annotated visual examples. Compare E17 and E19 cautiously because negative selection changes when proposal pool changes. No test data used. **Status at registration: running, no verified mAP yet.**


### E17 results: actual measured paired repair and human-checked visual examples

[Successful GitHub Action 43797974207](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797974207), [original visual/metric ZIP](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797974207/artifacts/14195860401). SAME 100 TRAIN /30 VAL images, with 156 annotated VAL birds; legacy positive-proposals + global mining versus train-only GT+IoU>=.55 jitter positives and post-NMS TRAIN-only hard false alarms; HOG+LBP, LoG+Otsu proposals, SVM C=.01, 3 rounds both; official single-class COCO 101-point mAP@IoU .50 maxDets100.

| Model | Round 0 mAP | Round 1 | Round 2 | Round 3 | Round-3 FP boxes @SVM margin>=0 | Round-3 missed GT birds |
|---|---:|---:|---:|---:|---:|---:|
| Legacy proposals+global hard mining | 6.24% | 5.77% | **7.84%** | 5.39% | **979** | 97 |
| GT+jitter positives + post-NMS TRAIN false alarms | 7.01% | 8.44% | **8.95%** | 8.64% | 1,172 | **92** |

The repair improves scores at round 3 **+3.25 percentage points mAP** over matched legacy but **increases** final output FP by 193 under the common margin>=0 threshold. Best round for both arms is **2**, not 3. mAP rank scores and thresholded FP are different operating points, so do not conclude the repair improved precision at the chosen decision threshold. The experiment is not the complete validation set and not comparable with E15 without paired design.

**Human visual review:** reviewed all seven archived comparison montages (GT/LoG+Otsu candidates/old vs repaired final boxes). `beach_bird_1379.jpg` (zero GT) generates dozens of false positives on seaweed; `beach_bird_1463.jpg` (zero GT) on dark rocks and waterfall edge; `beach_bird_1606.jpg` (zero GT) on an airborne drone; `beach_bird_1389.jpg` (16 GT) and `beach_bird_1443.jpg` (12 GT) have many loose, oversized and misplaced boxes around actual birds; `beach_bird_1452.jpg` (26 GT) shows small-bird localisation/recall losses. Most importantly, `beach_bird_1435.jpg` shows numerous visibly real shorebirds but **GT annotation count zero**, so detections in that image (legacy 23; repaired 42) are automatically scored as FP by the official annotations. Do NOT silently modify benchmark labels or claim that all 23/42 are actually real birds; many boxes also overlap background. This is a known dataset annotation quality caveat, *not* proof the full low mAP is entirely due to mislabeled GT.

**Original visual artifact:** `visuals/*_comparison.jpg`; exact per-image FP/TP/FN in `repaired_detector_per_image.csv`, full mining rounds in JSON. Further fixes should target: candidate boxes with better IoU (especially medium/large birds), classifier discrimination on strong textured backgrounds, annotation QA, and a properly tuned operating threshold; train on manually verified examples only where defensible.

### E18 — Dataset empty-label human review (measured)

[Successful GitHub Actions run 43798758419](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798758419), [contact-sheet archive](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798758419/artifacts/14196210026). All TRAIN and VAL image/label pairs were audited by code, zero-box images collected into JPEG contact sheets, 33 VAL zero-box scenes and 154 TRAIN zero-box scenes out of 291 and 1369 respectively. **Human inspection** of the VAL contact sheets and TRAIN pages identifies clearly visible birds in some zero-box images: `train/beach_bird_0149.jpg` (flock of dark airborne bird silhouettes), `train/beach_bird_0418.jpg` (two visible shorebirds), `val/beach_bird_1435.jpg` (many visible wading/shore birds). These are **manual QA flags, not revised ground-truth boxes**, and possible dataset target-scope exclusions should be adjudicated before rewriting labels. Images of drones, kites, bats, waves, seaweed, cattle and planes in many other zero-label files are genuine hard negatives. Because `labels(..., GT_empty)` treats every proposal as negative, any missed annotations may inject *true birds as negative training examples* during full-image hard-negative mining. False positives on this VAL image may likewise include detections of real birds not represented by the official labels.

**Recommended integrity protocol:** quarantine manually flagged suspicious TRAIN zero-label frames from *negative sampling* until checked/re-annotated, but retain all official VAL images in the primary benchmark for comparability; optionally report an explicitly labelled sensitivity analysis excluding confirmed corrupted annotation cases, **never as official mAP**. Don't modify the TEST split. Save source images, original labels, reviewer decisions, and corrected labels (if authorized) with provenance.


### E19 actual end-to-end hybrid detector results — REJECTED as current classifier integration

[Successful run 43798829411](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411), [raw experiment artifact 14196090985](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411/artifacts/14196090985), including real JPEG detections and GT/proposal images. Same 100 TRAIN /30 VAL images (156 birds) as E17; same GT+jitter positive crops, HOG+LBP, linear SVM, and post-NMS hard negative mining, but the candidate pool now includes ~960 LoG+Otsu small-object proposals **plus** ~960 spatially balanced medium/large proposals (1,920/image total).

| Hybrid mining round | Official COCO VAL mAP50 NMS=.35 | mAP50 NMS=.65 | FP on 30 VAL at margin0 | FN on 30 VAL at margin0 |
|---|---:|---:|---:|---:|
| 0 | 4.77% | 5.26% | 3,500 | 65 |
| **1** | **7.52%** | **9.22%** | 2,842 | 76 |
| 2 | 6.18% | 7.54% | 2,199 | 78 |

Compared with E17's old proposals and the same 100/30 images (**11.76% mAP50 NMS=.35, 1,196 FP, 83 FN at mining round2**), expanding candidate boxes massively worsens final ranking. The hybrid generator *does* significantly improve coverage (E18 recall 73.08%→90.38%, large-bird R50 54.24%→93.22%), but the linear HOG/LBP SVM assigns high scores to numerous large background windows. This demonstrates a **proposal–classification domain mismatch**, not failure of the new geometry alone.

**Visual verification:** Open `visual/contact_sheet.jpg` in run artifact. `beach_bird_1650.jpg`: 5 GT, 112 margin0 predictions, many red boxes on textured shoreline; `beach_bird_1606.jpg`: GT=0, 101 false detections on rocks/waves; `beach_bird_1452.jpg`: 26 GT, 70 predicted boxes with numerous oversized red rectangles; `beach_bird_1435.jpg`: officially GT0 despite visually apparent shorebirds, leading to annotation-confounded FP counts. These are real exported images, not predictions inferred from logs.

**Important caveat:** E17 and E19 cannot be interpreted as holding *training negative examples* identical: new candidate pool changes which safe negatives can be sampled and mined; training positives and evaluation images remain matched. Reproducible causal comparison of proposal geometry alone would require freezing all other samples or a trained scale-dependent classifier.

**Decision:** **Do not deploy or select the hybrid detector** based on improved proposal recall alone. The primary mAP and real qualitative output are worse. Retain large proposal geometry only for subsequent scale-specific classification, confidence calibration and richer appearance discrimination. Our best measured current 100/30 classical model remains E17's LoG/Otsu + GT-jitter + post-NMS mining at **11.76% mAP50** (and 12.65% at exploratory NMS=.65). Do not claim any result near 50%.

## E20 — Independent visual integrity check for zero-annotation validation images

[Run 43798758419](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798758419), [archived image montages](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798758419/artifacts/14196210026): train=1369 images / 154 zero-GT annotation scenes; VAL=291 images / **33 zero-GT annotation scenes** (11.34%). The human-visible photo of `beach_bird_1435.jpg` is among official zero-GT validation samples yet plainly appears to contain multiple birds. Such images can create mislabeled false positives and contaminate false-alarm training/evaluation. **Do not assume every zero-GT image is mislabeled**; only flagged examples should receive human review. Never silently alter official dataset annotations or remove samples from primary evaluation. A separate manually audited sensitivity analysis, if performed, must be clearly marked nonstandard and distinct from the official mAP.


### E19/E20 — Multiscale classical proposal repair and hybrid detector (separate completed experiments)

**Proposal stage:** [run 43798486226](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798486226), 30 fixed VAL images with 156 GT birds. Original LoG+Otsu proposal recall@0.5 **73.08%** with 959.17 candidates/image; image-edge-ranked large diverse windows recall@0.5 **39.10%** with 961 candidates; their GT-blind **union recall@0.5 90.38%** with 1,920.17 candidates. Crucially, scale-stratified original recall for 59 large birds was only **54.24%**, versus union **93.22%**. Small-bird recall stayed **91.30%**. So the original max LoG sigma and box extent missed large bird objects. This is a genuine architectural candidate geometry problem. These are proposal recall, NOT detection mAP.

**Hybrid end-to-end model:** [run 43798829411](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411), same fixed 100 TRAIN /30 VAL images as E17, trained classical HOG+LBP/LinearSVC using train GT+jitter positives and train-only full-image hard-negative mining. Official COCO single-class 101-point bbox mAP@.5:
- Round0 NMS.35 **4.77%**, NMS.65 **5.26%**; margin0 FP 3,500.
- Round1 NMS.35 **7.52%**, NMS.65 **9.22%**; margin0 FP 2,842.
- Round2 NMS.35 **6.18%**, NMS.65 **7.54%**; margin0 FP 2,199.
The NMS.65 ablation improves over NMS.35 under matched conditions (e.g. round1 +1.70 percentage points). But the 90.38% proposal recall **does not translate into high end-to-end mAP**: score ranking/background discrimination and huge numbers of wrongly scored boxes remain the primary issue. Hybrid visual [run artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798829411/artifacts/14196090985) includes actual validation photographs with many overlapping red false positives around rocks, seaweed, drones and legitimate birds; this visual QA supports the quantitative FP findings.

**Decision:** don't equate candidate recall with detection performance; no more unrestrained candidate-union expansion. The right next experiment is to improve score discrimination on real-image hard backgrounds (possibly nonlinear classical classifiers and HOG+colour/shape features), while controlling high-confidence false alarms. All model variants remain classical; no neural pretrained models were used.


### E21 — Repaired training-positive and mining ablation with NMS tuning

**Verified run:** [GitHub Actions 43800040330](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43800040330), artifact [classical-repair-visual-audit](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43800040330/artifacts/14196272268). 100 TRAIN images (the current manually flagged empty-label IDs were **not selected**; log `TRAIN QA QUARANTINED []`), SAME 30 VAL images and 156 VAL birds for every arm; unchanged LoG+Otsu proposals, HOG+spatial LBP and LinearSVC C=.01. The proposed/positive baseline and GT jitter variants differ by supervised positive crop construction; all training-only. Official primary metric COCO 101-point bird bbox mAP50 maxDets100; NMS IoU .35 and .65 are *explicit validation ablations*, not test metrics.

| Experiment | Mining rounds | VAL mAP50 @NMS.35 | VAL mAP50 @NMS.65 | FP margin0 @NMS.35 | Missed birds margin0 @NMS.35 |
|---|---:|---:|---:|---:|---:|
| A Proposal-positive SVM baseline | 0 | 8.02% | 8.15% | 1,999 | 78 |
| B GT+jitter-positive SVM | 0 | 9.53% | 9.86% | 2,072 | 74 |
| C GT+jitter + actual TRAIN full-image hard negatives | 1 | 7.09% | 8.42% | 1,732 | 90 |
| **C GT+jitter + actual TRAIN full-image hard negatives** | **2** | **11.76%** | **12.65%** | **1,196** | 83 |

**Findings:** GT positives improve mAP over proposal-only positives (+1.51pp NMS.35). More realistic full-image false alarm mining is nonmonotone: one round hurts, two rounds help. Less aggressive NMS .65 improves the best model to **12.65%** on 30 VAL images. Do not compare this directly to E15's 60 VAL-image result as they use different validation sets. Visual inspection of [contact sheet](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43800040330/artifacts/14196272268) finds continuing many red FP on shore/wave/seaweed/oversized bird windows, so model remains unreliable. Improvements are genuine but fall far below 50%. The annotation contamination risk found in E18 is **not the explanation for this particular 100-TRAIN result**, because no quarantined file was selected by the fixed-seed train sampling.

**Reproducible fixes:** `src/cloud_audit.py` now clips subpixel YOLO-boundary roundoff before accepting GT XYXY, with synthetic edge tests; `src/label_quality.py` stores manual QA training quarantine IDs; `src/rebuild_detector.py`, `src/hybrid_detector_eval.py`, and `src/repair_and_visual_audit.py` apply training-only quarantine and log affected filenames. All VAL/TST official annotations stay untouched.


## E21 — Scale-specific classical classifier architecture (preregistered 16 Nov 2026)

**Hypothesis:** The poor ranking of 1,920 hybrid proposals/image arises from a single global HOG+LBP classifier mixing incompatible bird sizes and from incomplete contextual/colour information. Test a **scale-gated three-SVM** detector against a matched single-SVM control and an additional HSV/edge feature ablation.

**Implementation:** `src/scale_specific_detector.py`, `.github/workflows/scale-specific-e21.yml`; [GitHub Actions run 43800881909](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43800881909). 100 seeded TRAIN (after pre-documented suspect TRAIN annotation quarantine) and 30 seeded VAL images; no TEST data. For calibration only, reserve ~20% of TRAIN images; fit and mine on disjoint 80% of TRAIN. All variants use **identical GT-blind image-only hybrid LoG+Otsu plus 96..512px spatially balanced large proposals**, and identical HOG/LBP features. They use TRAIN GT+jitter positives plus train-only proposal-positive examples and GT-safe negatives (IoU<.1 and no strong containment). Two rounds of post-NMS TRAIN hard-negative mining, separately within each scale specialist. Size bins defined by box area: small <1024px²; medium 1024..9999px²; large >=10000px². Per-scale logistic score calibration on held-out TRAIN images, never VAL. NMS IoU=.5 for all arms.

**Ablation:** A single model HOG+LBP + train-only calibration; B 3 specialist size-gated HOG+LBP models; C size-gated HOG+LBP+HSV histograms+intensity mean/std+gradient/edge occupancy features. Measure COCO API bbox mAP@0.5, 101 recall samples maxDets=100 for each; report final FP/FN with probability>=.5, and visual image overlays for each arm showing correct boxes blue, false alarms red, matched GT green, missed GT yellow. Calibration is solely from reserved TRAIN and cannot see validation labels.

**Status at registration:** cloud workflow queued; do not invent performance before run concludes. Consider subsequent classical box regression ONLY if this ablation demonstrates genuine classification gain. Existing E17 LoG/Otsu 11.76% (100 TRAIN/30 VAL) and E19 hybrid 6.18% are historical, not matched with E21 image-level calibration splitting.


### E21 initial measured run — scale-specific split alone did NOT help (calibration re-audit)

[Successful first Actions run 43800881909](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43800881909), [visual/CSV archive](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43800881909/artifacts/14196224342). Fixed dataset: 100 TRAIN images split 80 classifier fit / 20 calibration, 30 VAL images containing 156 GT birds; same hybrid GT-blind proposals, GT+jitter/proposal positives, 2 mining rounds and NMS IoU=.5. Primary metric: official COCO 101-point bbox mAP50.

| Variant | Primary VAL mAP50 | FP at calibrated p>=.5 | Missed birds at p>=.5 |
|---|---:|---:|---:|
| A single HOG+LBP linear SVM | **8.15%** | 6,141 | 35 |
| B three size-specific HOG+LBP SVMs | 6.54% | 7,316 | 33 |
| C B plus HSV/intensity/edge features | 6.87% | 7,353 | 33 |

**Do not claim improvement.** Novel architecture degrades mAP, though the simplistic calibrated 0.5 operating point is misleading because initial calibrators were fitted using *balanced GT-jitter positive samples* and only ~20 sampled negatives per calibration image instead of realistic candidate distributions; p>=.5 labels far too many backgrounds. In particular calibration of the small specialist used only 21 positive GT-jitter samples vs 407 negatives, indicating instability. An actual six-scene contact sheet shows hundreds of red large background rectangles over beach textures, shoreline and water.

**Corrective follow-up:** [Actions 43801802502](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43801802502) now fits calibration using **held-out TRAIN real generated proposal positives and safe negatives** with natural class prevalence (no `class_weight='balanced'`), not ideal GT jitter, and reports ranked top-20 FP/FN for additional post-NMS diagnostics. Crucially both training and calibration use TRAIN only. Until rerun completes, retain E21 first result as a *negative outcome*; do not use its p>=.5 FP count to compare with older raw SVM margin=0 counts.


## E23 — Quality-subset negative-learning gate (preregistered)

**Source:** `src/quality_subset_train.py`, workflow `.github/workflows/e23-quality-subset.yml`, [Actions run 43804273592](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43804273592).

**Motivation:** Inspection of train zero-GT contact sheet found visible birds in `beach_bird_0690.jpg` and `beach_bird_1343.jpg`, which could be wrongly treated as negative-only examples. E21 performance was very poor with large candidate/background confusion. Avoid blindly trusting empty labels.

**Quality subset:** exactly 45 seeded TRAIN images, selected only from images with >=1 annotated bird. No zero-annotation TRAIN images are used in this small diagnostic; additionally known suspect zero-GT TRAIN images are quarantined in `src/label_quality.py`. This conservative selection is NOT proof that every remaining image is completely annotated. Fixed 12 VAL images, with no test usage. Same GT+jitter positives (IoU>=0.55), identical GT-blind LoG+Otsu candidate boxes, HOG+LBP descriptors and balanced linear SVM C=.01.

**Paired comparison:** A baseline draws up to 20 GT-safe negatives per TRAIN image randomly; B selects **the same initial count** with deliberate coverage across image tiles and small/medium/large size bins. Both then receive two rounds of TRAIN-only post-NMS hard negative mining; B constrains mining to diverse image/tile/scale regions and removes near-duplicate selected boxes. Feature proposals and HOG+LBP descriptors are computed once per image, cached, and shared between variants/rounds.

**Primary acceptance gate:** B must beat A in official COCO 101-point single-class bbox mAP@.5 (maxDets100) AND reduce the number of FP among the top 20 ranked detections/image on the identical VAL set, with independent manual visual review. Other reported metrics: FP and FN @margin0, training patch accuracy, all three rounds, per-image overlays (blue=TP, red=FP, yellow=missed GT), exact selected TRAIN filenames. Do not expand to full TRAIN unless this small result is actually promising, visually credible and repeatable. VAL GT is never used to train, select negative samples, or select candidates.

**Status at registration:** waiting for completed GitHub Actions metrics; no outcome asserted.


### E23 results and human visual inspection — diagnostic completed; NOT a scale-up candidate

**Cloud run:** [successful 43804273592](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43804273592), [all CSV, JSON, PNG/JPEG audit artifacts](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43804273592/artifacts/14196769350). Exactly **45 TRAIN** images with **at least one annotated bird each** (308 GT birds; all zero-GT TRAIN excluded), **12 VAL** images (77 GT), 3 GT+jitter positive examples per GT plus original (1,232 positive crops), shared image-only LoG/Otsu proposals and HOG+LBP, matching balanced linear SVM C=.01 and 2 rounds of hard-negative mining. Never touched TEST. Full training intentionally not run.

| TRAIN-negative strategy | round 0 COCO mAP50 | round 1 | round 2 (pre-registered final) | Top-20 TP / 240 after round 2 | FP@margin0 round2 | FN@margin0 round2 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: 20 random GT-safe/image + post-NMS mining | 4.07% | **6.30%** | 5.24% | 24 / 240 | 646 | 42 |
| Quality: 20 tile+size diverse GT-safe/image + capped diverse post-NMS mining | 2.69% | 4.49% | **5.39%** | 24 / 240 | 640 | 42 |

**Result:** the quality arm improves fixed round-2 mAP by only **0.146 percentage points**, with **no reduction in top-20 FP** (216 vs 216), so it FAILS the pre-registered dual acceptance gate. The baseline's round1 6.30% exceeds both round2 scores; mining effects are non-monotonic. Quality arm round2 uses 1,127 negatives vs 1,346 baseline, since diversity guards reject redundant candidates. The experiment cannot establish that diverse negative sampling is better, nor that quarantine improves mAP in isolation; both arms use the same nonempty-GT quality subset.

**Direct original JPEG examination:** I personally inspected `baseline/visual/contact_sheet.jpg` and `quality/visual/contact_sheet.jpg`; the outputs are visually very similar. `beach_bird_1379.jpg` (GT0): multiple red rectangles over dark seaweed at shoreline; quality 65 active boxes vs baseline 54. `beach_bird_1419.jpg`: many FP red boxes on dense reeds/shore grass, several yellow missed actual birds. `beach_bird_1560.jpg`: many red rectangles on parts of real shorebirds and sand; duplicate-like nested proposals and yellow misses. `beach_bird_1606.jpg` (GT0): scores on aircraft-like metal/structures; 81 active quality boxes vs 78 baseline. Visuals strongly indicate background texture/shape confusion and multiple inadequate box geometries; changing the *distribution of GT-safe negatives* did not fix it. **GT0 does not guarantee truly bird-free**; original annotations remain the official evaluation reference. The 6-scene visual panels show excessive detections even though there are only 77 annotated birds over 12 VAL images.

**Decision:** NO full training. The next isolated diagnostic should test proposal *classification* capacity on confidently labelled background vs bird crops and/or a dedicated background rejection cascade; it should not run more unbounded mining, expand candidate boxes, or treat a 0.15-point gain as proof of success. Prefer a tiny frozen feature bank and PR/ranking plots. Report this as a negative outcome in the paper.


## E24 — Targeted post-detection false-positive rejection gate (registered)

**Question:** Can a second classical classifier distinguish the actual high-ranked LoG/Otsu false alarms (seaweed/water/beach texture/partial-bird crops) from genuine TRAIN proposal positives better than adding another round of SVM mining?

**Run:** [GitHub Actions 43805068857](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805068857), code `src/background_rejector.py` and workflow `.github/workflows/e24-background-rejection.yml`. Same exact 45 annotated-positive TRAIN images and 12 VAL images as E23, with no zero-annotation TRAIN images used and no TEST access.

**Controlled architecture:** E23 baseline GT+jitter positives, 20 random GT-safe negatives per TRAIN image, HOG+LBP balanced linear SVM, **one** TRAIN-only post-NMS hard-negative mining iteration (E23's best-performing mining round). Shared LoG+Otsu image-only proposal bank cached for stage1, stage2 and VAL. Train a second-stage **RandomForest** classifier using ONLY actual TRAIN proposal positives (IoU>=.5) and actual GT-safe TRAIN background negatives (IoU<.1, low containment), emphasizing high-margin post-NMS false alarms. Features: stage1 SVM margin; 18 HSV bins; 14 grayscale, edge-density, size, aspect and image-position statistics. Ambiguous overlaps are ignored, not mislabeled background.

**Paired ablation:** A stage1 only; B/C/D add the identical trained stage2 with score blending weight 0.25/0.50/0.75. All methods share the very same images, base SVM, candidate boxes, NMS .35, official COCO 101pt mAP50 maxDets 100; primary diagnostic false positive in top-20 ranked boxes/image and FN at fixed margin0 (note blended score changes interpretation of margin0). Save identical visually annotated top20 boxes for several backgrounds, crowded scenes and bird scenes, plus per-image errors. Validation GT is NEVER used for training.

**Promotion gate:** only consider larger training if a variant gives **at least +2 percentage points absolute mAP50** over common baseline **and** fewer top-20 false positives, followed by independent inspection of the actual annotated images and repeat with another train seed. Otherwise reject it. No promise of >=50% mAP. **Status:** running, no results verified yet.


### E24 measured results and visual adjudication — targeted FP reranking PASSES exploratory gate

[Successful Actions run 43805068857](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805068857), [raw artifact 14198260647](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805068857/artifacts/14198260647), contains `e24_results.json`, `e24_per_image.csv` and identical per-image *top-20 detection* overlays for base/three cascade blends. Fixed 45 TRAIN annotated-positive images (zero TRAIN-empty images excluded), 12 VAL images (77 labelled birds), no TEST. Stage 1 = same E23 single HOG/LBP+SVM with one TRAIN-only HNM round (preselected from E23); stage 2 = TRAIN proposal-domain RandomForest (633 IoU>=.5 positive proposals; 706 post-NMS hard negatives and 512 other GT-safe negatives, 43 distinct TRAIN images), trained only from TRAIN labels, 32 handcrafted color/edge/geometry features + SVM margin. Outputs are officially scored single-class COCO 101-point bbox mAP50 maxDets100.

| Model | mAP50 | Correct among top 240 (20 per VAL image) | Incorrect among top 240 | FP at score>=0 | Missed GT at score>=0 |
|---|---:|---:|---:|---:|---:|
| Stage1 baseline | 6.30% | 21 | 219 | 909 | 43 |
| Cascade score blend alpha=.25 | 9.37% | 28 | 212 | 609 | 42 |
| **Cascade alpha=.50** | **9.60%** | **29** | **211** | **471** | 44 |
| Cascade alpha=.75 | 6.26% | 23 | 217 | 423 | 48 |

**Acceptance:** alpha=.50 gains **+3.298 percentage points** mAP50 and reduces top20 FP by 8 on the exact same VAL images; clears numerical exploratory +2 points and lower top20 FP gate. Score>=0 threshold totals **cannot be interpreted as paired fixed-precision recall** because score blends change score scale; global COCO mAP and top20 matching are the primary comparable metrics. Alpha=.75 highlights trade-off: fewer FP at zero-score does not mean higher ranked mAP.

**Actual visual inspection** of original `visual/A_base/top20_contact.jpg` versus `visual/C_cascade50/top20_contact.jpg`, plus per-image records:
- `beach_bird_1379` (0 GT): top20 remains **20/20 red** on seaweed/debris. At score>=0 FP drops **88→34** but false alarms remain.
- `beach_bird_1419` (5 GT): top20 remains **20/20 red** mainly on vegetation/reeds, with actual birds missed. FPs at zero-score decrease **82→4**, but 5 GT are still missed; threshold decrease alone is not a solved detector.
- `beach_bird_1560` (7 GT): top20 stays 3 TP /17 FP; red nested/partial bird boxes persist. FP at score>=0 falls **102→59** but FN rises 2→3. The cascade may suppress some true bird patches.
- `beach_bird_1606` (0 GT): 20/20 top-ranked false detections remain over the drone/aircraft-like shape; zero-score FP **106→56**.
- `beach_bird_1389` top20 TP stays 5 on alpha=.50, but `beach_bird_1443` improves 4→6, `beach_bird_1516` improves 2→5 and `beach_bird_1559` improves 1→2. Gain originates from better ranking on some positive-bird scenes rather than eliminating hard all-negative top20 scenes.

**Scientific interpretation:** This is the strongest *controlled* FP-related improvement to date but not a solution to dominant seaweed/vegetation/aircraft false detections, and **9.60% remains far from 50%**. Improvement on a 12-image validation set, and alpha tuned among three variants on this same VAL, can be selection noise. Require a second fixed split/seed and visual review before expanding; continue to investigate whether exact GT annotations omit visible birds. Do not report this small-set best score as full-dataset or unbiased test performance.


### E25 — Pre-frozen cascade confirmation on 30 validation images: reproduced improvement

[Successful cloud run 43805517540](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805517540), [raw images/JSON/CSV artifact 14197594526](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43805517540/artifacts/14197594526). **No tuning in E25:** stage2 and stage1 identical algorithms/seed as E24, same 45 quality TRAIN images with TRAIN-only negative provenance, and *fixed alpha=.50 chosen in E24*. Expanded VAL from 12 to 30 images, total 156 annotated birds, same `data(root,"val",30)` reproducible seed43. Not fully independent image set from 12-image experiment (may overlap), but no re-tuning and most images differ. No TEST accessed.

| Metric | Single HOG+LBP SVM | Fixed alpha=.50 Stage2 RF cascade | Difference |
|---|---:|---:|---:|
| COCO bbox mAP@.5 101pt maxDets100 | **3.375%** | **6.499%** | **+3.124 percentage points** |
| Correct among top 600 detections (20/image) | 38 | **56** | **+18 TP** |
| Incorrect among top 600 | 562 | **544** | **−18 FP** |
| False detections at zero score (NOT fixed operating point across models) | 2,069 | 1,056 | −1,013 |
| Missed GT at zero score (NOT fixed operating point across models) | 95 | 94 | −1 |

**Image-level checks:** Across 30 images, correct top20 TP ranking increased in **13 images**, stayed unchanged in **16**, and worsened in **one**. Score>=0 FP totals decreased in all 30 images, but score scales differ across systems. Actual visual JPEG `visual/C_cascade50/top20_contact.jpg` independently inspected: 1379 (GT0) still top20 all false alarms on seaweed; 1419 birds under reeds still mostly missed; 1560 several genuine bird detections but many red nested partials; 1606 (GT0) still top20 all red on drone-like structure. Thus improvement in global mAP and some bird-scene ranking is reproducible, **but the worst semantic-background false alarms persist**.

**Engineering decision:** The TRAIN-only proposal-domain secondary classifier is worth retaining as a promising *research branch*, not replacing the existing detector as a solved model. More training images may help, but no evidence yet that this architecture can approach 50% mAP. Next focused test should directly curate *hardest false-positive categories*, require per-background image FP reduction without losing recall, and add a small independent set of audited negative-only TRAIN images; do not expand blindly and do not use VAL labels in training. Recognise official VAL missing-annotation uncertainty.


## E26 — Research-driven geometric refinement and suppression, registered before results

**Question:** Can a TRAIN-only classical box regressor resolve the pervasive oversize/partial-bird localisation errors, and can Gaussian Soft-NMS maintain multiple close bird detections better than hard NMS? This is distinct from E24–E25's background reranker.

**References:** Ross Girshick et al., *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation*, CVPR 2014 (use of regularized bounding box geometric deltas and high-IoU TRAIN proposals; here no CNN features are used); Navaneeth Bodla et al., *Soft-NMS*, ICCV 2017 (continuous overlap-dependent score decay rather than deleting boxes). Classical object discovery alternatives for later trials: Felzenszwalb et al., DPM/latent SVM (PAMI 2010); Uijlings et al., Selective Search (IJCV 2013); Zitnick/Dollar Edge Boxes (ECCV 2014).

**Protocol:** [E26 cloud run](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e26-bbox-refinement.yml); source `src/bbox_refinement_study.py`, 45 annotated-positive TRAIN images and common 30 seeded VAL images. Same E25 fixed SVM stage1, TRAIN-only RF stage2, 0.50 fusion alpha, TRAIN-only LoG/Otsu proposal bank. Train `StandardScaler+Ridge(alpha=75)` on *TRAIN proposals whose GT IoU≥0.5*, at most 30 per TRAIN image; 33 dimension input margin+32 image-only colour/edge/geometry descriptors; four standard normalized centre displacement and log-scale targets, clipped to prevent pathological boxes. Compare 6 predeclared configurations: base SVM hard NMS.35; RF cascade hard NMS.35; cascade+regression hard NMS.35; cascade+regression hard NMS.65; cascade Gaussian Soft-NMS sigma.5; cascade+regression Gaussian Soft-NMS sigma.5. All variants use same TRAIN/VAL images and official COCO 101pt bird bbox mAP@IoU=.5/maxDets100. Secondary: top20 paired TP/FP, margin>=0 FP/FN, *maximum raw proposal IoU before and after box correction* and recall@IoU .5/.75 (this is diagnostic, not detector mAP). Save actual preselected VAL photo contact sheets for all variants, matched scene filenames.

**Controls:** GT only for TRAIN regression fitting and diagnostic VAL evaluation; no GT-derived adjustments at inference; TEST untouched. E26 ablation intentionally exploratory because E24 fusion alpha already selected using prior VAL. Do not promote to full training without improvement in official mAP, top20 false ranking and true GT localisation, with human review of screenshots. The regressor is restricted to near-matched TRAIN proposals because regressing wholly background crops is ill-posed.

**Status:** code/workflow committed before results. Log observed result even if negative; no claimed gain prior to job conclusion.
