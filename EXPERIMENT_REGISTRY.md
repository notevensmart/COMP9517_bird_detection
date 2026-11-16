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
