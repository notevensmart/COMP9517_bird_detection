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
| E11 | Can LoG blobs + local Otsu segmentation reduce incorrect boxes? | Identical reserved seeded VAL subset, GT used only for evaluation | Pending cloud run | Record proposal recall@.5/.75, unmatched (IoU<.1), poorly localised (.1≤IoU<.5), duplicates and false boxes per image | Await measurement |
| E12 | Can rotation-pooled HOG/SVM reduce false detections? | Same TRAIN and VAL subsets as HOG baseline; identical pixel-derived candidate boxes | Pending cloud run | Record AP50 continuous, FP background/localisation/duplicates, missed birds, NMS and runtime | Await measurement |

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
