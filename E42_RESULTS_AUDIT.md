# E42 verified results — three paper-derived classical detectors (19 November 2026)

**Completed successful run:** https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/44493145567

**Protocol:** all three independent matrix jobs used exactly the same fixed 45 annotated-positive TRAIN images, 30 official VAL images containing 156 birds, 250 capped deterministic image-only Selective Search proposals per image, NMS 0.35, and official single-class COCO 101-point bbox AP50 maxDets100. No TEST access. Full code, image IDs, per-image CSV and real-image overlays are stored in the three individual GitHub Actions artifacts.

| Paper-derived detector | Official VAL mAP@0.5 | Top20 true positives / 600 | Top20 false positives | Training fit duration excluding feature bank |
|---|---:|---:|---:|---:|
| DPM-inspired latent HOG parts + linear SVM | 5.2482% | 48 | 552 | 3.35s |
| Improved Fisher Vector (dense SIFT + GMM) + linear SVM | 4.0242% | 35 | 565 | 0.83s |
| Aggregated Channel Features + boosted depth-2 trees | **8.5731%** | **58** | **542** | 86.56s |

**Shared proposal quality:** raw R50 64.10%, R75 19.23%; small R50 39.13%, medium R50 56.86%, large R50 89.83%; ~250 candidates/image. These are candidate recalls, not detector recalls. The proposals bound the maximum achievable true-positive recall and are considerably weaker than earlier E31/E38 studies with higher candidate budgets. Do not compare E42 AP with E33 full-TEST or E31/E38 AP as a controlled classifier comparison because candidate budgets, descriptor settings and training protocols differ.

## Visual audit of saved actual photographs

The uploaded artifacts include images 1379, 1399, 1419, 1435, 1452, 1560 and 1606 (GT green, true top20 detections blue, false top20 red). Verified per-image top20 true positive counts:

| Image | GT | DPM | Fisher | ACF | Main issue |
|---|---:|---:|---:|---:|---|
| 1399, large swans | 2 | 2 | 0 | 1 | whole-body localization remains unreliable; many head/neck/partial boxes |
| 1419, wetland vegetation | 5 | 0 | 0 | 0 | reeds/background rank highly across models |
| 1452, dense flock | 26 | 2 | 2 | 3 | severe small crowded bird misses; box/scale and ranking failure |
| 1560, shorebirds | 7 | 4 | 1 | 4 | correct detections coexist with many duplicates and partial boxes |

All images are genuine GitHub Actions predictions, not generated illustrations.

## Paper fidelity, limitations and decision

- **DPM:** implements HOG root and four deformable HOG part filters with learned quadratic/linear displacement features, alternating latent-part localization and SVM fitting, and TRAIN-only mined negatives. The original paper additionally uses multiscale HOG feature pyramids, mixture components, more extensive latent training and precise deformable-part inference. **This is a one-component bounded adaptation, not Felzenszwalb DPM reproduction.**
- **Fisher Vector:** follows the original improved Fisher mean and variance gradient terms, signed-square-root normalization and L2, with 8 TRAIN-fitted diagonal GMM components and dense SIFT. The paper originally studies classification with much larger codebooks; here local SIFT is pooled for candidate object windows and a linear SVM yields bbox detector scores.
- **ACF:** uses the original ten channel families (three LUV, normalized magnitude, six orientations), 4x4 aggregation and smoothing, depth-two boosted decision trees. It omits the paper's fast multiscale channel feature-pyramid detector and uses 160 instead of 2048 trees. Thus this is an ACF-channel-plus-boosting ablation, **not a complete historical ACF implementation**.

**Record correction:** The DPM JSON stores `train_initial_negative=764` and `training.mined_negatives=764`, because the original 539 negative examples list was mutated in-place with 225 mined examples before those metadata fields were recorded. The correct values are **539 initial**, **225 mined**, **764 total**. The model was trained using these 764 total negatives as intended; only the two reported count labels are wrong. This correction is based on code-level inspection and the shared identical training samples. It does not affect mAP, run status, or predictions.

**Decision:** ACF + boosted trees leads this specific paired 45/30/250 experiment, but with 8.57% AP and 542/600 top20 false alarms it is inadequate as a replacement for the frozen prior detector. DPM shows a localized benefit on large swans but not dense birds. Fisher is weakest here. No claims about which original paper's full algorithm is best may be made from reduced-scale versions.

**Primary citations:** see [E42_PAPER_REFERENCES.md](E42_PAPER_REFERENCES.md). 
