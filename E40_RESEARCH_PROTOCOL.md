# E40 — Three boosted bird experts, calibrated confidence, box refinement and crowd-aware NMS

**Research branch:** `research/e40-three-expert-refinement`. **Status:** cloud TRAIN/VAL evaluation launched; no verified E40 result yet.

This implements the complete architecture sketched in the user's diagram, subject to strictly classical ML and the prior failures observed in E34/E39. It does **not** use CNNs, pretrained encoders, or the already-evaluated TEST split.

1. **Deterministic multiscale proposals:** E37 canonical Selective Search with fixed 1200-box cap, and the same candidates for all five paired arms.
2. **Object and context features:** E39 native-aspect letterboxed 64×64 crop, masked 4×4 orientation/colour features, local LBP, native geometry and surrounding HSV context. Crop previews and finite-feature assertions saved.
3. **Three boosted experts:** LightGBM for candidate areas small <1024px², medium 1024–9999px² and large ≥10000px², with a global LightGBM comparator. TRAIN-only positive and hard-safe negative proposals; gating uses *candidate* area, never GT area at inference.
4. **Confidence calibration:** each expert and global booster is fitted on ~77% of TRAIN *images*; held-out ~23% of TRAIN image groups fits a logistic Platt mapping of raw log-odds to bird probability. No shared image IDs between booster fitting and calibration. The calibrated global and relevant expert probabilities are averaged.
5. **Learned box refinement:** TRAIN-only E32 ExtraTreesRegressor predicts bounded centre/size offsets from proposal appearance, stage1 and RBF features, then clamps to valid 512×512 image coordinates.
6. **Crowd-aware NMS:** retain neighbouring tiny detections when overlap IoU <.50, rather than the standard .35 threshold; large candidates retain .35 threshold. This is a simple scale-aware heuristic, not a learned crowd predictor.

## Paired controlled arms

- `rbf_035`: original PCA64/RBF Selective Search incumbent, NMS .35.
- `experts_raw_035`: uncalibrated 3-expert/global logit fusion, NMS .35.
- `experts_cal_035`: heldout-TRAIN calibrated 3-expert/global probability fusion, NMS .35.
- `experts_cal_refine_035`: calibrated scores + TRAIN-only learned box refinement, NMS .35.
- `experts_cal_refine_crowd`: all stages including crowd-aware small-object NMS.

Frozen **100 TRAIN /60 VAL** experiment; fixed GT-blind proposals; same image IDs and COCO 101-point bbox AP50; top20 TP, small and crowded top100 recall, and original annotated image overlays for failure analysis. Promotion requires ≥+2pp absolute AP50 and no top20 TP loss versus paired incumbent; if no arm passes, retain incumbent and report failures. These are exploratory validation metrics, **not** a fresh independent TEST score. The official TEST was already evaluated in E33.

## Important limitations

- E39 showed boosted classifiers alone and mild augmentation decreased AP; E40 does **not** assume more model capacity guarantees better detection.
- The E32 box refiner previously failed to improve AP; E40 explicitly measures its *incremental* effect separately.
- The crowd NMS heuristic may preserve more true small birds but also many more false boxes.
- Calibration holds out TRAIN images, but reduces booster fit data and is not a guarantee of well-calibrated probabilities on unseen VAL.
- The three experts are gated by *proposal* area, which can misclassify large birds represented by small part proposals; this remains a major geometry limitation.
- GT-missing backgrounds can contaminate negative training examples. No official annotations are modified.
- No augmentations are included in E40's primary experiment because E39's augmentation harmed overall AP; this avoids conflating multiple interventions.

[Workflow and live logs](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e40-three-experts.yml)
