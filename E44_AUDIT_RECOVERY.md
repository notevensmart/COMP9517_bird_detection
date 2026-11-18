# E44 NMS ablation — audited recovery and evidence log (19 Nov 2026)

## Mandatory evidence reviewed before modifying code
- `RESEARCH_LOG.md`, `EXPERIMENT_REGISTRY.md`, `E42_RESEARCH_PROTOCOL.md`, `MODEL_CHALLENGE_AUDIT.md`, `E37_STABLE_PROPOSALS_REPORT.md`, `CLASSIFIER_COMPARISON_REPORT.md`, `DATASET_QA_FLAGS.md`.
- Existing `src/e42_full_rbf_augmented.py`, `src/e44_nms_ablation.py`, `src/metrics_map.py`, `src/cloud_audit.py`, `src/nonlinear_appearance_study.py`, `src/selective_search_detection.py`, and relevant GitHub workflows.
- Downloaded and inspected the real successful E42 artifact 14291446569 and E43 artifact 14294729187, including metrics CSVs, JSON reports, and E42 original-photo overlays.
- E42 full run 43974171244: 1215 annotated-positive TRAIN, 291 VAL / 1843 GT; augmented RBF mAP50 0.20015650733804727, top20 TP=809, FP=5011; nonaug RBF mAP50 0.17432601719438245, TP=820, FP=5000. Raw proposal coverage=1395/1843=75.6918%.
- E43 full run 43974178323: global mAP50 0.17760682609500245, specialist-only 0.1704129023091761, 25/50/75% blends 0.16862633822850115 / 0.16764455283508495 / 0.168265954695999. All specialists/blends failed AP promotion gate.
- E44 original run 44421640677: FAILED at Python import, `ModuleNotFoundError: No module named 'skimage'`. Checkpoint download and Kaggle retrieval succeeded; no result artifacts were produced. `requirements.txt` already listed `scikit-image`; E44 workflow had bypassed that requirements file.

## Diagnosed issues, separated from measured effects
1. **Verified workflow failure:** uninstalled `scikit-image`. Addressed by installing repository requirements and testing imports, feature contract, metrics and NMS before dataset-intensive work.
2. **Soft-NMS score domain error:** original implementation applied a sigmoid to signed E42 fused confidence before multiplicative score decay; this changes cross-image COCO AP ranking versus original raw E42 signed scores. Revised to log-weight score decay: `s' = s + log(decay)`, equivalent to multiplying exp(s) by the original Soft-NMS decay, without applying a nonlinear image-crossing sigmoid. Raw E42 scores remain unchanged for nonoverlap. **Effect on mAP not yet measured**; this is a method definition requiring reported transparency, not a guarantee of improvement.
3. **Reproducibility gate:** full 291-image E44 run must reproduce E42 mAP50 0.20015650733804727 for hard NMS .35 within 1e-6, otherwise reject all cross-experiment NMS conclusions. Geometry and COCO evaluation remain unchanged.
4. **Checkpoint and cache:** reuse the existing 43974171244 trained_model checkpoint; cache is proposal-only and should not be assumed to persist between runner jobs unless successfully restored. The first E44 run explicitly logged `Cache not found`; avoid claiming a warm cache.
5. **Real image observations:** E42 augmented 1379 (GT0) ranks seaweed/wrack as birds; 1452 (26 GT) has 7/20 true detections and clustered misleading boxes; 1399 (2 GT swans) has 1/20 true detection and many partial/oversized boxes. Official VAL image 1435 contains possible unlabeled real birds; do not silently relabel.
6. **E42 augmentation vs top20:** measured change +2.58305 pp AP50 and -11 top20 TPs. Worse top20 TP images include 1393 (-4), 1453 (-3), and 1452, 1560 (-2 each). These data establish different operating-point behaviour; they do not prove a causal mechanism.
7. **Known negative prior results:** E19 added coverage but hurt AP; E31 unrestricted proposal union increased raw recall to 96.15% yet AP fell to 11.35%; E23/E27 extra negatives were not convincingly beneficial; E36 found uncapped SS set stable but order nondeterministic, repaired via E37 fixed GT-blind proposal cap. Avoid repeating these experiments unmodified.

## Current E44 branch protocol
Branch: `research/e44-nms-audited-repair` (forked from `research/e42-full-rbf-augmentation` without modifying E42 frozen baseline).
- Classical-only HOG/LBP/PCA/SVM; TRAIN-only frozen checkpoint. Full VAL 291 images / 1843 GT, COCO 101pt mAP50 maxDets100.
- Same candidate bank, E42 fused raw score, 300 max detections pre-COCO; hard NMS thresholds .35/.50 and linear/Gaussian Soft-NMS (sigma=.5). The new Soft-NMS operates in exp(raw-score) weights represented numerically in log-score units.
- Pre-run tests: synthetic COCO, E42 feature contract, NMS unit tests, 2-image checkpoint smoke test. Full run only after all gates pass.
- Results: **PENDING** until workflow actually executes and artifacts are inspected. Original E44 must remain recorded as FAILED, not overwritten.
- Preserve visuals, per-image top20 TP/FP, raw/post NMS proposal coverage, size-stratified raw coverage, worst images and full metric JSON.
- WARNING: duplicated/near-identical boxes with linear Soft-NMS may be retained at extremely low scores, but COCO maxDets=100 and score order limit their effect. This differs from hard NMS; report exact definitions.

## Follow-up
After workflow reports: inspect error logs if failed; inspect JSON/CSV and overlays if successful; enforce hard035 baseline reproduction. Compare mAP/TP by crowding and size using official VAL only. Report measured differences, including negative results. Do not run TEST or retrain E42 merely to change NMS.
