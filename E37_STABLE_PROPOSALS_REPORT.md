# E37 — Deterministic Selective Search candidate cap (TRAIN+VAL research only)

**Date:** 16 Nov 2026. **Branch:** `research/deterministic-selective-proposals`. **Source:** `src/selective_search_detection.py`; **validation:** `src/e36_reproducibility.py` and [successful E37 run 43831293647](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43831293647).

## Reason for intervention

E36 showed that OpenCV Selective Search's raw region SET is consistent on repeated invocations, including seeded and single-threaded calls, but its ORDER varies. The original detector deduplicated in first-seen order then kept the first 1200 boxes, so it retained different proposal sets for identical images with more than 1200 raw regions. This led to a hidden pipeline randomness even when the SVM/RBF training itself was deterministic on an identical feature bank.

## E37 change

After rejecting invalid sizes, canonicalize every proposal as a unique float32 `[x1,y1,x2,y2]` tuple with `np.unique(...,axis=0)`. If more than 1200 exist, select a fixed-seed (`9517`) GT-blind subset of 1200 indices from the canonical array, sorted into canonical order. Do not use labels, confidence scores, or the TEST split. This is a **new proposal distribution** and must be evaluated on TRAIN/VAL before being promoted to any final test model.

## Measured repeatability

- **VAL 1379:** original 1,508 raw regions. Old first-1200 cap changed between runs. E37 cap returned the identical 1200-box array on repeated calls and after another synthetic image was processed.
- **VAL 1389:** all 511 raw regions are retained; the canonical array and its hash were identical across calls.
- Repeat fits of the same 10-image TRAIN feature bank generated exactly identical PCA/RBF scores and NMS indices for both VAL scenes.

E37 asserts repeatability rather than merely reporting hashes. The successful action includes `e36_determinism.json` with full checksums and package versions (OpenCV 4.14.0, NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1).

## Scope / decision

**PASS on the two audited images.** Not proof of cross-version determinism or improved detector mAP. Training/validation re-evaluation is a separate prerequisite for promotion. E33 full test workflows already started with earlier frozen commit snapshots and have **not** been silently replaced with this branch.
