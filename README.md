# COMP9517 — Beach Bird Detection Cloud Audit

This repository runs a reproducible **CPU-based quick audit** of the public [Beach Bird Detection Dataset](https://www.kaggle.com/datasets/liugalhigh/beach-bird) for COMP9517. It is a first step towards a rigorous comparison, **not a final submission or a verified 50% mAP model**.

## Execute

Open [Actions → Bird Detection Audit](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/bird-audit.yml) to inspect the automatically triggered cloud run or choose **Run workflow** to execute again. The runner installs dependencies, performs synthetic box/IoU/AP tests, downloads Kaggle data, selects fixed-seed 80 training and 20 validation images, audits Canny proposal quality, trains a Canny + HOG + linear SVM classifier, and computes exploratory global AP50.

After completion, download `bird-quick-audit-results` under **Artifacts** on the run page; it contains `proposal_summary.csv`, `proposal_by_image.csv`, `metrics.json`, and `gate.json` when those stages execute successfully.

## Research safeguards

- `proposal_R50` is only **candidate-box recall**, an upper bound on achievable detection recall; it is not detection AP/mAP.
- `AP50_continuous` is the single-class, globally score-ranked continuous interpolation variant. It is not COCO AP50:95.
- These subset runs serve diagnosis, not unbiased validation or final testing.
- Training uses **train** images only. Validation is for exploratory measurement. **Test** is untouched.
- CPU runners may be unsuitable for full deep-neural-network training. A successful quick run does not unlock full training without passing tiny-set overfitting and proposal-quality gates.
- Current quick implementation is a clean-room experimental baseline and **does not exactly reproduce the existing Windows notebook's patch sampling**. Historical notebook metrics must be labelled separately.

## Local execution

```bash
python -m pip install -r requirements.txt
python src/cloud_audit.py --self-test
python src/cloud_audit.py --dataset /path/to/Dataset --output artifacts
```

No dataset images, pretrained weights, or private credentials are committed.


## E31: classical machine-learning classifier ablation (16 Nov 2026)

The [E31 cloud workflow](https://github.com/notevensmart/COMP9517_bird_detection/actions/workflows/e31-classifiers.yml) compares four **non-neural** classifiers on the same existing LoG/Otsu proposals and 64×64 HOG+LBP descriptors: Linear SVM, RBF SVM, Random Forest and LightGBM. It trains on 45 annotated TRAIN images and evaluates on a fixed 30-image VAL subset with COCO 101-point AP50, top-20 ranking, scale/crowding diagnostics, training time and scoring latency. No TEST images, missing-label zero-GT TRAIN negatives, extra hard-negative rounds or validation hyperparameter searches are used.

- Source: [src/classifier_benchmark.py](src/classifier_benchmark.py)
- Protocol, historical failures and status: [RESEARCH_LOG.md](RESEARCH_LOG.md)
- Download run artifact `e31-classical-classifier-results` for `e31_results.json` and `e31_table.csv`.
- **Measured results and real prediction overlays are now available in the linked run.** High patch-level training accuracy and proposal recall do not establish good validation mAP.


### E31-C measured outcome and actual image review

[Successful reproducible visual run](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43820800202): on 45 TRAIN / 30 VAL (156 GT birds), single-stage COCO AP50 results are Linear SVM **4.19%**, RBF SVM **5.46%**, Random Forest **5.88%**, LightGBM **6.56%**. All remain poor; no neural models were used, TEST remains untouched. Eight real 4-model image overlays show false positives on seaweed/reeds/drones, whole-bird box failures, and nonlinear classifiers missing all top20 birds in a 26-bird crowded scene. `beach_bird_1435.jpg` has zero official labels despite visible birds; do not treat every officially false box as semantically false.

- Full evidence and research decision: [CLASSIFIER_COMPARISON_REPORT.md](CLASSIFIER_COMPARISON_REPORT.md).
- Real model images, exact metrics and per-image CSV: [GitHub Actions artifact](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43820800202/artifacts/14204626929).
- The separate E31-SS Selective Search + nonlinear RBF stage-2 detector achieved **17.18%** AP50 on a matched 45/30 study; this is **not** a like-for-like LightGBM-versus-SS comparison, but supports prioritizing proposal alignment rather than a mere classifier swap.
