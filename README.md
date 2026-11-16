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
