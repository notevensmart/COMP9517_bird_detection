# E43 — 50% TRAIN classical paper-derived benchmark (19 Nov 2026)

## Fixed experimental protocol

- **Official TRAIN:** 1,369 images. Use exactly **685 images** (50.0365%) chosen by `training_subset(dataset,685)`, fixed seed 61. This selects from **annotated-positive TRAIN** images only because E30 identified unreliable empty-label scenes; do not claim all original TRAIN images were equally eligible.
- **Official VAL:** 120/291 images selected with the repository's fixed VAL seed 43, disjoint from TRAIN. This validation set has been repeatedly mined in prior experiments and is **exploratory**, not a fresh independent test. Official **TEST is not accessed**.
- All arms use GT-blind Selective Search fast, unique sorted coordinates and fixed-seed area-stratified cap **1200 regions/image** before HOG/SIFT/ACF descriptors. This follows the E36/E37 reproducibility fix and corrects E42's low 250-cap R50. A candidate budget change confounds any direct E42 vs E43 data-size-only inference; these results measure **improved pipeline plus scale-up**.
- Positive training patches from GT boxes (at most 6 GT per image, stratified small/medium/large) plus one jitter crop for each image; negatives are GT-safe TRAIN-only proposals, up to 6 original and 2 reserved hard candidates per image. All 685 TRAIN images participate even though dense birds are subsampled. This protects CPU/RAM while avoiding crowded images dominating the classifier. The full TRAIN set is **read and sampled**, not all ~4,000 TRAIN bird instances/patches used.
- Independent fit per method and streaming feature extraction. VAL never used in model fitting or threshold tuning; save model weights, configuration and sampled image IDs.
- **DPM:** one component, HOG root + 4 deformable parts, three alternate latent part / linear SVM steps, one TRAIN-only reserve negative mining pass. Still omits original authors' mixture components and multiscale convolution.
- **Improved Fisher:** dense SIFT, **32 component diagonal GMM fitted TRAIN only** (up from E42 8), first/second order Fisher statistics, signed square-root, L2 norm, linear class-balanced SVM and TRAIN-only hard negatives. Does not claim that increasing GMM components guarantees gains.
- **ACF:** LUV + normalized magnitude + six oriented gradient channels, 4x4 aggregation, **256** depth-two AdaBoost estimators (up from E42 160), with 96-estimator TRAIN-only prescreening for a negative-mining pass. Omits original ACF's dedicated fast pyramid and longer boosted cascade.
- Metric: **single-class COCO 101-point bbox AP@IoU=.5, maxDets100**. Report raw proposal recall 50/75 by GT size, top20 TP/FP and mean end-to-end VAL inference seconds (image load, proposal generation, descriptor extraction, scoring and NMS). Save actual validation prediction overlays, per-image CSV, model.joblib and JSON.
- **No hyperparameter search on VAL**, no TEST rerun and no claims of exact research-paper reproduction. Existing 11.89% primary frozen E33 full TEST remains historical and incomparable to a new exploratory VAL result.

## Original paper citations

See [E42_PAPER_REFERENCES.md](E42_PAPER_REFERENCES.md) for full citations and URLs:
1. Felzenszwalb et al. (2010), *Object Detection with Discriminatively Trained Part-Based Models*, IEEE TPAMI.
2. Perronnin et al. (2010), *Improving the Fisher Kernel for Large-Scale Image Classification*, ECCV.
3. Dollár et al. (2014), *Fast Feature Pyramids for Object Detection*, IEEE TPAMI.

## Previous failure-driven decisions

- E31/E38: stronger region alignment from Selective Search than original LoG-only; capped GT-blind stable set is preferred; avoid unrestricted union.
- E36/E37: must canonicalize SS coordinates **before** capping to avoid selection-order instability.
- E15/E23/E27: naive scaling and negative oversampling can plateau or regress; use bounded, scale-balanced sampling.
- E28/E29: near-perfect patch classification does **not** imply detection AP; measure actual full-scene COCO.
- E30: substantial small/crowded birds and incomplete empty-image labels; preserve all official VAL GT for scoring while excluding zero-GT TRAIN scenes from negative mining.
- E42: 250-region cap yields raw R50 just 64.1%; do not compare E43 1200 cap changes as training-data-only gains.

## Outcome

See the [E43 GitHub Actions workflow](.github/workflows/e43-half-train.yml) and its downloadable per-model artifacts for measured outcomes once completed. Do not fill in metrics before verifying run completion and artifacts.
