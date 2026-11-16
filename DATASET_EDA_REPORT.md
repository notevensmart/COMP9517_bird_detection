# E30 — Full Beach Bird Dataset Exploratory Data Analysis

**Executed:** 2026-11-16. **Grounding:** complete original Kaggle `liugalhigh/beach-bird` TRAIN and VAL splits, measured with `src/full_dataset_eda.py`. **Exact run:** [43811582657](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43811582657). **Original CSV, JSON, PNG plots and GT-annotated photo contact-sheet ZIP:** [E30 artifact 14200561882](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43811582657/artifacts/14200561882). The TEST split was never examined. EDA does not train a model.

## Executive findings: what actually matters for the detector

The task is **not a single-scale bird shape classification problem**. There is a severe **scale and density mixture**: distant ducks/waders measuring only ~8–30 pixels, dense flocks of 50–130 annotations, large isolated gulls/herons/waders occupying much of the image, flying birds, swimming birds, and ground birds. Visual source images include water, surf, sand, rocks, marsh plants, reeds, sky, blurred backgrounds and shadowed cliff edges. This explains why a uniform LoG box bank and globally resized 64×64 HOG crop produces excessive partial-bird/semantic-background false positives.

Three research priorities follow from measured data:
1. **Scale-aware proposal/verification architecture:** 34–35% of GT bird boxes have area <1,024 px², while ~18% exceed 10,000 px². At least two or three distinct scale/shape regimes are necessary to test; do not assume that resizing every candidate to a 64×64 square preserves bird structure.
2. **Crowding and tight box regression:** 555/1,369 TRAIN images and 105/291 VAL images have overlapping GT boxes at IoU >.1; 10–12% of bird boxes touch the image border (some legitimately truncated). Selection/NMS should avoid collapsing nearby individual birds. Validate IoU=.75 alongside .5.
3. **Difficult texture and annotation quality:** scene backgrounds include bird-like seaweed, rocks, surf and flying non-birds. 154 TRAIN /33 VAL images are zero-labelled, but manual review found some birds in zero-GT images (see `DATASET_QA_FLAGS.md`). Do not treat every empty-label image as a reliable bird-free training negative. Among annotated birds, ~36–40% have a low **crop-versus-expanded-box mean grayscale intensity difference** of <2 gray levels; this statistic is **not foreground-background separation** and should not be interpreted as a true segmentation contrast estimate. It suggests simple mean-colour thresholding deserves skepticism, not that birds are truly low contrast.

## Exact dataset-wide measurements

| Statistic | TRAIN | VAL |
|---|---:|---:|
| Images | 1,369 | 291 |
| Image dimensions | all **512×512** | all **512×512** |
| Annotated bird boxes | **8,663** | **1,843** |
| Images with zero annotated birds | 154 (11.25%) | 33 (11.34%) |
| Median boxes/image (including empty images) | 3 | 3 |
| Images with 10+ bird boxes | 240 (17.5%) | 50 (17.2%) |
| Images with 20+ bird boxes | 87 (6.35%) | 20 (6.87%) |
| Images with 50+ bird boxes | 12 | 3 |
| Max GT birds in one image | **130** | **100** |
| Bird bbox median area (px²) | 1,883 | 1,618 |
| Bird bbox area <256 px² | 772 (8.9%) | 154 (8.4%) |
| Bird bbox area <1,024 px² | 2,978 (34.4%) | 651 (35.3%) |
| Bird bbox area >=10,000 px² | 1,567 (18.1%) | 337 (18.3%) |
| Median shorter bbox side | 36.4 px | 35.1 px |
| Shorter bbox side <16 px | 1,117 (12.9%) | 230 (12.5%) |
| Bbox aspect ratio >2 (wide) | 633 (7.3%) | 78 (4.2%) |
| Bbox aspect ratio <0.5 (tall) | 280 (3.2%) | 70 (3.8%) |
| Bbox touching image boundary | 875 (10.1%) | 214 (11.6%) |
| Images containing pairs of GT with IoU>.1 | 555 (40.5%) | 105 (36.1%) |
| Median whole-image Laplacian-variance blur proxy | 771 | 1,072 |
| Median whole-image Canny edge density | 10.7% | 12.1% |
| Exact decoded pixel-hash duplicate groups | 0 | 0 |

The exact numbers follow from original YOLO boxes and OpenCV pixel measurements, not model-generated boxes. A dHash exact-match check also found zero groups, but does not rule out resized/recompressed/near-duplicate scenes. Median brightness differs mildly between splits (TRAIN 136.9, VAL 133.6), but descriptive statistics alone do **not** prove train/VAL distribution equivalence. File names are contiguous by split: TRAIN `beach_bird_0001...1369` and VAL `1370...1660`; this is **not evidence of a random image-level split**, and source/capture scene grouping deserves auditing.

### Image atlases and concrete qualitative observations

Use original E30 artifact contact sheets, not stock references:
- `contact_sheets/val_area_px2_smallest_01.jpg`: large stretches of shore/water and very small distant waders/ducks (e.g. `1373,1453,1455,1486,1490,1599`). Small GT boxes closely resemble water highlights at this resolution.
- `contact_sheets/val_area_px2_largest_01.jpg`: single birds occupying enormous regions (`1482,1508,1533,1545,1564`), including flying shapes, tall standing birds and varied complex outlines; fixed small LoG-based boxes cannot cover all of these.
- `contact_sheets/val_most_crowded_01.jpg`: `1658` has 100 GT birds, `1453` 58, `1373` 53; tightly adjacent birds in mixed scales. Hard NMS can suppress real birds and top-100 COCO maxDets may constrain images with exceptionally dense flocks.
- `contact_sheets/val_aspect_ratio_largest_01.jpg`: swimming wide birds, lateral flying birds and body-part occlusion. Square 64×64 stretching distorts those silhouettes.
- `contact_sheets/val_low_scene_contrast_01.jpg`: nearly uniform water/sky in some scenes, but same set includes kites and drones (negative-only images) alongside subtle waders. A 'low whole-scene contrast' proxy is **not** the same as low bird-to-background contrast.

## Important architectural scoping characteristics

### 1. Three different object regimes
- **Tiny (area <1,024 px²):** ~one-third of annotations. HOG spatial-cell sizes can be larger than the original bird silhouette. Prefer full-resolution small-object proposal stage, scale-specific descriptors, and controlled contrast normalization. Distinguish bird pixels from white foam, ripples, sand patterns and glints.
- **Medium (1,024–10,000 px²):** another substantial fraction (by arithmetic ~47%). Region-based proposals and texture/edge descriptors may generalize with less interpolation.
- **Large (>=10,000 px²):** ~18%. Need large candidate windows, meaningful parts, head/body/wing structure and better region alignment. The original LoG/Otsu maximum spatial extent had much poorer recall for this tier (earlier experiment: 54.2% at IoU .5 on 59 large GT).

### 2. Context is heterogeneous and sometimes misleading
Water/sand background can be easy, but visible scene families also include cliffs, dark seaweed, waves, rock outcrops, coastal grass, mudflats, open sky and drones/kites. A fixed absolute image-position rule or colour histogram would reject legitimate birds in other scenes. Train balanced, **scene-stratified** negatives and evaluate each real background family separately; curated more-negatives E27 failed, meaning more unstructured data is not sufficient.

### 3. Density has implications for evaluation
In 291 VAL images, 20 have at least 20 annotated birds and 3 have 50+; max is 100. COCO evaluation uses maxDets=100, so do not prune proposals below expected GT count in dense scenes. Explicitly inspect detection failures on `1658`, `1453`, `1373`, etc., and vary NMS only on separate validation subsets to avoid tuning bias. Some GT boxes overlap significantly, so overlapping predictions are not necessarily duplicates.

### 4. Bounding boxes are not segmentation masks
The data supplies rectangular ground-truth annotations only. A reported 'inside GT bbox' colour/texture statistic includes surrounding water/sand/vegetation for irregular bird shapes, and a local expanded crop also includes the object. **Do not claim these stats measure true bird/background colour separation**. To test actual segmentation viability, manually annotate a few foreground masks across representative scenes (or use GT-guided GrabCut diagnostically, with GT confined to TRAIN/visual QA).

### 5. Border and label problems
About 10–12% of boxes touch or nearly touch image edges; generic cropping/fixed box padding and regression must clip safely while preserving truncated targets. Two known manually suspicious zero-annotation TRAIN images and one VAL image demonstrate false-negative annotation risk. Maintain original labels for official metrics and archive human decisions separately. The entire dataset has not been manually checked object-by-object, so there may be other annotation omissions.

### 6. What not to conclude
- 512×512 source images do **not** imply uniform bird scale.
- LoG proposal recall and HOG patch training accuracy are not detection mAP.
- EDA cannot establish detector mAP improvement without a controlled paired training experiment.
- Zero exact duplicate image hashes does not guarantee no related frames or scene leakage.
- Overall edge density, Laplacian variance or broad colour statistics are imperfect proxies; they do not independently identify habitat type or image quality.

## Priority experiments motivated by the EDA

| Priority | Experiment | EDA evidence | Required success evidence |
|---|---|---|---|
| **P0** | Build scale-stratified EVAL breakdowns for every model | 35% small and 18% large birds; dense scenes | COCO mAP, per-size recall, GT-matched IoU, top20 FP, identical VAL subset |
| **P0** | Compare object-oriented EdgeBoxes/Selective Search vs LoG/Otsu on the full VAL | Tiny and huge birds, strong scenery texture | Proposal recall@.5/.75 by area+crowding; end-to-end mAP and inference time |
| **P1** | Separate small/large verification heads or feature pipelines | 12–13% GT with shorter side under 16 px; huge part scale variation | Paired mAP and per-tier PR, stronger generalisation over whole crop |
| **P1** | Explicit morphology/parts and aligned box refinement | Irregular flying/standing/swimming contours and weak IoU.75 | IoU.75 recall, fewer near-miss/localisation FP, visual box alignment |
| **P1** | Scene-stratified FP classifier with hard confuser categories | Water/rock/seaweed/reeds/sky/drone all present | Lower top20 FP *in each hard scene*, no severe TP loss |
| **P2** | Manual annotation integrity and group-wise split audit | Confirmed missing GT and contiguous split filenames | Reviewable QA sheet, source-scene grouping, no TEST-derived tuning |

**Integrity policy:** EDA was performed on TRAIN+VAL only. All visual confidence labels are based on supplied ground truth; no TEST was examined. Manual visual observations are labelled as observations, and all detector design suggestions are hypotheses rather than validated gains.
