# Dataset annotation integrity: manual visual QA

The dataset is supplied as `liugalhigh/beach-bird` with images and YOLO labels. An empty label file is *not proof* that the photograph contains no birds. We reviewed generated TRAIN/VAL contact sheets and independently validated the full-resolution examples.

## Measured scope
- All 1,369 TRAIN images checked: 154 (11.25%) have zero annotations.
- All 291 VAL images checked: 33 (11.34%) have zero annotations.
- Those counts do **not** mean all are mislabeled. Most visually inspected examples show rocks, waves, drones, aircraft, kites, livestock, etc. and seem legitimate negatives.

## High-priority review flags

| Split | Image | Official GT | Human-visible content | Action |
|---|---|---:|---|---|
| TRAIN | `beach_bird_0149.jpg` | 0 | Numerous flying bird-shaped silhouettes | **Quarantine from negative sampling pending target-scope review** |
| TRAIN | `beach_bird_0418.jpg` | 0 | Two obvious wading/shorebirds on wet sand | **Quarantine from negative sampling pending annotation review** |
| VAL | `beach_bird_1435.jpg` | 0 | Many standing shorebirds and two taller waders, unmistakable target-type birds | **Retain in official VAL metrics; flag as potential annotation error, do not reannotate silently** |

The optional machine-readable training filter is `src/label_quality.py`. It removes only manually flagged suspected zero-label TRAIN files from training or mining; it never touches VAL or TEST. Paired-model results **before the quarantine** remain unchanged and should be cited as historic experiments. New runs must report the number/names filtered (the fixed 100-image subset may contain none).

## Visual evidence
- [Human-checkable contact sheets, every zero-label TRAIN and VAL image](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43798758419/artifacts/14196210026).
- [Side-by-side detector/annotation outputs](https://github.com/notevensmart/COMP9517_bird_detection/actions/runs/43797974207/artifacts/14195860401).
- The official benchmark is based on supplied labels; errors due to missing annotations are still penalized unless the evaluation set is officially corrected.

## Paper discussion
A false-positive box can correspond to (a) truly irrelevant background, (b) a loose/oversized localisation near a labelled bird, (c) a duplicate prediction, or (d) a visually real bird without supplied GT. Categories (a)–(c) are measured by annotation geometry; category (d) requires human review. Report primary mAP on the untouched official validation/test data, and only report any annotation-corrected evaluation as a separately titled sensitivity analysis with auditable annotation corrections.

Do not use validation ground truth to fit or correct the classifier. No TEST images have been inspected here.
