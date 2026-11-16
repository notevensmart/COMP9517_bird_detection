# Model challenge audit — classical Beach Bird detector (16 Nov 2026)

This audit derives from the **completed original-image E30 EDA** and measured **E19/E23/E27/E29/E31 Selective Search** experiment logs, not from assumptions about modern neural object detectors. The complete image/box evidence is preserved in the original artifacts. The workstream excludes all neural networks and pretrained image embeddings.

## 1. Root causes established by existing evidence

| Bottleneck | Evidence and strength | Consequence | Current intervention |
|---|---|---|---|
| Proposal alignment, particularly large birds | E31 Selective Search raw R50=91.03% vs LoG 73.08%; R75=37.18% vs 19.23%; large R50=100% vs 54.24%, same VAL30/156 birds | Whole birds cannot be correctly localized if candidates are body parts | E32 trains 4D offset/size box regression on **TRAIN-only** matched proposals, comparing identical classifier scores before and after |
| Strong background/confuser false alarms | E31-SS only 88 correct of top600; seaweed 1379 and reeds 1419 remain; drone-like scene 1606; 1363 FP at score>=0 with SS vs 777 LoG under different score/feature domains | Candidate recall improvements may lower AP; naive negative additions also hurt (E27) | Log top20 FP and original photos for every variant; avoid more random negatives. Follow up with more discriminative object shape/context features only if data supports |
| Small birds / crowded flocks | E30: 34.4% train GT area<1024px²; 20+ birds in 87 TRAIN/20 VAL images; GT boxes overlap in 40.5% TRAIN scenes | Standard NMS may remove adjacent birds; SS small-object R50 78.26% vs LoG 91.30% | E32 evaluates NMS=.35 against .60 with top100 recall by scale and crowding, rejects changes if key regimes regress |
| Localization precision | SS raw R75 only 37.18% on E31 subset vs R50 91.03% | Many near-miss boxes and parts of birds are high-ranked; increasing classifier confidence cannot fix IoU | TRAIN-only classical regression as cautious test; report IoU50 and IoU75 separately |
| Label omissions / false negatives | E30 original annotated photos confirm some zero-GT scenes contain visible birds (VAL 1435); 154 TRAIN and 33 VAL images have no annotation | Some official FP may be genuine birds; blindly mining all empty-label patches can corrupt training | Exclude all zero-GT TRAIN from supervised experiments; preserve official VAL/TEST GT; flag misleading official FP separately; no manual score corrections |
| Small sample and overfitting | 17.18% SS score from only 45 TRAIN/30 VAL; E14 train-data plateau for a different LoG pipeline; E31 classifier train patch accuracy near 100% with low mAP | Overinterpretation risk; training scale might not generalize | E32 uses 100 TRAIN/60 VAL, then E33 trains on 200 TRAIN and confirms on VAL complementary to E32 before full 240-image TEST |
| Compute/time | SS generation ~1.51 seconds per image vs ~0.20 LoG in E31 | Proposal overhead makes full evaluation costly; candidate proliferation can overflow RAM | Keep one SS proposal cap=1200; stream validation/test images during E33; report total and per-image inference time |

## 2. Experiments deliberately not repeated

- E19 increased proposal coverage to ~90% but **lowered actual detector AP**: raw recall is not final ranked detection quality.
- E23/E27 diversified or added negatives without a convincing end-to-end gain; duplicate blind hard-negative experiments are not justified.
- E31 Selective Search+LoG **unrestricted union** raised raw R50 to 96.15% but dropped mAP to 11.35% from SS 17.18%; do not use unconstrained unions.
- E31-C tested Linear SVM, RBF SVM, RF, LightGBM on the same *LoG* features; LightGBM won that narrow ranking benchmark at 6.56% but **is not** evidence it improves the SS+stage2 RBF architecture.

## 3. Gate for a final test

The original E31-SS architecture remains the incumbent until E32 provides a verified better detector on paired VAL data. A corrected variant must raise COCO mAP50 by at least **2 percentage points**, not lose top20 TPs, and avoid a major loss of small or crowded recall. A separate complementary VAL confirmation is required. Freeze code, image identities, NMS threshold, feature definitions, classical fit procedures, proposed model and training size **before** accessing the 240-image test partition. Once TEST is evaluated, **no further model development from TEST feedback**.

All claimed scores must be labelled with their training count, validation/test count, mAP protocol, and whether the model changed proposal distribution. No claim of 30%, 50%, or any new score without actual run evidence.
