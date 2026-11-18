# COMP9517 — Classical Bird Detection Research Vertical Slice (Report Plan)

Status: agreed outline; **not** a completed report. Owner contribution to be checked against team division of work. Intended integration: IEEE two-column group report, approximately **2–3 pages** out of the **10-page main-text limit** (references excluded). No additional experiment should be claimed completed without verified artifacts.

## Editorial direction

Tell a **focused, evidence-based research story**, not a catalogue of 40+ exploratory runs. Present **five pivotal investigations** chosen for the information they yielded. Narrative arc: **problem → hypothesis → controlled comparison → quantitative and qualitative evidence → interpretation → next design decision**.

Working title: **An Experimental Investigation of Classical Bird Detection: From Region Proposals to Robust Recognition**.

Central question: *What limits classical bird detection in complex natural scenes, and how do proposal quality, appearance representation, generalization and post-processing interact to affect end-to-end accuracy?*

No neural networks or pretrained deep learning detectors. All development on TRAIN and VAL; prior frozen E33 TEST evaluation must not be used for further model selection or diagnostics. Include the other group method(s) in the final integrated report, with at least two meaningfully different approaches, and report runtime as required by specification.

## Five main investigations

1. **Region proposal quality — E31 Selective Search vs LoG/Otsu.**
   Hypothesis: better bird-aligned candidate boxes will improve recall and localization.
   Show representative proposal overlays, R50/R75 and size-dependent recall, and detection mAP in **paired matching conditions**. Discuss the paradox that an unrestricted union improved candidate recall but worsened detector AP. Avoid comparing small-subset E31 mAP as though it were directly paired with full-VAL E42.

2. **Handcrafted appearance and classification — E31-C / E28 (related evidence, clearly separated).**
   Question: why do HOG/LBP + classical classifiers confuse seaweed, reeds and bird parts? Show a controlled classifier comparison and real background false-positive overlays. Discuss high training-patch accuracy versus low end-to-end validation mAP **without calling their difference a valid generalization gap**: they measure different tasks. Describe two-stage Linear SVM and PCA + RBF SVM with colour/context features. E31-C and E28 use different experimental settings; report each denominator.

3. **Appearance augmentation — E42.**
   Compare paired full-VAL no-augmentation 17.43% vs augmentation 20.02% COCO 101-point mAP@IoU .50; top-20 TP 820 vs 809. Include original/augmented examples, mAP-vs-top20 ranking interpretation, analysis by scene density and, if prediction scores are recovered, precision–recall and confidence distributions. Training 1,215 annotated-positive TRAIN images; VAL 291 images / 1,843 annotated birds.

4. **Scale specialist model ablation — E43.**
   Global 17.76% vs specialist-only 17.04%; blends 25/50/75%: 16.86%, 16.76%, 16.83% VAL mAP50. Top-20 TP global 818; experts 832; 50% blend 837. Explain ranking/operating-point trade-off: higher top-20 TP did not yield higher global AP. Visualize the mAP / TP trade-off, ideally by object size using recoverable predictions; do not claim improved size-specific AP unless computed.

5. **Post-processing and failure audit — E44.**
   Compare Hard NMS IoU 0.35 and 0.50, linear Soft-NMS and Gaussian Soft-NMS on **frozen E42 augmented checkpoint**, 291-image VAL. Only report verified complete-run results. Check hard035 reproduces 20.01565% E42 baseline (allow only justified numeric tolerance); Soft-NMS scores must have consistent semantics and stable global ranking, with any transformed-score variants labeled appropriately. Include crowded-scene overlays, duplicate-vs-suppressed detections, per-image failures, size/crowd stratification, confidence ranking and localization diagnostics. Never silently replace the earlier baseline.

## Reproducibility note (not a sixth headline experiment)

Brief note on **E36–E37**: nondeterministic Selective Search output order, first-1200 cap caused different candidate sets; canonicalization + deterministic selection addressed the specific repeatability issue. Distinguish the newly selected proposal distribution from the original E33 TEST-frozen implementation.

## Visual evidence package (prioritize 3–4 figures in the 2–3 page slice)

- A single **detector evolution schematic**: original classical approach → Selective Search → stage 1 Linear SVM/HOG/LBP → PCA64/RBF SVM + 32D context → NMS.
- **Paired proposal/mAP trade-off panel** with explicit dataset counts; no unqualified cross-subset trend lines.
- **E42 vs E43 quantitative ablation**: two-axis plot or separate panels for mAP50 and top-20 TP (metrics defined), plus augmentation image pair.
- **Actual failure montage** (not synthetic or web images): seaweed/vegetation background confusers, crowded shorebirds, large or part-bird localization, missed proposals; ground truth and prediction legends.
- Optional appendix/research backup: training-size learning curves, hard-negative mining-round plots, error taxonomy, score histograms, PR curves, size/crowd breakdowns, reproducibility diagnostics.

### Accuracy rules for graphs

SVM fitting is **not epoch-based**, so do not invent epoch accuracy or loss curves. Prefer sample-size learning curves, mining-round performance and inference diagnostics only if earlier run artifacts contain those measurements. Distinguish train-patch accuracy from train detection mAP. Report actual confidence intervals only if computed from predictions. Never fabricate numbers, curves, visualizations or E44 results.

## Draft section allocation for integration

- Methods / design rationale: ~0.5 IEEE page.
- Selected experimental investigations and compact results: ~1.0 IEEE page.
- Failure analysis and discussion: ~0.75–1.0 IEEE page.
- Reserve figure/table footprints; final editor may rebalance.

## Research artifact and evidence handling

Preserve exact split sizes, validation denominators, model/checkpoint version, IoU=.50, COCO 101-point AP with maxDets=100, top-20 one-to-one TP matching and runtime. Link runs and artifacts in internal notes; the report should cite papers/software/dataset properly. Do not enumerate 40+ experiments in the submitted narrative; retain the complete EXperiment Registry and Research Log untouched for audit and honest provenance.

Sources for further drafting: EXPERIMENT_REGISTRY.md, RESEARCH_LOG.md, CLASSIFIER_COMPARISON_REPORT.md, MODEL_CHALLENGE_AUDIT.md, E37_STABLE_PROPOSALS_REPORT.md, E42_RESEARCH_PROTOCOL.md; E42 run 43974171244; E43 run 43974178323; E44 original failed 44421640677 and repaired run 44423104662 (verify final status and outputs before quoting).

## Recommended writing voice

Scientific, analytical and concise. Each investigation should communicate **what we believed, what we measured, why the result was surprising, and how the evidence changed our next decision**. Report negative results without disguising them. The vertical slice should merge cleanly with teammates' Introduction, Literature Review, second method and Conclusion.
