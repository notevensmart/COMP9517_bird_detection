# E42 — Primary classical detector papers and implementation fidelity

These references describe the original methods. E42 implements bounded adaptations and **must not be described as exact reproductions**.

1. Felzenszwalb, P. F., Girshick, R. B., McAllester, D., and Ramanan, D. (2010). **Object Detection with Discriminatively Trained Part-Based Models.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(9), 1627–1645. https://doi.org/10.1109/TPAMI.2009.167
   - Primary paper: https://faculty.cc.gatech.edu/~afb/classes/CS7495-Fall2014/readings/lsvm-pami.pdf
   - Official historical DPM source: https://www.rossgirshick.info/latent/
   - E42: one-component star model, HOG root and four deformable parts, displacement quadratic + linear penalties, alternating latent-part update and LinearSVC, one TRAIN-only hard-negative round. The original uses multiscale HOG feature pyramids, multiple components and considerably more latent training; this branch does **not** exactly reproduce it.

2. Perronnin, F., Sánchez, J., and Mensink, T. (2010). **Improving the Fisher Kernel for Large-Scale Image Classification.** European Conference on Computer Vision (ECCV), LNCS 6314, pp. 143–156. https://doi.org/10.1007/978-3-642-15561-1_11
   - Primary paper: https://www.micc.unifi.it/bagdanov/pdfs/peronnin_etal_ECCV10.pdf
   - E42: dense SIFT local descriptors, TRAIN-only eight-component diagonal GMM, first- and second-order mean/variance gradients with mixture-weight scaling, signed power normalization exponent 0.5, L2, linear SVM. The paper studies image classification rather than bird bbox detection; the small GMM, fixed grid and region proposals are documented adaptations.

3. Dollár, P., Appel, R., Belongie, S., and Perona, P. (2014). **Fast Feature Pyramids for Object Detection.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(8), 1532–1545. https://doi.org/10.1109/TPAMI.2014.2300479
   - Primary paper: https://vision.cornell.edu/se3/wp-content/uploads/2014/09/DollarPAMI14pyramids_0.pdf
   - E42: LUV colour (3), normalized gradient magnitude (1), histogram of six oriented gradients (6), four-by-four cell aggregation and smoothing, AdaBoost with depth-two trees. Unlike the original 2048 trees and fast approximate multiscale channel pyramids, E42 uses 160 learners and GT-blind Selective Search boxes to allow a paired low-resource comparison. Not an exact ACF reproduction.

## Shared protocol and scientific safeguards

- Same deterministic train sample, same validation sample, same capped GT-blind Selective Search regions, same NMS and COCO 101-point AP50 maxDets100 for all 3 arms.
- Ground truth is used to construct positive crops and safe negatives on TRAIN only. Zero-GT TRAIN images are excluded because some official annotations are incomplete. VAL labels are never used for training or selection.
- Each method runs independently and logs proposal R50/R75 by GT size, runtime, top20 matched TP/FP, per-image CSV and original-image overlays.
- A capped object proposal bank **is not** equivalent to the original multi-scale window detector for DPM/ACF. Classical method family similarities should not be overstated.
- The official test partition was already evaluated in E33 and is **not used again** in E42. No claim of novel independent TEST generalization from these VAL-only studies.
