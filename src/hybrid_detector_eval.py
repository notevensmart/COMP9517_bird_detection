"""End-to-end classical bird detector with repaired hybrid proposal scale coverage.

Uses large-window spatially balanced edge-score candidates + small-object LoG
and local Otsu segments. Train only on TRAIN GT+jitter positives and GT-safe
TRAIN full-image negatives. Evaluate actual COCO mAP50, FP/FN, NMS ablations,
and annotated real validation images. No pretrained neural models.
"""
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np
from label_quality import unsafe_training_negative
from cloud_audit import data,iou
from multiscale_coverage import methods
from rebuild_detector import (safe_labels,descriptor,make_train_features,fit,
    hard_negatives,scored_predictions,save_visuals,tests as rebuild_tests)

DIM=1924
def bank(samples,phase):
    records=[];start=time.perf_counter()
    for i,(path,gt) in enumerate(samples):
        image=cv2.imread(str(path))
        t=time.perf_counter()
        b=methods(image,per_shape=50)["Hybrid_union"]
        proposal_seconds=time.perf_counter()-t
        t=time.perf_counter()
        X=np.stack([descriptor(image,v) for v in b]) if len(b) else np.empty((0,DIM),np.float32)
        feat_seconds=time.perf_counter()-t
        p,n=safe_labels(b,gt)
        records.append(dict(path=path,boxes=b,gt=gt,pos=p,neg=n,X=X,
                            proposal_seconds=proposal_seconds,feature_seconds=feat_seconds))
        if (i+1)%20==0 or i+1==len(samples):
            print("HYBRID BANK",phase,i+1,"/",len(samples),"elapsed",round(time.perf_counter()-start,1),flush=True)
    return records

def main(root,out,train_limit,val_limit):
    out.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",train_limit);val=data(root,"val",val_limit)
    flagged=[p.name for p,gt in train if unsafe_training_negative(p.name,len(gt))]
    train=[(p,gt) for p,gt in train if not unsafe_training_negative(p.name,len(gt))]
    print("TRAIN QA QUARANTINED",flagged,flush=True)
    trainbank=bank(train,"train")
    valbank=bank(val,"val")
    pos_gt,pos_proposal,initial,trainbank,feature_counts=make_train_features(trainbank,out)
    results=[];records_by_variant={}
    negative=initial
    visual_records=None;visual_errors=None
    for rd in range(3):
        started=time.perf_counter()
        model,train_accuracy=fit(pos_gt,negative)
        train_sec=time.perf_counter()-started
        m35,records35,errors35=scored_predictions(model,valbank,.35)
        m65,records65,errors65=scored_predictions(model,valbank,.65)
        rec=dict(method="GT_jitter_hybrid_fullimage_mining",round=rd,
                 positive_training_patches=len(pos_gt),negative_training_patches=len(negative),
                 mean_proposals=float(np.mean([len(v["boxes"]) for v in valbank])),
                 train_accuracy=train_accuracy,train_fit_seconds=train_sec,
                 nms65_map50=m65["map50_coco101"],**m35)
        results.append(rec)
        if rd==2:
            visual_records={x["path"].name:x for x in records35}
            visual_errors=errors35
        if rd<2:
            extra,diagnostic=hard_negatives(model,trainbank,limit=600,per_image=10)
            negative=np.concatenate((negative,extra))
            rec["mining"]=diagnostic
        print("HYBRID FINAL DETECTOR ROUND",json.dumps(rec),flush=True)
    reporter={"hybrid":dict(records=visual_records,errors=visual_errors)}
    visuals=save_visuals(out,valbank,"hybrid",reporter)
    with (out/"hybrid_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=visual_errors[0]);wr.writeheader();wr.writerows(visual_errors)
    summary=dict(train_images=len(train),validation_images=len(val),
          ground_truth_birds=sum(len(v["gt"]) for v in valbank),feature_counts=feature_counts,
          methods="classical LoG + Otsu plus image-edge-ranked medium/large windows; HOG + LBP + LinearSVC",
          candidate_budget="small LoG/Otsu (~960) + large balanced grid (~960)",
          metric="COCO single-class bird bbox mAP@0.5 101pt maxDets100",
          results=results,visual_images=visuals,
          warnings=["All performance is exploratory VAL; no TEST samples used.",
                    "NMS .65 is a validation ablation, not yet frozen.",
                    "Training patch accuracy is not detector mAP.",
                    "Visual FP from official labels may include unannotated visible birds; manually review labels."])
    (out/"hybrid_detector_results.json").write_text(json.dumps(summary,indent=2))
    print("HYBRID DETECTOR COMPLETED",json.dumps(summary),flush=True)

def test():
    rebuild_tests()
    image=np.zeros((512,512,3),np.uint8)
    mask=methods(image,8)["Hybrid_union"]
    assert mask.ndim==2 and mask.shape[1]==4
    if len(mask):assert np.all(mask[:,2:]>mask[:,:2])
    print("PASS hybrid geometry and parent classical detector checks")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("hybrid_results"))
    ap.add_argument("--train-limit",type=int,default=100)
    ap.add_argument("--val-limit",type=int,default=30)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args();test()
    if not args.self_test:
        if not args.dataset:ap.error("--dataset required")
        main(args.dataset,args.output,args.train_limit,args.val_limit)
