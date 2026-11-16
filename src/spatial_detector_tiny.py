"""End-to-end tiny-image diagnostic with scale-balanced candidate generator.

Generates candidates solely from pixels, limits candidates to <=5400, mines
false positives, filters final predicted boxes with a frozen margin threshold,
NMS, and a 150-detection safety cap. This is NOT final validation/test mAP.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np, cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,feature,iou,ap50
from tiny_anchor_train import targets
from spatial_pruning import generate_budgeted
from anchor_proposals import best_coverage
from detection_postprocess import final_detections

def make_bank(path,gt,per_shape=200):
    im=cv2.imread(str(path))
    b,raw=generate_budgeted(im,per_shape)
    features=np.stack([feature(im,q) for q in b])
    return {"image":im,"boxes":b,"X":features,"gt":gt,"raw_grid":raw}

def test_model(banks,model,margin=0.0):
    predictions=[];false_positives=[];counts=[]
    for entry in banks:
        scores=model.decision_function(entry["X"])
        pred=final_detections(entry["boxes"],scores,margin,max_detections=150,nms_iou=.35)
        b=pred["boxes"];s=pred["scores"]
        predictions.append((b,s,entry["gt"]))
        best=iou(b,entry["gt"])
        m=best.max(axis=1) if len(entry["gt"]) else np.zeros(len(b))
        false_positives.append(int((m<.5).sum()))
        counts.append(len(b))
    return dict(AP50=ap50(predictions),mean_final_detections=float(np.mean(counts)),
                mean_final_FP=float(np.mean(false_positives)))

def run(root,out,rounds=6):
    out.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",0)
    positive=[r for r in train if 1<=len(r[1])<=8]
    negative=[r for r in train if not len(r[1])]
    rng=np.random.default_rng(42)
    chosen=[positive[int(i)] for i in rng.choice(len(positive),2,False)]
    if negative:chosen.append(negative[0])
    start=time.perf_counter()
    banks=[make_bank(path,gt) for path,gt in chosen]
    feature_seconds=time.perf_counter()-start
    allX=np.vstack([e["X"] for e in banks])
    positive_ix=[];negative_ix=[];offset=0
    for e in banks:
        pos,neg=targets(e)
        positive_ix.extend(offset+np.flatnonzero(pos))
        negative_ix.extend(offset+np.flatnonzero(neg))
        offset+=len(e["X"])
    positive_ix=np.asarray(positive_ix,int)
    negative_ix=np.asarray(negative_ix,int)
    if not len(positive_ix) or not len(negative_ix):
        raise RuntimeError(f"Insufficient proposals: {len(positive_ix)} positives, {len(negative_ix)} negatives")
    selected=set(rng.choice(negative_ix,min(200,len(negative_ix)),replace=False).tolist())
    results=[];best=None
    for round_id in range(rounds+1):
        negatives=np.asarray(sorted(selected),int)
        indices=np.r_[positive_ix,negatives]
        y=np.r_[np.ones(len(positive_ix)),np.zeros(len(negatives))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",random_state=42,max_iter=10000))
        model.fit(allX[indices],y)
        # Margin >=0 is only a diagnostic operating point; NOT a calibrated confidence.
        scored0=test_model(banks,model,margin=0.)
        scored1=test_model(banks,model,margin=1.)
        scores=model.decision_function(allX[negative_ix])
        hardest=negative_ix[np.argsort(-scores,kind="stable")]
        new=[int(i) for i in hardest if int(i) not in selected][:600]
        row=dict(round=round_id,train_patch_accuracy=float((model.predict(allX[indices])==y).mean()),
                 positives=len(positive_ix),negative_used=len(negatives),
                 AP50_margin0=float(scored0["AP50"]),AP50_margin1=float(scored1["AP50"]),
                 mean_FP_margin0=scored0["mean_final_FP"],mean_FP_margin1=scored1["mean_final_FP"],
                 mean_detections_margin0=scored0["mean_final_detections"],
                 mean_detections_margin1=scored1["mean_final_detections"],
                 hard_negatives_found=len(new))
        results.append(row)
        print("SPATIAL MINING ROUND",json.dumps(row),flush=True)
        if best is None or row["AP50_margin0"]>best["AP50_margin0"]:best=row.copy()
        if row["AP50_margin0"]>=.9 or not new:break
        selected.update(new)
    with (out/"spatial_mining_curve.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=results[0]);writer.writeheader();writer.writerows(results)
    result={"dataset_split":"TRAIN_ONLY","train_images":[x[0].name for x in chosen],
      "GT_birds":sum(len(gt) for _,gt in chosen),
      "proposal_recall50":sum(int((best_coverage(e["boxes"],e["gt"])>=.5).sum()) for e in banks)/sum(len(e["gt"]) for e in banks),
      "mean_proposals":float(np.mean([len(e["boxes"]) for e in banks])),
      "feature_seconds":feature_seconds,"best_round":best,"history":results,
      "tiny_overfit_gate_pass":any(x["AP50_margin0"]>=.9 for x in results),
      "caveat":"Training-set memorisation AP with margin filter; no test or validation results"}
    (out/"spatial_detector_tiny_results.json").write_text(json.dumps(result,indent=2))
    print("SPATIAL DETECTOR TINY RESULT",json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",default=Path("spatial_results"),type=Path)
    p.add_argument("--rounds",type=int,default=6)
    a=p.parse_args();run(a.dataset,a.output,a.rounds)
