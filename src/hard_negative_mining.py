"""Iterative hard-negative mining; tiny train images only. Diagnostic, not test mAP."""
import argparse,json,time,csv
from pathlib import Path
import numpy as np, cv2
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from cloud_audit import data,nms,ap50,iou
from tiny_anchor_train import make_bank,targets
from anchor_proposals import best_coverage

def evaluate_banks(banks,model):
    preds=[];mean_fp=[];mean_count=[]
    for e in banks:
        s=model.decision_function(e["X"])
        keep=nms(e["boxes"],s,.3,300)
        b=e["boxes"][keep];sc=s[keep]
        preds.append((b,sc,e["gt"]))
        m=iou(b,e["gt"])
        best=m.max(axis=1) if len(e["gt"]) else np.zeros(len(b))
        mean_fp.append(int(((sc>=0)&(best<.5)).sum()))
        mean_count.append(int((sc>=0).sum()))
    return ap50(preds),float(np.mean(mean_fp)),float(np.mean(mean_count))

def run(root,out,rounds):
    out.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",0)
    positive=[r for r in train if 1<=len(r[1])<=8]
    negative=[r for r in train if not len(r[1])]
    rng=np.random.default_rng(42)
    chosen=[positive[int(i)] for i in rng.choice(len(positive),2,False)]
    if negative:chosen.append(negative[0])
    banks=[make_bank(p,gt,450) for p,gt in chosen]
    allX=np.vstack([e["X"] for e in banks])
    assigned=[];offset=0;positive_ix=[];negative_ix=[]
    for e in banks:
        pos,neg=targets(e)
        positive_ix.extend(offset+np.flatnonzero(pos))
        negative_ix.extend(offset+np.flatnonzero(neg))
        offset+=len(e["X"])
    positive_ix=np.array(positive_ix,int);negative_ix=np.array(negative_ix,int)
    if not len(positive_ix) or not len(negative_ix):raise RuntimeError("No positive or negative patches")
    # Use all positive examples and start with a small uniform background sample.
    selected=set(rng.choice(negative_ix,min(200,len(negative_ix)),replace=False).tolist())
    history=[]
    for iteration in range(rounds+1):
        chosen_neg=np.array(sorted(selected),int)
        indices=np.r_[positive_ix,chosen_neg]
        labels=np.r_[np.ones(len(positive_ix)),np.zeros(len(chosen_neg))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=10000,random_state=42))
        start=time.perf_counter();model.fit(allX[indices],labels);fit_time=time.perf_counter()-start
        ap,fp,count=evaluate_banks(banks,model)
        scores=model.decision_function(allX)
        negatives=scores[negative_ix]
        # Hard negatives are highest-scoring backgrounds, including where SVM scores > 0.
        ranked=negative_ix[np.argsort(-negatives,kind="stable")]
        extra=[int(idx) for idx in ranked if int(idx) not in selected][:min(600,len(ranked))]
        row={"round":iteration,"AP50_tiny_train":float(ap),"mean_FP_margin":fp,"mean_positive_margin_detections":count,
             "positive_patches":len(positive_ix),"negative_used":len(selected),"training_patch_accuracy":float((model.predict(allX[indices])==labels).mean()),
             "fit_seconds":fit_time,"new_hard_negatives":len(extra)}
        history.append(row);print("HARD NEGATIVE ROUND",json.dumps(row),flush=True)
        if iteration==rounds or ap>=.9:break
        if not extra:break
        selected.update(extra)
    with (out/"hard_negative_curve.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    payload={"training_images":[p.name for p,_ in chosen],"GT_birds":sum(len(gt) for _,gt in chosen),
       "raw_proposal_recall":sum(int((best_coverage(e["boxes"],e["gt"])>=.5).sum()) for e in banks)/sum(len(e["gt"]) for e in banks),
       "history":history,"gate_pass":bool(max(r["AP50_tiny_train"] for r in history)>=.9),
       "test_split_used":False,"warning":"All AP numbers here are training-set memorisation diagnostics, not val/test performance"}
    (out/"hard_negative_results.json").write_text(json.dumps(payload,indent=2))
    print("HARD NEGATIVE RESULT",json.dumps(payload),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("artifacts"));p.add_argument("--rounds",type=int,default=6)
    a=p.parse_args();run(a.dataset,a.output,a.rounds)
