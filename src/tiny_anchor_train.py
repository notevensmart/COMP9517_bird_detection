"""Stage D gate: tiny-set raw-image HOG SVM training using image-only proposals.

All ground-truth access is restricted to training target assignment and scoring.
Inference generates candidates from pixels alone. Outputs metrics and overlay examples.
"""
import argparse, json, time
from pathlib import Path
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,iou,feature,nms,ap50
from anchor_proposals import generate,best_coverage

def make_bank(path,gt,per_shape,feature_limit=None):
    im=cv2.imread(str(path))
    start=time.perf_counter()
    b,raw=generate(im,"edge",per_shape)
    # Extract each proposal exactly once; inference reuses this feature bank.
    X=np.stack([feature(im,box) for box in b]) if len(b) else np.empty((0,1764),np.float32)
    return dict(image=im,boxes=b,X=X,gt=gt,seconds=time.perf_counter()-start)

def targets(entry):
    b=entry["boxes"];gt=entry["gt"]
    ov=iou(b,gt)
    best=ov.max(axis=1) if len(gt) else np.zeros(len(b))
    if len(gt) and len(b):
        wh=np.maximum(0,np.minimum(b[:,None,2:],gt[None,:,2:])-np.maximum(b[:,None,:2],gt[None,:,:2]))
        inter=wh[:,:,0]*wh[:,:,1]
        ba=np.prod(b[:,2:]-b[:,:2],axis=1)
        ga=np.prod(gt[:,2:]-gt[:,:2],axis=1)
        contain=(inter/np.maximum(np.minimum(ba[:,None],ga[None,:]),1e-9)).max(axis=1)
    else:contain=np.zeros(len(b))
    return (best>=.5),((best<.1)&(contain<.5))

def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",0)
    positives=[r for r in train if 1<=len(r[1])<=8]
    negatives=[r for r in train if len(r[1])==0]
    rng=np.random.default_rng(42)
    chosen=[positives[int(i)] for i in rng.choice(len(positives),2,replace=False)]
    if negatives:chosen.append(negatives[0])
    results=[]
    for per_shape in (450,900):
        print("BUILDING ANCHOR FEATURE BANK",per_shape,flush=True)
        bank=[make_bank(p,gt,per_shape) for p,gt in chosen]
        Xpos=[];Xneg=[];gt_total=hits=0
        for entry in bank:
            pos,neg=targets(entry)
            ix=np.flatnonzero(pos);iy=np.flatnonzero(neg)
            gt_total+=len(entry["gt"])
            hits+=int((best_coverage(entry["boxes"],entry["gt"])>=.5).sum())
            # Include varied positives; limit dominant near-duplicates to control memory.
            if len(ix):Xpos.append(entry["X"][rng.choice(ix,min(300,len(ix)),replace=False)])
            if len(iy):Xneg.append(entry["X"][rng.choice(iy,min(600,len(iy)),replace=False)])
        Xpos=np.vstack(Xpos) if Xpos else np.empty((0,1764))
        Xneg=np.vstack(Xneg) if Xneg else np.empty((0,1764))
        item=dict(config=f"edge{per_shape}",train_images=[p.name for p,_ in chosen],
                  birds=gt_total,proposal_recall=hits/gt_total if gt_total else None,
                  positive_patches=len(Xpos),negative_patches=len(Xneg),
                  mean_candidates=float(np.mean([len(x["boxes"]) for x in bank])),
                  bank_seconds=float(sum(x["seconds"] for x in bank)))
        if not len(Xpos) or not len(Xneg):
            item["gate"]="BLOCKED_MISSING_CLASS";results.append(item);continue
        X=np.vstack([Xpos,Xneg]);y=np.r_[np.ones(len(Xpos)),np.zeros(len(Xneg))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=10000,random_state=42))
        model.fit(X,y)
        item["train_patch_accuracy"]=float((model.predict(X)==y).mean())
        preds=[];pred_counts=[]
        for j,entry in enumerate(bank):
            b=entry["boxes"];s=model.decision_function(entry["X"])
            kept=nms(b,s,iou_limit=.3,max_det=300)
            pred=(b[kept],s[kept],entry["gt"])
            preds.append(pred)
            pred_counts.append(int((s[kept]>=0).sum()))
            if j<2:
                canvas=entry["image"].copy()
                for bb in entry["gt"].astype(int):
                    x1,y1,x2,y2=bb;cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
                for k in kept:
                    if s[k]<0:continue
                    x1,y1,x2,y2=b[k].astype(int)
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.imwrite(str(out/f"tiny_edge{per_shape}_image{j}.png"),canvas)
        item["train_raw_AP50"]=ap50(preds)
        item["mean_positive_margin_detections"]=float(np.mean(pred_counts))
        item["gate"]="PASS" if item["train_raw_AP50"]>=.9 else "FAIL_TINY_OVERFIT"
        results.append(item)
        print("TINY ANCHOR RESULT",json.dumps(item),flush=True)
    payload={"results":results,"ready_for_full_training":any(x["gate"]=="PASS" for x in results),
             "note":"Tiny training-set AP is a memorisation check, NOT generalisation and NOT validation AP."}
    (out/"tiny_anchor_results.json").write_text(json.dumps(payload,indent=2))
    print("TINY ANCHOR SUMMARY",json.dumps(payload),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True,type=Path);p.add_argument("--output",type=Path,default=Path("artifacts"))
    a=p.parse_args();run(a.dataset,a.output)
