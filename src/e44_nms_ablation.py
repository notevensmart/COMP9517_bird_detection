"""E44: frozen E42 checkpoint, paired NMS ablation on full VAL; never accesses TEST."""
import argparse,csv,json,joblib,os
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data,iou,nms
from e42_full_rbf_augmented import prep
from nonlinear_appearance_study import rbf_logits
from metrics_map import evaluate_map50
from rebuild_detector import draw_box

def soft_nms(boxes,scores,method,sigma=.5,threshold=.35,limit=300,min_score=0.00001):
    # Operates on copies. Decay in the original score domain; Gaussian always decays.
    scores=1/(1+np.exp(-np.clip(np.asarray(scores,dtype=float),-60,60)))
    remaining=np.arange(len(scores))
    kept=[]
    while len(remaining) and len(kept)<limit:
        best=int(np.argmax(scores[remaining]))
        idx=int(remaining[best]);remaining=np.delete(remaining,best)
        if scores[idx]<min_score:break
        kept.append(idx)
        if len(remaining):
            overlaps=iou(boxes[idx:idx+1],boxes[remaining])[0]
            if method=="linear":
                scores[remaining]*=np.where(overlaps>threshold,1-overlaps,1)
            elif method=="gaussian":
                scores[remaining]*=np.exp(-(overlaps**2)/sigma)
            else:raise ValueError(method)
    return np.asarray(kept,dtype=int),scores

def apply_nms(boxes,scores,mode):
    if mode=="hard035":return nms(boxes,scores,.35,300),scores
    if mode=="hard050":return nms(boxes,scores,.50,300),scores
    if mode=="linear":return soft_nms(boxes,scores,"linear")
    if mode=="gaussian":return soft_nms(boxes,scores,"gaussian")
    raise ValueError(mode)

def run(root,checkpoint,out):
    out.mkdir(parents=True,exist_ok=True)
    models=joblib.load(checkpoint)
    stage1=models["stage1"];model=models["augmented_model"]
    variants=("hard035","hard050","linear","gaussian")
    predictions={k:[] for k in variants}
    rows=[];summary={};visual=out/"visual";visual.mkdir(exist_ok=True)
    val=data(root,"val",0)
    for index,(path,gt) in enumerate(val,1):
        rec,im=prep(path,gt,stage1);boxes=rec["boxes"]
        scores=.5*rec["margin"]+.5*rbf_logits(model,rec) if len(boxes) else np.empty(0)
        coverage=float((iou(boxes,gt).max(axis=0)>=.5).sum()/len(gt)) if len(boxes) and len(gt) else 0.
        for mode in variants:
            ids,adjusted=apply_nms(boxes,scores,mode)
            pred=boxes[ids];conf=adjusted[ids]
            predictions[mode].append((pred,conf,gt))
            overlaps=iou(pred,gt) if len(pred) and len(gt) else np.empty((len(pred),len(gt)))
            used=set();tp=0
            for j in range(min(20,len(pred))):
                for g in np.argsort(-overlaps[j]):
                    if overlaps[j,g]>=.5 and int(g) not in used:
                        used.add(int(g));tp+=1;break
            rows.append(dict(image=path.name,mode=mode,gt=len(gt),proposals=len(boxes),proposal_recall50=coverage,top20_tp=tp,top20_fp=min(20,len(pred))-tp,top20_recall=tp/max(1,len(gt)),kept=len(ids)))
            if mode in ("hard035","gaussian") and (index<=12 or coverage<.5):
                canvas=im.copy()
                for box in gt:draw_box(canvas,box,(0,255,0),2)
                for j,box in enumerate(pred[:20]):
                    good=len(gt)>0 and len(overlaps)>j and overlaps[j].max(initial=0)>=.5
                    draw_box(canvas,box,(255,0,0) if good else (0,0,255),2)
                cv2.imwrite(str(visual/(path.stem+"_"+mode+".jpg")),canvas)
        if index%20==0:print("E44 VAL",index,len(val),flush=True)
    for mode in variants:
        m=evaluate_map50(predictions[mode])
        rr=[r for r in rows if r["mode"]==mode]
        m.update(top20_tp=sum(r["top20_tp"] for r in rr),top20_fp=sum(r["top20_fp"] for r in rr))
        summary[mode]=m
        print("E44 RESULT",mode,json.dumps(m),flush=True)
    with (out/"per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    baseline={r["image"]:r for r in rows if r["mode"]=="hard035"}
    failure=sorted(baseline.values(),key=lambda r:(r["proposal_recall50"],r["top20_recall"],-r["gt"]))
    (out/"worst_images.json").write_text(json.dumps(failure[:40],indent=2))
    (out/"summary.json").write_text(json.dumps(dict(validation_images=len(val),nms=summary,baseline="hard035",note="Full VAL only; same frozen E42 augmented checkpoint; no TEST"),indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("e44_results"))
    a=ap.parse_args();run(a.dataset,a.checkpoint,a.output)
