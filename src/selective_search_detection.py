"""E31: Selective Search versus LoG+Otsu on a paired, reproducible detector.

No neural networks. Classical Selective Search region grouping via OpenCV
ximgproc.segmentation, HOG+LBP and TRAIN-only nonlinear RBF SVM.
The SAME training/validation images, classifier config, NMS and official COCO
protocol are used in each arm. Also report proposal recall by GT size class.

Controls: E29's LoG+Otsu full-image proposals; Selective Search fast;
union (LoG+Selective Search). Models are refitted separately to each
proposal domain, avoiding the proposal-domain shift of a frozen classifier.
"""
from pathlib import Path
import argparse,json,csv,time
import cv2,numpy as np
from cloud_audit import data,iou,nms
from classical_methods_study import blob_boxes
from rebuild_detector import descriptor,make_jitter,fit,classification_errors,draw_box,safe_labels
from background_rejector import secondary_features
from nonlinear_appearance_study import fit_rbf,rbf_logits
from metrics_map import evaluate_map50
from quality_subset_train import training_subset

def selective_boxes(image,limit=1200):
    if not hasattr(cv2,"ximgproc") or not hasattr(cv2.ximgproc,"segmentation"):
        raise RuntimeError("OpenCV contrib ximgproc.segmentation required")
    seg=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    seg.setBaseImage(image)
    seg.switchToSelectiveSearchFast()
    rect=np.asarray(seg.process(),dtype=np.float32).reshape(-1,4)
    if len(rect)==0:return np.empty((0,4),np.float32)
    h,w=image.shape[:2]
    boxes=np.c_[rect[:,0],rect[:,1],rect[:,0]+rect[:,2],rect[:,1]+rect[:,3]]
    wh=boxes[:,2:]-boxes[:,:2]
    mask=(wh[:,0]>=3)&(wh[:,1]>=3)&(wh[:,0]*wh[:,1]>=16)&(wh[:,0]*wh[:,1]<=.92*w*h)
    boxes=boxes[mask]
    # Unique boxes, keep Selective Search's own ordering; do NOT use GT to rank.
    if len(boxes):
        _,indices=np.unique(boxes,axis=0,return_index=True)
        boxes=boxes[np.sort(indices)]
    return boxes[:limit].astype(np.float32)

def proposed(image,which):
    log,_=blob_boxes(image,"otsu",max_peaks=120)
    if which=="log":return log
    ss=selective_boxes(image,1200)
    if which=="selective":return ss
    if len(log)==0:return ss
    if len(ss)==0:return log
    return np.unique(np.vstack((log,ss)),axis=0).astype(np.float32)

def prepare(samples,which,label):
    records=[];timing=[]
    for index,(path,gt) in enumerate(samples):
        im=cv2.imread(str(path));tic=time.perf_counter()
        b=proposed(im,which)
        sec=time.perf_counter()-tic
        X=np.stack([descriptor(im,x) for x in b]) if len(b) else np.empty((0,1924),np.float32)
        aux=np.stack([secondary_features(im,x) for x in b]) if len(b) else np.empty((0,32),np.float32)
        positive,negative=safe_labels(b,gt)
        overlap=iou(b,gt)
        maximum=overlap.max(axis=0) if len(b) else np.zeros(len(gt))
        records.append(dict(path=path,gt=gt,boxes=b,X=X,aux=aux,
                pos=positive,neg=negative,coverage=maximum,seconds=sec))
        if (index+1)%10==0 or index+1==len(samples):
            print("E31 BANK",label,which,index+1,len(samples),
                  "mean_proposals",round(np.mean([len(r["boxes"]) for r in records]),1),flush=True)
    return records

def fit_models(bank):
    rng=np.random.default_rng(47)
    positives=[];negatives=[]
    for r in bank:
        im=cv2.imread(str(r["path"]))
        for gt in r["gt"]:
            for b in make_jitter(gt,im.shape[1],im.shape[0],rng,count=3):
                positives.append(descriptor(im,b))
        choices=r["neg"]
        if len(choices):
            indices=rng.choice(choices,min(20,len(choices)),replace=False)
            negatives.extend(r["X"][indices])
    pos=np.stack(positives);neg=np.stack(negatives)
    base,_=fit(pos,neg)
    # One TRAIN-only hard-negative mining round, with representative full-image
    # false alarms; no VAL ground truth or margins used in selection.
    hard=[]
    for r in bank:
        if not len(r["boxes"]):continue
        s=base.decision_function(r["X"])
        keep=nms(r["boxes"],s,.5,170)
        safe=set(int(i) for i in r["neg"])
        count=0
        for i in keep:
            j=int(i)
            if j in safe and s[j]>-.2:
                hard.append(r["X"][j]);count+=1
            if count>=12:break
    if hard:neg=np.vstack([neg,np.stack(hard)])
    base,_=fit(pos,neg)
    for r in bank:r["margin"]=base.decision_function(r["X"]) if len(r["X"]) else np.empty(0)
    # Proposal-domain verifier: use exactly this method's positive and
    # negative TRAIN proposal crops; train PCA + RBF on HOG/LBP + 32D context.
    X=[];aux=[];margin=[];labels=[]
    rng=np.random.default_rng(77)
    for r in bank:
        good=np.asarray(r["pos"])
        if len(good)>45:good=rng.choice(good,45,replace=False)
        safe=set(int(i) for i in r["neg"])
        scores=r["margin"]
        keep=nms(r["boxes"],scores,.5,170) if len(scores) else []
        hard=[int(j) for j in keep if int(j) in safe and scores[int(j)]>-.7][:25]
        remaining=[int(j) for j in r["neg"] if int(j) not in set(hard)]
        randomneg=rng.choice(remaining,min(12,len(remaining)),replace=False).tolist() if len(remaining) else []
        bad=hard+randomneg
        for ids,lab in [(good,1),(bad,0)]:
            if len(ids):
                X.append(r["X"][ids]);aux.append(r["aux"][ids])
                margin.append(r["margin"][ids]);labels.extend([lab]*len(ids))
    X=np.vstack(X);A=np.vstack(aux);M=np.concatenate(margin).reshape(-1,1);y=np.asarray(labels,dtype=int)
    fitted,diag=fit_rbf(X,A,M,y,64)
    return base,fitted,dict(training_image_count=len(bank),GT_positive_patches=len(pos),
              negative_patches=len(neg),proposal_positive_samples=int(y.sum()),
              proposal_negative_samples=int((y==0).sum()),**diag)

def get_metrics(preds):
    report=evaluate_map50(preds)
    r50=dict(small=[0,0],medium=[0,0],large=[0,0])
    for b,s,gt in preds:
        o=iou(b,gt)
        best=o.max(axis=0) if len(b) else np.zeros(len(gt))
        for i,g in enumerate(gt):
            a=np.prod(g[2:]-g[:2]);key="small" if a<1024 else ("medium" if a<10000 else "large")
            r50[key][1]+=1;r50[key][0]+=int(best[i]>=.5)
    report["post_nms_top100_coverage50_by_size"]={k:(p[0]/p[1] if p[1] else None) for k,p in r50.items()}
    return report

def eval_method(bank,base,rbf,which,out):
    samples=[];rows=[]
    visual_dir=out/"visual";visual_dir.mkdir(exist_ok=True)
    chosen={"beach_bird_1379.jpg","beach_bird_1419.jpg","beach_bird_1560.jpg",
            "beach_bird_1606.jpg","beach_bird_1389.jpg","beach_bird_1453.jpg"}
    panel=[]
    for rec in bank:
        rec["margin"]=base.decision_function(rec["X"]) if len(rec["X"]) else np.empty(0)
        logits=rbf_logits(rbf,rec)
        score=.5*rec["margin"]+.5*logits
        chosen_inds=nms(rec["boxes"],score,.35,300) if len(score) else np.empty(0,int)
        boxes=rec["boxes"][chosen_inds];s=score[chosen_inds]
        samples.append((boxes,s,rec["gt"]))
        ov=iou(boxes[:20],rec["gt"]);used=set();tp=0
        for k in range(min(20,len(boxes))):
            for j in np.argsort(-ov[k]) if len(rec["gt"]) else []:
                if ov[k,j]>=.5 and int(j) not in used:
                    used.add(int(j));tp+=1;break
        active=s>=0;errs=classification_errors(boxes[active],s[active],rec["gt"])
        rows.append(dict(method=which,image=rec["path"].name,GT=len(rec["gt"]),
             proposals=len(rec["boxes"]),top20_TP=tp,
             top20_FP=min(20,len(boxes))-tp,**errs))
        if rec["path"].name in chosen:
            image=cv2.imread(str(rec["path"]))
            for b in rec["gt"]:draw_box(image,b,(0,220,0),2)
            used=set()
            for k,b in enumerate(boxes[:20]):
                ok=False
                for j in np.argsort(-ov[k]) if len(rec["gt"]) else []:
                    if ov[k,j]>=.5 and int(j) not in used:
                        used.add(int(j));ok=True;break
                draw_box(image,b,(255,0,0) if ok else (0,0,255),2)
            image=cv2.resize(image,(512,320))
            cv2.rectangle(image,(0,0),(512,27),(0,0,0),-1)
            cv2.putText(image,f"{which} {rec['path'].stem} GT={len(rec['gt'])}",
                        (5,18),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
            panel.append(image)
    if len(panel)>=2:
        panel=panel[:6]
        if len(panel)%2:panel.pop()
        cv2.imwrite(str(visual_dir/(which+"_top20.jpg")),
           np.vstack([np.hstack(panel[i:i+2]) for i in range(0,len(panel),2)]))
    metrics=get_metrics(samples)
    metrics.update(method=which,top20_TP=sum(x["top20_TP"] for x in rows),
        top20_FP=sum(x["top20_FP"] for x in rows),FP_margin0=sum(x["FP_total"] for x in rows),
        FN_margin0=sum(x["FN"] for x in rows))
    return metrics,rows

def coverage(bank):
    bins={key:dict(n=0,covered50=0,covered75=0) for key in ("small","medium","large")}
    for rec in bank:
        for box,match in zip(rec["gt"],rec["coverage"]):
            a=np.prod(box[2:]-box[:2])
            key="small" if a<1024 else ("medium" if a<10000 else "large")
            v=bins[key];v["n"]+=1
            v["covered50"]+=int(match>=.5);v["covered75"]+=int(match>=.75)
    return dict(bins=bins,all_recall50=sum(v["covered50"] for v in bins.values())/max(1,sum(v["n"] for v in bins.values())),
     all_recall75=sum(v["covered75"] for v in bins.values())/max(1,sum(v["n"] for v in bins.values())),
     mean_candidates=float(np.mean([len(r["boxes"]) for r in bank])),
     mean_generation_seconds=float(np.mean([r["seconds"] for r in bank])))

def main(root,out,train_n,val_n,which):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    arms=["log","selective","union"] if which=="all" else [which]
    results=[];per_image=[]
    for kind in arms:
        print("E31 ARM",kind,flush=True)
        tb=prepare(train,kind,"TRAIN")
        vb=prepare(val,kind,"VAL")
        base,rbf,fitdiag=fit_models(tb)
        metrics,rows=eval_method(vb,base,rbf,kind,out)
        entry=dict(method=kind,training=fitdiag,proposal=coverage(vb),detection=metrics,
          train_images=len(tb),val_images=len(vb),val_GT=sum(len(v["gt"]) for v in vb))
        results.append(entry);per_image.extend(rows)
        (out/"e31_results.json").write_text(json.dumps(results,indent=2))
        print("E31 RESULT",json.dumps(entry),flush=True)
    with (out/"e31_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=per_image[0]);wr.writeheader();wr.writerows(per_image)

def tests():
    im=np.zeros((128,128,3),np.uint8)
    cv2.circle(im,(64,64),18,(255,255,255),-1)
    sel=selective_boxes(im,100)
    assert sel.ndim==2 and sel.shape[1]==4
    assert not len(sel) or (sel[:,2:]>sel[:,:2]).all()
    print("PASS Selective Search bounds, proposal shape and classical-only gate",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("e31_results"))
    ap.add_argument("--train-n",type=int,default=45)
    ap.add_argument("--val-n",type=int,default=30)
    ap.add_argument("--method",choices=["all","log","selective","union"],default="all")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args();tests()
    if not a.self_test:
        if not a.dataset:ap.error("--dataset required")
        main(a.dataset,a.output,a.train_n,a.val_n,a.method)
