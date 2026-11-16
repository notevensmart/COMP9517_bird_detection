"""Researcher-style diagnosis of classical bird detection failure.

Oracle tests are DIAGNOSTIC ONLY, not valid deployed bird detection.
1. Image/YOLO annotation sanity checks and pixel aspect statistics.
2. LoG+Otsu proposal recall / raw candidate count / scale and crowding.
3. Oracle-box classifier test: train on train-GT boxes, evaluate on val-GT
   boxes to isolate descriptor generalisation. Negative features from TRAIN
   image-only proposals. Never use validation for model fit.
4. Perfect-GT prediction sanity: COCO mAP50 should be 1 except possible
   crowded top-100 limit; no model and no detector is used in this oracle.
5. Oracle-proposal ranked boxes: score 1.0 for best proposal per GT plus
   very low scores for remaining proposals; compare no NMS / NMS=.35/.65
   and COCO maxDets=100. WARNING oracle uses VAL GT for scoring.
6. Save GT/candidate visual overlays for selected images (paper diagnostics).
"""
from pathlib import Path
import argparse,json,csv,time
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,iou,nms
from classical_methods_study import blob_boxes
from metrics_map import evaluate_map50
from hog_lbp_comparison import features

def descriptor(image,b,letterbox=False):
    xy=np.r_[np.floor(b[:2]),np.ceil(b[2:])].astype(int)
    crop=image[xy[1]:xy[3],xy[0]:xy[2]]
    if not crop.size:raise ValueError(f"Zero patch for {b}")
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    if letterbox:
        hh,ww=gray.shape
        scale=min(64/ww,64/hh)
        target=cv2.resize(gray,(max(1,round(ww*scale)),max(1,round(hh*scale))))
        result=np.full((64,64),int(np.median(gray)),np.uint8)
        x=(64-target.shape[1])//2;y=(64-target.shape[0])//2
        result[y:y+target.shape[0],x:x+target.shape[1]]=target
        gray=result
    else:gray=cv2.resize(gray,(64,64))
    return features(gray,True)

def oracle_preds(boxes,gt):
    """Best matching single proposal per GT; soft score prevents mis-ranking
    background. This uses GT and is *never* an inference method."""
    ov=iou(boxes,gt)
    s=np.full(len(boxes),-1000.,dtype=np.float64)
    if len(ov):
        for k in range(len(gt)):
            j=int(np.argmax(ov[:,k]))
            if ov[j,k]>=.5:s[j]=max(s[j],10+float(ov[j,k]))
    return s

def candidate_coverage(b,gt):
    mat=iou(b,gt)
    best=mat.max(axis=0) if len(b) else np.zeros(len(gt))
    return dict(recall50=int((best>=.5).sum()),recall75=int((best>=.75).sum()),
                gt=len(gt),candidates=len(b),gt_maxiou=best.tolist())

def evaluate_oracles(val):
    report=[];cases={}
    for path,gt in val:
        img=cv2.imread(str(path));b,_=blob_boxes(img,"otsu",120)
        cov=candidate_coverage(b,gt)
        scores=oracle_preds(b,gt) if len(b) else np.empty(0,float)
        cases[path.name]=dict(image=img,gt=gt,boxes=b,scores=scores)
        h,w=img.shape[:2]
        areas=np.prod(gt[:,2:]-gt[:,:2],axis=1) if len(gt) else np.empty(0)
        ar=(gt[:,2]-gt[:,0])/np.maximum(gt[:,3]-gt[:,1],1e-6) if len(gt) else np.empty(0)
        report.append(dict(image=path.name,width=w,height=h,
                           crowd_gt=len(gt),small_gt=int((areas<1024).sum()),
                           gt_aspect_gt_2=int((ar>2).sum()),
                           gt_aspect_lt_point5=int((ar<.5).sum()),
                           **{k:v for k,v in cov.items() if k!="gt_maxiou"}))
    def make_preds(style):
        entries=[]
        for c in cases.values():
            if style=="perfect_gt":
                b=c["gt"].copy()
                # Unique scores 1-ish to keep valid perfect detections ordered
                s=np.linspace(1,0.9,len(b)) if len(b) else np.empty(0)
            else:
                b=c["boxes"];s=c["scores"]
                if style.startswith("nms"):
                    thr=float(style[3:])
                    idx=nms(b,s,iou_limit=thr,max_det=300) if len(b) else np.empty(0,int)
                    b=b[idx];s=s[idx]
            entries.append((b,s,c["gt"]))
        return evaluate_map50(entries)
    oracle_results={key:make_preds(key) for key in ["perfect_gt","raw_proposals","nms0.35","nms0.65"]}
    total=sum(x["gt"] for x in report)
    cov=sum(x["recall50"] for x in report)/total if total else None
    summary=dict(n_val=len(val),birds=total,mean_candidates=float(np.mean([x["candidates"] for x in report])),
        proposal_recall50=cov,proposal_recall75=sum(x["recall75"] for x in report)/total if total else None,
        birds_gt_100=sum(x["gt"] for x in report if x["gt"]>100),
        images_gt_100=sum(x["gt"]>100 for x in report),
        small_gt=sum(x["small_gt"] for x in report),
        wide_gt=sum(x["gt_aspect_gt_2"] for x in report),
        tall_gt=sum(x["gt_aspect_lt_point5"] for x in report),
        oracle_metrics=oracle_results)
    return summary,report,cases

def gt_oracle_classifier(root,train_limit,val,cases):
    train=data(root,"train",train_limit)
    rng=np.random.default_rng(42)
    positives=[];negatives=[]
    for path,gt in train:
        im=cv2.imread(str(path))
        if len(gt):
            order=rng.choice(len(gt),min(12,len(gt)),replace=False)
            positives.extend((im,gt[k]) for k in order)
        b,_=blob_boxes(im,"otsu",60)
        if not len(b):continue
        ov=iou(b,gt)
        maximum=ov.max(axis=1) if len(gt) else np.zeros(len(b))
        ix=np.flatnonzero(maximum<.05)
        if len(ix):
            for k in rng.choice(ix,min(14,len(ix)),replace=False):negatives.append((im,b[k]))
    results=[]
    if not positives or not negatives:return dict(status="NO_TRAIN_CLASSES")
    print("ORACLE TRAIN feature counts",len(positives),len(negatives),flush=True)
    for letterbox in (False,True):
        start=time.perf_counter()
        X=np.stack([descriptor(im,b,letterbox) for im,b in positives+negatives])
        y=np.r_[np.ones(len(positives)),np.zeros(len(negatives))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",random_state=42,max_iter=10000))
        model.fit(X,y)
        feats=[];scores=[]
        for entry in cases.values():
            if not len(entry["gt"]):continue
            z=np.stack([descriptor(entry["image"],b,letterbox) for b in entry["gt"]])
            feats.append(z)
            scores.extend(model.decision_function(z).tolist())
        scores=np.asarray(scores)
        results.append(dict(letterbox=letterbox,train_GT_positive=len(positives),train_random_safe_negative=len(negatives),
           val_GT_bird_count=len(scores),val_true_birds_margin_ge_0=int((scores>=0).sum()),
           val_true_bird_positive_rate=float((scores>=0).mean()),
           val_true_bird_median_score=float(np.median(scores)),
           train_patch_accuracy=float((model.predict(X)==y).mean()),
           seconds=time.perf_counter()-start))
        print("GT ORACLE CLASSIFIER",json.dumps(results[-1]),flush=True)
    return results

def render(cases,out):
    order=sorted(cases.items(),key=lambda p:-len(p[1]["gt"]))
    choose=order[:2]+order[-2:]+[p for p in order if 5<=len(p[1]["gt"])<=15][:2]
    unique=set()
    for image_name,item in choose:
        if image_name in unique:continue
        unique.add(image_name)
        img=item["image"].copy()
        b=item["boxes"];gt=item["gt"]
        mat=iou(b,gt)
        matched=mat.max(axis=1)>=.5 if len(gt) else np.zeros(len(b),bool)
        # Draw at most 120 proposals, and all GT in separate images.
        for k in np.flatnonzero(matched)[:100]:
            x1,y1,x2,y2=np.rint(b[k]).astype(int)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
        for bb in gt:
            x1,y1,x2,y2=np.rint(bb).astype(int)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imwrite(str(out/(Path(image_name).stem+"_gt_green_candidates_blue.jpg")),img)

def run(root,out,train_limit,val_limit):
    out.mkdir(parents=True,exist_ok=True)
    val=data(root,"val",val_limit)
    summary,details,cases=evaluate_oracles(val)
    print("ORACLE GEOMETRY SUMMARY",json.dumps(summary),flush=True)
    classified=gt_oracle_classifier(root,train_limit,val,cases)
    render(cases,out)
    payload=dict(geometry=summary,classifier_gt_oracle=classified,
                 warning="All GT-derived scores and GT-box classification are diagnostic oracles, NOT deployment mAP.",
                 notes=["GT oracle mAP can be reduced by per-image maxDets=100 in very crowded scenes.",
                        "Test GT remains untouched.","Scores for GT box classifier are measured on exact VAL GT regions, a diagnostic upper-bound on recognition only."])
    (out/"audit_oracles.json").write_text(json.dumps(payload,indent=2))
    with (out/"audit_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=details[0].keys());writer.writeheader();writer.writerows(details)
    print("AUDIT COMPLETE",json.dumps(payload),flush=True)

def tests():
    x=np.array([[5,6,25,30]],np.float32)
    z=candidate_coverage(x,x)
    assert z["recall50"]==1
    pic=np.zeros((48,64,3),np.uint8)
    for mode in (False,True):
        h=descriptor(pic,np.array([2,3,34,25]),mode)
        assert h.shape==(1924,) and np.all(np.isfinite(h))
    print("PASS annotation and oracle descriptor tests",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("diagnostic_results"))
    ap.add_argument("--train-limit",type=int,default=100)
    ap.add_argument("--val-limit",type=int,default=60)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args();tests()
    if not args.self_test:
        if not args.dataset:ap.error("--dataset required")
        run(args.dataset,args.output,args.train_limit,args.val_limit)
