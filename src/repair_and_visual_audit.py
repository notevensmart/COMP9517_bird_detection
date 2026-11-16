"""Classical detector repair with visual verification and rigorous paired evaluation.

Controls:
* LEGACY: train proposal positives IoU >=0.5 and randomly sampled safe
  negatives, global mining, three rounds.
* REPAIRED: train true bird GT crops + IoU-controlled jitter + matched positive
  proposals, feature-exact inference, post-NMS TRAIN-only hard false alarm
  mining with per-image caps, three rounds.
Both train models use the same TRAIN/VAL images, same full-image GT-blind
LoG+Otsu proposals and HOG+LBP features, same COCO mAP@0.5 evaluator.
Only TRAIN GT is used to create positive examples or select negatives.
Outputs paired per-image FP taxonomy and annotated visual contact sheets.
"""
import argparse,csv,json,random,time
from pathlib import Path
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from label_quality import unsafe_training_negative
from cloud_audit import data,iou,nms
from classical_methods_study import blob_boxes,classification_errors
from hog_lbp_comparison import features
from metrics_map import evaluate_map50

def safe_crop(image,b):
    H,W=image.shape[:2]
    x1,y1,x2,y2=np.r_[np.floor(b[:2]),np.ceil(b[2:])].astype(int)
    x1=int(np.clip(x1,0,W));x2=int(np.clip(x2,0,W))
    y1=int(np.clip(y1,0,H));y2=int(np.clip(y2,0,H))
    if x2<=x1 or y2<=y1:return None
    return image[y1:y2,x1:x2]

def descriptor(image,b):
    crop=safe_crop(image,b)
    if crop is None:raise ValueError(f"invalid after clipping {b}")
    grey=cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2GRAY)
    z=features(grey,True)
    assert z.shape==(1924,) and np.isfinite(z).all()
    return z

def label_masks(boxes,gt):
    ov=iou(boxes,gt)
    best=ov.max(axis=1) if len(gt) else np.zeros(len(boxes))
    if len(gt) and len(boxes):
        wh=np.maximum(0,np.minimum(boxes[:,None,2:],gt[None,:,2:])-np.maximum(boxes[:,None,:2],gt[None,:,:2]))
        intersection=wh[:,:,0]*wh[:,:,1]
        boxarea=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
        gtarea=np.prod(gt[:,2:]-gt[:,:2],axis=1)
        cover=(intersection/np.maximum(np.minimum(boxarea[:,None],gtarea[None,:]),1e-9)).max(axis=1)
    else:cover=np.zeros(len(boxes))
    return best>=.5,(best<.1)&(cover<.5),best

def jittered(image,gt,rng,attempts=5):
    H,W=image.shape[:2];result=[]
    for original in gt:
        x1,y1,x2,y2=original
        cx=(x1+x2)/2;cy=(y1+y2)/2;bw=max(3,x2-x1);bh=max(3,y2-y1)
        result.append(np.array([max(0,x1),max(0,y1),min(W,x2),min(H,y2)],np.float32))
        for _ in range(attempts):
            # 5–12% centre jitter and ~10% width/height jitter preserve
            # bird appearance; reject proposals with IoU<0.55.
            nx=cx+rng.normal(0,.07*bw);ny=cy+rng.normal(0,.07*bh)
            nw=bw*np.exp(rng.normal(0,.10));nh=bh*np.exp(rng.normal(0,.10))
            b=np.array([max(0,nx-nw/2),max(0,ny-nh/2),
                        min(W,nx+nw/2),min(H,ny+nh/2)],np.float32)
            if b[2]-b[0]>=2 and b[3]-b[1]>=2 and iou(b[None,:],gt).max()>=.55:
                result.append(b)
    return result

def prepare(samples,kind,rng,max_peaks=120):
    bank=[];start=time.perf_counter()
    for k,(path,gt) in enumerate(samples):
        im=cv2.imread(str(path))
        t=time.perf_counter()
        b,_=blob_boxes(im,"otsu",max_peaks)
        proposal_time=time.perf_counter()-t
        t=time.perf_counter()
        X=np.stack([descriptor(im,v) for v in b]) if len(b) else np.empty((0,1924),np.float32)
        feature_time=time.perf_counter()-t
        pos,neg,best=label_masks(b,gt)
        item=dict(image=path.name,gt=gt,boxes=b,X=X,pos=pos,neg=neg,
                  match_iou=best,proposal_seconds=proposal_time,feature_seconds=feature_time)
        if kind=="train":
            # Extra GT positives are available only while fitting the model.
            # No validation or test GT is ever injected at inference.
            if len(gt):
                jit=jittered(im,gt,rng,5)
                item["gt_jitter_X"]=np.stack([descriptor(im,v) for v in jit])
                item["gt_jitter_count"]=len(jit)
            else:
                item["gt_jitter_X"]=np.empty((0,1924),np.float32)
                item["gt_jitter_count"]=0
        bank.append(item)
        if (k+1)%25==0 or k+1==len(samples):
            print("CACHED",kind,k+1,"/",len(samples),"seconds",round(time.perf_counter()-start,1),flush=True)
    return bank

def model_fit(pos,neg):
    X=np.concatenate([pos,neg]);y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",
                random_state=42,max_iter=10000))
    model.fit(X,y)
    return model,float((model.predict(X)==y).mean())

def mine_global(bank,model,selected,max_new=500):
    candidates=[]
    for imgindex,r in enumerate(bank):
        if not len(r["boxes"]):continue
        scores=model.decision_function(r["X"])
        for ix in np.flatnonzero(r["neg"]):
            key=(imgindex,int(ix))
            if key not in selected:candidates.append((float(scores[ix]),key))
    candidates.sort(key=lambda x:-x[0])
    return [key for _,key in candidates[:max_new]]

def mine_false_alarms(bank,model,selected,max_new=500,per_image=12):
    candidates=[]
    for imgindex,r in enumerate(bank):
        if not len(r["boxes"]):continue
        scores=model.decision_function(r["X"])
        # Mine from actual ranked full-image false alarms after NMS.
        keep=nms(r["boxes"],scores,.5,300)
        picked=0
        for ix in keep:
            key=(imgindex,int(ix))
            if key in selected or not r["neg"][ix]:continue
            # Score margin >=0: focus on actual false positives.
            if scores[ix]<0:break
            candidates.append((float(scores[ix]),key))
            picked+=1
            if picked>=per_image:break
    candidates.sort(key=lambda a:-a[0])
    return [key for _,key in candidates[:max_new]]

def evaluate(model,bank,threshold_nms=.35,visualize=False,out=None,tag=""):
    preds=[];rows=[]
    for r in bank:
        scores=model.decision_function(r["X"]) if len(r["X"]) else np.empty(0)
        order=nms(r["boxes"],scores,threshold_nms,300) if len(scores) else np.empty(0,int)
        b=r["boxes"][order];s=scores[order]
        preds.append((b,s,r["gt"]))
        active=s>=0;errs=classification_errors(b[active],s[active],r["gt"])
        rows.append(dict(image=r["image"],GT=len(r["gt"]),
             raw_proposals=len(r["boxes"]),final_boxes=int(active.sum()),**errs))
        if visualize:r["diagnostics_"+tag]=dict(boxes=b[active],scores=s[active],top_boxes=b[:50],top_scores=s[:50])
    result=evaluate_map50(preds)
    result.update(false_boxes=sum(v["FP_total"] for v in rows),
                  FP_background=sum(v["FP_background"] for v in rows),
                  FP_localization=sum(v["FP_localization"] for v in rows),
                  FP_duplicates=sum(v["FP_duplicates"] for v in rows),
                  missed_birds=sum(v["FN"] for v in rows),
                  mean_output_boxes=float(np.mean([v["final_boxes"] for v in rows])))
    return result,rows

def train(bank,variant,rounds,validation):
    rng=np.random.default_rng(42);positives=[];selected=set()
    if variant=="repaired":
        positives.extend([r["gt_jitter_X"] for r in bank if len(r["gt_jitter_X"])])
        # Also include the actual (GT-blind at inference) positive candidates
        # sampled from TRAIN detections to reduce GT->proposal distribution shift.
        for r in bank:
            ids=np.flatnonzero(r["pos"])
            if len(ids):positives.append(r["X"][rng.choice(ids,min(24,len(ids)),replace=False)])
    else:
        for r in bank:
            ids=np.flatnonzero(r["pos"])
            if len(ids):positives.append(r["X"][rng.choice(ids,min(24,len(ids)),replace=False)])
    if not positives:raise ValueError("No positive training examples")
    pos=np.concatenate(positives)
    for k,r in enumerate(bank):
        idx=np.flatnonzero(r["neg"])
        if len(idx):
            selected.update((k,int(i)) for i in rng.choice(idx,min(20,len(idx)),replace=False))
    history=[];last=None
    for rd in range(rounds+1):
        negatives=np.stack([bank[k]["X"][i] for k,i in sorted(selected)])
        t=time.perf_counter()
        model,accuracy=model_fit(pos,negatives)
        fit_time=time.perf_counter()-t
        perf,rows=evaluate(model,validation,.35)
        mined=mine_global(bank,model,selected) if variant=="legacy" else mine_false_alarms(bank,model,selected)
        result=dict(variant=variant,round=rd,positives=len(pos),negatives=len(negatives),
              train_patch_accuracy=accuracy,fit_seconds=fit_time,new_false_alarms=len(mined),**perf)
        history.append(result)
        print("REPAIR ROUND",json.dumps(result),flush=True)
        if rd<rounds:selected.update(mined)
        last=model
    return last,history

def render_panel(image,gt,proposal,model_detections,title):
    orig=image.copy()
    label=np.full((52,orig.shape[1],3),245,np.uint8)
    cv2.putText(label,title,(9,31),cv2.FONT_HERSHEY_SIMPLEX,.74,(22,22,22),2)
    for b in gt:
        x1,y1,x2,y2=np.rint(b).astype(int)
        cv2.rectangle(orig,(x1,y1),(x2,y2),(0,220,0),2)
    if proposal is not None:
        # BLUE: top 75 image-only candidate boxes; GREY: overlayed dense box locations
        for b in proposal[:75]:
            x1,y1,x2,y2=np.rint(b).astype(int)
            cv2.rectangle(orig,(x1,y1),(x2,y2),(245,160,25),1)
    if model_detections is not None:
        b,s=model_detections
        errors=iou(b,gt)
        for ix,box in enumerate(b[:65]):
            x1,y1,x2,y2=np.rint(box).astype(int)
            # Green predictions matched to some bird, red = background/near miss
            ok=(len(gt)>0 and len(errors) and errors[ix].max()>=.5)
            clr=(0,220,0) if ok else (0,0,230)
            cv2.rectangle(orig,(x1,y1),(x2,y2),clr,2)
            if ix<15:cv2.putText(orig,f"{s[ix]:.1f}",(x1,max(9,y1-3)),
                                cv2.FONT_HERSHEY_SIMPLEX,.36,clr,1)
    return np.vstack([label,orig])

def write_visuals(val,train_metrics,out,top_k=8):
    # Re-open only selected validation images for paper-quality overlays.
    order=sorted(val,key=lambda r:(-len(r["gt"]),r["image"]))
    indices=[r for r in order[:3]]+[r for r in order if len(r["gt"])==0][:2]+order[-3:]
    seen=set();visual_dir=out/"visuals";visual_dir.mkdir(exist_ok=True)
    for r in indices:
        if r["image"] in seen:continue
        seen.add(r["image"])
        path=r["path"]
        im=cv2.imread(str(path));g=r["gt"]
        baseline=r.get("diagnostics_legacy")
        repaired=r.get("diagnostics_repaired")
        # all panels same resolution
        imgs=[render_panel(im,g,None,None,"Ground truth (green)"),
              render_panel(im,g,r["boxes"],None,"LoG+Otsu candidates (blue)"),
              render_panel(im,g,None,(baseline["boxes"],baseline["scores"]) if baseline else None,"Original SVM detections"),
              render_panel(im,g,None,(repaired["boxes"],repaired["scores"]) if repaired else None,"Repaired SVM detections")]
        h,w=imgs[0].shape[:2]
        # limit display size without changing raw coordinates of drawn boxes
        scale=min(1.,600/max(w,h))
        if scale!=1:imgs=[cv2.resize(x,(int(x.shape[1]*scale),int(x.shape[0]*scale))) for x in imgs]
        montage=np.hstack(imgs)
        cv2.imwrite(str(visual_dir/(Path(r["image"]).stem+"_comparison.jpg")),montage)
    print("VISUAL EXAMPLES",len(seen),flush=True)

def main(root,out,train_limit,val_limit,rounds):
    out.mkdir(parents=True,exist_ok=True)
    tra=data(root,"train",train_limit);va=data(root,"val",val_limit)
    flagged=[p.name for p,gt in tra if unsafe_training_negative(p.name,len(gt))]
    tra=[(p,gt) for p,gt in tra if not unsafe_training_negative(p.name,len(gt))]
    print("TRAIN QA QUARANTINED",flagged,flush=True)
    rng=np.random.default_rng(123)
    train_bank=prepare(tra,"train",rng)
    val_bank=prepare(va,"val",rng)
    # track source path, for visual reading
    for entry,(path,_) in zip(val_bank,va):entry["path"]=path
    results={};per_image=[]
    for variant in ["legacy","repaired"]:
        model,history=train(train_bank,variant,rounds,val_bank)
        res,rows=evaluate(model,val_bank,.35,visualize=True,tag=variant)
        results[variant]=dict(history=history,final=res)
        per_image.extend([dict(variant=variant,**x) for x in rows])
    write_visuals(val_bank,results,out)
    with (out/"repaired_detector_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=per_image[0]);w.writeheader();w.writerows(per_image)
    (out/"repaired_detector_results.json").write_text(json.dumps(dict(
      train_images=len(tra),val_images=len(va),val_birds=sum(len(r["gt"]) for r in val_bank),
      primary_metric="official COCO 101-point bbox mAP@.5, maxDets=100",
      leakage="none: TRAIN only used for GT jitter and mining",
      warning="Experiments are exploratory matched VAL subset, not TEST",
      results=results),indent=2))
    print("REPAIR SUMMARY",json.dumps({k:v["final"] for k,v in results.items()}),flush=True)

def tests():
    x=np.zeros((50,80,3),np.uint8)
    assert safe_crop(x,np.array([-.001,10,20,30])) is not None
    assert descriptor(x,np.array([-.001,10,20,30])).shape==(1924,)
    gt=np.array([[10,10,25,25]],np.float32)
    rng=np.random.default_rng(12)
    js=jittered(x,gt,rng)
    assert js and all(iou(np.asarray(j)[None,:],gt)[0,0]>=.55 for j in js)
    ps,ns,_=label_masks(np.array([[10,10,25,25],[45,45,60,60]],np.float32),gt)
    assert ps.tolist()==[True,False] and ns.tolist()==[False,True]
    print("PASS clipped crops, jitter IoU, positive/negative labelling",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("repair_results"))
    p.add_argument("--train-limit",type=int,default=100)
    p.add_argument("--val-limit",type=int,default=30)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset is required")
        main(a.dataset,a.output,a.train_limit,a.val_limit,a.rounds)
