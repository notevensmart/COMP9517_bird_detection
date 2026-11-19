"""E39: paper-inspired HOG vs HOG+CS-LBP, on stable GT-blind Selective Search proposals.
Based on 'Crow Birds Detection using HOG and CS-LBP' (2012).
This is an adaptation, NOT an exact paper reproduction. TRAIN/VAL only; never TEST.
"""
import argparse, json, time, csv, random
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from cloud_audit import data, iou, nms
from metrics_map import evaluate_map50, self_test as metrics_test
from quality_subset_train import training_subset
from rebuild_detector import make_jitter

SEED = 61
WIN = (64, 128)  # width, height; paper's HOG window
NMS_IOU = .35

def cslbp(g):
    """16-code center-symmetric LBP, radius 1; normalized intensity threshold 0.01."""
    g = g.astype(np.float32) / 255.
    p = np.pad(g, 1, mode="reflect")
    offsets = [(p[:-2,:-2],p[2:,2:]),
               (p[:-2,1:-1],p[2:,1:-1]),
               (p[:-2,2:],p[2:,:-2]),
               (p[1:-1,2:],p[1:-1,:-2])]
    code = np.zeros_like(g,dtype=np.uint8)
    for bit,(a,b) in enumerate(offsets):
        code |= ((a-b)>.01).astype(np.uint8) << bit
    return code

def patch_desc(image, box):
    h,w=image.shape[:2]
    x1,y1,x2,y2=np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(int)
    x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
    if x2<=x1 or y2<=y1:raise ValueError("empty crop")
    gray=cv2.cvtColor(cv2.resize(image[y1:y2,x1:x2],WIN,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    hg=hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm="L2-Hys").astype(np.float32)
    codes=cslbp(gray)
    hist=[]
    for y in range(0,WIN[1],8):
        for x in range(0,WIN[0],8):
            cnt=np.bincount(codes[y:y+8,x:x+8].ravel(),minlength=16).astype(np.float32)
            hist.append(cnt/max(1.,float(cnt.sum())))
    return np.concatenate((hg,np.concatenate(hist))).astype(np.float32)

def stable_proposals(image,cap=500):
    """Deduplicate+sort BEFORE cap, removing E36's unstable SS prefix.
    Stratify by box area; selection is deterministic and never uses GT.
    """
    if not hasattr(cv2,"ximgproc") or not hasattr(cv2.ximgproc,"segmentation"):
        raise RuntimeError("opencv-contrib-python-headless required (Selective Search)")
    seg=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    seg.setBaseImage(image);seg.switchToSelectiveSearchFast()
    rect=np.asarray(seg.process(),dtype=np.float32).reshape(-1,4)
    h,w=image.shape[:2]
    if not len(rect):return np.empty((0,4),np.float32)
    boxes=np.c_[rect[:,0],rect[:,1],rect[:,0]+rect[:,2],rect[:,1]+rect[:,3]]
    wh=boxes[:,2:]-boxes[:,:2];area=wh[:,0]*wh[:,1]
    ok=(wh[:,0]>=3)&(wh[:,1]>=3)&(area>=16)&(area<=.92*w*h)
    boxes=np.unique(boxes[ok],axis=0).astype(np.float32)
    if len(boxes)<=cap:return boxes
    area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    tiers=np.digitize(area,[1024,10000])
    rng=np.random.default_rng(SEED)
    chosen=[]
    # Equal area-tier budgets, unused slots redistributed deterministically.
    budget=cap//3
    for tier in range(3):
        ix=np.flatnonzero(tiers==tier)
        chosen.extend(rng.choice(ix,min(budget,len(ix)),replace=False).tolist())
    remaining=cap-len(chosen)
    if remaining:
        pool=np.setdiff1d(np.arange(len(boxes)),np.asarray(chosen,dtype=int))
        chosen.extend(rng.choice(pool,min(remaining,len(pool)),replace=False).tolist())
    return boxes[np.sort(chosen)]

def make_bank(samples,label,cap):
    bank=[];times=[]
    for index,(path,gt) in enumerate(samples):
        image=cv2.imread(str(path))
        if image is None:raise ValueError(str(path))
        start=time.perf_counter()
        b=stable_proposals(image,cap)
        X=np.stack([patch_desc(image,q) for q in b]) if len(b) else np.empty((0,5828),np.float32)
        times.append(time.perf_counter()-start)
        overlaps=iou(b,gt)
        best=overlaps.max(axis=0) if len(b) else np.zeros(len(gt))
        bank.append(dict(path=path,gt=gt,boxes=b,X=X,coverage=best))
        if (index+1)%5==0 or index+1==len(samples):
            print("E39 BANK",label,index+1,len(samples),"proposals",len(b),flush=True)
    return bank,float(np.mean(times))

def make_train(bank):
    rng=np.random.default_rng(47); pos=[];neg=[]
    for r in bank:
        image=cv2.imread(str(r["path"]))
        for gt in r["gt"]:
            variants=make_jitter(gt,image.shape[1],image.shape[0],rng,count=2)
            for b in variants:
                if iou(np.asarray(b).reshape(1,4),gt.reshape(1,4))[0,0]>=.55:
                    pos.append(patch_desc(image,b))
        ov=iou(r["boxes"],r["gt"])
        high=ov.max(axis=1) if len(r["gt"]) else np.zeros(len(r["boxes"]))
        # Explicitly avoid using zero-GT TRAIN images (training_subset).
        safe=np.flatnonzero(high<.05)
        if len(safe):
            ids=rng.choice(safe,min(12,len(safe)),replace=False)
            neg.extend(r["X"][ids])
    if not pos or not neg:raise RuntimeError("No training class")
    X=np.stack(pos+neg).astype(np.float32)
    y=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]
    return X,y

def fit(X,y,feature):
    # Classical linear SVM + PCA of standard-scaled descriptors.
    width=3780 if feature=="hog" else 5828
    pca=min(96,len(y)-1,width)
    model=make_pipeline(StandardScaler(),PCA(n_components=pca,svd_solver="randomized",random_state=SEED),
                        LinearSVC(C=.1,class_weight="balanced",max_iter=10000,random_state=SEED))
    model.fit(X[:,:width],y)
    return model

def evaluate(model,bank,mode,out):
    preds=[]; rows=[]; times=[]
    visual=out/"visual";visual.mkdir(parents=True,exist_ok=True)
    interesting={"beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
                 "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg",
                 "beach_bird_1606.jpg"}
    for r in bank:
        start=time.perf_counter();b=r["boxes"]
        width=3780 if mode=="hog" else 5828
        s=model.decision_function(r["X"][:,:width]) if len(b) else np.array([])
        keep=nms(b,s,iou_limit=NMS_IOU,max_det=300) if len(b) else np.empty(0,int)
        boxes=b[keep];scores=s[keep]
        times.append(time.perf_counter()-start)
        preds.append((boxes,scores,r["gt"]))
        ov=iou(boxes[:20],r["gt"]);used=set();tp=0
        for row in ov:
            if len(row)==0:continue
            order=np.argsort(-row)
            for j in order:
                if row[j]>=.5 and int(j) not in used:
                    used.add(int(j));tp+=1;break
        rows.append(dict(mode=mode,image=r["path"].name,gt=len(r["gt"]),proposals=len(b),
                         top20_tp=tp,top20_fp=min(20,len(boxes))-tp,
                         raw_proposal_r50=float(np.mean(r["coverage"]>=.5)) if len(r["gt"]) else None))
        if r["path"].name in interesting:
            im=cv2.imread(str(r["path"]))
            for q in r["gt"]:
                x1,y1,x2,y2=q.astype(int);cv2.rectangle(im,(x1,y1),(x2,y2),(0,210,0),2)
            for k,q in enumerate(boxes[:20]):
                x1,y1,x2,y2=q.astype(int)
                cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,0) if k<5 else (0,0,255),1)
            cv2.imwrite(str(visual/(mode+"_"+r["path"].stem+".jpg")),im)
    metrics=evaluate_map50(preds)
    metrics.update(top20_TP=sum(x["top20_tp"] for x in rows),top20_FP=sum(x["top20_fp"] for x in rows),
                   mean_scoring_nms_seconds=float(np.mean(times)))
    return metrics,rows

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    train=training_subset(Path(a.dataset),a.train_n)
    val=data(Path(a.dataset),"val",a.val_n)
    tb,tgen=make_bank(train,"TRAIN",a.cap)
    vb,vgen=make_bank(val,"VAL",a.cap)
    X,y=make_train(tb)
    output=dict(protocol="E39 TRAIN/VAL only; COCO 101-point AP50 maxDets100",
                source="2012 Crow Birds Detection using HOG and CS-LBP; adapted not reproduced",
                train_images=len(train),val_images=len(val),train_positive=int(y.sum()),
                train_negative=int((y==0).sum()),val_GT=sum(len(r["gt"]) for r in vb),
                train_ids=[r["path"].name for r in tb],val_ids=[r["path"].name for r in vb],
                stable_cap=a.cap,nms_iou=NMS_IOU,mean_train_bank_seconds=tgen,
                mean_val_bank_seconds=vgen,arms=[])
    all_rows=[]
    for mode in ("hog","hog_cslbp"):
        tic=time.perf_counter();model=fit(X,y,mode);training=time.perf_counter()-tic
        metrics,rows=evaluate(model,vb,mode,out)
        output["arms"].append(dict(mode=mode,training_seconds=training,patch_train_accuracy=
            float(np.mean(model.predict(X[:,:3780 if mode=="hog" else 5828])==y)),**metrics))
        all_rows+=rows
        (out/"e39_results.json").write_text(json.dumps(output,indent=2))
        print("E39 ARM RESULT",json.dumps(output["arms"][-1]),flush=True)
    with (out/"e39_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    output["delta_AP50_pp"]=100*(output["arms"][1]["map50_coco101"]-output["arms"][0]["map50_coco101"])
    (out/"e39_results.json").write_text(json.dumps(output,indent=2))
    print("E39 DONE",json.dumps(dict(delta_pp=output["delta_AP50_pp"])),flush=True)

def self_test():
    metrics_test()
    im=np.zeros((128,64,3),np.uint8);cv2.circle(im,(32,64),15,(200,200,200),-1)
    v=patch_desc(im,np.array([0,0,64,128]))
    assert v.shape==(5828,) and np.isfinite(v).all()
    assert int(cslbp(np.zeros((4,4),np.uint8)).max())<=15
    a=stable_proposals(im,50);b=stable_proposals(im,50)
    assert np.array_equal(a,b),"SS cap not deterministic"
    assert (a[:,2:]>a[:,:2]).all() if len(a) else True
    print("E39 self-test PASS: CS-LBP, HOG dimensions, deterministic proposals, metric",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",default="e39_results")
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--cap",type=int,default=500)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test()
    else:
        if not args.dataset:p.error("--dataset required")
        run(args)
