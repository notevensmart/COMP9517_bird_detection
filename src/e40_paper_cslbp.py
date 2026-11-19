"""E40: mathematics-checked 2012 HOG + CS-LBP bird detector.
Paper-derived descriptor: 48x64 pixel window, 64 blocks, 4 HOG cells/block,
9 unsigned orientation bins, HOG=2304D; 16-bin CS-LBP/block, t=0.02,
CS-LBP=1024D, L2-Hys per block. Combined =3328D.
Paper-inspired linear SVM, one TRAIN-only hard negative mining iteration.
IMPORTANT: This GPU-free diagnostic uses capped multiscale sliding windows,
not the paper's unrestricted 1.12x/4px scan. It is NOT an exact reproduction.
All model selection/evaluation uses TRAIN/VAL only; official TEST untouched.
"""
import argparse, csv, json, time
from pathlib import Path
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from cloud_audit import data, iou, nms
from metrics_map import evaluate_map50, self_test as metric_tests
from quality_subset_train import training_subset
from rebuild_detector import make_jitter

W,H = 48,64
BLOCK_W,BLOCK_H = 6,8  # 8x8 blocks across 48x64, each with 2x2 cells
DIM_HOG,DIM_CSLBP = 2304,1024
SEED = 61
SCALES = (.35,.5,.75,1.,1.5,2.,3.,4.)
NMS = .35

def l2hys(blocks):
    """Per-block L2 norm, clamp histogram values to 0.2, renormalize L2."""
    x=np.asarray(blocks,np.float32)
    x=x/np.sqrt(np.sum(x*x,axis=1,keepdims=True)+1e-8)
    x=np.minimum(x,.2)
    return x/np.sqrt(np.sum(x*x,axis=1,keepdims=True)+1e-8)

def cslbp_codes(gray):
    """Original CS-LBP P=8,R=1: opposite neighbors at 0,45,90,135deg.
    Grayscale in [0,1]; threshold >=0.02. Bit order clockwise starting NW.
    """
    a=np.asarray(gray,np.float32)/255.
    p=np.pad(a,1,mode="reflect")
    opposite=((p[:-2,:-2],p[2:,2:]),
              (p[:-2,1:-1],p[2:,1:-1]),
              (p[:-2,2:],p[2:,:-2]),
              (p[1:-1,2:],p[1:-1,:-2]))
    codes=np.zeros(a.shape,np.uint8)
    for bit,(u,v) in enumerate(opposite):
        codes|=((u-v)>=.02).astype(np.uint8)<<bit
    return codes

def descriptor_gray(gray):
    if gray.shape!=(H,W) or gray.dtype!=np.uint8:
        raise ValueError("Expected uint8 grayscale window 64 high x 48 wide")
    # Paper's [-1,0,1] masks, unsigned orientation 0..180, nine 20-degree bins.
    gx=cv2.filter2D(gray.astype(np.float32),-1,np.array([[-1,0,1]],np.float32),
                    borderType=cv2.BORDER_REFLECT_101)
    gy=cv2.filter2D(gray.astype(np.float32),-1,np.array([[-1],[0],[1]],np.float32),
                    borderType=cv2.BORDER_REFLECT_101)
    magnitude=cv2.magnitude(gx,gy)
    angle=np.mod(np.degrees(np.arctan2(gy,gx)),180.)
    orientation=np.minimum((angle/20.).astype(np.int32),8)
    y,x=np.indices((H,W))
    block=(y//BLOCK_H)*8+(x//BLOCK_W)
    cell=((y%BLOCK_H)//4)*2+((x%BLOCK_W)//3)
    indices=(block*36+cell*9+orientation).ravel()
    hog_counts=np.bincount(indices,weights=magnitude.ravel(),minlength=DIM_HOG)
    hg=l2hys(hog_counts.reshape(64,36)).ravel()
    code=cslbp_codes(gray)
    inds=(block*16+code.astype(np.int32)).ravel()
    cs_counts=np.bincount(inds,minlength=DIM_CSLBP)
    cs=l2hys(cs_counts.reshape(64,16)).ravel()
    return np.concatenate((hg,cs)).astype(np.float32)

def descriptor(image,b):
    h,w=image.shape[:2]
    x1,y1,x2,y2=np.r_[np.floor(b[:2]),np.ceil(b[2:])].astype(int)
    x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
    if x2<=x1 or y2<=y1:raise ValueError("Empty image crop")
    g=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    return descriptor_gray(cv2.resize(g,(W,H),interpolation=cv2.INTER_AREA))

def grid_axis(limit,length,stride):
    last=max(0,limit-length)
    values=list(range(0,last+1,stride))
    if not values or values[-1]!=last:values.append(last)
    return values

def sliding_windows(image,cap=900,scales=SCALES,stride_fraction=.33):
    """Explicit resource-constrained scale-stratified sliding-window proposal bank.
    Input only, no labels, deterministic. Cap assigned per scale, not ranked by GT.
    A faithful FULL paper scan can be explored separately at 1.12x / stride4.
    """
    ih,iw=image.shape[:2]; selected=[]; counts={}
    quotas=[cap//len(scales)]*len(scales)
    for j in range(cap%len(scales)):quotas[j]+=1
    for k,scale in enumerate(scales):
        bw=int(round(W*scale));bh=int(round(H*scale))
        if bw<6 or bh<8 or bw>iw or bh>ih:
            counts[str(scale)]=0;continue
        step=max(4,int(round(min(bw,bh)*stride_fraction)))
        boxes=np.asarray([(x,y,x+bw,y+bh)
            for y in grid_axis(ih,bh,step)
            for x in grid_axis(iw,bw,step)],np.float32).reshape(-1,4)
        counts[str(scale)]=len(boxes)
        quota=quotas[k]
        if len(boxes)>quota:
            # Stable coverage across scan ordering, never annotation-dependent.
            ids=np.linspace(0,len(boxes)-1,quota,dtype=np.int64)
            boxes=boxes[ids]
        selected.extend(boxes)
    return np.asarray(selected,np.float32).reshape(-1,4),counts

def bank(samples,label,cap):
    records=[];times=[]
    for ix,(path,gt) in enumerate(samples):
        im=cv2.imread(str(path))
        if im is None:raise FileNotFoundError(str(path))
        start=time.perf_counter()
        b,unbounded=sliding_windows(im,cap)
        X=np.stack([descriptor(im,q) for q in b]) if len(b) else np.empty((0,3328),np.float32)
        times.append(time.perf_counter()-start)
        if not np.isfinite(X).all():raise ValueError("Nonfinite feature")
        ov=iou(b,gt)
        best=ov.max(axis=0) if len(b) else np.zeros(len(gt))
        # Reject negatives overlapping a large fraction of ANY GT, even at small IoU.
        if len(b) and len(gt):
            diff=np.maximum(0,np.minimum(b[:,None,2:],gt[None,:,2:])-np.maximum(b[:,None,:2],gt[None,:,:2]))
            inter=diff[:,:,0]*diff[:,:,1]
            area_b=np.prod(b[:,2:]-b[:,:2],axis=1)
            area_gt=np.prod(gt[:,2:]-gt[:,:2],axis=1)
            overlap_min=(inter/np.maximum(np.minimum(area_b[:,None],area_gt[None,:]),1e-9)).max(axis=1)
            best_per_candidate=ov.max(axis=1)
        else:
            overlap_min=np.zeros(len(b));best_per_candidate=np.zeros(len(b))
        safe=np.flatnonzero((best_per_candidate<.1)&(overlap_min<.5))
        records.append(dict(path=path,gt=gt,boxes=b,X=X,coverage=best,safe=safe,
                            uncapped_count=sum(unbounded.values())))
        if (ix+1)%5==0 or ix+1==len(samples):
            print("E40 BANK",label,ix+1,"/",len(samples),"candidate",len(b),flush=True)
    return records,float(np.mean(times))

def training_features(train_bank):
    rng=np.random.default_rng(47)
    positives=[];negatives=[]
    for r in train_bank:
        im=cv2.imread(str(r["path"]))
        for gt in r["gt"]:
            for b in make_jitter(gt,im.shape[1],im.shape[0],rng,count=2):
                if iou(np.asarray(b).reshape(1,4),gt.reshape(1,4))[0,0]>=.55:
                    positives.append(descriptor(im,b))
        safe=r["safe"]
        if len(safe):
            chosen=rng.choice(safe,min(22,len(safe)),replace=False)
            negatives.extend(r["X"][chosen])
    if not positives or not negatives:raise RuntimeError("Missing training class")
    return np.stack(positives),np.stack(negatives)

def model_fit(P,N,which):
    d=DIM_HOG if which=="hog" else DIM_HOG+DIM_CSLBP
    X=np.vstack((P[:,:d],N[:,:d]));y=np.r_[np.ones(len(P),int),np.zeros(len(N),int)]
    # No PCA or standardization: paper uses block-normalized descriptors + linear SVM.
    model=LinearSVC(C=1.,max_iter=10000,random_state=SEED)
    model.fit(X,y)
    return model,float((model.predict(X)==y).mean())

def hard_negative_mining(model,train_bank,which,limit=220):
    d=DIM_HOG if which=="hog" else DIM_HOG+DIM_CSLBP
    ranked=[]
    for index,r in enumerate(train_bank):
        if not len(r["safe"]):continue
        safe=r["safe"]
        score=model.decision_function(r["X"][safe,:d])
        top=np.argsort(-score)[:8]
        for j in top:
            if score[j]>-.3:
                ranked.append((float(score[j]),index,int(safe[j])))
    ranked.sort(key=lambda v:(-v[0],v[1],v[2]))
    selected=ranked[:limit]
    return np.stack([train_bank[i]["X"][j] for _,i,j in selected]) if selected else np.empty((0,3328),np.float32)

def proposal_report(records):
    result={}
    for key in ("all","small","medium","large"):
        values=[]
        for r in records:
            for box,hit in zip(r["gt"],r["coverage"]):
                area=np.prod(box[2:]-box[:2])
                name="small" if area<1024 else ("medium" if area<10000 else "large")
                if key=="all" or key==name:values.append(float(hit))
        result[key]=dict(n=len(values),R50=float(np.mean(np.asarray(values)>=.5)) if values else None,
                         R75=float(np.mean(np.asarray(values)>=.75)) if values else None)
    result["mean_candidates"]=float(np.mean([len(r["boxes"]) for r in records]))
    result["mean_uncapped_candidates"]=float(np.mean([r["uncapped_count"] for r in records]))
    return result

def evaluate(model,records,which,out):
    d=DIM_HOG if which=="hog" else DIM_HOG+DIM_CSLBP
    predictions=[];rows=[];secs=[]
    vis=out/"visual";vis.mkdir(parents=True,exist_ok=True)
    selected={"beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
              "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg",
              "beach_bird_1606.jpg"}
    for r in records:
        t=time.perf_counter()
        b=r["boxes"]
        s=model.decision_function(r["X"][:,:d]) if len(b) else np.empty(0)
        inds=nms(b,s,NMS,300) if len(b) else np.empty(0,dtype=int)
        b=b[inds];s=s[inds]
        secs.append(time.perf_counter()-t)
        predictions.append((b,s,r["gt"]))
        hits=iou(b[:20],r["gt"]);used=set();tp=0;matched=[]
        for row in hits:
            ok=False
            for j in np.argsort(-row):
                if row[j]>=.5 and int(j) not in used:
                    used.add(int(j));tp+=1;ok=True;break
            matched.append(ok)
        rows.append(dict(mode=which,image=r["path"].name,gt_count=len(r["gt"]),
                         candidates=len(r["boxes"]),top20_tp=tp,top20_fp=len(matched)-tp))
        if r["path"].name in selected:
            im=cv2.imread(str(r["path"]))
            for gt in r["gt"]:
                x1,y1,x2,y2=gt.astype(int);cv2.rectangle(im,(x1,y1),(x2,y2),(0,225,0),2)
            for k,box in enumerate(b[:20]):
                x1,y1,x2,y2=box.astype(int)
                cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,0) if matched[k] else (0,0,255),2)
            cv2.imwrite(str(vis/(which+"_"+r["path"].stem+".jpg")),im)
    metrics=evaluate_map50(predictions)
    metrics.update(top20_TP=sum(x["top20_tp"] for x in rows),
                   top20_FP=sum(x["top20_fp"] for x in rows),
                   mean_score_nms_seconds=float(np.mean(secs)))
    return metrics,rows

def validate_math():
    metric_tests()
    g=np.zeros((H,W),np.uint8)
    v=descriptor_gray(g)
    assert v.shape==(3328,) and np.isfinite(v).all()
    assert np.all(v[:DIM_HOG]==0), "Constant image should have zero gradients"
    assert int(cslbp_codes(g).max())==0
    # CS-LBP threshold is on [0,1] grayscale: 5/255<.02, 6/255>=.02
    c=np.zeros((H,W),np.uint8)
    c[9,9]=5
    assert (cslbp_codes(c)[10,10]&1)==0
    c[9,9]=6
    assert (cslbp_codes(c)[10,10]&1)==1
    c[11,11]=255
    assert (cslbp_codes(c)[10,10]&1)==0
    # Horizontal intensity ramp -> unsigned HOG is in bin zero (interior).
    ramp=np.broadcast_to(np.arange(W,dtype=np.uint8)[None,:]*4,(H,W)).copy()
    h=descriptor_gray(ramp)[:DIM_HOG].reshape(64,4,9)
    assert h[:,:,0].sum()>0 and h[:,:,1:].sum()<1e-4
    hys=l2hys(np.array([[3.,4.]],np.float32))
    assert np.isclose(np.linalg.norm(hys),1.,atol=1e-5)
    im=np.zeros((180,180,3),np.uint8)
    b1,_=sliding_windows(im,cap=300);b2,_=sliding_windows(im,cap=300)
    assert np.array_equal(b1,b2)
    assert len(b1)<=300 and np.all(b1[:,2:]>b1[:,:2])
    print("E40 PASS: CS-LBP exact threshold/pair, HOG orientation/dimensions, L2-Hys, stable boxes, COCO metrics",flush=True)

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=training_subset(args.dataset,args.train_n)
    val=data(args.dataset,"val",args.val_n)
    tb,tbank=bank(train,"TRAIN",args.cap)
    vb,vbank=bank(val,"VAL",args.cap)
    p,n=training_features(tb)
    report=dict(experiment="E40",source="2012 Crow Birds Detection using HOG and CS-LBP",
                reproduction_level="Paper descriptor dimensions and threshold; capped multiscale scan and dataset adapted",
                train_images=len(tb),val_images=len(vb),val_GT=int(sum(len(r["gt"]) for r in vb)),
                train_ids=[r["path"].name for r in tb],val_ids=[r["path"].name for r in vb],
                positive_samples=len(p),initial_negative_samples=len(n),scan_cap=args.cap,
                feature_dim_hog=DIM_HOG,feature_dim_cs_lbp=DIM_CSLBP,
                mean_train_bank_seconds=tbank,mean_val_bank_seconds=vbank,
                proposal=proposal_report(vb),arms=[])
    rows=[]
    for which in ("hog","hog_cslbp"):
        start=time.perf_counter()
        m,initial_acc=model_fit(p,n,which)
        extra=hard_negative_mining(m,tb,which)
        if len(extra):m,acc=model_fit(p,np.vstack((n,extra)),which)
        else:acc=initial_acc
        train_seconds=time.perf_counter()-start
        metrics,row=evaluate(m,vb,which,out)
        report["arms"].append(dict(mode=which,initial_train_accuracy=initial_acc,
                                   final_train_accuracy=acc,mined_train_negatives=len(extra),
                                   fit_seconds=train_seconds,**metrics))
        rows.extend(row)
        (out/"e40_results.json").write_text(json.dumps(report,indent=2))
        print("E40 RESULT",json.dumps(report["arms"][-1]),flush=True)
    report["delta_AP50_pp"]=100*(report["arms"][1]["map50_coco101"]-report["arms"][0]["map50_coco101"])
    (out/"e40_results.json").write_text(json.dumps(report,indent=2))
    with (out/"e40_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("E40 COMPLETE",json.dumps(dict(delta_AP50_pp=report["delta_AP50_pp"],proposal=report["proposal"])),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--output",default="e40_results")
    parser.add_argument("--train-n",type=int,default=45)
    parser.add_argument("--val-n",type=int,default=30)
    parser.add_argument("--cap",type=int,default=900)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:validate_math()
    else:
        if args.dataset is None:parser.error("--dataset required")
        main(args)
