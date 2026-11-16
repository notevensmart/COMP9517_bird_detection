"""CPU quick audit. Uses TRAIN for fitting and VAL for exploratory metrics; never touches TEST."""
import argparse, csv, json, random, time
from pathlib import Path
import cv2, numpy as np
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def boxes(path,w,h):
    out=[]
    if not path.exists(): raise FileNotFoundError(str(path))
    for i,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():continue
        v=np.fromstring(line,sep=" ")
        if len(v)!=5 or not np.all(np.isfinite(v)) or v[0]!=0 or not(0<=v[1]<=1 and 0<=v[2]<=1 and 0<v[3]<=1 and 0<v[4]<=1):
            raise ValueError(f"bad label {path}:{i}")
        _,cx,cy,bw,bh=v
        b=np.array([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2])*[w,h,w,h]
        if np.min(b)<-1e-3 or b[2]>w+1e-3 or b[3]>h+1e-3:raise ValueError(f"out of bounds {path}:{i}")
        # YOLO normalized labels near image boundaries can round to
        # -0.000256px. Clip small floating-point drift *before* crop flooring.
        # Never silently accept genuinely out-of-bounds annotations.
        b=np.clip(b,[0,0,0,0],[w,h,w,h])
        if b[2]<=b[0] or b[3]<=b[1]:raise ValueError(f"degenerate clipped label {path}:{i}")
        out.append(b)
    return np.asarray(out,dtype=np.float32).reshape(-1,4)

def data(root,split,limit):
    result=[]
    for p in sorted((root/"images"/split).iterdir()):
        if p.suffix.lower() not in (".jpg",".jpeg",".png"):continue
        image=cv2.imread(str(p))
        if image is None:raise ValueError(f"image unreadable {p}")
        h,w=image.shape[:2]
        result.append((p,boxes(root/"labels"/split/(p.stem+".txt"),w,h)))
    if not result:raise ValueError(f"no images in {root}/images/{split}")
    rng=random.Random(42 if split=="train" else 43)
    if limit and limit<len(result):result=sorted(rng.sample(result,limit),key=lambda x:x[0].name)
    return result

def iou(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh[:,:,0]*wh[:,:,1]
    aa=np.prod(a[:,2:]-a[:,:2],axis=1);bb=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)

def proposals(image,pad=.1,cap=200):
    h,w=image.shape[:2];g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    g=cv2.createCLAHE(2.,(8,8)).apply(g)
    g=cv2.GaussianBlur(g,(3,3),0)
    e=cv2.Canny(g,50,140)
    e=cv2.morphologyEx(e,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(e,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    all_boxes=set()
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if bw<2 or bh<2 or bw*bh<8 or bw*bh>.7*w*h or cv2.contourArea(c)<8:continue
        b=(max(0,int(np.floor(x-pad*bw))),max(0,int(np.floor(y-pad*bh))),min(w,int(np.ceil(x+bw+pad*bw))),min(h,int(np.ceil(y+bh+pad*bh))))
        if b[2]>b[0] and b[3]>b[1]:all_boxes.add(b)
    arr=np.asarray(sorted(all_boxes,key=lambda b:((b[2]-b[0])*(b[3]-b[1]),b)),dtype=np.float32).reshape(-1,4)
    if cap and len(arr)>cap:arr=arr[np.linspace(0,len(arr)-1,cap,dtype=int)]
    return arr,e

def feature(image,b):
    x1,y1,x2,y2=np.r_[np.floor(b[:2]),np.ceil(b[2:])].astype(int)
    patch=image[y1:y2,x1:x2]
    if patch.size==0:raise ValueError("empty crop")
    gray=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
    return hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm="L2-Hys").astype(np.float32)

def nms(b,s,iou_limit=.3,max_det=300):
    inds=np.argsort(-s,kind="stable");keep=[]
    while len(inds) and len(keep)<max_det:
        k=int(inds[0]);keep.append(k);inds=inds[1:]
        if len(inds):inds=inds[iou(b[k:k+1],b[inds])[0]<=iou_limit]
    return np.asarray(keep,dtype=int)

def ap50(preds):
    events=[];gt_total=0
    for b,s,gt in preds:
        gt_total+=len(gt);taken=np.zeros(len(gt),bool)
        for k in np.argsort(-s,kind="stable"):
            ov=iou(b[k:k+1],gt)[0]
            if len(ov):ov[taken]=-1
            j=int(ov.argmax()) if len(ov) else -1
            ok=j>=0 and ov[j]>=.5
            if ok:taken[j]=True
            events.append((float(s[k]),int(ok)))
    if not gt_total:return float("nan")
    events.sort(key=lambda a:-a[0])
    if not events:return 0.
    tp=np.cumsum([t for _,t in events])
    prec=tp/np.arange(1,len(tp)+1);rec=tp/gt_total
    mr=np.r_[0,rec,1];mp=np.r_[0,prec,0]
    mp=np.maximum.accumulate(mp[::-1])[::-1]
    changes=np.where(mr[1:]!=mr[:-1])[0]
    return float(np.sum((mr[changes+1]-mr[changes])*mp[changes+1]))

def audit(root,output,train_limit,val_limit):
    output.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",train_limit);val=data(root,"val",val_limit)
    rows=[];study=[]
    for name,pad,cap in [("canny_cap200",.1,200),("canny_uncapped",.1,None),("canny_pad25",.25,None)]:
        hits=hits75=total=nr=0
        for p,gt in val:
            im=cv2.imread(str(p));b,_=proposals(im,pad,cap)
            ov=iou(b,gt);best=ov.max(axis=0) if len(b) else np.zeros(len(gt))
            hits+=int((best>=.5).sum());hits75+=int((best>=.75).sum());total+=len(gt);nr+=len(b)
            study.append(dict(method=name,image=p.name,gt=len(gt),proposals=len(b),covered50=int((best>=.5).sum())))
        rows.append(dict(method=name,images=len(val),birds=total,proposal_R50=hits/total if total else None,proposal_R75=hits75/total if total else None,proposals_per_image=nr/len(val)))
    print("PROPOSAL RESULTS",rows,flush=True)
    for filename,items in [("proposal_summary.csv",rows),("proposal_by_image.csv",study)]:
        with (output/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=items[0].keys());w.writeheader();w.writerows(items)
    # The frozen quick-mode Canny HOG SVM baseline, fitted on train only
    positives=[];negatives=[];rng=np.random.default_rng(42)
    for p,gt in train:
        im=cv2.imread(str(p));b,_=proposals(im,.1,200);ov=iou(b,gt)
        best=ov.max(axis=1) if len(gt) else np.zeros(len(b))
        for j in np.flatnonzero(best>=.5):positives.append(feature(im,b[j]))
        bad=np.flatnonzero(best<.1)
        for j in rng.choice(bad,min(20,len(bad)),replace=False):negatives.append(feature(im,b[j]))
    counts=dict(positive=len(positives),negative=len(negatives))
    if not positives or not negatives:
        (output/"gate.json").write_text(json.dumps({"status":"BLOCKED","reason":"No trainable positive/negative proposal patches","counts":counts,"proposal_results":rows},indent=2))
        raise RuntimeError("TRAINING BLOCKED: no positive/negative proposals, inspect coverage first")
    X=np.asarray(positives[:600]+negatives[:1200]);y=np.r_[np.ones(min(len(positives),600)),np.zeros(min(len(negatives),1200))]
    model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=10000,random_state=42))
    start=time.perf_counter();model.fit(X,y);training_seconds=time.perf_counter()-start
    evaluated=[];score_rows=[]
    for p,gt in val:
        im=cv2.imread(str(p));start=time.perf_counter();b,_=proposals(im,.1,200)
        s=model.decision_function(np.stack([feature(im,box) for box in b])) if len(b) else np.empty(0)
        keep=nms(b,s);evaluated.append((b[keep],s[keep],gt))
        score_rows.append(dict(image=p.name,ground_truth=len(gt),detections=len(keep),seconds=time.perf_counter()-start))
    ap=ap50(evaluated)
    result=dict(status="QUICK_BASELINE_COMPLETE",train_images=len(train),val_images=len(val),train_patches=counts,training_seconds=training_seconds,AP50_continuous=ap,mean_inference_seconds=float(np.mean([x["seconds"] for x in score_rows])),caveat="Exploratory subset only; original notebook feature sampling not identical; not a full validation or test result")
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    (output/"gate.json").write_text(json.dumps({"status":"PROPOSAL_GATE_FAIL" if rows[0]["proposal_R50"]<.5 else "PROPOSAL_GATE_PASS","baseline":result,"proposal_results":rows},indent=2))
    print("BASELINE",result,flush=True)

def self_test():
    g=np.array([[0,0,10,10]],float)
    assert np.isclose(iou(g,g)[0,0],1)
    assert np.isclose(ap50([(g,np.array([.9]),g)]),1)
    assert np.isclose(ap50([(g+50,np.array([.9]),g)]),0)
    assert len(nms(np.vstack([g,g]),np.array([.9,.8])))==1
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"border.txt"
        p.write_text("0 0.16494975 0.43620600 0.32990000 0.25000000\\n".replace("\\\\n","\\n"))
        clipped=boxes(p,512,512)
        assert np.min(clipped)>=0 and len(clipped)==1
    print("Synthetic IoU, AP50, NMS and YOLO border-clipping checks PASS")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path);ap.add_argument("--output",type=Path,default=Path("artifacts"));ap.add_argument("--train-limit",type=int,default=80);ap.add_argument("--val-limit",type=int,default=20);ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args();self_test()
    if not args.self_test:
        if not args.dataset:ap.error("--dataset required")
        audit(args.dataset,args.output,args.train_limit,args.val_limit)
