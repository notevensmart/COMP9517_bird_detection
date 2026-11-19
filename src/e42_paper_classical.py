"""E42: THREE distinct classical bird detectors derived from primary papers.
Methods are computationally bounded adaptations, NOT exact reproductions of original systems.
DPM: Felzenszwalb et al. TPAMI 2010 -- HOG root + deformable HOG parts,
     latent part locations, alternating latent SVM and hard negative mining.
Fisher: Perronnin et al. ECCV 2010 -- dense SIFT, diagonal GMM Fisher
     first/second order gradients, signed sqrt and L2, linear SVM.
ACF: Dollar et al. TPAMI 2014 -- LUV, normalized magnitude, six oriented
     gradient channels, 4x4 aggregation, depth-2 boosted trees.
All methods use exactly same GT-blind deterministic Selective Search proposals.
TRAIN and VAL ONLY. Official TEST is never loaded. Benchmarks are independent
COCO 101-point single-class bbox AP50 maxDets100 with top20 and error overlays.
"""
import argparse, csv, json, time
from pathlib import Path
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from cloud_audit import data, iou, nms
from metrics_map import evaluate_map50, self_test as metric_self_test
from quality_subset_train import training_subset
from rebuild_detector import make_jitter, safe_labels

SEED=61
WINDOW=64
NMS_IOU=.35
SIFT=cv2.SIFT_create(nfeatures=0)
HOG=cv2.HOGDescriptor()

def crop(image,b):
    h,w=image.shape[:2]
    x1,y1,x2,y2=np.r_[np.floor(b[:2]),np.ceil(b[2:])].astype(int)
    x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
    if x2<=x1 or y2<=y1:raise ValueError("Empty crop")
    return cv2.resize(image[y1:y2,x1:x2],(WINDOW,WINDOW),interpolation=cv2.INTER_AREA)

def stable_ss(image,cap=250):
    if not hasattr(cv2,'ximgproc') or not hasattr(cv2.ximgproc,'segmentation'):
        raise RuntimeError('opencv-contrib-python-headless required')
    seg=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    seg.setBaseImage(image);seg.switchToSelectiveSearchFast()
    rect=np.asarray(seg.process(),dtype=np.float32).reshape(-1,4)
    h,w=image.shape[:2]
    if not len(rect):return np.empty((0,4),np.float32)
    b=np.c_[rect[:,0],rect[:,1],rect[:,0]+rect[:,2],rect[:,1]+rect[:,3]]
    wh=b[:,2:]-b[:,:2];a=wh[:,0]*wh[:,1]
    b=np.unique(b[(wh[:,0]>=3)&(wh[:,1]>=3)&(a>=16)&(a<=.92*w*h)],axis=0).astype(np.float32)
    if len(b)<=cap:return b
    tiers=np.digitize(np.prod(b[:,2:]-b[:,:2],axis=1),[1024,10000])
    rng=np.random.default_rng(SEED)
    ids=[]
    for t in range(3):
        subset=np.flatnonzero(tiers==t)
        ids.extend(rng.choice(subset,min(len(subset),cap//3),replace=False).tolist())
    remainder=cap-len(ids)
    if remainder:
        pool=np.setdiff1d(np.arange(len(b)),np.asarray(ids,dtype=int))
        ids.extend(rng.choice(pool,remainder,replace=False).tolist())
    return b[np.sort(ids)]

def hog_vec(g,size,cell=8):
    h,w=size
    return cv2.HOGDescriptor((w,h),(2*cell,2*cell),(cell,cell),(cell,cell),9).compute(
        cv2.resize(g,(w,h),interpolation=cv2.INTER_AREA)).ravel().astype(np.float32)

# DPM part search: four anchors, 3x3 latent displacements, HOG root+part.
# Deformation f = [-dx^2,-dx,-dy^2,-dy] as in star-structured DPM.
ANCHORS=((8,8),(40,8),(8,40),(40,40))
OFFSETS=((-4,-4),(-4,0),(-4,4),(0,-4),(0,0),(0,4),(4,-4),(4,0),(4,4))
DROOT=hog_vec(np.zeros((64,64),np.uint8),(64,64)).size
DPART=hog_vec(np.zeros((24,24),np.uint8),(24,24),cell=4).size
PART_DIM=DPART+4
DTOTAL=DROOT+4*PART_DIM

def dpm_candidates(patch):
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    root=hog_vec(gray,(64,64))
    partsets=[]
    for ax,ay in ANCHORS:
        options=[]
        for dx,dy in OFFSETS:
            x=np.clip(ax+dx,0,40);y=np.clip(ay+dy,0,40)
            ph=hog_vec(gray[y:y+24,x:x+24],(24,24),cell=4)
            deform=np.array([-(dx/4.)**2,-dx/4.,-(dy/4.)**2,-dy/4.],np.float32)
            options.append(np.r_[ph,deform])
        partsets.append(np.asarray(options,np.float32))
    return root,partsets

def dpm_latent(example,w=None):
    root,parts=example
    vector=[root]
    for j,choices in enumerate(parts):
        if w is None:
            chosen=4
        else:
            weights=w[DROOT+j*PART_DIM:DROOT+(j+1)*PART_DIM]
            chosen=int(np.argmax(choices@weights))
        vector.append(choices[chosen])
    return np.concatenate(vector).astype(np.float32)

def fisher_local(patch):
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    keys=[cv2.KeyPoint(float(x),float(y),12.) for y in (10,24,39,53) for x in (10,24,39,53)]
    kp,des=SIFT.compute(gray,keys)
    if des is None:return np.zeros((0,128),np.float32)
    return np.asarray(des,np.float32)/512.

def fisher_encode(local,gmm):
    K,D=gmm.means_.shape
    if not len(local):return np.zeros(2*K*D,np.float32)
    resp=gmm.predict_proba(local)
    diff=(local[:,None,:]-gmm.means_[None,:,:])/np.sqrt(gmm.covariances_[None,:,:]+1e-8)
    N=float(len(local)); weight=np.sqrt(gmm.weights_+1e-8)
    first=(resp[:,:,None]*diff).sum(axis=0)/(N*weight[:,None])
    second=(resp[:,:,None]*(diff*diff-1.)).sum(axis=0)/(N*np.sqrt(2.)*weight[:,None])
    vec=np.r_[first.ravel(),second.ravel()].astype(np.float32)
    vec=np.sign(vec)*np.sqrt(np.abs(vec)) # Perronnin power alpha=.5
    vec/=max(1e-9,float(np.linalg.norm(vec))) # Improved Fisher L2 normalization
    return vec

def acf_features(patch):
    """10 channel ACF: LUV(3)+normalized grad mag(1)+six oriented gradients.
    Exact aggregation over 4x4 pixel cells followed by smoothing.
    """
    sm=cv2.GaussianBlur(patch,(3,3),0.8)
    luv=cv2.cvtColor(sm,cv2.COLOR_BGR2Luv).astype(np.float32)/255.
    gray=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=1)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=1)
    mag=np.sqrt(gx*gx+gy*gy)
    norm=mag/(cv2.GaussianBlur(mag,(5,5),0)+.01)
    angle=np.mod(np.arctan2(gy,gx),np.pi)
    bins=np.minimum((angle/(np.pi/6)).astype(int),5)
    channels=[luv[:,:,i] for i in range(3)]+[norm]
    channels += [(mag*(bins==i)).astype(np.float32) for i in range(6)]
    aggregated=[]
    for ch in channels:
        agg=ch.reshape(16,4,16,4).sum(axis=(1,3))
        agg=cv2.GaussianBlur(agg,(3,3),.8)
        aggregated.append(agg.ravel())
    return np.concatenate(aggregated).astype(np.float32)

def extract(patch,method):
    if method=="dpm":return dpm_candidates(patch)
    if method=="fisher":return fisher_local(patch)
    if method=="acf":return acf_features(patch)
    raise ValueError(method)

def prepare(samples,method,cap,phase):
    out=[];sec=[]
    for i,(path,gt) in enumerate(samples):
        image=cv2.imread(str(path))
        if image is None:raise FileNotFoundError(path)
        tic=time.perf_counter();boxes=stable_ss(image,cap)
        records=[extract(crop(image,b),method) for b in boxes]
        sec.append(time.perf_counter()-tic)
        _,neg=safe_labels(boxes,gt)
        ov=iou(boxes,gt)
        cover=ov.max(axis=0) if len(boxes) else np.zeros(len(gt))
        out.append(dict(path=path,gt=gt,boxes=boxes,features=records,
                        safe=neg,cover=cover))
        if (i+1)%5==0 or i+1==len(samples):
            print("E42 BANK",method,phase,i+1,"/",len(samples),flush=True)
    return out,float(np.mean(sec))

def train_examples(bank,method):
    rng=np.random.default_rng(47)
    positives=[];negatives=[]
    for r in bank:
        im=cv2.imread(str(r["path"]))
        for gt in r["gt"]:
            for b in make_jitter(gt,im.shape[1],im.shape[0],rng,count=2):
                if iou(np.asarray(b).reshape(1,4),gt.reshape(1,4))[0,0]>=.55:
                    positives.append(extract(crop(im,b),method))
        neg=r["safe"]
        if len(neg):
            ids=rng.choice(neg,min(12,len(neg)),replace=False)
            negatives.extend(r["features"][int(j)] for j in ids)
    if not positives or not negatives:raise ValueError("Missing training classes")
    return positives,negatives

def train_dpm(pos,neg,train_bank):
    # Latent SVM alternating latent part positions + linear-margin training.
    # Single mixture component and four 24x24 parts; no full HOG-pyramid convolution.
    w=None;rounds=[]
    for iteration in range(3):
        X=np.stack([dpm_latent(x,w) for x in pos+neg])
        y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        clf=LinearSVC(C=.05,max_iter=6000,random_state=SEED)
        clf.fit(X,y);w=clf.coef_.ravel()
        rounds.append(dict(iteration=iteration,accuracy=float(np.mean(clf.predict(X)==y))))
        if iteration==0:
            mined=[]
            for r in train_bank:
                if not len(r["safe"]):continue
                ids=np.asarray(r["safe"],int)
                f=np.stack([dpm_latent(r["features"][j],w) for j in ids])
                scores=clf.decision_function(f)
                picked=ids[np.argsort(-scores)[:5]]
                mined.extend(r["features"][j] for j in picked)
            neg+=mined
    return (clf,w),dict(latent_rounds=rounds,mined_negatives=len(neg))

def train_fisher(pos,neg):
    rng=np.random.default_rng(SEED)
    all_local=np.vstack([x for x in pos+neg if len(x)])
    if len(all_local)>12000:
        all_local=all_local[rng.choice(len(all_local),12000,replace=False)]
    gmm=GaussianMixture(n_components=8,covariance_type="diag",reg_covar=1e-4,
                        max_iter=80,random_state=SEED)
    gmm.fit(all_local)
    X=np.stack([fisher_encode(x,gmm) for x in pos+neg])
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    clf=LinearSVC(C=1.,class_weight="balanced",max_iter=10000,random_state=SEED)
    clf.fit(X,y)
    return (clf,gmm),dict(gmm_components=8,local_descriptor_dimension=128,
                          fisher_dimension=X.shape[1],train_accuracy=float(np.mean(clf.predict(X)==y)),
                          gmm_converged=bool(gmm.converged_))

def train_acf(pos,neg):
    X=np.stack(pos+neg)
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    clf=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,
              min_samples_leaf=5,random_state=SEED),n_estimators=160,
              learning_rate=.6,random_state=SEED)
    clf.fit(X,y)
    return clf,dict(n_estimators=len(clf.estimators_),feature_dimension=X.shape[1],
                    tree_depth=2,train_accuracy=float(np.mean(clf.predict(X)==y)))

def score(model,features,method):
    if not features:return np.empty(0)
    if method=="dpm":
        clf,w=model
        X=np.stack([dpm_latent(f,w) for f in features])
        return clf.decision_function(X)
    if method=="fisher":
        clf,gmm=model
        X=np.stack([fisher_encode(f,gmm) for f in features])
        return clf.decision_function(X)
    X=np.stack(features)
    return model.decision_function(X)

def diagnostics(records):
    result={}
    for size in ("all","small","medium","large"):
        overlaps=[]
        for r in records:
            for b,q in zip(r["gt"],r["cover"]):
                area=np.prod(b[2:]-b[:2])
                label="small" if area<1024 else ("medium" if area<10000 else "large")
                if size=="all" or size==label:overlaps.append(q)
        result[size]=dict(n=len(overlaps),R50=float(np.mean(np.asarray(overlaps)>=.5)) if overlaps else None,
                         R75=float(np.mean(np.asarray(overlaps)>=.75)) if overlaps else None)
    result["avg_proposals"]=float(np.mean([len(r["boxes"]) for r in records]))
    return result

def evaluate(model,bank,method,out):
    predictions=[];rows=[];times=[]
    visual=out/"visual";visual.mkdir(parents=True,exist_ok=True)
    chosen={"beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
            "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg","beach_bird_1606.jpg"}
    for r in bank:
        t=time.perf_counter()
        s=np.asarray(score(model,r["features"],method),np.float32)
        boxes=r["boxes"]
        keep=nms(boxes,s,NMS_IOU,300) if len(boxes) else np.empty(0,int)
        boxes=boxes[keep];s=s[keep]
        times.append(time.perf_counter()-t)
        predictions.append((boxes,s,r["gt"]))
        overlap=iou(boxes[:20],r["gt"]);used=set();tp=0;correct=[]
        for row in overlap:
            match=False
            for j in np.argsort(-row):
                if row[j]>=.5 and int(j) not in used:
                    used.add(int(j));tp+=1;match=True;break
            correct.append(match)
        rows.append(dict(method=method,image=r["path"].name,gt=len(r["gt"]),
                         candidates=len(r["boxes"]),tp_top20=tp,fp_top20=len(correct)-tp))
        if r["path"].name in chosen:
            image=cv2.imread(str(r["path"]))
            for b in r["gt"]:
                x1,y1,x2,y2=b.astype(int);cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,0),2)
            for index,b in enumerate(boxes[:20]):
                x1,y1,x2,y2=b.astype(int)
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0) if correct[index] else (0,0,255),2)
            cv2.imwrite(str(visual/(method+"_"+r["path"].stem+".jpg")),image)
    metrics=evaluate_map50(predictions)
    metrics.update(tp_top20=sum(x["tp_top20"] for x in rows),
                   fp_top20=sum(x["fp_top20"] for x in rows),
                   avg_score_nms_seconds=float(np.mean(times)))
    return metrics,rows

def self_test(method):
    metric_self_test()
    image=np.zeros((128,128,3),np.uint8)
    cv2.rectangle(image,(30,30),(90,90),(180,200,50),-1)
    patch=crop(image,np.array([20,20,100,100]))
    a=extract(patch,method)
    if method=="dpm":
        assert len(a[0])==DROOT and len(a[1])==4
        assert all(v.shape==(9,PART_DIM) for v in a[1])
        assert dpm_latent(a).shape==(DTOTAL,)
        assert np.all(a[1][0][:,DPART]<=0)
    if method=="fisher":
        assert a.ndim==2 and a.shape[1]==128
        local=np.vstack([a,a+1e-3,np.ones_like(a)*.1])
        gm=GaussianMixture(n_components=2,covariance_type="diag",reg_covar=.001,random_state=SEED).fit(local)
        z=fisher_encode(a,gm)
        assert z.shape==(512,) and np.isfinite(z).all() and np.isclose(np.linalg.norm(z),1,atol=.01)
    if method=="acf":
        assert a.shape==(2560,) and np.isfinite(a).all()
    b=stable_ss(image,30);c=stable_ss(image,30)
    assert np.array_equal(b,c)
    print("E42 SELF TEST PASS",method,"feature shapes, GMM/part/channel math, deterministic proposals",flush=True)

def run(args):
    method=args.method
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=training_subset(args.dataset,args.train_n)
    val=data(args.dataset,"val",args.val_n)
    tbank,tsec=prepare(train,method,args.cap,"TRAIN")
    vbank,vsec=prepare(val,method,args.cap,"VAL")
    pos,neg=train_examples(tbank,method)
    tic=time.perf_counter()
    if method=="dpm":model,details=train_dpm(pos,neg,tbank)
    elif method=="fisher":model,details=train_fisher(pos,neg)
    else:model,details=train_acf(pos,neg)
    fit_seconds=time.perf_counter()-tic
    metrics,rows=evaluate(model,vbank,method,out)
    report=dict(experiment="E42",method=method,source_paper={
        "dpm":"Felzenszwalb et al. Object Detection with Discriminatively Trained Part-Based Models, TPAMI 2010",
        "fisher":"Perronnin et al. Improving the Fisher Kernel for Large-Scale Image Classification, ECCV 2010",
        "acf":"Dollar et al. Fast Feature Pyramids for Object Detection, TPAMI 2014"}[method],
        implementation_class="Computationally bounded paper-derived adaptation, not exact reproduction",
        adjustments={"dpm":"single component, 4 deformable 24x24 parts; GT patches and fixed SS region windows, no full feature pyramid",
                     "fisher":"8 component diagonal GMM, 4x4 dense SIFT grid, fixed SS proposals; Fisher paper originally classification not detection",
                     "acf":"10 paper-defined channels, block aggregation, 160 depth-2 SAMME trees rather than 2048; fixed SS boxes rather than full channel pyramid"}[method],
        train_images=len(train),val_images=len(val),val_gt=sum(len(r["gt"]) for r in vbank),
        train_positive=len(pos),train_initial_negative=len(neg),
        train_ids=[x[0].name for x in train],val_ids=[x[0].name for x in val],
        proposal="E37-inspired order-stable GT-blind capped Selective Search",
        cap=args.cap,nms=.35,training_feature_seconds_per_image=tsec,
        validation_feature_seconds_per_image=vsec,fit_seconds=fit_seconds,
        proposal_metrics=diagnostics(vbank),training=details,detection=metrics)
    (out/"results.json").write_text(json.dumps(report,indent=2))
    with (out/"per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print("E42 RESULT",json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=["dpm","fisher","acf"],required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",default="e42_results")
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--cap",type=int,default=250)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test(args.method)
    else:
        if args.dataset is None:p.error("--dataset required")
        run(args)
