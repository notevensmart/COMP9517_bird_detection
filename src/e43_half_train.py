"""E43: 50%-TRAIN, paper-grounded, resource-aware classical detection benchmark.

Uses 685 of 1369 official TRAIN (seed 61, positive-annotated images); 120/291
VAL (seed 43); NEVER TEST. DPM, Fisher and ACF retain their own paper-native
classification families. E42 feature extraction is reused, but training is
streamed and capped to avoid O(number of images * 1200 * feature dim) RAM.
Separate GT-blind, stable 1200 Selective Search candidate bank per image.

Fidelity: bounded adaptations, not certified historical reference implementations.
See E42_PAPER_REFERENCES.md and E43_50_PERCENT_PROTOCOL.md.
"""
import argparse, csv, gc, json, time
from pathlib import Path
import joblib
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import e42_paper_classical as e42
from cloud_audit import data,iou,nms
from metrics_map import evaluate_map50
from quality_subset_train import training_subset
from rebuild_detector import safe_labels, make_jitter

SEED=61
CAP=1200
NMS_IOU=.35

def pick_by_scale(boxes, n, rng):
    """Balance sampled GT patches across area scales, preserving image diversity."""
    if len(boxes)<=n:return np.arange(len(boxes),dtype=int)
    area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    tier=np.digitize(area,[1024,10000])
    chosen=[]
    for t in (0,1,2):
        ids=np.flatnonzero(tier==t)
        if len(ids):chosen.extend(rng.choice(ids,min(len(ids),max(1,n//3)),replace=False).tolist())
    if len(chosen)<n:
        pool=np.setdiff1d(np.arange(len(boxes)),np.asarray(chosen,dtype=int))
        chosen.extend(rng.choice(pool,min(len(pool),n-len(chosen)),replace=False).tolist())
    return np.asarray(chosen[:n],int)

def pick_negative(boxes,ids,n,rng):
    if not len(ids):return np.empty(0,int)
    area=np.prod(boxes[ids,2:]-boxes[ids,:2],axis=1)
    tier=np.digitize(area,[1024,10000])
    chosen=[]
    for t in (0,1,2):
        group=ids[np.flatnonzero(tier==t)]
        if len(group):chosen.extend(rng.choice(group,min(len(group),max(1,n//3)),replace=False).tolist())
    if len(chosen)<n:
        rest=np.setdiff1d(ids,np.asarray(chosen,dtype=int))
        chosen.extend(rng.choice(rest,min(len(rest),n-len(chosen)),replace=False).tolist())
    return np.asarray(chosen[:n],int)

def stream_training(samples,method,cap,out):
    rng=np.random.default_rng(47)
    pos=[];neg=[];reserve=[]
    diag={"sampled_train_images":len(samples),"all_train_gt":0,"sampled_gt":0,
          "selected_proposal_count":0,"initial_negative":0,"reserve_negative":0,
          "feature_seconds":0.0}
    tick=time.perf_counter()
    for index,(path,gt) in enumerate(samples,1):
        img=cv2.imread(str(path))
        if img is None:raise ValueError("unreadable "+str(path))
        boxes=e42.stable_ss(img,cap)
        positive_indices=pick_by_scale(gt,6,rng)
        positives=[gt[i] for i in positive_indices]
        for j,b in enumerate(positives):
            pos.append(e42.extract(e42.crop(img,b),method))
            # Extra local jitter per image (not all positives) for tolerance
            if j==0:
                variant=make_jitter(b,img.shape[1],img.shape[0],rng,count=1)
                if len(variant)>1:
                    pos.append(e42.extract(e42.crop(img,variant[1]),method))
        _,safe=safe_labels(boxes,gt)
        initial_ids=pick_negative(boxes,safe,6,rng)
        for j in initial_ids:neg.append(e42.extract(e42.crop(img,boxes[int(j)]),method))
        remaining=np.setdiff1d(safe,initial_ids)
        reserve_ids=pick_negative(boxes,remaining,2,rng)
        for j in reserve_ids:reserve.append(e42.extract(e42.crop(img,boxes[int(j)]),method))
        diag["all_train_gt"]+=len(gt);diag["sampled_gt"]+=len(positive_indices)
        diag["selected_proposal_count"]+=len(boxes)
        if index%25==0 or index==len(samples):
            print("E43 TRAIN FEATURES",method,index,"/",len(samples),
                  "pos",len(pos),"neg",len(neg),"reserve",len(reserve),flush=True)
            (out/"progress.json").write_text(json.dumps(
                dict(stage="TRAIN features",method=method,images=index,
                     total=len(samples),positives=len(pos),initial_negatives=len(neg),
                     reserve=len(reserve)),indent=2))
    diag.update(initial_negative=len(neg),reserve_negative=len(reserve),
                positives=len(pos),feature_seconds=time.perf_counter()-tick,
                mean_proposals=diag["selected_proposal_count"]/len(samples))
    return pos,neg,reserve,diag

def fit_dpm(pos,neg,reserve):
    """Three latent-part reestimations, then one TRAIN-only hard negative pass."""
    w=None;history=[];initial_negative=len(neg)
    for i in range(3):
        X=np.stack([e42.dpm_latent(x,w) for x in pos+neg]).astype(np.float32)
        labels=np.r_[np.ones(len(pos),np.int8),np.zeros(len(neg),np.int8)]
        clf=LinearSVC(C=.05,max_iter=7000,random_state=SEED)
        clf.fit(X,labels);w=clf.coef_.ravel()
        history.append(dict(round=i,train_patch_accuracy=float(np.mean(clf.predict(X)==labels))))
        del X;gc.collect()
        if i==0 and reserve:
            scores=np.array([float(w@e42.dpm_latent(x,w)+clf.intercept_[0]) for x in reserve])
            # Only mine confidently hard (misranked) TRAIN-safe negative crops.
            chosen=np.argsort(-scores)[:min(800,len(scores))]
            hard=[reserve[j] for j in chosen if scores[j]>-.2]
            neg=neg+hard
            print("E43 DPM hard negatives",len(hard),flush=True)
    return (clf,w),dict(history=history,initial_negative=initial_negative,
                         mined_negative=len(neg)-initial_negative)

def fit_fisher(pos,neg,reserve):
    """32 diagonal GMM components, improved Fisher pooling, hard-negative SVM refit."""
    rng=np.random.default_rng(SEED)
    all_local=np.vstack([x for x in pos+neg if len(x)])
    if len(all_local)>24000:
        all_local=all_local[rng.choice(len(all_local),24000,replace=False)]
    gmm=GaussianMixture(n_components=32,covariance_type="diag",reg_covar=1e-4,
                        max_iter=90,random_state=SEED)
    gmm.fit(all_local)
    Xp=np.stack([e42.fisher_encode(v,gmm) for v in pos])
    Xn=np.stack([e42.fisher_encode(v,gmm) for v in neg])
    X=np.vstack((Xp,Xn));y=np.r_[np.ones(len(Xp),int),np.zeros(len(Xn),int)]
    clf=LinearSVC(C=1.,class_weight="balanced",max_iter=12000,random_state=SEED)
    clf.fit(X,y)
    mined=0
    if reserve:
        # STREAM reserve vectors to avoid copying all raw SIFT arrays.
        margin=np.array([float(clf.decision_function(e42.fisher_encode(v,gmm)[None])[0])
                         for v in reserve])
        chosen=np.argsort(-margin)[:min(1000,len(margin))]
        extra=[e42.fisher_encode(reserve[j],gmm) for j in chosen if margin[j]>-.2]
        if extra:
            X=np.vstack((X,np.stack(extra)))
            y=np.r_[y,np.zeros(len(extra),int)]
            clf=LinearSVC(C=1.,class_weight="balanced",max_iter=12000,random_state=SEED)
            clf.fit(X,y);mined=len(extra)
    return (clf,gmm),dict(gmm_components=32,fisher_dim=8192,learned_gmm_on_train_only=True,
                          gmm_converged=bool(gmm.converged_),mined_negative=mined,
                          train_patch_accuracy=float(np.mean(clf.predict(X)==y)))

def fit_acf(pos,neg,reserve):
    """256 depth-two boosted decision trees; modest TRAIN-only hard-negative round."""
    Xp=np.stack(pos);Xn=np.stack(neg)
    X=np.vstack((Xp,Xn));y=np.r_[np.ones(len(Xp),int),np.zeros(len(Xn),int)]
    def classifier(estimators):
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(
            max_depth=2,min_samples_leaf=5,random_state=SEED),
            n_estimators=estimators,learning_rate=.6,random_state=SEED)
    first=classifier(96)
    first.fit(X,y)
    mined=0
    if reserve:
        R=np.stack(reserve)
        margins=first.decision_function(R)
        selected=np.argsort(-margins)[:min(850,len(R))]
        extra=R[[j for j in selected if margins[j]>-.2]]
        if len(extra):
            X=np.vstack((X,extra))
            y=np.r_[y,np.zeros(len(extra),int)]
            mined=len(extra)
    clf=classifier(256)
    clf.fit(X,y)
    return clf,dict(n_estimators=len(clf.estimators_),tree_depth=2,
                    mined_negative=mined,train_patch_accuracy=float(np.mean(clf.predict(X)==y)))

def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=training_subset(args.dataset,args.train_n)
    val=data(args.dataset,"val",args.val_n)
    if len(train)!=685:raise ValueError("50% protocol requires exactly 685 TRAIN images")
    if any(not len(gt) for _,gt in train):raise ValueError("zero-GT train image")
    names={path.name for path,_ in train}
    assert not(names & {path.name for path,_ in val})
    print("E43 DATA",args.method,len(train),len(val),"VAL_GT",sum(len(gt) for _,gt in val),flush=True)
    pos,neg,reserve,train_details=stream_training(train,args.method,args.cap,out)
    tic=time.perf_counter()
    if args.method=="dpm": model,fitdiag=fit_dpm(pos,neg,reserve)
    elif args.method=="fisher":model,fitdiag=fit_fisher(pos,neg,reserve)
    else:model,fitdiag=fit_acf(pos,neg,reserve)
    fit_seconds=time.perf_counter()-tic
    del pos,neg,reserve;gc.collect()
    joblib.dump(model,out/("model_"+args.method+".joblib"),compress=3)
    predictions=[];rows=[];image_seconds=[];counts={"all":[0,0,0],"small":[0,0,0],"medium":[0,0,0],"large":[0,0,0]}
    selected={"beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
              "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg",
              "beach_bird_1606.jpg"}
    visual=out/"visual";visual.mkdir(exist_ok=True)
    for idx,(path,gt) in enumerate(val,1):
        start=time.perf_counter()
        img=cv2.imread(str(path))
        if img is None:raise ValueError("unreadable "+str(path))
        boxes=e42.stable_ss(img,args.cap)
        features=[e42.extract(e42.crop(img,b),args.method) for b in boxes]
        scores=np.asarray(e42.score(model,features,args.method),np.float32)
        assert len(scores)==len(boxes) and np.isfinite(scores).all()
        keep=nms(boxes,scores,NMS_IOU,300) if len(boxes) else np.empty(0,int)
        bb=boxes[keep];ss=scores[keep]
        image_seconds.append(time.perf_counter()-start)
        predictions.append((bb,ss,gt))
        raw=iou(boxes,gt)
        matched=raw.max(axis=0) if len(boxes) else np.zeros(len(gt))
        for b,value in zip(gt,matched):
            area=np.prod(b[2:]-b[:2])
            sz="small" if area<1024 else ("medium" if area<10000 else "large")
            for key in ("all",sz):
                counts[key][0]+=1;counts[key][1]+=int(value>=.5);counts[key][2]+=int(value>=.75)
        overlaps=iou(bb[:20],gt);used=set();hits=[]
        for row in overlaps:
            ok=False
            for j in np.argsort(-row):
                if row[j]>=.5 and int(j) not in used:
                    used.add(int(j));ok=True;break
            hits.append(ok)
        rows.append(dict(method=args.method,image=path.name,GT=len(gt),
                         proposals=len(boxes),top20_TP=sum(hits),
                         top20_FP=len(hits)-sum(hits),inference_seconds=image_seconds[-1]))
        if path.name in selected:
            render=img.copy()
            for b in gt:
                x1,y1,x2,y2=b.astype(int);cv2.rectangle(render,(x1,y1),(x2,y2),(0,230,0),2)
            for b,ok in zip(bb[:20],hits):
                x1,y1,x2,y2=b.astype(int)
                cv2.rectangle(render,(x1,y1),(x2,y2),(255,0,0) if ok else (0,0,255),1)
            cv2.imwrite(str(visual/(args.method+"_"+path.stem+".jpg")),render)
        if idx%15==0 or idx==len(val):
            print("E43 VAL",args.method,idx,"/",len(val),"gt",len(gt),flush=True)
            (out/"progress.json").write_text(json.dumps(
                dict(stage="VAL",method=args.method,images=idx,total=len(val)),indent=2))
    metrics=evaluate_map50(predictions)
    raw_recall={k:dict(n=v[0],R50=v[1]/v[0] if v[0] else None,
                       R75=v[2]/v[0] if v[0] else None) for k,v in counts.items()}
    report=dict(experiment="E43",model=args.method,
        protocol="685 annotated TRAIN / 120 official VAL / 0 TEST, fixed seeds",
        paper_derived_bounded_adaptation=True,not_exact_historical_reproduction=True,
        train_images=len(train),official_total_train=1369,percent_all_train=100*len(train)/1369,
        val_images=len(val),val_GT=sum(len(gt) for _,gt in val),
        train_ids=[p.name for p,_ in train],val_ids=[p.name for p,_ in val],
        proposal="canonicalized Selective Search GT-blind stable 1200 cap",
        nms_iou=NMS_IOU,training=train_details,fit=fitdiag,fit_seconds=fit_seconds,
        mean_inference_seconds=float(np.mean(image_seconds)),
        proposal_recall=raw_recall,
        detection={**metrics,"top20_TP":sum(r["top20_TP"] for r in rows),
                   "top20_FP":sum(r["top20_FP"] for r in rows)})
    (out/"results.json").write_text(json.dumps(report,indent=2))
    with (out/"per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print("E43 COMPLETE",json.dumps(dict(method=args.method,train=len(train),
        val=len(val),mean_proposals=train_details["mean_proposals"],raw=raw_recall,
        result=report["detection"])),flush=True)

def test(method):
    e42.self_test(method)
    r=np.random.default_rng(17)
    gt=np.array([[1.,2.,8.,9.],[10.,10.,50.,60.],[20.,20.,120.,140.]],np.float32)
    selected=pick_by_scale(gt,2,r)
    assert len(selected)==2 and len(set(selected))==2
    if method=="acf":assert e42.acf_features(np.zeros((64,64,3),np.uint8)).shape==(2560,)
    print("E43 PASS",method,"scale-balanced sampling and underlying paper features")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=["dpm","fisher","acf"],required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",default="e43_results")
    p.add_argument("--train-n",type=int,default=685)
    p.add_argument("--val-n",type=int,default=120)
    p.add_argument("--cap",type=int,default=1200)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:test(args.method)
    else:
        if args.dataset is None:p.error("--dataset required")
        run(args)
