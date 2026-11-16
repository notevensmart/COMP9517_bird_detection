"""E43 full TRAIN RBF SVM, with paired TRAIN-only augmentation ablation.
Streaming Selective Search banks prevent retaining ~1.3M 1924D proposals.
Classical HOG/LBP PCA64 RBF, no neural models. VAL only, no TEST.
"""
import argparse,csv,json,time,gc
from pathlib import Path
import cv2,numpy as np
from sklearn.svm import SVC
from cloud_audit import data,iou,nms
from selective_search_detection import proposed,coverage
from rebuild_detector import descriptor,fit,make_jitter,safe_labels,draw_box
from background_rejector import secondary_features
from nonlinear_appearance_study import fit_rbf,rbf_logits
from metrics_map import evaluate_map50
from e32_failure_driven import per_image_stats
SEED=43
SCALES=("small","medium","large")
def scale_ids(boxes):
    b=np.asarray(boxes,np.float32).reshape(-1,4)
    area=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
    return np.where(area<1024,0,np.where(area<10000,1,2)).astype(np.int8)
def logit(p):
    p=np.clip(p,1e-4,1-1e-4)
    return np.log(p/(1-p))
FOCUS={"beach_bird_1379.jpg","beach_bird_1389.jpg","beach_bird_1399.jpg",
"beach_bird_1419.jpg","beach_bird_1435.jpg","beach_bird_1452.jpg",
"beach_bird_1560.jpg","beach_bird_1606.jpg"}
def aug_image(im,mode):
    if mode=="flip":return cv2.flip(im,1)
    if mode=="photo":
        lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        l=np.clip(l.astype(np.float32)*1.06+6,0,255).astype(np.uint8)
        out=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
        hsv=cv2.cvtColor(out,cv2.COLOR_BGR2HSV)
        hsv[:,:,1]=np.clip(hsv[:,:,1].astype(np.float32)*.94,0,255).astype(np.uint8)
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    raise ValueError(mode)
def map_box(b,w,mode):
    b=np.asarray(b,np.float32).copy()
    if mode=="flip":b[0],b[2]=w-b[2],w-b[0]
    return b
def prep(path,gt,stage1=None,need_features=True):
    im=cv2.imread(str(path))
    if im is None:raise ValueError("Unreadable "+str(path))
    b=proposed(im,"selective")
    p,n=safe_labels(b,gt)
    if need_features:
        X=np.stack([descriptor(im,x) for x in b]) if len(b) else np.empty((0,1924),np.float32)
    else:X=None
    rec=dict(path=path,gt=gt,boxes=b,pos=p,neg=n,X=X)
    if stage1 is not None:
        rec["margin"]=stage1.decision_function(X) if len(X) else np.empty(0)
    return rec,im
def first_pass(train,out):
    rng=np.random.default_rng(47)
    pos=[];neg=[];hard_pool=[]
    for k,(path,gt) in enumerate(train,1):
        rec,im=prep(path,gt,need_features=False)
        for g in gt:
            for b in make_jitter(g,im.shape[1],im.shape[0],rng,count=3):
                pos.append(descriptor(im,b))
        n=rec["neg"]
        if len(n):
            chosen=rng.choice(n,min(20,len(n)),replace=False)
            neg.extend(descriptor(im,rec["boxes"][i]) for i in chosen)
        if k%50==0:print("E43 STAGE1 PASS",k,len(train),"pos",len(pos),"neg",len(neg),flush=True)
    pos=np.asarray(pos,np.float32);neg=np.asarray(neg,np.float32)
    if len(pos)==0 or len(neg)==0:raise RuntimeError("Missing TRAIN examples")
    stage1,_=fit(pos,neg)
    hard=[]
    for k,(path,gt) in enumerate(train,1):
        rec,im=prep(path,gt,need_features=True)
        scores=stage1.decision_function(rec["X"]) if len(rec["X"]) else np.empty(0)
        keep=nms(rec["boxes"],scores,.5,170) if len(scores) else []
        safe=set(int(j) for j in rec["neg"]);count=0
        for j in keep:
            j=int(j)
            if j in safe and scores[j]>-.2:
                hard.append(rec["X"][j]);count+=1
            if count>=12:break
        if k%50==0:print("E43 HARD MINING",k,len(train),"hard",len(hard),flush=True)
    if hard:neg=np.vstack((neg,np.asarray(hard,np.float32)))
    stage1,_=fit(pos,neg)
    report=dict(gt_jitter=len(pos),initial_and_hard_neg=len(neg),hard_neg=len(hard))
    (out/"stage1_train.json").write_text(json.dumps(report,indent=2))
    del pos,neg,hard
    gc.collect()
    return stage1,report
def samples(train,stage1,out):
    rng=np.random.default_rng(77)
    X=[];A=[];M=[];Y=[];areas=[]
    previews=[];counts=dict(images=0,positive=0,negative=0,aug_positive=0,aug_negative=0)
    for k,(path,gt) in enumerate(train,1):
        rec,im=prep(path,gt,stage1)
        b=rec["boxes"];margin=rec["margin"]
        good=np.asarray(rec["pos"],dtype=int)
        if len(good)>12:good=rng.choice(good,12,replace=False)
        safe=set(int(j) for j in rec["neg"])
        keep=nms(b,margin,.5,170) if len(margin) else []
        hard=[int(j) for j in keep if int(j) in safe and margin[int(j)]>-.7][:12]
        remaining=[int(j) for j in rec["neg"] if int(j) not in set(hard)]
        rest=rng.choice(remaining,min(8,len(remaining)),replace=False).tolist() if remaining else []
        for ids,y in ((list(good),1),(hard+rest,0)):
            for j in ids:
                box=b[j]
                X.append(rec["X"][j]);A.append(secondary_features(im,box))
                M.append(margin[j]);Y.append(y);areas.append(int(scale_ids([box])[0]))
                counts["positive" if y else "negative"]+=1
        counts["images"]+=1
        if k%50==0:print("E43 STAGE2 TRAIN",k,len(train),json.dumps(counts),flush=True)
    assert len(X)==len(A)==len(M)==len(Y)
    if previews:
        d=out/"preprocessing";d.mkdir(exist_ok=True)
        cv2.imwrite(str(d/"augmentation_preview.jpg"),np.vstack(previews))
    def pack(x,a,m,y):
        x=np.asarray(x,np.float32);a=np.asarray(a,np.float32)
        m=np.asarray(m,np.float32).reshape(-1,1);y=np.asarray(y,int)
        if x.ndim!=2 or x.shape[1]!=1924 or a.shape[1]!=32 or not np.isfinite(x).all() or not np.isfinite(a).all() or not np.isfinite(m).all():
            raise ValueError("TRAIN features invalid")
        return x,a,m,y
    return pack(X,A,M,Y),np.asarray(areas,np.int8),counts
def eval_val(val,stage1,global_model,experts,out):
    variants=("global_rbf_full","experts_only_full","global_expert_blend_25","global_expert_blend_50","global_expert_blend_75")
    predictions={v:[] for v in variants};rows=[];vis=out/"visual";vis.mkdir(exist_ok=True)
    stats=dict(images=0,gt=0,covered50=0,mean_proposals=0)
    norm,pca,scaler,_=global_model
    for k,(path,gt) in enumerate(val,1):
        rec,im=prep(path,gt,stage1);b=rec["boxes"]
        stats["images"]+=1;stats["gt"]+=len(gt);stats["mean_proposals"]+=len(b)
        if len(b) and len(gt):stats["covered50"]+=int((iou(b,gt).max(axis=0)>=.5).sum())
        if len(b):
            global_log=rbf_logits(global_model,rec)
            x=scaler.transform(np.c_[pca.transform(norm.transform(rec["X"])),np.stack([secondary_features(im,box) for box in b]),rec["margin"]]).astype(np.float32)
            expert_log=global_log.copy()
            groups=scale_ids(b)
            for i,name in enumerate(SCALES):
                ids=np.flatnonzero(groups==i)
                if len(ids) and name in experts:
                    expert_log[ids]=logit(experts[name].predict_proba(x[ids])[:,1])
            margin=rec["margin"]
            scores={
                "global_rbf_full":.5*margin+.5*global_log,
                "experts_only_full":.5*margin+.5*expert_log,
                "global_expert_blend_25":.5*margin+.5*(.75*global_log+.25*expert_log),
                "global_expert_blend_50":.5*margin+.5*(.5*global_log+.5*expert_log),
                "global_expert_blend_75":.5*margin+.5*(.25*global_log+.75*expert_log)}
        else:scores={v:np.empty(0) for v in variants}
        for name in variants:
            inds=nms(b,scores[name],.35,300) if len(b) else np.empty(0,int)
            pred=b[inds];conf=scores[name][inds];predictions[name].append((pred,conf,gt))
            ov=iou(pred[:20],gt);matched=set();tp=0
            for ii in range(min(20,len(pred))):
                for jj in np.argsort(-ov[ii]) if len(gt) else []:
                    if ov[ii,jj]>=.5 and int(jj) not in matched:
                        matched.add(int(jj));tp+=1;break
            rows.append(dict(image=path.name,method=name,GT=len(gt),proposals=len(b),top20_TP=tp,top20_FP=min(20,len(pred))-tp))
            if path.name in FOCUS and name in ("global_rbf_full","experts_only_full","global_expert_blend_50"):
                canvas=im.copy()
                for box in gt:draw_box(canvas,box,(0,255,0),2)
                used=set()
                for ii,box in enumerate(pred[:20]):
                    ok=False
                    for jj in np.argsort(-ov[ii]) if len(gt) else []:
                        if ov[ii,jj]>=.5 and int(jj) not in used:
                            used.add(int(jj));ok=True;break
                    draw_box(canvas,box,(255,0,0) if ok else (0,0,255),2)
                cv2.putText(canvas,f"{name} GT {len(gt)} TP20 {tp}",(5,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2)
                cv2.imwrite(str(vis/(path.stem+"_"+name+".jpg")),canvas)
        if k%20==0:print("E43 VAL",k,len(val),flush=True)
    results=[]
    for name in variants:
        m=evaluate_map50(predictions[name]);r=[x for x in rows if x["method"]==name]
        m.update(method=name,top20_TP=sum(x["top20_TP"] for x in r),top20_FP=sum(x["top20_FP"] for x in r))
        results.append(m);print("E43 RESULT",json.dumps(m),flush=True)
    with (out/"e43_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats["mean_proposals"]/=max(1,len(val));stats["raw_recall50"]=stats["covered50"]/max(1,stats["gt"])
    return results,stats
def tests():
    im=np.arange(3*32*48,dtype=np.uint8).reshape(32,48,3)
    b=np.array([4,8,20,22],np.float32)
    flipped=aug_image(im,"flip");fb=map_box(b,48,"flip")
    assert np.array_equal(flipped[8:22,28:44],cv2.flip(im[8:22,4:20],1))
    assert np.array_equal(fb,[28,8,44,22])
    assert np.array_equal(map_box(b,48,"photo"),b)
    assert aug_image(im,"photo").shape==im.shape
    print("E43 augmentation and bbox geometry self-test PASS",flush=True)
def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    train=[v for v in data(root,"train",0) if len(v[1])]
    val=data(root,"val",0)
    print("E43 FULL TRAIN",len(train),"VAL",len(val),flush=True)
    stage1,stage1diag=first_pass(train,out)
    core,groups,counts=samples(train,stage1,out)
    global_model,diag0=fit_rbf(*core,pca_dim=64)
    norm,pca,scaler,_=global_model
    x,a,m,y=core
    features=scaler.transform(np.c_[pca.transform(norm.transform(x)),a,m]).astype(np.float32)
    del core,x,a,m
    gc.collect()
    experts={};expert_diag={}
    for i,name in enumerate(SCALES):
        ids=np.flatnonzero(groups==i)
        labels=np.bincount(y[ids],minlength=2)
        if len(ids)<50 or labels.min()<15:
            expert_diag[name]=dict(status="global_fallback",examples=len(ids),class_counts=labels.tolist())
            continue
        model=SVC(C=3.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=True,random_state=SEED+i,cache_size=700)
        tic=time.perf_counter();model.fit(features[ids],y[ids])
        experts[name]=model
        expert_diag[name]=dict(status="trained",examples=len(ids),class_counts=labels.tolist(),seconds=time.perf_counter()-tic,support_vectors=int(model.n_support_.sum()),fit_status=int(model.fit_status_),n_iter=model.n_iter_.tolist())
        print("E43 EXPERT",name,json.dumps(expert_diag[name]),flush=True)
    del features,y,groups
    gc.collect()
    results,proposals=eval_val(val,stage1,global_model,experts,out)
    baseline=results[0]
    gates={r["method"]:bool(r["map50_coco101"]>=baseline["map50_coco101"]+.02 and r["top20_TP"]>=baseline["top20_TP"]) for r in results[1:]}
    report=dict(status="COMPLETE",train_images=len(train),val_images=len(val),validation_GT=sum(len(g) for _,g in val),
      split="annotated-positive TRAIN only; complete VAL; no TEST",seed=SEED,stage1=stage1diag,
      global_training=diag0,experts=expert_diag,training_counts=counts,proposals=proposals,results=results,
      promotion_gates=gates,notes="TRAIN-only PCA/scaler and probability calibration; proposal-native GT-blind size routing. Fixed blends 25/50/75%; no validation-tuned weights. No augmentation in this experiment to isolate specialization.")
    (out/"e43_report.json").write_text(json.dumps(report,indent=2))
    print("E43 COMPLETE",json.dumps(dict(results=results,gates=gates)),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("e43_results"))
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:ap.error("--dataset required")
        run(a.dataset,a.output)
