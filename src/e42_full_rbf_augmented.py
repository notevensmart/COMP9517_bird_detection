"""E42 full TRAIN RBF SVM, with paired TRAIN-only augmentation ablation.
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
SEED=42
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
        if k%50==0:print("E42 STAGE1 PASS",k,len(train),"pos",len(pos),"neg",len(neg),flush=True)
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
        if k%50==0:print("E42 HARD MINING",k,len(train),"hard",len(hard),flush=True)
    if hard:neg=np.vstack((neg,np.asarray(hard,np.float32)))
    stage1,_=fit(pos,neg)
    report=dict(gt_jitter=len(pos),initial_and_hard_neg=len(neg),hard_neg=len(hard))
    (out/"stage1_train.json").write_text(json.dumps(report,indent=2))
    del pos,neg,hard
    gc.collect()
    return stage1,report
def samples(train,stage1,out):
    rng=np.random.default_rng(77)
    X=[];A=[];M=[];Y=[];augX=[];augA=[];augM=[];augY=[]
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
                M.append(margin[j]);Y.append(y)
                counts["positive" if y else "negative"]+=1
            # Max 6 patches per class/image/mode; avoid inflated correlated
            # examples and preserve whole-image context for aux descriptors.
            for mode in ("flip","photo"):
                transformed=aug_image(im,mode)
                for j in ids[:6]:
                    box=map_box(b[j],im.shape[1],mode)
                    f=descriptor(transformed,box)
                    aux=secondary_features(transformed,box)
                    # Stage1 sees augmented crop: no original-image score leakage.
                    score=float(stage1.decision_function(f.reshape(1,-1))[0])
                    augX.append(f);augA.append(aux);augM.append(score);augY.append(y)
                    counts["aug_positive" if y else "aug_negative"]+=1
                    if len(previews)<12:
                        raw=cv2.resize(im[max(0,int(b[j][1])):max(1,int(b[j][3])),
                                             max(0,int(b[j][0])):max(1,int(b[j][2]))],(96,96))
                        changed=cv2.resize(transformed[max(0,int(box[1])):max(1,int(box[3])),
                                                       max(0,int(box[0])):max(1,int(box[2]))],(96,96))
                        previews.append(np.hstack((raw,changed)))
        counts["images"]+=1
        if k%50==0:print("E42 STAGE2 TRAIN",k,len(train),json.dumps(counts),flush=True)
    assert len(X)==len(A)==len(M)==len(Y)
    assert len(augX)==len(augA)==len(augM)==len(augY)
    if previews:
        d=out/"preprocessing";d.mkdir(exist_ok=True)
        cv2.imwrite(str(d/"augmentation_preview.jpg"),np.vstack(previews))
    def pack(x,a,m,y):
        x=np.asarray(x,np.float32);a=np.asarray(a,np.float32)
        m=np.asarray(m,np.float32).reshape(-1,1);y=np.asarray(y,int)
        if x.ndim!=2 or x.shape[1]!=1924 or a.shape[1]!=32 or not np.isfinite(x).all() or not np.isfinite(a).all() or not np.isfinite(m).all():
            raise ValueError("TRAIN features invalid")
        return x,a,m,y
    return pack(X,A,M,Y),pack(augX,augA,augM,augY),counts
def eval_val(val,stage1,rbf0,rbfaug,out):
    variants=("global_rbf_full_no_aug","global_rbf_full_augmented")
    predictions={v:[] for v in variants};rows=[];vis=out/"visual";vis.mkdir(exist_ok=True)
    proposal_stat=dict(images=0,gt=0,covered50=0,mean_proposals=0)
    for k,(path,gt) in enumerate(val,1):
        rec,im=prep(path,gt,stage1)
        b=rec["boxes"];proposal_stat["images"]+=1;proposal_stat["gt"]+=len(gt)
        proposal_stat["mean_proposals"]+=len(b)
        if len(b) and len(gt):
            proposal_stat["covered50"]+=int((iou(b,gt).max(axis=0)>=.5).sum())
        for name,model in zip(variants,(rbf0,rbfaug)):
            scores=.5*rec["margin"]+.5*rbf_logits(model,rec) if len(b) else np.empty(0)
            inds=nms(b,scores,.35,300) if len(b) else np.empty(0,int)
            pred=b[inds];conf=scores[inds]
            predictions[name].append((pred,conf,gt))
            ov=iou(pred[:20],gt);matched=set();tp=0
            for ii in range(min(20,len(pred))):
                for jj in np.argsort(-ov[ii]) if len(gt) else []:
                    if ov[ii,jj]>=.5 and int(jj) not in matched:
                        matched.add(int(jj));tp+=1;break
            rows.append(dict(image=path.name,method=name,GT=len(gt),proposals=len(b),
                             top20_TP=tp,top20_FP=min(20,len(pred))-tp))
            if path.name in FOCUS:
                canvas=im.copy()
                for box in gt:draw_box(canvas,box,(0,255,0),2)
                used=set()
                for ii,box in enumerate(pred[:20]):
                    matched=False
                    for jj in np.argsort(-ov[ii]) if len(gt) else []:
                        if ov[ii,jj]>=.5 and int(jj) not in used:
                            used.add(int(jj));matched=True;break
                    draw_box(canvas,box,(255,0,0) if matched else (0,0,255),2)
                cv2.putText(canvas,f"{name} GT {len(gt)} TP20 {tp}",(5,22),
                    cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2)
                cv2.imwrite(str(vis/(path.stem+"_"+name+".jpg")),canvas)
        if k%20==0:print("E42 VAL",k,len(val),flush=True)
    results=[]
    for name in variants:
        m=evaluate_map50(predictions[name])
        r=[x for x in rows if x["method"]==name]
        m.update(method=name,top20_TP=sum(x["top20_TP"] for x in r),
                 top20_FP=sum(x["top20_FP"] for x in r))
        results.append(m)
        print("E42 RESULT",json.dumps(m),flush=True)
    with (out/"e42_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    proposal_stat["mean_proposals"]/=max(1,len(val))
    proposal_stat["raw_recall50"]=proposal_stat["covered50"]/max(1,proposal_stat["gt"])
    return results,proposal_stat
def tests():
    im=np.arange(3*32*48,dtype=np.uint8).reshape(32,48,3)
    b=np.array([4,8,20,22],np.float32)
    flipped=aug_image(im,"flip");fb=map_box(b,48,"flip")
    assert np.array_equal(flipped[8:22,28:44],cv2.flip(im[8:22,4:20],1))
    assert np.array_equal(fb,[28,8,44,22])
    assert np.array_equal(map_box(b,48,"photo"),b)
    assert aug_image(im,"photo").shape==im.shape
    print("E42 augmentation and bbox geometry self-test PASS",flush=True)
def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    train=[v for v in data(root,"train",0) if len(v[1])]
    val=data(root,"val",0)
    print("E42 FULL TRAIN",len(train),"VAL",len(val),flush=True)
    stage1,stage1diag=first_pass(train,out)
    core,aug,counts=samples(train,stage1,out)
    model0,diag0=fit_rbf(*core,pca_dim=64)
    # Freeze the baseline TRAIN-fitted PCA and standardizer for the augmented
    # arm: augmentation changes ONLY the SVM fitting examples, not representation.
    norm,pca,scaler,_=model0
    def transformed(batch):
        x,a,m,y=batch
        z=pca.transform(norm.transform(x))
        features=scaler.transform(np.c_[z,a,m]).astype(np.float32)
        if not np.isfinite(features).all():raise ValueError("Invalid PCA input")
        return features,y
    core_z,y0=transformed(core)
    aug_z,ya=transformed(aug)
    del core,aug
    gc.collect()
    augmented_svm=SVC(C=3.0,kernel="rbf",gamma="scale",class_weight="balanced",
         probability=True,random_state=47,cache_size=750)
    tic=time.perf_counter()
    augmented_svm.fit(np.vstack((core_z,aug_z)),np.r_[y0,ya])
    diag_aug=dict(train_examples=len(y0)+len(ya),augmented_examples=len(ya),
         train_seconds=time.perf_counter()-tic,support_vectors=int(augmented_svm.n_support_.sum()))
    model_aug=(norm,pca,scaler,augmented_svm)
    del core_z,aug_z,y0,ya
    gc.collect()
    results,proposals=eval_val(val,stage1,model0,model_aug,out)
    report=dict(status="COMPLETE",train_images=len(train),val_images=len(val),
      validation_GT=sum(len(gt) for _,gt in val),
      split="all annotated-positive TRAIN, all VAL; official TEST not accessed",
      augmented_methods=["whole-image horizontal flip with correct box mapping",
         "conservative LAB brightness/contrast + HSV saturation"],
      augmentation_counts=counts,stage1_training=stage1diag,
      global_rbf_training=diag0,augmented_rbf_training=diag_aug,
      proposals=proposals,results=results,
      note="Both SVCs share baseline TRAIN-fitted PCA/scaler; same proposal bank and NMS. Full TRAIN excludes zero-annotation images because known incomplete labels.")
    (out/"e42_report.json").write_text(json.dumps(report,indent=2))
    print("E42 COMPLETE",json.dumps(report),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("e42_results"))
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:ap.error("--dataset required")
        run(a.dataset,a.output)
