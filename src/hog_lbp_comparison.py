"""Controlled HOG versus HOG+LBP detector experiment.

Both models use exactly the same fixed training/validation images and pixel-only
spatially balanced proposals. The training set alone supplies class labels and
hard negatives. Validation AP50 uses globally sorted scores, with the same
NMS algorithm and no candidate GT injection. No test split is used.
"""
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np
from skimage.feature import hog,local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,iou,ap50,nms
from spatial_pruning import generate_budgeted

def features(gray,lbp_enabled):
    hog_vec=hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),
                block_norm="L2-Hys").astype(np.float32)
    if not lbp_enabled:return hog_vec
    texture=local_binary_pattern(gray,8,1,method="uniform")
    cells=[]
    for row in np.array_split(texture,4,axis=0):
        for cell in np.array_split(row,4,axis=1):
            h=np.bincount(np.minimum(cell.astype(np.int32).ravel(),9),minlength=10).astype(np.float32)
            cells.append(h/max(float(h.sum()),1.0))
    return np.concatenate([hog_vec,*cells]).astype(np.float32)

def extract_all(image,b,lbp_enabled):
    out=[]
    for box in b:
        x1,y1,x2,y2=np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(int)
        patch=image[y1:y2,x1:x2]
        if not patch.size:raise ValueError("empty candidate")
        gray=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
        out.append(features(gray,lbp_enabled))
    return np.stack(out)

def classes(boxes,gt):
    overlap=iou(boxes,gt)
    max_iou=overlap.max(axis=1) if len(gt) else np.zeros(len(boxes))
    if len(gt) and len(boxes):
        wh=np.maximum(0,np.minimum(boxes[:,None,2:],gt[None,:,2:])-np.maximum(boxes[:,None,:2],gt[None,:,:2]))
        intersection=wh[:,:,0]*wh[:,:,1]
        a=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
        b=np.prod(gt[:,2:]-gt[:,:2],axis=1)
        containment=(intersection/np.maximum(np.minimum(a[:,None],b[None,:]),1e-9)).max(axis=1)
    else:containment=np.zeros(len(boxes))
    return np.flatnonzero(max_iou>=.5),np.flatnonzero((max_iou<.1)&(containment<.5))

def prepare(samples,per_shape,lbp_enabled,for_training):
    bank=[];start=time.perf_counter()
    for idx,(path,gt) in enumerate(samples):
        im=cv2.imread(str(path))
        boxes,_=generate_budgeted(im,per_shape)
        X=extract_all(im,boxes,lbp_enabled)
        record=dict(image=path.name,gt=gt,boxes=boxes,X=X)
        if for_training:
            record["positives"],record["negatives"]=classes(boxes,gt)
        bank.append(record)
        print("FEATURE BANK", "train" if for_training else "val", "hog_lbp" if lbp_enabled else "hog",
              idx+1,"/",len(samples),path.name, len(boxes),"boxes",flush=True)
    return bank,time.perf_counter()-start

def train_model(train_bank,seed=42,mining_rounds=2):
    rng=np.random.default_rng(seed)
    positives=[];negatives=[];offset=0
    X=np.concatenate([e["X"] for e in train_bank])
    for e in train_bank:
        pos=e["positives"];neg=e["negatives"]
        positives.extend((offset+pos).tolist())
        if len(neg):
            k=min(10,len(neg))
            negatives.extend((offset+rng.choice(neg,k,replace=False)).tolist())
        offset+=len(e["X"])
    positives=np.asarray(positives,dtype=int)
    if not len(positives) or not len(negatives):
        raise RuntimeError(f"No trainable positives/negatives: {len(positives)} / {len(negatives)}")
    negatives=set(negatives)
    all_negative=np.concatenate([np.arange(sum(len(e["X"]) for e in train_bank[:i]),
                      sum(len(e["X"]) for e in train_bank[:i])+len(e["X"]))[e["negatives"]]
                      for i,e in enumerate(train_bank)]).astype(int)
    progress=[]
    for rd in range(mining_rounds+1):
        neg=np.asarray(sorted(negatives),dtype=int)
        indices=np.r_[positives,neg]
        y=np.r_[np.ones(len(positives)),np.zeros(len(neg))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=5000,random_state=seed))
        model.fit(X[indices],y)
        train_accuracy=float((model.predict(X[indices])==y).mean())
        scored=model.decision_function(X[all_negative])
        ranked=all_negative[np.argsort(-scored,kind="stable")]
        extra=[]
        for index in ranked:
            if int(index) not in negatives:extra.append(int(index))
            if len(extra)>=500:break
        progress.append(dict(round=rd,pos=len(positives),neg=len(neg),
                             patch_accuracy=train_accuracy,new_hard_negs=len(extra)))
        if rd<mining_rounds:negatives.update(extra)
    return model,progress

def evaluate(model,bank):
    records=[];predictions=[]
    for e in bank:
        start=time.perf_counter()
        s=model.decision_function(e["X"])
        # AP uses all NMS-kept boxes, including negative margins: no threshold
        # tuning on validation to inflate AP.
        keep=nms(e["boxes"],s,iou_limit=.35,max_det=300)
        b=e["boxes"][keep];score=s[keep]
        predictions.append((b,score,e["gt"]))
        records.append(dict(image=e["image"],gt=len(e["gt"]),raw_candidates=len(e["boxes"]),
             after_nms=len(keep),positive_margin=int((score>=0).sum()),
             scoring_seconds=time.perf_counter()-start))
    return ap50(predictions),records

def experiment(root,out,train_limit,val_limit,per_shape):
    out.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",train_limit)
    val=data(root,"val",val_limit)
    summary=[];details=[]
    for use_lbp in (False,True):
        name="hog_lbp" if use_lbp else "hog_only"
        train_bank,train_feature_time=prepare(train,per_shape,use_lbp,True)
        val_bank,val_feature_time=prepare(val,per_shape,use_lbp,False)
        start=time.perf_counter()
        model,progress=train_model(train_bank)
        train_time=time.perf_counter()-start
        ap,records=evaluate(model,val_bank)
        details.extend([dict(model=name,**r) for r in records])
        summary.append(dict(model=name,train_images=len(train),val_images=len(val),
            ground_truth_birds=sum(len(gt) for _,gt in val),per_shape=per_shape,
            mean_candidates=float(np.mean([r["raw_candidates"] for r in records])),
            mean_detections_after_nms=float(np.mean([r["after_nms"] for r in records])),
            mean_positive_margin=float(np.mean([r["positive_margin"] for r in records])),
            validation_AP50_continuous=ap,train_feature_seconds=train_feature_time,
            val_feature_seconds=val_feature_time,train_seconds=train_time,
            mean_scoring_seconds=float(np.mean([r["scoring_seconds"] for r in records])),
            mining=progress))
        print("ABLATION RESULT",json.dumps(summary[-1]),flush=True)
    (out/"hog_lbp_comparison.json").write_text(json.dumps(summary,indent=2))
    with (out/"hog_lbp_predictions_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=details[0].keys());w.writeheader();w.writerows(details)
    print("EXPERIMENT COMPLETED",flush=True)

def tests():
    gray=np.zeros((64,64),np.uint8)
    h=features(gray,False);l=features(gray,True)
    assert len(h)==1764 and len(l)==1924
    assert np.allclose(l[:1764],h)
    assert np.isfinite(l).all()
    print("HOG and LBP shape/identity tests PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("comparison_results"))
    p.add_argument("--train-limit",type=int,default=40)
    p.add_argument("--val-limit",type=int,default=12)
    p.add_argument("--per-shape",type=int,default=50)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args();tests()
    if not args.self_test:
        if not args.dataset:p.error("--dataset required")
        experiment(args.dataset,args.output,args.train_limit,args.val_limit,args.per_shape)
