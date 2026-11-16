"""CLASSICAL BIRD DETECTOR LEARNING CURVE — fixed validation subset.

Hypothesis: the HOG+LBP linear SVM underfits because it was trained on only
35 images. Use increasing NESTED train subsets, common validation images,
identical LoG+Otsu proposals, fixed class labels, and 3 rounds of hard-negative
mining. Primary metric: COCO 101-point single-class bbox mAP@0.50 (maxDets=100).

No test GT or val GT is used for training, candidate generation, or feature
selection. AP is computed from globally ranked detector scores after NMS.
The plotted scores are exploratory validation results, not test performance.

For CPU/GitHub RAM: retain a bounded number of train candidate descriptors
per image. This is explicitly DIFFERENT from the earlier E14 pipeline and
will be reported as a separately paired study, not conflated with E14.
"""
import argparse, csv, gc, json, random, time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data, iou, nms
from classical_methods_study import blob_boxes, classification_errors
from hog_lbp_comparison import features
from metrics_map import evaluate_map50

SEED = 44

def candidate_labels(boxes, gt):
    ov = iou(boxes, gt)
    best = ov.max(axis=1) if len(gt) else np.zeros(len(boxes))
    if len(gt):
        wh = np.maximum(0, np.minimum(boxes[:,None,2:],gt[None,:,2:])-
                            np.maximum(boxes[:,None,:2],gt[None,:,:2]))
        intersection = wh[:,:,0]*wh[:,:,1]
        boxarea = np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
        gtarea = np.prod(gt[:,2:]-gt[:,:2],axis=1)
        contained=(intersection/np.maximum(np.minimum(boxarea[:,None],gtarea[None,:]),1e-9)).max(axis=1)
    else: contained = np.zeros(len(boxes))
    return np.flatnonzero(best>=.5), np.flatnonzero((best<.1)&(contained<.5))

def descriptor(image, box):
    a=np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(int)
    patch=image[a[1]:a[3],a[0]:a[2]]
    if not patch.size: raise ValueError("Empty crop")
    gray=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
    return features(gray,True).astype(np.float32)

def pick_training(samples, max_pos=24, max_neg_pool=55, max_peaks=120):
    """Cache only TRAIN candidates, with positives and a reproducible safe neg pool."""
    records=[]; rng=np.random.default_rng(SEED)
    positive_total=0;negative_total=0;seconds=time.perf_counter()
    for k,(path,gt) in enumerate(samples):
        image=cv2.imread(str(path));boxes,_=blob_boxes(image,"otsu",max_peaks=max_peaks)
        if not len(boxes):
            records.append(dict(name=path.name,X=np.empty((0,1924),np.float32),
                is_pos=np.zeros(0,bool),is_initial_neg=np.zeros(0,bool)));continue
        p,n=candidate_labels(boxes,gt)
        if len(p)>max_pos:p=rng.choice(p,max_pos,replace=False)
        if len(n)>max_neg_pool:n=rng.choice(n,max_neg_pool,replace=False)
        # retain selected positives and a sampled pool of safe negatives
        ids=np.r_[p,n]
        if len(ids):X=np.stack([descriptor(image,boxes[i]) for i in ids])
        else:X=np.empty((0,1924),np.float32)
        is_pos=np.r_[np.ones(len(p),bool),np.zeros(len(n),bool)]
        initial=np.zeros(len(ids),bool)
        if len(n):initial[len(p):len(p)+min(20,len(n))]=True
        positive_total+=len(p);negative_total+=len(n)
        records.append(dict(name=path.name,X=X,is_pos=is_pos,is_initial_neg=initial))
        if (k+1)%50==0 or k+1==len(samples):
            print("TRAIN CACHE",k+1,"/",len(samples),"positive",positive_total,
                  "negative pool",negative_total,"seconds",round(time.perf_counter()-seconds,1),flush=True)
    return records

def validation_bank(samples,max_peaks=120):
    bank=[];start=time.perf_counter()
    for k,(path,gt) in enumerate(samples):
        image=cv2.imread(str(path));t=time.perf_counter()
        b,_=blob_boxes(image,"otsu",max_peaks=max_peaks)
        proposal_secs=time.perf_counter()-t
        t=time.perf_counter()
        X=np.stack([descriptor(image,v) for v in b]) if len(b) else np.empty((0,1924),np.float32)
        feature_secs=time.perf_counter()-t
        bank.append(dict(name=path.name,gt=gt,boxes=b,X=X,
                    proposal_seconds=proposal_secs,feature_seconds=feature_secs))
        if (k+1)%10==0 or k+1==len(samples):
            print("VAL CACHE",k+1,"/",len(samples),"elapsed",round(time.perf_counter()-start,1),flush=True)
    return bank

def validation_evaluate(model,bank):
    preds=[];tp=fp_bg=fp_loc=fp_dup=fn=total_boxes=empty_fp=0
    for record in bank:
        if len(record["boxes"]):
            scores=model.decision_function(record["X"])
            keep=nms(record["boxes"],scores,iou_limit=.35,max_det=300)
            b=record["boxes"][keep];s=scores[keep]
        else:
            b=np.empty((0,4),np.float32);s=np.empty(0,np.float32)
        preds.append((b,s,record["gt"]))
        active=s>=0.
        errors=classification_errors(b[active],s[active],record["gt"])
        tp+=errors["TP"];fp_bg+=errors["FP_background"];fp_loc+=errors["FP_localization"]
        fp_dup+=errors["FP_duplicates"];fn+=errors["FN"];total_boxes+=int(active.sum())
        if not len(record["gt"]):empty_fp+=errors["FP_total"]
    result=evaluate_map50(preds)
    result.update(TP_margin0=tp,FP_background_margin0=fp_bg,
       FP_localization_margin0=fp_loc,FP_duplicate_margin0=fp_dup,
       FP_total_margin0=fp_bg+fp_loc+fp_dup,FN_margin0=fn,
       mean_final_boxes_margin0=total_boxes/len(bank),
       empty_image_FP_margin0=empty_fp)
    return result

def train_size(records, size, bank, rounds=3):
    chosen=records[:size]
    X=np.concatenate([r["X"] for r in chosen])
    pos=[];neg_pool=[];initial=[];off=0
    for record in chosen:
        pos.extend((off+np.flatnonzero(record["is_pos"])).tolist())
        neg_pool.extend((off+np.flatnonzero(~record["is_pos"])).tolist())
        initial.extend((off+np.flatnonzero(record["is_initial_neg"])).tolist())
        off+=len(record["X"])
    pos=np.asarray(pos,dtype=int);neg_pool=np.asarray(neg_pool,dtype=int)
    if not len(pos) or not len(initial):raise RuntimeError(f"Untrainable size {size}: pos={len(pos)}, neg={len(initial)}")
    selected=set(initial);history=[]
    for rd in range(rounds+1):
        train_start=time.perf_counter()
        neg=np.asarray(sorted(selected),dtype=int)
        indices=np.r_[pos,neg]
        y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",
                             max_iter=10000,random_state=42))
        model.fit(X[indices],y)
        fit_seconds=time.perf_counter()-train_start
        evaluation=validation_evaluate(model,bank)
        # TRAIN only: mine top 500 confusing safe negatives not already selected.
        scores=model.decision_function(X[neg_pool])
        ranked=neg_pool[np.argsort(-scores,kind="stable")]
        extras=[]
        for idx in ranked:
            if int(idx) not in selected:extras.append(int(idx))
            if len(extras)>=500:break
        history.append(dict(train_images=size,round=rd,
            train_positive_features=len(pos),train_negative_features=len(neg),
            mining_candidates=len(neg_pool),new_hard_negatives=len(extras),
            fit_seconds=fit_seconds,**evaluation))
        print("CURVE POINT",json.dumps(history[-1]),flush=True)
        if rd<rounds:selected.update(extras)
    del X,model
    gc.collect()
    return history

def plot_curve(results,out):
    sizes=[r["train_images"] for r in results]
    values=[r["selected_map50"]*100 for r in results]
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    ax.plot(sizes,values,"o-",linewidth=2,markersize=7)
    for x,y in zip(sizes,values):
        ax.annotate(f"{y:.2f}%",(x,y),textcoords="offset points",xytext=(0,9),ha="center")
    ax.set_xlabel("Number of training images")
    ax.set_ylabel("Validation mAP@0.5 (%) — COCO 101-point")
    ax.set_title("Classical Bird Detector: Training-Set Learning Curve")
    ax.set_xticks(sizes)
    ax.grid(alpha=.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out/"learning_curve_map50.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    ax.plot(sizes,[r["selected_FP"] for r in results],"o-",label="False-positive boxes")
    ax.plot(sizes,[r["selected_FN"] for r in results],"s-",label="Missed birds")
    ax.set_xlabel("Number of training images")
    ax.set_ylabel("Count on common validation images (SVM margin ≥ 0)")
    ax.set_title("False Detections and Missed Birds vs Training Size")
    ax.set_xticks(sizes);ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/"learning_curve_errors.png",dpi=180);plt.close(fig)

def study(root,out,sizes,val_limit,rounds,max_peaks):
    out.mkdir(parents=True,exist_ok=True)
    total_start=time.perf_counter()
    all_train=data(root,"train",0)
    all_val=data(root,"val",val_limit)
    sizes=sorted(set(min(int(n),len(all_train)) for n in sizes))
    rng=random.Random(42)
    # NESTED training subsets so all learning-curve points are strictly comparable.
    ordered=rng.sample(all_train,len(all_train))
    chosen=ordered[:max(sizes)]
    print("SPLITS",json.dumps(dict(train_sizes=sizes,validation_images=len(all_val),
      validation_birds=sum(len(gt) for _,gt in all_val),total_train=len(all_train),
      train_seed=42,validation_seed=43)),flush=True)
    valbank=validation_bank(all_val,max_peaks=max_peaks)
    trainbank=pick_training(chosen,max_peaks=max_peaks)
    all_results=[];selected=[]
    for size in sizes:
        history=train_size(trainbank,size,valbank,rounds=rounds)
        all_results.extend(history)
        # Use same fixed 3-round protocol as earlier experiment, not an
        # optimised per-size mining-round selection on VAL.
        last=history[-1]
        selected.append(dict(train_images=size,
          selected_round=rounds,
          selected_map50=last["map50_coco101"],
          selected_FP=last["FP_total_margin0"],
          selected_FN=last["FN_margin0"],
          selected_mean_final_boxes=last["mean_final_boxes_margin0"],
          positive_train_patches=last["train_positive_features"],
          negative_train_patches=last["train_negative_features"]))
        # Save intermediate records to disk to avoid losing all output on timeout.
        (out/"progress.json").write_text(json.dumps(dict(selected=selected,history=all_results),indent=2))
        plot_curve(selected,out)
    with (out/"learning_curve.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=selected[0].keys())
        writer.writeheader();writer.writerows(selected)
    with (out/"learning_curve_all_rounds.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=all_results[0].keys())
        writer.writeheader();writer.writerows(all_results)
    result=dict(primary_metric="COCO 101-point single-class bbox mAP@IoU=0.50, maxDets100",
      train_sizes=sizes,train_seed=42,val_seed=43,
      val_images=len(all_val),val_birds=sum(len(gt) for _,gt in all_val),
      candidate_generator="LoG + local Otsu (image-only)",
      features="HOG + spatial uniform LBP",
      classifier="linear SVM balanced C=0.01",
      mining_rounds=rounds,
      test_split_used=False,
      warnings=["Training patch subsampling is bounded for cloud CPU/RAM, differing from older E14 runs.",
                "Validation is shared and reused; final generalisation must be assessed on untouched test.",
                "Incorrect-box counts at SVM margin>=0, not same operating point as mAP."],
      wall_seconds=time.perf_counter()-total_start,
      learning_curve=selected,all_rounds=all_results)
    (out/"learning_curve_results.json").write_text(json.dumps(result,indent=2))
    print("LEARNING CURVE RESULTS",json.dumps(selected),flush=True)

def tests():
    boxes=np.array([[0,0,10,10],[20,20,30,30]],np.float32)
    gt=np.array([[0,0,10,10]],np.float32)
    p,n=candidate_labels(boxes,gt)
    assert p.tolist()==[0] and n.tolist()==[1]
    assert len(features(np.zeros((64,64),np.uint8),True))==1924
    print("PASS: learning-curve label and feature checks",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("curve_results"))
    p.add_argument("--train-sizes",type=int,nargs="+",default=[35,100,300,700,1369])
    p.add_argument("--val-limit",type=int,default=60)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--max-peaks",type=int,default=120)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args();tests()
    if not args.self_test:
        if not args.dataset:p.error("--dataset required")
        study(args.dataset,args.output,args.train_sizes,args.val_limit,args.rounds,args.max_peaks)
