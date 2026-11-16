"""Controlled classical SVM mining ablation: global vs diverse vs positive-aware vs bounded.

This is an exploratory TRAIN / VAL study using a shared LoG+Otsu proposal bank
and HOG+LBP descriptors. It must never use validation labels for training.
For each round: train on TRAIN only, evaluate official COCO 101-point mAP@0.5
on VAL, and save FP/FN taxonomy. Selection based on validation is preliminary:
hold out untouched TEST for a single final evaluation.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data, iou, nms
from classical_methods_study import classification_errors
from metrics_map import evaluate_map50
from otsu_hog_lbp_map import prepare,labels

def make_index(bank):
    """Global training index with image, GT and proposal-size metadata."""
    X=np.concatenate([r["X"] for r in bank]).astype(np.float32)
    positives=[];negative=[];per_image=[];image_ids=[];boxes=[];offset=0
    for image_id,r in enumerate(bank):
        p,n=labels(r["boxes"],r["gt"])
        positives.extend((p+offset).tolist())
        negative.extend((n+offset).tolist())
        per_image.append({"pos":p+offset,"neg":n+offset,"start":offset,"end":offset+len(r["X"])})
        image_ids.extend([image_id]*len(r["X"]))
        boxes.append(r["boxes"])
        offset+=len(r["X"])
    return dict(X=X,positive=np.asarray(positives,dtype=int),
                negative=np.asarray(negative,dtype=int),
                image_id=np.asarray(image_ids,dtype=int),boxes=np.vstack(boxes),per_image=per_image)

def initial_negatives(meta,seed=44,per_image=20):
    rng=np.random.default_rng(seed)
    result=set()
    for rec in meta["per_image"]:
        neg=rec["neg"]
        if len(neg):
            result.update(int(i) for i in rng.choice(neg,min(per_image,len(neg)),replace=False))
    return result

def bbox_bucket(box):
    a=max(1.,float((box[2]-box[0])*(box[3]-box[1])))
    return int(np.clip(np.floor(np.log2(a)/3),0,5))

def choose_hard(meta,scored,selected,mode,limit=500):
    """Diverse negatives must differ by image / area bucket / overlapping region."""
    pool=meta["negative"]
    candidates=pool[np.argsort(-scored[pool],kind="stable")]
    if mode=="baseline":
        return [int(x) for x in candidates if int(x) not in selected][:limit]
    # Balanced representation: ≤35 new samples / image and ≤12 per image-scale.
    per_img={};per_scale={};accepted=[];accepted_by_image={}
    for global_idx in candidates:
        idx=int(global_idx)
        if idx in selected:continue
        image=int(meta["image_id"][idx])
        bucket=bbox_bucket(meta["boxes"][idx])
        if per_img.get(image,0)>=35 or per_scale.get((image,bucket),0)>=12:continue
        older=accepted_by_image.setdefault(image,[])
        if older and np.max(iou(meta["boxes"][idx:idx+1],meta["boxes"][older]))>=.5:
            continue
        accepted.append(idx);older.append(idx)
        per_img[image]=per_img.get(image,0)+1
        per_scale[(image,bucket)]=per_scale.get((image,bucket),0)+1
        if len(accepted)>=limit:break
    return accepted

def reduce_cache(meta,model,selected,max_negatives=1400,hard_fraction=.7,seed=42):
    """Keep informative hard negatives plus diverse low-score background.
    Never use val predictions or labels for cache pruning.
    """
    if len(selected)<=max_negatives:return set(selected)
    ids=np.asarray(sorted(selected),dtype=int)
    scores=model.decision_function(meta["X"][ids])
    count_hard=min(len(ids),round(max_negatives*hard_fraction))
    highest=ids[np.argsort(-scores,kind="stable")[:count_hard]]
    remainder=np.setdiff1d(ids,highest,assume_unique=False)
    # choose remainder reproducibly, evenly across contributing images
    rng=np.random.default_rng(seed)
    reserve=[];by_image={}
    for ind in remainder:
        by_image.setdefault(int(meta["image_id"][ind]),[]).append(int(ind))
    while len(reserve)<max_negatives-len(highest) and by_image:
        for k in sorted(list(by_image)):
            if not by_image[k]:del by_image[k];continue
            ix=int(rng.integers(len(by_image[k])))
            reserve.append(by_image[k].pop(ix))
            if len(reserve)>=max_negatives-len(highest):break
    return set(highest.tolist()+reserve)

def fit_model(meta,negatives,hard_positive=False,prior=None):
    X=meta["X"];pos=meta["positive"]
    n=np.asarray(sorted(negatives),dtype=int)
    idx=np.r_[pos,n]; y=np.r_[np.ones(len(pos)),np.zeros(len(n))]
    sample_weight=np.ones(len(idx),dtype=float)
    weak_count=0
    if hard_positive and prior is not None:
        # Boost difficult *existing positive training proposals* by SVM margin,
        # never GT from validation and never labels in ambiguous IoU range.
        ps=prior.decision_function(X[pos])
        weak=np.argsort(ps,kind="stable")[:min(120,len(ps))]
        sample_weight[weak]=2.5
        weak_count=len(weak)
    model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",
        max_iter=10000,random_state=42))
    model.fit(X[idx],y,linearsvc__sample_weight=sample_weight)
    return model,dict(positive_examples=len(pos),negative_examples=len(n),
                       weak_positive_weighted=weak_count,
                       train_patch_accuracy=float((model.predict(X[idx])==y).mean()))

def evaluate_round(model,valbank):
    predictions=[];rows=[]
    for r in valbank:
        started=time.perf_counter()
        scores=model.decision_function(r["X"])
        keep=nms(r["boxes"],scores,iou_limit=.35,max_det=300)
        b=r["boxes"][keep];s=scores[keep]
        predictions.append((b,s,r["gt"]))
        active=s>=0
        fp=classification_errors(b[active],s[active],r["gt"])
        rows.append(dict(image=r["path"],GT=len(r["gt"]),candidate_boxes=len(r["boxes"]),
              nms_boxes=len(keep),margin0_boxes=int(active.sum()),
              scoring_seconds=time.perf_counter()-started,**fp))
    metrics=evaluate_map50(predictions)
    metrics.update(mean_output_boxes=float(np.mean([r["margin0_boxes"] for r in rows])),
        false_positive_total=int(sum(r["FP_total"] for r in rows)),
        false_positive_background=int(sum(r["FP_background"] for r in rows)),
        false_positive_localisation=int(sum(r["FP_localization"] for r in rows)),
        false_positive_duplicate=int(sum(r["FP_duplicates"] for r in rows)),
        missed_birds=int(sum(r["FN"] for r in rows)),
        empty_scene_false_positive=int(sum(r["FP_total"] for r in rows if not r["GT"])),
        mean_classifier_seconds=float(np.mean([r["scoring_seconds"] for r in rows])))
    return metrics,rows

def study(dataset,out,train_limit,val_limit,rounds):
    out.mkdir(parents=True,exist_ok=True)
    train=data(dataset,"train",train_limit)
    val=data(dataset,"val",val_limit)
    trainbank=prepare(train,"train",120)
    valbank=prepare(val,"val",120)
    meta=make_index(trainbank)
    if not len(meta["positive"]) or not len(meta["negative"]):
        raise RuntimeError("no trainable class examples")
    variants=[
      ("A_global_mining", "baseline", False, False),
      ("B_diverse_negatives", "diverse", False, False),
      ("C_diverse_hard_positives", "diverse", True, False),
      ("D_bounded_balanced_cache", "diverse", True, True)]
    all_results=[];image_rows=[]
    for name,mode,use_hard_pos,use_bounded in variants:
        selected=initial_negatives(meta)
        history=[];prev=None
        for iteration in range(rounds+1):
            start=time.perf_counter()
            model,details=fit_model(meta,selected,use_hard_pos,prev)
            fit_seconds=time.perf_counter()-start
            evaluation,per_image=evaluate_round(model,valbank)
            mined=choose_hard(meta,model.decision_function(meta["X"]),selected,mode)
            row=dict(variant=name,round=iteration,**details,**evaluation,
                     new_hard_negatives=len(mined),fit_seconds=fit_seconds,
                     candidate_boxes_per_image=float(np.mean([len(r["boxes"]) for r in valbank])))
            history.append(row)
            image_rows.extend([dict(variant=name,round=iteration,**r) for r in per_image])
            print("MINING ABLATION ROUND",json.dumps(row),flush=True)
            if iteration<rounds:
                selected.update(mined)
                if use_bounded:selected=reduce_cache(meta,model,selected,max_negatives=1400,
                                    hard_fraction=.7,seed=42+iteration)
                prev=model
        best=max(history,key=lambda r:r["map50_coco101"])
        all_results.append(dict(variant=name,
           best_validation_round=best["round"],best_validation_map50=best["map50_coco101"],
           last_validation_map50=history[-1]["map50_coco101"],history=history))
        print("MINING ABLATION BEST",name,best["round"],best["map50_coco101"],flush=True)
    result=dict(dataset="Beach Bird",train_images=len(train),val_images=len(val),
        val_birds=sum(len(r["gt"]) for r in valbank),seed=42,
        proposals="LoG + local Otsu (image only)",
        descriptor="HOG + spatial uniform LBP",
        AP_metric="COCO 101-point single-class bbox mAP@IoU 0.50 maxDets100",
        notes=["Validation chosen for exploratory model selection; test split untouched.",
               "FP and missed-bird diagnostics use margin>=0, not all scores used for AP.",
               "Mining and weighting use TRAIN labels only.",
               "All runs share identical feature banks and train/val images."],
        variants=all_results)
    (out/"mining_ablation.json").write_text(json.dumps(result,indent=2))
    with (out/"mining_by_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=image_rows[0]);writer.writeheader();writer.writerows(image_rows)
    print("MINING ABLATION SUMMARY",json.dumps([dict(variant=v["variant"],
       best_round=v["best_validation_round"],mAP50=v["best_validation_map50"],
       last_mAP50=v["last_validation_map50"]) for v in all_results]),flush=True)

def unit_tests():
    a=np.array([[0,0,12,12],[0,0,12,12],[40,40,50,50]],float)
    m=dict(X=np.zeros((3,4)),negative=np.array([0,1,2]),
      boxes=a,image_id=np.array([0,0,1]))
    assert len(choose_hard(m,np.array([3.,2.,1.]),set(),"baseline",2))==2
    chosen=choose_hard(m,np.array([3.,2.,1.]),set(),"diverse",3)
    assert len(chosen)==2 and set(chosen)=={0,2}
    print("PASS: diverse-negative mining overlap and balancing checks")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("mining_results"))
    p.add_argument("--train-limit",type=int,default=35)
    p.add_argument("--val-limit",type=int,default=12)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();unit_tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        study(a.dataset,a.output,a.train_limit,a.val_limit,a.rounds)
