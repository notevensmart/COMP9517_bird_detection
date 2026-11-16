"""E31: paired classical ML comparison. Never use TEST or validation labels for fitting.
Fixed LoG/Otsu proposals, HOG+LBP, GT+jitter positives, safe negatives,
NMS=.35, COCO 101-point bbox AP50 maxDets100. No mining or tuning.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from cloud_audit import data, iou, nms
from quality_subset_train import training_subset, gt_positive_features, sample_initial
from rebuild_detector import build_bank
from metrics_map import evaluate_map50, self_test as metric_tests
from e31_visual_audit import audit as visual_audit, tests as visual_tests

def models():
    return {
        "linear_svm":make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=10000,random_state=61)),
        "rbf_svm":make_pipeline(StandardScaler(),SVC(C=1,gamma="scale",kernel="rbf",class_weight="balanced")),
        "random_forest":RandomForestClassifier(n_estimators=250,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=2,random_state=61),
        "lightgbm":LGBMClassifier(n_estimators=250,learning_rate=.05,num_leaves=15,min_child_samples=15,reg_lambda=2,max_bin=127,class_weight="balanced",n_jobs=2,verbosity=-1,random_state=61),
    }

def score(model,X):
    if hasattr(model,"decision_function"):return np.asarray(model.decision_function(X),float).ravel()
    return np.asarray(model.predict_proba(X)[:,1],float).ravel()

def matched_top20(preds):
    tp=fp=0; gt_size={"small":0,"medium":0,"large":0}; hits={k:0 for k in gt_size}
    crowd_gt=crowd_tp=0
    for boxes,s,gt in preds:
        areas=np.prod(gt[:,2:]-gt[:,:2],axis=1) if len(gt) else np.empty(0)
        tiers=np.where(areas<1024,"small",np.where(areas<10000,"medium","large"))
        for tier in tiers:gt_size[tier]+=1
        ix=np.argsort(-s,kind="stable")[:20]
        ov=iou(boxes[ix],gt)
        used=set()
        for row in ov:
            eligible=[(float(row[j]),j) for j in range(len(gt)) if j not in used and row[j]>=.5]
            if eligible:
                _,j=max(eligible);used.add(j);hits[tiers[j]]+=1;tp+=1
            else:fp+=1
        if len(gt)>=20:crowd_gt+=len(gt);crowd_tp+=len(used)
    return dict(top20_TP=tp,top20_FP=fp,top20_precision=tp/max(1,tp+fp),gt_by_scale=gt_size,
      top20_recall_by_scale={k:(hits[k]/v if v else None) for k,v in gt_size.items()},
      crowded_top20_recall=(crowd_tp/crowd_gt if crowd_gt else None),crowded_gt=crowd_gt)

def evaluate(model,bank):
    preds=[];timings=[];counts=[]
    for r in bank:
        t=time.perf_counter();b=r["boxes"]
        s=score(model,r["X"]) if len(b) else np.empty(0)
        if not np.isfinite(s).all():raise ValueError("Nonfinite model scores")
        keep=nms(b,s,iou_limit=.35,max_det=300) if len(b) else np.empty(0,int)
        preds.append((b[keep],s[keep],r["gt"]))
        timings.append(time.perf_counter()-t);counts.append(len(keep))
    return dict(**evaluate_map50(preds),**matched_top20(preds),
                mean_scoring_nms_s=float(np.mean(timings)),mean_post_nms=float(np.mean(counts)),
                mean_proposals=float(np.mean([len(x["boxes"]) for x in bank])))

def run(dataset,output,train_n,val_n):
    output.mkdir(parents=True,exist_ok=True)
    train=training_subset(dataset,train_n) # excludes ALL zero-GT train images
    val=data(dataset,"val",val_n)
    print("E31 BUILD SHARED TRAIN BANK",len(train),flush=True)
    trainbank=build_bank(train,"TRAIN",output)
    print("E31 BUILD SHARED VAL BANK",len(val),flush=True)
    valbank=build_bank(val,"VAL",output)
    pos=gt_positive_features(trainbank)
    neg,details=sample_initial(trainbank,"baseline",max_each=20)
    if not len(pos) or not len(neg):raise ValueError("Missing training class")
    X=np.vstack((pos,neg)).astype(np.float32)
    y=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]
    if not np.isfinite(X).all():raise ValueError("Nonfinite feature bank")
    results=[]
    trained={}
    for name,model in models().items():
        print("E31 FIT",name,"features",X.shape,flush=True)
        t=time.perf_counter();model.fit(X,y);seconds=time.perf_counter()-t
        trained[name]=model
        row=dict(method=name,train_seconds=seconds,patch_train_accuracy=float((model.predict(X)==y).mean()),
                 train_pos=len(pos),train_neg=len(neg),train_images=len(train),val_images=len(val),
                 val_birds=sum(len(gt) for _,gt in val),**evaluate(model,valbank))
        results.append(row)
        (output/"e31_partial.json").write_text(json.dumps(results,indent=2))
        print("E31 RESULT",json.dumps(row),flush=True)
    best=max(results,key=lambda r:r["map50_coco101"])
    base=next(x for x in results if x["method"]=="linear_svm")
    final=dict(experiment="E31",status="COMPLETE",dataset="Kaggle beach-bird",
       train_image_ids=[p.name for p,_ in train],val_image_ids=[p.name for p,_ in val],
       positives="TRAIN GT + jitter; IoU>=0.55",negatives="TRAIN only, IoU<.1 and containment<.5, 20 per image",
       excluded="All zero-GT TRAIN images, due to suspected annotation omissions",
       features="Original HOG+LBP 64x64 (same bank for all classifiers)",
       proposals="Original LoG/Otsu (same bank)",nms_iou=.35,hard_mining_rounds=0,
       metric="COCO 101-point single class AP50 maxDets100",results=results,
       exploratory_best=best["method"],best_delta_vs_linear_svm_pp=100*(best["map50_coco101"]-base["map50_coco101"]),
       limitations=["VAL exploratory only; TEST untouched","Classifier ranking only, no box regression",
         "SVM margins and tree probabilities are different scales; do not compare score>=0 FP",
         "E25 RandomForest was a two-stage cascade; this is standalone, not directly comparable",
         "No hyperparameter search; no extra mining","No deep learning"])
    (output/"e31_results.json").write_text(json.dumps(final,indent=2))
    with (output/"e31_table.csv").open("w",newline="") as f:
        columns=["method","map50_coco101","top20_TP","top20_FP","train_seconds","mean_scoring_nms_s"]
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r[k] for k in columns} for r in results)
    visual_audit(trained,valbank,output)
    print("E31 COMPLETE",json.dumps(dict(best=best["method"],delta_pp=final["best_delta_vs_linear_svm_pp"])),flush=True)

def tests():
    metric_tests()
    visual_tests()
    p=[(np.array([[0,0,10,10],[20,20,30,30]],float),np.array([.9,.8]),np.array([[0,0,10,10]],float))]
    v=matched_top20(p)
    assert v["top20_TP"]==1 and v["top20_FP"]==1 and len(models())==4
    print("E31 synthetic rank matching + registry PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e31_results"))
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
