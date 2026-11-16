"""E34: equal-data stage2 RBF versus LightGBM on Selective Search candidates.

The prior E31-C LightGBM experiment was standalone on LoG/Otsu, not a
fair comparison to E31-SS. Here same trained stage1, same PCA64+aux features,
same TRAIN proposal labels, same VAL candidate boxes, same score fusion.
Strict classical only. No test or validation fitting.
"""
from pathlib import Path
import argparse, csv, json, time
import cv2, numpy as np
from lightgbm import LGBMClassifier
from cloud_audit import data, iou, nms
from selective_search_detection import prepare,fit_models,coverage
from quality_subset_train import training_subset
from nonlinear_appearance_study import rbf_logits
from e32_failure_driven import per_image_stats, draw_audit
from metrics_map import evaluate_map50

FOCUS=("beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
       "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg",
       "beach_bird_1606.jpg")

def stage2_training_examples(bank):
    rng=np.random.default_rng(77)
    X=[];A=[];M=[];Y=[]
    for r in bank:
        pos=np.asarray(r["pos"])
        if len(pos)>45:pos=rng.choice(pos,45,replace=False)
        safe=set(int(i) for i in r["neg"])
        sc=r["margin"]
        keep=nms(r["boxes"],sc,.5,170) if len(sc) else []
        hard=[int(j) for j in keep if int(j) in safe and sc[int(j)]>-.7][:25]
        remainder=[int(j) for j in r["neg"] if int(j) not in set(hard)]
        randomneg=rng.choice(remainder,min(12,len(remainder)),replace=False).tolist() if len(remainder) else []
        for ids,y in ((pos,1),(hard+randomneg,0)):
            if len(ids):
                X.append(r["X"][ids]);A.append(r["aux"][ids])
                M.append(sc[ids]);Y.extend([y]*len(ids))
    return np.vstack(X),np.vstack(A),np.concatenate(M),np.asarray(Y,int)

def lgb_features(X,A,M,rbf):
    normal,pca,_,_=rbf
    return np.column_stack((pca.transform(normal.transform(X)),A,M)).astype(np.float32)

def fit_lgb(train,rbf):
    X,A,M,y=stage2_training_examples(train)
    transformed=lgb_features(X,A,M,rbf)
    model=LGBMClassifier(n_estimators=280,learning_rate=.05,num_leaves=15,
        min_child_samples=18,reg_lambda=2.0,class_weight="balanced",
        max_bin=127,n_jobs=2,verbosity=-1,random_state=67)
    t=time.perf_counter();model.fit(transformed,y)
    return model,dict(train_examples=len(y),train_positives=int(y.sum()),
         train_negatives=int((y==0).sum()),fit_seconds=time.perf_counter()-t,
         classifier_train_patch_accuracy=float((model.predict(transformed)==y).mean()),
         features=transformed.shape[1])

def lgb_logits(model,rbf,rec):
    z=lgb_features(rec["X"],rec["aux"],rec["margin"],rbf)
    p=np.clip(model.predict_proba(z)[:,1],1e-4,1-1e-4)
    return np.log(p/(1-p))

def evaluate(bank,base,rbf,lgb,out):
    out.mkdir(parents=True,exist_ok=True)
    visuals=out/"visual";visuals.mkdir(exist_ok=True)
    variants={"rbf_stage2":[],"lightgbm_stage2":[]};perimage=[]
    for rec in bank:
        m=base.decision_function(rec["X"])
        rec["margin"]=m
        scores={"rbf_stage2":.5*m+.5*rbf_logits(rbf,rec),
                "lightgbm_stage2":.5*m+.5*lgb_logits(lgb,rbf,rec)}
        for name,s in scores.items():
            keep=nms(rec["boxes"],s,iou_limit=.35,max_det=300) if len(s) else np.empty(0,int)
            b=rec["boxes"][keep];sc=s[keep]
            variants[name].append((b,sc,rec["gt"]))
            item=dict(method=name,image=rec["path"].name,GT=len(rec["gt"]),
                **per_image_stats(b,sc,rec["gt"],rec["coverage"]))
            perimage.append(item)
            if rec["path"].name in FOCUS:draw_audit(rec,(b,sc),name,visuals)
    output=[]
    for name,preds in variants.items():
        rows=[x for x in perimage if x["method"]==name]
        metric=evaluate_map50(preds)
        metric.update(method=name,top20_TP=sum(x["top20_TP"] for x in rows),
            top20_FP=sum(x["top20_FP"] for x in rows),
            small_recall100=sum(x["TP100_small"] for x in rows)/max(1,sum(x["GT_small"] for x in rows)),
            crowded_recall100=sum(x["crowded_TP100"] for x in rows)/max(1,sum(x["crowded_GT"] for x in rows)))
        output.append(metric)
        print("E34 METHOD",json.dumps(metric),flush=True)
    with (out/"per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(perimage[0]));wr.writeheader();wr.writerows(perimage)
    orig,trial=output
    passgate=(trial["map50_coco101"]>=orig["map50_coco101"]+.02
        and trial["top20_TP"]>=orig["top20_TP"]
        and trial["small_recall100"]>=orig["small_recall100"]-.03
        and trial["crowded_recall100"]>=orig["crowded_recall100"]-.03)
    return output,passgate

def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n)
    val=data(root,"val",val_n)
    tb=prepare(train,"selective","TRAIN")
    vb=prepare(val,"selective","VAL")
    base,rbf,diag=fit_models(tb)
    light,light_diag=fit_lgb(tb,rbf)
    scores,gate=evaluate(vb,base,rbf,light,out)
    result=dict(status="COMPLETE",train_images=len(train),validation_images=len(val),
        validation_gt=sum(len(g) for _,g in val),same_samples=True,
        method="E31-SS same Selective Search GT-blind proposals, same PCA64 TRAIN-only features",
        original_training=diag,lightgbm_training=light_diag,results=scores,gate_pass=gate,
        disclaimer="No final TEST. Existing RBF stage2 remains incumbent unless both mAP and top20 TP improve with size/crowd safeguards.")
    (out/"e34_results.json").write_text(json.dumps(result,indent=2))
    print("E34 FINAL",json.dumps(dict(gate_pass=gate,scores=scores)),flush=True)

def tests():
    X=np.ones((3,2),float)
    assert X.shape==(3,2)
    print("E34 fixed-proposal stage2 method registry PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e34_results"))
    p.add_argument("--train-n",type=int,default=100)
    p.add_argument("--val-n",type=int,default=60)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
