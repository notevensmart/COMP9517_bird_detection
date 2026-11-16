"""E40 classical three-scale boosted experts, held-out TRAIN calibration,
TRAIN-only box refinement and crowd-aware suppression. TEST never opened.
All candidate crops use audited E39 aspect-preserving 176D features.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMClassifier
from cloud_audit import data,iou,nms
from quality_subset_train import training_subset
from selective_search_detection import prepare,fit_models
from nonlinear_appearance_study import rbf_logits
from e39_scale_aware_boosting import features_for_bank,lightgbm_model,tests as preprocess_tests,FOCUS
from e32_failure_driven import fit_refiner,regression_features,apply_refiner,per_image_stats,draw_audit
from metrics_map import evaluate_map50

SEED=9444
def band(area):
    a=np.asarray(area)
    return np.where(a<1024,0,np.where(a<10000,1,2))
def areas(boxes):
    return np.prod(np.maximum(0,boxes[:,2:]-boxes[:,:2]),axis=1)
def proposal_examples(bank,stage1):
    rng=np.random.default_rng(SEED)
    X=[];Y=[];A=[];G=[]
    for rec in bank:
        if not len(rec["boxes"]):continue
        pos=np.asarray(rec["pos"],int);neg=np.asarray(rec["neg"],int)
        if len(pos)>55:pos=rng.choice(pos,55,replace=False)
        scores=stage1.decision_function(rec["X"])
        hard=neg[np.argsort(-scores[neg],kind="stable")[:55]] if len(neg) else neg
        for ids,y in ((pos,1),(hard,0)):
            if len(ids):
                X.append(rec["boost_X"][ids]);Y.extend([y]*len(ids))
                A.extend(areas(rec["boxes"][ids]));G.extend([rec["path"].name]*len(ids))
    if not X:raise RuntimeError("No TRAIN proposals")
    return np.vstack(X),np.asarray(Y),np.asarray(A),np.asarray(G)
def logit(model,x):
    p=np.clip(model.predict_proba(x)[:,1],1e-5,1-1e-5)
    return np.log(p/(1-p))
def fit_experts(bank,stage1):
    X,y,a,groups=proposal_examples(bank,stage1)
    tr,cal=next(GroupShuffleSplit(n_splits=1,test_size=.23,random_state=SEED).split(X,y,groups))
    models={};calibrators={};diagnostics={}
    for name,mask in [("global",np.ones(len(y),bool)),("small",band(a)==0),
                      ("medium",band(a)==1),("large",band(a)==2)]:
        ti=tr[mask[tr]];ci=cal[mask[cal]]
        counts=np.bincount(y[ti],minlength=2)
        ccounts=np.bincount(y[ci],minlength=2)
        if min(counts)<20 or min(ccounts)<8:
            raise RuntimeError(f"{name}: insufficient TRAIN-only fit/calibration class balance {counts}, {ccounts}")
        model=lightgbm_model(SEED+len(models))
        model.fit(X[ti],y[ti])
        logits=logit(model,X[ci]).reshape(-1,1)
        calibrator=LogisticRegression(random_state=SEED,max_iter=300)
        calibrator.fit(logits,y[ci])
        models[name]=model;calibrators[name]=calibrator
        diagnostics[name]=dict(fit_count=len(ti),calibration_count=len(ci),
                               fit_positive=int(y[ti].sum()),calibration_positive=int(y[ci].sum()),
                               calibration_groups=int(len(np.unique(groups[ci]))),
                               coef=float(calibrator.coef_[0,0]),
                               intercept=float(calibrator.intercept_[0]))
    assert not set(groups[tr]).intersection(groups[cal])
    return models,calibrators,dict(total_samples=len(y),train_image_groups=len(set(groups)),
                                  calibration_is_image_disjoint=True,experts=diagnostics)
def score_experts(rec,models,calibrators,calibrated):
    X=rec["boost_X"];b=rec["boxes"]
    if not len(b):return np.empty(0,float)
    bg=band(areas(b))
    global_s=logit(models["global"],X)
    out=np.empty(len(X),float)
    for k,name in enumerate(("small","medium","large")):
        mask=bg==k
        if not mask.any():continue
        expert=logit(models[name],X[mask])
        if calibrated:
            g=calibrators["global"].predict_proba(global_s[mask].reshape(-1,1))[:,1]
            e=calibrators[name].predict_proba(expert.reshape(-1,1))[:,1]
            # Blend calibrated TRAIN-heldout probabilities, not raw incomparable margins.
            out[mask]=.5*g+.5*e
        else:out[mask]=.5*global_s[mask]+.5*expert
    if not np.isfinite(out).all():raise ValueError("Nonfinite expert score")
    return out
def crowd_suppress(boxes,scores,limit=300):
    """Scale-aware NMS: protect small neighbouring birds from over-suppression."""
    if not len(boxes):return np.empty(0,int)
    order=np.argsort(-scores,kind="stable")
    retained=[]
    a=areas(boxes)
    while len(order) and len(retained)<limit:
        k=int(order[0]);retained.append(k);order=order[1:]
        if not len(order):break
        overlap=iou(boxes[order],boxes[[k]]).reshape(-1)
        # Conservative threshold .50 for two small proposals, .35 otherwise.
        threshold=np.where((a[order]<1024)&(a[k]<1024),.50,.35)
        order=order[overlap<threshold]
    return np.asarray(retained,int)
def evaluate(bank,stage1,rbf_model,models,calibrators,refiner,out):
    out.mkdir(parents=True,exist_ok=True);(out/"visual").mkdir(exist_ok=True)
    methods=("rbf_035","experts_raw_035","experts_cal_035",
             "experts_cal_refine_035","experts_cal_refine_crowd")
    predictions={k:[] for k in methods};rows=[]
    for count,rec in enumerate(bank,1):
        b=rec["boxes"]
        if len(b):
            rec["margin"]=stage1.decision_function(rec["X"])
            rbf=.5*rec["margin"]+.5*rbf_logits(rbf_model,rec)
            raw=score_experts(rec,models,calibrators,False)
            cal=score_experts(rec,models,calibrators,True)
            refined=apply_refiner(b,regression_features(rec,stage1,rbf_model),refiner)
            h,w=512,512
            assert np.isfinite(refined).all() and (refined[:,2:]>refined[:,:2]).all()
            arms={"rbf_035":(b,rbf),"experts_raw_035":(b,raw),
                  "experts_cal_035":(b,cal),"experts_cal_refine_035":(refined,cal),
                  "experts_cal_refine_crowd":(refined,cal)}
        else:
            arms={k:(b,np.empty(0)) for k in methods}
        for method,(boxes,score) in arms.items():
            if method.endswith("_crowd"):idx=crowd_suppress(boxes,score)
            else:idx=nms(boxes,score,.35,300) if len(boxes) else np.empty(0,int)
            bb=boxes[idx];ss=score[idx]
            predictions[method].append((bb,ss,rec["gt"]))
            stats=per_image_stats(bb,ss,rec["gt"],rec["coverage"])
            rows.append(dict(image=rec["path"].name,method=method,GT=len(rec["gt"]),**stats))
            if rec["path"].name in FOCUS:
                draw_audit(rec,(bb,ss),method,out/"visual")
        if count%10==0:print("E40 EVAL",count,len(bank),flush=True)
    results=[]
    for method in methods:
        result=evaluate_map50(predictions[method])
        subset=[r for r in rows if r["method"]==method]
        result.update(method=method,top20_TP=sum(x["top20_TP"] for x in subset),
                      top20_FP=sum(x["top20_FP"] for x in subset),
                      small_top100_recall=sum(x["TP100_small"] for x in subset)/max(1,sum(x["GT_small"] for x in subset)),
                      crowded_top100_recall=sum(x["crowded_TP100"] for x in subset)/max(1,sum(x["crowded_GT"] for x in subset)))
        results.append(result)
        print("E40 METHOD",json.dumps(result),flush=True)
    with (out/"e40_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    return results
def tests():
    preprocess_tests()
    b=np.array([[10,10,20,20],[11,11,21,21],[30,30,40,40]],np.float32)
    sc=np.array([.9,.8,.7])
    k=crowd_suppress(b,sc)
    assert np.array_equal(k,[0,1,2]) # small neighbours allowed at IoU .50
    big=np.array([[10,10,50,50],[11,11,51,51]],np.float32)
    assert np.array_equal(crowd_suppress(big,np.array([.9,.8])),[0])
    assert np.array_equal(band(np.array([100,1024,10000])),[0,1,2])
    print("E40 PASS three expert bins, crowd NMS, E39 preprocessing",flush=True)
def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    print("E40 SPLIT",len(train),len(val),sum(len(gt) for _,gt in val),flush=True)
    tb=prepare(train,"selective","TRAIN");vb=prepare(val,"selective","VAL")
    stage1,rbf_model,diag=fit_models(tb)
    features_for_bank(tb,out,"TRAIN");features_for_bank(vb,out,"VAL")
    models,calibrators,expert_diag=fit_experts(tb,stage1)
    refiner,refine_diag=fit_refiner(tb,stage1,rbf_model)
    results=evaluate(vb,stage1,rbf_model,models,calibrators,refiner,out)
    base=results[0];gate={r["method"]:(r["map50_coco101"]>=base["map50_coco101"]+.02
                      and r["top20_TP"]>=base["top20_TP"]) for r in results[1:]}
    report=dict(status="COMPLETE",protocol="TRAIN/VAL only; TEST not used",
       train_images=len(train),val_images=len(val),val_GT=sum(len(gt) for _,gt in val),
       preprocessing="E39 aspect-preserving 176D BGR letterbox and masked spatial descriptors",
       image_disjoint_calibration=True,three_scale_experts=True,
       training=expert_diag,regression=refine_diag,incumbent_training=diag,
       results=results,gate=gate)
    (out/"e40_report.json").write_text(json.dumps(report,indent=2))
    print("E40 FINAL",json.dumps(dict(results=results,gate=gate)),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e40_results"))
    p.add_argument("--train-n",type=int,default=100)
    p.add_argument("--val-n",type=int,default=60)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
