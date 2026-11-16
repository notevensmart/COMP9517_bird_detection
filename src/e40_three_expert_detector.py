"""E40: complete strictly classical 3-expert spatial bird detector.
The RBF baseline and boosted arms share deterministic SS proposals and TRAIN
images. Image-disjoint TRAIN calibration; GT is never used at VAL inference.
TEST is not opened (previously assessed in E33).
"""
import argparse, csv, json, time
from pathlib import Path
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from cloud_audit import data, iou, nms
from quality_subset_train import training_subset
from selective_search_detection import prepare, fit_models, coverage
from nonlinear_appearance_study import rbf_logits
from e39_scale_aware_boosting import features_for_bank, collect_training, lightgbm_model
from e32_failure_driven import target_offsets, apply_refiner, per_image_stats, draw_audit
from metrics_map import evaluate_map50

SEED=9440
BINS=("small","medium","large")
FOCUS={"beach_bird_1379.jpg","beach_bird_1389.jpg","beach_bird_1399.jpg",
       "beach_bird_1419.jpg","beach_bird_1435.jpg","beach_bird_1452.jpg",
       "beach_bird_1560.jpg","beach_bird_1606.jpg"}
MODES=("rbf_baseline","raw_three_experts","calibrated_three_experts",
       "calibrated_refined","calibrated_refined_crowd_soft_nms")

def scales(boxes):
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    a=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)
    return np.where(a<1024,0,np.where(a<10000,1,2)).astype(np.int32)

def logit_probs(probs):
    p=np.clip(np.asarray(probs,float),1e-5,1-1e-5)
    return np.log(p/(1-p))

def learner(seed):
    # Frozen capacity for the experiment; no VAL-driven hyperparameter search.
    return lightgbm_model(seed)

def train_experts(train):
    X,y,areas,train_counts=collect_training(train,train[0]["stage1"])
    # area bins rely on CANDIDATE dimensions (not GT size at inference)
    b=np.where(areas<1024,0,np.where(areas<10000,1,2))
    models={};counts={}
    for i,name in enumerate(BINS):
        mask=b==i
        c=np.bincount(y[mask],minlength=2)
        counts[name]=dict(positive=int(c[1]),negative=int(c[0]))
        if min(c)<20:raise RuntimeError(f"Insufficient TRAIN supervision for {name}: {c}")
        models[name]=learner(SEED+i)
        models[name].fit(X[mask],y[mask])
    return models,dict(training=train_counts,by_scale=counts,feature_dim=X.shape[1])

def expert_raw(rec,models):
    if not len(rec["boxes"]):return np.empty(0,float)
    ids=scales(rec["boxes"])
    X=rec["boost_X"]
    scores=np.zeros(len(X),float)
    for k,name in enumerate(BINS):
        where=np.flatnonzero(ids==k)
        if len(where):
            probs=models[name].predict_proba(X[where])[:,1]
            scores[where]=logit_probs(probs)
    if not np.isfinite(scores).all():raise ValueError("Invalid expert score")
    return scores

def collect_calibration(cal,models,base):
    # This is a separate image partition from all expert/classifier fitting.
    # Choose positives and negatives only from the TRAIN calibration images.
    rng=np.random.default_rng(44)
    xs={k:[] for k in BINS};ys={k:[] for k in BINS}
    stats={k:{"positive":0,"negative":0} for k in BINS}
    for rec in cal:
        if not len(rec["boxes"]):continue
        scores=expert_raw(rec,models)
        ids=scales(rec["boxes"])
        for k,name in enumerate(BINS):
            pos=np.intersect1d(np.flatnonzero(ids==k),rec["pos"])
            neg=np.intersect1d(np.flatnonzero(ids==k),rec["neg"])
            if len(pos)>70:pos=rng.choice(pos,70,replace=False)
            # Retain hard and randomly selected GT-safe background samples.
            order=neg[np.argsort(-scores[neg],kind="stable")]
            hard=order[:min(45,len(order))]
            others=order[len(hard):]
            extra=rng.choice(others,min(25,len(others)),replace=False) if len(others) else np.empty(0,dtype=np.int64)
            for index,label in [(pos,1),(np.r_[hard,extra].astype(np.int64),0)]:
                if len(index):
                    xs[name].extend(scores[index].tolist())
                    ys[name].extend([label]*len(index))
                    stats[name]["positive" if label else "negative"]+=len(index)
    calibrators={}
    for name in BINS:
        x=np.asarray(xs[name],float).reshape(-1,1)
        y=np.asarray(ys[name],int)
        if min(np.bincount(y,minlength=2))<10:
            raise RuntimeError(f"Not enough image-disjoint calibration examples for {name}: {stats[name]}")
        calibrators[name]=LogisticRegression(C=1.0,max_iter=1000,random_state=SEED)
        calibrators[name].fit(x,y)
    return calibrators,stats

def calibrated_score(rec,models,calibrators):
    raw=expert_raw(rec,models)
    ids=scales(rec["boxes"]);out=raw.copy()
    for k,name in enumerate(BINS):
        at=np.flatnonzero(ids==k)
        if len(at):
            p=calibrators[name].predict_proba(raw[at,None])[:,1]
            out[at]=logit_probs(p)
    return out

def refine_features(rec,confidence):
    b=rec["boxes"]
    if not len(b):return np.empty((0,rec["boost_X"].shape[1]+6),np.float32)
    w=np.maximum(1,b[:,2]-b[:,0]);h=np.maximum(1,b[:,3]-b[:,1])
    geom=np.column_stack((np.log1p(w),np.log1p(h),np.log(w/h),
                          (b[:,0]+b[:,2])/1024,(b[:,1]+b[:,3])/1024))
    X=np.column_stack((rec["boost_X"],geom,confidence))
    if not np.isfinite(X).all():raise ValueError("Refinement features must be finite")
    return X.astype(np.float32)

def train_regressor(core,experts,calibrators):
    rng=np.random.default_rng(47)
    xx=[];yy=[];scale_counts={key:0 for key in BINS}
    for rec in core:
        b=rec["boxes"];gt=rec["gt"]
        if not len(b) or not len(gt):continue
        overlaps=iou(b,gt)
        match=overlaps.argmax(axis=1);best=overlaps[np.arange(len(b)),match]
        positive=np.flatnonzero((best>=.35)&(best<.95))
        if len(positive)>100:positive=rng.choice(positive,100,replace=False)
        if not len(positive):continue
        score=calibrated_score(rec,experts,calibrators)
        xx.append(refine_features(rec,score)[positive])
        yy.append(target_offsets(b[positive],gt[match[positive]]))
        for j in scales(b[positive]):scale_counts[BINS[int(j)]]+=1
    if not xx:raise RuntimeError("No TRAIN-only regression targets")
    X=np.vstack(xx);Y=np.vstack(yy)
    # Multi-output classical box regression; train on the SAME core, not VAL.
    model=ExtraTreesRegressor(n_estimators=80,max_depth=12,
        min_samples_leaf=4,n_jobs=2,random_state=SEED)
    model.fit(X,Y)
    return model,dict(samples=len(X),by_proposal_scale=scale_counts,features=X.shape[1])

def crowd_soft_nms(boxes,scores,capacity=300):
    """Score-decaying soft suppression, gentler for neighbouring small boxes.
    GT-free and fixed thresholds. Return the final re-ranked box indexes/scores.
    The COCO evaluator subsequently restricts scoring to maxDets100.
    """
    b=np.asarray(boxes,np.float32).reshape(-1,4)
    scores=np.asarray(scores,float).copy()
    if not len(b):return np.empty(0,int),np.empty(0,float)
    # Restrict computational cost while preserving the strongest raw scores.
    pool=np.argsort(-scores,kind="stable")[:min(320,len(scores))]
    active=scores[pool].copy()
    area_bin=scales(b[pool])
    selected=[];values=[]
    while len(pool) and len(selected)<capacity:
        best=int(np.argmax(active));idx=int(pool[best]);value=float(active[best])
        if not np.isfinite(value):raise ValueError("Nonfinite Soft NMS score")
        selected.append(idx);values.append(value)
        other=np.arange(len(pool))!=best
        remaining=pool[other];rest=active[other]
        if not len(remaining):break
        overlap=iou(b[idx:idx+1],b[remaining])[0]
        # Small/small neighbours can represent distinct birds. Soften overlap
        # penalty only there; otherwise use conventional Gaussian suppression.
        both_small=(scales(b[remaining])==0)&(scales(b[idx:idx+1])[0]==0)
        sigma=np.where(both_small,.65,.35)
        decay=np.exp(-(overlap**2)/sigma)
        # Above .12 only; leave nonoverlapping proposals' ordering unchanged.
        rest=rest-np.where(overlap>.12,-np.log(np.clip(decay,1e-8,1)),0)
        pool=remaining
        active=rest
    order=np.argsort(-np.asarray(values),kind="stable")
    return np.asarray(selected,int)[order],np.asarray(values,float)[order]

def evaluate(val,base,rbf,experts,calibrators,refiner,out):
    visual=out/"visual";visual.mkdir(parents=True,exist_ok=True)
    preds={k:[] for k in MODES};rows=[]
    for rec in val:
        b=rec["boxes"]
        if len(b):
            rec["margin"]=base.decision_function(rec["X"])
            old=.5*rec["margin"]+.5*rbf_logits(rbf,rec)
            raw=expert_raw(rec,experts)
            cal=calibrated_score(rec,experts,calibrators)
            corrected=apply_refiner(b,refine_features(rec,cal),refiner)
        else:
            old=np.empty(0);raw=old;cal=old;corrected=b
        for mode in MODES:
            boxes=corrected if "refined" in mode else b
            scores=old if mode=="rbf_baseline" else raw if mode=="raw_three_experts" else cal
            if mode.endswith("soft_nms"):
                keep,finalscores=crowd_soft_nms(boxes,scores)
            else:
                keep=nms(boxes,scores,.35,300) if len(boxes) else np.empty(0,int)
                finalscores=scores[keep]
            outboxes=boxes[keep]
            preds[mode].append((outboxes,finalscores,rec["gt"]))
            rows.append(dict(method=mode,image=rec["path"].name,
                gt=len(rec["gt"]),candidates=len(b),
                **per_image_stats(outboxes,finalscores,rec["gt"],rec["coverage"])))
            if rec["path"].name in FOCUS:
                draw_audit(rec,(outboxes,finalscores),mode,visual)
    results=[]
    for name in MODES:
        matched=[r for r in rows if r["method"]==name]
        stats=evaluate_map50(preds[name])
        stats.update(method=name,top20_TP=sum(r["top20_TP"] for r in matched),
            top20_FP=sum(r["top20_FP"] for r in matched),
            small_top100_recall=sum(r["TP100_small"] for r in matched)/max(1,sum(r["GT_small"] for r in matched)),
            crowded_top100_recall=sum(r["crowded_TP100"] for r in matched)/max(1,sum(r["crowded_GT"] for r in matched)))
        results.append(stats)
        print("E40 METRIC",json.dumps(stats),flush=True)
    with (out/"e40_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)
    incumbent=results[0];gates={}
    for entry in results[1:]:
        gates[entry["method"]]=(entry["map50_coco101"]>=incumbent["map50_coco101"]+.02
            and entry["top20_TP"]>=incumbent["top20_TP"]
            and entry["small_top100_recall"]>=incumbent["small_top100_recall"]-.03
            and entry["crowded_top100_recall"]>=incumbent["crowded_top100_recall"]-.03)
    return results,gates

def self_test():
    im=np.zeros((80,128,3),np.uint8)
    from e39_scale_aware_boosting import normalized_features,letterbox
    patch=np.full((20,70,3),120,np.uint8)
    image,mask,original=letterbox(patch)
    assert image.shape==(64,64,3) and mask.sum()>0 and original==(20,70)
    assert np.isfinite(normalized_features(im,[0,0,50,40])).all()
    boxes=np.array([[0,0,12,12],[1,1,13,13],[40,40,60,60]],np.float32)
    scores=np.array([3.,2.8,1.1])
    ii,ss=crowd_soft_nms(boxes,scores)
    assert len(ii)==3 and np.array_equal(np.sort(ii),[0,1,2])
    assert np.isfinite(ss).all() and np.all(ss[:-1]>=ss[1:])
    assert scales(boxes).tolist()==[0,0,0]
    test=refine_features(dict(boxes=boxes,boost_X=np.ones((3,176),np.float32)),scores)
    assert test.shape==(3,182)
    print("E40 PREPROCESS/CROWD NMS/GEOMETRY self tests PASS",flush=True)

def run(dataset,out,train_n=60,cal_n=15,val_n=30):
    if train_n<=cal_n or cal_n<10:raise ValueError("Too few TRAIN-only calibration images")
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(dataset,train_n)
    # Image-disjoint, deterministic TRAIN split. CAL data only fits calibration.
    rng=np.random.default_rng(SEED)
    choice=set(rng.choice(len(train),cal_n,replace=False).tolist())
    core=[s for i,s in enumerate(train) if i not in choice]
    cal=[s for i,s in enumerate(train) if i in choice]
    val=data(dataset,"val",val_n)
    assert not set(p.name for p,_ in core)&set(p.name for p,_ in cal)
    assert not set(p.name for p,_ in val)&set(p.name for p,_ in train)
    print("E40 TRAIN CORE/CAL/VAL",len(core),len(cal),len(val),flush=True)
    started=time.perf_counter()
    trainbank=prepare(core,"selective","TRAIN_CORE")
    calbank=prepare(cal,"selective","TRAIN_CAL_ONLY")
    valbank=prepare(val,"selective","VAL")
    base,rbf,baseline_train=fit_models(trainbank)
    for rec in trainbank:rec["stage1"]=base
    for bank,phase in ((trainbank,"CORE"),(calbank,"CAL"),(valbank,"VAL")):
        features_for_bank(bank,out,phase)
    experts,diag=train_experts(trainbank)
    calibrators,calstat=collect_calibration(calbank,experts,base)
    refiner,refstat=train_regressor(trainbank,experts,calibrators)
    results,gate=evaluate(valbank,base,rbf,experts,calibrators,refiner,out)
    report=dict(status="E40_COMPLETED_VAL_ONLY",train_core=len(core),
        train_calibration=len(cal),val_images=len(val),
        val_GT=sum(len(gt) for _,gt in val),test_access=False,
        core_images=[p.name for p,_ in core],cal_images=[p.name for p,_ in cal],
        val_images_used=[p.name for p,_ in val],
        preprocess="176D masked aspect-preserving HOG-like gradients+HSV/LAB/LBP/context; E39 verified",
        binning="GT-blind proposal area <1024, [1024,10000), >=10000 px^2",
        calibration="TRAIN-only disjoint calibration images, logistic per expert",
        refinement="ExtraTrees 4D bbox offsets using TRAIN-core IoU .35-.95, confidence and geometry",
        soft_nms="Gaussian overlap score decay, gentler for two small proposals; fixed constants",
        train_stats=diag,calibration_stats=calstat,regression_stats=refstat,
        baseline_train=baseline_train,proposal_coverage=coverage(valbank),
        results=results,promote_gates=gate,elapsed_seconds=time.perf_counter()-started,
        caveats=["VAL only; previously used official TEST must not be reused for model selection",
                 "Three trained experts do not imply higher mAP: require positive validation evidence",
                 "Calibration images are disjoint from classifier fitting; raw proposals shared",
                 "Missing labels can contaminate safe negative proposals",
                 "Higher patch accuracy alone is not detection improvement"])
    (out/"e40_report.json").write_text(json.dumps(report,indent=2))
    print("E40 COMPLETE",json.dumps(dict(gates=gate,results=results)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e40_results"))
    p.add_argument("--train-n",type=int,default=60)
    p.add_argument("--cal-n",type=int,default=15)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();self_test()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.cal_n,a.val_n)
