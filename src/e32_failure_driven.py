"""E32: review dominant classical detector failures and test cautious corrections.
E31-SS Selective Search + staged PCA64 RBF is frozen. All fitting TRAIN-only.
Compare baseline hard NMS, relaxed NMS, and supervised classical bbox refinement.
VAL for evaluation only; no TEST, neural weights or target leakage.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np, cv2
from sklearn.ensemble import ExtraTreesRegressor
from cloud_audit import data, iou, nms
from quality_subset_train import training_subset
from selective_search_detection import prepare, fit_models, coverage
from nonlinear_appearance_study import rbf_logits
from metrics_map import evaluate_map50
from classical_methods_study import classification_errors

FOCUS=("beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
       "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg",
       "beach_bird_1606.jpg")
VARIANTS=("baseline_035","relaxed_060","refine_035","refine_060")

def regression_features(rec,base,rbf):
    b=rec["boxes"]
    if not len(b):return np.empty((0,38),np.float32)
    w=np.maximum(1,b[:,2]-b[:,0]);h=np.maximum(1,b[:,3]-b[:,1])
    geom=np.column_stack((np.log1p(w),np.log1p(h),np.log(w/h),
                          (b[:,0]+b[:,2])/1024.,(b[:,1]+b[:,3])/1024.))
    m=base.decision_function(rec["X"]).reshape(-1,1)
    rec["margin"]=m.ravel()  # RBF verifier consumes this TRAIN-derived stage1 score
    r=rbf_logits(rbf,rec).reshape(-1,1)
    result=np.column_stack((rec["aux"],geom,m,r)).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError("Nonfinite regression features")
    return result

def target_offsets(boxes,targets):
    wh=np.maximum(1.,boxes[:,2:]-boxes[:,:2])
    twh=np.maximum(1.,targets[:,2:]-targets[:,:2])
    shift=((targets[:,:2]+targets[:,2:])-(boxes[:,:2]+boxes[:,2:]))/(2*wh)
    size=np.log(twh/wh)
    return np.clip(np.column_stack((shift,size)),-1.5,1.5)

def fit_refiner(train,base,rbf):
    X=[];Y=[];total=0
    rng=np.random.default_rng(64)
    for rec in train:
        if not len(rec["gt"]) or not len(rec["boxes"]):continue
        overlap=iou(rec["boxes"],rec["gt"])
        best=overlap.argmax(axis=1);v=overlap[np.arange(len(best)),best]
        good=np.flatnonzero((v>=.35)&(v<.95))
        if len(good)>100:good=rng.choice(good,100,replace=False)
        if not len(good):continue
        inputs=regression_features(rec,base,rbf)[good]
        target=target_offsets(rec["boxes"][good],rec["gt"][best[good]])
        X.append(inputs);Y.append(target);total+=len(good)
    if not X:raise RuntimeError("No TRAIN matches for bbox regression")
    XX=np.vstack(X);yy=np.vstack(Y)
    model=ExtraTreesRegressor(n_estimators=90,max_depth=14,min_samples_leaf=4,
                              n_jobs=2,random_state=64)
    t=time.perf_counter();model.fit(XX,yy)
    return model,dict(train_regression_examples=total,train_regression_seconds=time.perf_counter()-t,
                      target_dimension=4,features=XX.shape[1],
                      training_input_source="TRAIN-only Selective Search/GT pairs with IoU [0.35,0.95)")

def apply_refiner(boxes,feats,model):
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    if not len(b):return b
    t=np.asarray(model.predict(feats),dtype=float)
    if t.shape!=(len(b),4):raise ValueError("Regression shape mismatch")
    # Conservative bound: limiting too-large predictions from unseen backgrounds.
    t[:,:2]=np.clip(t[:,:2],-.22,.22)
    t[:,2:]=np.clip(t[:,2:],-.32,.32)
    wh=b[:,2:]-b[:,:2]
    center=(b[:,:2]+b[:,2:])/2+t[:,:2]*wh
    newwh=wh*np.exp(t[:,2:])
    out=np.column_stack((center-newwh/2,center+newwh/2))
    out[:,0::2]=np.clip(out[:,0::2],0,512)
    out[:,1::2]=np.clip(out[:,1::2],0,512)
    bad=(out[:,2]<=out[:,0])|(out[:,3]<=out[:,1])|(~np.isfinite(out).all(axis=1))
    out[bad]=b[bad]
    return out.astype(np.float32)

def tier(box):
    area=float((box[2]-box[0])*(box[3]-box[1]))
    return "small" if area<1024 else ("medium" if area<10000 else "large")

def per_image_stats(boxes,scores,gt,raw):
    ov=iou(boxes,gt);used=set();tp=0;detected=set()
    for i in np.argsort(-scores,kind="stable")[:20]:
        eligible=[(float(ov[i,j]),j) for j in range(len(gt)) if j not in used and ov[i,j]>=.5]
        if eligible:
            _,j=max(eligible);used.add(j);tp+=1
    for i in np.argsort(-scores,kind="stable")[:100]:
        eligible=[(float(ov[i,j]),j) for j in range(len(gt)) if j not in detected and ov[i,j]>=.5]
        if eligible:detected.add(max(eligible)[1])
    allgt={k:0 for k in ("small","medium","large")}
    recalled={k:0 for k in allgt}
    for j,g in enumerate(gt):
        key=tier(g);allgt[key]+=1
        if j in detected:recalled[key]+=1
    return dict(top20_TP=tp,top20_FP=min(20,len(boxes))-tp,top100_recalled=len(detected),
       GT_small=allgt["small"],GT_medium=allgt["medium"],GT_large=allgt["large"],
       TP100_small=recalled["small"],TP100_medium=recalled["medium"],TP100_large=recalled["large"],
       crowded_GT=len(gt) if len(gt)>=20 else 0,
       crowded_TP100=len(detected) if len(gt)>=20 else 0,
       raw_R50=int((raw>=.5).sum()),raw_R75=int((raw>=.75).sum()))

def draw_audit(rec,predictions,method,out):
    im=cv2.imread(str(rec["path"]))
    gt=rec["gt"];b,s=predictions
    ov=iou(b,gt);used=set();top=np.argsort(-s,kind="stable")[:20]
    for g in gt:
        a=np.rint(g).astype(int);cv2.rectangle(im,tuple(a[:2]),tuple(a[2:]),(0,220,0),1)
    for i in top:
        eligible=[(float(ov[i,j]),j) for j in range(len(gt)) if j not in used and ov[i,j]>=.5]
        if eligible:
            _,j=max(eligible);used.add(j);colour=(255,0,0)
        else:colour=(0,0,255)
        a=np.rint(b[i]).astype(int);cv2.rectangle(im,tuple(a[:2]),tuple(a[2:]),colour,2)
    for j,g in enumerate(gt):
        if j not in used:
            a=np.rint(g).astype(int);cv2.rectangle(im,tuple(a[:2]),tuple(a[2:]),(0,220,255),2)
    # Original 512x512 full resolution retained in artifact
    cv2.putText(im,method+" "+rec["path"].stem[:18],(6,20),
                cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(out/(rec["path"].stem+"_"+method+".jpg")),im)

def score_bank(val,base,rbf,refiner,out):
    report=[];per_image=[];rows={k:[] for k in VARIANTS}
    out.mkdir(parents=True,exist_ok=True)
    panels=out/"visual";panels.mkdir(exist_ok=True)
    for idx,rec in enumerate(val):
        b=rec["boxes"]
        if len(b):
            m=base.decision_function(rec["X"])
            rec["margin"]=m  # Required by pretrained TRAIN-only PCA+RBF verifier
            score=.5*m+.5*rbf_logits(rbf,rec)
            features=regression_features(rec,base,rbf)
            refined=apply_refiner(b,features,refiner)
        else:score=np.empty(0);refined=b
        for variant in VARIANTS:
            use=refined if variant.startswith("refine") else b
            overlap=.60 if variant.endswith("060") else .35
            keep=nms(use,score,iou_limit=overlap,max_det=300) if len(use) else np.empty(0,int)
            bb=use[keep];ss=score[keep]
            rows[variant].append((bb,ss,rec["gt"]))
            item=dict(method=variant,image=rec["path"].name,GT=len(rec["gt"]),
                      proposals=len(b),nms_count=len(bb),**per_image_stats(bb,ss,rec["gt"],rec["coverage"]))
            per_image.append(item)
            if rec["path"].name in FOCUS:draw_audit(rec,(bb,ss),variant,panels)
        if (idx+1)%10==0:print("E32 EVAL",idx+1,len(val),flush=True)
    for variant in VARIANTS:
        data=rows[variant]
        metrics=evaluate_map50(data)
        by=[r for r in per_image if r["method"]==variant]
        result=dict(method=variant,**metrics,top20_TP=sum(x["top20_TP"] for x in by),
          top20_FP=sum(x["top20_FP"] for x in by),mean_nms_detections=float(np.mean([x["nms_count"] for x in by])),
          scale_recall_top100={k:sum(x["TP100_"+k] for x in by)/max(1,sum(x["GT_"+k] for x in by))
                               for k in ("small","medium","large")},
          crowded_recall_top100=(sum(x["crowded_TP100"] for x in by)/
              max(1,sum(x["crowded_GT"] for x in by))))
        report.append(result)
        print("E32 RESULT",json.dumps(result),flush=True)
    with (out/"e32_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=per_image[0].keys());w.writeheader();w.writerows(per_image)
    return report

def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    print("E32 TRAIN",len(train),"VAL",len(val),"GT",sum(len(gt) for _,gt in val),flush=True)
    tb=prepare(train,"selective","TRAIN")
    vb=prepare(val,"selective","VAL")
    base,rbf,diag=fit_models(tb)
    refiner,refdiag=fit_refiner(tb,base,rbf)
    results=score_bank(vb,base,rbf,refiner,out)
    baseline=next(x for x in results if x["method"]=="baseline_035")
    options=[x for x in results if x["method"]!="baseline_035"]
    # Conservative, predeclared gate, no automatic test usage from an exploratory validation.
    qualified=[x for x in options if x["map50_coco101"]>=baseline["map50_coco101"]+.02
        and x["top20_TP"]>=baseline["top20_TP"]
        and x["scale_recall_top100"]["small"]>=baseline["scale_recall_top100"]["small"]-.03
        and x["crowded_recall_top100"]>=baseline["crowded_recall_top100"]-.03]
    winner=max(qualified,key=lambda x:x["map50_coco101"]) if qualified else baseline
    summary=dict(status="E32_EXPLORATORY_VALIDATION_COMPLETE",
       train_images=len(train),val_images=len(val),val_gt=sum(len(g) for _,g in val),
       train_ids=[p.name for p,_ in train],val_ids=[p.name for p,_ in val],
       baseline_architecture="E31 Selective Search + GT jitter HOG/LBP SVM + PCA64 RBF",
       results=results,diagnostics=dict(training=diag,regression=refdiag,proposal=coverage(vb)),
       selected_for_confirmation=winner["method"],gate_pass=winner["method"]!="baseline_035",
       warning="VAL-only method selection; NOT confirmed on larger holdout; TEST untouched; raw proposal recall != detector AP",
       data_policy="No TEST data accessed. All regressors/classifiers trained only on TRAIN.")
    (out/"e32_summary.json").write_text(json.dumps(summary,indent=2))
    print("E32 FINAL",json.dumps({k:summary[k] for k in ("selected_for_confirmation","gate_pass","val_gt")}),flush=True)

def tests():
    b=np.array([[10,10,20,30],[0,0,15,15]],np.float32)
    z=target_offsets(b,b)
    assert np.max(np.abs(z))<1e-8
    class Zero:
        def predict(self,f):return np.zeros((len(f),4))
    corrected=apply_refiner(b,np.zeros((len(b),39)),Zero())
    assert np.allclose(corrected,b)
    assert tier(np.array([0,0,20,20]))=="small"
    p=per_image_stats(np.array([[10,10,20,30]]),np.array([.5]),np.array([[10,10,20,30]]),np.array([1.]))
    assert p["top20_TP"]==1 and p["TP100_small"]==1
    print("E32 regression identity/box and matching tests PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("e32_results"))
    p.add_argument("--train-n",type=int,default=100);p.add_argument("--val-n",type=int,default=60)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
