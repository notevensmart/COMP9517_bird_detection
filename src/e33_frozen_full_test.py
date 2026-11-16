"""E33: independent larger VAL confirmation then ONE full TEST evaluation.

Strictly classical pretrained-free feature pipeline. Freeze architecture and
hyperparameters before reading TEST. TRAIN labels are used for supervised fitting.
Test GT is loaded only AFTER the VAL-based variant decision is written.
"""
import argparse, csv, json, time, gc
from pathlib import Path
import numpy as np, cv2
from cloud_audit import data, iou, nms
from quality_subset_train import training_subset
from selective_search_detection import prepare, fit_models, coverage
from nonlinear_appearance_study import rbf_logits
from e32_failure_driven import fit_refiner, regression_features, apply_refiner, per_image_stats, draw_audit
from e34_stage2_lgbm import fit_lgb, lgb_logits
from metrics_map import evaluate_map50

VARIANTS=("baseline_035","relaxed_060","refine_035","refine_060","lightgbm_stage2")

def score_one(rec,base,rbf,variant,regressor=None,lightgbm=None):
    b=rec["boxes"]
    if not len(b):return np.empty((0,4),np.float32),np.empty(0,float)
    stage1=base.decision_function(rec["X"])
    rec["margin"]=stage1  # RBF input contract; computed without GT
    score=.5*stage1+.5*(lgb_logits(lightgbm,rbf,rec) if variant=="lightgbm_stage2" else rbf_logits(rbf,rec))
    if variant=="lightgbm_stage2" and lightgbm is None:raise ValueError("Missing frozen LightGBM stage2 model")
    if variant.startswith("refine"):
        if regressor is None:raise ValueError("Refiner required for frozen variant "+variant)
        b=apply_refiner(b,regression_features(rec,base,rbf),regressor)
    nms_iou=.6 if variant.endswith("060") else .35
    keep=nms(b,score,iou_limit=nms_iou,max_det=300)
    return b[keep],score[keep]

def eval_split(samples,base,rbf,variant,regressor,out,split,lightgbm=None):
    out.mkdir(parents=True,exist_ok=True)
    preds=[];rows=[];begin=time.perf_counter()
    focus={"beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
           "beach_bird_1435.jpg","beach_bird_1452.jpg","beach_bird_1560.jpg",
           "beach_bird_1606.jpg"}
    for i,(path,gt) in enumerate(samples,1):
        tic=time.perf_counter()
        # GT only for post-hoc audit; proposals/feature model do not use labels.
        rec=prepare([(path,gt)],"selective",split)[0]
        b,s=score_one(rec,base,rbf,variant,regressor,lightgbm)
        duration=time.perf_counter()-tic
        preds.append((b,s,gt))
        row=dict(image=path.name,method=variant,GT=len(gt),proposals=len(rec["boxes"]),
           predictions=len(b),inference_seconds=duration,
           **per_image_stats(b,s,gt,rec["coverage"]))
        rows.append(row)
        if path.name in focus and split=="VAL":draw_audit(rec,(b,s),variant,out)
        if i%25==0 or i==len(samples):
            print("E33",split,variant,i,len(samples),"elapsed",round(time.perf_counter()-begin,2),flush=True)
            (out/"progress.json").write_text(json.dumps(dict(split=split,method=variant,
                images_completed=i,total_images=len(samples),seconds=time.perf_counter()-begin),indent=2))
    metrics=evaluate_map50(preds)
    ret=dict(method=variant,split=split,images=len(samples),GT=sum(len(x[2]) for x in preds),
       **metrics,top20_TP=sum(x["top20_TP"] for x in rows),
       top20_FP=sum(x["top20_FP"] for x in rows),
       top100_GT_recall=sum(x["top100_recalled"] for x in rows)/max(1,sum(x["GT"] for x in rows)),
       scale_recall_top100={k:sum(x["TP100_"+k] for x in rows)/max(1,sum(x["GT_"+k] for x in rows))
                            for k in ("small","medium","large")},
       crowded_recall_top100=sum(x["crowded_TP100"] for x in rows)/max(1,sum(x["crowded_GT"] for x in rows)),
       mean_inference_seconds=float(np.mean([r["inference_seconds"] for r in rows])),
       total_inference_seconds=time.perf_counter()-begin)
    with (out/"per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (out/"metrics.json").write_text(json.dumps(ret,indent=2))
    print("E33 METRICS",json.dumps(ret),flush=True)
    return ret,rows

def pass_confirmation(baseline,candidate):
    return (candidate["map50_coco101"]>=baseline["map50_coco101"]
        and candidate["top20_TP"]>=baseline["top20_TP"]
        and candidate["scale_recall_top100"]["small"]>=baseline["scale_recall_top100"]["small"]-.03
        and candidate["crowded_recall_top100"]>=baseline["crowded_recall_top100"]-.03)

def run(dataset,out,train_n,candidate,confirm_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(dataset,train_n)
    # E32 used first 60 VAL chosen with seed 43. This complementary VAL split
    # is the independent confirmation set, not automatically the entire VAL.
    development=data(dataset,"val",60)
    development_names={p.name for p,_ in development}
    all_val=data(dataset,"val",0)
    independent=[x for x in all_val if x[0].name not in development_names]
    if confirm_n and confirm_n<len(independent):independent=independent[:confirm_n]
    assert not(set(p.name for p,_ in independent)&development_names)
    print("E33 TRAIN",len(train),"INDEPENDENT VAL",len(independent),flush=True)
    tic=time.perf_counter()
    bank=prepare(train,"selective","TRAIN")
    base,rbf,fitdiag=fit_models(bank)
    refiner=None;refdiag=None;lightgbm=None;lightdiag=None
    if candidate.startswith("refine"):
        refiner,refdiag=fit_refiner(bank,base,rbf)
    if candidate=="lightgbm_stage2":
        lightgbm,lightdiag=fit_lgb(bank,rbf)
    del bank;gc.collect()
    train_seconds=time.perf_counter()-tic
    basemetrics,_=eval_split(independent,base,rbf,"baseline_035",None,
                             out/"confirmation"/"baseline", "VAL")
    if candidate=="baseline_035":
        select="baseline_035";confirm=None;accepted=True
    else:
        confirm,_=eval_split(independent,base,rbf,candidate,refiner,
                            out/"confirmation"/"candidate","VAL",lightgbm=lightgbm)
        accepted=pass_confirmation(basemetrics,confirm)
        select=candidate if accepted else "baseline_035"
    # FREEZE: write and flush the decision BEFORE reading TEST records.
    selection=dict(status="FROZEN_BEFORE_TEST",choice=select,
          chosen_candidate_from_E32=candidate,confirmed=accepted,train_n=len(train),
          train_ids=[p.name for p,_ in train],development_val_n=60,
          independent_val_n=len(independent),confirm_val_ids=[p.name for p,_ in independent],
          train_seconds=train_seconds,training=fitdiag,refiner_training=refdiag,lightgbm_training=lightdiag,
          baseline_independent_val=basemetrics,candidate_independent_val=confirm,
          methodological_note="No TEST input has been read or used at this point; architecture chosen by TRAIN + complementary VAL.")
    (out/"model_freeze.json").write_text(json.dumps(selection,indent=2))
    print("E33 MODEL FROZEN BEFORE TEST",json.dumps({k:selection[k] for k in ("choice","confirmed","train_n","independent_val_n")}),flush=True)
    if select=="baseline_035":refiner=None;lightgbm=None
    # All 240 original test images, label set kept untouched until now.
    test=data(dataset,"test",0)
    if len(test)!=240:raise RuntimeError(f"Expected 240 TEST images, found {len(test)}")
    metrics,rows=eval_split(test,base,rbf,select,refiner,out/"test","TEST",lightgbm=lightgbm)
    # Save representative post-hoc test visuals without any further fitting or tuning.
    toshow=[]
    for rec in sorted(rows,key=lambda x:(-x["top20_FP"],x["image"]))[:3]:toshow.append(rec["image"])
    for rec in sorted(rows,key=lambda x:(-x["GT"],x["image"]))[:3]:toshow.append(rec["image"])
    for rec in sorted(rows,key=lambda x:(-x["GT_large"],x["image"]))[:2]:toshow.append(rec["image"])
    toshow=list(dict.fromkeys(toshow))
    visual=(out/"test"/"visual");visual.mkdir(exist_ok=True)
    picks={x[0].name:x for x in test if x[0].name in toshow}
    for name in toshow:
        path,gt=picks[name]
        rec=prepare([(path,gt)],"selective","TEST VISUAL")[0]
        b,s=score_one(rec,base,rbf,select,refiner,lightgbm)
        draw_audit(rec,(b,s),select,visual)
    record=dict(status="COMPLETED_FULL_240_IMAGE_TEST",frozen_model=select,
       training_images=len(train),test_images=len(test),test_GT=metrics["GT"],
       test_evaluation=metrics,selection_record="model_freeze.json",
       note="Once-only TEST assessment; do not alter any model using TEST feedback. Official labels preserved. No neural models.")
    (out/"final_test_report.json").write_text(json.dumps(record,indent=2))
    print("E33 FULL TEST FINAL",json.dumps(record),flush=True)

def tests():
    b={"map50_coco101":.1,"top20_TP":20,"scale_recall_top100":{"small":.5},
       "crowded_recall_top100":.5}
    assert pass_confirmation(b,b)
    b2=dict(b,map50_coco101=.09)
    assert not pass_confirmation(b,b2)
    print("E33 independent confirmation gate tests PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e33_full_evaluation"))
    p.add_argument("--train-n",type=int,default=200)
    p.add_argument("--candidate",choices=VARIANTS,default="baseline_035")
    p.add_argument("--confirm-n",type=int,default=0)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.candidate,a.confirm_n)
