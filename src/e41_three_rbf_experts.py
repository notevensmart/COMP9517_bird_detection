"""E41: three size-specialized classical RBF SVMs against matched global RBF.
TRAIN-only PCA, expert fitting and Platt calibration. Deterministic E37 SS bank.
No pretrained models, neural nets, TEST data or ground-truth at inference.
"""
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np
from sklearn.svm import SVC
from cloud_audit import data,nms
from quality_subset_train import training_subset
from selective_search_detection import prepare,fit_models,coverage
from nonlinear_appearance_study import collect_proposal_examples,rbf_logits
from metrics_map import evaluate_map50
from e32_failure_driven import per_image_stats,draw_audit,fit_refiner,apply_refiner,regression_features
from rebuild_detector import cropbox
from e39_scale_aware_boosting import letterbox
# The branch imports E39's image preprocessing only; it does NOT use LightGBM.
SEED=41
FOCUS={"beach_bird_1379.jpg","beach_bird_1389.jpg","beach_bird_1399.jpg",
       "beach_bird_1419.jpg","beach_bird_1435.jpg","beach_bird_1452.jpg",
       "beach_bird_1560.jpg","beach_bird_1606.jpg"}
SCALES=("small","medium","large")
def scale_ids(boxes):
    b=np.asarray(boxes,np.float32).reshape(-1,4)
    area=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)
    return np.where(area<1024,0,np.where(area<10000,1,2)).astype(np.int8)

def expert_features(global_rbf,rec):
    norm,pca,scaler,_=global_rbf
    if not len(rec["boxes"]):return np.empty((0,97),np.float32)
    z=pca.transform(norm.transform(rec["X"]))
    x=scaler.transform(np.c_[z,rec["aux"],rec["margin"]])
    if not np.isfinite(x).all():raise ValueError("Invalid expert input")
    return np.asarray(x,np.float32)

def train_experts(bank,global_rbf):
    X,A,M,y,counts=collect_proposal_examples(bank)
    norm,pca,scaler,_=global_rbf
    all_x=scaler.transform(np.c_[pca.transform(norm.transform(X)),A,M])
    # Every training proposal's native candidate area is reconstructed from
    # the exact same TRAIN-only sampling algorithm, with no GT size routing.
    rng=np.random.default_rng(77);areas=[]
    for r in bank:
        if not len(r["boxes"]):continue
        good=np.asarray(r["pos"])
        if len(good)>45:good=rng.choice(good,45,replace=False)
        scores=r["margin"]
        keep=nms(r["boxes"],scores,.5,170) if len(scores) else []
        safe=set(int(i) for i in r["neg"])
        hard=[int(j) for j in keep if int(j) in safe and scores[int(j)]>-.7][:25]
        rem=[int(j) for j in r["neg"] if int(j) not in set(hard)]
        rand=rng.choice(rem,min(12,len(rem)),replace=False).tolist() if rem else []
        ids=list(good)+hard+rand
        areas.extend(scale_ids(r["boxes"][ids]).tolist())
    if len(areas)!=len(y):raise AssertionError(f"TRAIN example routing mismatch: {len(areas)} != {len(y)}")
    groups=np.asarray(areas)
    experts={};diagnostics={}
    for i,name in enumerate(SCALES):
        ids=np.flatnonzero(groups==i)
        labels=np.bincount(y[ids],minlength=2)
        if len(ids)<50 or min(labels)<15:
            diagnostics[name]=dict(status="fallback_to_global",examples=len(ids),class_counts=labels.tolist())
            continue
        clf=SVC(C=3.0,kernel="rbf",gamma="scale",class_weight="balanced",
                probability=True,random_state=SEED+i,cache_size=700)
        tic=time.perf_counter();clf.fit(all_x[ids],y[ids])
        experts[name]=clf
        diagnostics[name]=dict(status="trained",examples=len(ids),class_counts=labels.tolist(),
              support_vectors=int(clf.n_support_.sum()),seconds=time.perf_counter()-tic)
    return experts,dict(train_examples=counts,scale=diagnostics,feature_dim=all_x.shape[1])

def logit(p):
    p=np.clip(p,1e-4,1-1e-4)
    return np.log(p/(1-p))

def scores(rec,base,global_rbf,experts):
    if not len(rec["boxes"]):return np.empty(0),np.empty(0),np.empty(0)
    margin=base.decision_function(rec["X"])
    rec["margin"]=margin
    global_log=rbf_logits(global_rbf,rec)
    baseline=.5*margin+.5*global_log
    # Probability=True performs TRAIN-only Platt fitting; probabilities are
    # comparable enough for fixed blending, but not proof of calibration.
    x=expert_features(global_rbf,rec)
    combined=global_log.copy()
    for i,name in enumerate(SCALES):
        mask=scale_ids(rec["boxes"])==i
        if mask.any() and name in experts:
            specialist=logit(experts[name].predict_proba(x[mask])[:,1])
            combined[mask]=.5*global_log[mask]+.5*specialist
    specialist_score=.5*margin+.5*combined
    return baseline,specialist_score,combined

def evaluate(bank,base,rbf,experts,refiner,out):
    variants=("global_rbf","three_experts","global_expert_blend","blend_refine","blend_crowd_nms")
    # three_experts: global score absent from RBF verifier; global stage1 kept
    # so only the specialization contribution changes.
    preds={v:[] for v in variants};rows=[];vis=out/"visual";vis.mkdir(parents=True,exist_ok=True)
    for k,r in enumerate(bank):
        b=r["boxes"]
        if len(b):
            g,blend,combined=scores(r,base,rbf,experts)
            expert_only=.5*r["margin"]+.5*(2*combined-rbf_logits(rbf,r))
            reg=regression_features(r,base,rbf)
            refined=apply_refiner(b,reg,refiner)
        else:g=blend=expert_only=np.empty(0);refined=b
        for name in variants:
            if name=="global_rbf": bb,sc,thr=b,g,.35
            elif name=="three_experts":bb,sc,thr=b,expert_only,.35
            elif name=="global_expert_blend":bb,sc,thr=b,blend,.35
            elif name=="blend_refine":bb,sc,thr=refined,blend,.35
            else:bb,sc,thr=b,blend,.55
            keep=nms(bb,sc,iou_limit=thr,max_det=300) if len(bb) else np.empty(0,int)
            selected=bb[keep];ranked=sc[keep]
            preds[name].append((selected,ranked,r["gt"]))
            stat=per_image_stats(selected,ranked,r["gt"],r["coverage"])
            rows.append(dict(image=r["path"].name,method=name,GT=len(r["gt"]),**stat))
            if r["path"].name in FOCUS:
                draw_audit(r,(selected,ranked),name,vis)
        if (k+1)%10==0:print("E41 VALIDATION",k+1,len(bank),flush=True)
    result=[]
    for name in variants:
        v=evaluate_map50(preds[name])
        rr=[x for x in rows if x["method"]==name]
        v.update(method=name,top20_TP=sum(x["top20_TP"] for x in rr),
            top20_FP=sum(x["top20_FP"] for x in rr),
            scale_recall_top100={z:sum(x["TP100_"+z] for x in rr)/max(1,sum(x["GT_"+z] for x in rr)) for z in SCALES},
            crowded_recall_top100=sum(x["crowded_TP100"] for x in rr)/max(1,sum(x["crowded_GT"] for x in rr)))
        result.append(v)
        print("E41 METHOD RESULT",json.dumps(v),flush=True)
    with (out/"e41_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    return result

def preprocessing_audit(train,out):
    imdir=out/"preprocessing";imdir.mkdir(exist_ok=True)
    previews=[];counts=dict(checked_images=0,checked_crops=0,extremely_small=0,invalid=0)
    for path,gt in train[:8]:
        im=cv2.imread(str(path))
        if im is None:raise ValueError("Unreadable TRAIN image")
        counts["checked_images"]+=1
        for box in gt[:4]:
            crop=cropbox(im,box)
            lb,mask,(h,w)=letterbox(crop)
            if lb.shape!=(64,64,3) or mask.shape!=(64,64):raise AssertionError("Bad letterbox")
            if h/w>2 and mask.sum()==4096:raise AssertionError("Aspect ratio destroyed")
            if min(h,w)<8:counts["extremely_small"]+=1
            counts["checked_crops"]+=1
            if len(previews)<12:
                orig=cv2.resize(crop,(128,128),interpolation=cv2.INTER_NEAREST)
                scaled=cv2.resize(lb,(128,128),interpolation=cv2.INTER_NEAREST)
                previews.append(np.hstack((orig,scaled)))
    if previews:cv2.imwrite(str(imdir/"source_vs_aspect_preserved.jpg"),np.vstack(previews))
    (imdir/"audit.json").write_text(json.dumps(counts,indent=2))
    print("E41 PREPROCESS AUDIT",json.dumps(counts),flush=True)

def tests():
    b=np.array([[0,0,10,10],[0,0,40,40],[0,0,150,100]],np.float32)
    assert scale_ids(b).tolist()==[0,1,2]
    from e39_scale_aware_boosting import tests as preprocessing_tests
    preprocessing_tests()
    assert np.isfinite(logit(np.array([0,1,.5]))).all()
    print("E41 self-test PASS",flush=True)

def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    preprocessing_audit(train,out)
    print("E41 TRAIN",len(train),"VAL",len(val),"GT",sum(len(g) for _,g in val),flush=True)
    tb=prepare(train,"selective","TRAIN");vb=prepare(val,"selective","VAL")
    base,rbf,diagnostics=fit_models(tb)
    experts,expert_diag=train_experts(tb,rbf)
    refiner,refiner_diag=fit_refiner(tb,base,rbf)
    results=evaluate(vb,base,rbf,experts,refiner,out)
    base_score=results[0];candidate=results[2]
    gate=(candidate["map50_coco101"]>=base_score["map50_coco101"]+.02
          and candidate["top20_TP"]>=base_score["top20_TP"]
          and candidate["scale_recall_top100"]["small"]>=base_score["scale_recall_top100"]["small"]-.03
          and candidate["crowded_recall_top100"]>=base_score["crowded_recall_top100"]-.03)
    report=dict(status="COMPLETE",split="TRAIN/VAL only, no TEST access",train_images=len(train),
      val_images=len(val),val_GT=sum(len(g) for _,g in val),seed=SEED,
      deterministic_proposals=True,features="TRAIN-only shared PCA64(HOG+LBP)+32D context+stage1",
      routing="GT-blind proposal native area <1024, 1024..9999, >=10000",
      classifier="Three RBF SVC probability=True with TRAIN-only calibration; fallback to global if sparse",
      baseline_training=diagnostics,expert_training=expert_diag,refiner_training=refiner_diag,
      proposals=coverage(vb),results=results,predeclared_promotion_gate=gate,
      note="Refinement and NMS are separate ablations; do not choose model on TEST.")
    (out/"e41_report.json").write_text(json.dumps(report,indent=2))
    print("E41 FINAL",json.dumps(dict(gate=gate,results=results)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e41_results"))
    p.add_argument("--train-n",type=int,default=100)
    p.add_argument("--val-n",type=int,default=60)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
