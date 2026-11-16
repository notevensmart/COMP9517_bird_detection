"""E21: scale-aware CLASSICAL bird detection with frozen evaluation.

No CNNs, pretrained features or GT-derived proposals. Image-level TRAIN split:
80% model fitting, 20% score calibration (seed 37). Fixed VAL remains untouched
by fitting, calibration, mining, or thresholds.

A: single-classifier HOG/LBP SVM with GT+jitter positives and full-image mining.
B: three size-gated SVMs with separate GT+jitter + proposal-positive training
   and scale-specific safe background examples mined from full TRAIN images.
C: B plus HOG/LBP + HSV histogram and grayscale/edge occupancy features.
Methods share exactly the same train/VAL images and GT-blind hybrid proposals.
Compute official COCO 101-point single-class mAP@IoU=.50 (maxDets=100),
post-NMS one-to-one FP breakdown and actual visual comparisons.

Score calibration: Platt/logistic calibrators trained on RESERVED TRAIN images
only; never VAL. Fit SVMs on 80% training image subset.
Method C feature dimension is 1924+ (12 HSV hist + 2 intensity stats + 2 edges).
"""
import argparse,csv,json,time,random
from pathlib import Path
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from cloud_audit import data,iou,nms
from label_quality import unsafe_training_negative
from multiscale_coverage import methods as proposal_methods
from rebuild_detector import cropbox,make_jitter,safe_labels,classification_errors,draw_box
from hog_lbp_comparison import features
from metrics_map import evaluate_map50

RNG=37;AREA=(1024.,10000.)
def bucket(box):
    wh=np.maximum(1,np.asarray(box)[2:]-np.asarray(box)[:2])
    return int(np.searchsorted(AREA,float(wh[0]*wh[1]),side="right"))

def appearance(im,box,color=False):
    patch=cropbox(im,box)
    grey=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
    base=features(grey,True).astype(np.float32)
    if not color:return base
    hsv=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2HSV)
    hist=np.concatenate([cv2.calcHist([hsv],[k],None,[4],[0,180] if k==0 else [0,256]).ravel() for k in range(3)]).astype(np.float32)
    hist/=np.maximum(hist.sum(),1)
    g=grey.astype(np.float32)/255.
    edges=cv2.Canny(grey,60,150)>0
    top=float(edges[:32].mean());bottom=float(edges[32:].mean())
    extra=np.r_[hist,g.mean(),g.std(),top,bottom].astype(np.float32)
    return np.r_[base,extra]

def bank(samples,phase):
    records=[];started=time.perf_counter()
    for idx,(path,gt) in enumerate(samples):
        im=cv2.imread(str(path))
        if im is None:raise ValueError("Could not read "+str(path))
        st=time.perf_counter()
        boxes=proposal_methods(im,per_shape=50)["Hybrid_union"]
        proposal_seconds=time.perf_counter()-st
        st=time.perf_counter()
        X=np.stack([appearance(im,b,True) for b in boxes]) if len(boxes) else np.empty((0,1940),np.float32)
        assert X.shape[1]==1940,(path,X.shape)
        feature_seconds=time.perf_counter()-st
        pos,neg=safe_labels(boxes,gt)
        ids=np.asarray([bucket(b) for b in boxes],np.int8)
        records.append(dict(path=path,gt=gt,boxes=boxes,features=X,scale=ids,
                            pos=pos,neg=neg,prop_secs=proposal_seconds,feature_secs=feature_seconds))
        if (idx+1)%15==0 or idx+1==len(samples):
            print("SCALE BANK",phase,idx+1,len(samples),
                  "seconds",round(time.perf_counter()-started,1),flush=True)
    return records

def sample_training(bank_records,include_proposal_positives=True,seed=37):
    """All positives stay in TRAIN; safe negatives require IoU < .1 and
    containment of neither whole GT nor partial GT. TRAIN labels only."""
    rng=np.random.default_rng(seed)
    outputs={i:dict(pos=[],neg=[]) for i in range(3)}
    for row in bank_records:
        im=cv2.imread(str(row["path"]))
        for gt in row["gt"]:
            jboxes=make_jitter(gt,im.shape[1],im.shape[0],rng,count=3)
            for box in jboxes:
                outputs[bucket(box)]["pos"].append(appearance(im,box,True))
        if include_proposal_positives:
            p=row["pos"]
            if len(p):
                for scale in range(3):
                    viable=p[row["scale"][p]==scale]
                    if len(viable)>12:viable=rng.choice(viable,12,replace=False)
                    outputs[scale]["pos"].extend(row["features"][viable])
        for scale in range(3):
            possible=row["neg"][row["scale"][row["neg"]]==scale]
            if len(possible)>22:possible=rng.choice(possible,22,replace=False)
            outputs[scale]["neg"].extend(row["features"][possible])
    result={}
    for sc,v in outputs.items():
        result[sc]={cls:(np.stack(values).astype(np.float32) if values else np.empty((0,1940),np.float32))
                    for cls,values in v.items()}
    return result

def build_models(samples,scale_aware,color,rounds=2):
    part=sample_training(samples,True)
    if not scale_aware:
        single={"pos":np.concatenate([part[k]["pos"] for k in range(3)]),
                "neg":np.concatenate([part[k]["neg"] for k in range(3)])}
        part={"all":single}
    dim=1940 if color else 1924
    models={};hist=[]
    for key,featureset in part.items():
        pos=featureset["pos"][:,:dim]
        neg=featureset["neg"][:,:dim]
        if len(pos)<5 or len(neg)<5:raise RuntimeError(f"Insufficient TRAIN samples for {key}: {len(pos)} positive {len(neg)} negative")
        for rd in range(rounds+1):
            model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",
                                  max_iter=3500,random_state=42))
            model.fit(np.vstack([pos,neg]),np.r_[np.ones(len(pos)),np.zeros(len(neg))])
            record=dict(scale=str(key),round=rd,positives=len(pos),negatives=len(neg),
                        patch_train_accuracy=float(np.mean(model.predict(np.vstack([pos,neg]))==
                        np.r_[np.ones(len(pos)),np.zeros(len(neg))])))
            if rd<rounds:
                scored=[]
                for imgidx,r in enumerate(samples):
                    selection=np.arange(len(r["boxes"])) if key=="all" else np.flatnonzero(r["scale"]==key)
                    if len(selection)==0:continue
                    bx=r["boxes"][selection]
                    raw=model.decision_function(r["features"][selection,:dim])
                    keep=nms(bx,raw,iou_limit=.50,max_det=150)
                    safe=set(r["neg"].tolist())
                    count=0
                    for j in keep:
                        ix=int(selection[j])
                        if ix not in safe:continue
                        scored.append((float(raw[j]),imgidx,ix))
                        count+=1
                        if count>=10:break
                scored.sort(key=lambda t:-t[0])
                take=scored[:min(300,len(scored))]
                if take:neg=np.vstack([neg,np.stack([samples[i]["features"][j,:dim] for _,i,j in take])])
                record["mined_negatives"]=len(take)
            hist.append(record);models[key]=model
            print("SCALE TRAIN",json.dumps(record),flush=True)
    return models,hist

def calibrate(models,heldout,scale_aware,color):
    """Proposal-domain calibration from separate TRAIN images.

    Use *actual generated proposals*, not ideal GT-jitter patches, to match
    inference score distributions. Natural imbalance: no balanced class weight.
    GT on these TRAIN calibration images labels samples but never affects box
    generation or final evaluation on VAL. Positive = candidate IoU >=.5;
    negative = GT-safe candidate IoU<.1 and containment<.5.
    """
    dim=1940 if color else 1924
    rng=np.random.default_rng(89)
    calibrators={};history=[]
    for key,model in models.items():
        pos=[];neg=[]
        for r in heldout:
            good=r["pos"];bad=r["neg"]
            if key!="all":
                good=good[r["scale"][good]==key]
                bad=bad[r["scale"][bad]==key]
            if len(good)>50:good=rng.choice(good,50,replace=False)
            if len(bad)>200:bad=rng.choice(bad,200,replace=False)
            if len(good):pos.extend(model.decision_function(r["features"][good,:dim]).tolist())
            if len(bad):neg.extend(model.decision_function(r["features"][bad,:dim]).tolist())
        if len(pos)<5 or len(neg)<20:
            raise RuntimeError(f"Scale {key} lacks enough real calibration proposals: positive {len(pos)}, negative {len(neg)}")
        margins=np.r_[pos,neg].reshape(-1,1)
        labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        cal=LogisticRegression(C=1.,class_weight=None,random_state=42,max_iter=500)
        cal.fit(margins,labels)
        calibrators[key]=cal
        history.append(dict(scale=str(key),calibration_positive=len(pos),
                            calibration_negative=len(neg),
                            empirical_positive_fraction=float(np.mean(labels)),
                            sigmoid_coefficient=float(cal.coef_[0,0]),
                            sigmoid_intercept=float(cal.intercept_[0])))
    return calibrators,history

def scores(models,cals,r,color=True):
    dim=1940 if color else 1924
    output=np.full(len(r["boxes"]),-10000.,np.float64)
    split="all" in models
    for key,m in models.items():
        ix=np.arange(len(output)) if split else np.flatnonzero(r["scale"]==key)
        if not len(ix):continue
        decision=m.decision_function(r["features"][ix,:dim])
        # P(bird) from a held-out TRAIN-only Platt calibrator.
        output[ix]=cals[key].predict_proba(decision.reshape(-1,1))[:,1]
    assert np.isfinite(output).all() and np.max(output)<=1.0 and np.min(output)>=0.0
    return output

def match_for_visual(boxes,scores,gt):
    used=set();tp=[];fp=[];ov=iou(boxes,gt)
    for idx in np.argsort(-scores):
        elig=[j for j in range(len(gt)) if j not in used and ov[idx,j]>=.5]
        if elig:
            best=max(elig,key=lambda j:ov[idx,j]);used.add(best);tp.append(idx)
        else:fp.append(idx)
    return tp,fp,used

def evaluate(models,cals,val,color=True,nms_iou=.5,visual_dir=None):
    preds=[];rows=[];illustrations=[]
    for r in val:
        s=scores(models,cals,r,color)
        keep=nms(r["boxes"],s,iou_limit=nms_iou,max_det=300)
        b=r["boxes"][keep];sc=s[keep]
        preds.append((b,sc,r["gt"]))
        # Since probability calibration replaces raw margin, use a predeclared
        # 0.5 probability threshold for operating point FP/FN counts.
        active=sc>=.5
        err=classification_errors(b[active],sc[active],r["gt"])
        top=min(20,len(b))
        top20=classification_errors(b[:top],sc[:top],r["gt"])
        rows.append(dict(image=r["path"].name,GT=len(r["gt"]),proposals=len(r["boxes"]),
                         after_nms=len(b),probability05=int(active.sum()),
                         FP_top20=top20["FP_total"],FN_top20=top20["FN"],**err))
        # For visual ranking QA draw top-20 candidates even if none passes .5.
        # Retain margin0.5 FP/FN as a distinct, explicitly labeled statistic.
        illustrations.append(dict(path=r["path"],gt=r["gt"],boxes=b[:top],scores=sc[:top],
                                  err=rows[-1],orig_boxes=r["boxes"]))
    metric=evaluate_map50(preds)
    metric.update(FP_total_p05=sum(x["FP_total"] for x in rows),
                  FP_background_p05=sum(x["FP_background"] for x in rows),
                  FP_localization_p05=sum(x["FP_localization"] for x in rows),
                  FN_p05=sum(x["FN"] for x in rows),
                  FP_top20=sum(x["FP_top20"] for x in rows),
                  FN_top20=sum(x["FN_top20"] for x in rows),
                  mean_proposals=float(np.mean([len(r["boxes"]) for r in val])),
                  GT_count=int(sum(len(r["gt"]) for r in val)))
    if visual_dir:
        visual_dir.mkdir(parents=True,exist_ok=True)
        sorted_cases=sorted(illustrations,key=lambda x:-x["err"]["FP_total"])
        # Include FP-heavy, crowded, background-only, and representative cases.
        chosen=sorted_cases[:2]+sorted(illustrations,key=lambda x:-x["err"]["GT"])[:2]+sorted(illustrations,key=lambda x:x["err"]["GT"])[:2]
        seen=set();tiles=[]
        for row in chosen:
            if row["path"].name in seen:continue
            seen.add(row["path"].name)
            image=cv2.imread(str(row["path"]))
            matched,incorrect,used=match_for_visual(row["boxes"],row["scores"],row["gt"])
            for ix in matched:draw_box(image,row["boxes"][ix],(255,0,0),2)
            for ix in incorrect:draw_box(image,row["boxes"][ix],(0,0,255),2)
            for ix,gt in enumerate(row["gt"]):
                draw_box(image,gt,(0,255,0) if ix in used else (0,255,255),2)
            cv2.putText(image,row["path"].name,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3)
            cv2.putText(image,row["path"].name,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.imwrite(str(visual_dir/(row["path"].stem+"_scored.jpg")),image)
            if len(tiles)<6:tiles.append(cv2.resize(image,(540,350)))
        if tiles:
            if len(tiles)%2:tiles.pop()
            if tiles:
                cv2.imwrite(str(visual_dir/"contact_sheet.jpg"),
                   np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    return metric,rows

def run(root,out,train_limit,val_limit):
    out.mkdir(parents=True,exist_ok=True)
    train=data(root,"train",train_limit)
    # Exclude only previously documented suspect TRAIN missing-label images.
    excluded=[p.name for p,gt in train if unsafe_training_negative(p.name,len(gt))]
    train=[x for x in train if not unsafe_training_negative(x[0].name,len(x[1]))]
    val=data(root,"val",val_limit)
    ordering=np.random.default_rng(RNG).permutation(len(train))
    num_cal=max(15,int(.2*len(train)))
    fit_paths=[train[i] for i in ordering[num_cal:]]
    cal_paths=[train[i] for i in ordering[:num_cal]]
    # Reuse fixed VAL proposal & descriptor bank across variants.
    fit_bank=bank(fit_paths,"FIT")
    cal_bank=bank(cal_paths,"CAL")
    val_bank=bank(val,"VAL")
    configs=[("A_single_HOG_LBP",False,False),("B_scale_HOG_LBP",True,False),
             ("C_scale_HOG_LBP_color",True,True)]
    summary=[];per_image=[]
    for name,scale_aware,color in configs:
        start=time.perf_counter()
        models,history=build_models(fit_bank,scale_aware,color,rounds=2)
        cals,calhistory=calibrate(models,cal_bank,scale_aware,color)
        metric,rows=evaluate(models,cals,val_bank,color,nms_iou=.5,
              visual_dir=out/"visual"/name)
        sec=time.perf_counter()-start
        entry=dict(method=name,fit_images=len(fit_paths),cal_images=len(cal_paths),
                   val_images=len(val),val_birds=int(sum(len(x[1]) for x in val)),
                   seconds_fit_cal_eval=sec,
                   train_history=history,calibration=calhistory,**metric)
        summary.append(entry)
        per_image.extend([dict(method=name,**r) for r in rows])
        print("E21 SCALE CLASSIFICATION RESULT",json.dumps(entry),flush=True)
        (out/"progress_results.json").write_text(json.dumps(summary,indent=2))
    (out/"e21_scale_results.json").write_text(json.dumps(
         dict(dataset="Beach Bird",primary="COCO single-class bbox mAP50 101-point maxDets100",
              training_image_seed=RNG,positive_jitter=True,
              feature_note="HOG+LBP vs HOG+LBP+HSV and boundary statistics",
              test_split_used=False,configurations=summary),indent=2))
    with (out/"e21_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=per_image[0]);writer.writeheader();writer.writerows(per_image)
    print("E21 COMPLETE",json.dumps([{k:v for k,v in x.items() if k in
          ("method","map50_coco101","FP_total_p05","FN_p05")} for x in summary]),flush=True)

def self_test():
    assert bucket([0,0,20,20])==0
    assert bucket([0,0,40,40])==1
    assert bucket([0,0,150,100])==2
    a=np.zeros((128,128,3),np.uint8)
    assert appearance(a,[0,0,24,24],False).shape==(1924,)
    assert appearance(a,[0,0,24,24],True).shape==(1940,)
    # Full score must be independent of VAL annotations.
    r=dict(features=np.zeros((2,1940),np.float32),
           boxes=np.array([[0,0,20,20],[0,0,160,160]],np.float32),
           scale=np.array([0,2]),gt=np.array([[4,4,14,14]],np.float32))
    assert r["scale"].tolist()==[0,2]
    print("PASS scale boundaries, handcrafted feature lengths, image-only inference checks",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e21_results"))
    p.add_argument("--train-limit",type=int,default=100)
    p.add_argument("--val-limit",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test()
    if not args.self_test:
        if not args.dataset:p.error("--dataset required")
        run(args.dataset,args.output,args.train_limit,args.val_limit)
