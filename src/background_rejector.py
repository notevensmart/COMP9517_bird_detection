"""E24 — targeted false-positive rejection, classical ML only.

Small paired experiment: 45 quality-subset TRAIN, 12 frozen VAL.
Stage 1: identical GT+jitter HOG/LBP SVM, random safe negatives and ONE
hard-negative round. Stage 2: RandomForest on TRAIN-only *proposal-domain*
positives and post-NMS hard false alarms. No validation scores or GT enter
fitting. All candidates/cache shared. Two operating configurations:
 - unchanged stage1 baseline
 - proposal-level stage2 score reranking based on stage1 margin plus
   RGB/HSV occupancy, gradients, LBP response, position and shape features.
Evaluate mAP@0.5 COCO, top-20 FP, recall, NMS and visual top-20 examples.
"""
import argparse,json,csv,time,random
from pathlib import Path
import cv2,numpy as np
from sklearn.ensemble import RandomForestClassifier
from cloud_audit import data,iou,nms
from rebuild_detector import (build_bank,descriptor,cropbox,fit,
    scored_predictions,classification_errors,draw_box)
from quality_subset_train import training_subset,gt_positive_features,sample_initial,mine
from metrics_map import evaluate_map50

def secondary_features(image,box):
    patch=cropbox(image,box)
    h,w=image.shape[:2]
    bh,bw=patch.shape[:2]
    rgb=cv2.resize(patch,(48,48))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
    grey=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    g=grey.astype(np.float32)/255.
    edges=cv2.Canny(grey,50,140)>0
    # 48x48 spatial pyramid, colour and edge features; all image-only.
    hist=[]
    for channel,bins,lim in [(0,6,180),(1,6,256),(2,6,256)]:
        hh=np.histogram(hsv[:,:,channel],bins=bins,range=(0,lim))[0].astype(float)
        hist.extend((hh/hh.sum()).tolist())
    stats=[g.mean(),g.std(),np.percentile(g,10),np.percentile(g,90),
           float(edges.mean()),float(edges[:24].mean()),float(edges[24:].mean()),
           float((hsv[:,:,1]>50).mean()),float((hsv[:,:,2]>170).mean()),
           float((hsv[:,:,2]<60).mean()),
           float(np.log1p(bw*bh)),float(np.log(max(bw,1)/max(bh,1))),
           float((box[0]+box[2])/(2*w)),float((box[1]+box[3])/(2*h))]
    return np.r_[hist,stats].astype(np.float32)

def prepare_secondary_banks(bank,stage1):
    for k,rec in enumerate(bank):
        image=cv2.imread(str(rec["path"]))
        rec["margin"]=stage1.decision_function(rec["X"]) if len(rec["boxes"]) else np.empty(0)
        rec["aux"]=np.stack([secondary_features(image,b) for b in rec["boxes"]]) if len(rec["boxes"]) else np.empty((0,32),np.float32)
        assert rec["aux"].shape==(len(rec["boxes"]),32)
        if (k+1)%15==0:print("E24 AUX",k+1,"/",len(bank),flush=True)
    return bank

def stage2_train(trainbank):
    """Train on TRAIN-proposal positives (IoU>=.5) vs safely negative, with
    emphasis on TRAIN full-image stage-1 high-confidence NMS false alarms.
    Ambiguous proposal IoU in (0.1,0.5) is never used as negative."""
    Xp=[];Xn=[];rng=np.random.default_rng(77)
    diagnostics=dict(positive=0,hard_negative=0,random_negative=0,negative_images=0)
    for r in trainbank:
        if not len(r["boxes"]):continue
        pos=r["pos"]
        # Cover true-bird candidate domain instead of GT-perfect crops.
        if len(pos)>45:pos=rng.choice(pos,45,replace=False)
        if len(pos):Xp.extend(np.c_[r["margin"][pos],r["aux"][pos]])
        scores=r["margin"]
        keep=nms(r["boxes"],scores,iou_limit=.5,max_det=170)
        safe=set(r["neg"].tolist())
        hard=[int(j) for j in keep if int(j) in safe and scores[int(j)]>-.7][:25]
        selected=set(hard)
        rest=[int(j) for j in r["neg"] if int(j) not in selected]
        random_neg=rng.choice(rest,min(12,len(rest)),replace=False).tolist() if rest else []
        negative=hard+random_neg
        if negative:Xn.extend(np.c_[scores[negative],r["aux"][negative]])
        diagnostics["positive"]+=len(pos)
        diagnostics["hard_negative"]+=len(hard)
        diagnostics["random_negative"]+=len(random_neg)
        diagnostics["negative_images"]+=bool(negative)
    Xp=np.asarray(Xp,np.float32).reshape(-1,33)
    Xn=np.asarray(Xn,np.float32).reshape(-1,33)
    if len(Xp)<30 or len(Xn)<100:raise RuntimeError(f"No stage2 usable proposals: {len(Xp)} pos, {len(Xn)} neg")
    X=np.vstack([Xp,Xn]);y=np.r_[np.ones(len(Xp)),np.zeros(len(Xn))]
    model=RandomForestClassifier(n_estimators=120,max_depth=9,min_samples_leaf=5,
                                 max_features="sqrt",class_weight="balanced_subsample",
                                 n_jobs=2,random_state=31)
    model.fit(X,y)
    diagnostics.update(stage2_train_patch_accuracy=float((model.predict(X)==y).mean()))
    print("E24 STAGE2 TRAIN",json.dumps(diagnostics),flush=True)
    return model,diagnostics

def evaluate_variant(valbank,stage1,stage2,alpha,out):
    preds=[];visual=[];rows=[]
    for r in valbank:
        b=r["boxes"];original=r["margin"]
        # Rank-based fusion; zero stage2 weight must reproduce baseline scores
        if stage2 is None or alpha==0:
            score=original.copy()
        else:
            prob=np.clip(stage2.predict_proba(np.c_[original,r["aux"]])[:,1],1e-4,1.-1e-4)
            logit=np.log(prob/(1-prob))
            score=(1-alpha)*original+alpha*logit
        keep=nms(b,score,iou_limit=.35,max_det=300) if len(b) else np.empty(0,dtype=int)
        final=b[keep];fs=score[keep]
        preds.append((final,fs,r["gt"]))
        active=fs>=0
        errors=classification_errors(final[active],fs[active],r["gt"])
        top=final[:20]
        ov=iou(top,r["gt"])
        used=set();top_tp=0;top_fp=0
        for i in range(len(top)):
            order=np.argsort(-ov[i]) if len(r["gt"]) else []
            matched=False
            for j in order:
                if ov[i,j]>=.5 and int(j) not in used:
                    used.add(int(j));top_tp+=1;matched=True;break
            if not matched:top_fp+=1
        rows.append(dict(image=r["path"].name,GT=len(r["gt"]),TP_top20=top_tp,FP_top20=top_fp,
                         margin0_predictions=int(active.sum()),**errors))
        visual.append(dict(path=r["path"],gt=r["gt"],boxes=final,scores=fs))
    result=evaluate_map50(preds)
    result.update(TP_top20=sum(x["TP_top20"] for x in rows),
                  FP_top20=sum(x["FP_top20"] for x in rows),
                  FP_margin0=sum(x["FP_total"] for x in rows),
                  FN_margin0=sum(x["FN"] for x in rows),
                  threshold_note="FP/FN at fused score>=0, different scale across variants; top20 and mAP are primary")
    # Save visual samples for all variants, irrespective of metric; same images.
    out.mkdir(parents=True,exist_ok=True)
    chosen=[v for v in visual if v["path"].name in (
        "beach_bird_1379.jpg","beach_bird_1419.jpg","beach_bird_1560.jpg","beach_bird_1606.jpg")]
    if len(chosen)<4:chosen=visual[:4]
    panels=[]
    for v in chosen:
        image=cv2.imread(str(v["path"]))
        boxes=v["boxes"][:20];sc=v["scores"][:20];ov=iou(boxes,v["gt"])
        matched=set()
        for i,b in enumerate(boxes):
            choices=[int(j) for j in np.argsort(-ov[i]) if ov[i,j]>=.5 and int(j) not in matched] if len(v["gt"]) else []
            if choices:
                matched.add(choices[0]);color=(255,0,0)
            else:color=(0,0,255)
            draw_box(image,b,color,2)
        for j,b in enumerate(v["gt"]):
            draw_box(image,b,(0,255,0) if j in matched else (0,255,255),2)
        cv2.putText(image,v["path"].name,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        cv2.imwrite(str(out/(v["path"].stem+".jpg")),image)
        panels.append(cv2.resize(image,(560,350)))
    if len(panels)>=2:
        if len(panels)%2:panels.pop()
        cv2.imwrite(str(out/"top20_contact.jpg"),
                    np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))
    return result,rows

def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    trainbank=build_bank(train,"TRAIN",out);valbank=build_bank(val,"VAL",out)
    positives=gt_positive_features(trainbank)
    neg,_=sample_initial(trainbank,"baseline",20)
    used=set()
    # Fixed baseline round 1 was E23's best round: train initial, mine once, fit.
    base,_=fit(positives,neg)
    extra,_=mine(base,trainbank,"baseline",used,250)
    if len(extra):neg=np.vstack([neg,extra])
    base,_=fit(positives,neg)
    allbank=prepare_secondary_banks(trainbank+valbank,base)
    trainbank=allbank[:len(trainbank)];valbank=allbank[len(trainbank):]
    secondary,diagnostics=stage2_train(trainbank)
    configs=[("A_base",None,0),("B_cascade25",secondary,.25),
             ("C_cascade50",secondary,.50),("D_cascade75",secondary,.75)]
    allresults=[];allrows=[]
    for name,model,weight in configs:
        metric,rows=evaluate_variant(valbank,base,model,weight,out/"visual"/name)
        record=dict(name=name,alpha=weight,**metric)
        allresults.append(record)
        allrows.extend([dict(variant=name,**r) for r in rows])
        print("E24 CASCADE RESULT",json.dumps(record),flush=True)
        (out/"progress.json").write_text(json.dumps(allresults,indent=2))
    baseline=allresults[0]
    winner=max(allresults[1:],key=lambda r:r["map50_coco101"])
    promising=(winner["map50_coco101"]>baseline["map50_coco101"]+.02 and
               winner["FP_top20"]<baseline["FP_top20"])
    report=dict(train_images=len(train),val_images=len(val),
        validation_birds=sum(len(gt) for _,gt in val),GT_jitter=True,
        source="LoG/Otsu GT-blind proposals HOG+LBP linear SVM + TRAIN-only background rejection RF",
        common_base="E23 best fixed round 1 on same seeded annotated-only 45 TRAIN and 12 VAL",
        negative_provenance="45 TRAIN images all have annotated birds; strict safe negatives IoU<.1 and containment<.5",
        stage2=diagnostics,all_variants=allresults,winner_exploratory=winner["name"],
        promising_for_larger_study=promising,
        caveats=["Alpha is selected on exploratory VAL; full final test must not reuse this VAL for unreported tuning.",
                 "Stage2 sees TRAIN proposal positives, not exact GT positives.",
                 "Only rank-based false alarms in top20 and mAP can establish benefit.",
                 "All validation photos are saved with top20 boxes; TEST untouched."])
    (out/"e24_results.json").write_text(json.dumps(report,indent=2))
    with (out/"e24_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=allrows[0]);wr.writeheader();wr.writerows(allrows)
    print("E24 COMPLETE",json.dumps(report)[:5500],flush=True)

def self_test():
    image=np.zeros((96,96,3),np.uint8)
    assert secondary_features(image,[5,5,55,55]).shape==(32,)
    from rebuild_detector import tests
    tests()
    print("PASS 32-dimensional classical background descriptor, GT crop, jitter and SVM labeling")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e24_results"))
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=12)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();self_test()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
