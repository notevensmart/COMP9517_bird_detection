"""E26: classical bbox refinement + Soft-NMS, paired with frozen E25 cascade.

Pre-register two orthogonal questions:
  1. Can TRAIN-proposal->GT box regression improve IoU on actual VAL true
     candidate matches and official COCO bbox mAP50?
  2. Does Gaussian Soft-NMS help crowded bird scenes without increasing
     empty-scene top20 FP?
Zero VAL GT enters training/inference decisions. Visual PNGs show all stages.
Research background: Girshick et al. 2014 box geometry targets, Bodla et al.
2017 Soft-NMS. No pretrained CNNs, no neural networks.
"""
import argparse,json,csv,time
from pathlib import Path
import cv2,numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cloud_audit import data,iou,nms
from rebuild_detector import build_bank,fit,draw_box,classification_errors
from quality_subset_train import training_subset,gt_positive_features,sample_initial,mine
from background_rejector import prepare_secondary_banks,stage2_train
from metrics_map import evaluate_map50

def encode(P,G):
    P=np.asarray(P,np.float64).reshape(-1,4);G=np.asarray(G,np.float64).reshape(-1,4)
    pw=np.maximum(P[:,2]-P[:,0],1);ph=np.maximum(P[:,3]-P[:,1],1)
    gw=np.maximum(G[:,2]-G[:,0],1);gh=np.maximum(G[:,3]-G[:,1],1)
    px=(P[:,0]+P[:,2])/2;py=(P[:,1]+P[:,3])/2
    gx=(G[:,0]+G[:,2])/2;gy=(G[:,1]+G[:,3])/2
    return np.c_[(gx-px)/pw,(gy-py)/ph,np.log(gw/pw),np.log(gh/ph)]

def decode(P,t,shape):
    P=np.asarray(P,np.float64).reshape(-1,4)
    t=np.clip(np.asarray(t,np.float64).reshape(-1,4),[-.6,-.6,-.9,-.9],[.6,.6,.9,.9])
    h,w=shape
    pw=np.maximum(P[:,2]-P[:,0],1);ph=np.maximum(P[:,3]-P[:,1],1)
    cx=(P[:,0]+P[:,2])/2+t[:,0]*pw;cy=(P[:,1]+P[:,3])/2+t[:,1]*ph
    nw=pw*np.exp(t[:,2]);nh=ph*np.exp(t[:,3])
    b=np.c_[cx-nw/2,cy-nh/2,cx+nw/2,cy+nh/2]
    b[:,0::2]=np.clip(b[:,0::2],0,w)
    b[:,1::2]=np.clip(b[:,1::2],0,h)
    # Retain original on degenerate outputs, rather than introducing bad boxes.
    invalid=((b[:,2]-b[:,0]<2)|(b[:,3]-b[:,1]<2))
    b[invalid]=P[invalid]
    return b.astype(np.float32)

def input_features(rec,indices):
    i=np.asarray(indices,dtype=int)
    # Proposal's fixed 32 secondary handcrafted colour/edge/geometry features
    # + original classifier margin (not full HOG/1924 to avoid high-p overfit).
    return np.c_[rec["margin"][i],rec["aux"][i]].astype(np.float32)

def learn_regressor(trainbank,min_iou=.5,per_image=30):
    xx=[];yy=[];counts=[];rng=np.random.default_rng(75)
    for r in trainbank:
        if not len(r["gt"]) or not len(r["boxes"]):continue
        overlaps=iou(r["boxes"],r["gt"])
        best=overlaps.max(axis=1)
        indexes=np.flatnonzero(best>=min_iou)
        if len(indexes)>per_image:indexes=rng.choice(indexes,per_image,replace=False)
        if not len(indexes):continue
        matched=np.argmax(overlaps[indexes],axis=1)
        target=encode(r["boxes"][indexes],r["gt"][matched])
        xx.append(input_features(r,indexes));yy.append(target)
        counts.append(len(indexes))
    if not xx:raise RuntimeError("No positive close matched TRAIN proposal for regression")
    X=np.vstack(xx);Y=np.vstack(yy)
    model=make_pipeline(StandardScaler(),Ridge(alpha=75.))
    model.fit(X,Y)
    mse=float(np.mean((model.predict(X)-Y)**2))
    return model,dict(training_pairs=len(X),training_images=len(counts),
                      train_mean_square_offset_error=mse,
                      delta_mean=Y.mean(axis=0).tolist(),delta_std=Y.std(axis=0).tolist())

def soft_nms(boxes,scores,iou_thr=.5,sigma=.5,limit=300):
    """Gaussian Soft-NMS, rank-and-decay all candidates, including low scores."""
    b=np.asarray(boxes,np.float32).reshape(-1,4);scores=np.asarray(scores,np.float64).copy()
    left=np.arange(len(b));chosen=[];out_scores=[]
    while len(left) and len(chosen)<limit:
        best=int(np.argmax(scores[left]));index=int(left[best])
        chosen.append(index);out_scores.append(float(scores[index]))
        left=np.delete(left,best)
        if not len(left):break
        overlaps=iou(b[index:index+1],b[left])[0]
        scores[left]*=np.exp(-(overlaps**2)/sigma)
    return np.asarray(chosen,dtype=int),np.asarray(out_scores,dtype=np.float64)

def gt_coverage(boxes,gt):
    ov=iou(boxes,gt)
    return (ov.max(axis=0) if len(boxes) else np.zeros(len(gt),float))

def matched_true_top20(boxes,gt):
    ov=iou(boxes[:20],gt);used=set();tp=0
    for ix in range(min(len(boxes),20)):
        ordered=np.argsort(-ov[ix]) if len(gt) else []
        for j in ordered:
            if ov[ix,j]>=.5 and int(j) not in used:
                used.add(int(j));tp+=1;break
    return tp,min(len(boxes),20)-tp

def draw(image,gt,b,s,name):
    im=image.copy()
    for g in gt:draw_box(im,g,(0,200,0),2)
    ov=iou(b[:20],gt)
    used=set()
    for k,p in enumerate(b[:20]):
        matches=[int(j) for j in np.argsort(-ov[k]) if ov[k,j]>=.5 and j not in used] if len(gt) else []
        if matches:used.add(matches[0]);color=(255,0,0)
        else:color=(0,0,245)
        draw_box(im,p,color,2)
    im=cv2.resize(im,(620,370))
    cv2.rectangle(im,(0,0),(620,27),(8,8,8),-1)
    cv2.putText(im,f"{name} | {len(gt)} GT", (10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return im

def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    trainbank=build_bank(train,"TRAIN",out);valbank=build_bank(val,"VAL",out)
    positives=gt_positive_features(trainbank)
    neg,_=sample_initial(trainbank,"baseline",20)
    stage1,_=fit(positives,neg)
    extra,_=mine(stage1,trainbank,"baseline",set(),250)
    if len(extra):neg=np.vstack([neg,extra])
    stage1,_=fit(positives,neg)
    both=prepare_secondary_banks(trainbank+valbank,stage1)
    trainbank=both[:len(trainbank)];valbank=both[len(trainbank):]
    stage2,stage2stats=stage2_train(trainbank)
    reg,regstats=learn_regressor(trainbank,.5,30)
    print("REGRESSION FIT",json.dumps(regstats),flush=True)
    options={
      "base_nms035":(False,"hard",.35),
      "cascade_nms035":(False,"hard",.35),
      "cascade_refined_nms035":(True,"hard",.35),
      "cascade_refined_nms065":(True,"hard",.65),
      "cascade_softnms":(False,"soft",.5),
      "cascade_refined_softnms":(True,"soft",.5),
    }
    preds={k:[] for k in options};per_img=[];contact={k:[] for k in options}
    geometry=dict(n_eligible_gt=0,baseline_iou_sum=0,refined_iou_sum=0,
                  gt_covered50_before=0,gt_covered50_after=0,
                  gt_covered75_before=0,gt_covered75_after=0)
    viz_names={"beach_bird_1379.jpg","beach_bird_1419.jpg","beach_bird_1560.jpg",
               "beach_bird_1606.jpg","beach_bird_1389.jpg","beach_bird_1443.jpg"}
    for k,r in enumerate(valbank):
        raw=r["boxes"]
        prob=np.clip(stage2.predict_proba(np.c_[r["margin"],r["aux"]])[:,1],1e-4,1-1e-4) if len(raw) else np.empty(0)
        logits=np.log(prob/(1-prob)) if len(prob) else prob
        fused=.5*r["margin"]+.5*logits
        refined=decode(raw,reg.predict(input_features(r,np.arange(len(raw)))) ,cv2.imread(str(r["path"])).shape[:2]) if len(raw) else raw
        before=gt_coverage(raw,r["gt"]);after=gt_coverage(refined,r["gt"])
        geometry["n_eligible_gt"]+=len(before)
        geometry["baseline_iou_sum"]+=float(before.sum())
        geometry["refined_iou_sum"]+=float(after.sum())
        for threshold in (.5,.75):
            geometry[f"gt_covered{int(threshold*100)}_before"]+=int((before>=threshold).sum())
            geometry[f"gt_covered{int(threshold*100)}_after"]+=int((after>=threshold).sum())
        for name,(use_reg,suppress,thr) in options.items():
            b=refined if use_reg else raw
            s=r["margin"] if name=="base_nms035" else fused
            if suppress=="hard":
                ix=nms(b,s,thr,max_det=300) if len(b) else np.empty(0,int)
                bs=b[ix];ss=s[ix]
            else:
                ix,ss=soft_nms(b,s,.5,.5,300)
                bs=b[ix]
            preds[name].append((bs,ss,r["gt"]))
            t,f=matched_true_top20(bs,r["gt"])
            active=ss>=0
            err=classification_errors(bs[active],ss[active],r["gt"])
            per_img.append(dict(image=r["path"].name,variant=name,gt=len(r["gt"]),
                TP_top20=t,FP_top20=f,**err))
            if r["path"].name in viz_names:
                contact[name].append(draw(cv2.imread(str(r["path"])),r["gt"],bs,ss,name))
        if (k+1)%10==0:print("REFINEMENT VAL",k+1,"/",len(valbank),flush=True)
    metrics=[]
    for name,records in preds.items():
        result=evaluate_map50(records)
        rows=[p for p in per_img if p["variant"]==name]
        result.update(variant=name,TP_top20=sum(p["TP_top20"] for p in rows),
          FP_top20=sum(p["FP_top20"] for p in rows),
          FP_margin0=sum(p["FP_total"] for p in rows),
          FN_margin0=sum(p["FN"] for p in rows))
        metrics.append(result)
        print("E26 MODEL",json.dumps(result),flush=True)
    visuals=out/"visual";visuals.mkdir(exist_ok=True)
    for name,panels in contact.items():
        if not panels:continue
        # 2-columns; same image order across every method, 6 visual scenes.
        if len(panels)%2:panels=panels[:-1]
        stacked=np.vstack([np.hstack(panels[j:j+2]) for j in range(0,len(panels),2)])
        cv2.imwrite(str(visuals/(name+".jpg")),stacked,[cv2.IMWRITE_JPEG_QUALITY,80])
    if geometry["n_eligible_gt"]:
        geometry["mean_maxiou_before"]=geometry["baseline_iou_sum"]/geometry["n_eligible_gt"]
        geometry["mean_maxiou_after"]=geometry["refined_iou_sum"]/geometry["n_eligible_gt"]
    payload=dict(protocol="45 annotated TRAIN, 30 fixed VAL, no TEST; train-only Ridge bbox regression and RF cascade",
        stage2=stage2stats,regression=regstats,geometry=geometry,models=metrics,
        caveat="No model selection based on validation GT; all options predeclared; per-variant NMS uses scores only.")
    (out/"e26_results.json").write_text(json.dumps(payload,indent=2))
    with (out/"e26_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=per_img[0]);writer.writeheader();writer.writerows(per_img)
    print("E26 COMPLETE",json.dumps(payload),flush=True)

def tests():
    p=np.array([[10,20,40,80],[0,0,50,50]],np.float32)
    gt=np.array([[13,18,39,71],[0,0,50,50]],np.float32)
    decoded=decode(p,encode(p,gt),(100,100))
    assert np.allclose(decoded,gt,atol=.01)
    b=np.array([[0,0,10,10],[0,0,10,10],[30,30,40,40]],np.float32)
    order,s=soft_nms(b,np.array([4.,3.,2.]))
    assert len(set(order))==3 and s[0]==4 and s[1]==2 and s[2]<3
    print("PASS invertible bbox delta geometry, Soft-NMS monotone decay and bounds",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e26_results"))
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
