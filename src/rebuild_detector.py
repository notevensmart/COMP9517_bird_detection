"""Classical detector repair: ground-truth positives, jitter, full-image hard-negative mining.

Strictly handcrafted LoG/Otsu + HOG/LBP + linear SVM. NO neural models.
Paired 100-TRAIN / 30-VAL diagnostic, fixed seeds; TEST remains untouched.

Reproducible gates:
 - synthetic YOLO-boundary clipping and all-positive GT crop features
 - identical fixed proposals for methods A/B/C
 - GT+jitter positive IoU >= .55, safe negatives IoU < .1 and containment < .5
 - hard negatives mined from TRAIN full-image post-NMS false alarms only
 - COCO 101-pt single-class bbox mAP@.5 on VAL
 - one-to-one FP taxonomy at fixed SVM margin >=0
 - photos of false/missed/correct boxes, plus training and GT overlays
 - val ground truth used only for scoring/auditing, NEVER model fitting.
"""
import argparse,base64,csv,json,random,time,gc
from pathlib import Path
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,iou,nms
from classical_methods_study import blob_boxes,classification_errors
from hog_lbp_comparison import features
from metrics_map import evaluate_map50

SEED=47; DIM=1924
def cropbox(image,box):
    h,w=image.shape[:2]
    b=np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(np.int64)
    b[0]=np.clip(b[0],0,w);b[2]=np.clip(b[2],0,w)
    b[1]=np.clip(b[1],0,h);b[3]=np.clip(b[3],0,h)
    if b[2]<=b[0] or b[3]<=b[1]:
        raise ValueError(f"invalid crop after clipping: {box}, dimensions {w}x{h}")
    return image[b[1]:b[3],b[0]:b[2]]

def descriptor(image,box):
    patch=cropbox(image,box)
    gray=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
    return features(gray,True).astype(np.float32)

def safe_labels(boxes,gt):
    ov=iou(boxes,gt)
    best=ov.max(axis=1) if len(gt) else np.zeros(len(boxes))
    if len(gt):
        diff=np.maximum(0,np.minimum(boxes[:,None,2:],gt[None,:,2:])
                          -np.maximum(boxes[:,None,:2],gt[None,:,:2]))
        area=diff[:,:,0]*diff[:,:,1]
        a=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
        g=np.prod(gt[:,2:]-gt[:,:2],axis=1)
        contained=(area/np.maximum(np.minimum(a[:,None],g[None,:]),1e-9)).max(axis=1)
    else:contained=np.zeros(len(boxes))
    return np.flatnonzero(best>=.5),np.flatnonzero((best<.1)&(contained<.5))

def make_jitter(gt,w,h,rng,count=3):
    """GT plus perturbations. No YOLO GT may be returned outside image bounds."""
    out=[np.asarray(gt,dtype=np.float32).copy()]
    x1,y1,x2,y2=map(float,gt);bw=x2-x1;bh=y2-y1
    center=np.array([(x1+x2)/2,(y1+y2)/2],float)
    tries=0
    while len(out)<count+1 and tries<60:
        tries+=1
        offset=np.array([rng.uniform(-.12,.12)*bw,rng.uniform(-.12,.12)*bh])
        scale=rng.uniform(.88,1.18)
        sx=bw*scale*rng.uniform(.94,1.06);sy=bh*scale*rng.uniform(.94,1.06)
        c=center+offset
        b=np.array([c[0]-sx/2,c[1]-sy/2,c[0]+sx/2,c[1]+sy/2],np.float32)
        b[0::2]=np.clip(b[0::2],0,w)
        b[1::2]=np.clip(b[1::2],0,h)
        if b[2]-b[0]<2 or b[3]-b[1]<2:continue
        if iou(b.reshape(1,4),np.asarray(gt).reshape(1,4))[0,0]<.55:continue
        out.append(b)
    return np.asarray(out,np.float32)

def build_bank(samples,phase,out,maxpeaks=120):
    bank=[];start=time.perf_counter()
    for k,(p,gt) in enumerate(samples):
        im=cv2.imread(str(p))
        if im is None:raise RuntimeError(f"Image missing {p}")
        t=time.perf_counter();boxes,_=blob_boxes(im,"otsu",max_peaks=maxpeaks)
        prop_time=time.perf_counter()-t
        pos,neg=safe_labels(boxes,gt)
        t=time.perf_counter()
        X=np.stack([descriptor(im,b) for b in boxes]) if len(boxes) else np.empty((0,DIM),np.float32)
        feat_time=time.perf_counter()-t
        bank.append(dict(path=p,gt=gt,image_shape=im.shape[:2],boxes=boxes,X=X,
                         pos=pos,neg=neg,proposal_seconds=prop_time,feature_seconds=feat_time))
        if (k+1)%20==0 or k+1==len(samples):
            print("BANK",phase,k+1,"/",len(samples),"elapsed_sec",round(time.perf_counter()-start,1),
                  "positive_proposals",sum(len(x["pos"]) for x in bank),flush=True)
    return bank

def make_train_features(bank, out):
    rng=np.random.default_rng(SEED)
    pos_can=[];neg_can=[];pos_gt=[]
    candidate_positives=[];hard_pool=[]
    counters=dict(gt_birds=0,gt_jitter_features=0,proposal_positive_features=0,
                  initial_safe_negatives=0,total_safe_negative_pool=0,birds_without_proposal=0)
    for r in bank:
        im=cv2.imread(str(r["path"]))
        p=r["pos"];n=r["neg"]
        ov=iou(r["boxes"],r["gt"])
        cover=ov.max(axis=0) if len(r["boxes"]) else np.zeros(len(r["gt"]))
        counters["birds_without_proposal"]+=int((cover<.5).sum())
        counters["gt_birds"]+=len(r["gt"])
        if len(p):
            take=rng.choice(p,min(24,len(p)),replace=False)
            pos_can.extend(r["X"][take])
        neg_seed=rng.choice(n,min(20,len(n)),replace=False) if len(n) else np.empty(0,int)
        if len(neg_seed):neg_can.extend(r["X"][neg_seed])
        hard_pool.append(r)
        counters["total_safe_negative_pool"]+=len(n)
        for box in r["gt"]:
            jj=make_jitter(box,im.shape[1],im.shape[0],rng,count=3)
            for b in jj:pos_gt.append(descriptor(im,b))
    counters["gt_jitter_features"]=len(pos_gt)
    counters["proposal_positive_features"]=len(pos_can)
    counters["initial_safe_negatives"]=len(neg_can)
    if not pos_gt or not pos_can or not neg_can:raise RuntimeError("Missing positive/negative training data")
    print("FEATURE LABEL COUNTS",json.dumps(counters),flush=True)
    return np.stack(pos_gt),np.stack(pos_can),np.stack(neg_can),hard_pool,counters

def fit(pos,neg):
    X=np.concatenate([pos,neg]);y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    clf=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",
                       max_iter=10000,random_state=42))
    clf.fit(X,y)
    return clf,float((clf.predict(X)==y).mean())

def hard_negatives(model,trainbank,limit=600,per_image=10):
    """Mine full TRAIN image detections after NMS; only safe negatives, diverse IoU."""
    candidates=[];ranked_count=0
    for image_id,r in enumerate(trainbank):
        if not len(r["boxes"]):continue
        score=model.decision_function(r["X"])
        keep=nms(r["boxes"],score,iou_limit=.5,max_det=200)
        safe=set(r["neg"].tolist())
        count=0
        for k in keep:
            j=int(k)
            if j not in safe:continue
            # Choose high margin and near-margin training false alarms,
            # not merely the highest random background patch.
            ranked_count+=1
            candidates.append((float(score[j]),image_id,j))
            count+=1
            if count>=per_image:break
    candidates.sort(key=lambda t:-t[0])
    selected=candidates[:limit]
    X=np.stack([trainbank[i]["X"][j] for _,i,j in selected]) if selected else np.empty((0,DIM),np.float32)
    return X,dict(post_nms_safe_candidates=ranked_count,mined_features=len(X),
                  mined_positive_margin=int(sum(score>=0 for score,_,_ in selected)),
                  mined_images=len(set(i for _,i,_ in selected)))

def scored_predictions(model,bank,nms_iou=.35):
    records=[];predictions=[];errors=[]
    for r in bank:
        t=time.perf_counter()
        b=r["boxes"]
        score=model.decision_function(r["X"]) if len(b) else np.empty(0,float)
        keep=nms(b,score,iou_limit=nms_iou,max_det=300) if len(b) else np.empty(0,int)
        out_b=b[keep];out_s=score[keep]
        predictions.append((out_b,out_s,r["gt"]))
        active=out_s>=0.
        fp=classification_errors(out_b[active],out_s[active],r["gt"])
        errors.append(dict(image=r["path"].name,gt=len(r["gt"]),candidates=len(b),nms=len(out_b),
             final_margin0=int(active.sum()),**fp,score_seconds=time.perf_counter()-t))
        records.append(dict(path=r["path"],gt=r["gt"],boxes=out_b,scores=out_s,
                       proposals=b,proposal_scores=score,all_gt_best_iou=(
                       iou(b,r["gt"]).max(axis=0).tolist() if len(b) else [0.]*len(r["gt"]))))
    metric=evaluate_map50(predictions)
    metric.update(FP_at_margin0=sum(x["FP_total"] for x in errors),
        FP_background=sum(x["FP_background"] for x in errors),
        FP_localization=sum(x["FP_localization"] for x in errors),
        FP_duplicates=sum(x["FP_duplicates"] for x in errors),
        FN_at_margin0=sum(x["FN"] for x in errors),
        mean_margin0_boxes=float(np.mean([x["final_margin0"] for x in errors])),
        empty_image_fp_at_margin0=sum(x["FP_total"] for x in errors if x["gt"]==0),
        mean_scoring_sec=float(np.mean([x["score_seconds"] for x in errors])))
    return metric,records,errors

def draw_box(canvas,b,color,thickness=2):
    h,w=canvas.shape[:2]
    a=np.round(b).astype(int)
    a[0::2]=np.clip(a[0::2],0,w-1);a[1::2]=np.clip(a[1::2],0,h-1)
    cv2.rectangle(canvas,(a[0],a[1]),(a[2],a[3]),color,thickness)

def decorate(record,variant,mode="detections"):
    im=cv2.imread(str(record["path"]))
    gt=record["gt"]
    if mode=="proposals":
        b=record["proposals"];ov=iou(b,gt)
        for i in range(len(b)):
            if len(gt) and ov[i].max()>=.5:draw_box(im,b[i],(255,0,0),1)
        for v in gt:draw_box(im,v,(0,255,0),2)
    else:
        pred=record["boxes"];scores=record["scores"]
        active=np.flatnonzero(scores>=0.)
        ov=iou(pred,gt)
        used=set()
        for idx in active:
            sim=ov[idx] if len(gt) else np.empty(0)
            viable=sorted((float(sim[k]),int(k)) for k in range(len(gt))
                    if sim[k]>=.5 and k not in used)
            if viable:
                used.add(viable[-1][1]);color=(255,0,0) # blue true positive
            else:color=(0,0,255) # red false alarm
            draw_box(im,pred[idx],color,2)
        for k,v in enumerate(gt):
            draw_box(im,v,(0,255,0) if k in used else (0,255,255),2) # green TP, yellow missed
    im=cv2.resize(im,(512,320))
    title=f"{variant}  {record['path'].name} GT={len(gt)}"
    if mode=="detections":title+=f"  predicted={(record['scores']>=0).sum()}"
    cv2.rectangle(im,(0,0),(512,27),(10,10,10),-1)
    cv2.putText(im,title[:75],(7,19),cv2.FONT_HERSHEY_SIMPLEX,.41,(255,255,255),1,cv2.LINE_AA)
    return im

def save_visuals(artifacts,bank,selected_variant,report):
    """Visuals use validation GT for ANALYSIS ONLY and are not inputs to model."""
    output=artifacts/"visual"
    output.mkdir(parents=True,exist_ok=True)
    vals={r["path"].name:r for r in bank}
    results=report[selected_variant]["errors"]
    selected=[]
    # Cover crowded, negative, many false alarms, poor recall and easy scenes.
    ranked=sorted(results,key=lambda r:-r["FP_total"])
    for row in ranked[:2]:selected.append(row["image"])
    for row in sorted(results,key=lambda r:-r["GT"])[:2]:selected.append(row["image"])
    for row in sorted(results,key=lambda r:r["GT"])[:2]:selected.append(row["image"])
    for row in sorted(results,key=lambda r:-r["TP"])[:2]:selected.append(row["image"])
    selected=list(dict.fromkeys(selected))[:6]
    tiles=[];per_image=[]
    for name in selected:
        rec=report[selected_variant]["records"][name]
        image=decorate(rec,selected_variant,"detections")
        tiles.append(image)
        cv2.imwrite(str(output/(Path(name).stem+"_annotated.jpg")),image,[cv2.IMWRITE_JPEG_QUALITY,82])
        per_image.append(next(x for x in results if x["image"]==name))
    canvas=np.vstack([np.hstack(tiles[k:k+2]) for k in range(0,len(tiles),2)]) if len(tiles)%2==0 else None
    if canvas is not None:
        cv2.imwrite(str(output/"contact_sheet.jpg"),canvas,[cv2.IMWRITE_JPEG_QUALITY,75])
        # A compact ASCII JSON screenshot makes it possible to inspect actual
        # images through the connected GitHub text-file action (no OCR).
        ok,jpeg=cv2.imencode(".jpg",canvas,[cv2.IMWRITE_JPEG_QUALITY,55])
        if not ok:raise ValueError("jpeg encoding failed")
        (artifacts/"contact_sheet_embed.json").write_text(json.dumps({
             "type":"image/jpeg","b64":base64.b64encode(jpeg.tobytes()).decode("ascii"),
             "legend":"Blue final TP, red final FP, green matched GT, yellow missed GT",
             "selected_images":selected,"per_image":per_image}))
    # Keep original-size GT/proposals overlays for spotting annotation errors
    for name in selected[:3]:
        r=vals[name]
        im=cv2.imread(str(r["path"]))
        for b in r["gt"]:draw_box(im,b,(0,255,0),2)
        for b in r["boxes"]:
            overlap=iou(np.asarray(b).reshape(1,4),r["gt"])
            if len(r["gt"]) and overlap.max()>=.5:draw_box(im,b,(255,0,0),1)
        cv2.imwrite(str(output/(Path(name).stem+"_gt_and_matching_proposals.jpg")),im)
    return selected

def study(dataset,out,train_limit,val_limit):
    out.mkdir(parents=True,exist_ok=True)
    train=data(dataset,"train",train_limit)
    val=data(dataset,"val",val_limit)
    # data() seeded selection identical to earlier 30/100 experiments.
    trainbank=build_bank(train,"TRAIN",out)
    valbank=build_bank(val,"VAL",out)
    gt_pos,proposal_pos,neg_pool,trainbank,feature_counts=make_train_features(trainbank,out)
    methods=[("A_proposal_positives",proposal_pos,False),
             ("B_GT_jitter_positives",gt_pos,False),
             ("C_GT_jitter_fullimage_mining",gt_pos,True)]
    report={};summary=[];rowlist=[]
    for name,positive,do_mine in methods:
        start=time.perf_counter()
        current_negs=neg_pool
        round_results=[]
        for rd in range(3 if do_mine else 1):
            t=time.perf_counter()
            clf,patch_accuracy=fit(positive,current_negs)
            train_sec=time.perf_counter()-t
            variants={}
            for suppression in (.35,.65):
                metric,recs,errors=scored_predictions(clf,valbank,suppression)
                variants[str(suppression)]=dict(metrics=metric,records={r["path"].name:r for r in recs},
                                               errors=errors)
            main=variants["0.35"]["metrics"]
            details=dict(method=name,round=rd,positive_patches=len(positive),
                         negative_patches=len(current_negs),train_patch_accuracy=patch_accuracy,
                         train_seconds=train_sec,**main,
                         nms65_mAP50=variants["0.65"]["metrics"]["map50_coco101"])
            if do_mine and rd<2:
                hard,hardstats=hard_negatives(clf,trainbank,limit=600,per_image=10)
                current_negs=np.concatenate([current_negs,hard])
                details.update(hardstats)
            round_results.append(details)
            print("REBUILD RESULT",json.dumps(details),flush=True)
        report[name]=dict(records=variants["0.35"]["records"],errors=variants["0.35"]["errors"])
        summary.extend(round_results)
        rowlist.extend([dict(method=name,**r) for r in variants["0.35"]["errors"]])
        gc.collect()
    winner=max((r for r in summary if r["round"]==0 or r["method"]=="C_GT_jitter_fullimage_mining"),
               key=lambda r:r["map50_coco101"])["method"]
    chosen=save_visuals(out,valbank,winner,report)
    metrics={r["method"]+"#round"+str(r["round"]):r for r in summary}
    result=dict(train_images=len(train),validation_images=len(val),
                validation_birds=sum(len(r["gt"]) for r in valbank),
                training_counts=feature_counts,
                models=metrics,chosen_for_visuals=winner,visual_images=chosen,
                method_notes=["All models use exact same LoG+local Otsu proposals and descriptor.",
                "A uses candidate positives only, B/C use ground-truth and jittered GT positive crops.",
                "C mines negatives only on TRAIN full-image post-NMS false alarm candidates.",
                "NMS 0.35 is primary, NMS 0.65 diagnostic only; test split untouched.",
                "Final boxes shown in red are FP at fixed decision margin >=0; yellow GT not matched.",
                "Visuals have VAL labels for diagnostics only, never training."])
    (out/"repair_study.json").write_text(json.dumps(result,indent=2))
    with (out/"repair_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rowlist[0]);writer.writeheader();writer.writerows(rowlist)
    print("REBUILD SUMMARY",json.dumps(summary),flush=True)

def tests():
    im=np.zeros((48,64,3),np.uint8)
    x=np.asarray([-0.000256,8.0,30.1,34.],np.float32)
    assert len(cropbox(im,x))>0
    gt=np.array([2,6,18,24],np.float32)
    rng=np.random.default_rng(SEED)
    jit=make_jitter(gt,64,48,rng,3)
    assert len(jit)==4 and np.min(iou(jit,gt.reshape(1,4)))>=.55
    testb=np.array([[2,6,18,24],[35,32,44,40]],np.float32)
    p,n=safe_labels(testb,gt.reshape(1,4))
    assert p.tolist()==[0] and n.tolist()==[1]
    assert descriptor(im,gt).shape==(DIM,)
    print("PASS boundary clipping, jitter IoU, safe negative, handcrafted feature checks",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("repair_results"))
    ap.add_argument("--train-limit",type=int,default=100)
    ap.add_argument("--val-limit",type=int,default=30)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args();tests()
    if not args.self_test:
        if not args.dataset:ap.error("--dataset required")
        study(args.dataset,args.output,args.train_limit,args.val_limit)
