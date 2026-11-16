"""Classical bird detector study: LoG+Otsu regions, HOG versus HOG+LBP and hard negatives.

Primary metric is COCO 101-point bbox mAP@0.5, single bird class, maxDets=100.
Train-only learning, GT-blind image proposals, common fixed train/val subsets.
Detections are scored and NMS-filtered; candidate box errors and final FP
are counted separately, including zero-GT validation images.
"""
from pathlib import Path
import argparse,csv,json,time
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,iou,nms
from classical_methods_study import blob_boxes,proposal_audit,classification_errors
from hog_lbp_comparison import features
from metrics_map import evaluate_map50

def feat_both(image,box):
    x1,y1,x2,y2=np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(int)
    patch=image[y1:y2,x1:x2]
    if patch.size==0:raise ValueError("empty candidate")
    grey=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
    f=features(grey,True)
    return f.astype(np.float32)

def prepare(samples,phase,max_peaks):
    bank=[];pstart=time.perf_counter()
    for i,(path,gt) in enumerate(samples):
        image=cv2.imread(str(path));t=time.perf_counter()
        boxes,proposal_scores=blob_boxes(image,"otsu",max_peaks=max_peaks)
        proposal_sec=time.perf_counter()-t
        if len(boxes)==0:raise ValueError("No candidate boxes; log and handle rather than hiding")
        t=time.perf_counter()
        X=np.stack([feat_both(image,b) for b in boxes])
        feature_sec=time.perf_counter()-t
        bank.append(dict(path=path.name,gt=gt,boxes=boxes,X=X,
                    proposal_seconds=proposal_sec,feature_seconds=feature_sec,
                    proposal_stats=proposal_audit(boxes,gt)))
        print(f"BANK {phase} {i+1}/{len(samples)} {path.name} {len(boxes)} candidates",flush=True)
    print("BANK COMPLETE",phase,"images",len(bank),"total_seconds",time.perf_counter()-pstart,flush=True)
    return bank

def labels(boxes,gt):
    overlaps=iou(boxes,gt)
    by_box=overlaps.max(axis=1) if len(gt) else np.zeros(len(boxes))
    if len(gt):
        iw=np.maximum(0,np.minimum(boxes[:,None,2:],gt[None,:,2:])-np.maximum(boxes[:,None,:2],gt[None,:,:2]))
        inter=iw[:,:,0]*iw[:,:,1]
        a=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
        b=np.prod(gt[:,2:]-gt[:,:2],axis=1)
        contained=(inter/np.maximum(np.minimum(a[:,None],b[None,:]),1e-9)).max(axis=1)
    else:contained=np.zeros(len(boxes))
    positive=np.flatnonzero(by_box>=.5)
    negative=np.flatnonzero((by_box<.1)&(contained<.5))
    return positive,negative

def train(bank,dim,with_lbp,mining_rounds):
    rng=np.random.default_rng(44)
    positives=[];negatives=[];offset=0
    for e in bank:
        pi,ni=labels(e["boxes"],e["gt"])
        if len(pi):positives.extend((pi+offset).tolist())
        if len(ni):negatives.extend((rng.choice(ni,min(20,len(ni)),replace=False)+offset).tolist())
        offset+=len(e["boxes"])
    Xfull=np.concatenate([e["X"][:,:dim] for e in bank])
    pos=np.asarray(positives,dtype=int);selected=set(negatives)
    negpool=[];offset=0
    for e in bank:
        _,ni=labels(e["boxes"],e["gt"]);negpool.extend((ni+offset).tolist());offset+=len(e["boxes"])
    negpool=np.asarray(negpool,dtype=int)
    if not len(pos) or not len(selected):raise RuntimeError("Missing positives or negatives")
    results=[]
    for rd in range(mining_rounds+1):
        neg=np.asarray(sorted(selected),int);idx=np.r_[pos,neg]
        y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",
                                               random_state=42,max_iter=10000))
        model.fit(Xfull[idx],y)
        accuracy=float((model.predict(Xfull[idx])==y).mean())
        scores=model.decision_function(Xfull[negpool])
        ranked=negpool[np.argsort(-scores,kind="stable")]
        extras=[]
        for j in ranked:
            if int(j) not in selected:extras.append(int(j))
            if len(extras)>=500:break
        row=dict(round=rd,positives=len(pos),negatives=len(neg),train_patch_accuracy=accuracy,
                 hard_negatives_found=len(extras))
        results.append(row)
        if rd<mining_rounds:selected.update(extras)
    return model,results

def evaluate(model,bank,dim):
    preds=[];rows=[]
    for e in bank:
        t=time.perf_counter()
        b=e["boxes"];s=model.decision_function(e["X"][:,:dim])
        keep=nms(b,s,iou_limit=.35,max_det=300)
        candidate_b=b[keep];candidate_s=s[keep]
        # For official COCO mAP the evaluator scores the highest 100 detections
        # without tuning a threshold on validation.
        preds.append((candidate_b,candidate_s,e["gt"]))
        good=candidate_s>=0.
        actual_b=candidate_b[good];actual_s=candidate_s[good]
        errs=classification_errors(actual_b,actual_s,e["gt"])
        rows.append(dict(image=e["path"],GT=len(e["gt"]),
             candidates=len(b),proposal_background=e["proposal_stats"]["background_candidates"],
             proposal_near_misses=e["proposal_stats"]["mislocalized_candidates"],
             proposal_excess=e["proposal_stats"]["excess_overlapping_candidates"],
             final_boxes=len(actual_b),after_nms=len(keep),**errs,
             scoring_seconds=time.perf_counter()-t,
             proposal_seconds=e["proposal_seconds"],feature_seconds=e["feature_seconds"]))
    return evaluate_map50(preds),rows

def study(dataset,out,train_limit,val_limit,max_peaks):
    out.mkdir(parents=True,exist_ok=True)
    trains=data(dataset,"train",train_limit);vals=data(dataset,"val",val_limit)
    trainbank=prepare(trains,"train",max_peaks)
    valbank=prepare(vals,"val",max_peaks)
    configs=[("HOG_SVM",1764,0),("HOG_LBP_SVM",1924,0),
             ("HOG_LBP_SVM_HNM",1924,2)]
    result=[];allrows=[]
    for name,dim,mining_rounds in configs:
        start=time.perf_counter()
        model,rounds=train(trainbank,dim,name!="HOG_SVM",mining_rounds)
        training_seconds=time.perf_counter()-start
        measures,rows=evaluate(model,valbank,dim)
        entries=dict(method=name,train_images=len(trains),validation_images=len(vals),
           validation_birds=sum(len(e["gt"]) for e in valbank),
           mean_candidates=float(np.mean([len(e["boxes"]) for e in valbank])),
           mean_output_boxes=float(np.mean([r["final_boxes"] for r in rows])),
           total_false_boxes=int(sum(r["FP_total"] for r in rows)),
           FP_background=int(sum(r["FP_background"] for r in rows)),
           FP_localization=int(sum(r["FP_localization"] for r in rows)),
           FP_duplicates=int(sum(r["FP_duplicates"] for r in rows)),
           false_positives_per_image=float(np.mean([r["FP_total"] for r in rows])),
           false_positives_empty_images=int(sum(r["FP_total"] for r in rows if r["GT"]==0)),
           missed_birds=int(sum(r["FN"] for r in rows)),
           mean_inference_seconds=float(np.mean([r["proposal_seconds"]+r["feature_seconds"]+r["scoring_seconds"] for r in rows])),
           training_seconds=training_seconds,
           mining_rounds=rounds,**measures)
        result.append(entries)
        allrows.extend([dict(method=name,**r) for r in rows])
        print("PRIMARY mAP50 RESULT",json.dumps(entries),flush=True)
    (out/"otsu_hog_lbp_map50.json").write_text(json.dumps(result,indent=2))
    with (out/"otsu_hog_lbp_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
    print("CLASSICAL ABLATION COMPLETE",flush=True)

def tests():
    image=np.zeros((64,64,3),np.uint8)
    cv2.circle(image,(30,30),8,(255,255,255),-1)
    vec=feat_both(image,np.array([10,10,50,50]))
    assert len(vec)==1924 and np.isfinite(vec).all()
    a=np.array([[0,0,20,20],[30,30,40,40]],np.float32)
    gt=np.array([[0,0,20,20]],np.float32)
    p,n=labels(a,gt)
    assert p.tolist()==[0] and n.tolist()==[1]
    print("HOG+LBP feature and target tests PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("ablation_results"))
    p.add_argument("--train-limit",type=int,default=35)
    p.add_argument("--val-limit",type=int,default=12)
    p.add_argument("--max-peaks",type=int,default=120)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args();tests()
    if not args.self_test:
        if not args.dataset:p.error("--dataset required")
        study(args.dataset,args.output,args.train_limit,args.val_limit,args.max_peaks)
