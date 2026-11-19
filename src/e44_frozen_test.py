"""E44: strict frozen E43 model evaluation on the complete official TEST partition.

No training, no validation fitting, no score threshold tuning, no parameter
selection, and no mutation of the joblib model files. Evaluates independent
frozen DPM, Fisher, and ACF E43 weights on all 240 TEST images.
Because official TEST was already consulted in E33, this is an explicitly
post-development test evaluation, not an independent unbiased holdout.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import cv2,joblib
import numpy as np
import e42_paper_classical as e42
from cloud_audit import data,iou,nms
from metrics_map import evaluate_map50,self_test as metric_tests

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def score_frozen(weights,dataset,out,method,cap=1200,nms_iou=.35):
    out.mkdir(parents=True,exist_ok=True)
    before=file_hash(weights)
    model=joblib.load(weights)
    samples=data(dataset,"test",0)
    if len(samples)!=240:raise RuntimeError(f"Expected full 240-image official TEST, got {len(samples)}")
    selected={"beach_bird_1700.jpg","beach_bird_1730.jpg","beach_bird_1750.jpg",
              "beach_bird_1800.jpg","beach_bird_1850.jpg","beach_bird_1900.jpg"}
    images_dir=out/"visual";images_dir.mkdir(exist_ok=True)
    predictions=[];rows=[];times=[];counts={k:[0,0,0] for k in ("all","small","medium","large")}
    total_proposals=0
    for index,(path,gt) in enumerate(samples,1):
        tic=time.perf_counter()
        image=cv2.imread(str(path))
        if image is None:raise ValueError(f"Could not read TEST image {path}")
        boxes=e42.stable_ss(image,cap)
        features=[e42.extract(e42.crop(image,b),method) for b in boxes]
        scores=np.asarray(e42.score(model,features,method),np.float32)
        if scores.shape!=(len(boxes),) or not np.isfinite(scores).all():
            raise RuntimeError(f"Bad scores {path}: {scores.shape}")
        keep=nms(boxes,scores,nms_iou,300) if len(boxes) else np.empty(0,int)
        det_boxes,det_scores=boxes[keep],scores[keep]
        duration=time.perf_counter()-tic
        times.append(duration);total_proposals+=len(boxes)
        predictions.append((det_boxes,det_scores,gt))

        # Ground truth is used only AFTER inference, never proposal selection.
        overlaps=iou(boxes,gt)
        best=overlaps.max(axis=0) if len(boxes) else np.zeros(len(gt))
        for b,overlap in zip(gt,best):
            area=np.prod(b[2:]-b[:2])
            size="small" if area<1024 else ("medium" if area<10000 else "large")
            for k in ("all",size):
                counts[k][0]+=1;counts[k][1]+=int(overlap>=.5);counts[k][2]+=int(overlap>=.75)
        matches=iou(det_boxes[:20],gt)
        seen=set();correct=[]
        for hitrow in matches:
            match=False
            for j in np.argsort(-hitrow):
                if hitrow[j]>=.5 and int(j) not in seen:
                    seen.add(int(j));match=True;break
            correct.append(match)
        rows.append(dict(method=method,image=path.name,gt=len(gt),
                         proposals=len(boxes),detections=len(det_boxes),
                         top20_tp=sum(correct),top20_fp=len(correct)-sum(correct),
                         inference_seconds=duration))
        if path.name in selected or (index in (1,40,80,120,160,200,240)):
            view=image.copy()
            for b in gt:
                x1,y1,x2,y2=b.astype(int)
                cv2.rectangle(view,(x1,y1),(x2,y2),(0,220,0),2)
            for b,match in zip(det_boxes[:20],correct):
                x1,y1,x2,y2=b.astype(int)
                cv2.rectangle(view,(x1,y1),(x2,y2),(255,0,0) if match else (0,0,255),2)
            cv2.imwrite(str(images_dir/(method+"_"+path.stem+".jpg")),view)
        if index%20==0 or index==len(samples):
            (out/"progress.json").write_text(json.dumps(
                dict(method=method,stage="TEST inference",processed=index,total=len(samples)),indent=2))
            print("E44 TEST PROGRESS",method,index,"/",len(samples),flush=True)
    metrics=evaluate_map50(predictions)
    after=file_hash(weights)
    if after!=before:raise AssertionError("Frozen model bytes changed during evaluation")
    recalls={k:dict(gt=v[0],R50=v[1]/v[0] if v[0] else None,
                    R75=v[2]/v[0] if v[0] else None) for k,v in counts.items()}
    summary=dict(experiment="E44",method=method,
         label="Post-development evaluation on previously used official TEST; not independent",
         model_source="Saved E43 model artifact, downloaded by actions/download-artifact",
         model_sha256_before=before,model_sha256_after=after,
         refit=False,changed_parameters=False,
         test_images=len(samples),
         test_ids=[path.name for path,_ in samples],
         proposal="Frozen E43 deterministic unique-then-stratified Selective Search",cap=cap,
         nms_iou=nms_iou,mean_proposals=total_proposals/len(samples),
         proposal_recall=recalls,mean_inference_seconds=float(np.mean(times)),
         **metrics,top20_TP=sum(r["top20_tp"] for r in rows),
         top20_FP=sum(r["top20_fp"] for r in rows))
    (out/"results.json").write_text(json.dumps(summary,indent=2))
    with (out/"per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("E44 COMPLETE",json.dumps({k:v for k,v in summary.items() if k not in ("test_ids",)}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=("dpm","fisher","acf"),required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--weights",type=Path)
    p.add_argument("--output",type=Path,default=Path("e44_results"))
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        metric_tests()
        im=np.zeros((128,128,3),np.uint8)
        assert np.array_equal(e42.stable_ss(im,50),e42.stable_ss(im,50))
        print("E44 PASS metrics and GT-blind deterministic proposal check")
    else:
        if args.dataset is None or args.weights is None:p.error("--dataset and --weights required")
        score_frozen(args.weights,args.dataset,args.output,args.method)
