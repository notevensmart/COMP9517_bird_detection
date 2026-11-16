"""E23: FAST quality-subset classical detector / negative-sampling ablation.

Use only TRAIN images with >=1 annotated bird, no suspected empty-label
TRAIN negatives. Paired baseline vs spatially/scale-diverse negatives;
same GT+jitter positives, LoG+Otsu boxes, HOG+LBP, linear SVM and
independent fixed VAL. Reuse one feature bank across all methods and mining
rounds. Do not upscale without a credible mAP + visual improvement.

GT never influences image-only proposals. TEST is not accessed.
"""
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np,cv2
from cloud_audit import data,nms,iou
from rebuild_detector import (build_bank,make_jitter,descriptor,fit,
  scored_predictions,save_visuals,tests as legacy_tests)

SEED=61

def training_subset(root,n):
    train=data(root,"train",0)
    eligible=[x for x in train if len(x[1])>0]
    assert len(eligible)>=n
    return sorted(random.Random(SEED).sample(eligible,n),key=lambda v:v[0].name)

def gt_positive_features(bank):
    rng=np.random.default_rng(47)
    pos=[]
    for rec in bank:
        image=cv2.imread(str(rec["path"]))
        for ground in rec["gt"]:
            boxes=make_jitter(ground,image.shape[1],image.shape[0],rng,count=3)
            for b in boxes:
                assert float(iou(np.asarray(b).reshape(1,4),ground.reshape(1,4))[0,0])>=.55-1e-6
                pos.append(descriptor(image,b))
    if not pos:raise ValueError("No TRAIN positives")
    return np.stack(pos)

def bin_label(rec,idx,tiles=4):
    h,w=rec["image_shape"]
    b=rec["boxes"][idx]
    cx=(b[0]+b[2])*.5;cy=(b[1]+b[3])*.5
    tilex=min(tiles-1,int(cx/max(w,1)*tiles))
    tiley=min(tiles-1,int(cy/max(h,1)*tiles))
    area=float((b[2]-b[0])*(b[3]-b[1]))
    size=0 if area<1024 else (1 if area<10000 else 2)
    return tiley*tiles+tilex,size

def sample_initial(bank,variant,max_each=20):
    rng=np.random.default_rng(91)
    patches=[];counts=[]
    for r in bank:
        neg=np.asarray(r["neg"],dtype=int)
        if not len(neg):
            counts.append(0);continue
        if variant=="baseline":
            picked=rng.choice(neg,min(max_each,len(neg)),replace=False).tolist()
        else:
            # Same per-image negative budget as baseline, but allocate across
            # distinct tile+size buckets before filling remaining slots.
            groups={}
            for j in neg:groups.setdefault(bin_label(r,int(j)),[]).append(int(j))
            keys=sorted(groups)
            for k in keys:rng.shuffle(groups[k])
            picked=[]
            while len(picked)<max_each and keys:
                fresh=[]
                for k in keys:
                    if len(picked)>=max_each:break
                    if groups[k]:picked.append(groups[k].pop());fresh.append(k)
                keys=[k for k in fresh if groups[k]]
        patches.extend(r["X"][picked])
        counts.append(len(picked))
    if not patches:raise ValueError("Missing negatives")
    return np.stack(patches),dict(images=len(bank),neg_features=sum(counts),per_image=counts)

def mine(model,bank,variant,used,limit=250):
    # Cache TRAIN feature descriptors. Negative crop can be selected only once.
    candidates=[]
    for imgid,r in enumerate(bank):
        margins=model.decision_function(r["X"])
        # Same image-only post-NMS boxes for both arms.
        keep=nms(r["boxes"],margins,iou_limit=.5,max_det=170)
        safe=set(int(v) for v in r["neg"])
        perimage=0
        for index in keep:
            j=int(index)
            if j not in safe or (imgid,j) in used:continue
            candidates.append((float(margins[j]),imgid,j))
            perimage+=1
            if perimage>=12:break
    candidates.sort(key=lambda v:-v[0])
    selected=[]
    perimage={}
    spatial=set()
    for margin,i,j in candidates:
        if margin<-.2:break
        if variant=="quality":
            if perimage.get(i,0)>=5:continue
            b=bank[i]["boxes"][j]
            group=(i,*bin_label(bank[i],j))
            if group in spatial:continue
            # exclude nearby redundant rectangles from same TRAIN image.
            old=[bank[k]["boxes"][x] for _,k,x in selected if k==i]
            if old and np.max(iou(np.asarray(old),b.reshape(1,4)))>.4:continue
            spatial.add(group)
        selected.append((margin,i,j));perimage[i]=perimage.get(i,0)+1
        if len(selected)>=limit:break
    for _,i,j in selected:used.add((i,j))
    X=np.stack([bank[i]["X"][j] for _,i,j in selected]) if selected else np.empty((0,1924),np.float32)
    return X,dict(new=len(X),from_images=len(perimage),positive_margin=int(sum(m>=0 for m,_,_ in selected)),
                  unique_hard_candidates=len(used))

def evaluate(model,bank):
    metric,recs,errs=scored_predictions(model,bank,nms_iou=.35)
    top20_tp=top20_fp=0
    for r in recs:
        b=r["boxes"][:20];s=r["scores"][:20];gt=r["gt"]
        overlaps=iou(b,gt);seen=set()
        for ii in range(len(b)):
            best=np.argsort(-overlaps[ii]) if len(gt) else []
            found=False
            for jj in best:
                if overlaps[ii,jj]>=.5 and int(jj) not in seen:
                    seen.add(int(jj));found=True;break
            if found:top20_tp+=1
            else:top20_fp+=1
    metric.update(top20_TP=top20_tp,top20_FP=top20_fp)
    return metric,recs,errs

def run(root,out,trainn,valn):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,trainn)
    val=data(root,"val",valn)
    print("QUALITY SUBSET",json.dumps(dict(
        eligible_train=len(train),GT_birds=sum(len(g) for _,g in train),
        zero_annotation_train_images=0,validation_images=len(val),
        validation_birds=sum(len(g) for _,g in val))),flush=True)
    trainbank=build_bank(train,"TRAIN",out)
    valbank=build_bank(val,"VAL",out)
    pos=gt_positive_features(trainbank)
    outcomes=[];images={}
    for mode in ("baseline","quality"):
        neg,selection=sample_initial(trainbank,mode,max_each=20)
        used=set()
        hist=[]
        for roundno in range(3):
            model,accuracy=fit(pos,neg)
            metrics,recs,errs=evaluate(model,valbank)
            result=dict(mode=mode,round=roundno,positive_features=len(pos),
                negative_features=len(neg),fit_patch_accuracy=accuracy,**metrics)
            if roundno<2:
                extra,mined=mine(model,trainbank,mode,used,limit=250)
                if len(extra):neg=np.vstack([neg,extra])
                result["mining"]=mined
            hist.append(result)
            print("E23 QUALITY SUBSET METRICS",json.dumps(result),flush=True)
        # Compare final round at a fixed protocol; earlier round values in log.
        final=hist[-1]
        outcomes.extend(hist)
        images[mode]=dict(records={r["path"].name:r for r in recs},errors=errs)
        save_visuals(out/mode,valbank,mode,images)
        (out/"progress.json").write_text(json.dumps(outcomes,indent=2))
    final={mode:[x for x in outcomes if x["mode"]==mode][-1] for mode in ("baseline","quality")}
    improved=(final["quality"]["map50_coco101"]>final["baseline"]["map50_coco101"]
        and final["quality"]["top20_FP"]<final["baseline"]["top20_FP"])
    report=dict(train_images=len(train),validation_images=len(val),
        validation_GT=sum(len(g) for _,g in val),
        negative_source="GT-safe regions in TRAIN images with >=1 annotated bird",
        quarantined_zero_annotation_images="ALL zero-GT TRAIN images excluded from this small diagnostic; known suspect names separately quarantined in label_quality.py",
        positive_source="TRAIN ground-truth and controlled jitter IoU>=.55",
        primary_metric="COCO 101-point single class bbox mAP@0.5 maxDets100",
        matched_training_subset=[p.name for p,_ in train],
        baseline_initial=20,quality_initial=20,
        mining_rounds=2,final=final,history=outcomes,
        gate_promising=improved,
        rationale="Quality must improve final mAP and top-20 false ranking; do not expand on marginal or negative result.",
        caveats=["VAL annotations not changed","TRAIN images with birds can still have incomplete boxes: safe negatives are based on provided GT",
                 "No full training; next step contingent on this diagnostic and visual review"])
    (out/"e23_results.json").write_text(json.dumps(report,indent=2))
    with (out/"e23_rounds.csv").open("w",newline="") as f:
        rows=[{k:v for k,v in item.items() if not isinstance(v,dict)} for item in outcomes]
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    print("E23 COMPLETE",json.dumps(dict(final=final,promising=improved)),flush=True)

def self_test():
    legacy_tests()
    dummy=dict(image_shape=(128,128),boxes=np.array([[0,0,10,10],[115,115,127,127],[20,20,100,100]],dtype=np.float32))
    assert len({bin_label(dummy,i) for i in range(3)})==3
    print("PASS negative tile/size diversity, positive-jitter and clipped crop tests")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--output",type=Path,default=Path("e23_results"))
    parser.add_argument("--train-n",type=int,default=45)
    parser.add_argument("--val-n",type=int,default=12)
    parser.add_argument("--self-test",action="store_true")
    a=parser.parse_args();self_test()
    if not a.self_test:
        if not a.dataset:parser.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
