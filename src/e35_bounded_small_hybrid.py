"""E35: bounded small-bird LoG supplement to Selective Search.
Avoid E31-SS's failed unrestricted union. No TEST, neural nets, or GT-driven
proposals. Same TRAIN/VAL, downstream features and fit for each proposal domain.
"""
from pathlib import Path
import argparse,json,csv,time
import cv2,numpy as np
from cloud_audit import data,iou
from quality_subset_train import training_subset
from classical_methods_study import blob_boxes
from rebuild_detector import descriptor,safe_labels
from background_rejector import secondary_features
from selective_search_detection import selective_boxes,prepare,fit_models,eval_method,coverage

def small_supplement(image,cap=160,quota_per_tile=7):
    ss=selective_boxes(image,1200)
    log,values=blob_boxes(image,"otsu",max_peaks=120)
    if not len(log):return ss
    area=np.prod(log[:,2:]-log[:,:2],axis=1)
    chosen=np.flatnonzero((area>=16)&(area<1024))
    chosen=chosen[np.argsort(-values[chosen],kind="stable")]
    selected=[];counts={}
    for i in chosen:
        b=log[i]
        tile=(int((b[0]+b[2])/128),int((b[1]+b[3])/128))
        if counts.get(tile,0)>=quota_per_tile:continue
        if len(ss) and np.max(iou(b.reshape(1,4),ss))>=.75:continue
        if len(selected) and np.max(iou(b.reshape(1,4),np.array(selected)))>=.75:continue
        selected.append(b);counts[tile]=counts.get(tile,0)+1
        if len(selected)>=cap:break
    return np.vstack((ss,np.asarray(selected,dtype=np.float32).reshape(-1,4))).astype(np.float32)

def prepare_hybrid(samples,label):
    records=[]
    for k,(path,gt) in enumerate(samples,1):
        image=cv2.imread(str(path));t=time.perf_counter()
        b=small_supplement(image)
        sec=time.perf_counter()-t
        X=np.stack([descriptor(image,box) for box in b]) if len(b) else np.empty((0,1924),np.float32)
        aux=np.stack([secondary_features(image,box) for box in b]) if len(b) else np.empty((0,32),np.float32)
        pos,neg=safe_labels(b,gt)
        ov=iou(b,gt);best=ov.max(axis=0) if len(b) else np.zeros(len(gt))
        records.append(dict(path=path,gt=gt,boxes=b,X=X,aux=aux,pos=pos,neg=neg,coverage=best,seconds=sec))
        if k%10==0 or k==len(samples):print("E35 BANK",label,k,len(samples),flush=True)
    return records

def run(dataset,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(dataset,train_n);val=data(dataset,"val",val_n)
    results=[];rows=[]
    for name in ("selective","bounded_small_hybrid"):
        print("E35 RUN",name,flush=True)
        if name=="selective":
            tr=prepare(train,"selective","TRAIN")
            va=prepare(val,"selective","VAL")
        else:
            tr=prepare_hybrid(train,"TRAIN")
            va=prepare_hybrid(val,"VAL")
        stage1,rbf,diag=fit_models(tr)
        metrics,per=eval_method(va,stage1,rbf,name,out)
        record=dict(method=name,training=diag,proposal=coverage(va),detection=metrics,
            train_images=len(tr),val_images=len(va),gt=sum(len(x["gt"]) for x in va))
        results.append(record);rows.extend(per)
        (out/"e35_partial.json").write_text(json.dumps(results,indent=2))
        print("E35 METHOD RESULT",json.dumps(record),flush=True)
    with (out/"e35_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    baseline,new=results
    bm=baseline["detection"];nm=new["detection"]
    bsmall=bm["post_nms_top100_coverage50_by_size"]["small"]
    nsmall=nm["post_nms_top100_coverage50_by_size"]["small"]
    bcrowd=sum(r["top20_TP"] for r in rows if r["method"]=="selective" and r["GT"]>=20)
    ncrowd=sum(r["top20_TP"] for r in rows if r["method"]=="bounded_small_hybrid" and r["GT"]>=20)
    gate=(nm["map50_coco101"]>=bm["map50_coco101"]+.02 and nm["top20_TP"]>=bm["top20_TP"]
          and nsmall>=bsmall-.03 and ncrowd>=bcrowd)
    summary=dict(status="E35_VAL_ONLY",train_n=len(train),val_n=len(val),
        val_gt=sum(len(g) for _,g in val),small_object_supplement_limit=160,
        spatial_tile_quota=7,variant="SS fast + image-only <=160 small LoG candidates",
        gate_pass=gate,results=results,
        caveats=["Same train and val IDs, independently fit models on each proposal domain",
                 "Strictly GT-blind proposals, TEST untouched",
                 "Avoid unrestricted E31 union, which previously reduced detection mAP",
                 "VAL score not sufficient for TEST promotion, requires complementary VAL confirmation"])
    (out/"e35_results.json").write_text(json.dumps(summary,indent=2))
    print("E35 COMPLETE",json.dumps(dict(gate_pass=gate,baseline_map=bm["map50_coco101"],
      bounded_map=nm["map50_coco101"],crowd_tp=(bcrowd,ncrowd))),flush=True)

def tests():
    im=np.zeros((80,80,3),np.uint8)
    cv2.rectangle(im,(10,10),(35,32),(255,255,255),-1)
    boxes=small_supplement(im,cap=30,quota_per_tile=4)
    assert boxes.ndim==2 and boxes.shape[1]==4
    if len(boxes):assert (boxes[:,2:]>boxes[:,:2]).all()
    print("E35 bounded GT-free small proposal geometry PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e35_results"))
    p.add_argument("--train-n",type=int,default=100)
    p.add_argument("--val-n",type=int,default=60)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
