"""Scientifically tested candidate-budget ablations with per-scale spatial diversity.

Unlike naive global top-k, this allocates a fixed budget independently to each
scale/aspect ratio, then allocates candidates across image tiles. GT is used
only AFTER candidate generation for recall audit.
"""
import argparse,json,csv,time
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data
from anchor_proposals import grid_boxes,best_coverage

def integral_edges(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    return cv2.integral((cv2.Canny(gray,50,140)>0).astype(np.uint8),sdepth=cv2.CV_32S)

def box_density(integ,b):
    q=b.astype(np.int32)
    n=(integ[q[:,3],q[:,2]]-integ[q[:,1],q[:,2]]-integ[q[:,3],q[:,0]]+integ[q[:,1],q[:,0]])
    return n/np.maximum(1,(q[:,2]-q[:,0])*(q[:,3]-q[:,1]))

def shape_spatial_select(boxes,score,budget,w,h,tiles=6):
    if len(boxes)<=budget:return boxes
    centers=(boxes[:,:2]+boxes[:,2:])*.5
    xx=np.minimum(tiles-1,(centers[:,0]/w*tiles).astype(int))
    yy=np.minimum(tiles-1,(centers[:,1]/h*tiles).astype(int))
    tile=yy*tiles+xx
    chosen=[]
    # Aim for coverage of all spatial cells before spending on extra texture.
    # Work only with bins containing proposals.
    present=np.unique(tile)
    quota=max(1,int(np.floor(.7*budget/len(present))))
    for t in present:
        indices=np.flatnonzero(tile==t)
        ranked=indices[np.argsort(-score[indices],kind="stable")]
        chosen.extend(ranked[:quota].tolist())
    chosen=np.asarray(chosen,dtype=int)
    if len(chosen)>budget:
        chosen=chosen[np.argsort(-score[chosen],kind="stable")[:budget]]
    remaining=budget-len(chosen)
    if remaining:
        mask=np.ones(len(boxes),bool);mask[chosen]=False
        extra=np.flatnonzero(mask)
        extra=extra[np.argsort(-score[extra],kind="stable")[:remaining]]
        chosen=np.r_[chosen,extra]
    return boxes[chosen].astype(np.float32)

def generate_budgeted(image,per_shape=100,tiles=6):
    h,w=image.shape[:2];integ=integral_edges(image);out=[]
    raw=0
    for _,_,boxes in grid_boxes(h,w):
        raw+=len(boxes)
        scores=box_density(integ,boxes)
        out.append(shape_spatial_select(boxes,scores,per_shape,w,h,tiles))
    return np.concatenate(out).reshape(-1,4),raw

def audit(dataset,out,limit):
    out.mkdir(parents=True,exist_ok=True)
    records=data(dataset,"val",limit)
    configs=[("spatial_50",50),("spatial_100",100),("spatial_200",200),("spatial_300",300),("spatial_450",450)]
    details=[];metrics=[]
    for path,gt in records:
        im=cv2.imread(str(path))
        small=np.prod(gt[:,2:]-gt[:,:2],axis=1)<1024 if len(gt) else np.zeros(0,bool)
        for name,budget in configs:
            t=time.perf_counter()
            boxes,raw=generate_budgeted(im,budget,tiles=6)
            seconds=time.perf_counter()-t
            best=best_coverage(boxes,gt)
            details.append(dict(image=path.name,variant=name,GT=len(gt),small_GT=int(small.sum()),
                 recalled50=int((best>=.5).sum()),recalled75=int((best>=.75).sum()),
                 small_recalled50=int(((best>=.5)&small).sum()),candidates=len(boxes),seconds=seconds))
    with (out/"spatial_proposal_by_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=details[0]);wr.writeheader();wr.writerows(details)
    for name,budget in configs:
        rows=[r for r in details if r["variant"]==name];gt=sum(r["GT"] for r in rows);small=sum(r["small_GT"] for r in rows)
        metrics.append(dict(variant=name,per_shape=budget,images=len(rows),
           proposal_R50=sum(r["recalled50"] for r in rows)/gt if gt else None,
           proposal_R75=sum(r["recalled75"] for r in rows)/gt if gt else None,
           small_R50=sum(r["small_recalled50"] for r in rows)/small if small else None,
           mean_candidates=float(np.mean([r["candidates"] for r in rows])),
           mean_generation_s=float(np.mean([r["seconds"] for r in rows]))))
    (out/"spatial_budget_results.json").write_text(json.dumps(metrics,indent=2))
    print("SPATIALLY BALANCED PRUNING",json.dumps(metrics,indent=2),flush=True)
    return metrics

def test():
    sample=np.zeros((64,64,3),np.uint8)
    b,raw=generate_budgeted(sample,per_shape=6)
    assert raw>=len(b)>0 and len(b)<=27*6
    assert np.all(b[:,2:]>b[:,:2])
    assert np.isfinite(b).all()
    print("Spatial budget and geometry tests passed",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("artifacts"));p.add_argument("--val-limit",type=int,default=20);p.add_argument("--self-test",action="store_true")
    a=p.parse_args();test()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        audit(a.dataset,a.output,a.val_limit)
