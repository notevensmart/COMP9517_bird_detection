"""Stage C: geometry-first, GT-free multi-scale proposals and coverage diagnostics.

Hypothesis: contour-only proposals miss small birds. A dense multi-scale anchor grid
with image-derived edge ranking should increase candidate recall at IoU >= 0.5.
GT boxes are read ONLY by audit, never used in candidate generation.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np, cv2
from cloud_audit import data, iou, proposals

SIZES=(8,12,16,24,32,48,64,96,144)
RATIOS=(0.67,1.0,1.5)
MAX_PER_SHAPE=450

def grid_boxes(h,w,sizes=SIZES,ratios=RATIOS,frac=.25,minimum_stride=3):
    for side in sizes:
        for ratio in ratios:
            bw=max(4,int(round(side*np.sqrt(ratio))))
            bh=max(4,int(round(side/np.sqrt(ratio))))
            if bw>w or bh>h:continue
            step=max(minimum_stride,int(round(min(bw,bh)*frac)))
            xs=np.unique(np.r_[np.arange(0,w-bw+1,step),w-bw]).astype(np.int32)
            ys=np.unique(np.r_[np.arange(0,h-bh+1,step),h-bh]).astype(np.int32)
            x,y=np.meshgrid(xs,ys)
            b=np.stack([x.ravel(),y.ravel(),x.ravel()+bw,y.ravel()+bh],axis=1).astype(np.float32)
            yield side,ratio,b

def edge_scores(image,boxes):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(3,3),0)
    edges=(cv2.Canny(blurred,50,140)>0).astype(np.uint8)
    # Integral image: O(1) score per anchor; interior edge density suppresses blank background.
    integ=cv2.integral(edges,sdepth=cv2.CV_32S)
    b=boxes.astype(np.int32)
    n=(integ[b[:,3],b[:,2]]-integ[b[:,1],b[:,2]]
       -integ[b[:,3],b[:,0]]+integ[b[:,1],b[:,0]]).astype(np.float32)
    a=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return n/np.maximum(a,1)

def generate(image,mode="edge",per_shape=MAX_PER_SHAPE):
    h,w=image.shape[:2];result=[];unfiltered_total=0
    for side,ratio,b in grid_boxes(h,w):
        unfiltered_total+=len(b)
        if mode=="dense":
            chosen=b
        elif mode=="edge":
            scores=edge_scores(image,b)
            # Preserve top edge-density windows and systematic spatial exploration.
            k=min(per_shape,len(b))
            selected=np.argpartition(scores,-k)[-k:] if k<len(b) else np.arange(len(b))
            # Stable ordering ensures repeatable output on equal scores.
            selected=selected[np.lexsort((selected,-scores[selected]))]
            chosen=b[selected]
        else:raise ValueError(mode)
        result.append(chosen)
    result=np.concatenate(result).reshape(-1,4)
    return result,unfiltered_total

def best_coverage(boxes,gt):
    if not len(gt):return np.empty(0)
    # Chunk to avoid large N x GT memory spikes.
    best=np.zeros(len(gt),dtype=float)
    for start in range(0,len(boxes),4000):
        ov=iou(boxes[start:start+4000],gt)
        if len(ov):best=np.maximum(best,ov.max(axis=0))
    return best

def run(root,out,limit):
    out.mkdir(parents=True,exist_ok=True)
    val=data(root,"val",limit)
    variants=["canny_cap200","anchor_dense","anchor_edge450","anchor_edge900"]
    stats=[];illustrations=0
    for path,gt in val:
        image=cv2.imread(str(path))
        for name in variants:
            start=time.perf_counter()
            if name=="canny_cap200":b,_=proposals(image,.1,200);raw=len(b)
            elif name=="anchor_dense":b,raw=generate(image,"dense")
            elif name=="anchor_edge450":b,raw=generate(image,"edge",450)
            else:b,raw=generate(image,"edge",900)
            elapsed=time.perf_counter()-start
            best=best_coverage(b,gt)
            small=np.prod(gt[:,2:]-gt[:,:2],axis=1)<1024 if len(gt) else np.zeros(0,dtype=bool)
            stats.append(dict(image=path.name,variant=name,gt=len(gt),small_gt=int(small.sum()),
                hits50=int((best>=.5).sum()),hits75=int((best>=.75).sum()),
                small_hits50=int(((best>=.5)&small).sum()),candidates=len(b),raw_grid=raw,seconds=elapsed))
            if name=="anchor_edge450" and illustrations<3:
                # Visualize only first 150 candidate boxes to avoid an unreadable wall of boxes.
                canvas=image.copy()
                for x1,y1,x2,y2 in gt.astype(int):
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),1)
                for x1,y1,x2,y2 in b[:150].astype(int):
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.imwrite(str(out/f"anchor_overlay_{illustrations}.png"),canvas)
        illustrations+=1
    with (out/"anchor_by_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=stats[0].keys());writer.writeheader();writer.writerows(stats)
    summary=[]
    for name in variants:
        rows=[s for s in stats if s["variant"]==name]
        total=sum(r["gt"] for r in rows);sm=sum(r["small_gt"] for r in rows)
        summary.append(dict(variant=name,images=len(rows),birds=total,
          proposal_R50=sum(r["hits50"] for r in rows)/total if total else None,
          proposal_R75=sum(r["hits75"] for r in rows)/total if total else None,
          small_R50=sum(r["small_hits50"] for r in rows)/sm if sm else None,
          mean_candidates=float(np.mean([r["candidates"] for r in rows])),
          mean_generation_s=float(np.mean([r["seconds"] for r in rows]))))
    with (out/"anchor_summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    (out/"anchor_summary.json").write_text(json.dumps(summary,indent=2))
    print("ANCHOR STUDY RESULTS",json.dumps(summary,indent=2),flush=True)

def test():
    anchors=np.concatenate([b for _,_,b in grid_boxes(64,64,sizes=(8,16),ratios=(1.0,))])
    assert len(anchors) and (anchors[:,2:]>anchors[:,:2]).all()
    im=np.zeros((64,64,3),np.uint8)
    b,total=generate(im,"edge",10)
    assert len(b)>0 and len(b)<=27*10 and total>=len(b)
    assert np.isclose(best_coverage(np.array([[0,0,10,10]]),np.array([[0,0,10,10]]))[0],1)
    print("Anchor geometry tests PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("artifacts"));p.add_argument("--val-limit",type=int,default=20);p.add_argument("--self-test",action="store_true")
    a=p.parse_args();test()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        run(a.dataset,a.output,a.val_limit)
