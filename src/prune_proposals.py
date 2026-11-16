"""Image-only proposal pruning evaluation. Ground truth is ONLY read after proposals.

Prune before expensive HOG scoring, then enforce a distinct post-classifier threshold,
NMS and output limit. IoU/recall audit determines whether pruning is acceptable.
"""
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data
from anchor_proposals import generate,edge_scores,best_coverage

def prune(image, boxes, budget, overlap=.65):
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    if len(boxes)<=budget:return boxes
    # Score each box using pixels only; no GT, class labels, or image count.
    scores=edge_scores(image,boxes).astype(np.float32)
    # Use a stable sort for ties, which otherwise overfocus on arbitrary regions.
    order=np.argsort(-scores,kind="stable")
    b=boxes[order]
    s=scores[order]
    xywh=np.column_stack((b[:,:2],b[:,2:]-b[:,:2])).astype(np.int32)
    # Suppress highly redundant proposals, but do not suppress overlapping
    # neighbouring birds as aggressively as final detection NMS.
    keep=cv2.dnn.NMSBoxes(xywh.tolist(),(s+1e-6).tolist(),score_threshold=0.0,nms_threshold=float(overlap),top_k=0)
    keep=np.asarray(keep).reshape(-1).astype(int)
    if len(keep)<budget:
        # NMS may remove too many candidates on highly crowded scenes; retain
        # a ranked reserve (still GT-free) to honour the candidate budget.
        keep_set=set(keep.tolist())
        reserve=[k for k in range(len(b)) if k not in keep_set][:budget-len(keep)]
        keep=np.r_[keep,np.asarray(reserve,int)]
    return b[keep[:budget]]

def audit(dataset,out,limit):
    out.mkdir(parents=True,exist_ok=True)
    records=data(dataset,"val",limit)
    configs=[("edge450_unpruned",None,None),("prune_400",400,.65),
             ("prune_800",800,.65),("prune_1600",1600,.65),
             ("prune_3200",3200,.65),("prune_1600_loose",1600,.85)]
    stats=[]
    for idx,(path,gt) in enumerate(records):
        image=cv2.imread(str(path))
        base,_=generate(image,"edge",450)
        for name,budget,overlap in configs:
            start=time.perf_counter()
            boxes=base if budget is None else prune(image,base,budget,overlap)
            duration=time.perf_counter()-start
            best=best_coverage(boxes,gt)
            small=(np.prod(gt[:,2:]-gt[:,:2],axis=1)<1024) if len(gt) else np.zeros(0,bool)
            stats.append(dict(image=path.name,variant=name,birds=len(gt),small_birds=int(small.sum()),
                hits50=int((best>=.5).sum()),hits75=int((best>=.75).sum()),
                small_hits50=int(((best>=.5)&small).sum()),candidates=len(boxes),prune_seconds=duration))
            if idx<2 and name in ("prune_800","prune_1600"):
                canvas=image.copy()
                # Show rank prefix (only 100) plus all GT: diagnostic, not inference.
                for x1,y1,x2,y2 in boxes[:100].astype(int):
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),1)
                for x1,y1,x2,y2 in gt.astype(int):
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.imwrite(str(out/f"{name}_{idx}.png"),canvas)
    with (out/"pruning_by_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=stats[0]);writer.writeheader();writer.writerows(stats)
    summary=[]
    for name,_,_ in configs:
        rows=[r for r in stats if r["variant"]==name]
        n=sum(r["birds"] for r in rows);small=sum(r["small_birds"] for r in rows)
        summary.append(dict(variant=name,images=len(rows),birds=n,
             recall50=sum(r["hits50"] for r in rows)/n if n else None,
             recall75=sum(r["hits75"] for r in rows)/n if n else None,
             small_recall50=sum(r["small_hits50"] for r in rows)/small if small else None,
             mean_candidates=float(np.mean([r["candidates"] for r in rows])),
             mean_prune_seconds=float(np.mean([r["prune_seconds"] for r in rows]))))
    (out/"pruning_results.json").write_text(json.dumps(summary,indent=2))
    with (out/"pruning_summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    print("PRUNING RESULTS",json.dumps(summary,indent=2),flush=True)
    # No automatic gate pass based on val exploration: preserve all results for selection.
    return summary

def test():
    im=np.zeros((64,64,3),np.uint8)
    b=np.array([[0,0,10,10],[0,0,10,10],[20,20,30,30],[40,40,50,50]],np.float32)
    out=prune(im,b,2)
    assert out.shape==(2,4) and np.isfinite(out).all()
    assert np.all(out[:,2:]>out[:,:2])
    assert prune(im,np.empty((0,4),np.float32),5).shape==(0,4)
    print("Pruning geometry/budget checks PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("artifacts"))
    p.add_argument("--val-limit",type=int,default=20);p.add_argument("--self-test",action="store_true")
    a=p.parse_args();test()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        audit(a.dataset,a.output,a.val_limit)
