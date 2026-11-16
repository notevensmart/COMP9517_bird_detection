"""Paper-inspired objectness baseline: OpenCV ximgproc EdgeBoxes.

Zitnick & Dollar ECCV 2014. Uses OpenCV structured edge detector pretrained
on external data; note this makes it NOT an entirely trained-from-scratch method.
Ground truth is never passed to edge detector or proposal generation.
"""
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data
from anchor_proposals import best_coverage
from spatial_pruning import generate_budgeted

def edgeboxes(image,edge_detector,maximum=2000):
    rgb=np.float32(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))/255.
    edge_map=edge_detector.detectEdges(rgb)
    orient=edge_detector.computeOrientation(edge_map)
    edge_nms=edge_detector.edgesNms(edge_map,orient)
    # Typical EdgeBoxes default minBoxArea=1000 is unsuitable for tiny birds,
    # so explicitly use 16 px^2 to avoid mechanically filtering them all.
    gen=cv2.ximgproc.createEdgeBoxes(maxBoxes=maximum,minBoxArea=16,
              maxAspectRatio=4.0,minScore=0.001)
    rectangles,scores=gen.getBoundingBoxes(edge_nms,orient)
    if rectangles is None or not len(rectangles):
        return np.empty((0,4),np.float32),np.empty(0,np.float32)
    rect=np.asarray(rectangles,np.float32).reshape(-1,4)
    return np.column_stack([rect[:,0],rect[:,1],rect[:,0]+rect[:,2],rect[:,1]+rect[:,3]]),np.asarray(scores).reshape(-1)

def audit(root,out,model,limit):
    out.mkdir(parents=True,exist_ok=True)
    if not hasattr(cv2,"ximgproc") or not hasattr(cv2.ximgproc,"createEdgeBoxes"):
        raise RuntimeError("Install opencv-contrib-python-headless")
    edge_detector=cv2.ximgproc.createStructuredEdgeDetection(str(model))
    val=data(root,"val",limit)
    names=["spatial1350","spatial2700","edgeboxes500","edgeboxes1000","edgeboxes2000"]
    rows=[]
    for path,gt in val:
        image=cv2.imread(str(path))
        first=time.perf_counter()
        eb,s=edgeboxes(image,edge_detector,2000)
        edge_seconds=time.perf_counter()-first
        small=np.prod(gt[:,2:]-gt[:,:2],axis=1)<1024 if len(gt) else np.zeros(0,bool)
        for name in names:
            start=time.perf_counter()
            if name.startswith("spatial"):
                b,_=generate_budgeted(image,per_shape=(50 if name=="spatial1350" else 100))
                sec=time.perf_counter()-start
            else:
                cap=int(name.removeprefix("edgeboxes"))
                b=eb[:cap]
                sec=edge_seconds
            cover=best_coverage(b,gt)
            rows.append(dict(image=path.name,method=name,gt=len(gt),
                small_gt=int(small.sum()),hits50=int((cover>=.5).sum()),
                hits75=int((cover>=.75).sum()),small_hits50=int(((cover>=.5)&small).sum()),
                boxes=len(b),seconds=sec))
        print("EDGEBOX IMAGE",path.name,"GT",len(gt),"proposals",len(eb),flush=True)
    with (out/"edgeboxes_per_image.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary=[]
    for name in names:
        r=[v for v in rows if v["method"]==name];total=sum(x["gt"] for x in r);sm=sum(x["small_gt"] for x in r)
        summary.append(dict(method=name,n_images=len(r),GT=total,
            recall50=sum(x["hits50"] for x in r)/total if total else None,
            recall75=sum(x["hits75"] for x in r)/total if total else None,
            small_R50=sum(x["small_hits50"] for x in r)/sm if sm else None,
            mean_boxes=float(np.mean([x["boxes"] for x in r])),
            seconds_per_image=float(np.mean([x["seconds"] for x in r]))))
    (out/"edgeboxes_vs_spatial.json").write_text(json.dumps(summary,indent=2))
    print("EDGEBOX COMPARISON",json.dumps(summary),flush=True)

def tests():
    assert hasattr(cv2,"ximgproc") and hasattr(cv2.ximgproc,"createEdgeBoxes")
    assert hasattr(cv2.ximgproc,"createStructuredEdgeDetection")
    print("EdgeBoxes extension available",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--output",type=Path,default=Path("edgeboxes_results"))
    p.add_argument("--val-limit",type=int,default=8)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset or not a.model:p.error("--dataset and --model required")
        audit(a.dataset,a.output,a.model,a.val_limit)
