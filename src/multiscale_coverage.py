"""Classical multiscale localization coverage study — no classifier training.

Design correction: original LoG sigma <=18 with factor <=2.4 cannot yield
large bird regions (e.g. near full image swans); supplement with large-window
image-edge-ranked grids 96..512 px and diverse aspect ratios. Preserve original
blob LoG+Otsu proposals for small birds; study union before training.

Ground truth is used *only* AFTER generating candidates to evaluate recall,
never to select or rank candidates for inference. Comparison uses fixed VAL.
"""
import argparse,json,csv,time
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data,iou
from classical_methods_study import blob_boxes
from anchor_proposals import grid_boxes
from spatial_pruning import integral_edges,box_density,shape_spatial_select

SIZES=(96,144,192,256,384,512)
RATIOS=(.45,.7,1.,1.5,2.2)

def larger_candidates(image,per_shape=50):
    h,w=image.shape[:2];integ=integral_edges(image);parts=[]
    for _,_,box in grid_boxes(h,w,sizes=SIZES,ratios=RATIOS,frac=.25,minimum_stride=4):
        if len(box)==0:continue
        score=box_density(integ,box)
        parts.append(shape_spatial_select(box,score,per_shape,w,h,tiles=5))
    return np.concatenate(parts).reshape(-1,4) if parts else np.empty((0,4),np.float32)

def methods(image,per_shape=50):
    log,_=blob_boxes(image,"otsu",max_peaks=120)
    large=larger_candidates(image,per_shape)
    union=np.vstack([log,large]) if len(log) and len(large) else (log if len(log) else large)
    return {"LoG_Otsu_original":log,"Large_diverse_grid":large,"Hybrid_union":union}

def eval_boxes(b,gt):
    ov=iou(b,gt)
    best=ov.max(axis=0) if len(b) else np.zeros(len(gt))
    areas=np.prod(gt[:,2:]-gt[:,:2],axis=1) if len(gt) else np.empty(0)
    result=dict(GT=len(gt),boxes=len(b),GT_R50=int((best>=.5).sum()),
          GT_R75=int((best>=.75).sum()))
    for name,mask in (("small",areas<1024),
                      ("medium",(areas>=1024)&(areas<10000)),
                      ("large",areas>=10000)):
        result[name+"_GT"]=int(mask.sum())
        result[name+"_R50"]=int(((best>=.5)&mask).sum())
    return result

def plot_example(image,gt,proposals,destination):
    canvas=image.copy()
    ov=iou(proposals,gt)
    if len(proposals) and len(gt):
        good=np.flatnonzero(ov.max(axis=1)>=.5)
        for j in good[:80]:
            b=np.rint(proposals[j]).astype(int)
            cv2.rectangle(canvas,(b[0],b[1]),(b[2],b[3]),(255,0,0),1)
    for b in gt:
        b=np.rint(b).astype(int)
        cv2.rectangle(canvas,(b[0],b[1]),(b[2],b[3]),(0,255,0),2)
    cv2.imwrite(str(destination),canvas,[cv2.IMWRITE_JPEG_QUALITY,75])

def run(root,out,val_limit,per_shape):
    out.mkdir(parents=True,exist_ok=True)
    val=data(root,"val",val_limit)
    rows=[];vis=[]
    for idx,(path,gt) in enumerate(val):
        im=cv2.imread(str(path));t=time.perf_counter()
        method=methods(im,per_shape)
        seconds=time.perf_counter()-t
        for name,b in method.items():
            rows.append(dict(image=path.name,method=name,seconds=seconds,**eval_boxes(b,gt)))
        if len(gt)>0 and len(vis)<4:
            sizes=np.prod(gt[:,2:]-gt[:,:2],axis=1)
            if sizes.max()>=10000 or (len(gt)>=14):
                plot_example(im,gt,method["Hybrid_union"],out/(path.stem+"_hybrid_gt_green_candidates_blue.jpg"))
                vis.append(path.name)
        if (idx+1)%10==0 or idx+1==len(val):
            print("SCALE AUDIT",idx+1,"/",len(val),flush=True)
    summary=[]
    for key in ("LoG_Otsu_original","Large_diverse_grid","Hybrid_union"):
        r=[x for x in rows if x["method"]==key]
        g=sum(x["GT"] for x in r)
        entry=dict(method=key,validation_images=len(r),validation_birds=g,
             mean_candidate_boxes=float(np.mean([x["boxes"] for x in r])),
             recall50=sum(x["GT_R50"] for x in r)/g,
             recall75=sum(x["GT_R75"] for x in r)/g)
        for area in ("small","medium","large"):
            cnt=sum(x[area+"_GT"] for x in r)
            entry[area+"_birds"]=cnt
            entry[area+"_recall50"]=sum(x[area+"_R50"] for x in r)/cnt if cnt else None
        summary.append(entry)
    (out/"multiscale_recall.json").write_text(json.dumps(summary,indent=2))
    with (out/"multiscale_by_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    print("MULTISCALE COVERAGE RESULTS",json.dumps(summary),flush=True)

def tests():
    im=np.zeros((300,400,3),np.uint8);cv2.rectangle(im,(30,30),(260,250),(255,255,255),3)
    m=methods(im,8);assert set(m)=={"LoG_Otsu_original","Large_diverse_grid","Hybrid_union"}
    for b in m.values():
        assert b.ndim==2 and b.shape[1]==4
        if len(b):
            assert (b[:,2:]>b[:,:2]).all()
    print("PASS multiscale geometry, bounds and hybrid union")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("scale_results"))
    p.add_argument("--val-limit",type=int,default=30)
    p.add_argument("--per-shape",type=int,default=50)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        run(a.dataset,a.output,a.val_limit,a.per_shape)
