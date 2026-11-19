"""E41: test E40 mathematically verified HOG/CS-LBP with deterministic SS regions.
Paired with E40 on the same fixed 45 TRAIN / 30 VAL splits, not a TEST run.
Includes GT-only diagnostic oracle for true 4px / 1.12x scan geometry; never
uses ground truth to create or rank detector proposals.
"""
import json, math, time
from pathlib import Path
import numpy as np
import cv2
import e40_paper_cslbp as e40

def ss_boxes(image, cap=900,scales=None,stride_fraction=None):
    """E36/E37 order-invariant Selective Search: unique full SET before cap."""
    seg=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    seg.setBaseImage(image)
    seg.switchToSelectiveSearchFast()
    rect=np.asarray(seg.process(),np.float32).reshape(-1,4)
    if not len(rect):return np.empty((0,4),np.float32),{"ss":0}
    h,w=image.shape[:2]
    b=np.column_stack((rect[:,:2],rect[:,:2]+rect[:,2:])).astype(np.float32)
    wh=b[:,2:]-b[:,:2];a=wh[:,0]*wh[:,1]
    valid=(wh[:,0]>=3)&(wh[:,1]>=3)&(a>=16)&(a<=.92*w*h)
    b=np.unique(b[valid],axis=0)
    count=len(b)
    if count>cap:
        area=np.prod(b[:,2:]-b[:,:2],axis=1)
        tier=np.digitize(area,[1024,10000])
        rng=np.random.default_rng(61)
        choices=[]
        for t in range(3):
            idx=np.flatnonzero(tier==t)
            choices.extend(rng.choice(idx,min(cap//3,len(idx)),replace=False).tolist())
        missing=cap-len(choices)
        if missing>0:
            rem=np.setdiff1d(np.arange(count),np.asarray(choices,dtype=int))
            choices.extend(rng.choice(rem,min(missing,len(rem)),replace=False).tolist())
        b=b[np.sort(choices)]
    return b.astype(np.float32),{"ss":count}

def dense_geometry_audit(val):
    """Measure unbounded proposal *geometry*, not detection; no descriptors.
    Paper stride4 and scale factor1.12 over adaptive scales 0.35..4.
    Only computed on VAL after experiment configuration frozen. No tuning.
    """
    values={"all":[],"small":[],"medium":[],"large":[]}
    mean_count=[]
    for ix,(path,gt) in enumerate(val):
        im=cv2.imread(str(path));ih,iw=im.shape[:2]
        best=np.zeros(len(gt),np.float32);count=0
        scale=.35
        while scale<=4.001:
            bw=int(round(48*scale));bh=int(round(64*scale))
            if bw>=6 and bh>=8 and bw<=iw and bh<=ih:
                xs=e40.grid_axis(iw,bw,4);ys=e40.grid_axis(ih,bh,4)
                xx,yy=np.meshgrid(xs,ys)
                xy=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
                bb=np.column_stack((xy,xy+np.array([bw,bh],np.float32)))
                count+=len(bb)
                if len(gt):
                    # Chunked oracle to bound RAM, no GT-dependent box selection.
                    for start in range(0,len(bb),4000):
                        overlap=e40.iou(bb[start:start+4000],gt)
                        best=np.maximum(best,overlap.max(axis=0))
            scale*=1.12
        mean_count.append(count)
        for gtbox,overlap in zip(gt,best):
            area=float(np.prod(gtbox[2:]-gtbox[:2]))
            kind="small" if area<1024 else ("medium" if area<10000 else "large")
            values[kind].append(float(overlap));values["all"].append(float(overlap))
        if (ix+1)%5==0 or ix+1==len(val):
            print("E41 DENSE ORACLE",ix+1,len(val),"windows",count,flush=True)
    result={k:{"n":len(v),"R50":float(np.mean(np.asarray(v)>=.5)) if v else None,
               "R75":float(np.mean(np.asarray(v)>=.75)) if v else None}
            for k,v in values.items()}
    result["mean_scanned_boxes"]=float(np.mean(mean_count))
    return result

def self_test():
    e40.validate_math()
    im=np.zeros((128,128,3),np.uint8)
    a,_=ss_boxes(im,120);b,_=ss_boxes(im,120)
    assert np.array_equal(a,b)
    print("E41 PASS deterministic SS unique-then-cap",flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",default="e41_results")
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--cap",type=int,default=900)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test()
    else:
        if not args.dataset:p.error("--dataset required")
        # Preserve E40's already fixed TRAIN/VAL ids and feature machinery.
        e40.sliding_windows=ss_boxes
        e40.main(args)
        path=Path(args.output)/"e40_results.json"
        report=json.loads(path.read_text())
        report["experiment"]="E41"
        report["proposal_generator"]="E36-stable Selective Search; GT-blind stratified cap"
        report["comparison"]="Same data/split, math and classification protocol as E40; different proposals and TRAIN negatives"
        dense=dense_geometry_audit(e40.data(args.dataset,"val",args.val_n))
        report["uncapped_paper_stride4_scale1_12_oracle_not_inference"]=dense
        path.unlink()
        (Path(args.output)/"e41_results.json").write_text(json.dumps(report,indent=2))
        print("E41 COMPLETE",json.dumps({"arms":report["arms"],"proposals":report["proposal"],
                                        "dense_geometry":dense}),flush=True)
