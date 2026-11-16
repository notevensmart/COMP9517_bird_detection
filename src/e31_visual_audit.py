"""E31 visual audit: compare real predicted bounding boxes across classical models.
Models, inputs, NMS, VAL selection and evaluation settings are unchanged.
GT is used for diagnostic colouring only, never to generate predictions.
"""
from pathlib import Path
import csv, json
import numpy as np
import cv2
from cloud_audit import iou, nms

METHODS=("linear_svm","rbf_svm","random_forest","lightgbm")
FOCUS=("beach_bird_1379.jpg","beach_bird_1399.jpg","beach_bird_1419.jpg",
       "beach_bird_1435.jpg","beach_bird_1560.jpg","beach_bird_1606.jpg")
COLORS={"TP":(255,0,0),"FP":(0,0,255),"GT":(0,210,0),"MISSED":(0,230,255)}

def rank_matches(b,s,gt,limit=20):
    b=np.asarray(b,dtype=float).reshape(-1,4)
    s=np.asarray(s,dtype=float).reshape(-1)
    gt=np.asarray(gt,dtype=float).reshape(-1,4)
    if len(b)!=len(s):raise ValueError("Box and score lengths disagree")
    ids=np.argsort(-s,kind="stable")[:limit]
    overlap=iou(b[ids],gt)
    chosen=set();marks=[]
    for i,j in enumerate(ids):
        candidates=[(float(overlap[i,k]),k) for k in range(len(gt))
                    if k not in chosen and overlap[i,k]>=.5]
        if candidates:
            _,k=max(candidates)
            chosen.add(k)
            marks.append((int(j),"TP",int(k)))
        else:
            marks.append((int(j),"FP",None))
    return marks,chosen

def one_panel(image,gt,b,s,method):
    marks,matched=rank_matches(b,s,gt,20)
    canvas=image.copy();height,width=canvas.shape[:2]
    # Draw original official GT; distinguish unmatched GT after predictions.
    for ground in gt:
        a=np.rint(ground).astype(int)
        cv2.rectangle(canvas,(max(0,a[0]),max(0,a[1])),(min(width-1,a[2]),min(height-1,a[3])),COLORS["GT"],1)
    for idx,kind,_ in marks:
        a=np.rint(b[idx]).astype(int)
        cv2.rectangle(canvas,(max(0,a[0]),max(0,a[1])),(min(width-1,a[2]),min(height-1,a[3])),COLORS[kind],2)
    for j,ground in enumerate(gt):
        if j not in matched:
            a=np.rint(ground).astype(int)
            cv2.rectangle(canvas,(max(0,a[0]),max(0,a[1])),(min(width-1,a[2]),min(height-1,a[3])),COLORS["MISSED"],2)
    band=np.full((70,width,3),245,np.uint8)
    text=f"{method} | GT={len(gt)} TP20={len(matched)} FP20={len(marks)-len(matched)}"
    cv2.putText(band,text,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.56,(15,15,15),1,cv2.LINE_AA)
    cv2.putText(band,"BLUE: TP | RED: FP | GREEN: GT | YELLOW: missed GT",(8,43),
                cv2.FONT_HERSHEY_SIMPLEX,.36,(50,50,50),1,cv2.LINE_AA)
    if not len(gt):
        cv2.putText(band,"CAUTION: zero official labels may contain real birds",(8,62),
                    cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,145),1,cv2.LINE_AA)
    return np.vstack((band,canvas)),dict(top20_TP=len(matched),
        top20_FP=len(marks)-len(matched),missed_GT_top20=len(gt)-len(matched))

def audit(models,valbank,out):
    output=Path(out)/"visual_review"
    output.mkdir(parents=True,exist_ok=True)
    names={x["path"].name:x for x in valbank}
    assert len(models)==4 and set(models)==set(METHODS)
    perimage=[];render_cache={}
    # Re-score exactly the original VAL feature/proposal bank for visual audit.
    for rec in valbank:
        name=rec["path"].name
        b=rec["boxes"]
        for method in METHODS:
            model=models[method]
            if len(b):
                s=model.decision_function(rec["X"]) if hasattr(model,"decision_function") else model.predict_proba(rec["X"])[:,1]
                s=np.asarray(s,float).ravel()
                if not np.isfinite(s).all():raise ValueError(f"nonfinite {method} scores")
                keep=nms(b,s,iou_limit=.35,max_det=300)
            else:
                keep=np.empty(0,int);s=np.empty(0)
            sb=b[keep];ss=s[keep]
            matched,chosen=rank_matches(sb,ss,rec["gt"])
            row=dict(image=name,method=method,GT=len(rec["gt"]),proposals=len(b),
                after_nms=len(keep),top20_TP=len(chosen),top20_FP=len(matched)-len(chosen),
                top20_missed=len(rec["gt"])-len(chosen),official_zero_gt=(len(rec["gt"])==0))
            perimage.append(row)
            if name not in render_cache:render_cache[name]={}
            render_cache[name][method]=(sb,ss)
    with (output/"per_image.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(perimage[0]))
        writer.writeheader();writer.writerows(perimage)
    # Always review previously known hard cases if they appear in this fixed subset.
    selected=[(name,"documented prior failure scene") for name in FOCUS if name in names]
    ids={name for name,_ in selected}
    priority=[
        (max(valbank,key=lambda x:len(x["gt"]))["path"].name,"highest crowding"),
        (max(valbank,key=lambda x:sum(np.prod(x["gt"][:,2:]-x["gt"][:,:2],axis=1)<1024) if len(x["gt"]) else 0)["path"].name,"most small GT birds"),
        (max(valbank,key=lambda x:np.max(np.prod(x["gt"][:,2:]-x["gt"][:,:2],axis=1)) if len(x["gt"]) else 0)["path"].name,"largest GT bird"),
    ]
    for name,reason in priority:
        if name not in ids:selected.append((name,reason));ids.add(name)
    results=[]
    for name,reason in selected:
        rec=names[name]
        image=cv2.imread(str(rec["path"]))
        if image is None:raise RuntimeError(f"Cannot read source image: {name}")
        panels=[];summary={}
        for method in METHODS:
            b,s=render_cache[name][method]
            panel,metrics=one_panel(image,rec["gt"],b,s,method)
            panels.append(panel);summary[method]=metrics
        comparison=np.hstack(panels)
        path=output/(Path(name).stem+"_4_models.jpg")
        if not cv2.imwrite(str(path),comparison,[cv2.IMWRITE_JPEG_QUALITY,88]):
            raise IOError(f"Could not save {path}")
        results.append(dict(image=name,reason=reason,GT=len(rec["gt"]),file=path.name,
                            models=summary))
    report=dict(status="VISUALS_GENERATED_NOT_YET_HUMAN_REVIEWED",selected=results,
        all_val_images=len(valbank),per_image_table="per_image.csv",
        interpretation="TP/FP are one-to-one IoU>=.5 versus official VAL GT among top20 after NMS; FP on zero-GT scenes may reflect missing annotations",
        no_val_fit=True,no_test_access=True)
    (output/"visual_audit.json").write_text(json.dumps(report,indent=2))
    print("E31 VISUAL AUDIT",json.dumps(dict(images=[x["image"] for x in results],count=len(results))),flush=True)
    return report

def tests():
    gt=np.array([[2,2,20,20]],float)
    boxes=np.array([[2,2,20,20],[2,2,20,20],[32,32,47,47]],float)
    score=np.array([.9,.8,.7])
    matches,seen=rank_matches(boxes,score,gt)
    assert [r[1] for r in matches]==["TP","FP","FP"]
    assert seen=={0}
    im=np.zeros((64,64,3),dtype=np.uint8)
    panel,metrics=one_panel(im,gt,boxes,score,"test")
    assert panel.shape==(134,64,3)
    assert metrics["top20_TP"]==1 and metrics["top20_FP"]==2
    print("E31 visual audit synthetic matching tests PASS",flush=True)

if __name__=="__main__":
    tests()
