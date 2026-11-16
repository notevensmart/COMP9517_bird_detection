"""Dataset annotation visual QA: empty-labelled images are not necessarily bird-free.

Do not silently delete or relabel any official val/test annotations. Generate
contact sheets of zero-labelled TRAIN/VAL images, positive-labelled controls,
and a CSV of annotation counts. Human review may flag scenes with obvious
birds but no annotations; independent adjudication is necessary for claims.
"""
import argparse,csv,json,math
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data

def tile(path,gt,tw=320,th=240):
    im=cv2.imread(str(path))
    h,w=im.shape[:2]
    factor=min(tw/w,(th-46)/h)
    resized=cv2.resize(im,(max(1,round(w*factor)),max(1,round(h*factor))))
    canvas=np.full((th,tw,3),247,np.uint8)
    x=(tw-resized.shape[1])//2;y=45+(th-45-resized.shape[0])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(canvas,f"{path.name}",(6,17),cv2.FONT_HERSHEY_SIMPLEX,.51,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(canvas,f"GT boxes: {len(gt)} | {w}x{h}",(6,37),cv2.FONT_HERSHEY_SIMPLEX,.49,(60,60,60),1,cv2.LINE_AA)
    return canvas

def sheets(records,folder,prefix,cols=4,rows=3):
    folder.mkdir(parents=True,exist_ok=True)
    size=cols*rows
    for k in range(0,len(records),size):
        ims=[tile(path,gt) for path,gt in records[k:k+size]]
        while len(ims)<size:ims.append(np.full_like(ims[0],255))
        canvas=np.vstack([np.hstack(ims[i:i+cols]) for i in range(0,size,cols)])
        cv2.imwrite(str(folder/f"{prefix}_{(k//size)+1:03d}.jpg"),canvas,
                    [cv2.IMWRITE_JPEG_QUALITY,85])
    return math.ceil(len(records)/size)

def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    report=[];summary={}
    for split in ("train","val"):
        records=data(root,split,0)
        empties=[r for r in records if len(r[1])==0]
        positives=[r for r in records if len(r[1])>0]
        # Generate every empty scene plus a few positive controls.
        pages=sheets(empties,out/"visual_empty_labels",split+"_empty")
        sample=positives[::max(1,len(positives)//24)][:24]
        sheets(sample,out/"visual_positive_controls",split+"_positive")
        for path,gt in records:
            report.append(dict(split=split,image=path.name,GT=len(gt),
                               is_empty_annotation=(len(gt)==0)))
        summary[split]=dict(images=len(records),empty_labeled_images=len(empties),
                            positive_labeled_images=len(positives),pages=pages,
                            fraction_empty=len(empties)/len(records))
        print("EMPTY LABEL REVIEW",split,json.dumps(summary[split]),flush=True)
    with (out/"annotation_count.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=report[0].keys())
        w.writeheader();w.writerows(report)
    (out/"annotation_review_summary.json").write_text(json.dumps(dict(summary=summary,
        warning="A zero-bird annotation does not guarantee a zero-bird image. Human review required.",
        policy="Never change official labels or exclude validation images without explicit diagnostic designation."),indent=2))
    print("ANNOTATION AUDIT DONE",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("label_review"))
    a=p.parse_args();run(a.dataset,a.output)
