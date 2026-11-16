"""E22 gate 1: visually audit true negative TRAIN images before retraining.

A zero-object annotation is only a *candidate* negative image: it is not proof
the photo has no birds. Save a labelled contact sheet for manual examination.
Analyse the exact E21 image split and report negative-image representation.
No classifier fitting, no validation or test GT used.
"""
import argparse,json,csv,random
from pathlib import Path
import cv2,numpy as np
from cloud_audit import data
from label_quality import unsafe_training_negative

def tile(path,annotation_count,target=(360,230)):
    img=cv2.imread(str(path))
    if img is None:raise ValueError(str(path))
    h,w=img.shape[:2]
    scale=min(target[0]/w,target[1]/h)
    re=cv2.resize(img,(max(1,round(w*scale)),max(1,round(h*scale))))
    panel=np.full((target[1]+36,target[0],3),232,np.uint8)
    x=(target[0]-re.shape[1])//2;y=(target[1]-re.shape[0])//2
    panel[y:y+re.shape[0],x:x+re.shape[1]]=re
    cv2.rectangle(panel,(0,target[1]),(target[0],target[1]+36),(30,30,30),-1)
    cv2.putText(panel,f"{path.name}  GT={annotation_count}",(7,target[1]+23),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return panel

def run(root,out,trainlimit):
    out.mkdir(parents=True,exist_ok=True)
    selected=data(root,"train",trainlimit)
    flagged=[p.name for p,gt in selected if unsafe_training_negative(p.name,len(gt))]
    used=[x for x in selected if not unsafe_training_negative(x[0].name,len(x[1]))]
    ordered=np.random.default_rng(37).permutation(len(used))
    ncal=max(15,int(.2*len(used)))
    splits={"FIT":[used[i] for i in ordered[ncal:]],
            "CAL":[used[i] for i in ordered[:ncal]]}
    rows=[]
    for split,items in splits.items():
        for p,gt in items:
            rows.append(dict(image=p.name,split=split,GT=len(gt),zero_GT=len(gt)==0))
    stat={}
    for split,items in splits.items():
        blank=[(p,gt) for p,gt in items if not len(gt)]
        stat[split]=dict(images=len(items),zero_GT_images=len(blank),
               percent_zero_gt=len(blank)/max(1,len(items)),
               named_zero_GT_images=[p.name for p,_ in blank],
               positive_images=len(items)-len(blank))
        # All sampled zero-GT TRAIN images, no cherry-picked examples.
        if blank:
            panels=[tile(p,len(gt)) for p,gt in blank]
            # Add spare blank tiles only to build a rectangular grid.
            columns=3
            while len(panels)%columns:
                panels.append(np.full_like(panels[0],245))
            canvas=np.vstack([np.hstack(panels[i:i+columns]) for i in range(0,len(panels),columns)])
            cv2.imwrite(str(out/(split.lower()+"_zero_annotation_contact.jpg")),canvas,
                        [cv2.IMWRITE_JPEG_QUALITY,88])
            for p,gt in blank:
                cv2.imwrite(str(out/("full_"+p.name)),cv2.imread(str(p)),
                            [cv2.IMWRITE_JPEG_QUALITY,78])
    report=dict(sampled_train_images=trainlimit,actual_eligible=len(used),
        exclusion_reason="Known suspicious zero-GT TRAIN annotations only",
        excluded_known_suspect_images=flagged,
        FIT=stat["FIT"],CAL=stat["CAL"],
        warnings=["Every zero-GT image is unverified until visually inspected.",
                  "Do not use visually suspicious zero-GT images as negative-only training.",
                  "Current sampler has no negative-only image quota and uses 22 candidate boxes per image and scale.",
                  "No validation or test image accessed in this audit."])
    (out/"negative_image_provenance.json").write_text(json.dumps(report,indent=2))
    with (out/"train_image_provenance.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    print("NEGATIVE IMAGE PROVENANCE",json.dumps(report),flush=True)

def self_test():
    panel=tile.__name__
    assert panel=="tile"
    assert unsafe_training_negative("beach_bird_0149.jpg",0)
    assert not unsafe_training_negative("beach_bird_0149.jpg",2)
    print("PASS explicit negative image QA checks",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path)
    ap.add_argument("--output",type=Path,default=Path("negative_audit"))
    ap.add_argument("--train-limit",type=int,default=100)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args();self_test()
    if not args.self_test:
        if not args.dataset:ap.error("--dataset required")
        run(args.dataset,args.output,args.train_limit)
