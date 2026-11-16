"""Raw-image tiny overfit experiment. Never inserts GT boxes into detector inference."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,proposals,iou,feature,nms,ap50

def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    all_train=data(root,"train",0)
    positives=[v for v in all_train if 1<=len(v[1])<=8]
    negatives=[v for v in all_train if len(v[1])==0]
    rng=np.random.default_rng(42)
    chosen=[positives[int(i)] for i in rng.choice(len(positives),min(2,len(positives)),replace=False)]
    if negatives:chosen.append(negatives[0])
    results=[]
    for name,pad,cap in [("capped",.1,200),("uncapped",.1,None),("uncapped_pad25",.25,None)]:
        bank=[];features=[];labels=[];tot=covered=contaminated=0
        for path,gt in chosen:
            im=cv2.imread(str(path));b,_=proposals(im,pad,cap)
            overlaps=iou(b,gt)
            best=overlaps.max(axis=1) if len(gt) else np.zeros(len(b))
            cover=overlaps.max(axis=0) if len(b) else np.zeros(len(gt))
            tot+=len(gt);covered+=int((cover>=.5).sum())
            # Reject negatives containing >=half of any bird (small birds inside large boxes).
            if len(gt) and len(b):
                wh=np.maximum(0,np.minimum(b[:,None,2:],gt[None,:,2:])-np.maximum(b[:,None,:2],gt[None,:,:2]))
                intersections=wh[:,:,0]*wh[:,:,1]
                box_area=np.prod(b[:,2:]-b[:,:2],axis=1)
                gt_area=np.prod(gt[:,2:]-gt[:,:2],axis=1)
                contain=(intersections/np.maximum(np.minimum(box_area[:,None],gt_area[None,:]),1e-9)).max(axis=1)
            else:contain=np.zeros(len(b))
            bad=(best<.1)&(contain>=.5);contaminated+=int(bad.sum())
            neg=(best<.1)&(~bad);pos=best>=.5
            indices=np.r_[np.flatnonzero(pos),np.flatnonzero(neg)[:500]]
            for j in indices:
                features.append(feature(im,b[j]));labels.append(int(pos[j]))
            bank.append((path,gt,im,b))
        positives_n=sum(labels);negatives_n=len(labels)-positives_n
        item=dict(variant=name,training_image_ids=[p.name for p,_,_,_ in bank],GT_birds=tot,
                  proposal_recall50=covered/tot if tot else None,positive_patches=positives_n,
                  negative_patches=negatives_n,contaminated_negatives_excluded=contaminated)
        if positives_n==0 or negatives_n==0:
            item.update(status="BLOCKED_NO_BOTH_CLASSES",train_patch_accuracy=None,raw_image_train_AP50=None)
            results.append(item);continue
        X=np.asarray(features);y=np.asarray(labels)
        model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=10000,random_state=42))
        model.fit(X,y)
        item["train_patch_accuracy"]=float((model.predict(X)==y).mean())
        predictions=[]
        for path,gt,im,b in bank:
            s=model.decision_function(np.stack([feature(im,bb) for bb in b])) if len(b) else np.empty(0)
            keep=nms(b,s)
            predictions.append((b[keep],s[keep],gt))
        item["raw_image_train_AP50"]=ap50(predictions)
        item["status"]="PASS" if item["raw_image_train_AP50"]>=.9 else "FAIL_TINY_DETECTION_OVERFIT"
        results.append(item)
        print(item,flush=True)
    output={"tiny_set_results":results,"gate_pass":any(r["status"]=="PASS" for r in results),
            "rule":"Full-scale experiments blocked unless tiny raw-image AP50 >= 0.9 on a fixed tiny training set",
            "test_split_used":False}
    (out/"tiny_overfit.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True,type=Path);p.add_argument("--output",default="artifacts",type=Path)
    a=p.parse_args();run(a.dataset,a.output)
