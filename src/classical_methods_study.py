"""Strictly classical LoG + local Otsu proposals and rotation-pooled HOG/SVM ablation.

Research-only preliminary quick experiment. Same fixed-seed TRAIN and VAL samples,
GT-blind proposal generation, and identical candidate sets for classifiers.
No CNN/transformer/pretrained image models. Logs candidate *and* final FP metrics.
Rotation pooling here is NOT Liu/Hodgson Fourier HOG; describe accurately.
"""
import argparse,csv,json,time
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_laplace,maximum_filter
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cloud_audit import data,iou,nms,ap50

def _clip(x1,y1,x2,y2,w,h):
    a=(max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2)))
    return a if a[2]-a[0]>=3 and a[3]-a[1]>=3 else None

def blob_boxes(image,mode="log",max_peaks=120):
    """Pixel-only multiscale scale-normalized LoG maxima, bright/dark polarity."""
    h,w=image.shape[:2]
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    smooth=cv2.GaussianBlur(grey,(3,3),0).astype(np.float32)/255.
    response_peaks=[]
    for sigma in (2.,3.5,5.5,8.,12.,18.):
        response=-(sigma*sigma)*gaussian_laplace(smooth,sigma,mode="reflect")
        for pol in (1.,-1.):
            s=response*pol
            window=max(3,2*int(sigma*.65)+1)
            mask=(s==maximum_filter(s,size=window)) & (s>max(.015,float(np.percentile(s,99.7))))
            yy,xx=np.nonzero(mask)
            if not len(xx):continue
            rank=np.argsort(-s[yy,xx],kind="stable")[:90]
            response_peaks.extend((float(s[yy[k],xx[k]]),int(xx[k]),int(yy[k]),sigma,int(pol)) for k in rank)
    response_peaks.sort(key=lambda t:-t[0])
    chosen=[]
    for p in response_peaks:
        _,x,y,s,_=p
        if any((x-q[1])**2+(y-q[2])**2<(min(s,q[3])*.8)**2 and abs(s-q[3])<max(s,q[3])*.55 for q in chosen):continue
        chosen.append(p)
        if len(chosen)>=max_peaks:break
    boxes=[];score=[];seen=set()
    for response,x,y,sigma,pol in chosen:
        for factor in (1.5,2.4):
            for ratio in (.7,1.0,1.5):
                rx=sigma*factor*np.sqrt(ratio);ry=sigma*factor/np.sqrt(ratio)
                b=_clip(np.floor(x-rx),np.floor(y-ry),np.ceil(x+rx),np.ceil(y+ry),w,h)
                if b is not None and b not in seen:seen.add(b);boxes.append(b);score.append(response)
        if mode!="otsu":continue
        r=max(7,int(round(3.5*sigma)))
        x1=max(0,x-r);y1=max(0,y-r);x2=min(w,x+r+1);y2=min(h,y+r+1)
        patch=grey[y1:y2,x1:x2]
        if min(patch.shape)<5:continue
        _,mask=cv2.threshold(patch,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        if pol<0:mask=255-mask
        n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
        if n<2:continue
        lx=x-x1;ly=y-y1;label=int(labels[ly,lx])
        if label==0:
            candidates=[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=5]
            if not candidates:continue
            label=min(candidates,key=lambda k:(centroids[k,0]-lx)**2+(centroids[k,1]-ly)**2)
        xx,yy,ww,hh,area=stats[label]
        if area<5:continue
        for pad in (.15,.4):
            b=_clip(np.floor(x1+xx-pad*ww),np.floor(y1+yy-pad*hh),
                    np.ceil(x1+xx+(1+pad)*ww),np.ceil(y1+yy+(1+pad)*hh),w,h)
            if b is not None and b not in seen:seen.add(b);boxes.append(b);score.append(response)
    return np.asarray(boxes,dtype=np.float32).reshape(-1,4),np.asarray(score,np.float32)

def _max_iou(boxes,gt):
    overlaps=iou(boxes,gt)
    by_box=overlaps.max(axis=1) if len(gt) else np.zeros(len(boxes))
    by_gt=overlaps.max(axis=0) if len(boxes) else np.zeros(len(gt))
    return overlaps,by_box,by_gt

def proposal_audit(boxes,gt):
    overlaps,by_box,by_gt=_max_iou(boxes,gt)
    # Dedup excess qualifying candidates per GT; a GT can have many proposal matches.
    qualified=int((by_box>=.5).sum())
    discovered=int((by_gt>=.5).sum())
    return dict(GT=len(gt),candidates=len(boxes),covered50=discovered,covered75=int((by_gt>=.75).sum()),
       background_candidates=int((by_box<.1).sum()),
       mislocalized_candidates=int(((by_box>=.1)&(by_box<.5)).sum()),
       overlapping_candidates=qualified,
       excess_overlapping_candidates=max(0,qualified-discovered),
       empty_scene_candidates=(len(boxes) if len(gt)==0 else 0))

def classification_errors(boxes,scores,gt):
    """One-to-one greedy matching on boxes already after NMS, sorted by score."""
    order=np.argsort(-scores,kind="stable")
    overlaps=iou(boxes,gt)
    used=np.zeros(len(gt),bool)
    tp=bg=local=dup=0
    for ix in order:
        ov=overlaps[ix] if len(gt) else np.empty(0)
        if len(ov) and np.any(ov>=.5):
            available=np.where((ov>=.5)&~used)[0]
            if len(available):
                j=available[np.argmax(ov[available])]
                used[j]=True;tp+=1
            else:dup+=1
        elif len(ov) and ov.max()>=.1:local+=1
        else:bg+=1
    return dict(TP=tp,FP_background=bg,FP_localization=local,FP_duplicates=dup,
       FP_total=bg+local+dup,FN=len(gt)-tp)

def descriptor(image,box,rotation=False):
    x1,y1,x2,y2=np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(int)
    patch=image[y1:y2,x1:x2]
    gray=cv2.cvtColor(cv2.resize(patch,(64,64)),cv2.COLOR_BGR2GRAY)
    def one(g):return hog(g,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm="L2-Hys").astype(np.float32)
    h=one(gray)
    if not rotation:return h
    # Rotation-pooled descriptor, NOT Fourier HOG, is an explicitly labelled
    # classical handcrafted feature hypothesis.
    variants=np.stack([one(cv2.rotate(gray,cv2.ROTATE_90_CLOCKWISE)),
                       one(cv2.rotate(gray,cv2.ROTATE_180)),
                       one(cv2.rotate(gray,cv2.ROTATE_90_COUNTERCLOCKWISE))])
    return np.r_[h,variants.mean(axis=0)].astype(np.float32)

def learn_candidates(train,mode,max_peaks):
    positives=[];negatives=[];rng=np.random.default_rng(42)
    for path,gt in train:
        image=cv2.imread(str(path));boxes,_=blob_boxes(image,mode,max_peaks)
        _,by_box,_=_max_iou(boxes,gt)
        pos=np.flatnonzero(by_box>=.5)
        neg=np.flatnonzero(by_box<.1)
        if len(pos):positives.extend((path,boxes[i]) for i in rng.choice(pos,min(len(pos),35),False))
        if len(neg):negatives.extend((path,boxes[i]) for i in rng.choice(neg,min(len(neg),45),False))
    return positives,negatives

def compare(dataset,out,train_limit=35,val_limit=12,max_peaks=120):
    out.mkdir(parents=True,exist_ok=True)
    train=data(dataset,"train",train_limit)
    val=data(dataset,"val",val_limit)
    proposal_rows=[];scores=[]
    for mode in ("log","otsu"):
        start=time.perf_counter()
        candidates={}
        for path,gt in val:
            image=cv2.imread(str(path));t=time.perf_counter()
            box,_=blob_boxes(image,mode,max_peaks)
            cost=time.perf_counter()-t
            candidates[path.name]=(box,gt,image,cost)
            proposal_rows.append(dict(method=mode,image=path.name,generate_seconds=cost,**proposal_audit(box,gt)))
        print("PROPOSAL MODE COMPLETE",mode,len(candidates),flush=True)
        pos,neg=learn_candidates(train,mode,max_peaks)
        train_sample_count={"positives":len(pos),"negatives":len(neg)}
        if not len(pos) or not len(neg):
            scores.append(dict(method=mode,status="BLOCKED_NO_CLASSES",**train_sample_count))
            continue
        # Read and cache images for repeated handcrafted descriptor extraction.
        imcache={p.name:cv2.imread(str(p)) for p,_ in train}
        for rotation in (False,True):
            name=mode+("_rot_hog" if rotation else "_hog")
            tt=time.perf_counter()
            X=np.stack([descriptor(imcache[p.name],b,rotation) for p,b in pos+neg])
            y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
            model=make_pipeline(StandardScaler(),LinearSVC(C=.01,class_weight="balanced",max_iter=10000,random_state=42))
            model.fit(X,y)
            fit_sec=time.perf_counter()-tt
            predictions=[];det_rows=[];feature_t=0.
            for path,gt in val:
                b,_,image,gtime=candidates[path.name]
                t=time.perf_counter()
                Xval=np.stack([descriptor(image,bb,rotation) for bb in b]) if len(b) else np.empty((0,X.shape[1]))
                margins=model.decision_function(Xval) if len(b) else np.empty(0)
                keep=nms(b,margins,iou_limit=.35,max_det=250) if len(b) else np.empty(0,int)
                post_b=b[keep];post_s=margins[keep]
                feature_t+=time.perf_counter()-t
                # Fixed margin 0 operating point, no VAL threshold tuning
                active=post_s>=0.
                errs=classification_errors(post_b[active],post_s[active],gt)
                det_rows.append(dict(model=name,image=path.name,GT=len(gt),
                   candidates=len(b),after_nms=len(keep),final_detections=int(active.sum()),**errs))
                predictions.append((post_b,post_s,gt))
            ap=ap50(predictions)
            result=dict(method=name,status="COMPLETE",train_images=len(train),val_images=len(val),
               val_birds=sum(len(gt) for _,gt in val),validation_AP50_continuous=float(ap),
               train_pos=len(pos),train_neg=len(neg),fit_seconds=fit_sec,
               val_features_seconds=feature_t,avg_final_boxes=float(np.mean([r["final_detections"] for r in det_rows])),
               FP_background=sum(r["FP_background"] for r in det_rows),
               FP_localization=sum(r["FP_localization"] for r in det_rows),
               FP_duplicates=sum(r["FP_duplicates"] for r in det_rows),
               FN=sum(r["FN"] for r in det_rows))
            scores.append(result)
            with (out/(name+"_per_image.csv")).open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=det_rows[0]);writer.writeheader();writer.writerows(det_rows)
            print("CLASSIFIER RESULT",json.dumps(result),flush=True)
        print("METHOD TIME",mode,time.perf_counter()-start,flush=True)
    with (out/"classical_candidates.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=proposal_rows[0]);writer.writeheader();writer.writerows(proposal_rows)
    summary=[]
    for mode in ("log","otsu"):
        subset=[r for r in proposal_rows if r["method"]==mode];gt=sum(r["GT"] for r in subset)
        summary.append(dict(method=mode,n_images=len(subset),GT=gt,
           recall50=sum(r["covered50"] for r in subset)/gt if gt else None,
           recall75=sum(r["covered75"] for r in subset)/gt if gt else None,
           mean_candidates=float(np.mean([r["candidates"] for r in subset])),
           background_candidates=sum(r["background_candidates"] for r in subset),
           near_miss_candidates=sum(r["mislocalized_candidates"] for r in subset),
           excess_matching_candidates=sum(r["excess_overlapping_candidates"] for r in subset),
           empty_scene_candidates=sum(r["empty_scene_candidates"] for r in subset),
           mean_proposal_seconds=float(np.mean([r["generate_seconds"] for r in subset]))))
    payload={"proposal":summary,"classification":scores,
      "notes":["GT is never input to candidate generation.","HOG rotation pooling is an approximation, NOT Fourier HOG.",
      "AP uses score-ranked post-NMS detections at all scores; fixed margin=0 for FP diagnostic.",
      "Metrics are preliminary seeded VAL subset, not test-set scores."]}
    (out/"classical_methods_results.json").write_text(json.dumps(payload,indent=2))
    print("CLASSICAL STUDY SUMMARY",json.dumps(payload),flush=True)

def tests():
    im=np.zeros((80,100,3),np.uint8)
    cv2.circle(im,(40,40),6,(255,255,255),-1)
    for mode in ("log","otsu"):
        b,_=blob_boxes(im,mode,20)
        assert b.ndim==2 and b.shape[1]==4 and np.all(b[:,2:]>b[:,:2])
    gt=np.array([[34,34,47,47]],np.float32)
    d=classification_errors(np.array([[34,34,47,47],[34,34,47,47],[0,0,10,10]],np.float32),
                            np.array([3.,2.,1.]),gt)
    assert (d["TP"],d["FP_duplicates"],d["FP_background"],d["FN"])==(1,1,1,0)
    h=descriptor(im,np.array([20,20,60,60]),False)
    r=descriptor(im,np.array([20,20,60,60]),True)
    assert len(h)==1764 and len(r)==3528 and np.isfinite(r).all()
    print("LoG/Otsu geometry, FP taxonomy, and rotation-descriptor tests PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("classical_results"))
    p.add_argument("--train-limit",type=int,default=35);p.add_argument("--val-limit",type=int,default=12)
    p.add_argument("--max-peaks",type=int,default=120);p.add_argument("--self-test",action="store_true")
    args=p.parse_args();tests()
    if not args.self_test:
        if not args.dataset:p.error("--dataset required")
        compare(args.dataset,args.output,args.train_limit,args.val_limit,args.max_peaks)
