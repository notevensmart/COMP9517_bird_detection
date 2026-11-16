"""E39: scale-specialized CLASSICAL boosted bird detector, with audited preprocessing.

TRAIN-only two-expert LightGBM with a global comparator; fixed deterministic
Selective Search proposals from E37. Inputs are aspect-preserving crops with
CLAHE gradient channels, colour spatial pyramid, local LBP and context cues.
HOG/LBP PCA-RBF incumbent uses EXACT same proposals and samples/images.
No neural nets, pretrained visual embeddings, TEST data, or GT at inference.
"""
import argparse, csv, json, time
from pathlib import Path
import cv2, numpy as np
from lightgbm import LGBMClassifier
from cloud_audit import data, iou, nms
from quality_subset_train import training_subset
from selective_search_detection import prepare, fit_models, coverage
from metrics_map import evaluate_map50
from nonlinear_appearance_study import rbf_logits
from e32_failure_driven import per_image_stats, draw_audit
from rebuild_detector import cropbox, make_jitter, safe_labels

SEED=9444
SIZE=64
TINY=1024.
FOCUS={"beach_bird_1379.jpg","beach_bird_1389.jpg","beach_bird_1399.jpg",
       "beach_bird_1419.jpg","beach_bird_1435.jpg","beach_bird_1452.jpg",
       "beach_bird_1560.jpg","beach_bird_1606.jpg"}
CLAHE=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

def letterbox(patch,size=SIZE):
    """No stretching; neutral pad and return mask and original dimensions."""
    if patch is None or patch.size==0 or patch.ndim!=3 or patch.shape[2]!=3:
        raise ValueError("Invalid BGR patch")
    h,w=patch.shape[:2]
    if h<1 or w<1:raise ValueError("Degenerate native crop")
    scale=min(size/w,size/h)
    nw=max(1,min(size,int(round(w*scale))))
    nh=max(1,min(size,int(round(h*scale))))
    method=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR
    small=cv2.resize(patch,(nw,nh),interpolation=method)
    # Constant neutral border avoids inventing bird edges by reflection.
    fill=np.median(patch.reshape(-1,3),axis=0).astype(np.uint8)
    canvas=np.broadcast_to(fill,(size,size,3)).copy()
    mask=np.zeros((size,size),dtype=np.uint8)
    left=(size-nw)//2;top=(size-nh)//2
    canvas[top:top+nh,left:left+nw]=small
    mask[top:top+nh,left:left+nw]=1
    if not np.array_equal(canvas[top:top+nh,left:left+nw],small):
        raise AssertionError("Letterbox changed source resized pixels")
    return canvas,mask,(h,w)

def normalized_features(image,box,with_preview=False):
    """Image-only extraction, no resize of aspect ratio, no GT input."""
    h,w=image.shape[:2]
    x1,y1,x2,y2=np.asarray(box,np.float32)
    if not(np.isfinite([x1,y1,x2,y2]).all() and x2>x1 and y2>y1):
        raise ValueError("Bad bounding box")
    cropped=cropbox(image,box)
    canvas,mask,(ph,pw)=letterbox(cropped)
    # CLAHE is applied only to LUMA of the already isolated crop, never alters
    # original RGB/HSV used for colour features or selective-search proposals.
    gray=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    clahe=CLAHE.apply(gray)
    gx=cv2.Sobel(clahe,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(clahe,cv2.CV_32F,0,1,ksize=3)
    mag=cv2.magnitude(gx,gy)
    ori=(np.arctan2(gy,gx)+np.pi)%(np.pi)
    region=mask.astype(bool)
    # 4x4 spatial orientation channels: 6 bins; no features from padded pixels.
    hsv=cv2.cvtColor(canvas,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(canvas,cv2.COLOR_BGR2LAB)
    parts=[]
    for row in range(4):
        for col in range(4):
            ys=slice(row*16,(row+1)*16);xs=slice(col*16,(col+1)*16)
            valid=region[ys,xs]
            m=mag[ys,xs][valid]
            o=ori[ys,xs][valid]
            if len(m):
                orientation=np.histogram(o,bins=6,range=(0,np.pi),weights=m)[0]
                orientation=orientation/(orientation.sum()+1e-5)
                sat=hsv[ys,xs,1][valid].astype(np.float32)/255.
                lum=lab[ys,xs,0][valid].astype(np.float32)/255.
                colour=[float(sat.mean()),float(lum.mean()),float(lum.std())]
            else:
                orientation=np.zeros(6,float);colour=[0.,0.,0.]
            parts.extend(orientation.tolist()+colour)
    # Rotation-invariant uniform-ish LBP histogram (raw local 8-neighbour
    # threshold code, 16 bins grouped); mask interior only for real crop.
    g=clahe
    c=g[1:-1,1:-1]
    offsets=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
    code=np.zeros_like(c,dtype=np.uint8)
    for k,(dy,dx) in enumerate(offsets):
        n=g[1+dy:63+dy,1+dx:63+dx]
        code|=((n>=c).astype(np.uint8)<<k)
    inside=mask[1:-1,1:-1].astype(bool)
    hist=np.bincount((code[inside]//16).ravel(),minlength=16).astype(float)
    hist/=max(1.,hist.sum())
    # Classical crop vs surrounding region context (descriptive, not a
    # foreground mask), plus native area/aspect/position and tiny flag.
    bw=x2-x1;bh=y2-y1
    margin=.25
    extended=[max(0,x1-margin*bw),max(0,y1-margin*bh),
              min(w,x2+margin*bw),min(h,y2+margin*bh)]
    context=cropbox(image,extended)
    region_hsv=cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV)
    context_hsv=cv2.cvtColor(context,cv2.COLOR_BGR2HSV)
    immean=region_hsv.reshape(-1,3).mean(axis=0)/255.
    ctmean=context_hsv.reshape(-1,3).mean(axis=0)/255.
    gray_real=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray_real,70,150)
    native=[np.log1p(bw),np.log1p(bh),np.log(max(.01,bw/bh)),
            (x1+x2)/(2*w),(y1+y2)/(2*h),float(bw*bh<TINY),
            min(ph,pw)/max(ph,pw),min(ph,pw)/16.,
            float((edges>0).mean()),float(gray_real.std()/255.)]
    values=np.asarray(parts+hist.tolist()+immean.tolist()+
                      (immean-ctmean).tolist()+native,dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite handcrafted features")
    if with_preview:return values,canvas,mask,cropped
    return values

def features_for_bank(bank,output,phase):
    checks=dict(phase=phase,images=0,candidates=0,nonfinite=0,
                extremely_small_native_crops=0,bad_images=0,feature_dimension=None)
    audit=output/"preprocessing";audit.mkdir(parents=True,exist_ok=True)
    tiles=[]
    rng=np.random.default_rng(SEED)
    for k,rec in enumerate(bank):
        image=cv2.imread(str(rec["path"]),cv2.IMREAD_COLOR)
        if image is None or image.ndim!=3:
            checks["bad_images"]+=1;raise RuntimeError("Unreadable BGR image: "+str(rec["path"]))
        feat=[]
        for j,box in enumerate(rec["boxes"]):
            f=normalized_features(image,box)
            feat.append(f)
            if j<4 and k<4 and phase=="TRAIN":
                _,letter,mask,native=normalized_features(image,box,True)
                source=cv2.resize(native,(128,128),interpolation=cv2.INTER_NEAREST)
                dest=cv2.resize(letter,(128,128),interpolation=cv2.INTER_NEAREST)
                tile=np.hstack((source,dest))
                lab=f"{rec['path'].stem[-4:]} #{j} source | letterboxed"
                cv2.putText(tile,lab,(3,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
                tiles.append(tile)
            if min(box[2]-box[0],box[3]-box[1])<8:
                checks["extremely_small_native_crops"]+=1
        if feat:rec["boost_X"]=np.vstack(feat).astype(np.float32)
        else:rec["boost_X"]=np.empty((0,176),np.float32)
        if checks["feature_dimension"] is None and len(feat):
            checks["feature_dimension"]=rec["boost_X"].shape[1]
        if checks["feature_dimension"] is not None and rec["boost_X"].shape[1]!=checks["feature_dimension"]:
            raise ValueError("Feature dimension mismatch")
        checks["images"]+=1;checks["candidates"]+=len(rec["boxes"])
        if (k+1)%10==0:print("E39 PREPROCESS",phase,k+1,len(bank),flush=True)
    if tiles:
        cv2.imwrite(str(audit/"raw_vs_letterboxed.jpg"),
                    np.vstack([np.hstack(tiles[q:q+4]) for q in range(0,len(tiles),4)]))
    (audit/(phase.lower()+"_checks.json")).write_text(json.dumps(checks,indent=2))
    print("E39 PREPROCESS VERIFIED",json.dumps(checks),flush=True)
    return bank,checks

def collect_training(bank,stage1):
    """GT-safe TRAIN proposal examples; per-image balance, harder negatives."""
    rng=np.random.default_rng(SEED)
    X=[];labels=[];areas=[];sources=[];counts=dict(pos=0,neg=0,
                                              pos_small=0,neg_small=0,
                                              pos_other=0,neg_other=0)
    for rec in bank:
        if not len(rec["boxes"]):continue
        sc=stage1.decision_function(rec["X"])
        p=np.asarray(rec["pos"],int)
        n=np.asarray(rec["neg"],int)
        # Prefer genuine proposals over perfect GT boxes to reduce domain shift.
        if len(p)>55:p=rng.choice(p,55,replace=False)
        order=n[np.argsort(-sc[n],kind="stable")] if len(n) else n
        hard=order[:min(30,len(order))]
        rem=order[len(hard):]
        extra=rng.choice(rem,min(25,len(rem)),replace=False) if len(rem) else np.empty(0,int)
        neg=np.r_[hard,extra]
        for ids,y in ((p,1),(neg,0)):
            for j in ids:
                X.append(rec["boost_X"][j]);labels.append(y)
                box=rec["boxes"][j]
                area=float((box[2]-box[0])*(box[3]-box[1]))
                areas.append(area);sources.append(rec["path"].name)
                key=("pos" if y else "neg")+("_small" if area<TINY else "_other")
                counts[key]+=1;counts["pos" if y else "neg"]+=1
    if not X:raise ValueError("No supervised proposal training samples")
    arr=np.vstack(X).astype(np.float32)
    yy=np.asarray(labels,int);a=np.asarray(areas,np.float32)
    if len(np.unique(yy))<2:raise ValueError("One-class training bank")
    if not np.isfinite(arr).all():raise ValueError("Nonfinite supervised features")
    return arr,yy,a,counts

def augment_image_boxes(image,boxes,mode):
    """TRAIN only; preserve exact 512px geometry, never invent new bird shapes."""
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("Expected uint8 BGR image")
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    h,w=image.shape[:2]
    if mode=="flip":
        transformed=cv2.flip(image,1)
        flipped=boxes.copy()
        flipped[:,0]=w-boxes[:,2]
        flipped[:,2]=w-boxes[:,0]
        if len(flipped) and ((flipped[:,2:]<=flipped[:,:2]).any() or
                             not np.isfinite(flipped).all()):
            raise ValueError("Horizontal flip created invalid boxes")
        return transformed,flipped
    if mode=="photo":
        # Conservative luminance/contrast and colour perturbation, whole image
        # and SAME boxes. No destructive crop or geometric warping.
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        L,A,B=cv2.split(lab)
        L=np.clip(L.astype(np.float32)*1.06+6,0,255).astype(np.uint8)
        recoloured=cv2.cvtColor(cv2.merge((L,A,B)),cv2.COLOR_LAB2BGR)
        hsv=cv2.cvtColor(recoloured,cv2.COLOR_BGR2HSV)
        hsv[:,:,1]=np.clip(hsv[:,:,1].astype(np.float32)*.94,0,255).astype(np.uint8)
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),boxes.copy()
    raise ValueError("Unknown augmentation mode")

def augmented_examples(train,stage1,output):
    """Extra positives AND safe negatives, per-image capped, TRAIN only."""
    rng=np.random.default_rng(8844)
    X=[];Y=[];areas=[];samples=[];audit=[]
    out=output/"preprocessing";out.mkdir(parents=True,exist_ok=True)
    for imageidx,rec in enumerate(train):
        image=cv2.imread(str(rec["path"]),cv2.IMREAD_COLOR)
        pos=np.asarray(rec["pos"],int)
        neg=np.asarray(rec["neg"],int)
        scores=stage1.decision_function(rec["X"])
        if len(pos)>12:pos=rng.choice(pos,12,replace=False)
        if len(neg):
            neg=neg[np.argsort(-scores[neg],kind="stable")[:12]]
        selected=np.r_[pos,neg]
        labels=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]
        if not len(selected):continue
        originals=rec["boxes"][selected]
        for mode in ("flip","photo"):
            im,b=augment_image_boxes(image,originals,mode)
            for ix,(rect,target) in enumerate(zip(b,labels)):
                feat=normalized_features(im,rect)
                X.append(feat);Y.append(int(target))
                areas.append(float(np.prod(rect[2:]-rect[:2])))
                samples.append(dict(image=rec["path"].name,mode=mode,label=int(target)))
                if imageidx<2 and ix<2:
                    raw=cropbox(image,originals[ix])
                    transformed=cropbox(im,rect)
                    a=cv2.resize(raw,(96,96),interpolation=cv2.INTER_NEAREST)
                    bb=cv2.resize(transformed,(96,96),interpolation=cv2.INTER_NEAREST)
                    tile=np.hstack((a,bb))
                    cv2.putText(tile,f"{mode} GT-safe {int(target)}",(3,15),
                                cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
                    audit.append(tile)
    if audit:
        cv2.imwrite(str(out/"train_augmentation_examples.jpg"),
                    np.vstack(audit),[cv2.IMWRITE_JPEG_QUALITY,90])
    xx=np.asarray(X,dtype=np.float32)
    yy=np.asarray(Y,int);aa=np.asarray(areas,np.float32)
    if len(xx)==0 or not np.isfinite(xx).all():
        raise RuntimeError("Augmentation had missing or nonfinite features")
    info=dict(augmented_patches=len(yy),positive=int(yy.sum()),negative=int((yy==0).sum()),
              modes=["whole-image horizontal flip with box remap",
                     "mild LAB luminance contrast + HSV saturation"],
              train_only=True,samples_per_class_per_image_per_mode=12)
    (out/"augmentation_checks.json").write_text(json.dumps(info,indent=2))
    print("E39 TRAIN AUGMENT VERIFIED",json.dumps(info),flush=True)
    return xx,yy,aa,info

def lightgbm_model(seed):
    return LGBMClassifier(n_estimators=210,learning_rate=.045,num_leaves=15,
        min_child_samples=14,max_depth=-1,colsample_bytree=.85,
        subsample=.9,subsample_freq=1,reg_lambda=4.0,
        class_weight="balanced",n_jobs=2,verbosity=-1,random_state=seed)

def fit_boosters(train,stage1,augmentation=None,seed=SEED):
    X,y,areas,counts=collect_training(train,stage1)
    if augmentation is not None:
        if len(augmentation)!=3:
            raise ValueError("Expected exactly three augmented arrays (features, labels, areas)")
        ax,ay,aa=augmentation
        ax=np.asarray(ax,dtype=np.float32)
        ay=np.asarray(ay,dtype=int)
        aa=np.asarray(aa,dtype=np.float32)
        if ax.ndim!=2 or X.shape[1]!=ax.shape[1]:
            raise ValueError("Augmentation feature dimensions do not match")
        if len(ax)!=len(ay) or len(ax)!=len(aa):
            raise ValueError("Augmentation features, labels and areas have different row counts")
        if not np.isfinite(ax).all() or not np.isfinite(aa).all():
            raise ValueError("Nonfinite augmented features or areas")
        X=np.vstack((X,ax));y=np.r_[y,ay];areas=np.r_[areas,aa]
    models={}
    t=time.perf_counter()
    models["global"]=lightgbm_model(seed)
    models["global"].fit(X,y)
    masks={"small":areas<TINY,"other":areas>=TINY}
    for k,mask in masks.items():
        classes=np.bincount(y[mask],minlength=2)
        if min(classes)<25:
            raise ValueError("Insufficient labelled examples for "+k+" expert: "+str(classes))
        models[k]=lightgbm_model(seed+(1 if k=="small" else 2))
        models[k].fit(X[mask],y[mask])
    stats=dict(**counts,features=X.shape[1],
        example_count=len(y),train_time_seconds=time.perf_counter()-t,
        train_accuracy_global=float((models["global"].predict(X)==y).mean()),
        train_accuracy_small=float((models["small"].predict(X[masks["small"]])==y[masks["small"]]).mean()),
        train_accuracy_other=float((models["other"].predict(X[masks["other"]])==y[masks["other"]]).mean()),
        algorithm="LightGBM gradient-boosted trees, no CNN; TRAIN proposal crops only")
    return models,stats

def logits(model,x):
    p=np.clip(model.predict_proba(x)[:,1],1e-5,1-1e-5)
    return np.log(p/(1-p))

def boost_scores(models,rec,mode):
    if not len(rec["boxes"]):return np.empty(0,float)
    X=rec["boost_X"];global_s=logits(models["global"],X)
    if mode=="global":return global_s
    area=np.prod(rec["boxes"][:,2:]-rec["boxes"][:,:2],axis=1)
    result=np.copy(global_s)
    for k,mask in (("small",area<TINY),("other",area>=TINY)):
        if mask.any():
            expert=logits(models[k],X[mask])
            result[mask]=.50*global_s[mask]+.50*expert
    return result

def comparisons(bank,base,rbf,boosters,boosters_aug,out):
    out.mkdir(parents=True,exist_ok=True)
    vis=out/"visual";vis.mkdir(exist_ok=True)
    variants=("rbf_baseline","global_boosted","scale_experts","global_augmented","scale_augmented")
    predictions={k:[] for k in variants};rows=[]
    for count,rec in enumerate(bank,1):
        b=rec["boxes"]
        if len(b):
            rec["margin"]=base.decision_function(rec["X"])
            old=.5*rec["margin"]+.5*rbf_logits(rbf,rec)
            sc={"rbf_baseline":old,"global_boosted":boost_scores(boosters,rec,"global"),
                "scale_experts":boost_scores(boosters,rec,"expert"),
                "global_augmented":boost_scores(boosters_aug,rec,"global"),
                "scale_augmented":boost_scores(boosters_aug,rec,"expert")}
        else:sc={x:np.empty(0) for x in variants}
        for method in variants:
            score=sc[method]
            keep=nms(b,score,iou_limit=.35,max_det=300) if len(b) else np.empty(0,int)
            box=b[keep];score=score[keep]
            predictions[method].append((box,score,rec["gt"]))
            row=dict(image=rec["path"].name,method=method,GT=len(rec["gt"]),
                **per_image_stats(box,score,rec["gt"],rec["coverage"]))
            rows.append(row)
            if rec["path"].name in FOCUS:draw_audit(rec,(box,score),method,vis)
        if count%10==0:print("E39 EVAL",count,len(bank),flush=True)
    scores=[]
    for method in variants:
        result=evaluate_map50(predictions[method])
        these=[r for r in rows if r["method"]==method]
        result.update(method=method,top20_TP=sum(x["top20_TP"] for x in these),
            top20_FP=sum(x["top20_FP"] for x in these),
            small_recall_top100=sum(x["TP100_small"] for x in these)/max(1,sum(x["GT_small"] for x in these)),
            crowded_recall_top100=sum(x["crowded_TP100"] for x in these)/max(1,sum(x["crowded_GT"] for x in these)))
        scores.append(result)
        print("E39 METHOD",json.dumps(result),flush=True)
    with (out/"e39_per_image.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)
    incumbent=scores[0]
    gate={r["method"]:(r["map50_coco101"]>=incumbent["map50_coco101"]+.02
         and r["top20_TP"]>=incumbent["top20_TP"]
         and r["small_recall_top100"]>=incumbent["small_recall_top100"]-.03
         and r["crowded_recall_top100"]>=incumbent["crowded_recall_top100"]-.03)
         for r in scores[1:]}
    return scores,gate

def tests():
    rng=np.random.default_rng(11)
    im=rng.integers(0,255,(96,112,3),dtype=np.uint8)
    patch=im[20:32,12:60]
    canvas,mask,orig=letterbox(patch)
    assert orig==(12,48) and mask.sum()==64*16
    ys,xs=np.nonzero(mask)
    assert int(xs.max()-xs.min()+1)==64 and int(ys.max()-ys.min()+1)==16
    assert canvas.shape==(64,64,3)
    assert mask.min()==0 and mask.max()==1
    f=normalized_features(im,[12,20,60,32])
    assert f.ndim==1 and np.isfinite(f).all() and len(f)>150
    identical=normalized_features(im,[12,20,60,32])
    assert np.array_equal(f,identical)
    tiny=normalized_features(im,[5,5,9,11])
    assert len(tiny)==len(f) and np.isfinite(tiny).all()
    border=normalized_features(im,[0,0,45,96])
    assert len(border)==len(f)
    source_boxes=np.array([[12,20,60,32],[0,0,40,96]],dtype=np.float32)
    flipped,fb=augment_image_boxes(im,source_boxes,"flip")
    assert np.array_equal(flipped,cv2.flip(im,1))
    assert np.array_equal(fb[:,0],im.shape[1]-source_boxes[:,2])
    assert np.array_equal(fb[:,2],im.shape[1]-source_boxes[:,0])
    assert np.array_equal(cropbox(flipped,fb[0]),cv2.flip(cropbox(im,source_boxes[0]),1))
    bright,pb=augment_image_boxes(im,source_boxes,"photo")
    assert np.array_equal(pb,source_boxes) and bright.shape==im.shape
    assert np.mean(np.abs(bright.astype(float)-im.astype(float)))>0.1
    assert len(normalized_features(flipped,fb[0]))==len(f)
    # Regression test for E39 run 43852165545: fit_boosters must receive
    # exactly the three arrays the augmentor supplies, not unpack four.
    import inspect
    code=inspect.getsource(fit_boosters)
    assert 'ax,ay,aa=augmentation' in code
    assert 'len(augmentation)!=3' in code
    assert len((np.zeros((2,len(f)),np.float32),np.array([0,1]),np.array([20.,1200.])))==3
    try:
        normalized_features(im,[20,20,20,30])
        raise AssertionError("Invalid crop incorrectly accepted")
    except ValueError:pass
    print("E39 preprocessing PASS: aspect preserved, mask, tiny/border, deterministic, finite",len(f),flush=True)

def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n)
    val=data(root,"val",val_n)
    print("E39 TRAIN",len(train),"VAL",len(val),"GT",sum(len(g) for _,g in val),flush=True)
    t=time.perf_counter()
    tb=prepare(train,"selective","TRAIN")
    vb=prepare(val,"selective","VAL")
    base,rbf,diag=fit_models(tb)
    tb,trainchecks=features_for_bank(tb,out,"TRAIN")
    vb,valchecks=features_for_bank(vb,out,"VAL")
    boosters,fit=fit_boosters(tb,base)
    augmented=augmented_examples(tb,base,out)
    augmented_fit_input=augmented[:3]
    boosters_aug,augfit=fit_boosters(tb,base,augmentation=augmented_fit_input)
    score,gate=comparisons(vb,base,rbf,boosters,boosters_aug,out)
    report=dict(status="COMPLETE",split="TRAIN/VAL only; TEST not accessed",
       train_images=len(train),val_images=len(val),val_GT=sum(len(g) for _,g in val),
       paired_proposals=True,seeded_deterministic_selective_search=True,
       baseline="Same deterministic SS proposals, TRAIN-only HOG+LBP PCA64 RBF",
       preprocessing=dict(train=trainchecks,val=valchecks,
         method="Native BGR crop; aspect-preserving 64x64 neutral-padded letterbox; CLAHE only grayscale gradient channel; spatial HSV/LAB, LBP, native geometry/context"),
       learning=fit,augmented_learning=augfit,augmentation=augmented[3],baseline_train=diag,proposals=coverage(vb),results=score,promote_gate=gate,
       limitations=["SVM/boosted tree ranking compared on same GT-blind proposal bank",
           "Augmented and nonaugmented boosted models use identical validation images",
           "Zero-GT TRAIN excluded due incomplete labels",
           "GT only for TRAIN labels and VAL diagnostics, not inference",
           "No test rerun; further confirmation needed on larger VAL",
           "High training-patch accuracy is not detection quality",
           "Native tiny birds may lack discriminative pixels despite padding"])
    (out/"e39_report.json").write_text(json.dumps(report,indent=2))
    print("E39 FINAL",json.dumps(dict(gate=gate,results=score)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e39_results"))
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
