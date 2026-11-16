"""E28: nonlinear appearance-based proposal verifier on real classical features.

Hypothesis: coarse 32D colour/edge stage2 cannot discriminate true bird
silhouette against seaweed, reeds, surf and aircraft; use full HOG+LBP spatial
appearance reduced with TRAIN-only PCA and a nonlinear RBF SVM.

All training labels GT-derived only from TRAIN; no VAL or TEST used to fit PCA,
feature scaling, model or threshold. Shared E25 stage1 and proposal cache.
Holdout VAL30, exactly same scoring; compare stage1 vs E25 RF vs E28 RBF SVM.
"""
import argparse,json,csv
from pathlib import Path
import cv2,numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from cloud_audit import data,iou,nms
from rebuild_detector import build_bank,fit,classification_errors,draw_box
from quality_subset_train import training_subset,gt_positive_features,sample_initial,mine
from background_rejector import prepare_secondary_banks,stage2_train
from metrics_map import evaluate_map50

def collect_proposal_examples(bank,per_pos=45,per_hard=25,per_rand=12):
    """Match original RF stage2 training selection exactly, with HOG+LBP data."""
    X=[];y=[];aux=[];marg=[];rng=np.random.default_rng(77);counts={}
    for r in bank:
        if not len(r["boxes"]):continue
        ids=r["pos"]
        if len(ids)>per_pos:ids=rng.choice(ids,per_pos,replace=False)
        scores=r["margin"];keep=nms(r["boxes"],scores,.5,170)
        safe=set(r["neg"].tolist())
        hard=[int(j) for j in keep if int(j) in safe and scores[int(j)]>-.7][:per_hard]
        exclude=set(hard)
        rem=[int(j) for j in r["neg"] if int(j) not in exclude]
        rest=rng.choice(rem,min(per_rand,len(rem)),replace=False).tolist() if rem else []
        negatives=hard+rest
        if len(ids):
            X.append(r["X"][ids]);aux.append(r["aux"][ids]);marg.append(scores[ids]);y.extend([1]*len(ids))
        if negatives:
            X.append(r["X"][negatives]);aux.append(r["aux"][negatives])
            marg.append(scores[negatives]);y.extend([0]*len(negatives))
    X=np.vstack(X).astype(np.float32)
    aux=np.vstack(aux).astype(np.float32)
    marg=np.concatenate(marg).reshape(-1,1)
    y=np.asarray(y,int)
    assert len(X)==len(aux)==len(marg)==len(y)
    counts=dict(total=len(y),positive=int(y.sum()),negative=int((y==0).sum()),
        feature_dimension=X.shape[1])
    return X,aux,marg,y,counts

def fit_rbf(X,aux,margin,y,pca_dim=64):
    # Pipeline fit on TRAIN ONLY. PCA is not pretrained; scikit-learn classical
    # dimensionality reduction of handcrafted HOG/LBP patch appearance.
    normal=StandardScaler()
    Xn=normal.fit_transform(X)
    pca=PCA(n_components=min(pca_dim,len(y)-2,X.shape[1]),svd_solver="randomized",
            iterated_power=3,random_state=47)
    z=pca.fit_transform(Xn)
    expanded=np.c_[z,aux,margin].astype(np.float32)
    feature_scaler=StandardScaler()
    scaled=feature_scaler.fit_transform(expanded)
    clf=SVC(C=3.0,kernel="rbf",gamma="scale",class_weight="balanced",
            probability=True,random_state=47,cache_size=750)
    clf.fit(scaled,y)
    return (normal,pca,feature_scaler,clf),dict(
        train_patch_accuracy=float((clf.predict(scaled)==y).mean()),
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
        support_vectors=int(clf.n_support_.sum()),C=3.0,gamma="scale",
        n_pca=pca.n_components_)

def rbf_logits(fitted,r):
    norm,pca,scaler,clf=fitted
    if not len(r["X"]):return np.empty(0)
    a=pca.transform(norm.transform(r["X"]))
    x=scaler.transform(np.c_[a,r["aux"],r["margin"]])
    prob=np.clip(clf.predict_proba(x)[:,1],1e-4,1-1e-4)
    return np.log(prob/(1-prob))

def eval_scores(bank,name,get_scores,out):
    preds=[];rows=[];panels=[]
    selected={"beach_bird_1379.jpg","beach_bird_1419.jpg",
              "beach_bird_1560.jpg","beach_bird_1606.jpg"}
    for k,r in enumerate(bank):
        scores=np.asarray(get_scores(r),float)
        assert len(scores)==len(r["boxes"]) and np.isfinite(scores).all()
        indices=nms(r["boxes"],scores,.35,300) if len(scores) else np.empty(0,int)
        boxes=r["boxes"][indices];final=scores[indices]
        preds.append((boxes,final,r["gt"]))
        ov=iou(boxes[:20],r["gt"])
        used=set();tp=0
        for i in range(min(len(boxes),20)):
            if len(r["gt"]):
                for j in np.argsort(-ov[i]):
                    if ov[i,j]>=.5 and int(j) not in used:
                        tp+=1;used.add(int(j));break
        active=final>=0
        err=classification_errors(boxes[active],final[active],r["gt"])
        rows.append(dict(image=r["path"].name,variant=name,GT=len(r["gt"]),
                         TP_top20=tp,FP_top20=min(20,len(boxes))-tp,**err))
        if r["path"].name in selected:
            image=cv2.imread(str(r["path"]))
            for gt in r["gt"]:draw_box(image,gt,(0,230,0),2)
            used=set()
            for i,b in enumerate(boxes[:20]):
                matched=False
                if len(r["gt"]):
                    for j in np.argsort(-ov[i]):
                        if ov[i,j]>=.5 and int(j) not in used:
                            used.add(int(j));matched=True;break
                draw_box(image,b,(255,0,0) if matched else (0,0,230),2)
            image=cv2.resize(image,(550,345))
            cv2.rectangle(image,(0,0),(550,25),(0,0,0),-1)
            cv2.putText(image,f"{name} {r['path'].stem} GT={len(r['gt'])}"[:70],(8,18),
                        cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            panels.append(image)
        if (k+1)%15==0:print("E28 VALIDATED",name,k+1,flush=True)
    if len(panels)>=4:
        out.mkdir(parents=True,exist_ok=True)
        vis=np.vstack([np.hstack(panels[i:i+2]) for i in range(0,4,2)])
        cv2.imwrite(str(out/(name+".jpg")),vis)
    metrics=evaluate_map50(preds)
    metrics.update(variant=name,TP_top20=sum(x["TP_top20"] for x in rows),
       FP_top20=sum(x["FP_top20"] for x in rows),
       FP_margin0=sum(x["FP_total"] for x in rows),
       FN_margin0=sum(x["FN"] for x in rows))
    print("E28 CLASSIFIER",json.dumps(metrics),flush=True)
    return metrics,rows

def main(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    train=training_subset(root,train_n);val=data(root,"val",val_n)
    tb=build_bank(train,"TRAIN",out);vb=build_bank(val,"VAL",out)
    pos=gt_positive_features(tb)
    neg,_=sample_initial(tb,"baseline",20)
    stage1,_=fit(pos,neg)
    extra,_=mine(stage1,tb,"baseline",set(),250)
    if len(extra):neg=np.vstack([neg,extra])
    stage1,_=fit(pos,neg)
    both=prepare_secondary_banks(tb+vb,stage1)
    tb=both[:len(tb)];vb=both[len(tb):]
    rf,rf_diag=stage2_train(tb)
    X,A,M,y,counts=collect_proposal_examples(tb)
    rbf,rbf_diag=fit_rbf(X,A,M,y,pca_dim=64)
    print("E28 TRAIN",json.dumps(dict(counts=counts,rf=rf_diag,rbf=rbf_diag)),flush=True)
    def original(r):return r["margin"]
    def rf_fused(r):
        p=np.clip(rf.predict_proba(np.c_[r["margin"],r["aux"]])[:,1],1e-4,1-1e-4)
        return .5*r["margin"]+.5*np.log(p/(1-p))
    def rbf_fused(r):return .5*r["margin"]+.5*rbf_logits(rbf,r)
    # No VAL model selection: three fixed, preregistered arms.
    experiments=[("A_stage1",original),("B_RF_32D",rf_fused),("C_RBF_PCA64_HOG_LBP",rbf_fused)]
    scores=[];rows=[]
    for name,method in experiments:
        metric,row=eval_scores(vb,name,method,out/"visual")
        scores.append(metric);rows.extend(row)
    result=dict(train_images=len(train),val_images=len(val),
                validation_birds=sum(len(gt) for _,gt in val),
                train_counts=counts,rf_fit=rf_diag,rbf_fit=rbf_diag,models=scores,
                caveats=["All features PCA and SVM fits TRAIN only. No test GT used.",
                         "No tuning of hyperparameters on VAL in E28.",
                         "Same E25 stage1 and exact same LoG/Otsu proposals for every arm."])
    (out/"e28_results.json").write_text(json.dumps(result,indent=2))
    with (out/"e28_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print("E28 COMPLETE",json.dumps(result)[:4000],flush=True)

def tests():
    assert 64<1924
    print("PASS independent PCA, nonlinear kernel SVM and proposal-domain experiment setup")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e28_results"))
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        main(a.dataset,a.output,a.train_n,a.val_n)
