"""E36 deterministic rerun test for classical Selective Search + PCA/RBF stage2.
Never reads TEST. Checks coordinate hashes and same-bank model rerun scores.
"""
from pathlib import Path
import argparse,hashlib,json
import cv2,numpy as np,sklearn,scipy
from cloud_audit import data,nms
from quality_subset_train import training_subset
from selective_search_detection import selective_boxes,prepare,fit_models
from nonlinear_appearance_study import rbf_logits
from e35_bounded_small_hybrid import small_supplement

def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def score(rec,base,rbf):
    rec["margin"]=base.decision_function(rec["X"])
    s=.5*rec["margin"]+.5*rbf_logits(rbf,rec)
    keep=nms(rec["boxes"],s,.35,300)
    return s,keep

def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    val=data(root,"val",0)
    names=["beach_bird_1379.jpg","beach_bird_1389.jpg"]
    subset=[(p,g) for p,g in val if p.name in names]
    assert len(subset)==len(names)
    proposal_hashes=[]
    for path,gt in subset:
        im=cv2.imread(str(path))
        first=selective_boxes(im)
        second=selective_boxes(im)
        synthetic=np.zeros((128,128,3),np.uint8)
        cv2.rectangle(synthetic,(10,10),(35,32),(255,255,255),-1)
        _=small_supplement(synthetic,cap=30,quota_per_tile=4)
        third=selective_boxes(im)
        proposal_hashes.append(dict(image=path.name,first=digest(first),second=digest(second),
            after_synthetic=digest(third),first_count=len(first),same_after_repeat=np.array_equal(first,second),
            same_after_synthetic=np.array_equal(first,third)))
    train=training_subset(root,10)
    bank=prepare(train,"selective","TRAIN")
    vbank=prepare(subset,"selective","VAL")
    b1,r1,_=fit_models(bank)
    first=[score(r,b1,r1) for r in vbank]
    b2,r2,_=fit_models(bank)
    second=[score(r,b2,r2) for r in vbank]
    compare=[]
    for (p,_),(s1,k1),(s2,k2) in zip(subset,first,second):
        compare.append(dict(image=p.name,score_same=np.array_equal(s1,s2),
           max_abs_score_diff=float(np.max(np.abs(s1-s2))) if len(s1) else 0,
           chosen_nms_same=np.array_equal(k1,k2),first_score_hash=digest(s1),
           second_score_hash=digest(s2)))
    out_json=dict(status="COMPLETE",train_n=10,val_n=2,
       version=dict(opencv=cv2.__version__,numpy=np.__version__,
                    sklearn=sklearn.__version__,scipy=scipy.__version__),
       proposal_hashes=proposal_hashes,repeated_fits=compare,test_split_used=False,
       finding="Identical hashes across repeated local calls do NOT establish cross-run reproducibility; compare package versions and image/feature hashes for remote runs.")
    (out/"e36_determinism.json").write_text(json.dumps(out_json,indent=2))
    print("E36 DETERMINISM",json.dumps(out_json),flush=True)

def tests():
    assert digest(np.zeros((1,4),np.float32))==digest(np.zeros((1,4),np.float32))
    print("E36 reproducibility hash self-test PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path,default=Path("e36_results"))
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if a.dataset is None:p.error("--dataset required")
        main(a.dataset,a.output)
