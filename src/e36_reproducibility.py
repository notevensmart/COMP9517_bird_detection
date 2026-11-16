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

def ordered_hash(boxes):
    arr=np.asarray(boxes,np.float32).reshape(-1,4)
    if len(arr):
        arr=arr[np.lexsort((arr[:,3],arr[:,2],arr[:,1],arr[:,0]))]
    return digest(arr)

def selective_raw(image,seed=None):
    """Diagnostic access to raw Selective Search proposals BEFORE the 1200 cap."""
    if seed is not None:cv2.setRNGSeed(int(seed))
    seg=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    seg.setBaseImage(image)
    seg.switchToSelectiveSearchFast()
    raw=np.asarray(seg.process(),dtype=np.float32).reshape(-1,4)
    h,w=image.shape[:2]
    if not len(raw):return np.empty((0,4),np.float32)
    b=np.c_[raw[:,0],raw[:,1],raw[:,0]+raw[:,2],raw[:,1]+raw[:,3]]
    wh=b[:,2:]-b[:,:2]
    keep=(wh[:,0]>=3)&(wh[:,1]>=3)&(wh[:,0]*wh[:,1]>=16)&(wh[:,0]*wh[:,1]<=.92*w*h)
    b=b[keep]
    if len(b):
        _,idx=np.unique(b,axis=0,return_index=True)
        b=b[np.sort(idx)]
    return b.astype(np.float32)

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
        raw1=selective_raw(im)
        raw2=selective_raw(im)
        seed1=selective_raw(im,seed=42)
        seed2=selective_raw(im,seed=42)
        old_threads=cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            one_thread1=selective_raw(im,seed=42)
            one_thread2=selective_raw(im,seed=42)
        finally:
            cv2.setNumThreads(old_threads)
        proposal_hashes.append(dict(image=path.name,first=digest(first),second=digest(second),
            prefix_sorted_hashes=[ordered_hash(first),ordered_hash(second),ordered_hash(third)],
            same_prefix_set_order_invariant=(ordered_hash(first)==ordered_hash(second)),
            raw_proposal_counts=[len(raw1),len(raw2)],
            raw_set_same=ordered_hash(raw1)==ordered_hash(raw2),
            seeded_raw_counts=[len(seed1),len(seed2)],
            seeded_raw_set_same=ordered_hash(seed1)==ordered_hash(seed2),
            seeded_raw_order_same=np.array_equal(seed1,seed2),
            singlethread_seeded_counts=[len(one_thread1),len(one_thread2)],
            singlethread_seeded_set_same=ordered_hash(one_thread1)==ordered_hash(one_thread2),
            singlethread_seeded_order_same=np.array_equal(one_thread1,one_thread2),
            after_synthetic=digest(third),first_count=len(first),same_after_repeat=np.array_equal(first,second),
            same_after_synthetic=np.array_equal(first,third)))
    # E37 hard gate: the capped proposal array must remain identical across
    # identical calls and after unrelated images have been processed.
    if not all(p["same_after_repeat"] and p["same_after_synthetic"]
               for p in proposal_hashes):
        raise AssertionError("Capped Selective Search remains nondeterministic")
    print("E37 CAP REPEATABILITY PASSED",flush=True)
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
