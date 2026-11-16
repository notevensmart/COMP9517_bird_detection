"""Deployment-facing final detection filtering, independent of proposal pruning.

SVM margins are NOT probabilities. Threshold must be selected on VAL, frozen,
then reused unchanged for TEST. Explicitly count proposals versus final detections.
"""
import numpy as np
from cloud_audit import nms

def final_detections(boxes,scores,threshold,max_detections=150,nms_iou=.35):
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    s=np.asarray(scores,dtype=np.float64).reshape(-1)
    if len(b)!=len(s):raise ValueError("boxes and scores have different lengths")
    if not (np.isfinite(b).all() and np.isfinite(s).all()):raise ValueError("non-finite boxes/scores")
    if len(b) and np.any(b[:,2:]<=b[:,:2]):raise ValueError("invalid box size")
    # Reject background BEFORE NMS, then suppress duplicates, then enforce safety limit.
    use=np.flatnonzero(s>=float(threshold))
    if not len(use):
        return {"boxes":np.empty((0,4),np.float32),"scores":np.empty(0,np.float64),
                "candidate_count":len(b),"passing_threshold":0,"final_count":0}
    ordered=use[np.argsort(-s[use],kind="stable")]
    keep=nms(b[ordered],s[ordered],iou_limit=nms_iou,max_det=max_detections)
    chosen=ordered[keep]
    return {"boxes":b[chosen],"scores":s[chosen],
            "candidate_count":len(b),"passing_threshold":len(use),"final_count":len(chosen)}

def tests():
    b=np.array([[0,0,10,10],[0,0,10,10],[20,20,30,30],[40,40,50,50]],np.float32)
    s=np.array([1.,.9,-.5,2.])
    out=final_detections(b,s,threshold=.5,max_detections=2)
    assert out["candidate_count"]==4 and out["passing_threshold"]==3
    assert out["final_count"]==2 and len(out["scores"])==2
    assert len(final_detections(b,s,threshold=10)["boxes"])==0
    assert len(final_detections(np.empty((0,4)),[],threshold=0)["boxes"])==0
    print("Final detection filtering tests PASS")

if __name__=="__main__":
    tests()
