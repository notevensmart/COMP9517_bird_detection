"""Primary COMP9517 evaluation: COCO API bbox mAP@IoU=0.5, single 'bird' class.

Official COCO protocol: 101 recall samples and maxDets=100 per image.
Keep the legacy continuous-interpolation AP in a separate diagnostic field.
Never report proposal coverage or training memorisation as validation/test mAP.
"""
import contextlib, io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from cloud_audit import ap50 as continuous_ap50, iou

def evaluate_map50(predictions):
    """predictions list of (boxes XYXY, decision scores, ground truth XYXY).

    Includes empty images. Boxes and scores must be finite.
    All predictions are already NMS-filtered, but only top-100 are evaluated
    by official COCO evaluation. Returns COCO 101-point mAP@0.5.
    """
    images=[];anns=[];dets=[];annotation_id=1
    for index,(boxes,scores,gt) in enumerate(predictions,1):
        boxes=np.asarray(boxes,dtype=float).reshape(-1,4)
        scores=np.asarray(scores,dtype=float).reshape(-1)
        gt=np.asarray(gt,dtype=float).reshape(-1,4)
        if len(boxes)!=len(scores):raise ValueError("Prediction length mismatch")
        if not(np.isfinite(boxes).all() and np.isfinite(gt).all() and np.isfinite(scores).all()):
            raise ValueError("NaN or inf in predictions or GT")
        images.append(dict(id=index,width=100000,height=100000,file_name=f"image_{index}"))
        for b in gt:
            x1,y1,x2,y2=b.tolist()
            if x2<=x1 or y2<=y1:raise ValueError("Invalid GT box")
            anns.append(dict(id=annotation_id,image_id=index,category_id=1,
                  bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
            annotation_id+=1
        for k in np.argsort(-scores,kind="stable")[:100]:
            x1,y1,x2,y2=boxes[k].tolist()
            if x2<=x1 or y2<=y1:raise ValueError("Invalid prediction box")
            dets.append(dict(image_id=index,category_id=1,
                  bbox=[x1,y1,x2-x1,y2-y1],score=float(scores[k])))
    if not anns:
        return dict(map50_coco101=None,AP50_continuous=None,gt_count=0,detection_count=len(dets))
    ground=COCO()
    ground.dataset=dict(info=dict(description="COMP9517 validation"),licenses=[],
                        categories=[dict(id=1,name="bird",supercategory="bird")],
                        images=images,annotations=anns)
    with contextlib.redirect_stdout(io.StringIO()):ground.createIndex()
    if not dets:
        return dict(map50_coco101=0.0,AP50_continuous=0.0,gt_count=len(anns),detection_count=0)
    with contextlib.redirect_stdout(io.StringIO()):
        detected=ground.loadRes(dets)
        ev=COCOeval(ground,detected,"bbox")
        ev.params.imgIds=[x["id"] for x in images]
        ev.params.catIds=[1]
        ev.params.iouThrs=np.array([0.5])
        ev.params.recThrs=np.linspace(0,1,101)
        ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate()
    values=ev.eval["precision"][0,:,0,0,2]
    valid=values[values>=0]
    ap=float(valid.mean()) if len(valid) else 0.0
    # Continuous AP is only a backwards-compatibility diagnostic; it may not
    # be numerically identical to official COCO 101-point AP.
    truncated=[]
    for boxes,scores,gt in predictions:
        indices=np.argsort(-np.asarray(scores),kind="stable")[:100]
        truncated.append((np.asarray(boxes).reshape(-1,4)[indices],
                          np.asarray(scores)[indices],np.asarray(gt).reshape(-1,4)))
    return dict(map50_coco101=ap,AP50_continuous=float(continuous_ap50(truncated)),
                gt_count=len(anns),detection_count=len(dets))

def self_test():
    b=np.array([[10,10,30,30]],np.float32)
    assert np.isclose(evaluate_map50([(b,np.array([.9]),b)])["map50_coco101"],1.)
    wrong=b+50
    assert np.isclose(evaluate_map50([(wrong,np.array([.9]),b)])["map50_coco101"],0.)
    assert np.isclose(evaluate_map50([(np.empty((0,4)),np.array([]),b)])["map50_coco101"],0.)
    assert np.isclose(evaluate_map50([(b,np.array([.9]),b),(np.empty((0,4)),np.array([]),np.empty((0,4)))])["map50_coco101"],1.)
    assert evaluate_map50([(np.empty((0,4)),np.array([]),np.empty((0,4)))])["map50_coco101"] is None
    print("PASS: official COCO 101-point bbox mAP50 tests (perfect, missing, wrong, empty)",flush=True)

if __name__=="__main__":self_test()
