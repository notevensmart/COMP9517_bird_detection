"""E27: manually audited bird-free scenes for a proposal-domain background rejector.

Stage1 (GT jitter + 1 round hard negative) held constant and trained on the
same 45 annotated-positive TRAIN. Compare stage2 RandomForest trained on 45
positive scenes against stage2 using 45 positive scenes + 14 MANUALLY VIEWED
bird-free TRAIN negatives. Both scores use alpha .50 and identical val30,
proposals, NMS, top20 and COCO evaluation. No test/val GT used in learning.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
from cloud_audit import data
from rebuild_detector import build_bank,fit
from quality_subset_train import training_subset,gt_positive_features,sample_initial,mine
from background_rejector import prepare_secondary_banks,stage2_train,evaluate_variant

AUDITED_NEGATIVE_TRAIN={
  "beach_bird_0012.jpg":"Rock formations in ocean, no discernible birds",
  "beach_bird_0051.jpg":"Seaweed on rocks, no discernible birds",
  "beach_bird_0052.jpg":"Open sea, no discernible birds",
  "beach_bird_0055.jpg":"Rock outcrops, no discernible birds",
  "beach_bird_0057.jpg":"Shadow on sand, no discernible birds",
  "beach_bird_0067.jpg":"Quadcopter in sky (not bird)",
  "beach_bird_0091.jpg":"Colourful kite in sky",
  "beach_bird_0150.jpg":"Bat close-up (not bird)",
  "beach_bird_0301.jpg":"Waves breaking on rocks",
  "beach_bird_0302.jpg":"Foamy water",
  "beach_bird_0304.jpg":"Beach and breakers",
  "beach_bird_0305.jpg":"Sunset and surf",
  "beach_bird_0306.jpg":"Sea and columns",
  "beach_bird_0307.jpg":"Rock pillars in ocean",
}
def run(root,out,train_n,val_n):
    out.mkdir(parents=True,exist_ok=True)
    pos_train=training_subset(root,train_n)
    whole=data(root,"train",0)
    index={p.name:(p,gt) for p,gt in whole}
    missing=set(AUDITED_NEGATIVE_TRAIN)-index.keys()
    if missing:raise ValueError(f"Missing manually reviewed image files: {missing}")
    negatives=[index[n] for n in sorted(AUDITED_NEGATIVE_TRAIN)]
    if any(len(gt) for _,gt in negatives):
        raise ValueError("Expected empty supplied annotations on manually reviewed negatives")
    assert not (set(p.name for p,_ in pos_train)&set(AUDITED_NEGATIVE_TRAIN))
    val=data(root,"val",val_n)
    bank=build_bank(pos_train,"POS_TRAIN",out)
    nb=build_bank(negatives,"AUDITED_NEGATIVE_TRAIN",out)
    vb=build_bank(val,"VAL",out)
    positive=gt_positive_features(bank)
    initial,_=sample_initial(bank,"baseline",20)
    stage1,_=fit(positive,initial)
    extra,_=mine(stage1,bank,"baseline",set(),250)
    if len(extra):initial=np.vstack((initial,extra))
    stage1,_=fit(positive,initial)
    both=prepare_secondary_banks(bank+nb+vb,stage1)
    bank=both[:len(bank)];nb=both[len(bank):len(bank)+len(nb)];vb=both[len(bank)+len(nb):]
    basemodel,basecounts=stage2_train(bank)
    audmodel,audcounts=stage2_train(bank+nb)
    metrics=[];rows=[]
    for name,model in [("E25_stage2_positive_scenes_only",basemodel),
                       ("E27_stage2_plus_manually_audited_negative_scenes",audmodel)]:
        result,rr=evaluate_variant(vb,stage1,model,.50,out/"visual"/name)
        metrics.append(dict(name=name,**result))
        rows.extend(dict(variant=name,**x) for x in rr)
        print("E27 PAIRED MODEL",json.dumps(metrics[-1]),flush=True)
    audit=dict(train_positive_images=len(pos_train),additional_negative_images=len(nb),
      additional_negative_names=sorted(AUDITED_NEGATIVE_TRAIN),validation_images=len(vb),
      validation_gt=sum(len(gt) for _,gt in val),
      stage2_baseline=basecounts,stage2_with_negative_scenes=audcounts,
      models=metrics,protocol="Train only for fitting; same val30 and fixed RF score blend alpha.50",
      caveat="Manual visual review of thumbnails is not a formal independently verified relabeling; zero-GT label risk remains. Do not touch TEST.")
    (out/"e27_results.json").write_text(json.dumps(audit,indent=2))
    with (out/"e27_per_image.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print("E27 COMPLETE",json.dumps(audit),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("e27_results"))
    p.add_argument("--train-n",type=int,default=45)
    p.add_argument("--val-n",type=int,default=30)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    assert len(AUDITED_NEGATIVE_TRAIN)==14
    print("PASS manually reviewed negative-scene manifest",flush=True)
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        run(a.dataset,a.output,a.train_n,a.val_n)
