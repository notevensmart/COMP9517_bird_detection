"""Full Beach Bird dataset EDA: TRAIN+VAL only, no TEST and no ML fitting.
All measurements are on original imagery/YOLO boxes, not detector predictions.
JSON, CSV, charts and real annotated exemplar contact sheets are exported.
"""
from pathlib import Path
import argparse,csv,json,hashlib,collections,math
import cv2,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cloud_audit import data,iou

def quant(values):
    a=np.asarray(values,dtype=float)
    return {str(k):float(v) for k,v in zip(("min","p05","p25","median","p75","p95","p99","max","mean"),
        ([np.min(a),*np.percentile(a,[5,25,50,75,95,99]),np.max(a),np.mean(a)] if len(a) else [float("nan")]*9))}

def safe_stats(a):
    a=np.asarray(a,dtype=np.float32)
    return [float(np.mean(a)),float(np.std(a))] if a.size else [None,None]

def image_metrics(path,gt,split):
    im=cv2.imread(str(path));h,w=im.shape[:2]
    grey=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    grad=cv2.Laplacian(grey,cv2.CV_32F)
    edge=cv2.Canny(grey,70,150)
    thumb=cv2.resize(grey,(9,8),interpolation=cv2.INTER_AREA).astype(np.int16)
    dhash="".join("1" if v else "0" for v in (thumb[:,1:]>thumb[:,:-1]).ravel())
    count=len(gt)
    row=dict(split=split,image=path.name,path=str(path),width=w,height=h,
      pixel_area=w*h,orientation="landscape" if w>h else ("portrait" if h>w else "square"),
      annotations=count,grayscale_mean=float(grey.mean()),grayscale_std=float(grey.std()),
      laplacian_variance=float(grad.var()),canny_edge_density=float((edge>0).mean()),
      hsv_mean_h=float(hsv[:,:,0].mean()),hsv_mean_s=float(hsv[:,:,1].mean()),
      hsv_mean_v=float(hsv[:,:,2].mean()),lab_mean_a=float(lab[:,:,1].mean()),
      lab_mean_b=float(lab[:,:,2].mean()),sha256=hashlib.sha256(im.tobytes()).hexdigest(),
      dhash=dhash,border_birds=0,close_bird_pairs_iou01=0,gt_max_overlap=0.)
    boxes=[]
    for i,b in enumerate(gt):
        x1,y1,x2,y2=map(float,b);bw=x2-x1;bh=y2-y1
        xlo=max(0,int(np.floor(x1)));xhi=min(w,int(np.ceil(x2)))
        ylo=max(0,int(np.floor(y1)));yhi=min(h,int(np.ceil(y2)))
        crop=grey[ylo:yhi,xlo:xhi]
        local=grey[max(0,ylo-3):min(h,yhi+3),max(0,xlo-3):min(w,xhi+3)]
        # surrounding patch's mean is not a perfect background proxy; report as local contrast
        local_contrast=abs(float(crop.mean())-float(local.mean())) if crop.size and local.size else float("nan")
        edgefrac=float((edge[ylo:yhi,xlo:xhi]>0).mean()) if crop.size else float("nan")
        area=bw*bh;fraction=area/(w*h)
        border=int(x1<=1 or y1<=1 or x2>=w-1 or y2>=h-1)
        row["border_birds"]+=border
        boxes.append(dict(split=split,image=path.name,idx=i,width_px=bw,height_px=bh,
            area_px2=area,image_area_fraction=fraction,relative_width=bw/w,
            relative_height=bh/h,aspect_ratio=bw/max(bh,1e-9),
            center_x_norm=(x1+x2)/(2*w),center_y_norm=(y1+y2)/(2*h),
            short_side_px=min(bw,bh),long_side_px=max(bw,bh),
            border_touch=border,foreground_gray_std=float(crop.std()) if crop.size else float("nan"),
            foreground_edge_density=edgefrac,local_contrast=local_contrast,
            crowded_image_gt_count=count))
    if len(gt)>1:
        ov=iou(gt,gt)
        np.fill_diagonal(ov,0.)
        row["close_bird_pairs_iou01"]=int(np.triu(ov>.1,1).sum())
        row["gt_max_overlap"]=float(ov.max())
    return row,boxes

def save_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

def make_charts(images,boxes,out):
    out.mkdir(exist_ok=True)
    from collections import Counter
    by={split:[r for r in images if r["split"]==split] for split in ("train","val")}
    for metric,title,log in [
       ("annotations","Bird annotations per image",False),
       ("laplacian_variance","Image sharpness proxy: Laplacian variance",True),
       ("grayscale_std","Image grayscale contrast",False)]:
        plt.figure(figsize=(9,5))
        for split,recs in by.items():
            z=np.array([r[metric] for r in recs],dtype=float)
            if log:z=np.log10(1+z)
            plt.hist(z,bins=40,alpha=.52,label=f"{split} ({len(recs)} images)",density=True)
        plt.legend();plt.xlabel(("log10(1+x) " if log else "")+metric);plt.ylabel("Density")
        plt.title(title);plt.tight_layout();plt.savefig(out/(metric+".png"),dpi=130);plt.close()
    for metric,title,log in [
        ("area_px2","Ground truth bird box pixel area",True),
        ("short_side_px","Bird shorter bounding-box side (px)",True),
        ("aspect_ratio","Bird bounding-box width/height",True),
        ("center_y_norm","Bird centre vertical position",False),
        ("local_contrast","Bird patch vs surrounding-box intensity",False)]:
        plt.figure(figsize=(9,5))
        for split in ("train","val"):
            z=np.array([r[metric] for r in boxes if r["split"]==split],dtype=float)
            z=z[np.isfinite(z)]
            if log:z=np.log10(np.maximum(z,.01))
            plt.hist(z,bins=45,alpha=.52,label=split,density=True)
        plt.legend();plt.xlabel(("log10 " if log else "")+metric);plt.ylabel("Density")
        plt.title(title);plt.tight_layout();plt.savefig(out/(metric+".png"),dpi=130);plt.close()
    # Scale vs vertical location illustrates perspective bias when present.
    plt.figure(figsize=(9,5))
    for split in ("train","val"):
        bb=[r for r in boxes if r["split"]==split]
        plt.scatter([r["center_y_norm"] for r in bb],
            [np.log10(max(r["area_px2"],1.)) for r in bb],s=4,alpha=.20,label=split)
    plt.xlabel("Vertical image centre fraction (top=0)");plt.ylabel("log10 bbox area px²")
    plt.legend();plt.title("Bird position versus apparent object size")
    plt.tight_layout();plt.savefig(out/"position_vs_scale.png",dpi=130);plt.close()

def tile(item,label):
    path=item["path"];im=cv2.imread(path);h,w=im.shape[:2]
    # draw exact GT boxes on original resolution, then resize
    if item["annotations"]:
        # get GT separately from filenames to guarantee exact same label processing
        from cloud_audit import boxes as load_boxes
        gt=load_boxes(Path(path).parent.parent.parent/"labels"/item["split"]/(Path(path).stem+".txt"),w,h)
        for b in gt:
            x1,y1,x2,y2=np.rint(b).astype(int)
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
    dim=(320,225);scale=min(dim[0]/w,(dim[1]-45)/h)
    small=cv2.resize(im,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
    canvas=np.full((dim[1],dim[0],3),241,np.uint8)
    left=(dim[0]-small.shape[1])//2;top=45+(dim[1]-45-small.shape[0])//2
    canvas[top:top+small.shape[0],left:left+small.shape[1]]=small
    cv2.putText(canvas,Path(path).stem,(7,17),cv2.FONT_HERSHEY_SIMPLEX,.46,(15,15,15),1)
    cv2.putText(canvas,f"{label[:28]} | GT {item['annotations']}",(7,35),
                cv2.FONT_HERSHEY_SIMPLEX,.42,(15,15,15),1)
    return canvas

def sheet(items,out,name,cols=4):
    if not items:return
    lim=24
    for page in range(math.ceil(min(len(items),lim)/(cols*3))):
        pairs=items[page*cols*3:(page+1)*cols*3]
        ims=[tile(r,caption) for r,caption in pairs]
        while len(ims)<cols*3:ims.append(np.full((225,320,3),255,np.uint8))
        contact=np.vstack([np.hstack(ims[k:k+cols]) for k in range(0,len(ims),cols)])
        cv2.imwrite(str(out/(f"{name}_{page+1:02d}.jpg")),contact,[cv2.IMWRITE_JPEG_QUALITY,86])

def dedupe_recommendations(image_rows):
    # Exact pixel checksum repeated across splits and coarse perceptual dHash matches.
    sha=collections.defaultdict(list);dhash=collections.defaultdict(list)
    for r in image_rows:
        sha[r["sha256"]].append((r["split"],r["image"]))
        dhash[r["dhash"]].append((r["split"],r["image"]))
    exact=[v for v in sha.values() if len(v)>1]
    perceptual=[v for v in dhash.values() if len(v)>1]
    return dict(exact_pixel_groups=exact[:70],exact_pixel_group_count=len(exact),
         exact_dhash_group_count=len(perceptual),dhash_groups_sample=perceptual[:35],
         note="dHash exact matches indicate possible near-duplicates, but can coincide on uniform images; review visually before asserting leakage.")

def summary(images,boxes):
    result={}
    for split in ("train","val"):
        ir=[r for r in images if r["split"]==split]
        br=[b for b in boxes if b["split"]==split]
        sizes=[(r["width"],r["height"]) for r in ir]
        result[split]=dict(
            images=len(ir),annotated_birds=len(br),images_without_annotation=sum(r["annotations"]==0 for r in ir),
            width_px=quant([r["width"] for r in ir]),height_px=quant([r["height"] for r in ir]),
            dimensions_top=collections.Counter(sizes).most_common(12),
            birds_per_image=quant([r["annotations"] for r in ir]),
            images_with_at_least_10_birds=sum(r["annotations"]>=10 for r in ir),
            images_with_at_least_20_birds=sum(r["annotations"]>=20 for r in ir),
            images_with_at_least_50_birds=sum(r["annotations"]>=50 for r in ir),
            bounding_box_area_px2=quant([b["area_px2"] for b in br]),
            bbox_area_fraction=quant([b["image_area_fraction"] for b in br]),
            shorter_box_side_px=quant([b["short_side_px"] for b in br]),
            aspect_ratio=quant([b["aspect_ratio"] for b in br]),
            boxes_area_less_256=sum(b["area_px2"]<256 for b in br),
            boxes_area_less_1024=sum(b["area_px2"]<1024 for b in br),
            boxes_short_side_less_8=sum(b["short_side_px"]<8 for b in br),
            boxes_short_side_less_16=sum(b["short_side_px"]<16 for b in br),
            boxes_area_greater_10000=sum(b["area_px2"]>=10000 for b in br),
            boxes_aspect_wide_2=sum(b["aspect_ratio"]>2 for b in br),
            boxes_aspect_tall_half=sum(b["aspect_ratio"]<.5 for b in br),
            boxes_touches_border=sum(b["border_touch"] for b in br),
            crowding_images_with_gt_overlap_01=sum(r["close_bird_pairs_iou01"]>0 for r in ir),
            bird_center_y=quant([b["center_y_norm"] for b in br]),
            bird_local_contrast=quant([b["local_contrast"] for b in br if np.isfinite(b["local_contrast"])]),
            image_laplacian_var=quant([r["laplacian_variance"] for r in ir]),
            image_edge_fraction=quant([r["canny_edge_density"] for r in ir]),
            image_brightness=quant([r["grayscale_mean"] for r in ir]))
    return result

def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    imrows=[];boxrows=[];samples={}
    for split in ("train","val"):
        rec=data(root,split,0)
        for k,(p,gt) in enumerate(rec):
            ir,br=image_metrics(p,gt,split)
            imrows.append(ir);boxrows.extend(br)
            if (k+1)%100==0 or (k+1)==len(rec):
                print("EDA",split,k+1,"/",len(rec),"boxes",len(boxrows),flush=True)
    save_csv(out/"images.csv",imrows);save_csv(out/"boxes.csv",boxrows)
    stats=summary(imrows,boxrows);duplicates=dedupe_recommendations(imrows)
    (out/"statistics.json").write_text(json.dumps(stats,indent=2,allow_nan=False))
    (out/"potential_duplicates.json").write_text(json.dumps(duplicates,indent=2))
    make_charts(imrows,boxrows,out/"charts")
    visuals=out/"contact_sheets";visuals.mkdir(exist_ok=True)
    bybox=collections.defaultdict(list)
    for b in boxrows:bybox[(b["split"],b["image"])].append(b)
    def exemplars(split,key,reverse=False,positive_only=False):
        subset=[r for r in imrows if r["split"]==split and (not positive_only or r["annotations"]>0)]
        return sorted(subset,key=lambda r:(r[key],r["image"]),reverse=reverse)
    for split in ("train","val"):
        cases={
            "most_crowded":[(r,"high bird count") for r in exemplars(split,"annotations",True)[:12]],
            "empty_labels":[(r,"no GT labels") for r in exemplars(split,"image") if not r["annotations"]][:12],
            "blur_or_low_laplacian":[(r,"low sharpness") for r in exemplars(split,"laplacian_variance")[:12]],
            "high_background_edges":[(r,"busy edges") for r in exemplars(split,"canny_edge_density",True)[:12]],
            "low_scene_contrast":[(r,"low contrast") for r in exemplars(split,"grayscale_std")[:12]],
        }
        for objkey,reverse in [("area_px2",False),("area_px2",True),
                               ("aspect_ratio",False),("aspect_ratio",True),
                               ("local_contrast",False)]:
            ordered=sorted((b for b in boxrows if b["split"]==split and np.isfinite(b[objkey])),
                 key=lambda b:(b[objkey],b["image"]),reverse=reverse)
            imgs=[];seen=set()
            for b in ordered:
                if b["image"] not in seen:
                    r=next(r for r in imrows if r["image"]==b["image"] and r["split"]==split)
                    imgs.append((r,f"{objkey} {b[objkey]:.2f}"))
                    seen.add(b["image"])
                if len(imgs)>=12:break
            cases[objkey+("_largest" if reverse else "_smallest")]=imgs
        for name,items in cases.items():sheet(items,visuals,split+"_"+name)
    print("EDA SUMMARY",json.dumps({s:{k:v for k,v in stats[s].items() if k in (
          "images","annotated_birds","images_without_annotation","images_with_at_least_20_birds",
          "boxes_area_less_1024","boxes_short_side_less_8","boxes_touches_border",
          "boxes_area_greater_10000","dimensions_top")} for s in stats}),flush=True)
    print("EDA DUPLICATES",json.dumps({k:v for k,v in duplicates.items() if k.endswith("count")}),flush=True)

def tests():
    assert quant([1,2,3])["median"]==2
    assert abs(quant([1,2,3])["mean"]-2)<1e-10
    print("PASS distribution quantile and summary checks",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,default=Path("full_dataset_eda"))
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();tests()
    if not a.self_test:
        if not a.dataset:p.error("--dataset required")
        main(a.dataset,a.output)
