from pathlib import Path
import argparse, csv, json, time, traceback
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Expected location: C:/FitnessVision/training/evaluate_phase3_pose_model.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "MPII Dataset" / "Full_MPII_Dataset" / "train_validation_dataset"
OUTPUT_ROOT = PROJECT_ROOT / "training_outputs" / "phase3_full_mpii"
CHECKPOINT = OUTPUT_ROOT / "checkpoints" / "best_model.keras"
FINAL_MODEL = OUTPUT_ROOT / "fitnessvision_pose_v2.keras"

IMAGE_SIZE, HEATMAP_SIZE, NUM_JOINTS = 256, 64, 16
PCK_THRESHOLD = 3.0
JOINT_NAMES = [
    "Right ankle", "Right knee", "Right hip", "Left hip", "Left knee",
    "Left ankle", "Pelvis", "Thorax", "Upper neck", "Head top",
    "Right wrist", "Right elbow", "Right shoulder", "Left shoulder",
    "Left elbow", "Left wrist"
]
EDGES = [
    (0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,7),(7,8),(8,9),
    (12,11),(11,10),(13,14),(14,15),(12,7),(13,7)
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Optional explicit .keras model path")
    p.add_argument("--samples", type=int, default=16, help="Number of validation images to visualize")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()

def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("persons") if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records found in {path}")
    return records

def select_model(explicit):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    if CHECKPOINT.is_file():
        return CHECKPOINT
    if FINAL_MODEL.is_file():
        return FINAL_MODEL
    raise FileNotFoundError(f"No model found at {CHECKPOINT} or {FINAL_MODEL}")

def load_batch(records, indices, image_dir, heatmap_dir):
    xs, ys, masks = [], [], []
    for i in indices:
        r = records[int(i)]
        name = r["crop_image"]
        with Image.open(image_dir / name) as im:
            rgb = np.asarray(im.convert("RGB").resize((256,256)), dtype=np.float32)
        with np.load(heatmap_dir / (Path(name).stem + ".npz"), allow_pickle=False) as z:
            hm, mask = z["heatmaps"].astype(np.float32), z["joint_mask"].astype(np.float32)
        if hm.shape != (16,64,64) or mask.shape != (16,):
            raise ValueError(f"Invalid heatmap/mask shape for {name}: {hm.shape}, {mask.shape}")
        xs.append(rgb / 127.5 - 1.0)
        ys.append(hm.transpose(1,2,0))
        masks.append(mask)
    return np.asarray(xs), np.asarray(ys), np.asarray(masks)

def evaluate(model, records, image_dir, heatmap_dir, batch_size):
    correct = np.zeros(16, dtype=np.int64)
    counts = np.zeros(16, dtype=np.int64)
    mse_sum, masked_pixels, inference_seconds = 0.0, 0, 0.0
    for start in range(0, len(records), batch_size):
        ids = np.arange(start, min(start + batch_size, len(records)))
        x, y, masks = load_batch(records, ids, image_dir, heatmap_dir)
        t = time.perf_counter()
        pred = model.predict(x, verbose=0)
        inference_seconds += time.perf_counter() - t
        mask_map = masks[:,None,None,:]
        mse_sum += float(np.sum(np.square(y-pred) * mask_map))
        masked_pixels += int(np.sum(mask_map) * 64 * 64)
        for b in range(len(ids)):
            for j in range(16):
                if masks[b,j] <= 0.5:
                    continue
                gy,gx = np.unravel_index(np.argmax(y[b,:,:,j]), (64,64))
                py,px = np.unravel_index(np.argmax(pred[b,:,:,j]), (64,64))
                counts[j] += 1
                if np.hypot(px-gx, py-gy) <= PCK_THRESHOLD:
                    correct[j] += 1
    total = int(counts.sum())
    pck = 100.0 * correct.sum()/total if total else float("nan")
    per_joint = {JOINT_NAMES[j]: (100.0*correct[j]/counts[j] if counts[j] else float("nan")) for j in range(16)}
    return {
        "masked_mse": mse_sum/masked_pixels if masked_pixels else float("nan"),
        "pck": pck, "joint_count": total, "per_joint": per_joint,
        "inference_seconds": inference_seconds
    }

def draw_pose(ax, points, color, label):
    for a,b in EDGES:
        if points[a] is not None and points[b] is not None:
            ax.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]], color=color, linewidth=1.5)
    pts = [p for p in points if p is not None]
    if pts:
        ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=25,c=color,label=label,edgecolors="white",linewidths=.4)

def visualize(model, records, image_dir, heatmap_dir, out_dir, count):
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(records), size=min(max(1,count),len(records)), replace=False)
    rows = []
    for n, idx in enumerate(chosen,1):
        r = records[int(idx)]
        name = r["crop_image"]
        with Image.open(image_dir / name) as im:
            rgb = np.asarray(im.convert("RGB").resize((256,256)))
        with np.load(heatmap_dir / (Path(name).stem + ".npz"), allow_pickle=False) as z:
            target = z["heatmaps"].astype(np.float32).transpose(1,2,0)
            mask = z["joint_mask"].astype(np.float32)
        pred = model.predict((rgb.astype(np.float32)/127.5-1.0)[None,...], verbose=0)[0]
        gt_pts, pr_pts = [None]*16, [None]*16
        good = seen = 0
        for j in range(16):
            if mask[j] <= .5: continue
            gy,gx = np.unravel_index(np.argmax(target[:,:,j]), (64,64))
            py,px = np.unravel_index(np.argmax(pred[:,:,j]), (64,64))
            gt_pts[j], pr_pts[j] = (float(gx*4),float(gy*4)), (float(px*4),float(py*4))
            seen += 1
            good += int(np.hypot(px-gx,py-gy) <= PCK_THRESHOLD)
        sample_pck = 100*good/seen if seen else float("nan")
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].imshow(rgb); draw_pose(axes[0],gt_pts,"lime","Ground truth")
        axes[0].set_title("Ground-truth visible joints"); axes[0].axis("off")
        axes[1].imshow(rgb); draw_pose(axes[1],pr_pts,"red","Prediction")
        axes[1].set_title("Model prediction"); axes[1].axis("off")
        for ax in axes: ax.legend(loc="lower right",fontsize=8)
        fig.suptitle(f"{name} | visible joints: {seen} | sample PCK@3: {sample_pck:.1f}%")
        fig.tight_layout()
        image_out = viz_dir / f"prediction_{n:02d}_{Path(name).stem}.png"
        fig.savefig(image_out,dpi=130,bbox_inches="tight"); plt.close(fig)
        rows.append({"validation_record":int(idx),"crop_image":name,"visible_joints":seen,"sample_pck_at_3":sample_pck,"visualization":str(image_out)})
    with (viz_dir/"visualization_samples.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return viz_dir, rows

def main():
    args=parse_args()
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    report=OUTPUT_ROOT/"evaluation_report.txt"
    try:
        model_path=select_model(args.model)
        ann=DATA_ROOT/"validation"/"annotations.json"
        if not ann.is_file(): raise FileNotFoundError(f"Validation annotations not found: {ann}")
        records=load_records(ann)
        image_dir=DATA_ROOT/"validation"/"images"
        heatmap_dir=DATA_ROOT/"validation"/"heatmaps"
        print(f"Model: {model_path}\nValidation samples: {len(records):,}\nLoading model...")
        model=tf.keras.models.load_model(model_path,compile=False)
        print("Evaluating all validation samples...")
        m=evaluate(model,records,image_dir,heatmap_dir,args.batch_size)
        print("Generating visualizations...")
        viz_dir,rows=visualize(model,records,image_dir,heatmap_dir,OUTPUT_ROOT,args.samples)
        lines=[
            "FitnessVision Phase 3 Evaluation Report","="*42,
            f"Model: {model_path}",f"Validation samples: {len(records):,}",
            "Input: 256x256x3", "Output: 64x64x16",
            f"Visible joint targets: {m['joint_count']:,}",
            f"Masked heatmap MSE: {m['masked_mse']:.10e}",
            f"PCK@3 heatmap pixels: {m['pck']:.2f}%",
            f"Inference time (all validation samples): {m['inference_seconds']:.2f} seconds",
            "","Per-joint PCK@3:"
        ]
        lines += [f"  {k}: {v:.2f}%" for k,v in m["per_joint"].items()]
        lines += ["",f"Visualized samples: {len(rows)}",f"Visualizations: {viz_dir}",
                  "","PCK uses argmax coordinates from target/predicted 64x64 heatmaps;",
                  "only joints with joint_mask=1 are evaluated. Threshold = 3 heatmap pixels."]
        report.write_text("\n".join(lines),encoding="utf-8")
        print("\n"+"\n".join(lines)); print(f"\nReport saved: {report}")
    except Exception as e:
        report.write_text(f"Evaluation FAILED\n{e}\n\n{traceback.format_exc()}",encoding="utf-8")
        raise

if __name__=="__main__":
    main()
