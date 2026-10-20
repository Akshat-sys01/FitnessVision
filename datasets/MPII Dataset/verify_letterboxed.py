"""
Verification Script for letterboxed_dataset:
1. Audits mathematical alignment between original annotations, preprocessing metadata,
   and the letterboxed annotations across all 2,270 images.
2. Selects random diverse samples from the letterboxed dataset and draws the keypoints,
   limbs, and head boxes to visually verify perfect correspondence.
3. Saves annotated sample images and a multi-panel montage.
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw

# Paths
BASE_DIR = 'filtered_exercise_dataset'
LB_DIR = os.path.join(BASE_DIR, 'letterboxed_dataset')
ORIG_JSON = os.path.join(BASE_DIR, 'exercise_annotations.json')
LB_JSON = os.path.join(LB_DIR, 'exercise_annotations.json')
META_JSON = os.path.join(LB_DIR, 'preprocessing_metadata.json')
LB_IMAGES = os.path.join(LB_DIR, 'images')
OUTPUT_DIR = os.path.join(LB_DIR, 'verification_samples')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 16 MPII Joints and Skeleton Connections
JOINT_NAMES = {
    0: 'r_ankle', 1: 'r_knee', 2: 'r_hip', 3: 'l_hip', 4: 'l_knee', 5: 'l_ankle',
    6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 9: 'head_top', 10: 'r_wrist',
    11: 'r_elbow', 12: 'r_shoulder', 13: 'l_shoulder', 14: 'l_elbow', 15: 'l_wrist'
}

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 6), (3, 6), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]

RIGHT_JOINTS = {0, 1, 2, 10, 11, 12}
LEFT_JOINTS = {3, 4, 5, 13, 14, 15}


def run_mathematical_audit():
    print("=" * 60)
    print("STEP 1: MATHEMATICAL & ALIGNMENT AUDIT ACROSS ALL SAMPLES")
    print("=" * 60)
    
    with open(ORIG_JSON, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)
    with open(LB_JSON, 'r', encoding='utf-8') as f:
        lb_data = json.load(f)
    with open(META_JSON, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
        
    orig_map = {e['image_name']: e for e in orig_data['annotations']}
    lb_map = {e['image_name']: e for e in lb_data['annotations']}
    meta_images = meta_data['images']
    
    total_images_checked = 0
    total_joints_checked = 0
    max_discrepancy = 0.0
    out_of_bounds_count = 0
    
    for img_name, orig_entry in orig_map.items():
        if img_name not in lb_map or img_name not in meta_images:
            continue
            
        lb_entry = lb_map[img_name]
        m = meta_images[img_name]
        scale_x = m['scale_x']
        scale_y = m['scale_y']
        pad_left = m['pad_left']
        pad_top = m['pad_top']
        
        orig_persons = orig_entry.get('persons', [])
        lb_persons = lb_entry.get('persons', [])
        
        if len(orig_persons) != len(lb_persons):
            print(f"Person count mismatch in {img_name}")
            continue
            
        total_images_checked += 1
        
        for p_idx in range(len(orig_persons)):
            op = orig_persons[p_idx]
            lp = lb_persons[p_idx]
            
            orig_pts = {pt['id']: pt for pt in op.get('points', [])}
            lb_pts = {pt['id']: pt for pt in lp.get('points', [])}
            
            for jid, opt in orig_pts.items():
                if jid in lb_pts:
                    total_joints_checked += 1
                    ox, oy = opt['x'], opt['y']
                    lx, ly = lb_pts[jid]['x'], lb_pts[jid]['y']
                    
                    expected_lx = ox * scale_x + pad_left
                    expected_ly = oy * scale_y + pad_top
                    
                    diff = max(abs(lx - expected_lx), abs(ly - expected_ly))
                    if diff > max_discrepancy:
                        max_discrepancy = diff
                        
                    if lx < 0 or lx > 256 or ly < 0 or ly > 256:
                        out_of_bounds_count += 1
                        
    print(f"Total Images Audited: {total_images_checked}")
    print(f"Total Keypoints Audited: {total_joints_checked}")
    print(f"Maximum Transformation Discrepancy: {max_discrepancy:.6f} px")
    print(f"Out-of-Bounds Points: {out_of_bounds_count}")
    if max_discrepancy < 1e-4:
        print(">> MATHEMATICAL VERIFICATION PASSED: Exact 1-to-1 affine correspondence confirmed!")
    print("=" * 60 + "\n")


def render_letterboxed_sample(img_name, lb_entry, lb_images_dir):
    img_path = os.path.join(lb_images_dir, img_name)
    if not os.path.exists(img_path):
        return None
        
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size # Should be 256x256
    
    # Overlay header banner
    caption = f"{lb_entry.get('activity_caption')[:28]}"
    draw.rectangle([(0, 0), (w, 18)], fill=(15, 23, 42))
    draw.text((6, 2), caption, fill=(250, 204, 21))
    
    line_w = 2
    pt_r = 3
    
    for person in lb_entry.get('persons', []):
        pts = {p['id']: (p['x'], p['y']) for p in person.get('points', []) if p['x'] > 0 and p['y'] > 0}
        
        # Limbs
        for j1, j2 in SKELETON_EDGES:
            if j1 in pts and j2 in pts:
                if j1 in RIGHT_JOINTS and j2 in RIGHT_JOINTS:
                    color = (239, 68, 68)    # Red for right side
                elif j1 in LEFT_JOINTS and j2 in LEFT_JOINTS:
                    color = (59, 130, 246)   # Blue for left side
                else:
                    color = (34, 197, 94)    # Green for spine/center
                draw.line([pts[j1], pts[j2]], fill=color, width=line_w)
                
        # Joints
        for jid, (x, y) in pts.items():
            if jid in RIGHT_JOINTS:
                color = (239, 68, 68)
            elif jid in LEFT_JOINTS:
                color = (59, 130, 246)
            else:
                color = (34, 197, 94)
            draw.ellipse([(x - pt_r, y - pt_r), (x + pt_r, y + pt_r)], fill=color, outline=(255, 255, 255))
            
        # Head box
        hb = person.get('head_box')
        if hb and hb.get('x1') is not None and hb['x1'] > 0:
            draw.rectangle([(hb['x1'], hb['y1']), (hb['x2'], hb['y2'])], outline=(234, 179, 8), width=1)
            
    return img


def generate_random_visualizations(num_samples=6):
    print("=" * 60)
    print(f"STEP 2: GENERATING VISUAL VERIFICATION SAMPLES ({num_samples} RANDOM EXERCISES)")
    print("=" * 60)
    
    with open(LB_JSON, 'r', encoding='utf-8') as f:
        lb_data = json.load(f)
        
    # Filter entries that have at least one person with valid joints
    valid_entries = []
    for e in lb_data['annotations']:
        has_pts = any(len(p.get('points', [])) >= 8 for p in e.get('persons', []))
        if has_pts:
            valid_entries.append(e)
            
    print(f"Pool of well-annotated images: {len(valid_entries)}")
    
    # Group by activity to sample diverse exercises
    by_activity = {}
    for e in valid_entries:
        act = e.get('activity_caption', 'other')
        if act not in by_activity:
            by_activity[act] = []
        by_activity[act].append(e)
        
    random.seed(2026) # Reproducible random selection
    chosen_activities = [
        'calisthenics',
        'resistance training',
        'circuit training',
        'yoga, Power',
        'rowing, stationary',
        'rope skipping, general'
    ]
    
    selected_samples = []
    for act in chosen_activities:
        if act in by_activity:
            sample = random.choice(by_activity[act])
            selected_samples.append(sample)
            
    rendered_list = []
    for idx, entry in enumerate(selected_samples):
        img_name = entry['image_name']
        act = entry['activity_caption']
        rendered = render_letterboxed_sample(img_name, entry, LB_IMAGES)
        if rendered:
            clean_act = act.replace(',', '').replace(' ', '_')
            out_file = os.path.join(OUTPUT_DIR, f"random_sample_{idx+1}_{clean_act}_{img_name}")
            rendered.save(out_file, quality=95)
            print(f"  Sample {idx+1}: [{act}] -> Saved: {out_file}")
            rendered_list.append((act, img_name, rendered))
            
    # Build a 3x2 Grid Montage of the 256x256 Letterboxed Images
    if len(rendered_list) >= 6:
        grid_w = 256 * 3 + 16
        grid_h = 256 * 2 + 16
        grid_img = Image.new('RGB', (grid_w, grid_h), color=(15, 23, 42))
        
        for idx, (act, img_name, r_img) in enumerate(rendered_list[:6]):
            row = idx // 3
            col = idx % 3
            x_pos = 4 + col * (256 + 4)
            y_pos = 4 + row * (256 + 4)
            grid_img.paste(r_img, (x_pos, y_pos))
            
        montage_path = os.path.join(OUTPUT_DIR, "letterbox_verification_grid_6_samples.jpg")
        grid_img.save(montage_path, quality=95)
        print(f"\nSaved 6-sample verification grid montage: {montage_path}")
        
    print("=" * 60)
    print("VISUAL VERIFICATION GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    run_mathematical_audit()
    generate_random_visualizations(num_samples=6)
