"""
Verification Script:
Loads the recovered exercise_annotations.json, verifies keypoints on real images,
and saves annotated verification images (both individual and a 4-panel comparison montage).
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont

# Path setup
JSON_PATH = 'filtered_exercise_dataset/exercise_annotations.json'
IMAGES_DIR = 'filtered_exercise_dataset/images'
OUTPUT_DIR = 'filtered_exercise_dataset/verification_samples'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 16 MPII Joints
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


def render_annotation(image_name, entry, images_dir):
    img_path = os.path.join(images_dir, image_name)
    if not os.path.exists(img_path):
        return None
        
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Header banner
    caption = f"Exercise: {entry.get('activity_caption')} | Image: {image_name}"
    banner_h = max(36, int(h * 0.04))
    draw.rectangle([(0, 0), (w, banner_h)], fill=(15, 23, 42))
    draw.text((16, int(banner_h * 0.2)), caption, fill=(250, 204, 21))
    
    scale_factor = max(1.0, min(w, h) / 700.0)
    line_w = max(3, int(4 * scale_factor))
    pt_r = max(5, int(6 * scale_factor))
    
    for person in entry.get('persons', []):
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
            draw.ellipse([(x - pt_r, y - pt_r), (x + pt_r, y + pt_r)], fill=color, outline=(255, 255, 255), width=2)
            
        # Head box
        hb = person.get('head_box')
        if hb and hb.get('x1') is not None and hb['x1'] > 0:
            draw.rectangle([(hb['x1'], hb['y1']), (hb['x2'], hb['y2'])], outline=(234, 179, 8), width=max(2, int(2 * scale_factor)))
            
    return img


def main():
    print(f"Reading recovered annotations from: {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total annotations in JSON: {len(data['annotations'])}")
    
    # Pick distinct exercise activities with clean annotations
    targets = [
        'calisthenics',
        'resistance training',
        'yoga, Power',
        'rowing, stationary'
    ]
    
    selected = {}
    for entry in data['annotations']:
        act = entry.get('activity_caption')
        if act in targets and act not in selected:
            # check that it has at least 1 person with >= 12 joints
            for p in entry.get('persons', []):
                if len(p.get('points', [])) >= 12:
                    selected[act] = entry
                    break
        if len(selected) == len(targets):
            break
            
    rendered_images = []
    
    for act, entry in selected.items():
        img_name = entry['image_name']
        rendered = render_annotation(img_name, entry, IMAGES_DIR)
        if rendered:
            clean_act = act.replace(',', '').replace(' ', '_')
            single_path = os.path.join(OUTPUT_DIR, f"verify_{clean_act}_{img_name}")
            rendered.save(single_path, quality=95)
            print(f"Saved verified sample [{act}]: {single_path}")
            rendered_images.append((act, rendered))

    # Also build a 2x2 comparison montage (1280x960)
    if len(rendered_images) >= 4:
        montage_w, montage_h = 1600, 1000
        montage = Image.new('RGB', (montage_w, montage_h), color=(30, 41, 59))
        
        cell_w, cell_h = montage_w // 2, montage_h // 2
        positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
        
        for idx, (act, img) in enumerate(rendered_images[:4]):
            resized_cell = img.resize((cell_w - 4, cell_h - 4))
            pos = positions[idx]
            montage.paste(resized_cell, (pos[0] + 2, pos[1] + 2))
            
        montage_path = os.path.join(OUTPUT_DIR, "verification_montage_4_exercises.jpg")
        montage.save(montage_path, quality=95)
        print(f"\nSaved 4-exercise verification montage to: {montage_path}")
        
    print("\nVerification completed successfully!")

if __name__ == '__main__':
    main()
