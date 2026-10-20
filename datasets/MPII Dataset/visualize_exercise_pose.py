"""
MPII Exercise Pose Visualizer
Draws joint keypoints, skeleton connections, and activity captions onto filtered exercise images.
Uses only Pillow (PIL) which is standard in Python environments.
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

# 16 MPII Joint Names and Skeleton edges
SKELETON_EDGES = [
    (0, 1),   # r_ankle -> r_knee
    (1, 2),   # r_knee -> r_hip
    (2, 6),   # r_hip -> pelvis
    (3, 6),   # l_hip -> pelvis
    (3, 4),   # l_hip -> l_knee
    (4, 5),   # l_knee -> l_ankle
    (6, 7),   # pelvis -> thorax
    (7, 8),   # thorax -> upper_neck
    (8, 9),   # upper_neck -> head_top
    (7, 12),  # thorax -> r_shoulder
    (12, 11), # r_shoulder -> r_elbow
    (11, 10), # r_elbow -> r_wrist
    (7, 13),  # thorax -> l_shoulder
    (13, 14), # l_shoulder -> l_elbow
    (14, 15)  # l_elbow -> l_wrist
]

RIGHT_JOINTS = {0, 1, 2, 10, 11, 12}
LEFT_JOINTS = {3, 4, 5, 13, 14, 15}


def visualize_image(image_name, annotations_json_path, images_dir, output_path):
    with open(annotations_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    anno_match = None
    for entry in data['annotations']:
        if entry['image_name'] == image_name:
            anno_match = entry
            break
            
    if not anno_match:
        print(f"Error: Image {image_name} not found in {annotations_json_path}")
        return False
        
    src_img_path = os.path.join(images_dir, image_name)
    if not os.path.exists(src_img_path):
        print(f"Error: Source image file {src_img_path} does not exist.")
        return False
        
    img = Image.open(src_img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Overlay caption at the top
    caption_text = f"Activity: {anno_match['activity_caption']} | {anno_match['category']}"
    draw.rectangle([(0, 0), (img.width, 30)], fill=(0, 0, 0, 180))
    draw.text((10, 8), caption_text, fill=(255, 255, 0))
    
    # Draw each person
    for p_idx, person in enumerate(anno_match.get('persons', [])):
        pts = {p['id']: (p['x'], p['y']) for p in person.get('points', []) if p['x'] > 0 and p['y'] > 0}
        
        # Draw skeleton limbs
        for j1, j2 in SKELETON_EDGES:
            if j1 in pts and j2 in pts:
                if j1 in RIGHT_JOINTS and j2 in RIGHT_JOINTS:
                    edge_color = (255, 80, 80)     # Red for right side
                elif j1 in LEFT_JOINTS and j2 in LEFT_JOINTS:
                    edge_color = (80, 150, 255)    # Blue for left side
                else:
                    edge_color = (0, 255, 120)     # Green for spine/center
                draw.line([pts[j1], pts[j2]], fill=edge_color, width=4)
                
        # Draw joint keypoints
        for j_id, (x, y) in pts.items():
            r = 5
            color = (255, 255, 255)
            if j_id in RIGHT_JOINTS:
                color = (255, 50, 50)
            elif j_id in LEFT_JOINTS:
                color = (50, 100, 255)
            elif j_id in (6, 7, 8, 9):
                color = (50, 255, 50)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline=(0, 0, 0))
            
        # Draw head bounding box if available
        hb = person.get('head_box')
        if hb and hb.get('x1') is not None and hb['x1'] > 0:
            draw.rectangle([(hb['x1'], hb['y1']), (hb['x2'], hb['y2'])], outline=(255, 215, 0), width=2)
            
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"Saved visualization to: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize MPII Exercise Pose Keypoints.")
    parser.add_argument('--image', default='098996023.jpg', help='Image filename (e.g. 098996023.jpg)')
    parser.add_argument('--annotations', default='filtered_exercise_dataset/exercise_annotations.json')
    parser.add_argument('--images-dir', default='images')
    parser.add_argument('--output', default='filtered_exercise_dataset/sample_visualization.jpg')
    args = parser.parse_args()
    
    visualize_image(args.image, args.annotations, args.images_dir, args.output)
