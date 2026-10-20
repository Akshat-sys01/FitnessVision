"""
MPII Human Pose Dataset - Exercise & Workout Filter
Filters images and annotations related to gym workouts, home exercises,
calisthenics (push-ups, pull-ups), resistance training (curls, shoulder press),
weight lifting, stationary cardio, yoga, pilates, and conditioning.
"""

import os
import sys
import json
import csv
import shutil
import argparse
from collections import Counter
import numpy as np
import scipy.io as sio

# Official MPII 16 body joints mapping
JOINT_NAMES = {
    0: 'r_ankle',
    1: 'r_knee',
    2: 'r_hip',
    3: 'l_hip',
    4: 'l_knee',
    5: 'l_ankle',
    6: 'pelvis',
    7: 'thorax',
    8: 'upper_neck',
    9: 'head_top',
    10: 'r_wrist',
    11: 'r_elbow',
    12: 'r_shoulder',
    13: 'l_shoulder',
    14: 'l_elbow',
    15: 'l_wrist'
}

# Core conditioning exercise activities (all from 'conditioning exercise' category)
CORE_EXERCISE_ACTIVITIES = {
    'calisthenics',                                                       # Push-ups, pull-ups, sit-ups, squats, dips
    'resistance training',                                                # Bicep curls, shoulder press, bench press, dumbbells
    'weight lifting, power lifting',                                      # Powerlifting, heavy barbell squats, deadlifts
    'circuit training',                                                   # Full-body conditioning circuits
    'home exercise, general',                                             # General home workouts
    'video exercise workouts, TV conditioning programs',                  # TV & video workout programs
    'health club exerceise classes',                                      # Gym & health club exercise classes
    'therapeutic exercise ball, Fitball exercise',                        # Stability/Swiss ball exercises
    'pilates, general',                                                   # Pilates mat & apparatus exercises
    'stretching, mild',                                                   # Flexibility & stretching routines
    'yoga, Power',                                                        # Power yoga workouts
    'yoga, Nadisodhana',                                                  # Yoga breathing & postures
    'slide board exercise, general',                                      # Slide board workouts
    'slimnastics, jazzercise',                                            # Aerobic conditioning & jazzercise
    'rowing, stationary',                                                 # Gym rowing ergometer
    'bicycling, stationary',                                              # Spin bike & stationary bike
    'Elliptical trainer, moderate effort',                                # Elliptical trainer
    'ski machine, general',                                               # Nordic ski machine
    'upper body exercise, stationary bicycle - Airdyne (arms only) 40 rpm, moderate',  # Upper body cardio
    'native New Zealander physical activities',                           # Conditioning
    'teaching exercise class'                                             # Fitness instructor
}

# Extended workout activities from other categories
EXTENDED_EXERCISE_ACTIVITIES = {
    ('dancing', 'aerobic, step'),                                         # Step aerobics
    ('dancing', 'aerobic, general'),                                      # Aerobics workouts
    ('sports', 'rope skipping, general'),                                 # Jump rope / conditioning
    ('sports', 'boxing, punching bag'),                                   # Punching bag workout
    ('sports', 'gymnastics, general'),                                    # Bodyweight strength & acrobatics
    ('water activities', 'water aerobics, water calisthenics')            # Aqua aerobics & calisthenics
}


def sanitize_filename(name):
    """Sanitize string for folder names."""
    keepchars = (' ', '.', '_', '-')
    return "".join(c for c in name if c.isalnum() or c in keepchars).rstrip()


def parse_points(annopoints):
    """Extract joint coordinates and visibility safely from annopoints struct."""
    if annopoints is None or (hasattr(annopoints, 'size') and annopoints.size == 0):
        return []
    pts = getattr(annopoints, 'point', None)
    if pts is None or (hasattr(pts, 'size') and pts.size == 0):
        return []
    if not isinstance(pts, (list, tuple)) and not hasattr(pts, '__len__'):
        pts = [pts]
    
    results = []
    for p in pts:
        try:
            jid = int(getattr(p, 'id', -1))
            x = float(getattr(p, 'x', 0.0))
            y = float(getattr(p, 'y', 0.0))
            vis = getattr(p, 'is_visible', 1)
            if hasattr(vis, 'size') and vis.size == 0:
                vis = 1
            elif isinstance(vis, (list, tuple)):
                vis = int(vis[0]) if len(vis) > 0 else 1
            else:
                vis = int(vis) if vis is not None else 1
            
            results.append({
                'id': jid,
                'name': JOINT_NAMES.get(jid, f'joint_{jid}'),
                'x': round(x, 2),
                'y': round(y, 2),
                'is_visible': vis
            })
        except Exception:
            continue
    return results


def parse_annorect(annorect):
    """Extract person bounding boxes, scale, objpos, and keypoints safely."""
    if annorect is None or (hasattr(annorect, 'size') and annorect.size == 0):
        return []
    if not isinstance(annorect, (list, tuple)) and not hasattr(annorect, '__len__'):
        annorect = [annorect]
        
    persons = []
    for r in annorect:
        try:
            head_box = {
                'x1': float(getattr(r, 'x1', -1)) if getattr(r, 'x1', None) is not None and getattr(r, 'x1') != [] else None,
                'y1': float(getattr(r, 'y1', -1)) if getattr(r, 'y1', None) is not None and getattr(r, 'y1') != [] else None,
                'x2': float(getattr(r, 'x2', -1)) if getattr(r, 'x2', None) is not None and getattr(r, 'x2') != [] else None,
                'y2': float(getattr(r, 'y2', -1)) if getattr(r, 'y2', None) is not None and getattr(r, 'y2') != [] else None,
            }
        except Exception:
            head_box = None
            
        try:
            scale_val = getattr(r, 'scale', None)
            scale = float(scale_val) if scale_val is not None and scale_val != [] else None
        except Exception:
            scale = None
            
        objpos = None
        objpos_val = getattr(r, 'objpos', None)
        if objpos_val is not None:
            try:
                objpos = {
                    'x': float(getattr(objpos_val, 'x', -1)),
                    'y': float(getattr(objpos_val, 'y', -1)),
                }
            except Exception:
                pass
                
        annopoints = getattr(r, 'annopoints', None)
        points = parse_points(annopoints)
        
        persons.append({
            'head_box': head_box,
            'scale': scale,
            'objpos': objpos,
            'points': points
        })
    return persons


def filter_dataset(mat_path, images_dir, output_dir, mode='core', copy_images=False, organize_by_activity=False):
    """Main filtering function."""
    print(f"Loading annotations from: {mat_path}...")
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    release = mat['RELEASE']
    
    total_entries = len(release.annolist)
    print(f"Total entries in dataset: {total_entries}")
    
    existing_images = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
    print(f"Total images found in {images_dir}: {len(existing_images)}")
    
    filtered_indices = []
    metadata_rows = []
    detailed_annotations = []
    activity_counts = Counter()
    
    for i in range(total_entries):
        anno = release.annolist[i]
        img_name = str(getattr(anno.image, 'name', ''))
        act_obj = release.act[i] if i < len(release.act) else None
        
        cat = str(getattr(act_obj, 'cat_name', '')).strip() if act_obj else ''
        act = str(getattr(act_obj, 'act_name', '')).strip() if act_obj else ''
        act_id = int(getattr(act_obj, 'act_id', -1)) if act_obj and getattr(act_obj, 'act_id', None) is not None else -1
        
        # Determine match
        is_match = False
        if mode in ('core', 'all'):
            if cat == 'conditioning exercise' and act in CORE_EXERCISE_ACTIVITIES:
                is_match = True
        if mode in ('extended', 'all'):
            if (cat, act) in EXTENDED_EXERCISE_ACTIVITIES:
                is_match = True
        
        if not is_match:
            continue
            
        filtered_indices.append(i)
        activity_counts[f"{cat} -> {act}"] += 1
        
        # Extract train flag
        is_train = int(release.img_train[i]) if hasattr(release, 'img_train') and i < len(release.img_train) else -1
        
        # Extract video information
        vididx = getattr(anno, 'vididx', None)
        video_id = ''
        if vididx is not None and hasattr(release, 'video_list'):
            try:
                v_idx = int(vididx) - 1 # MATLAB 1-based indexing
                if 0 <= v_idx < len(release.video_list):
                    video_id = str(release.video_list[v_idx])
            except Exception:
                pass
                
        frame_sec = float(getattr(anno, 'frame_sec', -1)) if getattr(anno, 'frame_sec', None) is not None else None
        
        # Extract person annotations
        parsed_persons = parse_annorect(getattr(anno, 'annorect', None))
        total_annotated_joints = sum(len(p['points']) for p in parsed_persons)
        
        exists_on_disk = img_name in existing_images
        
        # Append to metadata rows (for CSV)
        metadata_rows.append({
            'index': i,
            'image_name': img_name,
            'category': cat,
            'activity_caption': act,
            'activity_id': act_id,
            'is_train': is_train,
            'video_id': video_id,
            'frame_sec': frame_sec,
            'num_persons': len(parsed_persons),
            'num_annotated_joints': total_annotated_joints,
            'image_exists_on_disk': exists_on_disk
        })
        
        # Append to detailed annotations (for JSON)
        detailed_annotations.append({
            'index': i,
            'image_name': img_name,
            'category': cat,
            'activity_caption': act,
            'activity_id': act_id,
            'is_train': is_train,
            'video_id': video_id,
            'frame_sec': frame_sec,
            'persons': parsed_persons
        })

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save CSV
    csv_path = os.path.join(output_dir, 'exercise_metadata.csv')
    print(f"\nWriting CSV metadata to {csv_path}...")
    fieldnames = [
        'index', 'image_name', 'category', 'activity_caption', 'activity_id',
        'is_train', 'video_id', 'frame_sec', 'num_persons', 'num_annotated_joints',
        'image_exists_on_disk'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
        
    # 2. Save JSON
    json_path = os.path.join(output_dir, 'exercise_annotations.json')
    print(f"Writing detailed annotations with joint coordinates to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': 'MPII Human Pose Dataset - Filtered Exercise & Workout Subset',
            'mode': mode,
            'total_images': len(detailed_annotations),
            'joint_names': JOINT_NAMES,
            'annotations': detailed_annotations
        }, f, indent=2)

    # 3. Save filtered subset as MAT file
    mat_out_path = os.path.join(output_dir, 'mpii_exercise_subset.mat')
    print(f"Saving subset MATLAB structure to {mat_out_path}...")
    filtered_annolist = release.annolist[filtered_indices]
    filtered_act = release.act[filtered_indices]
    filtered_train = release.img_train[filtered_indices] if hasattr(release, 'img_train') else []
    filtered_single = release.single_person[filtered_indices] if hasattr(release, 'single_person') else []
    
    filtered_release = {
        'annolist': filtered_annolist,
        'act': filtered_act,
        'img_train': filtered_train,
        'single_person': filtered_single,
        'video_list': release.video_list if hasattr(release, 'video_list') else []
    }
    sio.savemat(mat_out_path, {'RELEASE': filtered_release}, do_compression=True)

    # 4. Copy images if requested
    if copy_images:
        print(f"\nCopying {len(filtered_indices)} images to {output_dir}...")
        copied_count = 0
        for row in metadata_rows:
            img_name = row['image_name']
            src_file = os.path.join(images_dir, img_name)
            if not os.path.exists(src_file):
                continue
                
            if organize_by_activity:
                folder_name = sanitize_filename(row['activity_caption'])
                target_folder = os.path.join(output_dir, 'images_by_activity', folder_name)
            else:
                target_folder = os.path.join(output_dir, 'images')
                
            os.makedirs(target_folder, exist_ok=True)
            dst_file = os.path.join(target_folder, img_name)
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        print(f"Successfully copied {copied_count} images.")

    # 5. Save summary report
    report_path = os.path.join(output_dir, 'filter_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MPII DATASET EXERCISE FILTERING SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total matching images: {len(filtered_indices)}\n")
        f.write(f"Images verified on disk: {sum(1 for r in metadata_rows if r['image_exists_on_disk'])}\n")
        f.write(f"Total annotated persons: {sum(r['num_persons'] for r in metadata_rows)}\n")
        f.write(f"Total annotated joints: {sum(r['num_annotated_joints'] for r in metadata_rows)}\n\n")
        f.write("BREAKDOWN BY ACTIVITY CAPTION:\n")
        f.write("-" * 60 + "\n")
        for act_name, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {act_name:<50} : {count:4d} images\n")
        f.write("-" * 60 + "\n")

    print("\n" + "=" * 60)
    print("FILTERING COMPLETE!")
    print(f"Mode: {mode}")
    print(f"Total exercise images filtered: {len(filtered_indices)}")
    print(f"Output files generated in: {output_dir}")
    print(f"  - CSV Metadata:          {csv_path}")
    print(f"  - Full Annotations JSON: {json_path}")
    print(f"  - Filtered MAT file:     {mat_out_path}")
    print(f"  - Summary Report:        {report_path}")
    if copy_images:
        print(f"  - Filtered Images:       {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter MPII Dataset for Gym and Home Workouts.")
    parser.add_argument('--mat', default='annotations/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat',
                        help='Path to mpii_human_pose_v1_u12_1.mat')
    parser.add_argument('--images', default='images', help='Path to images directory')
    parser.add_argument('--output-dir', default='filtered_exercise_dataset', help='Output directory')
    parser.add_argument('--mode', choices=['core', 'extended', 'all'], default='all',
                        help='core: conditioning exercises only (1755), extended: core + aerobics/rope/punching/gymnastics (2270), all: all workouts')
    parser.add_argument('--copy-images', action='store_true', help='Copy images into output directory')
    parser.add_argument('--organize-by-activity', action='store_true', help='Organize copied images into subfolders by activity name')
    
    args = parser.parse_args()
    filter_dataset(
        mat_path=args.mat,
        images_dir=args.images,
        output_dir=args.output_dir,
        mode=args.mode,
        copy_images=args.copy_images,
        organize_by_activity=args.organize_by_activity
    )
