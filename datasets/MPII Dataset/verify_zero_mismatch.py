"""
Verification script: Validates that every filtered image precisely matches
its original entry in mpii_human_pose_v1_u12_1.mat with zero index shift or mismatch.
"""
import scipy.io as sio
import json
import csv

print("Loading original MPII ground truth .mat file...")
mat = sio.loadmat('annotations/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat', struct_as_record=False, squeeze_me=True)
release = mat['RELEASE']

print("Loading filtered JSON annotations...")
with open('filtered_exercise_dataset/exercise_annotations.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

print("Loading filtered CSV metadata...")
with open('filtered_exercise_dataset/exercise_metadata.csv', 'r', encoding='utf-8') as f:
    csv_rows = list(csv.DictReader(f))

mismatches = 0
total_checked = len(json_data['annotations'])

for idx, entry in enumerate(json_data['annotations']):
    orig_idx = entry['index']
    orig_img_name = str(release.annolist[orig_idx].image.name)
    orig_cat = str(getattr(release.act[orig_idx], 'cat_name', '')).strip()
    orig_act = str(getattr(release.act[orig_idx], 'act_name', '')).strip()
    
    # Check JSON vs Original MAT
    if orig_img_name != entry['image_name']:
        print(f"MISMATCH at {idx}: JSON image {entry['image_name']} != MAT image {orig_img_name}")
        mismatches += 1
    if orig_cat != entry['category']:
        print(f"MISMATCH at {idx}: JSON cat {entry['category']} != MAT cat {orig_cat}")
        mismatches += 1
    if orig_act != entry['activity_caption']:
        print(f"MISMATCH at {idx}: JSON act {entry['activity_caption']} != MAT act {orig_act}")
        mismatches += 1
        
    # Check CSV vs JSON
    csv_row = csv_rows[idx]
    if csv_row['image_name'] != entry['image_name']:
        print(f"MISMATCH CSV vs JSON image: {csv_row['image_name']} vs {entry['image_name']}")
        mismatches += 1
    if csv_row['activity_caption'] != entry['activity_caption']:
        print(f"MISMATCH CSV vs JSON act: {csv_row['activity_caption']} vs {entry['activity_caption']}")
        mismatches += 1

print("\n" + "=" * 50)
print(f"VERIFICATION COMPLETE across {total_checked} images:")
print(f"Total Mismatches Detected: {mismatches}")
if mismatches == 0:
    print("STATUS: 100% PERFECT 1-TO-1 ALIGNMENT CONFIRMED (0 ERRORS)")
print("=" * 50)
