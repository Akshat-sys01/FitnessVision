import scipy.io as sio
import os
import json
from collections import Counter

mat = sio.loadmat('annotations/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat', struct_as_record=False, squeeze_me=True)
release = mat['RELEASE']

images_dir = 'images'
existing_files = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()

core_conditioning = []
extended_workouts = []

tier2_acts = {
    ('dancing', 'aerobic, step'),
    ('dancing', 'aerobic, general'),
    ('water activities', 'water aerobics, water calisthenics'),
    ('sports', 'rope skipping, general'),
    ('sports', 'boxing, punching bag'),
    ('sports', 'gymnastics, general')
}

core_by_act = Counter()
ext_by_act = Counter()

for i in range(len(release.annolist)):
    img_name = str(release.annolist[i].image.name)
    cat = str(getattr(release.act[i], 'cat_name', '')).strip()
    act = str(getattr(release.act[i], 'act_name', '')).strip()
    
    if cat == 'conditioning exercise':
        exists = img_name in existing_files
        core_conditioning.append({
            'index': i,
            'image': img_name,
            'cat': cat,
            'act': act,
            'exists': exists
        })
        core_by_act[act] += 1
    elif (cat, act) in tier2_acts:
        exists = img_name in existing_files
        extended_workouts.append({
            'index': i,
            'image': img_name,
            'cat': cat,
            'act': act,
            'exists': exists
        })
        ext_by_act[f"{cat}: {act}"] += 1

print(f"Total Core Conditioning Exercise images: {len(core_conditioning)}")
print(f"  Existing on disk: {sum(1 for x in core_conditioning if x['exists'])}")
print("\nBreakdown by activity in Core Conditioning:")
for act, cnt in core_by_act.most_common():
    print(f"  {act}: {cnt}")

print(f"\nTotal Extended Workout images: {len(extended_workouts)}")
print(f"  Existing on disk: {sum(1 for x in extended_workouts if x['exists'])}")
print("\nBreakdown by activity in Extended:")
for act, cnt in ext_by_act.most_common():
    print(f"  {act}: {cnt}")
