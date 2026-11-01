import json
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("cleaned_dataset_folder")
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "exercise_annotations.json"

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

total_images = 0
total_persons = 0
total_keypoints = 0

invalid_ids = 0
duplicate_ids = 0
out_of_bounds = 0
invalid_visibility = 0
missing_images = 0
missing_scales = 0
invalid_scales = 0
empty_persons = 0

for annotation in data["annotations"]:

    total_images += 1
    image_name = annotation["image_name"]

    if not (IMAGE_DIR / image_name).exists():
        missing_images += 1

    persons = annotation.get("persons", [])

    if not persons:
        empty_persons += 1

    for person in persons:

        total_persons += 1

        scale = person.get("scale")

        if scale is None:
            missing_scales += 1
        elif not isinstance(scale, (int, float)) or scale <= 0:
            invalid_scales += 1

        points = person.get("points", [])
        point_ids = [p["id"] for p in points]

        if len(point_ids) != len(set(point_ids)):
            duplicate_ids += 1

        for point in points:

            total_keypoints += 1

            if point["id"] not in range(16):
                invalid_ids += 1

            if point["is_visible"] not in (0, 1):
                invalid_visibility += 1

            x = point["x"]
            y = point["y"]

            if not (0 <= x < 512 and 0 <= y < 512):
                out_of_bounds += 1


print("\n========== DATASET VALIDATION ==========")

print("Total images:", total_images)
print("Total persons:", total_persons)
print("Total keypoints:", total_keypoints)

print("\n========== VALIDATION ERRORS ==========")

print("Missing images:", missing_images)
print("Empty person annotations:", empty_persons)
print("Invalid keypoint IDs:", invalid_ids)
print("Duplicate keypoint IDs:", duplicate_ids)
print("Out-of-bounds keypoints:", out_of_bounds)
print("Invalid visibility values:", invalid_visibility)
print("Missing scales:", missing_scales)
print("Invalid scales:", invalid_scales)

print("\n=======================================")

if any([
    missing_images,
    empty_persons,
    invalid_ids,
    duplicate_ids,
    out_of_bounds,
    invalid_visibility,
    invalid_scales
]):
    print("RESULT: VALIDATION FAILED")
else:
    print("RESULT: VALIDATION PASSED")