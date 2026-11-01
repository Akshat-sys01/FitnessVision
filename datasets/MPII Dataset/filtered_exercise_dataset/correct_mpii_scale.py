import json
import math
import shutil
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent

ORIGINAL_IMAGE_DIR = BASE_DIR / "images_copy"

DATASETS = [
    BASE_DIR / "letterboxed_512",
    BASE_DIR / "cleaned_dataset_folder"
]

# Original MPII annotations
ORIGINAL_JSON = BASE_DIR / "exercise_annotations.json"

with open(ORIGINAL_JSON, "r", encoding="utf-8") as f:
    original_data = json.load(f)

original_annotations = {
    item["image_name"]: item
    for item in original_data["annotations"]
}

for dataset_dir in DATASETS:

    annotation_file = dataset_dir / "exercise_annotations.json"
    backup_file = dataset_dir / "exercise_annotations.scale_backup.json"

    print(f"\nProcessing: {dataset_dir.name}")

    # Create a backup only if one does not already exist.
    if backup_file.exists():
        print("Existing backup found. Restoring original annotations...")
        shutil.copy2(backup_file, annotation_file)
    else:
        shutil.copy2(annotation_file, backup_file)
        print("Created backup:", backup_file)
    
    # Load the original annotations from the restored file.
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected_persons = 0
    skipped_null_scales = 0               
        
        
    
    for annotation in data["annotations"]:

        image_name = annotation["image_name"]

        original_annotation = original_annotations.get(image_name)

        if original_annotation is None:
            raise ValueError(f"Original annotation missing: {image_name}")

        image_path = ORIGINAL_IMAGE_DIR / image_name

        with Image.open(image_path) as image:
            original_width, original_height = image.size

        resize_factor = min(
            512 / original_width,
            512 / original_height
        )

        new_width = round(original_width * resize_factor)
        new_height = round(original_height * resize_factor)

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        actual_scale_factor = math.sqrt(scale_x * scale_y)

        for person in annotation.get("persons", []):
            if person.get("scale") is not None:
                person["scale"] = (
                    float(person["scale"]) * actual_scale_factor
                )
                corrected_persons += 1
            else:
                skipped_null_scales += 1


    # Save corrected annotations
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

print("Corrected persons:", corrected_persons)
print("Skipped null scales:", skipped_null_scales)
print("Backup:", backup_file)

print("\nScale correction completed!")