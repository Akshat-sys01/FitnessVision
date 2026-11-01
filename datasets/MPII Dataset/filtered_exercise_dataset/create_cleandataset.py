import json
import shutil
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

SOURCE_DIR = BASE_DIR / "letterboxed_512"
SOURCE_IMAGE_DIR = SOURCE_DIR / "images"
SOURCE_ANNOTATION_FILE = SOURCE_DIR / "exercise_annotations.json"

OUTPUT_DIR = BASE_DIR / "cleaned_dataset_folder"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_ANNOTATION_FILE = OUTPUT_DIR / "exercise_annotations.json"

# --------------------------------------------------
# LOAD ANNOTATIONS
# --------------------------------------------------

print("Loading annotations...")

with open(SOURCE_ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

original_annotations = data["annotations"]

# Keep only images containing at least one annotated person
filtered_annotations = [
    annotation
    for annotation in original_annotations
    if annotation.get("persons")
]

print(f"Original images: {len(original_annotations)}")
print(f"Images with persons: {len(filtered_annotations)}")

# --------------------------------------------------
# CHECK IMAGE AVAILABILITY
# --------------------------------------------------

missing_images = []

for annotation in filtered_annotations:
    image_name = annotation["image_name"]
    image_path = SOURCE_IMAGE_DIR / image_name

    if not image_path.is_file():
        missing_images.append(image_name)

if missing_images:
    print("\nERROR: Some images are missing.")

    for name in missing_images[:20]:
        print(name)

    raise FileNotFoundError(
        f"{len(missing_images)} images are missing. "
        "Dataset creation cancelled."
    )

# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# COPY FILTERED IMAGES
# --------------------------------------------------

print("\nCopying images...")

copied_images = 0

for annotation in filtered_annotations:
    image_name = annotation["image_name"]

    source_path = SOURCE_IMAGE_DIR / image_name
    destination_path = OUTPUT_IMAGE_DIR / image_name

    shutil.copy2(source_path, destination_path)

    copied_images += 1

# --------------------------------------------------
# CREATE FILTERED ANNOTATION JSON
# --------------------------------------------------

cleaned_data = data.copy()

cleaned_data["annotations"] = filtered_annotations
cleaned_data["total_images"] = len(filtered_annotations)

with open(OUTPUT_ANNOTATION_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

# --------------------------------------------------
# GENERATE CLEANING REPORT
# --------------------------------------------------

total_persons = sum(
    len(annotation.get("persons", []))
    for annotation in filtered_annotations
)

report = f"""
FITNESSVISION - CLEANED DATASET REPORT

Original images: {len(original_annotations)}
Excluded images without persons: {
    len(original_annotations) - len(filtered_annotations)
}

Images retained: {len(filtered_annotations)}
Images copied: {copied_images}

Total annotated persons: {total_persons}

Image dimensions: 512 x 512
Preprocessing: Direct letterboxing

Source dataset: letterboxed_512
Output dataset: cleaned_dataset_folder
"""

report_path = OUTPUT_DIR / "cleaning_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 45)
print("CLEANED DATASET CREATED SUCCESSFULLY!")
print("=" * 45)

print(f"Original images: {len(original_annotations)}")
print(f"Excluded images: {len(original_annotations) - len(filtered_annotations)}")
print(f"Retained images: {len(filtered_annotations)}")
print(f"Copied images: {copied_images}")
print(f"Annotated persons: {total_persons}")

print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"Annotations: {OUTPUT_ANNOTATION_FILE}")
print(f"Report: {report_path}")