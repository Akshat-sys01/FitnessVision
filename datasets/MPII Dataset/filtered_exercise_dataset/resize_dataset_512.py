import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "padded_1920" / "images"
ANNOTATION_FILE = BASE_DIR / "padded_1920" / "exercise_annotations.json"

OUTPUT_DIR = BASE_DIR / "resized_512"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_ANNOTATION_FILE = OUTPUT_DIR / "exercise_annotations.json"

TARGET_SIZE = 512
SCALE = TARGET_SIZE / 1920

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Load annotations
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = 0
skipped = 0

for annotation in data["annotations"]:

    image_name = annotation["image_name"]
    image_path = IMAGE_DIR / image_name

    if not image_path.exists():
        print(f"Missing image: {image_name}")
        skipped += 1
        continue

    with Image.open(image_path) as image:

        # Resize image
        image = image.convert("RGB")
        image = image.resize(
            (TARGET_SIZE, TARGET_SIZE),
            Image.Resampling.LANCZOS
        )

        output_path = OUTPUT_IMAGE_DIR / image_name
        image.save(output_path, quality=95)

    # Transform annotations
    for person in annotation.get("persons", []):

        # Transform keypoints
        for point in person.get("points", []):
            point["x"] *= SCALE
            point["y"] *= SCALE

        # Transform head bounding box
        if person.get("head_box"):
            box = person["head_box"]

            box["x1"] *= SCALE
            box["x2"] *= SCALE
            box["y1"] *= SCALE
            box["y2"] *= SCALE

        # Transform object position
        if person.get("objpos"):
            person["objpos"]["x"] *= SCALE
            person["objpos"]["y"] *= SCALE

    processed += 1

# Save annotations
with open(OUTPUT_ANNOTATION_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("\nStage 2 completed!")
print(f"Processed images: {processed}")
print(f"Skipped images: {skipped}")
print(f"Scaling factor: {SCALE}")
print(f"Output directory: {OUTPUT_DIR}")