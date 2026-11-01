import json
from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "images_copy"
ANNOTATION_FILE = BASE_DIR / "exercise_annotations.json"

OUTPUT_DIR = BASE_DIR / "padded_1920"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_ANNOTATION_FILE = OUTPUT_DIR / "exercise_annotations.json"

# Configuration
CANVAS_SIZE = 1920
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

        image = image.convert("RGB")
        width, height = image.size

        if width > CANVAS_SIZE or height > CANVAS_SIZE:
            print(f"Image exceeds canvas size: {image_name}")
            skipped += 1
            continue

        # Calculate padding
        pad_x = (CANVAS_SIZE - width) // 2
        pad_y = (CANVAS_SIZE - height) // 2

        # Create square canvas
        canvas = Image.new(
            "RGB",
            (CANVAS_SIZE, CANVAS_SIZE),
            (0, 0, 0)
        )

        canvas.paste(image, (pad_x, pad_y))

        # Save image
        output_path = OUTPUT_IMAGE_DIR / image_name
        canvas.save(output_path, quality=95)

        # Transform person annotations
        for person in annotation.get("persons", []):

            # Transform keypoints
            for point in person.get("points", []):
                point["x"] += pad_x
                point["y"] += pad_y

            # Transform head bounding box
            if person.get("head_box"):
                box = person["head_box"]
                box["x1"] += pad_x
                box["x2"] += pad_x
                box["y1"] += pad_y
                box["y2"] += pad_y

            # Transform object position
            if person.get("objpos"):
                person["objpos"]["x"] += pad_x
                person["objpos"]["y"] += pad_y

        processed += 1

# Save transformed annotations
with open(OUTPUT_ANNOTATION_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("\nPreprocessing completed!")
print(f"Processed images: {processed}")
print(f"Skipped images: {skipped}")
print(f"Output directory: {OUTPUT_DIR}")