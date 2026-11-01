import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "images_copy"
ANNOTATION_FILE = BASE_DIR / "exercise_annotations.json"

OUTPUT_DIR = BASE_DIR / "letterboxed_512"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_ANNOTATION_FILE = OUTPUT_DIR / "exercise_annotations.json"

TARGET_SIZE = 512

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def letterbox_image(image):
    """Resize proportionally and add padding to create a 512x512 image."""

    width, height = image.size

    # Calculate proportional scaling factor
    scale = min(TARGET_SIZE / width, TARGET_SIZE / height)

    # Calculate resized dimensions
    new_width = round(width * scale)
    new_height = round(height * scale)

    # Resize image
    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Calculate centered padding
    pad_x = (TARGET_SIZE - new_width) // 2
    pad_y = (TARGET_SIZE - new_height) // 2

    # Create square canvas
    canvas = Image.new(
        "RGB",
        (TARGET_SIZE, TARGET_SIZE),
        (0, 0, 0)
    )

    canvas.paste(resized, (pad_x, pad_y))

    # Actual scale factors account for integer rounding
    scale_x = new_width / width
    scale_y = new_height / height

    return canvas, scale_x, scale_y, pad_x, pad_y


# Load original annotations
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

    with Image.open(image_path) as original:

        image_format = original.format

        image = original.convert("RGB")

        canvas, sx, sy, pad_x, pad_y = letterbox_image(image)

        output_path = OUTPUT_IMAGE_DIR / image_name

        if image_format == "JPEG":
            canvas.save(output_path, quality=95)
        else:
            canvas.save(output_path, format=image_format)

    # Transform annotations
    for person in annotation.get("persons", []):

        # Transform keypoints
        for point in person.get("points", []):

            point["x"] = point["x"] * sx + pad_x
            point["y"] = point["y"] * sy + pad_y

        # Transform head bounding box
        if person.get("head_box"):

            box = person["head_box"]

            box["x1"] = box["x1"] * sx + pad_x
            box["x2"] = box["x2"] * sx + pad_x
            box["y1"] = box["y1"] * sy + pad_y
            box["y2"] = box["y2"] * sy + pad_y

        # Transform object position
        if person.get("objpos"):

            person["objpos"]["x"] = (
                person["objpos"]["x"] * sx + pad_x
            )

            person["objpos"]["y"] = (
                person["objpos"]["y"] * sy + pad_y
            )

    processed += 1


# Save transformed annotations separately
with open(OUTPUT_ANNOTATION_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("\nLetterboxing completed!")
print(f"Processed images: {processed}")
print(f"Skipped images: {skipped}")
print(f"Output directory: {OUTPUT_DIR}")