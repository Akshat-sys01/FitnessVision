import json
import math
from pathlib import Path

from PIL import Image


# ==================================================
# CONFIGURATION
# ==================================================

DATASET_DIR = Path("cleaned_dataset_folder")
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "exercise_annotations.json"

OUTPUT_DIR = Path("person_training_dataset")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"

INPUT_SIZE = 256

# Expand the bounding box by 20% on each side.
BBOX_MARGIN = 0.20

# ==================================================


def get_person_bbox(person, image_width, image_height):
    """
    Derive a person bounding box from MPII keypoints
    and the optional head_box annotation.

    Returns:
        x1, y1, x2, y2
    """

    xs = []
    ys = []

    # Collect available keypoints.
    for point in person.get("points", []):

        x = point.get("x")
        y = point.get("y")

        if x is None or y is None:
            continue

        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue

        xs.append(float(x))
        ys.append(float(y))

    # Include head_box when available.
    head_box = person.get("head_box")

    if head_box:

        for x_key in ("x1", "x2"):
            value = head_box.get(x_key)
            if value is not None and math.isfinite(float(value)):
                xs.append(float(value))

        for y_key in ("y1", "y2"):
            value = head_box.get(y_key)
            if value is not None and math.isfinite(float(value)):
                ys.append(float(value))

    if not xs or not ys:
        return None

    # Initial bounding box.
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    # Avoid zero-sized bounding boxes.
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)

    # Add margin.
    margin_x = box_width * BBOX_MARGIN
    margin_y = box_height * BBOX_MARGIN

    x1 = max(0, math.floor(x1 - margin_x))
    y1 = max(0, math.floor(y1 - margin_y))

    x2 = min(image_width, math.ceil(x2 + margin_x))
    y2 = min(image_height, math.ceil(y2 + margin_y))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def letterbox_image(image, points):
    """
    Resize a person crop while preserving aspect ratio.

    Returns:
        letterboxed_image
        transformed_points
        transformation_metadata
    """

    original_width, original_height = image.size

    scale = min(
        INPUT_SIZE / original_width,
        INPUT_SIZE / original_height
    )

    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    # Use actual resize ratios to account for rounding.
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (INPUT_SIZE, INPUT_SIZE),
        (0, 0, 0)
    )

    pad_x = (INPUT_SIZE - new_width) // 2
    pad_y = (INPUT_SIZE - new_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    # Transform keypoints into letterboxed coordinates.
    transformed_points = []

    for point in points:

        new_point = point.copy()

        x = point.get("x")
        y = point.get("y")

        if x is not None and y is not None:

            new_point["x"] = float(x) * scale_x + pad_x
            new_point["y"] = float(y) * scale_y + pad_y

        transformed_points.append(new_point)

    metadata = {
        "original_crop_width": original_width,
        "original_crop_height": original_height,
        "resize_scale_x": scale_x,
        "resize_scale_y": scale_y,
        "padding_x": pad_x,
        "padding_y": pad_y
    }

    return canvas, transformed_points, metadata


def main():

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read annotations.
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    person_annotations = []

    total_persons = 0
    skipped_persons = 0

    person_id = 0

    print("\nPreparing person-centred dataset...\n")

    for annotation in dataset["annotations"]:

        image_name = annotation["image_name"]

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print("Missing image:", image_name)
            skipped_persons += len(annotation.get("persons", []))
            continue

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image_width, image_height = image.size

        for person_index, person in enumerate(
            annotation.get("persons", [])
        ):

            total_persons += 1

            points = person.get("points", [])

            if not points:
                skipped_persons += 1
                continue

            bbox = get_person_bbox(
                person,
                image_width,
                image_height
            )

            if bbox is None:
                skipped_persons += 1
                continue

            x1, y1, x2, y2 = bbox

            # Crop the person.
            person_crop = image.crop(
                (x1, y1, x2, y2)
            )

            # Transform original coordinates into crop coordinates.
            crop_points = []

            for point in points:

                new_point = point.copy()

                if point.get("x") is not None:
                    new_point["x"] = float(point["x"]) - x1

                if point.get("y") is not None:
                    new_point["y"] = float(point["y"]) - y1

                crop_points.append(new_point)

            # Resize and letterbox.
            final_image, final_points, transform = letterbox_image(
                person_crop,
                crop_points
            )

            person_id += 1

            output_name = f"person_{person_id:06d}.jpg"

            output_path = OUTPUT_IMAGE_DIR / output_name

            final_image.save(
                output_path,
                quality=95
            )

            # Save transformed annotations.
            person_annotations.append({

                "person_id": person_id,

                "image_name": output_name,

                "source_image": image_name,

                "source_person_index": person_index,

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },

                "input_size": INPUT_SIZE,

                "original_scale": person.get("scale"),

                "transform": transform,

                "points": final_points

            })

    # Save dataset annotations.
    output_data = {

        "dataset_name": "FitnessVision Person Dataset",

        "input_size": INPUT_SIZE,

        "total_persons": len(person_annotations),

        "annotations": person_annotations

    }

    output_annotation_file = OUTPUT_DIR / "annotations.json"

    with open(
        output_annotation_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2
        )

    print("\n========== DATASET SUMMARY ==========")

    print("Persons processed:", len(person_annotations))

    print("Persons skipped:", skipped_persons)

    print("Total persons encountered:", total_persons)

    print("CNN input size:", INPUT_SIZE, "x", INPUT_SIZE)

    print("Images directory:", OUTPUT_IMAGE_DIR)

    print("Annotations:", output_annotation_file)

    print("\nPerson dataset preparation completed!")


if __name__ == "__main__":
    main()
