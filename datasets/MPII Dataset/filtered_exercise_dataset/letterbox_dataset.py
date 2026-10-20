import json
from pathlib import Path

from PIL import Image, ImageOps


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent

IMAGE_DIR = DATASET_DIR / "images_copy"
ANNOTATION_FILE = DATASET_DIR / "exercise_annotations.json"

OUTPUT_DIR = DATASET_DIR / "letterboxed_dataset"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"

TARGET_WIDTH = 256
TARGET_HEIGHT = 256

# Padding color: black
PADDING_COLOR = (0, 0, 0)


# --------------------------------------------------
# LETTERBOX IMAGE AND TRANSFORM COORDINATES
# --------------------------------------------------

def letterbox_image(image, target_width, target_height):
    """
    Resize an image while preserving its aspect ratio,
    then add padding to reach the target dimensions.

    Returns:
        letterboxed_image
        scale
        pad_left
        pad_top
    """

    original_width, original_height = image.size

    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    # Calculate resized dimensions.
    # Use the same dimensions for the image and coordinates.
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    # Resize without distorting the aspect ratio.
    resized_image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR
    )

    # Calculate padding.
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2

    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top

    # Create the final canvas.
    canvas = Image.new(
        "RGB",
        (target_width, target_height),
        PADDING_COLOR
    )

    canvas.paste(resized_image, (pad_left, pad_top))

    # Use the actual resized dimensions to transform coordinates.
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    return (
        canvas,
        scale_x,
        scale_y,
        pad_left,
        pad_top
    )


def transform_point(point, scale_x, scale_y, pad_left, pad_top):
    """
    Transform one keypoint into the letterboxed image's
    coordinate system.
    """

    point["x"] = point["x"] * scale_x + pad_left
    point["y"] = point["y"] * scale_y + pad_top

    return point


def transform_person(
    person,
    scale_x,
    scale_y,
    pad_left,
    pad_top
):
    """
    Transform geometric annotations for one person.
    """

    # Transform the 16 body keypoints.
    for point in person.get("points", []):
        transform_point(
            point,
            scale_x,
            scale_y,
            pad_left,
            pad_top
        )

    # Transform the head bounding box, if present.
    head_box = person.get("head_box")

    if head_box is not None:
        head_box["x1"] = (
            head_box["x1"] * scale_x + pad_left
        )
        head_box["x2"] = (
            head_box["x2"] * scale_x + pad_left
        )

        head_box["y1"] = (
            head_box["y1"] * scale_y + pad_top
        )
        head_box["y2"] = (
            head_box["y2"] * scale_y + pad_top
        )

    # Transform the object position, if present.
    objpos = person.get("objpos")

    if objpos is not None:
        objpos["x"] = (
            objpos["x"] * scale_x + pad_left
        )
        objpos["y"] = (
            objpos["y"] * scale_y + pad_top
        )

    # Preserve scale, visibility flags, joint IDs,
    # joint names, and other non-geometric metadata.


# --------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# --------------------------------------------------

def main():

    # Check that input paths exist.
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not ANNOTATION_FILE.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    # Load original annotations.
    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        original_data = json.load(file)

    if not isinstance(original_data, dict):
        raise ValueError(
            "Expected the annotation JSON to contain an object."
        )

    if not isinstance(original_data.get("annotations"), list):
        raise ValueError(
            "Expected an 'annotations' list in the JSON file."
        )

    # Work on a separate in-memory copy.
    # The original JSON file will never be overwritten.
    transformed_data = json.loads(
        json.dumps(original_data)
    )

    # Create output directories.
    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = transformed_data["annotations"]

    processed = 0
    missing_images = []
    invalid_images = []
    failed_records = []

    preprocessing_metadata = {
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "padding_color_rgb": list(PADDING_COLOR),
        "images": {}
    }

    total = len(records)

    print(f"Annotation records found: {total}")
    print("Starting letterboxing...\n")

    for index, record in enumerate(records, start=1):

        image_name = record.get("image_name")

        if not image_name:
            failed_records.append({
                "record_index": index,
                "reason": "Missing image_name"
            })
            continue

        # Prevent paths from escaping the dataset image folder.
        if Path(image_name).name != image_name:
            failed_records.append({
                "record_index": index,
                "image_name": image_name,
                "reason": "Unexpected image path"
            })
            continue

        image_path = IMAGE_DIR / image_name

        if not image_path.is_file():
            missing_images.append(image_name)
            continue

        try:
            # Load the original image.
            with Image.open(image_path) as source:
                source.load()

                # Convert to RGB for consistent model input.
                image = source.convert("RGB")

            original_width, original_height = image.size

            # Apply letterboxing.
            (
                letterboxed_image,
                scale_x,
                scale_y,
                pad_left,
                pad_top
            ) = letterbox_image(
                image,
                TARGET_WIDTH,
                TARGET_HEIGHT
            )

            # Transform annotations for every person.
            for person in record.get("persons", []):
                transform_person(
                    person,
                    scale_x,
                    scale_y,
                    pad_left,
                    pad_top
                )

            # Save the processed image separately.
            output_image_path = OUTPUT_IMAGE_DIR / image_name
            letterboxed_image.save(
                output_image_path,
                format="JPEG",
                quality=95
            )

            # Store transformation information for later use.
            preprocessing_metadata["images"][image_name] = {
                "original_width": original_width,
                "original_height": original_height,
                "target_width": TARGET_WIDTH,
                "target_height": TARGET_HEIGHT,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "pad_left": pad_left,
                "pad_top": pad_top
            }

            processed += 1

        except Exception as error:
            invalid_images.append({
                "image_name": image_name,
                "error": str(error)
            })
            continue

        if index % 100 == 0 or index == total:
            print(
                f"Progress: {index}/{total} annotation records "
                f"| Successfully processed: {processed}"
            )

    # Save transformed annotations separately.
    output_annotation_file = (
        OUTPUT_DIR / "exercise_annotations.json"
    )

    with output_annotation_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            transformed_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Save transformation metadata.
    metadata_file = (
        OUTPUT_DIR / "preprocessing_metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            preprocessing_metadata,
            file,
            indent=2
        )

    # Save an audit report.
    report = {
        "annotation_records": total,
        "successfully_processed": processed,
        "missing_images": missing_images,
        "invalid_images": invalid_images,
        "failed_records": failed_records
    }

    report_file = OUTPUT_DIR / "preprocessing_report.json"

    with report_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    # Print final summary.
    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Annotation records: {total}")
    print(f"Successfully processed: {processed}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Invalid images: {len(invalid_images)}")
    print(f"Failed records: {len(failed_records)}")

    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Processed images: {OUTPUT_IMAGE_DIR}")
    print(f"Transformed annotations: {output_annotation_file}")
    print(f"Transformation metadata: {metadata_file}")
    print(f"Audit report: {report_file}")

    if missing_images or invalid_images or failed_records:
        print(
            "\nWARNING: Some records were not processed. "
            "Review preprocessing_report.json before training."
        )

    print("\nOriginal annotations remain unchanged.")


if __name__ == "__main__":
    main()