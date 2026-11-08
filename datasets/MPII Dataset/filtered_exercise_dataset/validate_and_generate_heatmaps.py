import json
from pathlib import Path

import numpy as np
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path("person_training_dataset")

IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations.json"

OUTPUT_DIR = DATASET_DIR / "heatmaps"

INPUT_SIZE = 256
HEATMAP_SIZE = 64

NUM_KEYPOINTS = 16
SIGMA = 2.0

# ==========================================


def generate_gaussian_heatmap(x, y):
    """
    Generate one Gaussian heatmap for a keypoint.

    Coordinates are converted from 256x256
    image space to 64x64 heatmap space.
    """

    heatmap = np.zeros(
        (HEATMAP_SIZE, HEATMAP_SIZE),
        dtype=np.float32
    )

    x = x / (INPUT_SIZE / HEATMAP_SIZE)
    y = y / (INPUT_SIZE / HEATMAP_SIZE)

    radius = int(3 * SIGMA)

    center_x = int(round(x))
    center_y = int(round(y))

    x_min = max(0, center_x - radius)
    x_max = min(HEATMAP_SIZE, center_x + radius + 1)

    y_min = max(0, center_y - radius)
    y_max = min(HEATMAP_SIZE, center_y + radius + 1)

    if x_min >= x_max or y_min >= y_max:
        return heatmap

    grid_x = np.arange(x_min, x_max)
    grid_y = np.arange(y_min, y_max)

    xx, yy = np.meshgrid(grid_x, grid_y)

    gaussian = np.exp(
        -(
            (xx - x) ** 2 +
            (yy - y) ** 2
        ) / (2 * SIGMA ** 2)
    )

    heatmap[y_min:y_max, x_min:x_max] = gaussian

    return heatmap


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        dataset = json.load(f)

    annotations = dataset["annotations"]

    # Validation counters
    total_images = 0
    total_persons = 0
    total_keypoints = 0

    missing_images = 0
    invalid_image_sizes = 0
    invalid_ids = 0
    duplicate_ids = 0
    out_of_bounds = 0
    invalid_visibility = 0
    missing_coordinates = 0

    invisible_keypoints = 0
    missing_keypoints = 0

    validation_errors = []

    # ======================================
    # VALIDATION
    # ======================================

    print("\nStarting dataset validation...")

    for annotation in annotations:

        total_images += 1

        image_name = annotation["image_name"]

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():

            missing_images += 1

            validation_errors.append(
                f"Missing image: {image_name}"
            )

            continue

        with Image.open(image_path) as image:

            if image.size != (INPUT_SIZE, INPUT_SIZE):

                invalid_image_sizes += 1

                validation_errors.append(
                    f"Invalid image size: {image_name}"
                )

        for person in annotation["annotations"] if False else []:
            pass

        # Each generated image represents one person.
        total_persons += 1

        points = annotation["points"]

        point_ids = [p["id"] for p in points]

        if len(point_ids) != len(set(point_ids)):

            duplicate_ids += 1

            validation_errors.append(
                f"Duplicate keypoint IDs: {image_name}"
            )

        valid_ids = set()

        for point in points:

            total_keypoints += 1

            point_id = point["id"]

            if (
                not isinstance(point_id, int)
                or point_id < 0
                or point_id >= NUM_KEYPOINTS
            ):

                invalid_ids += 1

                validation_errors.append(
                    f"Invalid keypoint ID: {image_name}"
                )

                continue

            valid_ids.add(point_id)

            x = point.get("x")
            y = point.get("y")

            if x is None or y is None:

                missing_coordinates += 1

                validation_errors.append(
                    f"Missing coordinates: {image_name}, ID {point_id}"
                )

                continue

            if not np.isfinite(x) or not np.isfinite(y):

                missing_coordinates += 1

                validation_errors.append(
                    f"Non-finite coordinates: {image_name}, ID {point_id}"
                )

                continue

            if not (0 <= x < INPUT_SIZE and 0 <= y < INPUT_SIZE):

                out_of_bounds += 1

                validation_errors.append(
                    f"Out-of-bounds keypoint: {image_name}, ID {point_id}"
                )

            visibility = point.get("is_visible")

            if visibility not in (0, 1):

                invalid_visibility += 1

                validation_errors.append(
                    f"Invalid visibility: {image_name}, ID {point_id}"
                )

            elif visibility == 0:

                invisible_keypoints += 1

        missing_keypoints += NUM_KEYPOINTS - len(valid_ids)

    # ======================================
    # VALIDATION REPORT
    # ======================================

    print("\n========== VALIDATION REPORT ==========")

    print("Images:", total_images)
    print("Persons:", total_persons)
    print("Annotated keypoints:", total_keypoints)

    print("\nMissing images:", missing_images)
    print("Invalid image sizes:", invalid_image_sizes)
    print("Invalid keypoint IDs:", invalid_ids)
    print("Duplicate keypoint IDs:", duplicate_ids)
    print("Out-of-bounds coordinates:", out_of_bounds)
    print("Missing coordinates:", missing_coordinates)
    print("Invalid visibility:", invalid_visibility)

    print("Invisible annotated keypoints:", invisible_keypoints)
    print("Missing keypoints:", missing_keypoints)

    # Stop if data is invalid.
    if validation_errors:

        print("\nVALIDATION FAILED!")

        print("\nFirst 20 errors:")

        for error in validation_errors[:20]:
            print("-", error)

        print("\nHeatmap generation cancelled.")

        return

    print("\nVALIDATION PASSED!")

    # ======================================
    # HEATMAP GENERATION
    # ======================================

    print("\nGenerating Gaussian heatmaps...")

    generated = 0

    for annotation in annotations:

        image_name = annotation["image_name"]

        heatmaps = np.zeros(
            (
                NUM_KEYPOINTS,
                HEATMAP_SIZE,
                HEATMAP_SIZE
            ),
            dtype=np.float16
        )

        joint_mask = np.zeros(
            NUM_KEYPOINTS,
            dtype=np.float32
        )

        visibility = np.full(
            NUM_KEYPOINTS,
            -1,
            dtype=np.int8
        )

        for point in annotation["points"]:

            point_id = point["id"]

            x = float(point["x"])
            y = float(point["y"])

            heatmaps[point_id] = generate_gaussian_heatmap(
                x,
                y
            )

            # Indicates that this joint has an annotation.
            joint_mask[point_id] = 1

            # Preserve visibility separately.
            visibility[point_id] = point["is_visible"]

        output_name = Path(image_name).stem + ".npz"

        output_path = OUTPUT_DIR / output_name

        np.savez_compressed(
            output_path,
            heatmaps=heatmaps,
            joint_mask=joint_mask,
            visibility=visibility
        )

        generated += 1

        if generated % 500 == 0:

            print(
                f"Generated {generated}/{total_persons} heatmaps"
            )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print("\n========== HEATMAP SUMMARY ==========")

    print("Generated samples:", generated)

    print(
        "Heatmap shape:",
        (NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE)
    )

    print("Gaussian sigma:", SIGMA)

    print("Output directory:", OUTPUT_DIR)

    print("\nDATASET READY FOR KERAS TRAINING!")


if __name__ == "__main__":
    main()