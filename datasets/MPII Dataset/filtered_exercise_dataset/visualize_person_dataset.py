import json
import random
from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("person_training_dataset")
IMAGE_DIR = DATASET_DIR / "images"
ANNOTATION_FILE = DATASET_DIR / "annotations.json"

OUTPUT_DIR = DATASET_DIR / "visualizations"

NUM_SAMPLES = 10

SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]

    samples = random.sample(
        annotations,
        min(NUM_SAMPLES, len(annotations))
    )

    for index, annotation in enumerate(samples, start=1):

        image_path = IMAGE_DIR / annotation["image_name"]

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path)
            continue

        points = {
            point["id"]: point
            for point in annotation["points"]
        }

        # Draw skeleton connections.
        for start_id, end_id in SKELETON:

            if start_id not in points or end_id not in points:
                continue

            p1 = points[start_id]
            p2 = points[end_id]

            # Draw only connections with visible endpoints.
            if not (
                p1["is_visible"] == 1
                and p2["is_visible"] == 1
            ):
                continue

            x1, y1 = int(p1["x"]), int(p1["y"])
            x2, y2 = int(p2["x"]), int(p2["y"])

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # Draw keypoints.
        for point in annotation["points"]:

            if point["is_visible"] != 1:
                continue

            x = int(point["x"])
            y = int(point["y"])

            cv2.circle(
                image,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

        output_path = OUTPUT_DIR / f"sample_{index:02d}.jpg"

        cv2.imwrite(str(output_path), image)

        print("Saved:", output_path)

    print("\nVisualization completed!")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()