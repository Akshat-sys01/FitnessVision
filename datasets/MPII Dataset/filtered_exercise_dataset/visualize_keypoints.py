import json
import random
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "letterboxed_512" / "images"
ANNOTATION_FILE = BASE_DIR / "letterboxed_512" / "exercise_annotations.json"

OUTPUT_DIR = BASE_DIR / "keypoint_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 10

# MPII skeleton connections (keypoint IDs)
SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]

# Load annotations
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]

# Select random annotations
selected = random.sample(
    annotations,
    min(NUM_IMAGES, len(annotations))
)

for annotation in selected:

    image_name = annotation["image_name"]
    image_path = IMAGE_DIR / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_name}")
        continue

    for person in annotation.get("persons", []):

        points = {
            point["id"]: point
            for point in person.get("points", [])
        }

        # Draw skeleton connections
        for start_id, end_id in SKELETON:

            if start_id not in points or end_id not in points:
                continue

            p1 = points[start_id]
            p2 = points[end_id]

            if not p1["is_visible"] or not p2["is_visible"]:
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

        # Draw keypoints
        for point in points.values():

            if not point["is_visible"]:
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

    output_path = OUTPUT_DIR / image_name
    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")

print("\nVisualization completed!")
print(f"Output folder: {OUTPUT_DIR}")