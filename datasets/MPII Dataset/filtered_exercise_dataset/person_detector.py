from pathlib import Path
import cv2
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "yolo11n.pt"

INPUT_IMAGE = Path(
    "images/004976720.jpg"
)

OUTPUT_DIR = Path("phase1_outputs")
CROPS_DIR = OUTPUT_DIR / "person_crops"

CONFIDENCE = 0.25
IMAGE_SIZE = 640

# ------------------------------------------------


def detect_and_crop(image_path):

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Read input image
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    # Create output directories
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Run person detection (COCO class 0 = person)
    results = model.predict(
        source=image,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    annotated_image = image.copy()

    person_count = 0

    for box in result.boxes:

        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        confidence = float(box.conf[0])

        # Convert coordinates to integers
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        # Ignore invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop detected person
        person_crop = image[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        person_count += 1

        # Save individual person crop
        crop_path = CROPS_DIR / f"person_{person_count}.jpg"

        cv2.imwrite(str(crop_path), person_crop)

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display confidence
        label = f"Person {person_count}: {confidence:.2f}"

        cv2.putText(
            annotated_image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(f"\nPerson {person_count}")
        print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Crop saved: {crop_path}")

    # Save annotated image
    annotated_path = OUTPUT_DIR / "detected_people.jpg"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(annotated_path), annotated_image)

    print("\n========== DETECTION SUMMARY ==========")
    print("Input image:", image_path)
    print("Image dimensions:", width, "x", height)
    print("People detected:", person_count)
    print("Annotated image:", annotated_path)
    print("Person crops directory:", CROPS_DIR)


if __name__ == "__main__":
    detect_and_crop(INPUT_IMAGE)