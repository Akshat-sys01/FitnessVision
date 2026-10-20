from pathlib import Path
from PIL import Image

# Replace this with your dataset folder path
dataset_dir = Path("images/")

dimensions = {}
total_images = 0
unreadable_images = []

# Search for images in the folder and its subfolders
for image_path in dataset_dir.rglob("*"):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            size = (width, height)

            dimensions[size] = dimensions.get(size, 0) + 1
            total_images += 1

    except (OSError, ValueError):
        unreadable_images.append(str(image_path))

# Display results
print("Total images:", total_images)
print("\nImage dimensions found:")

for size, count in sorted(dimensions.items()):
    print(f"{size[0]} x {size[1]} pixels: {count} images")

if len(dimensions) == 1 and total_images > 0:
    print("\nRESULT: All readable images have the same dimensions.")
elif len(dimensions) > 1:
    print("\nRESULT: Images have different dimensions.")
else:
    print("\nRESULT: No readable images found.")

if unreadable_images:
    print(f"\nUnreadable images: {len(unreadable_images)}")