import json
from pathlib import Path

folder = Path("letterboxed_512")
annotation_file = folder / "exercise_annotations.json"
backup_file = folder / "exercise_annotations.backup.json"
temp_file = folder / "exercise_annotations.temp.json"

# Load existing annotations
with open(annotation_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create backup
if not backup_file.exists():
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

removed_persons = 0

# Remove only persons having at least one out-of-bounds keypoint
for annotation in data["annotations"]:
    valid_persons = []

    for person in annotation.get("persons", []):
        invalid = any(
            not (0 <= point["x"] < 512 and 0 <= point["y"] < 512)
            for point in person.get("points", [])
        )

        if invalid:
            removed_persons += 1
            print(f"Removed person from: {annotation['image_name']}")
        else:
            valid_persons.append(person)

    annotation["persons"] = valid_persons

# Write cleaned annotations to a temporary file
with open(temp_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

# Replace the existing annotation file
temp_file.replace(annotation_file)

total_persons = sum(
    len(a.get("persons", [])) for a in data["annotations"]
)

print("\nCleaning completed!")
print("Removed person annotations:", removed_persons)
print("Remaining annotated persons:", total_persons)
print("Total images:", len(data["annotations"]))
print("Backup:", backup_file)
