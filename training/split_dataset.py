import json
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("../datasets/MPII Dataset/filtered_exercise_dataset/person_training_dataset")
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"
OUTPUT_FILE = DATASET_DIR / "dataset_split.json"
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

def main():
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing: {ANNOTATIONS_FILE.resolve()}")
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("annotations", data) if isinstance(data, dict) else data
    ids = []
    for record in records:
        person_id = record.get("person_id", record.get("id"))
        if person_id is not None:
            ids.append(int(person_id))
    ids = sorted(set(ids))
    train_ids, val_ids = train_test_split(
        ids, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"train": list(map(int, train_ids)),
                   "validation": list(map(int, val_ids))}, f, indent=4)
    print("Total:", len(ids), "| Train:", len(train_ids), "| Validation:", len(val_ids))
    print("Saved:", OUTPUT_FILE.resolve())

if __name__ == "__main__":
    main()
