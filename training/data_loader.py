import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Adjust this path if your dataset directory differs.
DATASET_DIR = Path("./datasets/MPII Dataset/filtered_exercise_dataset/person_training_dataset")

IMAGE_DIR = DATASET_DIR / "images"
HEATMAP_DIR = DATASET_DIR / "heatmaps"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"
SPLIT_FILE = DATASET_DIR / "dataset_split.json"

INPUT_HEIGHT, INPUT_WIDTH = 256, 256
HEATMAP_HEIGHT, HEATMAP_WIDTH = 64, 64
NUM_JOINTS = 16
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

def _read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _records_from_json(data):
    return data.get("annotations", data) if isinstance(data, dict) else data

def _get_person_id(record):
    value = record.get("person_id", record.get("id"))
    return None if value is None else int(value)

def _load_sample(person_id, annotation_by_id):
    record = annotation_by_id[int(person_id)]
    image_name = record.get("image_name") or record.get("image")
    if not image_name:
        image_name = f"person_{int(person_id):06d}.jpg"

    image_path = IMAGE_DIR / image_name
    heatmap_path = HEATMAP_DIR / (Path(image_name).stem + ".npz")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")

    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [INPUT_HEIGHT, INPUT_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    with np.load(heatmap_path) as data:
        heatmaps = data["heatmaps"].astype(np.float32)
        visibility = data["visibility"].astype(np.int8)

    if heatmaps.shape == (NUM_JOINTS, 64, 64):
        heatmaps = np.transpose(heatmaps, (1, 2, 0))
    if heatmaps.shape != (64, 64, NUM_JOINTS):
        raise ValueError(f"Unexpected heatmap shape {heatmaps.shape}: {heatmap_path}")

    # Use visible joints only: visibility 1; invisible 0 and missing -1 excluded.
    joint_mask = (visibility == 1).astype(np.float32)
    return image.numpy(), heatmaps, joint_mask

def _generator(ids, annotation_by_id):
    for person_id in ids:
        yield _load_sample(person_id, annotation_by_id)

def _make_dataset(ids, annotation_by_id, training):
    ds = tf.data.Dataset.from_generator(
        lambda: _generator(ids, annotation_by_id),
        output_signature=(
            tf.TensorSpec((256, 256, 3), tf.float32),
            tf.TensorSpec((64, 64, 16), tf.float32),
            tf.TensorSpec((16,), tf.float32)
        )
    )
    if training:
        ds = ds.shuffle(min(len(ids), 1000), reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

_annotations = _read_json(ANNOTATIONS_FILE)
_records = _records_from_json(_annotations)
annotation_by_id = {}
for _record in _records:
    _pid = _get_person_id(_record)
    if _pid is not None:
        annotation_by_id[_pid] = _record

splits = _read_json(SPLIT_FILE)
train_ids = splits["train"]
val_ids = splits["validation"]

train_dataset = _make_dataset(train_ids, annotation_by_id, True)
val_dataset = _make_dataset(val_ids, annotation_by_id, False)

print(f"Training samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Batch size: {BATCH_SIZE}")
