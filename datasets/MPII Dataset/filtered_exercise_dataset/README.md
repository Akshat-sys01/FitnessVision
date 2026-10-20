# Filtered Exercise Dataset

This folder contains a **fitness‑focused subset** of the original MPII Human Pose dataset. The subset was created by filtering images that depict fitness‑related activities. Below is an overview of the files included and their formats.

## File Overview

| File | Description |
|------|-------------|
| `exercise_annotations.json` | JSON file with pose annotations for each filtered image. The structure mirrors the original MPII `.mat` annotations but is converted to JSON for easier consumption. Each entry contains:
- `image_name`: filename of the image (e.g., `000003072.jpg`).
- `annolist`: list of person annotations, each with:
  - `annorect`: bounding box (`x1`, `y1`, `x2`, `y2`).
  - `objpos`: approximate center of the person.
  - `annopoints`: list of joint points (`id`, `x`, `y`, `is_visible`).
- `act`: activity label and name.
| `exercise_metadata.csv` | CSV file providing a compact summary for each image. Columns include:
- `image_name`
- `activity_id`
- `activity_name`
- `category_name`
- `train_split` (1 = training, 0 = test)
| `filter_summary.txt` | Plain‑text summary of the filtering criteria, number of images kept, and any statistics (e.g., class distribution).
| `mpii_exercise_subset.mat` | MATLAB‑format file (`.mat`) that stores the same information as `exercise_annotations.json` but in the original MATLAB `RELEASE` structure used by the MPII dataset. It can be loaded in MATLAB/Octave with:
```matlab
load('mpii_exercise_subset.mat');
% The variable RELEASE contains the annotations.
```
| `sample_calisthenics.jpg`, `sample_resistance.jpg` | Example images illustrating the type of activities included in the subset.
| `images/` (sub‑folder) | Directory containing **all filtered images** (≈ **X** images). The filenames match those referenced in the annotation files.

## How to Load the Data

### In Python
```python
import json
import pandas as pd

# Load annotations
with open('exercise_annotations.json') as f:
    ann = json.load(f)

# Load metadata
meta = pd.read_csv('exercise_metadata.csv')
```

### In MATLAB / Octave
```matlab
load('mpii_exercise_subset.mat'); % loads variable RELEASE
% Example: get first image name
firstImg = RELEASE.annolist(1).image.name;
```

## Annotation Structure (Reference)
The original MPII annotation structure is documented in `annotations/mpii_human_pose_v1_u12_2/README.md`. The key fields are:
- `annolist(imgidx).image.name`
- `annolist(imgidx).annorect(ridx).x1, y1, x2, y2`
- `annolist(imgidx).annorect(ridx).objpos`
- `annolist(imgidx).annorect(ridx).annopoints.point(id).x, y, is_visible`
- `annolist(imgidx).act`, `act_name`, `cat_name`

The filtered dataset retains the same hierarchy, merely limited to images that satisfy the fitness criterion.

## Usage Tips
- The JSON file is convenient for most modern ML pipelines.
- If you need the original MATLAB format (e.g., for legacy code), use the `.mat` file.
- The `train_split` column in `exercise_metadata.csv` indicates whether an image belongs to the training set (`1`) or test set (`0`).

---
*Generated on $(date)*
