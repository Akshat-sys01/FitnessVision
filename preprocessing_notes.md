### 1. Dataset Acquisition
Downloaded the MPII Human Pose Dataset containing 24,984 images and MATLAB annotations.

**Why:** Provides 16-joint human pose annotations for training our custom pose estimation model.

### 2. Annotation Structure Inspection
Inspected the `.mat` file using MATLAB to understand image-level and person-level annotations, including keypoints, visibility, scale and bounding boxes.

**Why:** To identify the information required for pose estimation.

### 3. MAT to JSON Conversion
Converted the original MATLAB annotations into a simplified JSON structure, retaining image filenames, person information, keypoints, joint IDs and visibility flags. Removed unnecessary activity and video metadata.

**Why:** JSON is easier to process and integrate into our Python training pipeline.

### 4. Annotation Filtering
Retained people with at least one usable annotated keypoint. Excluded people without keypoint annotations.

**Why:** Supervised pose estimation requires ground-truth keypoint coordinates.

### 5. Initial Conversion Results
- Original images: 24,984
- Exported image records: 17,408
- People with keypoints: 28,883
- People with all 16 keypoints: 19,919
- People with partial keypoints: 8,964
- People without keypoints: 12,009
- Missing image references: 3

### 6. JSON Validation
Audited the converted JSON and verified all 16 joint IDs, keypoint coordinates and visibility annotations.

**Why:** To ensure annotation integrity before further preprocessing.

### 7. MPII Joint Mapping Verification

Verified all 16 original MPII joint IDs against the existing FitnessVision joint order.

**Result:** Both use identical joint ordering (IDs 0–15).

**Why:** Ensures compatibility with the existing model architecture and avoids unnecessary keypoint remapping.


### 8. Original Dataset Partition Analysis

Inspected the original MPII `img_train` field.

- Training partition (`1`): 18,079 images.
- Test partition (`0`): 6,908 images.

**Why:** To preserve the original dataset partition and avoid using test images as supervised training samples.

### 9. Person-Centred Cropping

Extracted individual person crops using annotated keypoints and MPII head bounding boxes. Added 20% margin and applied aspect-ratio-preserving resizing with letterboxing to 256×256.

Keypoint coordinates were transformed into the corresponding cropped-image coordinate system.

**Why:** To generate standardized, individual-person training images while preserving keypoint alignment.


### 9. Person-Centred Cropping

Generated individual person crops using MPII keypoints and head bounding boxes. Added 20% margin and applied aspect-ratio-preserving resizing with letterboxing to 256×256. Transformed keypoint coordinates accordingly.

**Result:** 28,883 person crops generated.

**Why:** To standardize individual human images and reduce background interference during pose estimation.

### 10. Person Crop Validation

Validated all generated person crops against their annotations.

**Result:** 28,883 valid images, all sized 256×256, with zero missing or corrupted images.

**Why:** To ensure image integrity and consistent dimensions before generating training heatmaps.

### 11. Gaussian Heatmap Generation

Generated 16-channel Gaussian heatmaps from transformed MPII keypoints. Heatmaps have dimensions 64×64 with Gaussian sigma 2.0. Only visible joints receive supervised heatmaps; corresponding visibility masks are generated.

**Why:** Converts keypoint coordinates into spatial training targets for the CNN.


### 12. Full 16-Joint Dataset Separation

Separated person-centred samples containing annotations for all 16 MPII joints.

- Original samples: 28,883
- Selected samples: 19,919
- Excluded samples: 8,964
- Missing images/heatmaps: 0

Invisible joints were retained, with their corresponding heatmaps masked during training.

**Why:** Creates a consistent 16-joint dataset while preserving partially visible body poses for training.


### 13. 16-Joint Dataset Quality Validation

Validated all 19,919 samples for image integrity, heatmap dimensions, joint masks, annotation consistency and duplicate images.

**Results:**
- Valid images: 19,919
- Valid heatmaps: 19,919
- Invalid images: 0
- Invalid heatmaps: 0
- Invalid annotations: 0
- Duplicate images: 0

**Status:** PASS

**Why:** Ensures the dataset contains structurally valid images, annotations and heatmaps before model training.