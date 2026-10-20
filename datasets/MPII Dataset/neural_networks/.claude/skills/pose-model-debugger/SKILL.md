---
name: pose-model-debugger
description: Diagnose and debug CNN-based human pose estimation models, especially collapsed keypoint predictions, coordinate mismatches, annotation problems, masked losses, and incorrect PCK evaluation. Use when investigating training or inference failures.
---

# Pose Model Debugger

## Objective

Diagnose problems in the existing human pose estimation pipeline
without unnecessarily rewriting the project.

## Workflow

1. Inspect the existing model, dataset, loss, training, evaluation,
   and inference implementations.

2. Trace a sample from the original image and annotation through
   preprocessing, model prediction, evaluation, and visualization.

3. Check coordinate conventions, image resizing, cropping,
   keypoint ordering, annotation masks, and output decoding.

4. Verify that the loss function and PCK implementation are
   consistent with the coordinate representation.

5. Run inexpensive diagnostic tests before full retraining.

6. Make minimal, targeted changes only after identifying a
   problem.

7. Preserve existing checkpoints and datasets.

8. Test every change and report actual results.

## Constraints

- Do not rebuild the project from scratch.
- Do not assume the model architecture is the root cause.
- Do not change multiple components simultaneously without
  justification.
- Do not delete existing checkpoints or datasets.
- Never fabricate training or evaluation results.
- Explain technical findings in simple language.