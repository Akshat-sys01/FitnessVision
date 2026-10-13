# FitnessVision System Architecture

## 1. Overview
FitnessVision separates computer vision capture, spatial keypoint extraction, geometric kinematics, temporal state-machine tracking, and web presentation into clean, decoupled layers.

```
+-------------------------------------------------------------+
|                     PRESENTATION LAYER                      |
|  - Django Templates (base, home, exercise_selection, etc.)  |
|  - Modern Dark CSS3 Design System with CSS Variables        |
|  - JavaScript Workout Controller (workout.js, camera.js)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   COMPUTER VISION LAYER                     |
|  - Camera Stream Capture (getUserMedia)                     |
|  - BlazePose 33 Body Keypoint Estimator                     |
|  - Client WebAssembly Canvas Overlay / Server OpenCV        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                KEYPOINT PREPROCESSING LAYER                 |
|  - Keypoint Sanitizer & Parser (parse_keypoints)            |
|  - Visibility & Confidence Checker                          |
|  - Exercise-Specific Landmark Selection                     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 KINEMATICS & GEOMETRY LAYER                 |
|  - 3-Point Angle Calculation (cosine dot product + clamp)   |
|  - Euclidean & Normalized Distances                         |
|  - Torso Inclination Vector & Bilateral Symmetry            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               EXERCISE ANALYSIS & RULES LAYER               |
|  - SquatAnalyzer (depth, torso lean, knee valgus)           |
|  - BicepCurlAnalyzer (drift, sway, extension/flexion)       |
|  - Error Catalog & Priority Feedback Synthesizer            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 TEMPORAL STATE MACHINE LAYER                |
|  - SquatRepetitionCounter (Standing -> Descend -> Bottom)   |
|  - BicepCurlRepetitionCounter (Extended -> Flex -> Lower)   |
|  - Anti-Jitter Moving Average Filter                        |
|  - Incomplete Repetition Rejection & Lost Frame Handler     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     PERSISTENCE LAYER                       |
|  - Session-Isolated State Manager (ActiveWorkoutState)      |
|  - Django ORM Models (WorkoutSession in SQLite)             |
+-------------------------------------------------------------+
```

## 2. Component Directory Responsibilities
- `pose_estimation/`: Encapsulates model interactions with MediaPipe Pose so the rest of the application never depends directly on raw model formats.
- `posture_analysis/`: Pure, stateless mathematical and geometric algorithms. Fully unit-tested with synthetic landmark coordinates.
- `exercises/`: Encapsulates exercise definitions, target muscle groups, setup guidance, and temporal repetition state machines.
- `main_app/`: Standard Django web routes, API endpoints, request validation, and database operations.
