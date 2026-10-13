# FitnessVision 🏋️‍♂️🤖
### AI-Powered Fitness Exercise Analysis & Real-Time Posture Correction System

FitnessVision is a computer vision web application designed to help individuals exercise safely without requiring an in-person trainer. Using a webcam, FitnessVision performs real-time body pose estimation, calculates exercise-specific geometric kinematics, detects common form errors, and accurately tracks repetitions using temporal movement state machines.

---

## 1. Project Overview

FitnessVision analyzes human exercise kinematics directly in the web browser and backend using pretrained pose estimation (MediaPipe BlazePose). It evaluates exercise form against sports science principles, offers instant corrective feedback cues, and provides post-workout summary analytics.

### Key Goals
- **Real-Time Guidance**: Provide immediate visual and textual feedback on exercise form.
- **Accurate Repetition Tracking**: Employ temporal phase state machines with hysteresis and jitter suppression to prevent false or duplicate repetition counts.
- **Accessibility**: Run on standard laptop webcams without dedicated hardware, sensors, or GPU requirements.
- **Biomechanical Safety**: Clearly detect hazardous movement deviations such as knee valgus or excessive spinal flexion.

---

## 2. Features

- **Live Webcam Integration**: Browser-based camera stream management with clean lifecycle and permission handling.
- **Pretrained 33-Keypoint Pose Estimation**: Pretrained MediaPipe BlazePose integration (zero model training from scratch).
- **Exercise Selection Studio**: Support for **Bodyweight Squats** and **Bicep Curls** with detailed execution cues and target muscle groups.
- **Interpretable Geometric Analysis**:
  - 3-point joint angle calculations (e.g. knee flexion, elbow flexion, hip angle).
  - Torso inclination relative to vertical axis.
  - Limb symmetry evaluation between left and right sides.
- **Form Error Detection Engine**:
  - *Squat*: Excessive forward torso lean, insufficient depth, knee valgus collapse, lateral asymmetry.
  - *Bicep Curl*: Upper arm drift/flaring, torso momentum sway, incomplete flexion/extension.
- **Multi-Phase Repetition Counting**:
  - Hysteresis-backed state machines (e.g. `STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING`).
  - Strict rejection of incomplete movement cycles and micro-fluctuations.
- **Dynamic HUD & Biomechanical Gauges**: Live angles, movement phase indicators, and prioritized corrective feedback.
- **Session Results & Persistence**: Post-workout summary with repetition totals, error breakdown, and technique advice stored in SQLite via Django ORM.
- **Modern Dark Tech Design**: High-contrast, responsive UI styled with modern CSS variables, glassmorphism, and clear typography.

---

## 3. Technology Stack

- **Backend**: Python 3.11+, Django 4.2 (ORM, views, template engine, REST-style APIs).
- **Computer Vision**: MediaPipe 0.10+ (BlazePose), OpenCV (`opencv-python`), NumPy.
- **Frontend**: HTML5 Semantic Markup, Modern CSS3 Design System, Vanilla JavaScript (ES6+), HTML5 Canvas.
- **Database**: SQLite3 (persisting workout history and error telemetry).

---

## 4. Application Architecture

```
                       BROWSER WEBCAM STREAM
                                 |
                                 v
                       HTML5 CANVAS RENDERER
                                 |
        +------------------------+------------------------+
        | (Primary Mode)                                  | (Fallback Server Mode)
        v                                                 v
MediaPipe JS (WebAssembly)                      Django Frame Endpoint
33 Spatial Keypoints                            (Python cv2 + MediaPipe)
        |                                                 |
        +------------------------+------------------------+
                                 |
                                 v
             REST JSON API (/api/analyze-pose/)
                                 |
                     KEYPOINT PREPROCESSING
           (Extract required landmarks & check visibility)
                                 |
                    GEOMETRIC FEATURE EXTRACTION
        (Vector dot-product angles, torso tilt, symmetry)
                                 |
                                 +-------------------------+
                                 |                         |
                                 v                         v
                       POSTURE ERROR RULES       TEMPORAL REP COUNTER
                    (Valgus, drift, lean)      (State machine transitions)
                                 |                         |
                                 +------------+------------+
                                              |
                                              v
                                     FEEDBACK SYNTHESIZER
                                 (Prioritized coaching cues)
                                              |
                                              v
                                       TELEMETRY HUD
                                              |
                                              v
                                       WORKOUT SUMMARY
```

### Architectural Decisions: Browser & Server Division
- **Webcam Ownership**: The browser owns the video stream via `navigator.mediaDevices.getUserMedia`.
- **Hybrid Processing**:
  - The browser runs client-side MediaPipe Pose to provide immediate 30 FPS skeleton visualization on the `<canvas>` overlay.
  - The structured coordinates are sent to Django's `/api/analyze-pose/` endpoint where the authoritative Python geometric engine, isolated session state machine, and error catalog evaluate the repetition.
  - A fallback endpoint (`/api/analyze-frame/`) provides full server-side OpenCV and Python MediaPipe processing if WebAssembly or external CDNs are unavailable.

---

## 5. Project Directory Structure

```
FitnessVision/
|-- manage.py                    # Django management utility
|-- requirements.txt             # Project Python dependencies
|-- README.md                    # System documentation
|-- .gitignore                   # Git exclusions
|
|-- config/                      # Project root configuration
|   |-- __init__.py
|   |-- settings.py              # Django settings & static/template paths
|   |-- urls.py                  # Root URL configuration
|   |-- asgi.py
|   |-- wsgi.py
|
|-- main_app/                    # Core application app
|   |-- __init__.py
|   |-- models.py                # WorkoutSession ORM model
|   |-- views.py                 # Views and API endpoints
|   |-- urls.py                  # App URL routes
|   |-- session_manager.py       # Session-isolated state & counters
|   |-- admin.py                 # Django admin registration
|   |-- apps.py
|
|-- pose_estimation/             # Pose detector and keypoint abstraction
|   |-- __init__.py
|   |-- detector.py              # Server-side MediaPipe Pose wrapper
|   |-- keypoints.py             # 33 landmarks mapping, validation, sanitization
|   |-- visualization.py         # OpenCV skeleton rendering utilities
|
|-- posture_analysis/            # Biomechanical rule engine
|   |-- __init__.py
|   |-- geometry.py              # Angle, distance, tilt, symmetry math
|   |-- squat_analysis.py        # Squat mechanics & error checks
|   |-- curl_analysis.py         # Bicep curl mechanics & error checks
|   |-- error_detection.py       # Structured error catalog & codes
|   |-- feedback.py              # Priority-based coaching feedback generator
|
|-- exercises/                   # Exercise metadata and repetition state machines
|   |-- __init__.py
|   |-- registry.py              # Exercise catalog, muscles, instructions
|   |-- repetition_counter.py    # Temporal state machines (Squat & Curl)
|
|-- templates/                   # Django HTML templates
|   |-- base.html                # Navigation, footer, and styling shell
|   |-- home.html                # Landing page
|   |-- exercise_selection.html  # Exercise cards library
|   |-- workout.html             # Real-time workout video & HUD studio
|   |-- results.html             # Session completion summary
|
|-- static/                      # Static assets
|   |-- css/
|   |   |-- style.css            # Dark aesthetic fitness design system
|   |-- js/
|   |   |-- camera.js            # Webcam lifecycle & permission manager
|   |   |-- workout.js           # Client controller, pose loop & HUD updater
|
|-- tests/                       # Automated test suite
|   |-- __init__.py
|   |-- test_geometry.py         # Angle & distance edge case tests
|   |-- test_repetition_counter.py# State machine transition & jitter tests
|   |-- test_posture_analysis.py # Form rules & synthetic pose tests
|   |-- test_django_views.py     # HTTP views & API route tests
|
|-- Documentation/               # Additional technical references
```

---

## 6. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- A functional webcam or external USB camera.
- A modern web browser (Google Chrome, Microsoft Edge, Firefox, or Safari).

### Step 1: Create and Activate Virtual Environment (Optional but Recommended)
```bash
# Windows (PowerShell / Command Prompt)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Database Migrations
```bash
python manage.py migrate
```

### Step 4: Run Automated Tests
Verify that all unit and integration tests pass:
```bash
python manage.py test tests
```

### Step 5: Start the Development Server
```bash
python manage.py runserver 127.0.0.1:8000
```

### Step 6: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```

---

## 7. Posture Analysis & Kinematic Rules

### Joint Angle Calculation
The angle $\theta$ between points $A$, $B$ (vertex joint), and $C$ is computed via vector dot product:
$$\mathbf{u} = A - B, \quad \mathbf{v} = C - B$$
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
$$\theta = \arccos(\text{clamp}(\cos(\theta), -1.0, 1.0)) \times \frac{180}{\pi}$$

Robust edge case handling:
- Zero-length limb vectors are detected to avoid division by zero.
- Numerical precision rounding is clamped to $[-1.0, 1.0]$.
- Missing or low-visibility keypoints return explicit `None` instead of arbitrary zeros.

### Squat Form Rules
- **Depth**: Reaching knee flexion $\le 100^\circ$ signifies acceptable depth. Ascending before reaching $\le 115^\circ$ flags `INSUFFICIENT_SQUAT_DEPTH`.
- **Torso Lean**: Inclination of shoulder-to-hip line relative to vertical $> 48^\circ$ flags `EXCESSIVE_FORWARD_LEAN`.
- **Knee Valgus**: Narrowing of inter-knee distance relative to inter-ankle distance ($< 65\%$) flags `KNEE_VALGUS`.

### Bicep Curl Form Rules
- **Elbow Stability**: Upper arm inclination $> 40^\circ$ relative to torso flags `EXCESSIVE_ELBOW_DRIFT`.
- **Momentum / Sway**: Torso inclination change $> 22^\circ$ flags `TORSO_SWAY`.
- **Range of Motion**: Flexion to $\le 65^\circ$ at top and return to $\ge 145^\circ$ at bottom.

---

## 8. Repetition Counting Logic

Repetition tracking is implemented as a deterministic finite-state machine (FSM) rather than an instant threshold crossing:

### Squat State Transitions
1. `START`: User moves into camera view.
2. `STANDING`: User stands upright (knee angle $\ge 150^\circ$).
3. `DESCENDING`: Knee angle drops below $140^\circ$.
4. `BOTTOM`: Knee angle reaches $\le 100^\circ$. If the user reverses upward early ($> 115^\circ$), an incomplete repetition is flagged and rep count is not incremented.
5. `ASCENDING`: Knee angle increases past $115^\circ$.
6. `STANDING (COMPLETED)`: Knee angle returns to $\ge 150^\circ$. The repetition count increments by exactly 1.

A 4-frame rolling moving average suppresses sensor noise and micro-jitter.

---

## 9. Current Limitations & Known Considerations

- **2D Perspective Ambiguity**: 2D monocular webcams cannot measure out-of-plane depth with the precision of multi-camera optical setups. Certain checks (e.g. knee valgus) require a direct frontal perspective, while torso lean is best observed from a slight 45-degree or sagittal perspective.
- **Occlusion**: Loose clothing or objects obstructing the camera view may reduce keypoint visibility. The system flags this with a visual warning prompt.
- **Lighting**: Low-contrast or poorly lit environments can reduce BlazePose landmark confidence.

---

## 10. Medical Disclaimer

FitnessVision is designed exclusively for educational, posture awareness, and fitness guidance purposes. It is **not** a certified medical device, orthopedic diagnosis instrument, or substitute for professional medical care, physiotherapy, or personal coaching. Users should immediately cease exercising if experiencing pain, strain, or discomfort.
