/**
 * FitnessVision - Workout Session Coordinator
 * Handles camera capture, pose estimation (MediaPipe client or Python server fallback),
 * canvas skeleton rendering, API communication, and HUD telemetry updates.
 */

class WorkoutSessionController {
  constructor(config) {
    this.exerciseSlug = config.exerciseSlug || 'squat';
    this.sessionToken = config.sessionToken || 'default_session';
    this.useServerDetector = config.useServerDetector || false;

    // DOM Elements
    this.videoEl = document.getElementById('webcam-video');
    this.canvasEl = document.getElementById('pose-canvas');
    this.ctx = this.canvasEl ? this.canvasEl.getContext('2d') : null;
    this.placeholderEl = document.getElementById('viewport-placeholder');
    this.cameraStatusText = document.getElementById('camera-status-text');
    this.statusDot = document.getElementById('status-dot');

    // Telemetry Elements
    this.repCountEl = document.getElementById('rep-count-val');
    this.phaseBadgeEl = document.getElementById('phase-badge');
    this.statusBannerEl = document.getElementById('status-banner');
    this.statusTitleEl = document.getElementById('status-title');
    this.feedbackTextEl = document.getElementById('feedback-text');
    this.primaryAngleValEl = document.getElementById('primary-angle-val');
    this.primaryAngleLabelEl = document.getElementById('primary-angle-label');
    this.torsoIncValEl = document.getElementById('torso-inc-val');
    this.symmetryValEl = document.getElementById('symmetry-val');
    this.timerValEl = document.getElementById('session-timer-val');

    // Controls
    this.btnStart = document.getElementById('btn-start-workout');
    this.btnPause = document.getElementById('btn-pause-workout');
    this.btnResume = document.getElementById('btn-resume-workout');
    this.btnEnd = document.getElementById('btn-end-workout');

    // State
    this.isRunning = false;
    this.isPaused = false;
    this.isProcessingFrame = false;
    this.lastApiCallTime = 0;
    this.apiThrottleMs = 120; // ~8 requests/sec to server analysis API
    this.startTime = null;
    this.timerInterval = null;
    this.elapsedSeconds = 0;
    this.latestRepCount = 0;

    // Keypoint connections mapping for client rendering
    this.connections = [
      // Torso
      [11, 12], [11, 23], [12, 24], [23, 24],
      // Left Arm
      [11, 13], [13, 15],
      // Right Arm
      [12, 14], [14, 16],
      // Left Leg
      [23, 25], [25, 27], [27, 29], [29, 31],
      // Right Leg
      [24, 26], [26, 28], [28, 30], [30, 32],
    ];

    this.landmarkNames = [
      "nose", "left_eye_inner", "left_eye", "left_eye_outer",
      "right_eye_inner", "right_eye", "right_eye_outer",
      "left_ear", "right_ear", "mouth_left", "mouth_right",
      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
      "left_wrist", "right_wrist", "left_pinky", "right_pinky",
      "left_index", "right_index", "left_thumb", "right_thumb",
      "left_hip", "right_hip", "left_knee", "right_knee",
      "left_ankle", "right_ankle", "left_heel", "right_heel",
      "left_foot_index", "right_foot_index"
    ];

    this.cameraManager = new CameraManager(this.videoEl, {
      onStreamReady: () => this.onCameraReady(),
      onError: (msg) => this.onCameraError(msg),
      onStatusChange: (status, msg) => this.onCameraStatus(status, msg),
    });

    this.mpPose = null;
    this.initEventListeners();
  }

  initEventListeners() {
    if (this.btnStart) this.btnStart.addEventListener('click', () => this.startWorkout());
    if (this.btnPause) this.btnPause.addEventListener('click', () => this.pauseWorkout());
    if (this.btnResume) this.btnResume.addEventListener('click', () => this.resumeWorkout());
    if (this.btnEnd) this.btnEnd.addEventListener('click', () => this.endWorkout());
  }

  onCameraStatus(status, message) {
    if (this.cameraStatusText) this.cameraStatusText.textContent = message;
    if (this.statusDot) {
      this.statusDot.className = 'status-dot';
      if (status === 'active') this.statusDot.classList.add('active');
      else if (status === 'error') this.statusDot.classList.add('error');
    }
  }

  onCameraError(errorMessage) {
    if (this.statusBannerEl) {
      this.updateStatusBanner('warning', 'Camera Access Required', errorMessage);
    }
    if (this.placeholderEl) {
      this.placeholderEl.classList.remove('hidden');
      this.placeholderEl.innerHTML = `
        <div style="color: #ef4444; font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
        <h3 style="color: #ffffff; margin-bottom: 0.5rem;">Camera Error</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; max-width: 400px;">${errorMessage}</p>
        <button class="btn btn-secondary" style="margin-top: 1rem;" onclick="location.reload()">Retry</button>
      `;
    }
  }

  async startWorkout() {
    if (this.isRunning) return;

    if (this.btnStart) this.btnStart.disabled = true;
    const cameraStarted = await this.cameraManager.start();
    if (!cameraStarted) {
      if (this.btnStart) this.btnStart.disabled = false;
      return;
    }

    if (this.placeholderEl) this.placeholderEl.classList.add('hidden');
    this.isRunning = true;
    this.isPaused = false;
    this.startTime = Date.now();
    this.startTimer();

    // Toggle button visibility
    if (this.btnStart) this.btnStart.style.display = 'none';
    if (this.btnPause) this.btnPause.style.display = 'inline-flex';
    if (this.btnEnd) this.btnEnd.style.display = 'inline-flex';

    await this.initPoseDetector();
    this.animationLoop();
  }

  pauseWorkout() {
    this.isPaused = true;
    if (this.btnPause) this.btnPause.style.display = 'none';
    if (this.btnResume) this.btnResume.style.display = 'inline-flex';
    this.updateStatusBanner('warning', 'Workout Paused', 'Press Resume when you are ready to continue.');
  }

  resumeWorkout() {
    this.isPaused = false;
    if (this.btnResume) this.btnResume.style.display = 'none';
    if (this.btnPause) this.btnPause.style.display = 'inline-flex';
    this.updateStatusBanner('correct', 'Resumed', 'Resume your exercise movement.');
  }

  async endWorkout() {
    this.isRunning = false;
    clearInterval(this.timerInterval);
    this.cameraManager.stop();

    if (this.btnEnd) this.btnEnd.disabled = true;
    if (this.feedbackTextEl) this.feedbackTextEl.textContent = 'Saving session summary...';

    try {
      const resp = await fetch('/api/finish-workout/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_token: this.sessionToken,
          exercise_slug: this.exerciseSlug,
          rep_count: this.latestRepCount,
          duration_seconds: this.elapsedSeconds,
        }),
      });

      const data = await resp.json();
      if (data.success && data.results_url) {
        window.location.href = data.results_url;
      } else {
        window.location.href = '/results/';
      }
    } catch (e) {
      console.error('Error saving workout:', e);
      window.location.href = '/results/';
    }
  }

  startTimer() {
    clearInterval(this.timerInterval);
    this.timerInterval = setInterval(() => {
      if (!this.isPaused && this.isRunning) {
        this.elapsedSeconds++;
        const mins = String(Math.floor(this.elapsedSeconds / 60)).padStart(2, '0');
        const secs = String(this.elapsedSeconds % 60).padStart(2, '0');
        if (this.timerValEl) this.timerValEl.textContent = `${mins}:${secs}`;
      }
    }, 1000);
  }

  onCameraReady() {
    if (this.canvasEl && this.videoEl) {
      this.canvasEl.width = this.videoEl.videoWidth || 640;
      this.canvasEl.height = this.videoEl.videoHeight || 480;
    }
  }

  async initPoseDetector() {
    // If window.Pose (MediaPipe JS) is loaded from CDN, initialize it
    if (window.Pose && !this.useServerDetector) {
      try {
        this.mpPose = new window.Pose({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`,
        });

        this.mpPose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        this.mpPose.onResults((results) => this.onPoseResults(results));
        console.log('Client-side MediaPipe Pose initialized successfully.');
        return;
      } catch (err) {
        console.warn('Failed to initialize client MediaPipe Pose, falling back to server engine:', err);
      }
    }

    // Server fallback mode
    console.log('Using backend server pose analysis engine.');
    this.useServerDetector = true;
  }

  async animationLoop() {
    if (!this.isRunning) return;

    if (!this.isPaused && this.videoEl && this.videoEl.readyState >= 2) {
      if (this.canvasEl && this.videoEl) {
        if (this.canvasEl.width !== this.videoEl.videoWidth) {
          this.canvasEl.width = this.videoEl.videoWidth;
          this.canvasEl.height = this.videoEl.videoHeight;
        }
      }

      const now = Date.now();
      if (!this.isProcessingFrame) {
        if (this.mpPose && !this.useServerDetector) {
          // Client MediaPipe frame processing
          this.isProcessingFrame = true;
          try {
            await this.mpPose.send({ image: this.videoEl });
          } catch (e) {
            console.error('MediaPipe error:', e);
          } finally {
            this.isProcessingFrame = false;
          }
        } else if (this.useServerDetector && (now - this.lastApiCallTime >= this.apiThrottleMs)) {
          // Server backend frame processing fallback
          this.isProcessingFrame = true;
          this.lastApiCallTime = now;
          this.sendFrameToServer();
        }
      }
    }

    requestAnimationFrame(() => this.animationLoop());
  }

  onPoseResults(results) {
    if (!this.ctx || !this.canvasEl) return;

    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);

    if (!results.poseLandmarks) {
      this.updateStatusBanner('warning', 'Tracking Inactive', 'Step back so your full body is in the camera frame.');
      return;
    }

    // Render skeleton
    this.drawSkeleton(results.poseLandmarks);

    // Throttle server posture analysis calls
    const now = Date.now();
    if (now - this.lastApiCallTime >= this.apiThrottleMs) {
      this.lastApiCallTime = now;
      this.sendKeypointsToServer(results.poseLandmarks);
    }
  }

  drawSkeleton(landmarks) {
    const w = this.canvasEl.width;
    const h = this.canvasEl.height;

    // Draw connection lines
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = '#00f59b';
    this.ctx.lineCap = 'round';

    for (const [idx1, idx2] of this.connections) {
      const p1 = landmarks[idx1];
      const p2 = landmarks[idx2];
      if (p1 && p2 && (p1.visibility || 1.0) > 0.45 && (p2.visibility || 1.0) > 0.45) {
        this.ctx.beginPath();
        this.ctx.moveTo(p1.x * w, p1.y * h);
        this.ctx.lineTo(p2.x * w, p2.y * h);
        this.ctx.stroke();
      }
    }

    // Draw joint dots
    for (const lm of landmarks) {
      if ((lm.visibility || 1.0) > 0.45) {
        this.ctx.beginPath();
        this.ctx.arc(lm.x * w, lm.y * h, 5, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#10b981';
        this.ctx.stroke();
      }
    }
  }

  async sendKeypointsToServer(landmarks) {
    const keypointDict = {};
    for (let i = 0; i < landmarks.length; i++) {
      const name = this.landmarkNames[i];
      if (name) {
        keypointDict[name] = {
          x: landmarks[i].x,
          y: landmarks[i].y,
          z: landmarks[i].z || 0,
          visibility: landmarks[i].visibility || 1.0,
        };
      }
    }

    try {
      const resp = await fetch('/api/analyze-pose/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_token: this.sessionToken,
          exercise: this.exerciseSlug,
          keypoints: keypointDict,
        }),
      });

      const data = await resp.json();
      if (data.success) {
        this.updateHUD(data);
      }
    } catch (err) {
      console.warn('API sync error:', err);
    }
  }

  async sendFrameToServer() {
    if (!this.canvasEl || !this.videoEl) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 480;
    tempCanvas.height = 360;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(this.videoEl, 0, 0, 480, 360);
    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.65);

    try {
      const resp = await fetch('/api/analyze-frame/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_token: this.sessionToken,
          exercise: this.exerciseSlug,
          image: dataUrl,
        }),
      });

      const data = await resp.json();
      if (data.success) {
        this.updateHUD(data);
        if (data.keypoints && this.ctx) {
          this.ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
          this.drawKeypointDict(data.keypoints);
        }
      }
    } catch (err) {
      console.warn('Server frame analyze error:', err);
    } finally {
      this.isProcessingFrame = false;
    }
  }

  drawKeypointDict(keypointDict) {
    const w = this.canvasEl.width;
    const h = this.canvasEl.height;

    // Draw lines
    const connPairs = [
      ["left_shoulder", "right_shoulder"],
      ["left_shoulder", "left_hip"],
      ["right_shoulder", "right_hip"],
      ["left_hip", "right_hip"],
      ["left_shoulder", "left_elbow"],
      ["left_elbow", "left_wrist"],
      ["right_shoulder", "right_elbow"],
      ["right_elbow", "right_wrist"],
      ["left_hip", "left_knee"],
      ["left_knee", "left_ankle"],
      ["right_hip", "right_knee"],
      ["right_knee", "right_ankle"],
    ];

    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = '#00f59b';
    for (const [k1, k2] of connPairs) {
      const p1 = keypointDict[k1];
      const p2 = keypointDict[k2];
      if (p1 && p2 && (p1.visibility || 1) > 0.45 && (p2.visibility || 1) > 0.45) {
        this.ctx.beginPath();
        this.ctx.moveTo(p1.x * w, p1.y * h);
        this.ctx.lineTo(p2.x * w, p2.y * h);
        this.ctx.stroke();
      }
    }

    for (const [k, p] of Object.entries(keypointDict)) {
      if ((p.visibility || 1) > 0.45) {
        this.ctx.beginPath();
        this.ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.strokeStyle = '#10b981';
        this.ctx.stroke();
      }
    }
  }

  updateHUD(data) {
    // 1. Repetition counter with bump effect
    const count = data.repetition_count || 0;
    if (count !== this.latestRepCount) {
      this.latestRepCount = count;
      if (this.repCountEl) {
        this.repCountEl.textContent = count;
        this.repCountEl.classList.add('bump');
        setTimeout(() => this.repCountEl.classList.remove('bump'), 300);
      }
    }

    // 2. Movement phase
    if (this.phaseBadgeEl && data.movement_phase) {
      this.phaseBadgeEl.textContent = data.movement_phase.toUpperCase();
    }

    // 3. Posture Status & Feedback
    const status = data.posture_status || 'warning';
    const feedback = data.feedback || '';
    const statusTitle =
      status === 'correct' ? 'Form: Good' :
      status === 'incorrect' ? 'Form Deviation' : 'Tracking Notice';

    this.updateStatusBanner(status, statusTitle, feedback);

    // 4. Live Measurements
    const m = data.measurements || {};
    if (this.exerciseSlug === 'squat') {
      if (this.primaryAngleValEl) {
        const val = m.primary_knee_angle ?? m.left_knee_angle ?? m.right_knee_angle;
        this.primaryAngleValEl.textContent = val !== null && val !== undefined ? `${val}°` : '--';
      }
      if (this.primaryAngleLabelEl) this.primaryAngleLabelEl.textContent = 'Knee Angle';
      if (this.torsoIncValEl) {
        this.torsoIncValEl.textContent = m.torso_inclination !== null && m.torso_inclination !== undefined ? `${m.torso_inclination}°` : '--';
      }
      if (this.symmetryValEl) {
        this.symmetryValEl.textContent = m.knee_symmetry_diff !== null && m.knee_symmetry_diff !== undefined ? `${m.knee_symmetry_diff}°` : '--';
      }
    } else {
      if (this.primaryAngleValEl) {
        const val = m.primary_elbow_angle ?? m.left_elbow_angle ?? m.right_elbow_angle;
        this.primaryAngleValEl.textContent = val !== null && val !== undefined ? `${val}°` : '--';
      }
      if (this.primaryAngleLabelEl) this.primaryAngleLabelEl.textContent = 'Elbow Angle';
      if (this.torsoIncValEl) {
        this.torsoIncValEl.textContent = m.torso_inclination !== null && m.torso_inclination !== undefined ? `${m.torso_inclination}°` : '--';
      }
      if (this.symmetryValEl) {
        this.symmetryValEl.textContent = m.elbow_symmetry_diff !== null && m.elbow_symmetry_diff !== undefined ? `${m.elbow_symmetry_diff}°` : '--';
      }
    }
  }

  updateStatusBanner(statusClass, title, message) {
    if (!this.statusBannerEl) return;
    this.statusBannerEl.className = `status-banner ${statusClass}`;
    if (this.statusTitleEl) this.statusTitleEl.textContent = title;
    if (this.feedbackTextEl) this.feedbackTextEl.textContent = message;

    const iconEl = this.statusBannerEl.querySelector('.status-icon');
    if (iconEl) {
      if (statusClass === 'correct') iconEl.textContent = '✓';
      else if (statusClass === 'incorrect') iconEl.textContent = '✗';
      else iconEl.textContent = '!';
    }
  }
}

window.WorkoutSessionController = WorkoutSessionController;
