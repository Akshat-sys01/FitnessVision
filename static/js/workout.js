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

    // Speech & Voice Feedback
    this.lastSpokenFeedback = '';
    this.lastAnnouncedRep = 0;
    this.exerciseCompleted = false;
    this.speechEnabled = true;

    if ('speechSynthesis' in window) {
      setTimeout(() => {
        const test = new SpeechSynthesisUtterance('Fitness Vision audio test. Your audio is working.');
        test.lang = 'en-US';
        window.speechSynthesis.speak(test);
      }, 1000);
    }
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

  /**
   * Converts detected landmarks or keypoint map into canonical 16 joints.
   */
  getCanonical16(landmarks) {
    const mpIndices = {
      right_ankle: 28,
      right_knee: 26,
      right_hip: 24,
      left_hip: 23,
      left_knee: 25,
      left_ankle: 27,
      right_wrist: 16,
      right_elbow: 14,
      right_shoulder: 12,
      left_shoulder: 11,
      left_elbow: 13,
      left_wrist: 15,
      nose: 0,
    };

    const canonical = {};

    if (Array.isArray(landmarks)) {
      for (const [name, idx] of Object.entries(mpIndices)) {
        const pt = landmarks[idx];
        if (pt) {
          const vis = pt.visibility !== undefined ? pt.visibility : 1.0;
          if (vis >= 0.45) {
            canonical[name] = { x: pt.x, y: pt.y, visibility: vis };
          }
        }
      }
    } else if (landmarks && typeof landmarks === 'object') {
      for (const [name, idx] of Object.entries(mpIndices)) {
        const pt = landmarks[name] || landmarks[idx] || landmarks[String(idx)];
        if (pt) {
          const vis = pt.visibility !== undefined ? pt.visibility : (pt.score !== undefined ? pt.score : 1.0);
          if (vis >= 0.45) {
            canonical[name] = { x: pt.x, y: pt.y, visibility: vis };
          }
        }
      }
      if (!canonical.nose && (landmarks.nose || landmarks[0] || landmarks['0'])) {
        const n = landmarks.nose || landmarks[0] || landmarks['0'];
        const vis = n.visibility !== undefined ? n.visibility : 1.0;
        if (vis >= 0.45) {
          canonical.nose = { x: n.x, y: n.y, visibility: vis };
        }
      }
    }

    // Derive Joint 6: Pelvis (midpoint of left & right hips)
    if (canonical.left_hip && canonical.right_hip) {
      canonical.pelvis = {
        x: (canonical.left_hip.x + canonical.right_hip.x) / 2,
        y: (canonical.left_hip.y + canonical.right_hip.y) / 2,
        visibility: Math.min(canonical.left_hip.visibility, canonical.right_hip.visibility),
      };
    }

    // Derive Joint 7: Thorax (midpoint of left & right shoulders)
    if (canonical.left_shoulder && canonical.right_shoulder) {
      canonical.thorax = {
        x: (canonical.left_shoulder.x + canonical.right_shoulder.x) / 2,
        y: (canonical.left_shoulder.y + canonical.right_shoulder.y) / 2,
        visibility: Math.min(canonical.left_shoulder.visibility, canonical.right_shoulder.visibility),
      };
    }

    // Derive Joint 8: Upper Neck (midpoint between thorax and nose)
    if (canonical.thorax && canonical.nose) {
      canonical.upper_neck = {
        x: (canonical.thorax.x + canonical.nose.x) / 2,
        y: (canonical.thorax.y + canonical.nose.y) / 2,
        visibility: Math.min(canonical.thorax.visibility, canonical.nose.visibility),
      };
    }

    // Derive Joint 9: Head Top (approximate position above head extrapolated from nose and upper_neck)
    if (canonical.nose && canonical.upper_neck) {
      canonical.head_top = {
        x: canonical.nose.x + 0.35 * (canonical.nose.x - canonical.upper_neck.x),
        y: canonical.nose.y + 0.35 * (canonical.nose.y - canonical.upper_neck.y),
        visibility: Math.min(canonical.nose.visibility, canonical.upper_neck.visibility),
      };
    } else if (canonical.nose) {
      canonical.head_top = {
        x: canonical.nose.x,
        y: canonical.nose.y - 0.08,
        visibility: canonical.nose.visibility,
      };
    }

    return canonical;
  }

  /**
   * Renders strictly the canonical 16 joints and connections on the canvas.
   */
  renderCanonicalSkeleton(canonical) {
    if (!this.ctx || !this.canvasEl) return;
    const w = this.canvasEl.width;
    const h = this.canvasEl.height;

    // Canonical 16-joint list in exact required order
    const canonicalJointOrder = [
      'right_ankle',     // 0
      'right_knee',      // 1
      'right_hip',       // 2
      'left_hip',        // 3
      'left_knee',       // 4
      'left_ankle',      // 5
      'pelvis',          // 6
      'thorax',          // 7
      'upper_neck',      // 8
      'head_top',        // 9
      'right_wrist',     // 10
      'right_elbow',     // 11
      'right_shoulder',  // 12
      'left_shoulder',   // 13
      'left_elbow',      // 14
      'left_wrist',      // 15
    ];

    // Canonical connections between the 16 joints
    const connections = [
      // Head and neck
      ['head_top', 'upper_neck'],
      ['upper_neck', 'thorax'],
      // Torso
      ['thorax', 'left_shoulder'],
      ['thorax', 'right_shoulder'],
      ['thorax', 'pelvis'],
      ['pelvis', 'left_hip'],
      ['pelvis', 'right_hip'],
      // Left arm
      ['left_shoulder', 'left_elbow'],
      ['left_elbow', 'left_wrist'],
      // Right arm
      ['right_shoulder', 'right_elbow'],
      ['right_elbow', 'right_wrist'],
      // Left leg
      ['left_hip', 'left_knee'],
      ['left_knee', 'left_ankle'],
      // Right leg
      ['right_hip', 'right_knee'],
      ['right_knee', 'right_ankle'],
    ];

    // 1. Draw connection lines
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = '#00f59b';
    this.ctx.lineCap = 'round';

    for (const [j1, j2] of connections) {
      const p1 = canonical[j1];
      const p2 = canonical[j2];
      if (p1 && p2 && p1.visibility >= 0.45 && p2.visibility >= 0.45) {
        this.ctx.beginPath();
        this.ctx.moveTo(p1.x * w, p1.y * h);
        this.ctx.lineTo(p2.x * w, p2.y * h);
        this.ctx.stroke();
      }
    }

    // 2. Draw circles for only the 16 canonical joints (NO face, fingers, or toes)
    for (const name of canonicalJointOrder) {
      const pt = canonical[name];
      if (pt && pt.visibility >= 0.45) {
        this.ctx.beginPath();
        this.ctx.arc(pt.x * w, pt.y * h, 5, 0, 2 * Math.PI);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#10b981';
        this.ctx.stroke();
      }
    }
  }

  drawSkeleton(landmarks) {
    const canonical = this.getCanonical16(landmarks);
    this.renderCanonicalSkeleton(canonical);
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
    const canonical = this.getCanonical16(keypointDict);
    this.renderCanonicalSkeleton(canonical);
  }

  updateHUD(data) {
    // 1. Repetition counter with bump effect
    const count = data.repetition_count || 0;
    if (count !== this.latestRepCount) {
      this.latestRepCount = count;
      this.announceRepProgress(count);
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

  announceRepProgress(repCount) {
    if (repCount <= this.lastAnnouncedRep) return;
    this.lastAnnouncedRep = repCount;

    let message = `${repCount} repetitions completed.`;
    if (repCount === 1) {
      message = 'First repetition completed. Good start!';
    } else if (repCount === 5) {
      message = 'Halfway there! Five more repetitions to go!';
    } else if (repCount === 9) {
      message = 'Just one more repetition! You can do it!';
    } else if (repCount >= 10 && !this.exerciseCompleted) {
      this.exerciseCompleted = true;
      message = 'Excellent! All 10 repetitions completed. Exercise complete. Great work!';
    } else if (repCount > 10) {
      return;
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const speech = new SpeechSynthesisUtterance(message);
      speech.lang = 'en-US';
      speech.rate = 1;
      speech.pitch = 1;
      window.speechSynthesis.speak(speech);
    }
  }

  updateStatusBanner(statusClass, title, message) {
    if (!this.statusBannerEl) return;
    this.statusBannerEl.className = `status-banner ${statusClass}`;
    if (this.statusTitleEl) this.statusTitleEl.textContent = title;
    if (this.feedbackTextEl) this.feedbackTextEl.textContent = message;

    // Speak only when the feedback message changes
    if (
      this.speechEnabled &&
      message &&
      message !== this.lastSpokenFeedback &&
      'speechSynthesis' in window
    ) {
      this.lastSpokenFeedback = message;
      window.speechSynthesis.cancel();
      const speech = new SpeechSynthesisUtterance(message);
      speech.lang = 'en-US';
      speech.rate = 1;
      speech.pitch = 1;
      window.speechSynthesis.speak(speech);
    }

    const iconEl = this.statusBannerEl.querySelector('.status-icon');
    if (iconEl) {
      if (statusClass === 'correct') iconEl.textContent = '✓';
      else if (statusClass === 'incorrect') iconEl.textContent = '✗';
      else iconEl.textContent = '!';
    }
  }
}

window.WorkoutSessionController = WorkoutSessionController;
