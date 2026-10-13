/**
 * CameraManager - Manages browser webcam lifecycle, device acquisition,
 * permission error handling, and clean stream termination.
 */

class CameraManager {
  constructor(videoElement, options = {}) {
    this.videoElement = videoElement;
    this.stream = null;
    this.isActive = false;
    this.facingMode = options.facingMode || 'user';
    this.desiredWidth = options.width || 640;
    this.desiredHeight = options.height || 480;

    // Callbacks
    this.onStreamReady = options.onStreamReady || null;
    this.onError = options.onError || null;
    this.onStatusChange = options.onStatusChange || null;

    // Ensure cleanup on page navigation/unload
    window.addEventListener('beforeunload', () => this.stop());
  }

  updateStatus(status, message = '') {
    if (this.onStatusChange) {
      this.onStatusChange(status, message);
    }
  }

  async start() {
    if (this.isActive && this.stream) {
      console.warn('Camera is already running.');
      return true;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const err = new Error('Camera access API is not supported in this browser.');
      this.handleError(err);
      return false;
    }

    this.updateStatus('requesting', 'Requesting camera permissions...');

    const constraints = {
      audio: false,
      video: {
        width: { ideal: this.desiredWidth },
        height: { ideal: this.desiredHeight },
        facingMode: this.facingMode,
      },
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.videoElement.srcObject = this.stream;

      // Wait until video metadata loads and playback begins
      await new Promise((resolve) => {
        this.videoElement.onloadedmetadata = () => {
          this.videoElement.play();
          resolve();
        };
      });

      this.isActive = true;
      this.updateStatus('active', 'Camera active');

      if (this.onStreamReady) {
        this.onStreamReady(this.videoElement, this.stream);
      }
      return true;
    } catch (err) {
      this.handleError(err);
      return false;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {
          console.warn('Error stopping track:', e);
        }
      });
      this.stream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    this.isActive = false;
    this.updateStatus('stopped', 'Camera stopped');
  }

  handleError(err) {
    this.isActive = false;
    let friendlyMessage = 'Unable to access camera.';

    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      friendlyMessage = 'Camera permission was denied. Please allow camera access in your browser settings to continue.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      friendlyMessage = 'No camera device found on your system. Please connect a webcam.';
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      friendlyMessage = 'Camera is in use by another application or operating system process.';
    } else if (err.name === 'OverconstrainedError') {
      friendlyMessage = 'Camera does not meet requested resolution constraints.';
    } else {
      friendlyMessage = err.message || 'An unexpected error occurred while initializing the webcam.';
    }

    this.updateStatus('error', friendlyMessage);
    if (this.onError) {
      this.onError(friendlyMessage, err);
    }
  }
}

window.CameraManager = CameraManager;
