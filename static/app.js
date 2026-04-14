function vjepa2App() {
  return {
    tab: 'upload',
    modelReady: false,
    modelName: '',
    modelDevice: '',
    streaming: false,
    recording: false,
    sessionId: null,
    results: [],
    dragover: false,
    topK: 3,
    numFrames: 16,
    stride: 16,
    videoUrl: null,
    selectedFile: null,
    rtspUrl: '',
    ws: null,
    mediaRecorder: null,
    cameraStream: null,
    recordingSeconds: 0,
    totalClips: 0,
    _recordingTimer: null,
    cameraAvailable: false,
    cameraError: '',

    get previewUrl() {
      if (this.sessionId) {
        return `/v2/models/vjepa2/sessions/${this.sessionId}/preview`;
      }
      return '';
    },

    init() {
      this.checkModelReady();
      this.checkCameraAvailable();
    },

    async checkCameraAvailable() {
      // Camera requires secure context (HTTPS or localhost)
      if (!window.isSecureContext) {
        this.cameraError = 'Camera requires HTTPS';
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.cameraError = 'Camera API not available';
        return;
      }
      // If we're in a secure context with mediaDevices API, assume camera might be available
      // The actual getUserMedia call in startCamera() will handle permission/detection errors
      this.cameraAvailable = true;
      this.cameraError = '';
    },

    async checkModelReady() {
      try {
        const resp = await fetch('/v2/health/ready');
        if (resp.ok) {
          const data = await resp.json();
          this.modelReady = true;
          this.modelName = data.model || '';
          this.modelDevice = data.device || '';
        } else {
          this.modelReady = false;
          try {
            const data = await resp.json();
            this.modelName = data.model || '';
          } catch {}
        }
      } catch {
        this.modelReady = false;
      }
      if (!this.modelReady) {
        setTimeout(() => this.checkModelReady(), 3000);
      }
    },

    formatTime(ms) {
      const s = Math.floor(ms / 1000);
      const m = Math.floor(s / 60);
      const sec = s % 60;
      const frac = Math.floor((ms % 1000) / 100);
      return `${m}:${String(sec).padStart(2, '0')}.${frac}`;
    },

    handleFileSelect(event) {
      const file = event.target.files[0];
      if (file) {
        this.selectedFile = file;
        this.videoUrl = URL.createObjectURL(file);
      }
    },

    handleDrop(event) {
      this.dragover = false;
      const file = event.dataTransfer.files[0];
      if (file && file.type.startsWith('video/')) {
        this.selectedFile = file;
        this.videoUrl = URL.createObjectURL(file);
      }
    },

    async start() {
      this.results = [];
      this.sessionId = null;
      this.totalClips = 0;
      this.recordingSeconds = 0;
      if (this.tab === 'upload') {
        await this.startUpload();
      } else if (this.tab === 'camera') {
        await this.startCamera();
      } else if (this.tab === 'rtsp') {
        await this.startRtsp();
      }
    },

    stop() {
      if (this.tab === 'camera' && this.mediaRecorder) {
        this.mediaRecorder.stop();
        // onstop handler will send the stop action after final data
      } else if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'stop' }));
      }
      this.recording = false;
      this._stopRecordingTimer();
      this.stopCamera();
      // streaming stays true until server sends 'complete'
    },

    connectWs(path) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      this.ws = new WebSocket(`${proto}//${location.host}${path}`);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'session') {
          this.sessionId = msg.session_id;
        } else if (msg.type === 'progress') {
          this.totalClips = msg.clips_queued;
        } else if (msg.type === 'prediction') {
          this.results.push(msg);
          this.$nextTick(() => {
            const list = document.querySelector('.results-list');
            if (list) list.scrollTop = list.scrollHeight;
          });
        } else if (msg.type === 'complete') {
          this.streaming = false;
          this.recording = false;
          this._stopRecordingTimer();
          this.stopCamera();
        } else if (msg.type === 'error') {
          console.error('Server error:', msg.message);
          this.streaming = false;
          this.recording = false;
          this._stopRecordingTimer();
          this.stopCamera();
        }
      };
      this.ws.onclose = () => {
        this.streaming = false;
        this.recording = false;
        this._stopRecordingTimer();
        this.stopCamera();
      };
      return new Promise((resolve, reject) => {
        this.ws.onopen = resolve;
        this.ws.onerror = reject;
      });
    },

    async startUpload() {
      if (!this.selectedFile) return;
      await this.connectWs('/v2/models/vjepa2/stream/browser');
      this.ws.send(JSON.stringify({
        top_k: this.topK,
        num_frames: this.numFrames,
        stride: this.stride,
        media_type: this.selectedFile.type || null,
      }));
      this.streaming = true;
      this.recording = true;
      const buffer = await this.selectedFile.arrayBuffer();
      // Send in 1 MiB chunks to avoid WebSocket frame size limits
      const CHUNK = 1024 * 1024;
      for (let offset = 0; offset < buffer.byteLength; offset += CHUNK) {
        this.ws.send(buffer.slice(offset, offset + CHUNK));
        // Wait for buffer to drain before sending more
        while (this.ws.bufferedAmount > CHUNK * 2) {
          await new Promise(r => setTimeout(r, 50));
        }
      }
      // Wait for all data to be sent before signaling stop
      while (this.ws.bufferedAmount > 0) {
        await new Promise(r => setTimeout(r, 50));
      }
      this.ws.send(JSON.stringify({ action: 'stop' }));
      this.recording = false;
    },

    async startCamera() {
      // Check for secure context first
      if (!window.isSecureContext) {
        this.cameraError = 'Camera requires HTTPS';
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.cameraError = 'Camera API not available';
        return;
      }
      try {
        this.cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
      } catch (err) {
        this.cameraError = err.name === 'NotAllowedError' ? 'Camera permission denied' : 'Camera error: ' + err.message;
        return;
      }
      this.$nextTick(() => {
        const video = this.$refs.cameraPreview;
        if (video) video.srcObject = this.cameraStream;
      });
      await this.connectWs('/v2/models/vjepa2/stream/browser');
      this.ws.send(JSON.stringify({
        top_k: this.topK,
        num_frames: this.numFrames,
        stride: this.stride,
      }));
      this.streaming = true;
      this.recording = true;
      this.recordingSeconds = 0;
      this._startRecordingTimer();
      this.mediaRecorder = new MediaRecorder(this.cameraStream, {
        mimeType: 'video/webm;codecs=vp8',
      });
      this.mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && this.ws.readyState === WebSocket.OPEN) {
          const buffer = await event.data.arrayBuffer();
          this.ws.send(buffer);
        }
      };
      this.mediaRecorder.onstop = () => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: 'stop' }));
        }
      };
      this.mediaRecorder.start(1000);
    },

    async startRtsp() {
      if (!this.rtspUrl) return;
      await this.connectWs('/v2/models/vjepa2/stream/rtsp');
      this.ws.send(JSON.stringify({
        rtsp_url: this.rtspUrl,
        top_k: this.topK,
        num_frames: this.numFrames,
        stride: this.stride,
      }));
      this.streaming = true;
      this.recording = true;
      this.recordingSeconds = 0;
      this._startRecordingTimer();
    },

    stopCamera() {
      if (this.cameraStream) {
        this.cameraStream.getTracks().forEach(t => t.stop());
        this.cameraStream = null;
      }
    },

    _startRecordingTimer() {
      this._recordingTimer = setInterval(() => {
        this.recordingSeconds++;
      }, 1000);
    },

    _stopRecordingTimer() {
      if (this._recordingTimer) {
        clearInterval(this._recordingTimer);
        this._recordingTimer = null;
      }
    },

    formatDuration(seconds) {
      const m = Math.floor(seconds / 60);
      const s = seconds % 60;
      return `${m}:${String(s).padStart(2, '0')}`;
    },
  };
}
