function vjepa2App() {
  return {
    tab: 'upload',
    modelReady: false,
    streaming: false,
    sessionId: null,
    results: [],
    dragover: false,
    topK: 5,
    numFrames: 16,
    stride: 8,
    videoUrl: null,
    selectedFile: null,
    rtspUrl: '',
    ws: null,
    mediaRecorder: null,
    cameraStream: null,

    get previewUrl() {
      if (this.sessionId) {
        return `/v2/models/vjepa2/sessions/${this.sessionId}/preview`;
      }
      return '';
    },

    init() {
      this.checkModelReady();
    },

    async checkModelReady() {
      try {
        const resp = await fetch('/v2/health/ready');
        this.modelReady = resp.ok;
      } catch {
        this.modelReady = false;
      }
      if (!this.modelReady) {
        setTimeout(() => this.checkModelReady(), 5000);
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
      }
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'stop' }));
      }
      this.streaming = false;
      this.stopCamera();
    },

    connectWs(path) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      this.ws = new WebSocket(`${proto}//${location.host}${path}`);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'session') {
          this.sessionId = msg.session_id;
        } else if (msg.type === 'prediction') {
          this.results.push(msg);
          this.$nextTick(() => {
            const list = document.querySelector('.results-list');
            if (list) list.scrollTop = list.scrollHeight;
          });
        } else if (msg.type === 'complete') {
          this.streaming = false;
          this.stopCamera();
        } else if (msg.type === 'error') {
          console.error('Server error:', msg.message);
          this.streaming = false;
          this.stopCamera();
        }
      };
      this.ws.onclose = () => {
        this.streaming = false;
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
      }));
      this.streaming = true;
      const buffer = await this.selectedFile.arrayBuffer();
      this.ws.send(buffer);
      this.ws.send(JSON.stringify({ action: 'stop' }));
    },

    async startCamera() {
      try {
        this.cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
      } catch (err) {
        console.error('Camera access denied:', err);
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
      this.mediaRecorder = new MediaRecorder(this.cameraStream, {
        mimeType: 'video/webm;codecs=vp8',
      });
      const chunks = [];
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.ws.readyState === WebSocket.OPEN) {
          chunks.push(event.data);
        }
      };
      this.mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const buffer = await blob.arrayBuffer();
        if (this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(buffer);
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
    },

    stopCamera() {
      if (this.cameraStream) {
        this.cameraStream.getTracks().forEach(t => t.stop());
        this.cameraStream = null;
      }
    },
  };
}
