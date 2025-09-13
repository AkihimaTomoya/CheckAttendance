document.addEventListener('DOMContentLoaded', () => {
  // ---------- DOM helpers ----------
  const $ = (id) => document.getElementById(id);
  const video = $('videoStream');
  const canvas = $('canvasOverlay');
  const ctx = canvas.getContext('2d');

  const overlayChk = $('overlayChk');
  const toggleCameraBtn = $('toggleCameraBtn');
  const reloadFacebankBtn = $('reloadFacebankBtn');
  const connectionStatus = $('connectionStatus');
  const statusLine = $('status');
  const debugLine = $('debugLine');

  // ---------- State ----------
  let ws = null;
  let streaming = false;
  let overlayOn = overlayChk ? overlayChk.checked : true;
  let frameRAF = null;
  let overlayRAF = null;
  let lastLocs = [];
  let recogMap = {};
  let srcW = 0, srcH = 0;
  let lastSend = 0;
  const SEND_INTERVAL_MS = 120; // throttle WS frame sends

  // ---------- UI helpers ----------
  function setPill(text, cls) {
    connectionStatus.textContent = text;
    connectionStatus.className = 'pill ' + (cls || '');
  }
  function setStatus(text) {
    statusLine.textContent = text;
  }
  function setDebug(text) {
    if (debugLine) debugLine.textContent = text;
    // console.log('[DEBUG]', text);
  }

  // ---------- Canvas & drawing ----------
  function fitCanvasToVideo() {
    const w = video.videoWidth || video.clientWidth || 800;
    const h = video.videoHeight || video.clientHeight || 600;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
  }

  function drawOverlay() {
    if (!streaming) return;
    fitCanvasToVideo();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!overlayOn) {
      overlayRAF = requestAnimationFrame(drawOverlay);
      return;
    }

    const faces = Array.isArray(lastLocs) ? lastLocs : [];
    setStatus(faces.length > 0 ? `Tracking ${faces.length} face(s)` : 'Waiting for frames…');

    const sw = srcW || canvas.width;
    const sh = srcH || canvas.height;
    const sx = canvas.width / sw;
    const sy = canvas.height / sh;

    for (const loc of faces) {
      let [x1, y1, x2, y2] = loc.bbox;
      x1 = Math.round(x1 * sx);
      y1 = Math.round(y1 * sy);
      x2 = Math.round(x2 * sx);
      y2 = Math.round(y2 * sy);

      const rec = recogMap[loc.id] || {};
      const name = rec.name || rec.name_top1 || 'Unknown';

      ctx.lineWidth = 2;
      ctx.strokeStyle = '#10b981';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = name;
      ctx.font = '14px ui-sans-serif, system-ui';
      const tw = ctx.measureText(label).width;
      const ly = Math.max(0, y1 - 22);
      ctx.fillStyle = '#10b981';
      ctx.fillRect(x1, ly, tw + 10, 22);
      ctx.fillStyle = '#052e1e';
      ctx.fillText(label, x1 + 5, ly + 16);
    }

    overlayRAF = requestAnimationFrame(drawOverlay);
  }

  // ---------- Camera ----------
  function explainGetUserMediaError(e) {
    const httpsHint =
      location.hostname !== 'localhost' && location.protocol !== 'https:'
        ? ' (Hint: use http://localhost:<port> on the machine running the server, or HTTPS when accessing via LAN IP).'
        : '';
    if (e && e.name) {
      switch (e.name) {
        case 'NotAllowedError':
          return 'You denied camera permission.' + httpsHint;
        case 'NotFoundError':
          return 'No camera device found.';
        case 'NotReadableError':
          return 'The device is busy or blocked by the OS.';
        case 'OverconstrainedError':
          return 'Requested resolution is not supported by the camera.';
        case 'SecurityError':
          return 'Security policy blocked camera access.' + httpsHint;
        default:
          return `Camera error: ${e.name}.` + httpsHint;
      }
    }
    return 'Unable to open the camera.' + httpsHint;
  }

  async function startCamera() {
    if (streaming) return;
    try {
      toggleCameraBtn.disabled = true;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      streaming = true;
      toggleCameraBtn.disabled = false;
      toggleCameraBtn.textContent = 'Turn off camera';
      setPill('Connected', 'ok');
      setStatus('Camera is on');
      overlayRAF = requestAnimationFrame(drawOverlay);
      frameRAF = requestAnimationFrame(tickSend);
    } catch (e) {
      toggleCameraBtn.disabled = false;
      setPill('Camera error', 'err');
      setStatus(explainGetUserMediaError(e));
      setDebug('getUserMedia error: ' + (e && e.message ? e.message : e));
      console.error(e);
    }
  }

  function stopCamera() {
    streaming = false;
    if (frameRAF) cancelAnimationFrame(frameRAF), (frameRAF = null);
    if (overlayRAF) cancelAnimationFrame(overlayRAF), (overlayRAF = null);
    const so = video.srcObject;
    if (so) {
      so.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    toggleCameraBtn.textContent = 'Turn on camera';
    setPill('Off', 'warn');
    setStatus('Camera is off');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // ---------- WS frame loop ----------
  function sendFrame() {
    if (!ws || ws.readyState !== 1 || !streaming) return;
    const cap = document.createElement('canvas');
    cap.width = video.videoWidth || canvas.width;
    cap.height = video.videoHeight || canvas.height;
    const cctx = cap.getContext('2d');
    cctx.drawImage(video, 0, 0, cap.width, cap.height);
    srcW = cap.width;
    srcH = cap.height;
    const dataURL = cap.toDataURL('image/jpeg', 0.8);
    try {
      ws.send(dataURL);
    } catch (e) {
      setDebug('WS send error: ' + e);
    }
  }

  function tickSend(ts) {
    if (!streaming) return;
    if (!lastSend || ts - lastSend >= SEND_INTERVAL_MS) {
      sendFrame();
      lastSend = ts || performance.now();
    }
    frameRAF = requestAnimationFrame(tickSend);
  }

  // ---------- WebSocket ----------
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;
    setDebug('Connecting WS: ' + url);
    try {
      ws = new WebSocket(url);
      ws.onopen = () => {
        setPill('WS connected', 'ok');
        setDebug('WS open');
      };
      ws.onclose = () => {
        setPill('WS disconnected', 'err');
        setDebug('WS close');
      };
      ws.onerror = (ev) => {
        setPill('WS error', 'err');
        setDebug('WS error');
        console.error(ev);
      };
      ws.onmessage = (ev) => {
        let m;
        try {
          m = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (m.type === 'frame_result') {
          if (Array.isArray(m.locs)) lastLocs = m.locs;
          recogMap = m.data || {};
          if (m.meta && typeof m.meta.threshold === 'number') {
            // Optionally show in debug, if needed.
            // setDebug(`Threshold: ${m.meta.threshold}`);
          }
        } else if (m.type === 'config') {
          // Server broadcast config updates
          const cfg = m.data || {};
          if ('show_bbox' in cfg) {
            overlayOn = !!cfg.show_bbox;
            if (overlayChk) overlayChk.checked = overlayOn;
          }
        } else if (m.type === 'infer_status') {
          // Could toggle a spinner here
        }
      };
    } catch (e) {
      setDebug('WS connect exception: ' + e);
    }
  }

  // ---------- UI bindings ----------
  toggleCameraBtn.addEventListener('click', () => {
    if (streaming) stopCamera();
    else startCamera();
  });

  overlayChk.addEventListener('change', async () => {
    overlayOn = overlayChk.checked;
    try {
      await fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_bbox: overlayOn, show_label: overlayOn }),
      });
      setDebug('Overlay config sent -> ' + overlayOn);
    } catch (e) {
      setDebug('Overlay config error: ' + e);
    }
  });

  reloadFacebankBtn.addEventListener('click', async () => {
    reloadFacebankBtn.disabled = true;
    try {
      const res = await fetch('/reload-facebank', { method: 'POST' });
      const js = await res.json();
      setStatus((js.status === 'success' ? 'Facebank: ' : 'Error: ') + (js.message || ''));
      setDebug('Reload facebank -> ' + (js.status || 'unknown'));
    } catch (e) {
      setStatus('Reload error: ' + e);
      setDebug('Reload error: ' + e);
    } finally {
      reloadFacebankBtn.disabled = false;
    }
  });

  window.addEventListener('beforeunload', () => {
    try { if (ws) ws.close(); } catch {}
    try { stopCamera(); } catch {}
  });

  // ---------- Bootstrap ----------
  (async () => {
    overlayOn = overlayChk ? overlayChk.checked : true;
    setPill('Connecting WS...', 'warn');
    setStatus('Ready.');
    await bootstrapConfig();
    connectWS();
    // draw loop starts when camera is on
    setDebug('Ready. Click "Turn on camera" to start.');
  })();

  async function bootstrapConfig() {
    try {
      const res = await fetch('/config');
      const js = await res.json();
      if (js && js.config) {
        const cfg = js.config;
        if ('show_bbox' in cfg) {
          overlayOn = !!cfg.show_bbox;
          if (overlayChk) overlayChk.checked = overlayOn;
        }
      }
    } catch (e) {
      // ignore
    }
  }
});
