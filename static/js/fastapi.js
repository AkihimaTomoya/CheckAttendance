document.addEventListener('DOMContentLoaded', () => {

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);
  const video             = $('videoStream');
  const canvas            = $('canvasOverlay');
  const ctx               = canvas.getContext('2d');
  const overlayChk        = $('overlayChk');
  const toggleCameraBtn   = $('toggleCameraBtn');
  const reloadFacebankBtn = $('reloadFacebankBtn');
  const connectionStatus  = $('connectionStatus');
  const statusLine        = $('status');
  const debugLine         = $('debugLine');

  // ---------- State ----------
  let ws         = null;
  let streaming  = false;
  let overlayOn  = overlayChk ? overlayChk.checked : true;
  let overlayRAF = null;
  let lastLocs     = [];
  let lastRecogMap = {};

  const SEND_INTERVAL_MS = 120;

  // Canvas tái sử dụng để capture frame — không tạo mới mỗi lần gửi
  let capCanvas = null;
  let capCtx    = null;

  // ---------- UI ----------
  const setPill   = (t, c) => { connectionStatus.textContent = t; connectionStatus.className = 'pill ' + (c || ''); };
  const setStatus = (t, c) => { if (!statusLine) return; statusLine.textContent = t || ''; statusLine.className = 'status-badge' + (c ? ' '+c : ''); statusLine.title = t || ''; };
  const setDebug  = (t, c) => { if (!debugLine)  return; debugLine.textContent  = t || ''; debugLine.className  = 'status-badge debug-badge' + (c ? ' '+c : ''); debugLine.title = t || ''; };

  // ---------- Canvas sizing ----------
  function syncCanvas() {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return false;
    if (canvas.width !== w)  canvas.width  = w;
    if (canvas.height !== h) canvas.height = h;
    return true;
  }

  // ---------- Draw loop ----------
  // Video element hiển thị live feed bình thường.
  // Canvas overlay chỉ vẽ bbox — không vẽ lại frame video.
  function drawOverlay() {
    if (!streaming) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (overlayOn) {
      const faces = lastLocs;
      if (faces.length > 0) {
        setStatus(`Tracking ${faces.length} face(s)`, 'ok');
      } else {
        setStatus('Waiting for frames…', 'warn');
      }

      for (const loc of faces) {
        const [x1, y1, x2, y2] = loc.bbox;
        const rec     = lastRecogMap[loc.id] || {};
        const name    = rec.name || rec.name_top1 || 'Unknown';
        const isKnown = name !== 'Unknown';
        const color   = isKnown ? '#10b981' : '#ef4444';

        ctx.lineWidth   = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.font = '14px ui-sans-serif, system-ui';
        const tw = ctx.measureText(name).width;
        const ly = Math.max(0, y1 - 22);
        ctx.fillStyle = color;
        ctx.fillRect(x1, ly, tw + 10, 22);
        ctx.fillStyle = '#fff';
        ctx.fillText(name, x1 + 5, ly + 16);
      }
    } else {
      setStatus('Camera is on (overlay hidden)', 'warn');
    }

    overlayRAF = requestAnimationFrame(drawOverlay);
  }

  // ---------- Camera ----------
  function explainCameraError(e) {
    const hint = location.hostname !== 'localhost' && location.protocol !== 'https:'
      ? ' (Dùng http://localhost:<port> hoặc HTTPS qua LAN.)' : '';
    if (!e?.name) return 'Không mở được camera.' + hint;
    return ({ NotAllowedError:  'Bạn từ chối quyền camera.' + hint,
              NotFoundError:    'Không tìm thấy camera.',
              NotReadableError: 'Camera đang bị chiếm bởi ứng dụng khác.',
              SecurityError:    'Security policy chặn camera.' + hint })[e.name]
           ?? `Lỗi camera: ${e.name}.` + hint;
  }

  async function startCamera() {
    if (streaming) return;
    toggleCameraBtn.disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();

      if (!video.videoWidth) {
        await new Promise(res => video.addEventListener('loadedmetadata', res, { once: true }));
      }

      syncCanvas();

      // Khởi tạo capture canvas một lần duy nhất, tái sử dụng mãi
      capCanvas = document.createElement('canvas');
      capCanvas.width  = video.videoWidth;
      capCanvas.height = video.videoHeight;
      capCtx = capCanvas.getContext('2d', { willReadFrequently: false });

      streaming = true;
      toggleCameraBtn.disabled = false;
      toggleCameraBtn.textContent = 'Turn off camera';
      setPill('Connected', 'ok');
      setStatus('Waiting for frames…', 'warn');

      overlayRAF = requestAnimationFrame(drawOverlay);
      // Dùng setInterval thay vì RAF để send frame đúng nhịp 120ms
      // RAF bị throttle khi tab mất focus; setInterval vẫn chạy
      startSendLoop();
    } catch (e) {
      toggleCameraBtn.disabled = false;
      setPill('Camera error', 'err');
      setStatus(explainCameraError(e), 'err');
      setDebug('getUserMedia: ' + (e?.message ?? e), 'err');
    }
  }

  function stopCamera() {
    streaming = false;
    if (overlayRAF) cancelAnimationFrame(overlayRAF), (overlayRAF = null);
    stopSendLoop();
    video.srcObject?.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    capCanvas = null; capCtx = null;
    lastLocs = []; lastRecogMap = {};
    toggleCameraBtn.textContent = 'Turn on camera';
    setPill('Off', 'warn');
    setStatus('Camera is off', 'warn');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // ---------- WS send loop ----------
  let _sendTimer = null;

  function startSendLoop() {
    if (_sendTimer) return;
    _sendTimer = setInterval(sendFrame, SEND_INTERVAL_MS);
  }

  function stopSendLoop() {
    if (_sendTimer) clearInterval(_sendTimer), (_sendTimer = null);
  }

  function sendFrame() {
    if (!ws || ws.readyState !== 1 || !streaming || !video.videoWidth || !capCtx) return;
    // Tái sử dụng capCanvas — không allocate DOM element mới mỗi frame
    capCtx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
    try { ws.send(capCanvas.toDataURL('image/jpeg', 0.75)); }
    catch (e) { setDebug('WS send error: ' + e, 'err'); }
  }

  // ---------- WebSocket ----------
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url   = `${proto}//${location.host}/ws`;
    setDebug('Connecting: ' + url, 'warn');
    try {
      ws = new WebSocket(url);
      ws.onopen  = ()  => { setPill('WS connected', 'ok');     setDebug('WS open', 'ok'); };
      ws.onclose = ()  => { setPill('WS disconnected', 'err'); setDebug('WS closed', 'err'); };
      ws.onerror = (e) => { setPill('WS error', 'err');         setDebug('WS error', 'err'); console.error(e); };
      ws.onmessage = (ev) => {
        let m;
        try { m = JSON.parse(ev.data); } catch { return; }
        if (m.type === 'frame_result') {
          lastLocs     = Array.isArray(m.locs) ? m.locs : [];
          lastRecogMap = m.data || {};
        } else if (m.type === 'config') {
          const cfg = m.data || {};
          if ('show_bbox' in cfg) {
            overlayOn = !!cfg.show_bbox;
            if (overlayChk) overlayChk.checked = overlayOn;
          }
        }
      };
    } catch (e) {
      setDebug('WS exception: ' + e, 'err');
    }
  }

  // ---------- UI bindings ----------
  toggleCameraBtn.addEventListener('click', () => streaming ? stopCamera() : startCamera());

  overlayChk.addEventListener('change', async () => {
    overlayOn = overlayChk.checked;
    try {
      await fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_bbox: overlayOn, show_label: overlayOn }),
      });
    } catch (e) { setDebug('Config error: ' + e, 'err'); }
  });

  reloadFacebankBtn.addEventListener('click', async () => {
    reloadFacebankBtn.disabled = true;
    try {
      const res = await fetch('/reload-facebank', { method: 'POST' });
      const js  = await res.json();
      setStatus((js.status === 'success' ? 'Facebank: ' : 'Error: ') + (js.message || ''),
                js.status === 'success' ? 'ok' : 'err');
    } catch (e) { setStatus('Reload error: ' + e, 'err'); }
    finally { reloadFacebankBtn.disabled = false; }
  });

  window.addEventListener('beforeunload', () => {
    try { ws?.close(); }  catch {}
    try { stopCamera(); } catch {}
  });

  // ---------- Bootstrap ----------
  (async () => {
    setPill('Connecting WS...', 'warn');
    setStatus('Ready.', 'warn');
    try {
      const res = await fetch('/config');
      const js  = await res.json();
      if (js?.config && 'show_bbox' in js.config) {
        overlayOn = !!js.config.show_bbox;
        if (overlayChk) overlayChk.checked = overlayOn;
      }
    } catch {}
    connectWS();
    setDebug('Ready. Click "Turn on camera" to start.', 'warn');
  })();

});