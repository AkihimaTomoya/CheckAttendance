document.addEventListener('DOMContentLoaded', () => {
  const user = ATT.requireSession('student');
  if (!user) return;
  document.getElementById('navStudentName').textContent = `${user.display_name} (${user.uid})`;

  let isCheckedIn = false, checkInTime = null;
  let faceVisible = false, lastFaceSeen = Date.now();
  let totalAbsentMs = 0, absenceStartTime = 0;

  // ─── 1. XỬ LÝ ĐÓNG TAB (GHI LOG CHECK_OUT & RỜI LỚP) ───
  window.addEventListener('beforeunload', () => {
    if (isCheckedIn || streaming) {
      const now = Date.now();
      const currentAbs = (!faceVisible && absenceStartTime > 0) ? (now - absenceStartTime) : 0;

      // Trạng thái 'left' -> Đã rời lớp
      ATT.writeRecord(user.uid, { status: 'left', check_out: now, last_seen: now });

      const payload = JSON.stringify({
        student_id: user.uid, display_name: user.display_name, face_name: user.face_name,
        event: 'check_out', total_absent_sec: Math.floor((totalAbsentMs + currentAbs)/1000)
      });
      navigator.sendBeacon('/log-entry', new Blob([payload], { type: 'application/json' }));
    }
  });

  // ─── 2. XỬ LÝ NÚT ĐĂNG XUẤT (GHI LOG CHECK_OUT) ───
  document.getElementById('logoutBtn').addEventListener('click', () => {
    if (isCheckedIn || streaming) {
      const now = Date.now();
      const currentAbs = (!faceVisible && absenceStartTime > 0) ? (now - absenceStartTime) : 0;

      ATT.logEntry({ student_id: user.uid, display_name: user.display_name, face_name: user.face_name, event: 'check_out', total_absent_sec: Math.floor((totalAbsentMs + currentAbs)/1000) });
      ATT.writeRecord(user.uid, { status: 'left', check_out: now, last_seen: now });
    }
    ATT.clearSession(); location.href = '/';
  });

  const video = document.getElementById('videoStream');
  const canvas = document.getElementById('canvasOverlay');
  const ctx = canvas.getContext('2d');
  const toggleCamBtn = document.getElementById('toggleCameraBtn');
  const overlayChk = document.getElementById('overlayChk');
  const $attStatus = document.getElementById('attStatus');
  const $attDuration = document.getElementById('attDuration');

  let ws = null, streaming = false, overlayOn = true;
  let capCanvas = document.createElement('canvas');
  let capCtx = capCanvas.getContext('2d');
  let sendTimer = null, overlayRAF = null;
  let lastLocs = [], lastRecogMap = {};

  function updateState(foundFace) {
    const now = Date.now();
    if (foundFace) {
      lastFaceSeen = now;
      if (!faceVisible) {
        faceVisible = true;
        if (absenceStartTime > 0) { totalAbsentMs += (now - absenceStartTime); absenceStartTime = 0; }
        if (isCheckedIn) ATT.logEntry({ student_id: user.uid, display_name: user.display_name, face_name: user.face_name, event: 'face_found', total_absent_sec: Math.floor(totalAbsentMs/1000) });
      }
      if (!isCheckedIn) {
        isCheckedIn = true; checkInTime = now; faceVisible = true;
        // Lần đầu tiên thấy mặt -> check_in
        ATT.logEntry({ student_id: user.uid, display_name: user.display_name, face_name: user.face_name, event: 'check_in', total_absent_sec: 0 });
      }
    } else {
      if (faceVisible && (now - lastFaceSeen > 5000)) {
        faceVisible = false; absenceStartTime = now - 5000;
        ATT.logEntry({ student_id: user.uid, display_name: user.display_name, face_name: user.face_name, event: 'face_lost', total_absent_sec: Math.floor(totalAbsentMs/1000) });
      }
    }

    if (isCheckedIn) {
      const curAbs = (!faceVisible && absenceStartTime > 0) ? (Date.now() - absenceStartTime) : 0;
      ATT.writeRecord(user.uid, { display_name: user.display_name, face_name: user.face_name, status: faceVisible ? 'present' : 'absent', check_in: checkInTime, last_seen: lastFaceSeen, total_absent_ms: totalAbsentMs + curAbs });
    }
  }

  setInterval(() => {
    if (isCheckedIn) {
      const now = Date.now();
      const curAbs = (!faceVisible && absenceStartTime > 0) ? (now - absenceStartTime) : 0;
      $attDuration.innerHTML = `Đã học: <b>${ATT.fmtDur(now - checkInTime)}</b> | Tổng vắng: <b><span style="color:red">${ATT.fmtDur(totalAbsentMs + curAbs)}</span></b>`;
      $attStatus.textContent = faceVisible ? "🟢 Đang trên hình" : "🔴 Vắng mặt";

      // Heartbeat sync -> Admin thấy Realtime
      ATT.writeRecord(user.uid, { display_name: user.display_name, face_name: user.face_name, status: faceVisible ? 'present' : 'absent', check_in: checkInTime, last_seen: lastFaceSeen, total_absent_ms: totalAbsentMs + curAbs });
    }
  }, 1000);

  function drawOverlay() {
    if (!streaming) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (overlayOn) {
      for (const loc of lastLocs) {
        const rec = lastRecogMap[loc.id] || {};
        const detectedName = rec.name || rec.name_top1 || 'Unknown';

        let color = '#ef4444';
        let label = detectedName;

        if (rec.passed_threshold && detectedName === user.face_name) {
            color = '#10b981';
        } else if (rec.passed_threshold && detectedName !== 'Unknown') {
            color = '#f59e0b';
            label = `Sai người (${detectedName})`;
        }

        ctx.lineWidth = 2; ctx.strokeStyle = color;
        ctx.strokeRect(loc.bbox[0], loc.bbox[1], loc.bbox[2]-loc.bbox[0], loc.bbox[3]-loc.bbox[1]);

        ctx.font = '14px ui-sans-serif, system-ui';
        const tw = ctx.measureText(label).width;
        const ly = Math.max(0, loc.bbox[1] - 22);

        ctx.fillStyle = color; ctx.fillRect(loc.bbox[0], ly, tw + 10, 22);
        ctx.fillStyle = '#fff'; ctx.fillText(label, loc.bbox[0] + 5, ly + 16);
      }
    }
    overlayRAF = requestAnimationFrame(drawOverlay);
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream; await video.play();
      canvas.width = capCanvas.width = video.videoWidth; canvas.height = capCanvas.height = video.videoHeight;
      streaming = true; toggleCamBtn.textContent = "Tắt Camera";

      // ─── 3. LOG CAMERA_ON ───
      ATT.logEntry({ student_id: user.uid, display_name: user.display_name, face_name: user.face_name, event: 'camera_on', total_absent_sec: Math.floor(totalAbsentMs/1000) });

      overlayRAF = requestAnimationFrame(drawOverlay);
      ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);

      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          lastLocs = m.locs || []; lastRecogMap = m.data || {};
          const found = Object.values(lastRecogMap).some(r => r.passed_threshold && r.name === user.face_name);
          updateState(found);
        } catch {}
      };

      sendTimer = setInterval(() => {
        if(ws?.readyState === 1 && streaming) { capCtx.drawImage(video, 0, 0); ws.send(capCanvas.toDataURL('image/jpeg', 0.7)); }
      }, 150);
    } catch (e) { alert("Lỗi camera! Vui lòng cấp quyền."); }
  }

  toggleCamBtn.addEventListener('click', () => {
    if (streaming) {
      streaming = false; toggleCamBtn.textContent = "Bật Camera";
      video.srcObject?.getTracks().forEach(t => t.stop()); video.srcObject = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      clearInterval(sendTimer); if (overlayRAF) cancelAnimationFrame(overlayRAF);
      if (ws) { ws.close(); ws = null; }

      // ─── 4. LOG CAMERA_OFF ───
      if (isCheckedIn && faceVisible) {
        faceVisible = false; absenceStartTime = Date.now();
      }
      ATT.logEntry({ student_id: user.uid, display_name: user.display_name, face_name: user.face_name, event: 'camera_off', total_absent_sec: Math.floor(totalAbsentMs/1000) });

    } else { startCamera(); }
  });

  if (overlayChk) overlayChk.addEventListener('change', () => overlayOn = overlayChk.checked);
});