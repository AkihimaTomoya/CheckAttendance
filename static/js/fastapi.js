document.addEventListener('DOMContentLoaded', () => {
  const $ = id => document.getElementById(id);

  const video = $('videoStream');
  const canvas = $('canvasOverlay');
  const ctx = canvas.getContext('2d');

  const connectionStatus = $('connectionStatus');
  const serverThrEl = $('serverThreshold');
  const faceCountEl = $('faceCount');
  const statusLine = $('status');

  const stopBtn = $('stopBtn');
  const reloadFacebankBtn = $('reloadFacebankBtn');

  const thrRange = $('thrRange'), thrNum = $('thrNum'), applyThrBtn = $('applyThrBtn');
  const ttaChk = $('ttaChk'), top1Chk = $('top1Chk'), refreshRuntimeBtn = $('refreshRuntimeBtn');

  const resultPre = $('resultPre'), runtimePre = $('runtimePre');
  const serverDebugPre = $('serverDebugPre'), facebankDebugPre = $('facebankDebugPre');
  const refreshDebugBtn = $('refreshDebugBtn'), refreshFacebankBtn = $('refreshFacebankBtn');

  let ws = null, streaming = false, sendTimer = null;
  let lastLocs = [], recogMap = {}, cfg = { threshold: null };
  // kích thước nguồn dùng để scale bbox -> canvas hiển thị
  let srcW = null, srcH = null;

  const setPill = (t,c) => { connectionStatus.textContent=t; connectionStatus.className='pill '+(c||''); };
  const setStatus = (t) => { statusLine.textContent = t; };
  const fetchJSON = async (u,o={}) => (await fetch(u,o)).json();

  function fitCanvas(){
    const w = video.videoWidth || video.clientWidth || 800;
    const h = video.videoHeight|| video.clientHeight|| 600;
    canvas.width = w; canvas.height = h;
  }

  function drawOverlay(){
    if (!streaming) return;
    fitCanvas();
    ctx.clearRect(0,0,canvas.width,canvas.height);

    const faces = Array.isArray(lastLocs) ? lastLocs : [];
    faceCountEl.textContent = faces.length;
    setStatus(faces.length>0 ? `Nhận diện ${faces.length} khuôn mặt` : 'Chưa phát hiện khuôn mặt');

    const sw = (typeof srcW==='number' && srcW>0) ? srcW : canvas.width;
    const sh = (typeof srcH==='number' && srcH>0) ? srcH : canvas.height;
    const sx = canvas.width / sw;
    const sy = canvas.height/ sh;

    faces.forEach(loc => {
      let [x1,y1,x2,y2] = loc.bbox;
      // scale bbox theo kích thước hiển thị
      x1 = Math.round(x1*sx); y1 = Math.round(y1*sy);
      x2 = Math.round(x2*sx); y2 = Math.round(y2*sy);

      const rec = recogMap[loc.id] || {};
      const useTop1 = !!top1Chk.checked;
      const name = (useTop1 && rec.name_top1) ? rec.name_top1 : (rec.name || 'Unknown');
      const dist = (typeof rec.distance === 'number') ? rec.distance : null;
      const thr  = (typeof cfg.threshold === 'number') ? cfg.threshold : (rec.threshold || 1.56);
      const pass = (typeof dist === 'number') ? (dist < thr) : false;

      // Dùng đúng 3 tông: pass = #198cf0, not-pass = hsl(201,97%,72%), text = #fff
      const color = pass ? '#198cf0' : 'hsl(201, 97%, 72%)';

      ctx.lineWidth = 2; ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);

      const label = `${name} — d=${dist==null?'—':dist.toFixed(3)}`;
      ctx.font = '14px ui-sans-serif, system-ui';
      const tw = ctx.measureText(label).width;
      const y = Math.max(0, y1-22);
      ctx.fillStyle = color; ctx.fillRect(x1, y, tw+10, 22);
      ctx.fillStyle = '#fff'; ctx.fillText(label, x1+5, y+16);
    });

    requestAnimationFrame(drawOverlay);
  }

  async function startCamera(){
    if (streaming) return;
    try{
      const stream = await navigator.mediaDevices.getUserMedia({ video:{width:{ideal:1280}, height:{ideal:720}}, audio:false });
      video.srcObject = stream; await video.play();
      streaming = true; setPill('Đã kết nối','ok'); setStatus('Camera đã bật');
      connectWS(); kickLoops(); requestAnimationFrame(drawOverlay);
    }catch(e){
      setPill('Lỗi camera','err'); setStatus('Không mở được camera. Hãy cấp quyền.');
    }
  }
  function stopCamera(){
    streaming = false;
    if (sendTimer) { clearInterval(sendTimer); sendTimer=null; }
    if (ws){ try{ ws.close(); }catch{} ws=null; }
    const so = video.srcObject; if (so){ so.getTracks().forEach(t=>t.stop()); video.srcObject=null; }
    setPill('Đã tắt','warn'); setStatus('Camera đã tắt');
    faceCountEl.textContent = '0'; lastLocs=[]; recogMap={}; ctx.clearRect(0,0,canvas.width,canvas.height);
  }
  function kickLoops(){
    if (sendTimer) clearInterval(sendTimer);
    sendTimer = setInterval(sendFrame, 120);
  }
  function sendFrame(){
    if (!ws || ws.readyState!==1 || !streaming) return;
    const cap = document.createElement('canvas');
    cap.width = video.videoWidth || canvas.width;
    cap.height= video.videoHeight|| canvas.height;
    cap.getContext('2d').drawImage(video,0,0,cap.width,cap.height);
    const dataURL = cap.toDataURL('image/jpeg', 0.82);
    // lưu srcW/srcH local để scale bbox; tiếp tục gửi base64 (tương thích server hiện tại)
    srcW = cap.width; srcH = cap.height;
    ws.send(dataURL);
    setPill('Đang xử lý…','warn');
  }

  function connectWS(){
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => setPill('Kết nối WS','ok');
    ws.onclose = () => setPill('Mất kết nối WS','err');
    ws.onerror = () => setPill('Lỗi WS','err');

    ws.onmessage = ev => {
      let m; try{ m = JSON.parse(ev.data); }catch{ return; }
      if (m.type==='frame_result'){
        if (Array.isArray(m.locs)) lastLocs = m.locs;
        recogMap = m.data || {};
        // nếu server có trả meta src_w/h thì override (không bắt buộc)
        if (m.meta){
          if (typeof m.meta.threshold==='number'){
            cfg.threshold = m.meta.threshold;
            serverThrEl.textContent = cfg.threshold.toFixed(2);
            thrRange.value = cfg.threshold.toFixed(2);
            thrNum.value = cfg.threshold.toFixed(2);
          }
          if (typeof m.meta.src_w==='number') srcW = m.meta.src_w;
          if (typeof m.meta.src_h==='number') srcH = m.meta.src_h;
        }
        if (resultPre) resultPre.textContent = JSON.stringify({locs:lastLocs, data:recogMap}, null, 2);
        setPill('Xử lý xong','ok');
      } else if (m.type==='config' && m.data && typeof m.data.threshold==='number'){
        cfg.threshold = m.data.threshold;
        serverThrEl.textContent = cfg.threshold.toFixed(2);
      }
    };
  }

  // API helpers
  async function fetchServerDebug(){ try{ serverDebugPre.textContent = JSON.stringify(await fetchJSON('/debug'), null, 2); }catch(e){ serverDebugPre.textContent='Error: '+e; } }
  async function fetchFacebankInfo(){ try{ facebankDebugPre.textContent = JSON.stringify(await fetchJSON('/facebank-info'), null, 2); }catch(e){ facebankDebugPre.textContent='Error: '+e; } }
  async function fetchRuntimeConfig(){
    try {
      const j = await fetchJSON('/runtime-config');
      runtimePre.textContent = JSON.stringify(j, null, 2);
      if (j.status==='success'){
        const d = j.data || {};
        if (typeof d.threshold==='number'){
          cfg.threshold = d.threshold;
          serverThrEl.textContent = d.threshold.toFixed(2);
          thrRange.value = d.threshold.toFixed(2);
          thrNum.value = d.threshold.toFixed(2);
        }
        if (typeof d.tta==='boolean') ttaChk.checked = d.tta;
        if (typeof d.debug_top1==='boolean') top1Chk.checked = d.debug_top1;
      }
    } catch(e){ runtimePre.textContent = 'Error: '+e; }
  }
  async function setThreshold(v){ try{ await fetchJSON('/set-threshold',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({threshold:v})}); await fetchRuntimeConfig(); }catch{} }
  async function setTTA(on){ try{ await fetchJSON('/set-tta',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({tta:!!on})}); await fetchRuntimeConfig(); }catch{} }
  async function setDebugTop1(on){ try{ await fetchJSON('/set-debug-top1',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({debug_top1:!!on})}); await fetchRuntimeConfig(); }catch{} }

  // Bind UI
  stopBtn.onclick = stopCamera;
  reloadFacebankBtn.onclick = async () => {
    reloadFacebankBtn.disabled = true;
    try{ const j = await fetchJSON('/reload-facebank', {method:'POST'});
      statusLine.textContent = (j.status==='success' ? j.message : ('Error: '+j.message));
      await fetchFacebankInfo(); await fetchServerDebug();
    } finally { reloadFacebankBtn.disabled = false; }
  };
  thrRange.oninput = () => { thrNum.value = thrRange.value; };
  thrNum.oninput   = () => { thrRange.value = thrNum.value; };
  applyThrBtn.onclick = async () => {
    const v = parseFloat(thrNum.value); if (isFinite(v)) await setThreshold(v);
  };
  ttaChk.onchange = async () => { await setTTA(ttaChk.checked); };
  top1Chk.onchange = async () => { await setDebugTop1(top1Chk.checked); };
  refreshRuntimeBtn.onclick = fetchRuntimeConfig;
  refreshDebugBtn.onclick = fetchServerDebug;
  refreshFacebankBtn.onclick = fetchFacebankInfo;

  // Boot
  setPill('Đang khởi tạo…','warn');
  fetchServerDebug(); fetchFacebankInfo(); fetchRuntimeConfig();
  startCamera();
});
