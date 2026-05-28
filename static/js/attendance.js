const ATT = (() => {
  let _cfg = {};
  fetch('/attendance-config').then(r => r.json()).then(d => _cfg = d).catch(()=>{});

  return {
    getConfig: () => _cfg,
    login: async (u, p) => {
      try {
        const r = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: u, password: p }) });
        return await r.json();
      } catch (e) { return { status: 'error', message: 'Mất kết nối server' }; }
    },
    fetchUsers: async () => { try { return await (await fetch('/api/users')).json(); } catch { return {}; } },
    logEntry: async (data) => { try { await fetch('/log-entry', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }); } catch{} },

    readRecords: () => JSON.parse(localStorage.getItem('att_records') || '{}'),
    writeRecord: (uid, data) => {
      const r = JSON.parse(localStorage.getItem('att_records') || '{}');
      r[uid] = { ...r[uid], ...data };
      localStorage.setItem('att_records', JSON.stringify(r));
    },

    setSession: (u) => sessionStorage.setItem('att_u', JSON.stringify(u)),
    getSession: () => JSON.parse(sessionStorage.getItem('att_u') || 'null'),
    clearSession: () => sessionStorage.removeItem('att_u'),
    requireSession: (role) => {
      const u = JSON.parse(sessionStorage.getItem('att_u') || 'null');
      if (!u || (role && u.role !== role)) { location.href = '/'; return null; }
      return u;
    },

    fmtTime: (ts) => ts ? new Date(ts).toLocaleTimeString('vi-VN', {hour12: false}) : '--:--',
    fmtDur: (ms) => {
      if (!ms || ms <= 0) return '0p';
      const s = Math.floor(ms / 1000);
      return `${Math.floor(s / 60)}p ${s % 60}s`;
    }
  };
})();