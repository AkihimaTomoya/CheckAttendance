document.addEventListener('DOMContentLoaded', async () => {
  const user = ATT.requireSession('admin');
  if (!user) return;
  document.getElementById('logoutBtn').addEventListener('click', () => { ATT.clearSession(); location.href = '/'; });

  let usersDb = {};
  let students = [];

  async function loadUsers() {
    usersDb = await ATT.fetchUsers();
    students = Object.entries(usersDb).filter(([, u]) => u.role === 'student');
  }
  await loadUsers();

  function renderTable() {
    const records = ATT.readRecords();
    const tbody = document.getElementById('attendanceBody');
    if(!tbody) return;
    tbody.innerHTML = '';
    const now = Date.now();

    students.forEach(([uid, u]) => {
      const r = records[uid] || {};
      const status = r.status || 'not_joined';

      // Phân tích trạng thái
      const isAbsent = status === 'absent';
      const hasLeft = status === 'left';

      // Chỉ đếm "Đang vắng" nếu status thực sự là absent (đang ở trong lớp nhưng không thấy mặt/tắt cam)
      // Nếu đã "left" (Rời lớp) thì không đếm "Đang vắng" nữa
      const currentAbsMs = (isAbsent && r.last_seen) ? (now - r.last_seen) : 0;
      const totalAbsStr = ATT.fmtDur(r.total_absent_ms || 0);
      const currAbsStr = isAbsent ? `<br><span style="color:red; font-size:11px">Đang vắng: ${ATT.fmtDur(currentAbsMs)}</span>` : '';

      const badges = {
        present: '<span class="adm-badge adm-present">● Có mặt</span>',
        absent: '<span class="adm-badge adm-absent">● Vắng mặt</span>',
        left: '<span class="adm-badge adm-left" style="background:#f3f4f6; color:#6b7280; border: 1px solid #e5e7eb">○ Đã rời lớp</span>',
        not_joined: '<span class="adm-badge adm-idle">◌ Chưa vào</span>'
      };

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>${u.display_name}</b><br><small>${uid} | Face: ${u.face_name}</small></td>
        <td>${badges[status] || badges.not_joined} ${currAbsStr}</td>
        <td>${ATT.fmtTime(r.check_in)}</td>
        <td style="color:red; font-weight:bold">${totalAbsStr}</td>
        <td><button class="btn btn-delete btn-text-danger" data-uid="${uid}" style="padding: 4px 8px;">Xóa</button></td>
      `;
      tbody.appendChild(tr);
    });
  }

  setInterval(renderTable, 1000);

  // Xóa Sinh viên
  document.getElementById('attendanceBody').addEventListener('click', async (e) => {
    if (e.target.classList.contains('btn-delete')) {
      const uid = e.target.getAttribute('data-uid');
      if (confirm(`Bạn có chắc chắn muốn xóa [${uid}]?`)) {
        const res = await (await fetch(`/api/users/${uid}`, { method: 'DELETE' })).json();
        if (res.status === 'ok') { alert(res.message); await loadUsers(); }
        else { alert(`Lỗi: ${res.message}`); }
      }
    }
  });

  // Thêm Sinh viên
  document.getElementById('addStudentForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const msgBox = document.getElementById('addResultMsg');
    msgBox.textContent = "Đang xử lý...";

    const newStudent = {
      uid: document.getElementById('newUid').value,
      display_name: document.getElementById('newName').value,
      face_name: document.getElementById('newFaceName').value,
      password: document.getElementById('newPassword').value
    };

    const res = await (await fetch('/api/users', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newStudent)
    })).json();

    if (res.status === 'ok') {
      msgBox.textContent = "✅ " + res.message; msgBox.style.color = "var(--ok)";
      e.target.reset(); await loadUsers();
    } else {
      msgBox.textContent = "❌ " + res.message; msgBox.style.color = "var(--err)";
    }
  });

  // Reload Facebank
  document.getElementById('reloadFacebankBtn')?.addEventListener('click', async (e) => {
    const btn = e.target;
    const msgBox = document.getElementById('fbResultMsg');
    btn.disabled = true; btn.textContent = "Đang tải lại..."; msgBox.textContent = "";

    try {
      const res = await (await fetch('/api/reload-facebank', { method: 'POST' })).json();
      if (res.status === 'ok') {
        msgBox.textContent = "✅ " + res.message; msgBox.style.color = "var(--ok)";
      } else {
        msgBox.textContent = "❌ " + res.message; msgBox.style.color = "var(--err)";
      }
    } catch (err) {
      msgBox.textContent = "❌ Lỗi kết nối server!"; msgBox.style.color = "var(--err)";
    } finally {
      btn.disabled = false; btn.textContent = "Tải lại Facebank";
    }
  });
});