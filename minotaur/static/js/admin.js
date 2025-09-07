// admin.js — Admin modal + diagnostics helpers
;(function () {
  function openModal() {
    const m = document.getElementById('modal');
    if (!m) return;
    m.classList.remove('hidden');
    fetch('/minotaur/settings', { credentials: 'same-origin' })
      .then(r => r.text())
      .then(html => { const b = document.getElementById('modal-body'); if (b) b.innerHTML = html; })
      .catch(() => { const b = document.getElementById('modal-body'); if (b) b.innerText = '読み込みに失敗しました'; });
  }
  function closeModal() {
    const m = document.getElementById('modal');
    if (m) m.classList.add('hidden');
  }
  function submitSettings(ev) {
    ev.preventDefault();
    const form = ev.target;
    const fd = new FormData(form);
    fetch('/minotaur/settings', { method: 'POST', body: fd, credentials: 'same-origin' })
      .then(() => {
        // Reflect OFFICIAL_BASE change in header immediately
        try {
          const ob = (fd.get('OFFICIAL_BASE') || '').toString();
          const el = document.getElementById('official-base-text');
          if (el) el.textContent = ob ? ob : '(mock)';
        } catch (_) {}
        closeModal();
      })
      .catch(() => { closeModal(); })
      .finally(() => { try { window.loadStatus && window.loadStatus(); } catch (_) {} });
    return false;
  }
  function downloadDB() {
    window.location.href = '/minotaur/download_db';
    return false;
  }

  // Expose for inline onclicks
  window.openModal = openModal;
  window.closeModal = closeModal;
  window.submitSettings = submitSettings;
  window.downloadDB = downloadDB;
})();
