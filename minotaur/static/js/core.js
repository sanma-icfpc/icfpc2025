// core.js — SSE, auto-update, identicons, navbar helpers, init glue
;(function () {
  let autoUpdateEnabled = true;
  let es = null;

  function updateIdenticons(scope) {
    const run = () => {
      try {
        if (typeof window.jdenticon === 'function') {
          window.jdenticon();
        } else if (window.jdenticon && typeof window.jdenticon.update === 'function') {
          if (scope) window.jdenticon.update(scope); else window.jdenticon.update();
        }
      } catch (_) {}
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run, { once: true }); else requestAnimationFrame(() => setTimeout(run, 0));
  }

  function setSSEStatus(isOnline) {
    const el = document.getElementById('sse-status');
    if (!el) return;
    if (isOnline) { el.textContent = '🟢 ONLINE'; el.className = 'badge badge-online ml-2'; }
    else { el.textContent = '🔴 OFFLINE'; el.className = 'badge badge-offline ml-2'; }
  }

  function startSSE() {
    if (!autoUpdateEnabled) return;
    stopSSE();
    try {
      es = new EventSource('/minotaur/stream');
      setSSEStatus(true);
      es.addEventListener('open', function () { if (autoUpdateEnabled) setSSEStatus(true); });
      es.onopen = function () { if (autoUpdateEnabled) setSSEStatus(true); };
      es.addEventListener('change', function () {
        if (!autoUpdateEnabled) return;
        try {
          const recentTab = document.getElementById('tab-recent');
          const recentVisible = !!(recentTab && !recentTab.classList.contains('hidden'));
          if (!recentVisible) { window.loadStatus && window.loadStatus({ skipRecent: true }); }
        } catch (_) {}
        try { window.updateAgentCount && window.updateAgentCount(); } catch (_) {}
        try { window.refreshPriorityStatuses && window.refreshPriorityStatuses(); } catch (_) {}
        try { window.refreshOfficialBase && window.refreshOfficialBase(); } catch (_) {}
        try { window.refreshAnalyticsIfVisible && window.refreshAnalyticsIfVisible(); } catch (_) {}
        try { window.refreshDBInfoIfVisible && window.refreshDBInfoIfVisible(); } catch (_) {}
        try { window.refreshThreadsIfVisible && window.refreshThreadsIfVisible(); } catch (_) {}
        try { window.refreshMemInfoIfVisible && window.refreshMemInfoIfVisible(); } catch (_) {}
      });
      es.onerror = function () { if (autoUpdateEnabled) setSSEStatus(false); };
    } catch (_) {}
  }
  function stopSSE() { try { if (es) es.close(); } catch (_) {} es = null; }

  function updateAutoToggleUI() {
    const t = document.getElementById('auto-toggle');
    if (!t) return;
    if (autoUpdateEnabled) { t.classList.add('on'); t.setAttribute('aria-checked', 'true'); }
    else { t.classList.remove('on'); t.setAttribute('aria-checked', 'false'); }
  }
  function toggleAutoUpdate() {
    autoUpdateEnabled = !autoUpdateEnabled;
    updateAutoToggleUI();
    if (autoUpdateEnabled) {
      startSSE();
      try { window.loadStatus && window.loadStatus(); } catch (_) {}
      try { window.updateAgentCount && window.updateAgentCount(); } catch (_) {}
      try { window.refreshPriorityStatuses && window.refreshPriorityStatuses(); } catch (_) {}
      try { window.refreshAnalyticsIfVisible && window.refreshAnalyticsIfVisible(); } catch (_) {}
      try { window.refreshDBInfoIfVisible && window.refreshDBInfoIfVisible(); } catch (_) {}
      try { window.refreshThreadsIfVisible && window.refreshThreadsIfVisible(); } catch (_) {}
      try { window.refreshMemInfoIfVisible && window.refreshMemInfoIfVisible(); } catch (_) {}
    } else { stopSSE(); setSSEStatus(false); }
  }

  function _cleanupSSE() { try { stopSSE(); } catch (_) {} }
  window.addEventListener('beforeunload', _cleanupSSE);
  window.addEventListener('pagehide', _cleanupSSE);
  document.addEventListener('visibilitychange', function () {
    try { if (document.hidden) stopSSE(); else startSSE(); } catch (_) {}
  });

  // Expose
  window.updateIdenticons = updateIdenticons;
  window.setSSEStatus = setSSEStatus;
  window.startSSE = startSSE;
  window.stopSSE = stopSSE;
  window.updateAutoToggleUI = updateAutoToggleUI;
  window.toggleAutoUpdate = toggleAutoUpdate;
})();
