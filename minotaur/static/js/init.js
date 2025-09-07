// init.js — bootstraps initial loads and timers
;(function () {
  function onReady() {
    try { window.initRecentPageFromURL && window.initRecentPageFromURL(); } catch (_) {}
    try { window.initRecentFilterFromURL && window.initRecentFilterFromURL(); } catch (_) {}
    try { window.loadStatus && window.loadStatus(); } catch (_) {}
    try { if (document.getElementById('priority-pane')) window.loadPriorities && window.loadPriorities(true); } catch (_) {}
    try { window.updateAgentCount && window.updateAgentCount(); } catch (_) {}
    try { window.setSSEStatus && window.setSSEStatus(true); } catch (_) {}
    try { window.updateAutoToggleUI && window.updateAutoToggleUI(); } catch (_) {}
    try { window.startSSE && window.startSSE(); } catch (_) {}
    try { window.updateRelativeTimes && setInterval(window.updateRelativeTimes, 1000); } catch (_) {}
    // Light periodic refresh for Admin thread view
    setInterval(function () {
      try { window.refreshThreadsIfVisible && window.refreshThreadsIfVisible(); } catch (_) {}
      try { window.refreshMemInfoIfVisible && window.refreshMemInfoIfVisible(); } catch (_) {}
    }, 2000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', onReady, { once: true }); else onReady();
})();

