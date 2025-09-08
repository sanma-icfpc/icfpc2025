// status.js — Running/Queued/Recent panes + status refresh
;(function () {
  let statusInFlight = false;
  let recentPage = 1;
  let recentFilter = '';
  let expandedKeys = new Set();
  let detailsOpenIdx = [];
  let innerScrollTops = [];

  function captureUIState() {
    expandedKeys.clear();
    document.querySelectorAll('#running-pane .flow-row.expanded, #queued-pane .flow-row.expanded, #recent-pane .flow-row.expanded').forEach(el => {
      const k = el.getAttribute('data-key');
      if (k) expandedKeys.add(k);
    });
    detailsOpenIdx = [];
    const allDetails = Array.from(document.querySelectorAll('#running-pane details, #queued-pane details, #recent-pane details'));
    allDetails.forEach((d, i) => { if (d.open) detailsOpenIdx.push(i); });
    innerScrollTops = Array.from(document.querySelectorAll('#running-pane .overflow-auto, #running-pane .overflow-y-auto, #queued-pane .overflow-auto, #queued-pane .overflow-y-auto, #recent-pane .overflow-auto, #recent-pane .overflow-y-auto')).map(el => el.scrollTop);
  }

  function restoreUIState() {
    document.querySelectorAll('#running-pane .flow-row, #queued-pane .flow-row, #recent-pane .flow-row').forEach(el => {
      const k = el.getAttribute('data-key');
      if (k && expandedKeys.has(k)) {
        el.classList.add('expanded');
        const detail = el.nextElementSibling;
        if (detail) detail.classList.remove('hidden');
      }
    });
    const allDetails = Array.from(document.querySelectorAll('#running-pane details, #queued-pane details, #recent-pane details'));
    detailsOpenIdx.forEach(i => { if (allDetails[i]) allDetails[i].open = true; });
    const inner = Array.from(document.querySelectorAll('#running-pane .overflow-auto, #running-pane .overflow-y-auto, #queued-pane .overflow-auto, #queued-pane .overflow-y-auto, #recent-pane .overflow-auto, #recent-pane .overflow-y-auto'));
    for (let i = 0; i < inner.length && i < innerScrollTops.length; i++) {
      try { inner[i].scrollTop = innerScrollTops[i]; } catch (_) {}
    }
  }

  function updateRelativeTimes() {
    const now = Date.now();
    document.querySelectorAll('#running-pane .rt, #queued-pane .rt, #recent-pane .rt, #section-running .rt, #section-queued .rt, #section-recent .rt').forEach(el => {
      const ts = el.getAttribute('data-ts');
      if (!ts) return;
      try {
        const dt = new Date(ts);
        const diff = Math.max(0, Math.floor((now - dt.getTime()) / 1000));
        let rel = '';
        if (diff < 60) rel = `${diff}秒前`;
        else if (diff < 3600) rel = `${Math.floor(diff / 60)}分前`;
        else rel = `${Math.floor(diff / 3600)}時間前`;
        const y = dt.getFullYear();
        const m = ('0' + (dt.getMonth() + 1)).slice(-2);
        const d = ('0' + dt.getDate()).slice(-2);
        const hh = ('0' + dt.getHours()).slice(-2);
        const mm = ('0' + dt.getMinutes()).slice(-2);
        const ss = ('0' + dt.getSeconds()).slice(-2);
        el.textContent = `${y}/${m}/${d} ${hh}:${mm}:${ss} (${rel})`;
      } catch (_) {}
    });
    // Shallow inline "ago" labels next to agent identity
    document.querySelectorAll('#running-pane .rt-ago, #queued-pane .rt-ago, #recent-pane .rt-ago, #section-running .rt-ago, #section-queued .rt-ago, #section-recent .rt-ago').forEach(el => {
      const ts = el.getAttribute('data-ts');
      if (!ts) return;
      try {
        const dt = new Date(ts);
        const diff = Math.max(0, Math.floor((now - dt.getTime()) / 1000));
        let rel = '';
        if (diff < 60) {
          const v = diff;
          const u = v === 1 ? 'sec' : 'secs';
          rel = `${v} ${u}`;
        } else if (diff < 3600) {
          const v = Math.floor(diff / 60);
          const u = v === 1 ? 'min' : 'mins';
          rel = `${v} ${u}`;
        } else {
          const v = Math.floor(diff / 3600);
          const u = v === 1 ? 'hour' : 'hours';
          rel = `${v} ${u}`;
        }
        el.textContent = `${rel} ago`;
      } catch (_) {}
    });

    // Durations for recent cards (finished_at - started_at)
    document.querySelectorAll('#recent-pane .rt-dur, #section-recent .rt-dur').forEach(el => {
      const s = el.getAttribute('data-start');
      const e = el.getAttribute('data-end');
      if (!s || !e) return;
      try {
        const st = new Date(s).getTime();
        const et = new Date(e).getTime();
        const diff = Math.max(0, Math.floor((et - st) / 1000));
        const h = Math.floor(diff / 3600);
        const m = Math.floor((diff % 3600) / 60);
        const sec = diff % 60;
        let txt = '';
        if (h > 0) txt = `${h}h ${m}m ${sec}s`;
        else if (m > 0) txt = `${m}m ${sec}s`;
        else txt = `${sec}s`;
        el.textContent = txt;
      } catch (_) {}
    });
  }

  function getRecentPagesFromDOM() {
    try {
      const pager = document.getElementById('recent-pager');
      if (!pager) return { page: recentPage, pages: 1 };
      const page = parseInt(pager.getAttribute('data-page') || `${recentPage}`, 10) || recentPage;
      const pages = parseInt(pager.getAttribute('data-pages') || '1', 10) || 1;
      return { page, pages };
    } catch (_) { return { page: recentPage, pages: 1 }; }
  }

  function setURLParam(name, value) {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set(name, String(value));
      window.history.replaceState({}, '', url.toString());
    } catch (_) {}
  }

  function initRecentPageFromURL() {
    try {
      const url = new URL(window.location.href);
      const v = parseInt(url.searchParams.get('recent_page') || '', 10);
      if (!Number.isNaN(v) && v > 0) recentPage = v;
    } catch (_) {}
  }
  function initRecentFilterFromURL() {
    try {
      const url = new URL(window.location.href);
      const v = url.searchParams.get('recent_filter') || '';
      recentFilter = v;
    } catch (_) { recentFilter = ''; }
  }

  function recentPrev() {
    const { pages } = getRecentPagesFromDOM();
    if (recentPage > 1) { recentPage -= 1; setURLParam('recent_page', recentPage); loadStatus(); }
  }
  function recentNext() {
    const { pages } = getRecentPagesFromDOM();
    if (recentPage < pages) { recentPage += 1; setURLParam('recent_page', recentPage); loadStatus(); }
  }

  function bindRecentPagerEvents() {
    try {
      const pager = document.getElementById('recent-pager');
      if (!pager) return;
      const input = pager.querySelector('#recent-page-input');
      const pages = parseInt(pager.getAttribute('data-pages') || '1', 10) || 1;
      if (!input) return;
      input.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') {
          ev.preventDefault();
          let v = parseInt(input.value || '1', 10);
          if (Number.isNaN(v) || v < 1) v = 1;
          if (v > pages) v = pages;
          if (v !== recentPage) {
            recentPage = v;
            setURLParam('recent_page', recentPage);
            loadStatus();
          }
        }
      });
      input.addEventListener('change', function (_) {
        let v = parseInt(input.value || '1', 10);
        if (Number.isNaN(v) || v < 1) v = 1;
        if (v > pages) v = pages;
        if (v !== recentPage) {
          recentPage = v;
          setURLParam('recent_page', recentPage);
          loadStatus();
        }
      });
    } catch (_) {}
  }

  function loadStatus(opts) {
    opts = opts || {};
    const skipRecent = !!opts.skipRecent;
    if (statusInFlight) return;
    const rPane = document.getElementById('running-pane');
    const qPane = document.getElementById('queued-pane');
    const rcPane = document.getElementById('recent-pane');
    if (!rPane || !qPane || !rcPane) return;
    const prevScrolls = [rPane.scrollTop, qPane.scrollTop, rcPane.scrollTop];
    const prevWinY = window.scrollY || window.pageYOffset || 0;
    captureUIState();
    statusInFlight = true;
    const tasks = [];
    tasks.push(
      fetch('/minotaur/pane_running', { credentials: 'same-origin' })
        .then(r => r.text())
        .then(html => { rPane.innerHTML = html; try { window.updateIdenticons && window.updateIdenticons(rPane); } catch (_) {} })
        .catch(() => {})
    );
    tasks.push(
      fetch('/minotaur/pane_queued', { credentials: 'same-origin' })
        .then(r => r.text())
        .then(html => { qPane.innerHTML = html; try { window.updateIdenticons && window.updateIdenticons(qPane); } catch (_) {} })
        .catch(() => {})
    );
    if (!skipRecent) {
      let url = '/minotaur/pane_recent?recent_page=' + encodeURIComponent(recentPage);
      if (typeof recentFilter === 'string') url += '&recent_filter=' + encodeURIComponent(recentFilter);
      tasks.push(
        fetch(url, { credentials: 'same-origin' })
          .then(r => r.text())
          .then(html => {
            rcPane.innerHTML = html;
            try { window.updateIdenticons && window.updateIdenticons(rcPane); } catch (_) {}
            try {
              const pager = rcPane.querySelector('#recent-pager');
              if (pager) {
                const pg = parseInt(pager.getAttribute('data-page') || `${recentPage}`, 10) || recentPage;
                recentPage = pg;
                setURLParam('recent_page', recentPage);
                bindRecentPagerEvents();
              }
            } catch (_) {}
            try { bindRecentFilterEvents(); } catch (_) {}
          })
          .catch(() => {})
      );
    }
    Promise.all(tasks)
      .catch(() => {})
      .finally(() => {
        statusInFlight = false;
        [rPane, qPane, rcPane].forEach((el, i) => { try { el.scrollTop = prevScrolls[i]; } catch (_) {} });
        try { window.scrollTo(0, prevWinY); } catch (_) { }
        restoreUIState(); updateRelativeTimes();
      });
  }

  function updateAgentCount() {
    fetch('/minotaur/agent_count', { credentials: 'same-origin' })
      .then(r => r.json())
      .then(j => { const el = document.getElementById('agent-count'); if (el && typeof j.n === 'number') el.textContent = j.n; })
      .catch(() => {});
  }

  function bindRecentFilterEvents() {
    try {
      const input = document.getElementById('recent-filter-input');
      if (!input) return;
      input.value = typeof recentFilter === 'string' ? recentFilter : '';
      const apply = () => {
        const v = (input.value || '').trim();
        recentFilter = v;
        setURLParam('recent_filter', v);
        recentPage = 1;
        setURLParam('recent_page', recentPage);
        loadStatus();
      };
      input.addEventListener('keydown', function (ev) { if (ev.key === 'Enter') { ev.preventDefault(); apply(); } });
      input.addEventListener('change', function () { apply(); });
    } catch (_) {}
  }

  function toggleRecentRequests(headerEl) {
    try {
      if (!headerEl) return false;
      let scope = headerEl.parentElement || null;
      let container = null;
      if (scope) container = scope.querySelector('[data-recent-requests]');
      if (!container) container = headerEl.nextElementSibling;
      if (!container) return false;
      container.classList.toggle('hidden');
      const caret = headerEl.querySelector('[data-caret]');
      if (caret) caret.textContent = container.classList.contains('hidden') ? '▸' : '▾';
      return false;
    } catch (_) { return false; }
  }
  function toggleCardRequests(headerEl) {
    try {
      if (!headerEl) return false;
      let scope = headerEl.parentElement || null;
      let container = null;
      if (scope) container = scope.querySelector('[data-card-requests]');
      if (!container) container = headerEl.nextElementSibling;
      if (!container) return false;
      container.classList.toggle('hidden');
      const caret = headerEl.querySelector('[data-caret]');
      if (caret) caret.textContent = container.classList.contains('hidden') ? '▸' : '▾';
      return false;
    } catch (_) { return false; }
  }
  // Delegate click to Recent toggle headers
  (function bindRecentToggleDelegation() {
    const once = function () {
      try {
        const pane = document.getElementById('recent-pane');
        if (!pane) return;
        if (pane._recentDelegationBound) return;
        pane._recentDelegationBound = true;
        pane.addEventListener('click', function (ev) {
          try {
            const target = ev.target;
            if (!target || !target.closest) return;
            const header = target.closest('[data-recent-toggle]');
            if (header && pane.contains(header)) { ev.preventDefault(); toggleRecentRequests(header); }
          } catch (_) { }
        });
        pane.addEventListener('keydown', function (ev) {
          try {
            if (ev.key !== 'Enter' && ev.key !== ' ') return;
            const target = ev.target;
            if (!target || !target.closest) return;
            const header = target.closest('[data-recent-toggle]');
            if (header && pane.contains(header)) { ev.preventDefault(); toggleRecentRequests(header); }
          } catch (_) { }
        });
      } catch (_) { }
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', once, { once: true }); else once();
  })();

  // Expose globals
  window.loadStatus = loadStatus;
  window.updateAgentCount = updateAgentCount;
  window.initRecentPageFromURL = initRecentPageFromURL;
  window.initRecentFilterFromURL = initRecentFilterFromURL;
  window.recentPrev = recentPrev;
  window.recentNext = recentNext;
  window.bindRecentPagerEvents = bindRecentPagerEvents;
  window.bindRecentFilterEvents = bindRecentFilterEvents;
  window.updateRelativeTimes = updateRelativeTimes;
  window.toggleRecentRequests = toggleRecentRequests;
  window.toggleCardRequests = toggleCardRequests;
})();
