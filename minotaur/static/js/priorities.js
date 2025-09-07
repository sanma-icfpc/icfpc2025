// priorities.js — functions for the Scheduler/Priorities pane
;(function () {
  function esc(s) { try { return CSS.escape(s); } catch (_) { return s; } }

  function collectPriorityItems() {
    const rows = document.querySelectorAll('#priority-pane [data-row]');
    const items = [];
    rows.forEach(tr => {
      const nameAttr = tr.getAttribute('data-name');
      const cells = tr.querySelectorAll('td');
      if (!cells || cells.length === 0) return;
      if (nameAttr === '__new__') {
        const name = (tr.querySelector('input[type=text]')?.value || '').trim();
        const pri = parseInt(tr.querySelector('input[type=number]')?.value || '0', 10);
        if (name) items.push({ name, priority: pri });
        return;
      }
      const name = nameAttr;
      const pri = parseInt(tr.querySelector('input[type=number]')?.value || '0', 10);
      items.push({ name, priority: pri });
    });
    return items;
  }

  function loadPriorities(sort) {
    const el = document.getElementById('priority-pane');
    if (!el) return;
    const url = '/minotaur/priorities' + (sort ? '?sort=1' : '');
    fetch(url, { credentials: 'same-origin' })
      .then(r => r.text())
      .then(html => { el.innerHTML = html; try { window.updateIdenticons && window.updateIdenticons(el); } catch (_) {} })
      .catch(() => {});
  }

  function applyAgentPriorities(sortAfter) {
    const items = collectPriorityItems();
    fetch('/minotaur/priorities/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
      credentials: 'same-origin'
    }).then(() => { loadPriorities(!!sortAfter); try { window.loadStatus && window.loadStatus(); } catch (_) {} })
      .catch(() => { loadPriorities(!!sortAfter); try { window.loadStatus && window.loadStatus(); } catch (_) {} });
    return false;
  }

  function togglePin(name, pinned) {
    fetch('/minotaur/priorities/pin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, pinned }),
      credentials: 'same-origin'
    }).then(() => { refreshPriorityStatuses(); })
      .catch(() => {});
    return false;
  }

  function deleteAgentPriority(name) {
    if (!confirm(`Delete priority for ${name}?`)) return false;
    const row = document.querySelector(`#priority-pane [data-row][data-name="${esc(name)}"]`);
    if (row) row.style.opacity = '0.5';
    fetch('/minotaur/priorities/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
      credentials: 'same-origin'
    }).then(() => { loadPriorities(false); })
      .catch(() => { loadPriorities(false); });
    return false;
  }

  function selectNextAgent(name) {
    fetch('/minotaur/priorities/select_next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
      credentials: 'same-origin'
    }).then(() => { try { window.loadStatus && window.loadStatus(); } catch (_) {} ; refreshPriorityStatuses(); })
      .catch(() => { try { window.loadStatus && window.loadStatus(); } catch (_) {} ; refreshPriorityStatuses(); });
    return false;
  }

  function refreshPriorityStatuses() {
    const pane = document.getElementById('priority-pane');
    if (!pane) return;
    fetch('/minotaur/priorities/data', { credentials: 'same-origin' })
      .then(r => r.json())
      .then(j => {
        const rows = j.rows || [];
        const tableBody = pane.querySelector('#priority-rows');
        if (!tableBody) return;
        const existing = new Map();
        tableBody.querySelectorAll('tr[data-row]').forEach(tr => { existing.set(tr.getAttribute('data-name'), tr); });
        rows.forEach(row => {
          const name = row.name;
          const prio = row.priority;
          const status = row.status;
          const _pinval = (typeof row.pinned === 'number') ? row.pinned : (row.pinned ? 1 : 0);
          const pinned = (_pinval === 1) ? 1 : 0; // persistent pin only
          const nextFlag = (_pinval === 2) ? 1 : 0; // one-shot select-next
          let tr = existing.get(name);
          if (!tr) {
            const newRow = document.createElement('tr');
            newRow.setAttribute('data-row', '');
            newRow.setAttribute('data-name', name);
            newRow.innerHTML = `
              <td class="p-1 font-mono truncate">
                <div class="flex items-center gap-2">
                  ${name === '*' ? '' : '<div class=\'w-5 h-5 rounded-md border overflow-hidden\'><svg width=\'20\' height=\'20\' data-jdenticon-value=\'' + name + '\'></svg></div>'}
                  ${name === '*' ? '<span class=\'text-gray-500\'>(default)</span>' : name}
                </div>
              </td>
              <td class="p-1"><span class="badge badge-${status}">${status}</span></td>
              <td class="p-1"><input type="number" min="1" max="100" step="1" class="border rounded px-2 py-1 w-20" value="${prio}" /></td>
              <td class="p-1 text-xs text-gray-500"><span data-share>w=${prio} v=${(row.vtime||0).toFixed ? row.vtime.toFixed(2) : '0.00'} s=${(row.service||0).toFixed ? row.service.toFixed(1) : '0.0'}</span></td>
              <td class="p-1">${name !== '*' ? '<button data-pin class="px-2 py-1 text-xs ' + (pinned ? 'bg-yellow-500 text-white' : 'bg-gray-200') + '" onclick="togglePin(\'' + name + '\',' + (pinned ? 0 : 1) + '); return false;">' + (pinned ? 'Unpin' : 'Pin') + '</button>' : '<span class="text-gray-400">—</span>'}</td>
              <td class="p-1">${name !== '*' ? '<button class="px-2 py-1 text-xs bg-red-600 text-white rounded" onclick="deleteAgentPriority(\'' + name + '\'); return false;">Delete</button>' : '<span class="text-gray-400">—</span>'}</td>
              <td class="p-1">${name !== '*' ? '<div class="flex items-center gap-2"><button class="px-2 py-1 text-xs bg-blue-600 text-white rounded" onclick="selectNextAgent(\'' + name + '\'); return false;">Select for Next</button><span data-next-flag class="' + (nextFlag ? 'badge badge-queued' : 'hidden badge badge-queued') + '">Next</span></div>' : '<span class="text-gray-400">—</span>'}</td>
            `;
            const newRowInput = tableBody.querySelector('tr[data-name="__new__"]');
            tableBody.insertBefore(newRow, newRowInput ? newRowInput : null);
            return;
          }
          // Update badge
          const badge = tr.querySelector('.badge');
          if (badge) {
            badge.textContent = status;
            badge.className = `badge badge-${status}`;
          }
          // Update priority input if not focused and not dirty
          const inp = tr.querySelector('input[type=number]');
          const isDirty = !!(inp && inp.getAttribute('data-dirty') === '1');
          if (inp && document.activeElement !== inp && !isDirty) {
            const cur = parseInt(inp.value || '0', 10);
            if (cur !== prio) inp.value = prio;
          }
          // Update share (weight/vtime/service)
          const share = tr.querySelector('[data-share]');
          if (share) {
            const v = (typeof row.vtime === 'number') ? row.vtime : 0;
            const s = (typeof row.service === 'number') ? row.service : 0;
            let dispW = prio;
            try {
              if (inp && inp.getAttribute('data-dirty') === '1') {
                const cur = parseInt(inp.value || `${prio}`, 10);
                if (!Number.isNaN(cur) && cur > 0) dispW = cur;
              }
            } catch (_) {}
            share.textContent = `w=${dispW} v=${v.toFixed(2)} s=${s.toFixed(1)}`;
          }
          // Update pin button
          let pinBtn = tr.querySelector('[data-pin]');
          if (pinBtn && name !== '*') {
            pinBtn.textContent = pinned ? 'Unpin' : 'Pin';
            pinBtn.className = 'px-2 py-1 text-xs ' + (pinned ? 'bg-yellow-500 text-white' : 'bg-gray-200');
            pinBtn.setAttribute('onclick', `togglePin('${name}', ${pinned ? 0 : 1}); return false;`);
          }
          // Update next flag indicator
          const nf = tr.querySelector('[data-next-flag]');
          if (nf) {
            if (nextFlag) {
              nf.className = 'badge badge-queued';
            } else {
              nf.className = 'hidden badge badge-queued';
            }
          }
        });
        try { window.updateIdenticons && window.updateIdenticons(tableBody); } catch (_) {}
      })
      .catch(() => {});
  }

  // Mark inputs dirty on user edit to prevent auto-reload from overwriting
  document.addEventListener('input', function (ev) {
    const t = ev.target;
    if (!t) return;
    try {
      if (t.closest && t.closest('#priority-pane') && t.matches && t.matches('input[type=number]')) {
        t.setAttribute('data-dirty', '1');
      }
    } catch (_) {}
  }, true);
  document.addEventListener('change', function (ev) {
    const t = ev.target;
    if (!t) return;
    try {
      if (t.closest && t.closest('#priority-pane') && t.matches && t.matches('input[type=number]')) {
        t.setAttribute('data-dirty', '1');
      }
    } catch (_) {}
  }, true);

  // Expose global API for inline onclicks
  window.loadPriorities = loadPriorities;
  window.applyAgentPriorities = applyAgentPriorities;
  window.togglePin = togglePin;
  window.deleteAgentPriority = deleteAgentPriority;
  window.selectNextAgent = selectNextAgent;
  window.refreshPriorityStatuses = refreshPriorityStatuses;
})();

