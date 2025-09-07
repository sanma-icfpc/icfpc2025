let chart;
let allData = {};
let highlightTeams = [];
let filterMode = false; // when true, show only highlighted teams
let hoverTs = null;      // current hover timestamp (Date)
let selectedTs = null;   // persisted selection (Date)
let isDragging = false;
let hoverTeam = null;    // team name hovered when no manual highlights
let zeroOnlyTeams = new Set(); // teams whose scores are always 0

async function fetchData() {
  const res = await fetch("/api/data");
  allData = await res.json();
  // compute teams that are always 0
  zeroOnlyTeams = new Set(
    Object.keys(allData).filter(team =>
      (allData[team] || []).length > 0 && (allData[team] || []).every(p => Number(p.score) === 0)
    )
  );
  updateChart();
  const latest = latestTimestamp();
  if (latest) {
    selectedTs = latest;
    showRankingAt(selectedTs);
  }
}

function parseTimestamp(ts) {
  // "YYYYMMDD_HHMMSS"
  const year = ts.substring(0,4);
  const month = ts.substring(4,6);
  const day = ts.substring(6,8);
  const hour = ts.substring(9,11);
  const minute = ts.substring(11,13);
  const second = ts.substring(13,15);
  return new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}+09:00`);
}

function updateChart() {
  const ctx = document.getElementById("leaderboardChart").getContext("2d");
  const datasets = [];

  Object.keys(allData).forEach((team, idx) => {
    // Omit teams with score 0 for all timestamps
    if (zeroOnlyTeams.has(team)) return;
    // Deduplicate by timestamp: keep max score per timestamp
    const byTs = new Map();
    for (const p of allData[team]) {
      const ts = p.timestamp;
      const s = Number(p.score);
      const prev = byTs.get(ts);
      if (prev === undefined || s > prev) byTs.set(ts, s);
    }
    const points = Array.from(byTs.entries())
      .map(([ts, s]) => ({ x: parseTimestamp(ts), y: s }))
      .sort((a, b) => a.x - b.x);
    const isHighlight = highlightTeams.includes(team);
    if (filterMode && highlightTeams.length > 0 && !isHighlight) return; // show only highlighted teams when filtering and has selection

    const hue = (idx * 47) % 360;
    const baseColor = `hsl(${hue},70%,50%)`;
    const dimColor = `hsla(${hue},70%,50%,0.2)`;
    let color = baseColor;
    let width = isHighlight ? 3 : 1;
    // In highlight mode (not filtering), dim non-selected only if any selection exists
    if (!filterMode && highlightTeams.length > 0 && !isHighlight) {
      color = `hsla(${hue},70%,50%,0.2)`; // dim non-highlight lines
      width = 1;
    }

    datasets.push({
      label: team,
      data: points,
      borderColor: color,
      borderWidth: width,
      tension: 0,
      cubicInterpolationMode: 'default',
      fill: false,
      // store for dynamic hover dimming
      _baseColor: baseColor,
      _dimColor: dimColor,
      _baseWidth: width,
    });
  });

  if (chart) chart.destroy();

  // Crosshair plugin: draws hover and selection vertical lines
  const crosshairPlugin = {
    id: 'crosshair',
    afterDatasetsDraw(chartInstance) {
      const { ctx, chartArea, scales } = chartInstance;
      const xScale = scales.x;
      const top = chartArea.top;
      const bottom = chartArea.bottom;
      ctx.save();
      if (hoverTs && !isDragging) {
        const x = xScale.getPixelForValue(hoverTs);
        if (x >= chartArea.left && x <= chartArea.right) {
          ctx.strokeStyle = 'rgba(0,0,0,0.4)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, top);
          ctx.lineTo(x, bottom);
          ctx.stroke();
        }
      }
      if (selectedTs) {
        const x = xScale.getPixelForValue(selectedTs);
        if (x >= chartArea.left && x <= chartArea.right) {
          ctx.strokeStyle = 'rgba(0,0,0,0.8)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x, top);
          ctx.lineTo(x, bottom);
          ctx.stroke();
        }
      }
      // No manual redraw for hovered team; we adjust dataset styles instead
      ctx.restore();
    }
  };

  // Right-end labels plugin: draws team names at right edge
  const rightLabelsPlugin = {
    id: 'rightLabels',
    afterDatasetsDraw(chartInstance) {
      const { ctx, chartArea, data } = chartInstance;
      const rightX = chartArea.right + 4;
      ctx.save();
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      // Compute final snapshot ranks/scores (nearest to latest timestamp)
      const latest = latestTimestamp();
      let finalTsStr = null;
      if (latest) finalTsStr = nearestTimestampStringForDate(latest);
      const finalSnap = finalTsStr ? buildSnapshot(finalTsStr) : { teamRank: {}, teamScore: {} };
      data.datasets.forEach((ds, i) => {
        const meta = chartInstance.getDatasetMeta(i);
        if (!meta || !meta.data || meta.data.length === 0) return;
        const last = meta.data[meta.data.length - 1];
        if (!last) return;
        const y = Math.max(chartArea.top + 6, Math.min(chartArea.bottom - 6, last.y));
        const isHover = (highlightTeams.length === 0 && hoverTeam && ds.label === hoverTeam);
        ctx.fillStyle = ds.borderColor || '#333';
        if (isHover) {
          ctx.font = 'bold 12px sans-serif';
        }
        const rk = finalSnap.teamRank[ds.label];
        const sc = finalSnap.teamScore[ds.label];
        const labelText = (rk !== undefined && sc !== undefined)
          ? `${ds.label} (${ordinal(rk)}): ${sc}`
          : ds.label;
        ctx.fillText(labelText, rightX, y);
      });
      ctx.restore();
    }
  };

  // Hover tooltip plugin: shows team name and score at nearest point on hovered team
  const hoverTooltipPlugin = {
    id: 'hoverTooltip',
    afterDatasetsDraw(chartInstance) {
      if (!hoverTeam || highlightTeams.length > 0 || !hoverTs) return;
      const { ctx, chartArea, data, scales } = chartInstance;
      const xScale = scales.x;
      const yScale = scales.y;
      const xHover = xScale.getPixelForValue(hoverTs);
      let dsIndex = data.datasets.findIndex(d => d.label === hoverTeam);
      if (dsIndex < 0) return;
      const ds = data.datasets[dsIndex];
      const meta = chartInstance.getDatasetMeta(dsIndex);
      const points = meta?.data || [];
      if (!points.length) return;
      // find nearest point to xHover
      let bestIdx = 0;
      let bestDx = Infinity;
      points.forEach((pt, i) => {
        const dx = Math.abs(pt.x - xHover);
        if (dx < bestDx) { bestDx = dx; bestIdx = i; }
      });
      const pt = points[bestIdx];
      const datum = ds.data[bestIdx];
      const score = datum?.y ?? '';
      // compute rank at nearest timestamp to hoverTs
      const hoverTsStr = nearestTimestampStringForDate(hoverTs);
      const snap = hoverTsStr ? buildSnapshot(hoverTsStr) : { teamRank: {} };
      const r = snap.teamRank[ds.label];

      // draw a small marker
      ctx.save();
      ctx.fillStyle = ds.borderColor || '#000';
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2); ctx.fill();

      // tooltip text
      const text = (r !== undefined) ? `${ds.label} (${ordinal(r)}): ${score}` : `${ds.label}: ${score}`;
      ctx.font = '12px sans-serif';
      const padding = 6;
      const tw = ctx.measureText(text).width;
      const th = 18;
      let tx = pt.x + 8;
      let ty = pt.y - th - 4;
      // keep inside chartArea
      if (tx + tw + padding * 2 > chartArea.right) tx = pt.x - (tw + padding * 2) - 8;
      if (ty < chartArea.top) ty = pt.y + 8;
      // box
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(tx, ty, tw + padding * 2, th);
      ctx.fill();
      ctx.stroke();
      // text
      ctx.fillStyle = '#000';
      ctx.fillText(text, tx + padding, ty + th - 6);
      ctx.restore();
    }
  };

  // Dynamically reserve enough right padding for labels
  const adjustRightPaddingPlugin = {
    id: 'adjustRightPadding',
    beforeLayout(chartInstance) {
      const { ctx } = chartInstance;
      ctx.save();
      ctx.font = '12px sans-serif';
      // compute final snapshot ranks/scores for label text
      const latest = latestTimestamp();
      let finalTsStr = null;
      if (latest) finalTsStr = nearestTimestampStringForDate(latest);
      const finalSnap = finalTsStr ? buildSnapshot(finalTsStr) : { teamRank: {}, teamScore: {} };
      let maxWidth = 0;
      (chartInstance.data.datasets || []).forEach((ds) => {
        const rk = finalSnap.teamRank[ds.label];
        const sc = finalSnap.teamScore[ds.label];
        const text = (rk !== undefined && sc !== undefined)
          ? `${ds.label} (${ordinal(rk)}): ${sc}`
          : ds.label;
        const w = ctx.measureText(text).width;
        if (w > maxWidth) maxWidth = w;
      });
      ctx.restore();
      const desired = Math.ceil(maxWidth + 12); // 8px padding + small margin
      if (!chartInstance.options.layout) chartInstance.options.layout = { padding: {} };
      if (!chartInstance.options.layout.padding) chartInstance.options.layout.padding = {};
      chartInstance.options.layout.padding.right = Math.max(desired, 120);
    }
  };

  chart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      layout: { padding: { right: 120 } },
      interaction: { mode: undefined },
      plugins: {
        tooltip: { enabled: false },
        legend: { display: false }
      },
      elements: { point: { radius: 2, hoverRadius: 2 } },
      scales: {
        x: {
          type: "time",
          time: { unit: "hour" },
          ticks: {
            source: 'data',
            callback: (value) => {
              const d = new Date(value);
              const m = d.getMonth() + 1;
              const day = d.getDate();
              const hh = String(d.getHours()).padStart(2, '0');
              const mm = String(d.getMinutes()).padStart(2, '0');
              return `${m}/${day} ${hh}:${mm} JST`;
            }
          }
        },
        y: { beginAtZero: true }
      }
    },
    plugins: [adjustRightPaddingPlugin, crosshairPlugin, hoverTooltipPlugin, rightLabelsPlugin]
  });

  // Mouse interactions for hover/drag selection
  const canvas = chart.canvas;
  const xScale = chart.scales.x;
  function getTsFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const v = xScale.getValueForPixel(x);
    return v instanceof Date ? v : new Date(v);
  }
  function getMouseXY(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }
  function updateHoverTeamFromMouse(e) {
    if (highlightTeams.length > 0) { hoverTeam = null; return; }
    const { x: mx, y: my } = getMouseXY(e);
    let bestTeam = null;
    let best = Infinity;
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      const pts = meta?.data || [];
      pts.forEach((pt) => {
        const dx = mx - pt.x;
        const dy = my - pt.y;
        const d2 = dx*dx + dy*dy;
        if (d2 < best) {
          best = d2;
          bestTeam = ds.label;
        }
      });
    });
    hoverTeam = bestTeam;
  }
  function redraw() { chart.update('none'); }
  function applyHoverDim() {
    if (!chart) return;
    const hasManual = highlightTeams.length > 0;
    if (hasManual) return; // manual highlight takes precedence
    chart.data.datasets.forEach((ds) => {
      if (hoverTeam && ds.label !== hoverTeam) {
        ds.borderColor = ds._dimColor || ds.borderColor;
        ds.borderWidth = 1;
      } else {
        ds.borderColor = ds._baseColor || ds.borderColor;
        ds.borderWidth = (hoverTeam && ds.label === hoverTeam) ? 4 : (ds._baseWidth || 1);
      }
    });
  }
  canvas.addEventListener('mousemove', (e) => {
    hoverTs = getTsFromEvent(e);
    updateHoverTeamFromMouse(e);
    if (isDragging) {
      selectedTs = hoverTs;
      showRankingAt(selectedTs);
    }
    applyHoverDim();
    redraw();
  });
  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    selectedTs = getTsFromEvent(e);
    showRankingAt(selectedTs);
    redraw();
  });
  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      redraw();
    }
  });
  canvas.addEventListener('mouseleave', () => {
    hoverTs = null;
    hoverTeam = null;
    applyHoverDim();
    redraw();
  });
}

function latestTimestamp() {
  let latest = null;
  for (const team in allData) {
    for (const p of allData[team]) {
      const d = parseTimestamp(p.timestamp);
      if (!latest || d > latest) latest = d;
    }
  }
  return latest;
}

// Find the timestamp string closest to a given Date across all teams
function nearestTimestampStringForDate(d) {
  if (!d) return null;
  let nearestTs = null;
  let minDiff = Infinity;
  for (const team in allData) {
    for (const p of allData[team]) {
      const t = parseTimestamp(p.timestamp);
      const diff = Math.abs(t - d);
      if (diff < minDiff) { minDiff = diff; nearestTs = p.timestamp; }
    }
  }
  return nearestTs;
}

// Build snapshot ranking and score maps at an exact timestamp string
function buildSnapshot(tsStr) {
  const allRows = [];
  for (const team in allData) {
    if (zeroOnlyTeams.has(team)) continue;
    let maxScore = null;
    for (const p of allData[team]) {
      if (p.timestamp === tsStr) {
        const s = Number(p.score);
        if (maxScore === null || s > maxScore) maxScore = s;
      }
    }
    if (maxScore !== null) {
      allRows.push({ team, score: maxScore });
    }
  }
  allRows.sort((a,b) => b.score - a.score);
  const teamRank = {};
  const teamScore = {};
  allRows.forEach((r, i) => { teamRank[r.team] = i + 1; teamScore[r.team] = r.score; });
  return { teamRank, teamScore };
}

function ordinal(n) {
  const s = ["th","st","nd","rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function showRankingAt(ts) {
  // nearest timestamp in data
  let nearestTs = null;
  let minDiff = Infinity;
  for (const team in allData) {
    for (const p of allData[team]) {
      const t = parseTimestamp(p.timestamp);
      const diff = Math.abs(t - ts);
      if (diff < minDiff) {
        minDiff = diff;
        nearestTs = p.timestamp;
      }
    }
  }

  if (!nearestTs) return;

  // Build human-readable time label in JST
  const d = parseTimestamp(nearestTs);
  const mm = String(d.getMonth() + 1);
  const dd = String(d.getDate());
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  const whenLabel = `${mm}/${dd} ${hh}:${mi} JST`;

  // Build full rows to compute true global ranks
  const allRows = [];
  for (const team in allData) {
    if (zeroOnlyTeams.has(team)) continue; // omit teams always 0
    let maxScore = null;
    for (const p of allData[team]) {
      if (p.timestamp === nearestTs) {
        const s = Number(p.score);
        if (maxScore === null || s > maxScore) maxScore = s;
      }
    }
    if (maxScore !== null) {
      allRows.push({ team, score: maxScore });
    }
  }
  allRows.sort((a,b) => b.score - a.score);
  const teamRank = {};
  allRows.forEach((r, i) => { teamRank[r.team] = i + 1; });

  // In filter mode, omit non-highlighted rows but keep original ranks
  const rows = (filterMode && highlightTeams.length > 0)
    ? allRows.filter(r => highlightTeams.includes(r.team))
    : allRows;

  let html = `<div style="margin-bottom:6px; font-size:12px; color:#555;">${whenLabel}</div>`;
  html += "<table><tr><th>Rank</th><th>Team</th><th>Score</th></tr>";
  rows.forEach((r) => {
    const cls = highlightTeams.includes(r.team) ? "highlight" : "";
    const rk = teamRank[r.team] ?? '';
    html += `<tr class="${cls}"><td>${rk}</td><td>${r.team}</td><td>${r.score}</td></tr>`;
  });
  html += "</table>";

  document.getElementById("rankingTable").innerHTML = html;
}

document.addEventListener("DOMContentLoaded", () => {
  // Restore settings from URL params (?hl=a,b&filter=1)
  const highlightInput = document.getElementById("highlightInput");
  const filterCheckbox = document.getElementById('filterMode');
  const params = new URLSearchParams(window.location.search);
  const hlParam = params.get('hl') || '';
  const filterParam = params.get('filter');
  highlightInput.value = hlParam;
  highlightTeams = hlParam.split(",").map(s => s.trim()).filter(s => s);
  filterMode = filterParam === '1' || filterParam === 'true';
  filterCheckbox.checked = filterMode;

  function setUrlParam(name, value) {
    const p = new URLSearchParams(window.location.search);
    if (value === undefined || value === null || value === '' || value === false) {
      p.delete(name);
    } else {
      p.set(name, String(value));
    }
    const q = p.toString();
    const newUrl = q ? `${location.pathname}?${q}` : location.pathname;
    history.replaceState({}, '', newUrl);
  }

  fetchData();

  highlightInput.addEventListener("input", (e) => {
    highlightTeams = e.target.value.split(",").map(s => s.trim()).filter(s => s);
    setUrlParam('hl', e.target.value);
    updateChart();
    if (selectedTs) {
      showRankingAt(selectedTs);
    } else {
      const latest = latestTimestamp();
      if (latest) showRankingAt(latest);
    }
  });

  filterCheckbox.addEventListener('change', (e) => {
    filterMode = e.target.checked;
    setUrlParam('filter', filterMode ? '1' : null);
    updateChart();
    if (selectedTs) {
      showRankingAt(selectedTs);
    } else {
      const latest = latestTimestamp();
      if (latest) showRankingAt(latest);
    }
  });
});
