// Data injected from Streamlit
const injected = window.__INITIAL_DATA__ || {};
const meta = injected.meta || {};
const lgas = Array.isArray(injected.lgas) ? injected.lgas : [];
const hotspotsPayload = Array.isArray(injected.hotspots) ? injected.hotspots : [];
const stateOptions = ['All Nigeria', ...(injected.states || [])];

let currentState = meta.state_filter || 'All Nigeria';
let currentDepth = Number(meta.depth || 0);
let currentFocus = meta.focus || 'All risk';
let currentYear = meta.year != null ? String(meta.year) : '2018';
let currentLayer = 'Risk score';

const lgaById = new Map(lgas.map((l) => [String(l.id), l]));
const featureLayerById = new Map();
const fieldValuesCache = new Map();

const selectedId = meta.selected_lga || (injected.selected && injected.selected.id);
let selectedLGA = null;
if (selectedId) {
  const base = lgaById.get(String(selectedId)) || {};
  selectedLGA = { ...base, ...(injected.selected || {}) };
} else if (injected.selected) {
  selectedLGA = injected.selected;
}

let compareLGAs = (meta.compare_lgas || [])
  .map((id) => lgaById.get(String(id)))
  .filter(Boolean);

const riskLookup = {};
(injected.map?.choropleth || []).forEach((item) => {
  const val = item.risk_norm ?? item.risk;
  riskLookup[String(item.id)] = val != null ? Number(val) : null;
});

let mapInstance = null;
let geoLayer = null;
let baseGeoJson = null;
let hasFitBounds = false;

let stateSyncTimer = null;
let pendingStateUrl = '';
let stateSyncLocked = false;

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function sanitizeText(value, fallback = '') {
  if (value == null) return fallback;
  return String(value);
}

function safeNum(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Number(value);
}

function riskLabel(r) {
  return r == null || Number.isNaN(Number(r)) ? 'NA' : (Number(r) * 100).toFixed(0);
}

function confidenceBadge(conf) {
  const n = safeNum(conf);
  if (n == null) return { emoji: '🟡', label: 'Unknown' };
  if (n >= 80) return { emoji: '🟢', label: `${n.toFixed(0)}%` };
  if (n >= 60) return { emoji: '🟡', label: `${n.toFixed(0)}%` };
  return { emoji: '🔴', label: `${n.toFixed(0)}%` };
}

function fmtMetric(v) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  const num = Number(v);
  if (Math.abs(num) >= 10) return num.toFixed(0);
  return num.toFixed(1);
}

function metricClass(value, lowThreshold, midThreshold, invert = false) {
  const n = safeNum(value);
  if (n == null) return 'metric-value-yellow';
  if (!invert) {
    if (n < lowThreshold) return 'metric-value-red';
    if (n < midThreshold) return 'metric-value-yellow';
    return 'metric-value-green';
  }
  if (n > midThreshold) return 'metric-value-red';
  if (n > lowThreshold) return 'metric-value-yellow';
  return 'metric-value-green';
}

function riskColorHex(r) {
  if (r == null || Number.isNaN(Number(r))) return 'rgba(52, 211, 153, 0.6)';
  const v = Math.max(0, Math.min(1, Number(r)));
  if (v > 0.66) {
    const t = (v - 0.66) / 0.34;
    return `rgba(239, ${Math.round(68 - 68 * t)}, ${Math.round(68 - 68 * t)}, ${0.7 + t * 0.2})`;
  }
  if (v > 0.33) {
    const t = (v - 0.33) / 0.33;
    return `rgba(${Math.round(234 * t + 34 * (1 - t))}, ${Math.round(179 * t + 197 * (1 - t))}, ${Math.round(8 * t + 94 * (1 - t))}, 0.75)`;
  }
  return `rgba(34, 197, 94, ${0.4 + v * 0.8})`;
}

function setApplyStatus(label, mode = 'applied') {
  const status = document.getElementById('apply-status');
  if (!status) return;

  const dot = document.createElement('span');
  dot.className = 'status-dot';
  if (mode === 'applied') dot.classList.add('status-dot-static');

  status.replaceChildren(dot, document.createTextNode(label));
}

function syncHeader() {
  const metaEl = document.getElementById('dataset-meta-text');
  if (metaEl) {
    const count = meta.lga_count || lgas.length || '—';
    const yearText = currentYear || '—';
    metaEl.textContent = `${count} LGAs · ${yearText}`;
  }

  const focusScope = document.getElementById('focus-scope');
  if (focusScope) focusScope.textContent = currentState;

  document.querySelectorAll('.depth-btn').forEach((btn) => {
    const depthVal = Number(btn.dataset.depth || 0);
    const active = depthVal === currentDepth;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });

  document.querySelectorAll('.chip').forEach((chip) => {
    const active = chip.dataset.focus === currentFocus;
    chip.classList.toggle('active', active);
    chip.setAttribute('aria-pressed', String(active));
  });

  setApplyStatus('Applied', 'applied');
}

function applyDepthVisibility() {
  document.querySelectorAll('.depth-gate').forEach((el) => {
    const depthClass = [...el.classList].find((c) => c.startsWith('depth-')) || 'depth-99';
    const requiredDepth = Number(depthClass.replace('depth-', ''));
    const visible = currentDepth >= requiredDepth;
    el.hidden = !visible;
  });

  const strip = document.getElementById('compare-strip');
  if (strip) strip.classList.toggle('visible', currentDepth >= 1);
}

function buildStateUrl() {
  const params = new URLSearchParams(window.parent.location.search);
  params.set('state', currentState);
  params.set('focus', currentFocus);
  params.set('depth', String(currentDepth));
  params.set('year', currentYear);

  if (selectedLGA?.id) params.set('lga', String(selectedLGA.id));
  else params.delete('lga');

  if (compareLGAs.length) params.set('compare', compareLGAs.map((l) => l.id).join(','));
  else params.delete('compare');

  return `${window.parent.location.pathname}?${params.toString()}`;
}

function flushStateToPython() {
  if (stateSyncLocked || !pendingStateUrl) return;

  const currentUrl = `${window.parent.location.pathname}${window.parent.location.search}`;
  if (pendingStateUrl === currentUrl) {
    setApplyStatus('Applied', 'applied');
    pendingStateUrl = '';
    return;
  }

  stateSyncLocked = true;
  window.parent.location.replace(pendingStateUrl);
}

function pushStateToPython({ immediate = false } = {}) {
  pendingStateUrl = buildStateUrl();

  if (stateSyncTimer) {
    clearTimeout(stateSyncTimer);
    stateSyncTimer = null;
  }

  if (stateSyncLocked) return;

  setApplyStatus('Updating…', 'updating');
  if (immediate) {
    flushStateToPython();
    return;
  }

  stateSyncTimer = window.setTimeout(() => {
    stateSyncTimer = null;
    flushStateToPython();
  }, 250);
}

function hotspotsBase() {
  if (hotspotsPayload.length) return hotspotsPayload;
  return [...lgas]
    .sort((a, b) => (Number(b.risk ?? 0) - Number(a.risk ?? 0)))
    .map((l, i) => ({ ...l, rank: i + 1 }));
}

function renderHotspots() {
  const list = document.getElementById('hotspot-list');
  if (!list) return;

  const query = (document.getElementById('search-input')?.value || '').toLowerCase();
  const base = hotspotsBase();
  const filtered = query
    ? base.filter((l) => `${sanitizeText(l.name)} ${sanitizeText(l.state)}`.toLowerCase().includes(query))
    : base;

  list.replaceChildren();
  filtered.slice(0, 12).forEach((lga, i) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'hotspot-card';
    if (String(selectedLGA?.id) === String(lga.id)) card.classList.add('active');
    card.style.animationDelay = `${i * 0.04}s`;
    card.setAttribute('aria-label', `Select ${sanitizeText(lga.name, 'LGA')}`);

    const rank = document.createElement('div');
    rank.className = 'hotspot-rank';
    rank.textContent = String(lga.rank ?? i + 1).padStart(2, '0');

    const info = document.createElement('div');
    info.className = 'hotspot-info';

    const name = document.createElement('div');
    name.className = 'hotspot-name';
    name.textContent = sanitizeText(lga.name, 'Unnamed LGA');

    const state = document.createElement('div');
    state.className = 'hotspot-state';
    state.textContent = sanitizeText(lga.state, 'Unknown state');

    info.append(name, state);

    const scoreWrap = document.createElement('div');
    scoreWrap.className = 'hotspot-score';

    const conf = confidenceBadge(lga.confidence_pct);
    const confEl = document.createElement('div');
    confEl.className = 'hotspot-state';
    confEl.textContent = `${conf.emoji} ${conf.label}`;

    const badge = document.createElement('div');
    const risk = safeNum(lga.risk);
    const bucket = risk != null && risk > 0.66 ? 'risk-high' : risk != null && risk > 0.33 ? 'risk-med' : 'risk-low';
    badge.className = `risk-badge ${bucket}`;
    badge.textContent = riskLabel(risk);

    scoreWrap.append(confEl, badge);
    card.append(rank, info, scoreWrap);
    card.addEventListener('click', () => selectLGA(lga.id));
    list.appendChild(card);
  });
}

function openDrawer() {
  const drawer = document.getElementById('detail-drawer');
  if (drawer) drawer.classList.add('open');
}

function closeDrawer() {
  const drawer = document.getElementById('detail-drawer');
  if (drawer) drawer.classList.remove('open');
  selectedLGA = null;
  renderHotspots();
  renderMap();
  pushStateToPython();
}

function percentileRank(field, value) {
  if (value == null) return null;

  if (!fieldValuesCache.has(field)) {
    const vals = lgas
      .map((l) => safeNum(l[field]))
      .filter((v) => v != null)
      .sort((a, b) => a - b);
    fieldValuesCache.set(field, vals);
  }

  const vals = fieldValuesCache.get(field);
  if (!vals || !vals.length) return null;

  let low = 0;
  let high = vals.length - 1;
  let idx = vals.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (vals[mid] >= value) {
      idx = mid;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return Math.round((idx / Math.max(vals.length - 1, 1)) * 100);
}

function renderDetail() {
  const inner = document.getElementById('detail-inner');
  if (!inner) return;

  if (!selectedLGA) {
    inner.replaceChildren();
    return;
  }

  const lga = selectedLGA;
  const distPct = 100 - (percentileRank('dist', safeNum(lga.dist)) ?? 50);
  const facPct = percentileRank('fac', safeNum(lga.fac)) ?? 50;
  const u5Pct = 100 - (percentileRank('u5mr', safeNum(lga.u5mr)) ?? 50);
  const covPct = percentileRank('cov', safeNum(lga.cov)) ?? 50;

  let action = 'Review these access barriers alongside local knowledge before making planning decisions.';
  if (safeNum(lga.fac) != null && safeNum(lga.dist) != null && Number(lga.fac) < 0.5 && Number(lga.dist) > 5) {
    action = 'Very few facilities and long travel times. Consider mobile clinic deployment.';
  } else if (safeNum(lga.u5mr) != null && Number(lga.u5mr) > 150) {
    action = 'Under-5 mortality is among the highest in the dataset. Cross-check with immunisation coverage.';
  } else if (safeNum(lga.cov) != null && Number(lga.cov) < 20) {
    action = 'Less than 20% of this LGA is within 5km of a facility. A new fixed post would significantly improve access.';
  }

  const shapRows = lga.shap
    ? Object.entries(lga.shap).sort((a, b) => Math.abs(Number(b[1] ?? 0)) - Math.abs(Number(a[1] ?? 0)))
    : [];

  const riskNum = safeNum(lga.risk);
  const conf = confidenceBadge(lga.confidence_pct);
  const riskClass = riskNum != null && riskNum > 0.66
    ? 'metric-risk-red'
    : riskNum != null && riskNum > 0.33
      ? 'metric-risk-yellow'
      : 'metric-risk-green';

  const pctRows = [
    { label: 'Facility access', pct: facPct },
    { label: 'Travel distance', pct: distPct },
    { label: 'Child survival', pct: u5Pct },
    { label: '5km coverage', pct: covPct },
  ];

  inner.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-lga">${escapeHtml(sanitizeText(lga.name, 'Unknown LGA'))}</div>
        <div class="detail-state-tag">
          ${escapeHtml(sanitizeText(lga.state, 'Unknown state'))} · Risk score:
          <span class="metric-risk ${riskClass}">${escapeHtml(riskLabel(riskNum))}</span>
        </div>
        <div class="detail-state-tag">Data confidence: ${conf.emoji} ${escapeHtml(conf.label)}</div>
      </div>
      <button type="button" class="close-btn" id="detail-close-btn" aria-label="Close details">×</button>
    </div>

    <div class="metric-grid">
      <div class="metric-cell">
        <div class="metric-label">Facilities / 10k</div>
        <div class="metric-value ${metricClass(lga.fac, 0.5, 1.5)}">${escapeHtml(fmtMetric(lga.fac))}</div>
        <div class="metric-unit">per 10,000 pop</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Avg distance</div>
        <div class="metric-value ${metricClass(lga.dist, 4, 8, true)}">${escapeHtml(fmtMetric(lga.dist))}</div>
        <div class="metric-unit">km to nearest</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">Under-5 mortality</div>
        <div class="metric-value ${metricClass(lga.u5mr, 80, 150, true)}">${escapeHtml(fmtMetric(lga.u5mr))}</div>
        <div class="metric-unit">per 1,000 births</div>
      </div>
      <div class="metric-cell">
        <div class="metric-label">5km coverage</div>
        <div class="metric-value ${metricClass(lga.cov, 20, 50)}">${escapeHtml(fmtMetric(lga.cov))}</div>
        <div class="metric-unit">area within 5km</div>
      </div>
    </div>

    <div class="section-label">vs Nigeria</div>
    <div class="pct-bars">
      ${pctRows.map((r) => {
        const pct = Math.max(0, Math.min(100, Number(r.pct ?? 0)));
        const fillClass = pct >= 66 ? 'better' : pct >= 33 ? 'mid' : 'worse';
        return `
          <div class="pct-row">
            <div class="pct-label">${escapeHtml(r.label)}</div>
            <div class="pct-track"><div class="pct-fill ${fillClass}" style="width:${pct}%"></div></div>
            <div class="pct-val">${pct}%</div>
          </div>
        `;
      }).join('')}
    </div>

    <div class="action-prompt">
      <p>${escapeHtml(action)}</p>
    </div>
    <p class="action-note">Decision-support only. Always combine with local knowledge and community input.</p>

    <div class="depth-section ${currentDepth >= 2 && shapRows.length ? 'visible' : ''}" id="shap-section">
      <div class="section-label">Feature contribution (SHAP)</div>
      ${shapRows.length
        ? shapRows.map(([k, v]) => {
          const n = safeNum(v) ?? 0;
          const width = Math.min(Math.abs(n) * 200, 100);
          const sign = n >= 0 ? '+' : '';
          return `
            <div class="shap-row">
              <div class="shap-feature">${escapeHtml(k)}</div>
              <div class="shap-track"><div class="shap-fill ${n >= 0 ? 'pos' : 'neg'}" style="width:${width}%"></div></div>
              <div class="shap-val">${sign}${n.toFixed(2)}</div>
            </div>
          `;
        }).join('')
        : '<div class="shap-row"><div class="shap-feature">No SHAP data</div></div>'}
      <button type="button" class="dl-btn" id="download-lga-btn">↓ Download LGA data (CSV)</button>
    </div>
  `;

  document.getElementById('detail-close-btn')?.addEventListener('click', closeDrawer);
  document.getElementById('download-lga-btn')?.addEventListener('click', downloadLGA);
}

function setDepth(depth) {
  currentDepth = Number(depth);
  applyDepthVisibility();
  syncHeader();
  if (selectedLGA) renderDetail();
  renderMap();
  pushStateToPython();
}

function setFocus(focus) {
  currentFocus = focus;
  syncHeader();
  pushStateToPython();
}

function selectLGA(id) {
  const base = lgaById.get(String(id));
  if (!base) return;

  selectedLGA = { ...base };
  if (injected.selected && String(injected.selected.id) === String(id)) {
    selectedLGA = { ...selectedLGA, ...injected.selected };
  }

  renderHotspots();
  renderDetail();
  openDrawer();
  renderMap();
  pushStateToPython();
}

function addCompareSlot() {
  if (!selectedLGA) return;
  if (compareLGAs.find((l) => String(l.id) === String(selectedLGA.id))) return;
  if (compareLGAs.length >= 4) return;

  compareLGAs.push(selectedLGA);
  renderCompareSlots();
  pushStateToPython();
}

function removeCompare(id) {
  compareLGAs = compareLGAs.filter((l) => String(l.id) !== String(id));
  renderCompareSlots();
  pushStateToPython();
}

function renderCompareSlots() {
  const slots = document.getElementById('compare-slots');
  const compareBtn = document.getElementById('compare-go-btn');
  if (!slots || !compareBtn) return;

  slots.replaceChildren();
  compareLGAs.forEach((l) => {
    const slot = document.createElement('div');
    slot.className = 'compare-slot filled';

    const label = document.createElement('span');
    label.textContent = sanitizeText(l.name, 'Unknown LGA');

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'compare-remove';
    removeBtn.textContent = '×';
    removeBtn.setAttribute('aria-label', `Remove ${sanitizeText(l.name, 'LGA')} from comparison`);
    removeBtn.addEventListener('click', () => removeCompare(l.id));

    slot.append(label, removeBtn);
    slots.appendChild(slot);
  });

  if (compareLGAs.length < 4) {
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'compare-slot compare-add-btn';
    addBtn.textContent = '+ Add LGA';
    addBtn.addEventListener('click', addCompareSlot);
    slots.appendChild(addBtn);
  }

  compareBtn.disabled = compareLGAs.length < 2;
}

function runCompare() {
  const names = compareLGAs.map((l) => sanitizeText(l.name, 'LGA'));
  alert(`Compare view: ${names.join(' vs ')}`);
}

function initMap() {
  if (mapInstance) return;
  mapInstance = L.map('map-leaflet', {
    zoomControl: true,
    attributionControl: false,
    minZoom: 5,
    maxZoom: 12,
  }).setView([9.1, 8.7], 6);

  if (mapInstance.fullscreenControl == null && L.Control.Fullscreen) {
    mapInstance.addControl(new L.Control.Fullscreen({ position: 'topleft' }));
  }
}

function normalizeGeoJson() {
  if (baseGeoJson) return baseGeoJson;

  let gj = injected.map?.geojson ? JSON.parse(injected.map.geojson) : null;
  if (!gj) return null;

  if (Array.isArray(gj.features) && typeof turf !== 'undefined') {
    try {
      gj = {
        ...gj,
        features: gj.features.map((f) => turf.transformScale(f, 0.8, { origin: 'centroid' })),
      };
    } catch (e) {
      // fall back to original geometry
    }
  }

  baseGeoJson = gj;
  return baseGeoJson;
}

function valueForLayer(lga, layer) {
  if (!lga) return null;
  switch (layer) {
    case 'Facilities': return safeNum(lga.fac);
    case 'Population': return safeNum(lga.density ?? lga.pop);
    case 'Towers': return safeNum(lga.towers);
    case 'SHAP':
      if (!lga.shap) return null;
      return safeNum(Object.values(lga.shap)[0]);
    default: return safeNum(lga.risk);
  }
}

function layerLabel(layer) {
  switch (layer) {
    case 'Facilities': return 'Facilities / 10k';
    case 'Population': return 'Population';
    case 'Towers': return 'Towers / 10k';
    case 'SHAP': return 'SHAP';
    default: return 'Risk score';
  }
}

function getLayerRange(layer) {
  if (fieldValuesCache.has(`range:${layer}`)) return fieldValuesCache.get(`range:${layer}`);

  const values = lgas
    .map((l) => valueForLayer(l, layer))
    .filter((v) => v != null && !Number.isNaN(v));

  const range = {
    min: values.length ? Math.min(...values) : 0,
    max: values.length ? Math.max(...values) : 1,
  };
  fieldValuesCache.set(`range:${layer}`, range);
  return range;
}

function scaledLayerValue(id) {
  const lga = lgaById.get(String(id));
  const val = valueForLayer(lga, currentLayer);
  if (val == null || Number.isNaN(val)) return null;

  if (currentLayer === 'Risk score') {
    return riskLookup[String(id)] ?? val;
  }

  const range = getLayerRange(currentLayer);
  if (range.max === range.min) return 0.5;
  return (val - range.min) / (range.max - range.min);
}

function displayValue(id) {
  const lga = lgaById.get(String(id));
  const val = valueForLayer(lga, currentLayer);
  return currentLayer === 'Risk score' ? riskLabel(val) : fmtMetric(val);
}

function tooltipHtml(id, fallbackName, fallbackState) {
  const lga = lgaById.get(String(id));
  const safeName = escapeHtml(sanitizeText(lga?.name ?? fallbackName ?? id, 'Unknown LGA'));
  const safeState = escapeHtml(sanitizeText(lga?.state ?? fallbackState, ''));
  const safeLabel = escapeHtml(layerLabel(currentLayer));
  const safeVal = escapeHtml(displayValue(id));

  return `<div class="hd-tip"><div class="hd-tip-title">${safeName}</div><div class="hd-tip-sub">${safeState}</div><div class="hd-tip-risk">${safeLabel}: ${safeVal}</div></div>`;
}

function styleForFeatureId(id) {
  const norm = scaledLayerValue(id);
  const fill = riskColorHex(norm);
  const selected = selectedLGA && String(selectedLGA.id) === String(id);
  return {
    color: selected ? '#ffffff' : '#0f172a',
    weight: selected ? 2.5 : 0.25,
    fillColor: fill,
    fillOpacity: 0.45,
  };
}

function initGeoLayer() {
  if (!mapInstance || geoLayer) return;

  const gj = normalizeGeoJson();
  if (!gj) return;

  geoLayer = L.geoJSON(gj, {
    style: (feature) => {
      const id = String(feature.properties?.lga_uid ?? feature.properties?.lga_name ?? '');
      return styleForFeatureId(id);
    },
    onEachFeature: (feature, layer) => {
      const id = String(feature.properties?.lga_uid ?? feature.properties?.lga_name ?? '');
      const name = feature.properties?.lga_name;
      const state = feature.properties?.state_name;

      featureLayerById.set(id, layer);
      layer.bindTooltip(tooltipHtml(id, name, state), {
        sticky: true,
        direction: 'top',
        className: 'hd-tooltip',
      });
      layer.on('click', () => selectLGA(id));
    },
  }).addTo(mapInstance);

  if (!hasFitBounds) {
    try {
      mapInstance.fitBounds(geoLayer.getBounds(), { padding: [10, 10] });
      hasFitBounds = true;
    } catch (e) {
      // ignore fit errors
    }
  }
}

function renderMap() {
  initMap();
  if (!mapInstance) return;

  initGeoLayer();
  if (!geoLayer) return;

  featureLayerById.forEach((layer, id) => {
    layer.setStyle(styleForFeatureId(id));
    const feature = layer.feature || {};
    const name = feature.properties?.lga_name;
    const state = feature.properties?.state_name;
    layer.setTooltipContent(tooltipHtml(id, name, state));
  });
}

function csvSafe(value) {
  const text = sanitizeText(value, '');
  const prefixed = /^[=+\-@]/.test(text) ? `'${text}` : text;
  return `"${prefixed.replace(/"/g, '""')}"`;
}

function downloadLGA() {
  if (!selectedLGA) return;

  const l = selectedLGA;
  const row = [
    csvSafe(l.name),
    csvSafe(l.state),
    csvSafe(l.risk ?? ''),
    csvSafe(l.fac ?? ''),
    csvSafe(l.dist ?? ''),
    csvSafe(l.u5mr ?? ''),
    csvSafe(l.pop ?? ''),
    csvSafe(l.cov ?? ''),
  ].join(',');

  const csv = `lga_name,state,risk_score,facilities_per_10k,avg_distance_km,u5mr,population,coverage_5km_pct\n${row}`;
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${sanitizeText(l.name, 'lga').replace(/\s+/g, '_')}_health_data.csv`;
  a.click();
}

function toggleLayer(layerName) {
  currentLayer = layerName;

  document.querySelectorAll('.layer-btn').forEach((btn) => {
    const active = btn.dataset.layer === currentLayer;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });

  const legendTitle = document.querySelector('.legend-title');
  if (legendTitle) legendTitle.textContent = currentLayer;

  renderMap();
}

function wireEvents() {
  const stateSelect = document.getElementById('state-select');
  stateOptions.forEach((st) => {
    const opt = document.createElement('option');
    opt.value = st;
    opt.textContent = st;
    stateSelect.appendChild(opt);
  });
  stateSelect.value = currentState;
  stateSelect.addEventListener('change', (e) => {
    currentState = e.target.value;
    syncHeader();
    pushStateToPython();
  });

  const yearSelect = document.getElementById('year-select');
  yearSelect.value = currentYear;
  yearSelect.addEventListener('change', (e) => {
    currentYear = e.target.value;
    syncHeader();
    pushStateToPython();
  });

  const searchInput = document.getElementById('search-input');
  if (searchInput) searchInput.addEventListener('input', renderHotspots);

  document.querySelectorAll('.depth-btn').forEach((btn) => {
    btn.addEventListener('click', () => setDepth(btn.dataset.depth));
  });

  document.querySelectorAll('.chip').forEach((btn) => {
    btn.addEventListener('click', () => setFocus(btn.dataset.focus || 'All risk'));
  });

  document.querySelectorAll('.layer-btn').forEach((btn) => {
    btn.addEventListener('click', () => toggleLayer(btn.dataset.layer || 'Risk score'));
  });

  document.getElementById('compare-go-btn')?.addEventListener('click', runCompare);
  document.getElementById('compare-add-btn')?.addEventListener('click', addCompareSlot);
}

wireEvents();
syncHeader();
applyDepthVisibility();
renderHotspots();
renderCompareSlots();
renderMap();

if (selectedLGA) {
  openDrawer();
  renderDetail();
}
