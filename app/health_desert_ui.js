// Data injected from Streamlit
const injected = window.__INITIAL_DATA__ || {};
const meta = injected.meta || {};
const lgas = Array.isArray(injected.lgas) ? injected.lgas : [];
const hotspotsPayload = Array.isArray(injected.hotspots) ? injected.hotspots : [];
const stateOptions = ['All Nigeria', ...(injected.states || [])];
const urlParams = new URLSearchParams(window.parent.location.search);
const testingMode = ['1', 'true', 'yes'].includes((urlParams.get('testing') || '').toLowerCase());
const testPersona = urlParams.get('persona') || 'unknown';
let testSession = urlParams.get('session') || '';
if (testingMode && !testSession) {
  testSession = String(Date.now());
}

let currentState = meta.state_filter || 'All Nigeria';
let currentDepth = Number(meta.depth || 0);
let currentFocus = meta.focus || 'All risk';
let currentYear = meta.year != null ? String(meta.year) : '2018';
let currentLayer = 'Risk score';
let isMobile = false;
let pendingEvent = null;

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

const hasTowerConnectivityData = lgas.some((lga) => {
  const value = safeNum(lga.towers);
  return value != null && value > 0;
});

function riskLabel(r, total) {
  const t = safeNum(total);
  if (t != null) return t.toFixed(1);
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

  const footnote = document.getElementById('data-footnote');
  if (footnote) {
    const modelVersion = Array.isArray(meta.model_version) ? meta.model_version.join(', ') : meta.model_version;
    const updated = meta.data_last_updated ? `Updated ${meta.data_last_updated}` : 'Update date unknown';
    footnote.textContent = `DHS 2013/2018 · NHFR · OpenCellID · Model ${modelVersion || 'v1.2'} · ${updated}`;
  }

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
  params.set('mobile', isMobile ? '1' : '0');

  if (selectedLGA?.id) params.set('lga', String(selectedLGA.id));
  else params.delete('lga');

  if (compareLGAs.length) params.set('compare', compareLGAs.map((l) => l.id).join(','));
  else params.delete('compare');

  if (testingMode) {
    params.set('testing', '1');
    if (testPersona) params.set('persona', testPersona);
    if (testSession) params.set('session', testSession);
    if (pendingEvent) {
      params.set('evt', JSON.stringify(pendingEvent));
    } else {
      params.delete('evt');
    }
  }

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
  pendingEvent = null;
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

function queueEvent(type, details = {}) {
  if (!testingMode) return;
  pendingEvent = { type, details };
  pushStateToPython();
}

function hotspotsBase() {
  if (hotspotsPayload.length) return hotspotsPayload;
  return [...lgas]
    .sort((a, b) => {
      const aScore = safeNum(a.risk_total) != null ? safeNum(a.risk_total) / 10 : Number(a.risk ?? 0);
      const bScore = safeNum(b.risk_total) != null ? safeNum(b.risk_total) / 10 : Number(b.risk ?? 0);
      return bScore - aScore;
    })
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
    const riskTotal = safeNum(lga.risk_total);
    const riskScore = riskTotal != null ? riskTotal / 10 : risk;
    const bucket = riskScore != null && riskScore > 0.66 ? 'risk-high' : riskScore != null && riskScore > 0.33 ? 'risk-med' : 'risk-low';
    badge.className = `risk-badge ${bucket}`;
    badge.textContent = riskLabel(risk, riskTotal);

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

function buildInterventions(lga) {
  const mapping = {
    'Low facility density': [
      'Mobile clinic outreach',
      'Primary health center upgrades',
      'Community health worker expansion',
    ],
    'Limited facility proximity coverage': [
      'New fixed posts in low-coverage wards',
      'Targeted outreach for isolated settlements',
      'Transport support for referrals',
    ],
    'Elevated under-5 mortality indicators': [
      'Maternal and child health outreach',
      'Immunization drives',
      'Nutrition and antenatal support',
    ],
    'Limited mobile network coverage': [
      'Offline-first health data tools',
      'Community radio health programs',
      'USSD or SMS service channels',
    ],
  };

  const barriers = String(lga.primary_barriers || '')
    .split('|')
    .map((b) => b.trim())
    .filter(Boolean);

  const items = [];
  barriers.forEach((barrier) => {
    if (mapping[barrier]) {
      items.push({ title: barrier, actions: mapping[barrier] });
    }
  });

  if (!items.length) {
    return '';
  }

  return items
    .map((item) => `
      <div class=\"intervention-card\">
        <div class=\"intervention-title\">${escapeHtml(item.title)}</div>
        <ul class=\"intervention-list\">
          ${item.actions.map((act) => `<li>${escapeHtml(act)}</li>`).join('')}
        </ul>
      </div>
    `)
    .join('');
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

  let action = lga.recommendation || 'Review these access barriers alongside local knowledge before making planning decisions.';
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
  const riskTotal = safeNum(lga.risk_total);
  const conf = confidenceBadge(lga.confidence_pct);
  const riskScore = riskTotal != null ? riskTotal / 10 : riskNum;
  const riskClass = riskScore != null && riskScore > 0.66
    ? 'metric-risk-red'
    : riskScore != null && riskScore > 0.33
      ? 'metric-risk-yellow'
      : 'metric-risk-green';

  const pctRows = [
    { label: 'Facility access', pct: facPct },
    { label: 'Travel distance', pct: distPct },
    { label: 'Child survival', pct: u5Pct },
    { label: '5km coverage', pct: covPct },
  ];

  const interventionsHtml = buildInterventions(lga);
  const interventionsSection = interventionsHtml
    ? `
    <div class="section-label">What can help</div>
    <div class="intervention-grid">
      ${interventionsHtml}
    </div>
  `
    : '';

  inner.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-lga">${escapeHtml(sanitizeText(lga.name, 'Unknown LGA'))}</div>
        <div class="detail-state-tag">
          ${escapeHtml(sanitizeText(lga.state, 'Unknown state'))} · Risk score:
          <span class="metric-risk ${riskClass}">${escapeHtml(riskLabel(riskNum, riskTotal))}</span>
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

    ${interventionsSection}

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
  queueEvent('depth_change', { depth: currentDepth });
}

function setFocus(focus) {
  currentFocus = focus;
  syncHeader();
  pushStateToPython();
  queueEvent('focus_change', { focus: currentFocus });
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
  queueEvent('lga_select', { id: String(id), name: selectedLGA?.name });
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
    case 'Connectivity':
      // Fall back to 5km coverage proxy when tower feed is unavailable in this release.
      return hasTowerConnectivityData ? safeNum(lga.towers) : safeNum(lga.cov);
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
    case 'Connectivity': return hasTowerConnectivityData ? 'Towers / 10k' : 'Connectivity (5km coverage)';
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

function renderMapTable() {
  const tableWrap = document.getElementById('map-table');
  if (!tableWrap) return;
  const rows = lgas.slice(0, 200);
  const header = `
    <thead>
      <tr>
        <th>LGA</th>
        <th>State</th>
        <th>Risk score</th>
        <th>Confidence</th>
      </tr>
    </thead>
  `;
  const body = `
    <tbody>
      ${rows.map((lga) => {
        const riskTotal = safeNum(lga.risk_total);
        const risk = riskLabel(lga.risk, riskTotal);
        return `
          <tr>
            <td>${escapeHtml(sanitizeText(lga.name, 'Unknown'))}</td>
            <td>${escapeHtml(sanitizeText(lga.state, ''))}</td>
            <td>${escapeHtml(risk)}</td>
            <td>${escapeHtml(String(safeNum(lga.confidence_pct) ?? '—'))}</td>
          </tr>
        `;
      }).join('')}
    </tbody>
  `;
  tableWrap.innerHTML = `<table class="map-table-inner">${header}${body}</table>`;
}

function toggleMapTable() {
  const tableWrap = document.getElementById('map-table');
  const toggleBtn = document.getElementById('map-table-toggle');
  if (!tableWrap || !toggleBtn) return;
  const isOpen = tableWrap.classList.toggle('open');
  tableWrap.setAttribute('aria-hidden', String(!isOpen));
  toggleBtn.setAttribute('aria-expanded', String(isOpen));
  toggleBtn.textContent = isOpen ? 'Hide map table' : 'View map as table';
  if (isOpen) renderMapTable();
}

let currentExportMode = 'csv';

function buildShareUrl() {
  const params = new URLSearchParams();
  params.set('state', currentState);
  params.set('focus', currentFocus);
  params.set('depth', String(currentDepth));
  params.set('year', currentYear);
  if (selectedLGA?.id) params.set('lga', String(selectedLGA.id));
  if (compareLGAs.length) params.set('compare', compareLGAs.map((l) => l.id).join(','));
  params.set('mobile', isMobile ? '1' : '0');
  return `${window.parent.location.origin}/static/share_preview.html?${params.toString()}`;
}

function updateShareDrawer() {
  const shareUrl = buildShareUrl();
  const urlEl = document.getElementById('share-url');
  if (urlEl) urlEl.textContent = shareUrl;
  const x = document.getElementById('share-x');
  const li = document.getElementById('share-linkedin');
  const wa = document.getElementById('share-whatsapp');
  if (x) x.href = `https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}`;
  if (li) li.href = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
  if (wa) wa.href = `https://wa.me/?text=${encodeURIComponent(shareUrl)}`;
}

function openOverlay(id) {
  const overlay = document.getElementById(id);
  if (overlay) {
    overlay.classList.add('open');
    overlay.setAttribute('aria-hidden', 'false');
  }
}

function closeOverlay(id) {
  const overlay = document.getElementById(id);
  if (overlay) {
    overlay.classList.remove('open');
    overlay.setAttribute('aria-hidden', 'true');
  }
}

function buildExportMetadata() {
  const modelVersion = Array.isArray(meta.model_version) ? meta.model_version.join(', ') : meta.model_version || 'v1.2';
  const updated = meta.data_last_updated || 'Unknown';
  return [
    '# Health Desert Scorer - Data Export',
    `# Generated: ${new Date().toISOString()}`,
    `# Filters: State=${currentState}, Year=${currentYear}, Focus=${currentFocus}`,
    `# LGAs included: ${lgas.length}`,
    '#',
    '# IMPORTANT DISCLAIMER:',
    '# This is a planning tool output. Scores indicate access barriers, not health outcomes.',
    '# Always validate with local knowledge before decisions.',
    '#',
    '# Data Sources: DHS 2013, 2018 · NHFR 2020 · WorldPop 2020 · OpenCellID 2019',
    `# Model: ${modelVersion}`,
    `# Data last updated: ${updated}`,
    '# Citation: Bello, B.A. (2026). Health Desert Scorer.',
    '',
  ];
}

function exportFieldDefs() {
  return [
    { key: 'name', label: 'lga_name' },
    { key: 'state', label: 'state_name' },
    { key: 'year', label: 'year' },
    { key: 'risk_total', label: 'risk_score_total' },
    { key: 'risk', label: 'risk_score' },
    { key: 'fac', label: 'facilities_per_10k' },
    { key: 'dist', label: 'avg_distance_km' },
    { key: 'u5mr', label: 'u5mr_mean' },
    { key: 'cov', label: 'coverage_5km' },
    { key: 'towers', label: 'towers_per_10k' },
    { key: 'confidence_pct', label: 'confidence_pct' },
    { key: 'confidence_reason_codes', label: 'confidence_reason_codes' },
    { key: 'primary_barriers', label: 'primary_barriers' },
    { key: 'recommendation', label: 'recommendation' },
  ];
}

function buildExportRows() {
  return lgas.map((lga) => {
    const row = { ...lga };
    row.risk_total = safeNum(lga.risk_total);
    row.risk = safeNum(lga.risk);
    return row;
  });
}

function buildCsvExport() {
  const headers = exportFieldDefs().map((f) => f.label).join(',');
  const rows = buildExportRows().map((row) =>
    exportFieldDefs().map((f) => csvSafe(row[f.key] ?? '')).join(',')
  );
  const csv = [...buildExportMetadata(), headers, ...rows].join('\n');
  return { name: `health_desert_${currentState}_${currentYear}.csv`, data: csv, type: 'text/csv' };
}

function buildGeoJsonExport() {
  const gj = normalizeGeoJson();
  if (!gj) {
    return { name: 'health_desert.geojson', data: JSON.stringify({ type: 'FeatureCollection', features: [] }), type: 'application/geo+json' };
  }
  const dataLookup = new Map(lgas.map((l) => [String(l.id), l]));
  const metadata = {
    export_date: new Date().toISOString(),
    filters: { state: currentState, year: currentYear, focus: currentFocus },
    model_version: meta.model_version || 'v1.2',
  };
  const features = gj.features.map((f) => {
    const id = String(f.properties?.lga_uid ?? f.properties?.lga_name ?? '');
    const data = dataLookup.get(id);
    return {
      ...f,
      properties: {
        ...f.properties,
        ...(data || {}),
        export_metadata: JSON.stringify(metadata),
      },
    };
  });
  const out = { ...gj, features };
  return { name: `health_desert_${currentState}_${currentYear}.geojson`, data: JSON.stringify(out), type: 'application/geo+json' };
}

function buildSummaryExport() {
  const rows = buildExportRows();
  const scores = rows.map((r) => safeNum(r.risk_total) ?? (safeNum(r.risk) != null ? safeNum(r.risk) * 10 : null)).filter((v) => v != null);
  const avg = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length) : 0;
  const high = scores.filter((v) => v >= 7).length;
  const medium = scores.filter((v) => v >= 4 && v < 7).length;
  const low = scores.filter((v) => v < 3).length;
  const top = [...rows].sort((a, b) => (safeNum(b.risk_total) ?? 0) - (safeNum(a.risk_total) ?? 0)).slice(0, 10);

  let report = '';
  report += 'HEALTH DESERT SCORER - SUMMARY REPORT\\n';
  report += '========================================\\n\\n';
  report += `Generated: ${new Date().toISOString()}\\n`;
  report += `Geographic Scope: ${currentState}\\n`;
  report += `Year: ${currentYear}\\n`;
  report += `Focus Mode: ${currentFocus}\\n\\n`;
  report += 'SUMMARY STATISTICS\\n';
  report += '----------------------------------------\\n';
  report += `Total LGAs: ${rows.length}\\n`;
  report += `Average Risk Score: ${avg.toFixed(2)} / 10\\n`;
  report += `High Risk (7-10): ${high}\\n`;
  report += `Medium Risk (4-6): ${medium}\\n`;
  report += `Lower Risk (0-3): ${low}\\n\\n`;
  report += 'TOP 10 HIGHEST-NEED LGAs\\n';
  report += '----------------------------------------\\n';
  top.forEach((row, idx) => {
    const score = safeNum(row.risk_total) ?? (safeNum(row.risk) != null ? safeNum(row.risk) * 10 : null);
    report += `${idx + 1}. ${row.name} (${row.state}) - ${score != null ? score.toFixed(2) : 'NA'}\\n`;
  });
  report += '\\nIMPORTANT DISCLAIMER\\n';
  report += 'This is a planning tool output, not a diagnosis system.\\n';
  report += 'Always validate with local knowledge and field checks.\\n';
  return { name: `health_desert_report_${currentState}_${currentYear}.txt`, data: report, type: 'text/plain' };
}

async function buildBundleExport() {
  const zip = new JSZip();
  const csv = buildCsvExport();
  const geojson = buildGeoJsonExport();
  const summary = buildSummaryExport();
  zip.file('data.csv', csv.data);
  zip.file('data.geojson', geojson.data);
  zip.file('summary_report.txt', summary.data);
  zip.file('README.txt', `Health Desert Scorer Export Bundle\\nGenerated: ${new Date().toISOString()}\\nFilters: ${currentState} · ${currentYear} · ${currentFocus}\\n`);
  const content = await zip.generateAsync({ type: 'blob' });
  return { name: `health_desert_bundle_${currentState}_${currentYear}.zip`, data: content, type: 'application/zip' };
}

async function downloadExport() {
  let payload;
  if (currentExportMode === 'csv') payload = buildCsvExport();
  if (currentExportMode === 'geojson') payload = buildGeoJsonExport();
  if (currentExportMode === 'summary') payload = buildSummaryExport();
  if (currentExportMode === 'bundle') payload = await buildBundleExport();
  if (!payload) return;

  const blob = payload.data instanceof Blob ? payload.data : new Blob([payload.data], { type: payload.type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = payload.name;
  a.click();
  queueEvent('export', { format: currentExportMode });
}

function setExportMode(mode) {
  currentExportMode = mode;
  document.querySelectorAll('.export-chip').forEach((chip) => {
    const active = chip.dataset.export === mode;
    chip.classList.toggle('active', active);
    chip.setAttribute('aria-pressed', String(active));
  });
  const help = document.getElementById('export-help');
  if (!help) return;
  const text = {
    csv: 'CSV is compatible with Excel and Google Sheets.',
    geojson: 'GeoJSON works in GIS tools like QGIS or ArcGIS.',
    summary: 'Summary is a text report for proposals and briefs.',
    bundle: 'Bundle includes CSV, GeoJSON, summary, and README.',
  };
  help.textContent = text[mode] || '';
}

const tourSteps = [
  {
    title: 'Welcome',
    body: 'This tool highlights LGAs facing healthcare access barriers. It is designed for planning, not diagnosis.',
  },
  {
    title: 'Step 1 of 4: The map',
    body: 'The map shows relative access barriers. Click any LGA to see details and confidence.',
  },
  {
    title: 'Step 2 of 4: Filters',
    body: 'Use State, Year, and Focus to explore different access barriers.',
  },
  {
    title: 'Step 3 of 4: Highest-need list',
    body: 'The list ranks LGAs by the selected focus so you can prioritize outreach.',
  },
  {
    title: 'Step 4 of 4: Ready',
    body: 'Try selecting a state, review the top LGAs, and export a list for planning.',
  },
];

let tourIndex = 0;

function openTour() {
  const overlay = document.getElementById('tour-overlay');
  if (!overlay) return;
  overlay.classList.add('open');
  overlay.setAttribute('aria-hidden', 'false');
  renderTourStep();
}

function closeTour(markComplete = false) {
  const overlay = document.getElementById('tour-overlay');
  if (!overlay) return;
  overlay.classList.remove('open');
  overlay.setAttribute('aria-hidden', 'true');
  if (markComplete) {
    localStorage.setItem('hd_tour_v1_completed', '1');
  }
}

function renderTourStep() {
  const titleEl = document.getElementById('tour-step-title');
  const bodyEl = document.getElementById('tour-step-body');
  if (!titleEl || !bodyEl) return;
  const step = tourSteps[tourIndex] || tourSteps[0];
  titleEl.textContent = step.title;
  bodyEl.textContent = step.body;
  const nextBtn = document.getElementById('tour-next-btn');
  if (nextBtn) nextBtn.textContent = tourIndex >= tourSteps.length - 1 ? 'Start' : 'Next';
}

function maybeStartTour() {
  const completed = localStorage.getItem('hd_tour_v1_completed') === '1';
  if (!completed) openTour();
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
  queueEvent('export', { format: 'lga_csv' });
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

function applyHelpTooltips() {
  document.querySelectorAll('.help-icon').forEach((icon) => {
    const help = icon.getAttribute('data-help');
    if (help) {
      icon.setAttribute('title', help);
      icon.setAttribute('aria-label', help);
    }
  });
}

function detectMobile() {
  const next = window.matchMedia('(max-width: 768px)').matches;
  if (next !== isMobile) {
    isMobile = next;
    document.body.classList.toggle('is-mobile', isMobile);
    pushStateToPython({ immediate: true });
  } else {
    document.body.classList.toggle('is-mobile', isMobile);
  }
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
    queueEvent('filter_change', { state: currentState });
  });

  const yearSelect = document.getElementById('year-select');
  yearSelect.value = currentYear;
  yearSelect.addEventListener('change', (e) => {
    currentYear = e.target.value;
    syncHeader();
    pushStateToPython();
    queueEvent('filter_change', { year: currentYear });
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

  document.getElementById('map-table-toggle')?.addEventListener('click', toggleMapTable);
  const mapToggle = document.getElementById('map-table-toggle');
  if (mapToggle) mapToggle.setAttribute('aria-expanded', 'false');

  document.getElementById('share-open-btn')?.addEventListener('click', () => {
    updateShareDrawer();
    openOverlay('share-drawer');
    queueEvent('share', { method: 'open' });
  });
  document.getElementById('share-close-btn')?.addEventListener('click', () => closeOverlay('share-drawer'));
  document.getElementById('copy-share-btn')?.addEventListener('click', async () => {
    const url = buildShareUrl();
    try {
      await navigator.clipboard.writeText(url);
    } catch (e) {
      // ignore clipboard errors
    }
    queueEvent('share', { method: 'copy' });
  });
  document.getElementById('share-x')?.addEventListener('click', () => queueEvent('share', { method: 'x' }));
  document.getElementById('share-linkedin')?.addEventListener('click', () => queueEvent('share', { method: 'linkedin' }));
  document.getElementById('share-whatsapp')?.addEventListener('click', () => queueEvent('share', { method: 'whatsapp' }));

  document.getElementById('export-open-btn')?.addEventListener('click', () => {
    setExportMode(currentExportMode);
    openOverlay('export-drawer');
  });
  document.getElementById('export-close-btn')?.addEventListener('click', () => closeOverlay('export-drawer'));
  document.querySelectorAll('.export-chip').forEach((chip) => {
    chip.addEventListener('click', () => setExportMode(chip.dataset.export || 'csv'));
  });
  document.getElementById('export-download-btn')?.addEventListener('click', downloadExport);

  document.getElementById('tour-restart-btn')?.addEventListener('click', () => {
    tourIndex = 0;
    openTour();
  });
  document.getElementById('tour-skip-btn')?.addEventListener('click', () => closeTour(true));
  document.getElementById('tour-next-btn')?.addEventListener('click', () => {
    if (tourIndex >= tourSteps.length - 1) {
      closeTour(true);
      return;
    }
    tourIndex += 1;
    renderTourStep();
  });

  applyHelpTooltips();

  const methodLink = document.getElementById('methodology-link');
  const glossaryLink = document.getElementById('glossary-link');
  [methodLink, glossaryLink].forEach((link) => {
    if (!link) return;

    const rawHref = link.getAttribute('href') || '';
    const parsed = new URL(rawHref, `${window.parent.location.origin}${window.parent.location.pathname}`);
    const navUrl = new URL(parsed.href);
    if (testingMode) {
      navUrl.searchParams.set('testing', '1');
      if (testPersona) navUrl.searchParams.set('persona', testPersona);
      if (testSession) navUrl.searchParams.set('session', testSession);
    }

    const targetHref = navUrl.search ? `${navUrl.pathname}?${navUrl.searchParams.toString()}` : navUrl.pathname;
    link.setAttribute('href', targetHref);
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });
}

wireEvents();
detectMobile();
window.addEventListener('resize', detectMobile);
if (testingMode && testSession) {
  pushStateToPython({ immediate: true });
}
syncHeader();
applyDepthVisibility();
renderHotspots();
renderCompareSlots();
renderMap();
maybeStartTour();

if (selectedLGA) {
  openDrawer();
  renderDetail();
}

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  closeOverlay('share-drawer');
  closeOverlay('export-drawer');
  closeTour(false);
});
