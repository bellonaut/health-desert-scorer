// Detect mobile early and round-trip state to Streamlit before app init.
(function detectAndReportMobile() {
  const isMobileNow = window.innerWidth <= 768
    || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '');

  document.documentElement.setAttribute('data-mobile', isMobileNow ? '1' : '0');
  if (isMobileNow) document.body.classList.add('is-mobile');
  else document.body.classList.remove('is-mobile');

  try {
    const url = new URL(window.parent.location.href);
    const current = url.searchParams.get('mobile');
    const next = isMobileNow ? '1' : '0';
    if (current !== next) {
      url.searchParams.set('mobile', next);
      window.parent.history.replaceState({}, '', url.toString());
    }
    window.parent.postMessage({ type: 'hd_mobile_detected', mobile: next }, '*');
  } catch (e) {
    // Cross-origin iframe restrictions: no-op.
  }
})();

function getParentUrl() {
  try {
    if (window.parent?.location?.href) {
      return new URL(window.parent.location.href);
    }
  } catch (e) {
    // iframe parent access may be blocked by sandbox/navigation policy
  }

  try {
    if (document.referrer) {
      return new URL(document.referrer);
    }
  } catch (e) {
    // ignore invalid or unavailable referrer values
  }

  try {
    return new URL(window.location.href);
  } catch (e) {
    return null;
  }
}

function getParentDocument() {
  try {
    if (window.parent && window.parent !== window) {
      return window.parent.document;
    }
  } catch (e) {
    // ignore blocked parent DOM access
  }
  return null;
}

function replaceParentLocation(nextUrl) {
  try {
    if (window.parent?.location) {
      window.parent.location.replace(nextUrl);
      return true;
    }
  } catch (e) {
    // ignore blocked parent navigation
  }
  return false;
}

// Data injected from Streamlit or hydrated for standalone mode.
if (!window.__INITIAL_DATA__) {
  const standaloneUrl = new URL(window.location.href);
  const standaloneParams = standaloneUrl.searchParams;
  const year = standaloneParams.get('year') || '2024';
  const focus = standaloneParams.get('focus') || 'All risk';
  const state = standaloneParams.get('state') || 'All Nigeria';
  const query = new URLSearchParams({ year, focus, state });
  const hotspotList = document.getElementById('hotspot-list');
  const statusText = document.getElementById('apply-status-text');
  const countText = document.getElementById('lga-count');
  const resNote = document.querySelector('.res-note');
  const mapArea = document.querySelector('.map-area');

  if (statusText) statusText.textContent = 'loading';
  if (countText) countText.textContent = '...';
  if (resNote) resNote.textContent = 'Loading map data...';
  if (hotspotList) {
    hotspotList.innerHTML = '<div class="hotspot-empty">Loading map data...</div>';
  }
  if (mapArea && !document.getElementById('map-empty-state')) {
    const empty = document.createElement('div');
    empty.id = 'map-empty-state';
    empty.className = 'map-empty-state';
    empty.textContent = 'Loading map data...';
    mapArea.appendChild(empty);
  }

  fetch(`/api/data?${query.toString()}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Standalone hydration failed: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      window.__INITIAL_DATA__ = data;
      applyPayloadData(data, { preserveSelection: false, preserveCompare: false });
      attachStandaloneMessageHandler();
      onLeafletReady(() => bootApp());
    })
    .catch((error) => {
      console.error('[HDS] standalone hydration error:', error);
      if (statusText) statusText.textContent = 'error';
      if (hotspotList) {
        hotspotList.innerHTML = '<div class="hotspot-empty">Failed to load map data. Try refreshing.</div>';
      }
      const empty = document.getElementById('map-empty-state');
      if (empty) empty.textContent = 'Failed to load map data. Try refreshing.';
    });
}

const APP_STATE_MESSAGE_TYPE = 'hd_apply_state';
const standaloneMode = window.parent === window;
const discoveredParentUrl = getParentUrl();
let configuredParentUrl = null;

let injected = {};
let meta = {};
let lgas = [];
let hotspotsPayload = [];
let allLgasForSearch = [];
let stateOptions = ['All Nigeria'];
let currentState = 'All Nigeria';
let currentDepth = 0;
let currentFocus = 'All risk';
let currentYear = '2024';
let currentLayer = 'Risk score';
let currentMapMode = 'polygon';
let isMobile = document.documentElement.getAttribute('data-mobile') === '1';
let pendingEvent = null;
let bootstrappingLatestYear = false;
let selectedLGA = null;
let compareLGAs = [];
let hasTowerConnectivityData = false;
let mapInstance = null;
let geoLayer = null;
let baseGeoJson = null;
let hasFitBounds = false;
let baseTileLayer = null;
let pendingMapInitFrame = null;
let standaloneMessageHandlerAttached = false;
let standaloneRefreshToken = 0;
let standaloneFetchController = null;

const lgaById = new Map();
const featureLayerById = new Map();
const fieldValuesCache = new Map();
const riskLookup = {};

function onLeafletReady(callback) {
  if (window.L) {
    callback();
    return;
  }
  window.addEventListener('hd:leaflet-ready', callback, { once: true });
}

function buildConfiguredParentUrl() {
  if (!meta.parent_app_path) return null;

  let baseOrigin = 'http://localhost:8501';
  if (discoveredParentUrl?.origin) {
    baseOrigin = discoveredParentUrl.origin;
  } else {
    try {
      if (document.referrer) {
        baseOrigin = new URL(document.referrer).origin;
      }
    } catch (e) {
      // ignore referrer parse issues and keep fallback origin
    }
  }

  try {
    return new URL(meta.parent_app_path, `${baseOrigin}/`);
  } catch (e) {
    return null;
  }
}

function refreshConfiguredParentUrl() {
  configuredParentUrl = buildConfiguredParentUrl();
}

function getAppStateBaseUrl() {
  if (configuredParentUrl) return new URL(configuredParentUrl.toString());

  const liveParent = getParentUrl();
  if (liveParent) return new URL(liveParent.toString());
  return null;
}

const parentUrl = getAppStateBaseUrl();
const urlParams = parentUrl ? new URLSearchParams(parentUrl.search) : new URLSearchParams();
const hasExplicitYearParam = urlParams.has('year');
const pwaMode = ['1', 'true', 'yes'].includes((urlParams.get('pwa') || '').toLowerCase());
const testingMode = ['1', 'true', 'yes'].includes((urlParams.get('testing') || '').toLowerCase());
const testPersona = urlParams.get('persona') || 'unknown';
let testSession = urlParams.get('session') || '';
if (testingMode && !testSession) {
  testSession = String(Date.now());
}

function normalizeCompareIds(value) {
  if (Array.isArray(value)) return value.map((item) => String(item)).filter(Boolean);
  if (typeof value === 'string') return value.split(',').map((item) => item.trim()).filter(Boolean);
  return [];
}

function resetMapDataState() {
  baseGeoJson = null;
  hasFitBounds = false;
  featureLayerById.clear();
  if (geoLayer && mapInstance) {
    try {
      mapInstance.removeLayer(geoLayer);
    } catch (e) {
      // ignore layer removal issues during data refresh
    }
  }
  geoLayer = null;
}

function applyPayloadData(data, { preserveSelection = true, preserveCompare = true } = {}) {
  const preservedSelectedId = preserveSelection ? String(selectedLGA?.id || meta.selected_lga || '') : '';
  const preservedCompareIds = preserveCompare ? compareLGAs.map((item) => String(item.id)) : [];

  injected = data || {};
  meta = injected.meta || {};
  lgas = Array.isArray(injected.lgas) ? injected.lgas : [];
  hotspotsPayload = Array.isArray(injected.hotspots) ? injected.hotspots : [];
  allLgasForSearch = Array.isArray(injected.all_lgas_for_search) ? injected.all_lgas_for_search : [];
  stateOptions = ['All Nigeria', ...(injected.states || [])];

  currentState = meta.state_filter || currentState || 'All Nigeria';
  currentDepth = Number(meta.depth ?? currentDepth ?? 0);
  if (Number.isNaN(currentDepth) || currentDepth < 0) currentDepth = 0;
  if (currentDepth > 1) currentDepth = 1;
  currentFocus = meta.focus || currentFocus || 'All risk';
  currentYear = meta.year != null ? String(meta.year) : (currentYear || '2024');

  refreshConfiguredParentUrl();

  lgaById.clear();
  lgas.forEach((lga) => {
    lgaById.set(String(lga.id), lga);
  });

  fieldValuesCache.clear();
  Object.keys(riskLookup).forEach((key) => delete riskLookup[key]);
  (injected.map?.choropleth || []).forEach((item) => {
    const val = item.risk_norm ?? item.risk;
    riskLookup[String(item.id)] = val != null ? Number(val) : null;
  });

  hasTowerConnectivityData = lgas.some((lga) => {
    const value = safeNum(lga.towers);
    return value != null && value > 0;
  });

  const compareIds = normalizeCompareIds(meta.compare_lgas).length
    ? normalizeCompareIds(meta.compare_lgas)
    : preservedCompareIds;
  compareLGAs = compareIds
    .map((id) => lgaById.get(String(id)))
    .filter(Boolean)
    .map((item) => ({ ...item }));

  const selectedId = meta.selected_lga || (injected.selected && injected.selected.id) || preservedSelectedId;
  if (selectedId) {
    const base = lgaById.get(String(selectedId)) || {};
    const detail = injected.selected && String(injected.selected.id) === String(selectedId)
      ? injected.selected
      : {};
    const merged = mergeLga(base, detail);
    selectedLGA = Object.keys(merged).length ? merged : null;
  } else {
    selectedLGA = null;
  }
}

if (window.__INITIAL_DATA__) {
  applyPayloadData(window.__INITIAL_DATA__, { preserveSelection: false, preserveCompare: false });
}

if (!hasExplicitYearParam && currentYear !== '2024') {
  try {
    const url = getAppStateBaseUrl();
    if (!url) throw new Error('Parent URL unavailable');
    url.searchParams.set('year', '2024');
    bootstrappingLatestYear = true;
    replaceParentLocation(url.toString());
  } catch (e) {
    // Ignore cross-origin or history access issues and continue with injected state.
  }
}

function mergeLga(base, detail) {
  const merged = { ...(base || {}), ...(detail || {}) };
  if ((merged.shap == null) && base && base.shap != null) {
    merged.shap = base.shap;
  }
  return merged;
}

const BASE_TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';
const TOUR_STORAGE_KEY = 'hd_tour_v2_completed';
const STATUS_COLOR_GOOD = '#0072B2';
const STATUS_COLOR_WARN = '#E69F00';
const STATUS_COLOR_BAD = '#D55E00';
const TEXT_COLOR_MUTED = '#C2CCD9';
const TEXT_COLOR_DIM = '#A5B0C2';
const SCORE_COLOR_LOW = '#8ED3FF';
const SCORE_COLOR_HIGH = '#FF9B6B';
const STATE_BOUNDS = {
  Abia: { s: 4.7, w: 7.1, n: 5.9, e: 8.1 },
  Adamawa: { s: 7.8, w: 11.5, n: 10.9, e: 13.7 },
  'Akwa Ibom': { s: 4.3, w: 7.4, n: 5.3, e: 8.6 },
  Anambra: { s: 5.7, w: 6.6, n: 6.8, e: 7.3 },
  Bauchi: { s: 9.3, w: 8.8, n: 12.3, e: 11.1 },
  Bayelsa: { s: 4.1, w: 5.7, n: 5.1, e: 6.8 },
  Benue: { s: 6.2, w: 7.7, n: 8.2, e: 10.0 },
  Borno: { s: 9.9, w: 11.0, n: 13.9, e: 15.1 },
  'Cross River': { s: 4.3, w: 7.8, n: 6.9, e: 9.6 },
  Delta: { s: 4.9, w: 5.3, n: 6.5, e: 7.0 },
  Ebonyi: { s: 5.7, w: 7.7, n: 6.7, e: 8.6 },
  Edo: { s: 5.7, w: 5.0, n: 7.2, e: 6.9 },
  Ekiti: { s: 7.4, w: 4.8, n: 8.2, e: 5.9 },
  Enugu: { s: 5.9, w: 6.9, n: 7.2, e: 8.1 },
  FCT: { s: 8.3, w: 6.8, n: 9.4, e: 7.8 },
  Gombe: { s: 9.5, w: 10.0, n: 11.2, e: 12.1 },
  Imo: { s: 4.9, w: 6.6, n: 5.9, e: 7.5 },
  Jigawa: { s: 11.0, w: 8.2, n: 13.0, e: 10.7 },
  Kaduna: { s: 9.0, w: 6.8, n: 11.4, e: 9.5 },
  Kano: { s: 11.0, w: 8.0, n: 13.0, e: 9.5 },
  Katsina: { s: 11.5, w: 6.5, n: 13.9, e: 9.3 },
  Kebbi: { s: 10.1, w: 3.8, n: 13.2, e: 6.3 },
  Kogi: { s: 6.7, w: 5.7, n: 8.9, e: 7.9 },
  Kwara: { s: 7.7, w: 2.5, n: 9.8, e: 6.6 },
  Lagos: { s: 6.3, w: 2.7, n: 6.8, e: 3.9 },
  Nasarawa: { s: 7.8, w: 7.5, n: 9.4, e: 9.3 },
  Niger: { s: 8.2, w: 3.8, n: 11.7, e: 7.3 },
  Ogun: { s: 6.3, w: 2.7, n: 7.8, e: 4.3 },
  Ondo: { s: 5.7, w: 4.5, n: 7.7, e: 6.4 },
  Osun: { s: 7.0, w: 4.0, n: 8.2, e: 5.1 },
  Oyo: { s: 6.8, w: 2.7, n: 9.1, e: 4.6 },
  Plateau: { s: 8.2, w: 8.2, n: 10.4, e: 10.5 },
  Rivers: { s: 4.2, w: 6.4, n: 5.5, e: 7.5 },
  Sokoto: { s: 11.5, w: 4.2, n: 13.9, e: 6.8 },
  Taraba: { s: 6.5, w: 9.8, n: 9.4, e: 12.5 },
  Yobe: { s: 10.5, w: 10.8, n: 13.8, e: 13.5 },
  Zamfara: { s: 11.1, w: 5.8, n: 13.9, e: 8.0 },
  'All Nigeria': null,
};
const MAP_MODES = {
  polygon: {
    label: 'Data',
    tile: null,
    bg: '#090b10',
    fillOpacity: 0.95,
    strokeColor: '#1a1a2e',
    strokeWeight: 0.7,
    strokeOpacity: 0.95,
  },
  dark: {
    label: 'Map',
    tile: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    tileOpts: { attribution: BASE_TILE_ATTRIBUTION, subdomains: 'abcd', maxZoom: 19 },
    bg: '#090b10',
    fillOpacity: 0.82,
    strokeColor: '#0f172a',
    strokeWeight: 0.25,
    strokeOpacity: 0.75,
  },
  print: {
    label: 'Print',
    tile: 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
    tileOpts: { attribution: BASE_TILE_ATTRIBUTION, subdomains: 'abcd', maxZoom: 19 },
    bg: '#f3efe7',
    fillOpacity: 0.88,
    colorOverride: true,
    strokeColor: '#f1d2b4',
    strokeWeight: 0.35,
    strokeOpacity: 0.9,
  },
};

let stateSyncTimer = null;
let pendingStateUrl = '';
let stateSyncLocked = false;

function hideBootOverlay() {
  const dismissFromDocument = (doc) => {
    if (!doc) return false;
    const overlay = doc.getElementById('hd-boot-overlay');
    if (!overlay) return false;
    overlay.classList.add('is-hidden');
    window.setTimeout(() => {
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }, 320);
    return true;
  };

  let dismissed = false;
  try {
    const parentDoc = getParentDocument();
    if (parentDoc) {
      dismissed = dismissFromDocument(parentDoc);
    }
  } catch (err) {
    // Cross-origin iframe restrictions: ignore and try local document.
  }
  if (!dismissed) dismissFromDocument(document);
}

function scheduleBootOverlayHide(delayMs = 0) {
  window.setTimeout(() => {
    window.requestAnimationFrame(() => hideBootOverlay());
  }, Math.max(0, Number(delayMs) || 0));
}

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

function scoreOutOfTen(source) {
  const riskScoreTotal = safeNum(source?.risk_score_total);
  if (riskScoreTotal != null) return riskScoreTotal;

  const riskTotal = safeNum(source?.risk_total);
  if (riskTotal != null) return riskTotal;

  const riskScore = safeNum(source?.risk_score);
  if (riskScore != null) return riskScore <= 1 ? riskScore * 10 : riskScore;

  const risk = safeNum(source?.risk);
  if (risk != null) return risk <= 1 ? risk * 10 : risk;

  return null;
}

function riskLabel(r, total) {
  const t = safeNum(total);
  if (t != null) return t.toFixed(2);
  return r == null || Number.isNaN(Number(r)) ? 'NA' : (Number(r) * 10).toFixed(2);
}

function confidenceBadge(conf) {
  const n = safeNum(conf);
  if (n == null) return {
    emoji: '🟡', label: 'Unknown',
    title: 'Confidence could not be estimated for this LGA.'
  };
  if (n >= 85) return {
    emoji: '🟢', label: '>85%',
    title: 'Relatively complete data. Some gaps remain — always validate locally.'
  };
  if (n >= 65) return {
    emoji: '🟡', label: '65–85%',
    title: 'Moderate data gaps may affect this estimate. Use with field verification.'
  };
  return {
    emoji: '🔴', label: '<65%',
    title: 'Significant data gaps — treat this score with extra caution.'
  };
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

function interpolateColor(hex1, hex2, t) {
  const r1 = parseInt(hex1.slice(1, 3), 16);
  const g1 = parseInt(hex1.slice(3, 5), 16);
  const b1 = parseInt(hex1.slice(5, 7), 16);
  const r2 = parseInt(hex2.slice(1, 3), 16);
  const g2 = parseInt(hex2.slice(3, 5), 16);
  const b2 = parseInt(hex2.slice(5, 7), 16);
  return `rgb(${Math.round(r1 + (r2 - r1) * t)},${Math.round(g1 + (g2 - g1) * t)},${Math.round(b1 + (b2 - b1) * t)})`;
}

function getColor(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return 'rgba(80,80,90,0.4)';
  const MEDIAN = 5.5;
  const s = Math.max(0, Math.min(10, Number(score)));
  if (s <= MEDIAN) {
    const t = 1 - (s / MEDIAN);
    return interpolateColor('#aec4d6', '#2166ac', t);
  }
  const t = (s - MEDIAN) / (10 - MEDIAN);
  return interpolateColor('#e8a49a', '#d73027', t);
}

function riskColorHex(r) {
  if (r == null || Number.isNaN(Number(r))) return 'rgba(80,80,90,0.4)';
  const v = Math.max(0, Math.min(1, Number(r)));
  if (v <= 0.5) return interpolateColor('#2166ac', '#aec4d6', v / 0.5);
  return interpolateColor('#e8a49a', '#d73027', (v - 0.5) / 0.5);
}

function getPrintColor(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return '#e5e5e5';
  const s = Math.max(0, Math.min(10, Number(score)));
  return interpolateColor('#ffe0b2', '#bf360c', s / 10);
}

function getCoverageColor(pct) {
  if (pct === null || pct === undefined || Number.isNaN(Number(pct))) return 'rgba(80,80,90,0.5)';
  const t = Math.max(0, Math.min(100, Number(pct)));
  if (t < 40) {
    const s = 1 - (t / 40);
    return interpolateColor('#ff6b35', '#d73027', s);
  }
  if (t < 70) {
    const s = (t - 40) / 30;
    return interpolateColor('#ff6b35', '#f7c59f', s);
  }
  const s = (t - 70) / 30;
  return interpolateColor('#f7c59f', '#3c4150', s);
}

function legendHtml(title, gradient, labels, rows) {
  const labelHtml = Array.isArray(labels) && labels.length
    ? `
      <div class="legend-labels">
        ${labels.map((label) => `<span>${label}</span>`).join('')}
      </div>
    `
    : '';
  const rowHtml = Array.isArray(rows)
    ? rows.map((row) => `
        <div class="legend-row">
          <span class="legend-dot" style="background:${row.color}"></span>
          <span>${row.text}</span>
        </div>
      `).join('')
    : '';

  return `
    <div style="font-family:'IBM Plex Mono',monospace">
      <div class="legend-title">${escapeHtml(title)}</div>
      <div class="legend-gradient" style="background:${gradient}"></div>
      ${labelHtml}
      ${rowHtml}
    </div>
  `;
}

function updateLegend() {
  const legend = document.querySelector('.map-legend');
  if (!legend) return;

  if (currentMapMode === 'print') {
    legend.innerHTML = legendHtml(
      'PRINT SCALE',
      'linear-gradient(to right, #ffe0b2, #bf360c)',
      ['0', '5', '10'],
      [
        { color: '#ffe0b2', text: 'Lower score / lighter fill' },
        { color: '#bf360c', text: 'Higher score / darker fill' },
      ],
    );
    return;
  }

  if (currentLayer === 'Risk score' && currentFocus === '60-min coverage') {
    legend.innerHTML = legendHtml(
      '60-MIN DRIVE',
      'linear-gradient(to right, #d73027, #ff6b35, #f7c59f, #3c4150)',
      ['< 40%', '70%+'],
      [
        { color: '#d73027', text: 'Low road access (< 40% pop within 60-min drive)' },
        { color: '#ff6b35', text: 'Moderate access' },
        { color: '#3c4150', text: 'Well served (> 70%)' },
      ],
    );
    return;
  }

  if (currentLayer === 'SHAP') {
    legend.innerHTML = legendHtml(
      'SHAP',
      'linear-gradient(to right, #2166ac, #f7f7f7, #d73027)',
      ['-', '0', '+'],
      [
        { color: '#2166ac', text: 'Decreases risk' },
        { color: '#d73027', text: 'Increases risk' },
      ],
    );
    return;
  }

  const title = currentLayer === 'Risk score' ? currentFocus : currentLayer;
  legend.innerHTML = legendHtml(
    String(title || 'Risk score').toUpperCase(),
    'linear-gradient(to right, #2166ac, #aec4d6, #e8a49a, #d73027)',
    ['0', '5.5', '10'],
    [
      { color: '#2166ac', text: 'Lower barrier / lower risk' },
      { color: '#aec4d6', text: 'Near median' },
      { color: '#d73027', text: 'Higher barrier / higher risk' },
    ],
  );
}

const CHIP_COLOR_FIELD = {
  'All risk': {
    field: 'risk_total',
    lowerIsWorse: false,
    getValue: (source) => scoreOutOfTen(source),
  },
  'Child mortality': {
    field: 'u5mr',
    lowerIsWorse: false,
    getValue: (source) => safeNum(source?.u5mr ?? source?.u5_mortality_rate ?? source?.u5mr_mean),
  },
  'Facility access': {
    field: 'fac',
    lowerIsWorse: true,
    getValue: (source) => safeNum(source?.fac ?? source?.facilities_per_10k),
  },
  Connectivity: {
    field: 'towers',
    lowerIsWorse: true,
    getValue: (source) => safeNum(source?.towers ?? source?.towers_per_10k ?? source?.connectivity_score),
  },
  '5km coverage': {
    field: 'cov',
    lowerIsWorse: true,
    getValue: (source) => safeNum(source?.cov ?? source?.coverage_5km),
  },
  '60-min coverage': {
    field: 'pop_pct_60min',
    lowerIsWorse: true,
    getValue: (source) => safeNum(source?.pop_pct_60min),
  },
};

function getMapModeConfig(mode = currentMapMode) {
  return MAP_MODES[mode] || MAP_MODES.polygon;
}

function getFeatureStateName(source) {
  return sanitizeText(source?.state_name || source?.state || '', '');
}

function isFeatureInCurrentState(source) {
  return currentState === 'All Nigeria' || getFeatureStateName(source) === currentState;
}

function fillOpacityForFeature(source) {
  return isFeatureInCurrentState(source) ? getMapModeConfig().fillOpacity : 0.15;
}

function focusColorCacheKey(focus = currentFocus) {
  return `focus-color:${focus}`;
}

function getFocusColorScore(source, focus = currentFocus) {
  const cfg = CHIP_COLOR_FIELD[focus] || CHIP_COLOR_FIELD['All risk'];
  if (cfg.field === 'risk_total') {
    return scoreOutOfTen(source);
  }

  const cacheKey = focusColorCacheKey(focus);
  if (!fieldValuesCache.has(cacheKey)) {
    const values = lgas
      .map((lga) => cfg.getValue(lga))
      .filter((value) => value != null && !Number.isNaN(value));
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 0;
    const range = max - min || 1;
    const scoreMap = new Map();
    lgas.forEach((lga) => {
      const raw = cfg.getValue(lga);
      if (raw == null || Number.isNaN(raw)) return;
      const norm = ((Number(raw) - min) / range) * 10;
      const score = cfg.lowerIsWorse ? 10 - norm : norm;
      scoreMap.set(String(lga.id), score);
    });
    fieldValuesCache.set(cacheKey, scoreMap);
  }

  const scoreMap = fieldValuesCache.get(cacheKey);
  const sourceId = source?.id ?? source?.lga_id ?? source?.lga_uid;
  if (sourceId != null && scoreMap?.has(String(sourceId))) {
    return scoreMap.get(String(sourceId));
  }

  const raw = cfg.getValue(source);
  if (raw == null || Number.isNaN(raw)) return null;
  const values = Array.from(scoreMap?.values?.() || []);
  if (!values.length) return null;
  const sourceNorm = Math.max(0, Math.min(10, Number(raw)));
  return cfg.lowerIsWorse ? 10 - sourceNorm : sourceNorm;
}

function scoreForCurrentLayer(source, layer = currentLayer) {
  if (!source) return null;
  if (layer === 'Risk score') return getFocusColorScore(source, currentFocus);
  if (layer === 'Facilities' || layer === 'Connectivity' || layer === 'Towers') {
    return badnessForLayer(source, layer);
  }
  const norm = scaledLayerValue(source.id ?? source.lga_id ?? source.lga_uid);
  return norm == null ? null : norm * 10;
}

function fillColorForFeature(source, layer = currentLayer) {
  const score = scoreForCurrentLayer(source, layer);
  if (getMapModeConfig().colorOverride) {
    return getPrintColor(score);
  }
  if (layer === 'SHAP') {
    const norm = scaledLayerValue(source.id ?? source.lga_id ?? source.lga_uid);
    return riskColorHex(norm);
  }
  if (layer === 'Risk score' && currentFocus === '60-min coverage') {
    return getCoverageColor(safeNum(source?.pop_pct_60min));
  }
  return getColor(score);
}

function applyStateDimming() {
  if (!mapInstance) return;
  mapInstance.eachLayer((layer) => {
    if (!layer.feature || typeof layer.setStyle !== 'function') return;
    layer.setStyle({ fillOpacity: fillOpacityForFeature(layer.feature.properties) });
  });
}

function fitMapToCurrentState() {
  if (!mapInstance) return;
  const bb = STATE_BOUNDS[currentState];
  if (bb) {
    mapInstance.fitBounds([[bb.s, bb.w], [bb.n, bb.e]], { padding: [40, 40], maxZoom: 9 });
    return;
  }
  mapInstance.fitBounds([[4.1, 2.5], [13.9, 15.1]], { padding: [20, 20] });
}

function applyMapModeBaseLayer() {
  const mapEl = document.getElementById('map-leaflet');
  if (mapEl) {
    mapEl.style.background = getMapModeConfig().bg || '#090b10';
  }

  if (!mapInstance) return;

  if (baseTileLayer) {
    mapInstance.removeLayer(baseTileLayer);
    baseTileLayer = null;
  }

  const cfg = getMapModeConfig();
  if (cfg.tile) {
    baseTileLayer = L.tileLayer(cfg.tile, cfg.tileOpts).addTo(mapInstance);
    baseTileLayer.on('tileerror', (e) => {
      console.warn('[HDS] tile error:', e?.tile?.src);
    });
    if (typeof baseTileLayer.bringToBack === 'function') {
      baseTileLayer.bringToBack();
    }
  }

  if (geoLayer && typeof geoLayer.bringToFront === 'function') {
    geoLayer.bringToFront();
  }
}

function switchMapMode(mode) {
  if (!MAP_MODES[mode]) return;
  currentMapMode = mode;
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  applyMapModeBaseLayer();
  updateLegend();
  renderMap();
}

function handleStateChange(selectedState) {
  currentState = selectedState;
  syncHeader();
  renderHotspots();
  renderMap();
  fitMapToCurrentState();
  pushStateToPython();
  queueEvent('filter_change', { state: currentState });
}

function markTourComplete() {
  localStorage.setItem(TOUR_STORAGE_KEY, '1');
}

function isTourOpen() {
  const overlay = document.getElementById('tour-overlay');
  return Boolean(overlay?.classList.contains('open'));
}

function setApplyStatus(label, mode = 'applied') {
  const statusText = document.getElementById('apply-status-text');
  if (statusText) {
    statusText.textContent = mode === 'updating' ? 'updating' : 'ready';
  }

  const dot = document.querySelector('.header-status .status-dot');
  if (dot) {
    dot.classList.toggle('is-updating', mode === 'updating');
  }
}

function syncHeader() {
  const headerCount = meta.lga_count || lgas.length || '-';
  const headerYear = currentYear || '-';
  const stateDisplay = document.getElementById('state-display');
  if (stateDisplay) stateDisplay.textContent = currentState;

  const yearDisplay = document.getElementById('year-display');
  if (yearDisplay) yearDisplay.textContent = headerYear;

  const lgaCount = document.getElementById('lga-count');
  if (lgaCount) lgaCount.textContent = String(headerCount);

  const metaEl = document.getElementById('dataset-meta-text');
  if (metaEl) {
    const count = meta.lga_count || lgas.length || '—';
    const yearText = currentYear || '—';
    metaEl.textContent = `${count} LGAs · ${yearText}`;
  }

  const focusScope = document.getElementById('focus-scope');
  if (focusScope) focusScope.textContent = currentState;

  const stateSelect = document.getElementById('state-select');
  if (stateSelect && stateSelect.value !== currentState) {
    stateSelect.value = currentState;
  }
  const stateSelectMobile = document.getElementById('state-select-mobile');
  if (stateSelectMobile && stateSelectMobile.value !== currentState) {
    stateSelectMobile.value = currentState;
  }
  const yearSelect = document.getElementById('year-select');
  if (yearSelect && yearSelect.value !== currentYear) {
    yearSelect.value = currentYear;
  }
  const yearSelectMobile = document.getElementById('year-select-mobile');
  if (yearSelectMobile && yearSelectMobile.value !== currentYear) {
    yearSelectMobile.value = currentYear;
  }

  document.querySelectorAll('.depth-btn').forEach((btn) => {
    const depthVal = Number(btn.dataset.depth || 0);
    const active = depthVal === currentDepth;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });

  document.querySelectorAll('.focus-section .chip').forEach((chip) => {
    const active = chip.dataset.focus === currentFocus;
    chip.classList.toggle('active', active);
    chip.classList.toggle('chip-sel', active);
    chip.setAttribute('aria-pressed', String(active));
  });

  const footnote = document.getElementById('data-footnote');
  if (footnote) {
    const modelVersion = Array.isArray(meta.model_version) ? meta.model_version.join(', ') : meta.model_version;
    const updated = meta.data_last_updated ? `Updated ${meta.data_last_updated}` : 'Update date unknown';
    footnote.textContent = `DHS 2013/2018/2024 · NHFR · ORS isochrones · OpenCellID · Model ${modelVersion || 'v1.4'} · ${updated}`;
  }

  const resNote = document.querySelector('.res-note');
  if (resNote) {
    if (currentDepth === 1) {
      resNote.textContent = 'Research mode - SHAP attribution - Select an LGA to analyze';
      resNote.style.borderBottom = '1px solid rgba(249,115,22,0.2)';
      resNote.style.color = 'rgba(249,115,22,0.7)';
    } else {
      resNote.textContent = 'Tap any LGA on the map to see details';
      resNote.style.borderBottom = '';
      resNote.style.color = '';
    }
  }

  setApplyStatus('Applied', 'applied');
  syncMobileMoreMeta();
}

function applyDepthVisibility() {
  document.querySelectorAll('.depth-gate').forEach((el) => {
    const depthClass = [...el.classList].find((c) => c.startsWith('depth-')) || 'depth-99';
    const requiredDepth = Number(depthClass.replace('depth-', ''));
    const visible = currentDepth >= requiredDepth;
    el.hidden = !visible;
  });

  const strip = document.getElementById('compare-strip');
  if (strip) strip.classList.add('visible');
}

function buildStateUrl() {
  const url = getAppStateBaseUrl();
  if (!url) return null;

  const params = new URLSearchParams(url.search);
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

  return `${url.pathname}?${params.toString()}`;
}

function buildStateMessagePayload() {
  return {
    state: currentState,
    focus: currentFocus,
    depth: currentDepth,
    year: currentYear,
    mobile: isMobile ? '1' : '0',
    lga_uid: selectedLGA?.id ? String(selectedLGA.id) : null,
    compare: compareLGAs.map((item) => String(item.id)),
    layer: currentLayer,
    map_mode: currentMapMode,
    url: buildStateUrl(),
  };
}

function scheduleNonCriticalWork(task, timeout = 600) {
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(() => task(), { timeout });
    return;
  }
  window.setTimeout(task, 0);
}

const deferredScriptLoads = new Map();

function ensureExternalScript(src, globalName) {
  if (globalName && typeof window[globalName] !== 'undefined') {
    return Promise.resolve();
  }
  if (deferredScriptLoads.has(src)) {
    return deferredScriptLoads.get(src);
  }

  const promise = new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.body.appendChild(script);
  });

  deferredScriptLoads.set(src, promise);
  return promise;
}

function postStateMessage(payload) {
  try {
    window.parent.postMessage({ type: APP_STATE_MESSAGE_TYPE, state: payload }, '*');
    return true;
  } catch (e) {
    return false;
  }
}

function normalizeStandaloneState(rawState = {}) {
  const compareIds = normalizeCompareIds(rawState.compare);
  const normalizedDepth = Number(rawState.depth ?? currentDepth ?? 0);
  const lgaId = rawState.lga_uid ?? rawState.lga_id ?? rawState.lga ?? null;
  return {
    state: rawState.state || currentState,
    focus: rawState.focus || currentFocus,
    depth: Number.isNaN(normalizedDepth) ? currentDepth : Math.max(0, Math.min(1, normalizedDepth)),
    year: rawState.year != null ? String(rawState.year) : currentYear,
    mobile: rawState.mobile != null ? String(rawState.mobile) : (isMobile ? '1' : '0'),
    lga_uid: lgaId != null && String(lgaId).trim() !== '' ? String(lgaId) : '',
    compare: compareIds,
    layer: rawState.layer || currentLayer,
    map_mode: rawState.map_mode || currentMapMode,
    url: rawState.url || '',
  };
}

async function refreshStandalonePayload(rawState = {}) {
  if (!standaloneMode) return;

  const nextState = normalizeStandaloneState(rawState);
  const refreshToken = ++standaloneRefreshToken;
  currentLayer = nextState.layer || currentLayer;
  currentMapMode = nextState.map_mode || currentMapMode;

  const url = nextState.url
    ? new URL(nextState.url, window.location.origin)
    : new URL(window.location.href);
  url.searchParams.set('state', nextState.state);
  url.searchParams.set('focus', nextState.focus);
  url.searchParams.set('depth', String(nextState.depth));
  url.searchParams.set('year', nextState.year);
  url.searchParams.set('mobile', nextState.mobile);
  if (nextState.lga_uid) url.searchParams.set('lga', nextState.lga_uid);
  else url.searchParams.delete('lga');
  if (nextState.compare.length) url.searchParams.set('compare', nextState.compare.join(','));
  else url.searchParams.delete('compare');
  window.history.replaceState({}, '', `${url.pathname}?${url.searchParams.toString()}`);

  const query = new URLSearchParams({
    state: nextState.state,
    focus: nextState.focus,
    depth: String(nextState.depth),
    year: nextState.year,
    mobile: nextState.mobile,
  });
  if (nextState.lga_uid) query.set('lga', nextState.lga_uid);
  if (nextState.compare.length) query.set('compare', nextState.compare.join(','));

  if (standaloneFetchController) {
    standaloneFetchController.abort();
  }
  standaloneFetchController = typeof AbortController !== 'undefined' ? new AbortController() : null;

  setApplyStatus('Updating', 'updating');

  try {
    const response = await fetch(
      `/api/data?${query.toString()}`,
      standaloneFetchController ? { signal: standaloneFetchController.signal } : undefined,
    );
    if (!response.ok) {
      throw new Error(`Standalone refresh failed: ${response.status}`);
    }
    const data = await response.json();
    if (refreshToken !== standaloneRefreshToken) return;

    resetMapDataState();
    window.__INITIAL_DATA__ = data;
    applyPayloadData(data, { preserveSelection: false, preserveCompare: false });
    bootApp({ force: true, skipDeferredWork: true });
  } catch (error) {
    if (error?.name === 'AbortError') return;
    console.error('[HDS] standalone refresh error:', error);
    setApplyStatus('Applied', 'applied');
  }
}

function attachStandaloneMessageHandler() {
  if (!standaloneMode || standaloneMessageHandlerAttached) return;
  standaloneMessageHandlerAttached = true;
  window.addEventListener('message', (event) => {
    const message = event.data;
    if (!message || message.type !== APP_STATE_MESSAGE_TYPE) return;
    if (event.source && event.source !== window) return;
    refreshStandalonePayload(message.state || {});
  });
}

function flushStateToPython() {
  if (stateSyncLocked || !pendingStateUrl) return;

  const url = getAppStateBaseUrl() || getParentUrl();
  if (!url) {
    pendingStateUrl = '';
    return;
  }

  const currentUrl = `${url.pathname}${url.search}`;
  if (pendingStateUrl === currentUrl) {
    setApplyStatus('Applied', 'applied');
    pendingStateUrl = '';
    return;
  }

  const statePayload = buildStateMessagePayload();
  if (standaloneMode) {
    pendingEvent = null;
    pendingStateUrl = '';
    postStateMessage(statePayload);
    return;
  }

  stateSyncLocked = true;
  pendingEvent = null;
  postStateMessage(statePayload);
  if (!replaceParentLocation(pendingStateUrl)) {
    stateSyncLocked = false;
    pendingStateUrl = '';
  }
}

function pushStateToPython({ immediate = false } = {}) {
  pendingStateUrl = buildStateUrl();
  if (!pendingStateUrl) return;

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

// Map focus labels to data columns
const FOCUS_COLUMNS = {
  'All risk': 'risk_total',
  'Child mortality': 'u5mr',
  'Facility access': 'fac',
  'Connectivity': 'towers',
  '5km coverage': 'cov',
  '60-min coverage': 'pop_pct_60min'
};

// Columns where higher values are better (ascending sort)
const ASCENDING_FOCUS = {
  'fac': true,
  'towers': true,
  'cov': true,
  'pop_pct_60min': true
};

function hotspotsBase() {
  // Use all_lgas_for_search if available, otherwise fall back to lgas
  // This ensures comprehensive search across all LGAs in the selected state
  const searchSource = allLgasForSearch.length > 0 ? allLgasForSearch : lgas;
  
  // Always use the full lgas array and sort by current focus
  const focusColumn = FOCUS_COLUMNS[currentFocus] || 'risk_total';
  const ascending = ASCENDING_FOCUS[focusColumn] || false;
  
  // Filter by current state first
  let filtered = searchSource;
  if (currentState && currentState !== 'All Nigeria') {
    filtered = searchSource.filter(l => String(l.state) === String(currentState));
  }
  
  return [...filtered]
    .sort((a, b) => {
      let aScore, bScore;
      
      if (focusColumn === 'risk_total') {
        aScore = safeNum(a.risk_total) != null ? safeNum(a.risk_total) : Number(a.risk ?? 0) * 10;
        bScore = safeNum(b.risk_total) != null ? safeNum(b.risk_total) : Number(b.risk ?? 0) * 10;
      } else {
        aScore = safeNum(a[focusColumn]) ?? 0;
        bScore = safeNum(b[focusColumn]) ?? 0;
      }
      
      if (ascending) {
        return aScore - bScore; // Lower is better for these metrics
      }
      return bScore - aScore; // Higher is better (or higher risk)
    })
    .map((l, i) => ({ ...l, rank: i + 1 }));
}

// Configurable limit for hotspots display - can be overridden via URL param
let hotspotDisplayLimit = 12;
if (urlParams.get('hotspot_limit')) {
  const parsed = parseInt(urlParams.get('hotspot_limit'), 10);
  if (!isNaN(parsed) && parsed > 0) {
    hotspotDisplayLimit = parsed;
  }
}

function renderHotspotCard(item, index) {
  const lgaId = sanitizeText(item.lga_id || item.id || '');
  const score = scoreOutOfTen(item) || 0;
  const scoreColor = score > 5.5 ? SCORE_COLOR_HIGH : SCORE_COLOR_LOW;
  const barColor = scoreColor;
  const barWidth = Math.min(100, Math.max(0, score * 10)).toFixed(1);
  const driver = sanitizeText(item.worst_driver, '');
  const driverHtml = driver
    ? `<span class="lga-driver">${escapeHtml(driver)}</span>`
    : '';
  const selectedClass = String(selectedLGA?.id) === String(lgaId) ? ' active' : '';
  const name = sanitizeText(item.lga_name || item.name || '', '');
  const state = sanitizeText(item.state_name || item.state || '', '');

  return `
    <div class="hotspot-item${selectedClass}" role="button" tabindex="0" data-lga-id="${escapeHtml(lgaId)}" onclick="selectLGA('${escapeHtml(lgaId)}')">
      <span class="hotspot-rank">${String(index + 1).padStart(2, '0')}</span>
      <div class="hotspot-info">
        <div class="hotspot-name">${escapeHtml(name)}</div>
        <div class="hotspot-meta">
          <span class="hotspot-state">${escapeHtml(state)}</span>
          ${driverHtml}
        </div>
      </div>
      <div class="hotspot-score-col">
        <span class="hotspot-score" style="color:${scoreColor}">${score.toFixed(2)}</span>
        <div class="hotspot-bar-wrap">
          <div class="hotspot-bar" style="width:${barWidth}%;background:${barColor}"></div>
        </div>
      </div>
    </div>
  `;
}

function renderHotspots() {
  const list = document.getElementById('hotspot-list');
  if (!list) return;

  const query = (document.getElementById('search-input')?.value || '').toLowerCase();
  const base = hotspotsBase();
  const filtered = query
    ? base.filter((l) => {
      const name = sanitizeText(l.lga_name || l.name);
      const state = sanitizeText(l.state_name || l.state);
      return `${name} ${state}`.toLowerCase().includes(query);
    })
    : base;
  window.__hdHotspots = filtered.length ? filtered : base;

  list.replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement('div');
    empty.className = 'hotspot-empty';
    empty.textContent = query
      ? `No LGAs match "${query}".`
      : `No LGAs available for ${currentState} in ${currentYear}.`;
    list.appendChild(empty);
    return;
  }

  list.innerHTML = filtered
    .slice(0, hotspotDisplayLimit)
    .map((lga, i) => renderHotspotCard(lga, i))
    .join('');

  list.querySelectorAll('.hotspot-item').forEach((item) => {
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectLGA(item.dataset.lgaId || '');
      }
    });
  });
}

function openDrawer() {
  const drawer = document.getElementById('detail-drawer');
  if (drawer) drawer.classList.add('open');

  if (isMobile && !document.getElementById('drawer-backdrop')) {
    const bd = document.createElement('div');
    bd.id = 'drawer-backdrop';
    bd.style.cssText = 'position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,0.4);';
    bd.addEventListener('click', closeDrawer);
    (document.querySelector('.app-shell') || document.body).appendChild(bd);
  }
}

function closeDrawer() {
  const drawer = document.getElementById('detail-drawer');
  if (drawer) drawer.classList.remove('open');
  document.getElementById('drawer-backdrop')?.remove();
  selectedLGA = null;
  renderHotspots();
  renderMap();
  pushStateToPython();
}

function initBottomSheetDrag() {
  const drawer = document.getElementById('detail-drawer');
  if (!drawer) return;

  if (!drawer.querySelector('.drag-handle')) {
    const handle = document.createElement('div');
    handle.className = 'drag-handle';
    drawer.insertBefore(handle, drawer.firstChild);
  }

  let startY = 0;
  let isDragging = false;

  drawer.addEventListener('touchstart', (e) => {
    startY = e.touches[0].clientY;
    isDragging = true;
  }, { passive: true });

  drawer.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    const delta = e.touches[0].clientY - startY;
    if (delta > 0) {
      drawer.style.transform = `translateY(${delta}px)`;
    }
  }, { passive: true });

  drawer.addEventListener('touchend', (e) => {
    if (!isDragging) return;
    isDragging = false;
    const delta = e.changedTouches[0].clientY - startY;
    drawer.style.transform = '';
    if (delta > 80) closeDrawer();
  }, { passive: true });
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
  if (!inner || !selectedLGA) {
    if (inner) inner.replaceChildren();
    return;
  }

  let lga = selectedLGA;
  if (selectedLGA?.id) {
    const base = lgaById.get(String(selectedLGA.id));
    if (base) {
      lga = mergeLga(base, selectedLGA);
      selectedLGA = lga;
    }
  }
  const isResearch = currentDepth >= 1;

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
    ? Object.entries(lga.shap)
      .filter(([k, v]) => k !== 'is_synthetic' && v != null && !Number.isNaN(Number(v)))
      .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))
    : [];
  const maxShap = shapRows.length
    ? Math.max(...shapRows.map(([, v]) => Math.abs(Number(v))))
    : 1;

  const heroScore = scoreOutOfTen(lga);
  const heroScoreColor = heroScore != null && heroScore > 5.5 ? SCORE_COLOR_HIGH : SCORE_COLOR_LOW;
  const heroScoreDisplay = heroScore != null ? heroScore.toFixed(2) : 'NA';
  const stateLabel = sanitizeText(lga.state_name || lga.state, 'Unknown state');
  const conf = confidenceBadge(lga.confidence_pct);

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

  const isInCompare = compareLGAs.some((item) => String(item.id) === String(lga.id));
  const compareBtnHtml = `
    <button type="button"
            class="compare-inline-btn ${isInCompare ? 'compare-btn-added' : ''}"
            id="add-to-compare-btn">
      ${isInCompare ? 'Added to compare' : 'Add to compare'}
    </button>
  `;

  const shapChartHtml = shapRows.map(([k, v]) => {
    const n = Number(v);
    const pct = Math.min((Math.abs(n) / maxShap) * 100, 100);
    const isPos = n >= 0;
    const label = k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    const absVal = Math.abs(n);
    const magnitude = absVal > 0.1 ? 'high' : absVal > 0.04 ? 'medium' : 'low';

    return `
      <div class="shap-row-full" data-magnitude="${magnitude}">
        <div class="shap-feature-full" title="${escapeHtml(k)}">${escapeHtml(label)}</div>
        <div class="shap-bar-wrap">
          <div class="shap-bar-fill ${isPos ? 'pos' : 'neg'}"
               style="width:${pct}%"
               title="${isPos ? 'Increases' : 'Decreases'} risk"></div>
        </div>
        <div class="shap-val-full ${isPos ? 'shap-pos-text' : 'shap-neg-text'}">
          ${isPos ? '+' : '-'} ${absVal.toFixed(3)}
        </div>
      </div>
    `;
  }).join('');

  const shapBodyHtml = shapRows.length
    ? `
      <div class="shap-chart">${shapChartHtml}</div>
      <div class="shap-legend">
        <span class="shap-legend-item pos">+ Increases risk</span>
        <span class="shap-legend-item neg">- Decreases risk</span>
      </div>
    `
    : `
      <div class="research-empty">
        ${lga.shap == null
          ? (String(currentYear).toLowerCase() === 'both'
            ? 'Feature attribution is available for single-year views. Select 2013 or 2018.'
            : 'Feature attribution is still loading for this LGA. If it persists, run scripts/generate_shap.py.')
          : 'No SHAP data available for this LGA/year combination.'}
      </div>
    `;

  const shapHtml = isResearch
    ? `
      <div class="research-section">
        <div class="research-section-title">
          Feature Attribution
          <span class="research-badge">SHAP</span>
        </div>
        <div class="research-explainer">
          Each bar shows how much a feature pushes the risk score up (red) or down (green)
          relative to the national average.
        </div>
        ${shapBodyHtml}
      </div>
    `
    : '';

  const confReasons = String(lga.confidence_reason_codes || '')
    .split('|')
    .map((r) => r.trim())
    .filter((r) => r && r.toLowerCase() !== 'none');
  const confReasonHtml = confReasons.length
    ? confReasons.map((r) => `<div class="conf-reason-item">${escapeHtml(r)}</div>`).join('')
    : '<div class="conf-reason-item muted">No reason codes available</div>';

  const confDetailHtml = isResearch
    ? `
      <div class="research-section">
        <div class="research-section-title">
          Model Confidence
          <span class="research-badge">${conf.emoji} ${conf.label}</span>
        </div>
        <div class="conf-reason-list">${confReasonHtml}</div>
      </div>
    `
    : '';

  const modelVersion = Array.isArray(meta.model_version)
    ? meta.model_version.join(', ')
    : (meta.model_version || 'v1.4');
  const metaHtml = isResearch
    ? `
      <div class="research-section research-meta">
        <div class="research-meta-row">
          <span class="research-meta-label">DATA SOURCES</span>
          <span>DHS ${escapeHtml(String(currentYear))} - NHFR 2020 - OpenCellID - WorldPop</span>
        </div>
        <div class="research-meta-row">
          <span class="research-meta-label">MODEL</span>
          <span>${escapeHtml(String(modelVersion))}</span>
        </div>
        <div class="research-meta-row">
          <span class="research-meta-label">LAST UPDATED</span>
          <span>${escapeHtml(String(meta.data_last_updated || 'Unknown'))}</span>
        </div>
        <div class="research-meta-row">
          <span class="research-meta-label">BOUNDARY RES</span>
          <span>${escapeHtml(String(meta.boundary_resolution || 'auto'))}</span>
        </div>
      </div>
    `
    : '';

  const downloadHtml = isResearch
    ? '<button type="button" class="dl-btn" id="download-lga-btn">Download LGA data (CSV)</button>'
    : '';

  inner.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-lga">${escapeHtml(sanitizeText(lga.name, 'Unknown LGA'))}</div>
        <div class="detail-state-tag">
          ${escapeHtml(stateLabel)}
          ${isResearch ? '<span class="research-mode-tag">RESEARCH</span>' : ''}
        </div>
        <div class="detail-state-tag" title="${escapeHtml(conf.title || '')}">Data confidence: ${conf.emoji} ${escapeHtml(conf.label)} <span class="conf-band-note">(estimated band)</span></div>
      </div>
      <button type="button" class="close-btn" id="detail-close-btn" aria-label="Close">&times;</button>
    </div>

    <div class="detail-score-hero">
      <div class="detail-score-val" style="color:${heroScoreColor}">
        ${heroScoreDisplay}
      </div>
      <div class="detail-score-label">RISK SCORE &middot; ${escapeHtml(stateLabel)}</div>
    </div>

    ${compareBtnHtml}

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
      <div class="metric-cell">
        <div class="metric-label">60-min coverage</div>
        <div class="metric-value ${metricClass(lga.pop_pct_60min, 30, 60)}">${escapeHtml(fmtMetric(lga.pop_pct_60min))}</div>
        <div class="metric-unit">% pop within 60min drive</div>
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
    ${shapHtml}
    ${confDetailHtml}
    ${metaHtml}
    ${downloadHtml}
  `;
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
  renderHotspots();
  updateLegend();
  renderMap();
  pushStateToPython();
  queueEvent('focus_change', { focus: currentFocus });
}

function selectLGA(id) {
  const base = lgaById.get(String(id));
  if (!base) return;

  selectedLGA = { ...base };
  if (injected.selected && String(injected.selected.id) === String(id)) {
    selectedLGA = mergeLga(selectedLGA, injected.selected);
  }

  renderHotspots();
  renderDetail();
  openDrawer();
  renderMap();
  pushStateToPython();
  queueEvent('lga_select', { id: String(id), name: selectedLGA?.name });
}

function addToCompare(lgaId) {
  const target = lgaById.get(String(lgaId));
  if (!target) return false;
  if (compareLGAs.find((l) => String(l.id) === String(target.id))) return false;
  if (compareLGAs.length >= 4) return false;

  compareLGAs.push({ ...target });
  renderCompareSlots();
  if (selectedLGA) renderDetail();
  pushStateToPython();
  return true;
}

function addCompareSlot() {
  if (!selectedLGA) return;
  addToCompare(selectedLGA.id);
}

function seedCompareFromHotspots() {
  const rankedSets = [
    Array.isArray(window.__hdHotspots) ? window.__hdHotspots : [],
    hotspotsBase(),
  ];
  for (const ranked of rankedSets) {
    for (const lga of ranked) {
      const id = String(lga?.lga_id ?? lga?.id ?? '');
      if (!id) continue;
      addToCompare(id);
      if (compareLGAs.length >= 2) break;
    }
    if (compareLGAs.length >= 2) break;
  }
  return compareLGAs.length >= 2;
}

function removeCompare(id) {
  compareLGAs = compareLGAs.filter((l) => String(l.id) !== String(id));
  renderCompareSlots();
  if (selectedLGA) renderDetail();
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
    addBtn.textContent = '+ Add current LGA';
    addBtn.addEventListener('click', addCompareSlot);
    slots.appendChild(addBtn);
  }

  const fabCount = document.getElementById('compare-fab-count');
  if (fabCount) {
    fabCount.textContent = compareLGAs.length > 0 ? String(compareLGAs.length) : '';
  }

  compareBtn.disabled = compareLGAs.length < 2;
}

function runCompare() {
  if (compareLGAs.length < 2) {
    alert('Please add at least 2 LGAs to compare.');
    return;
  }

  const metrics = [
    { key: 'year',       label: 'Year',               higherIsBad: null,  format: (v) => v != null ? String(v) : '—' },
    { key: 'risk_total', label: 'Risk score (0–10)',   higherIsBad: true,  format: (v) => v != null ? Number(v).toFixed(1) : '—' },
    { key: 'fac',        label: 'Facilities / 10k',    higherIsBad: false, format: fmtMetric },
    { key: 'dist',       label: 'Avg distance (km)',   higherIsBad: true,  format: fmtMetric },
    { key: 'u5mr',       label: 'Under-5 mortality',   higherIsBad: true,  format: fmtMetric },
    { key: 'cov',        label: '5km coverage %',      higherIsBad: false, format: fmtMetric },
    { key: 'towers',     label: 'Towers / 10k',        higherIsBad: false, format: fmtMetric },
  ];

  function cellStyle(rawValues, idx, higherIsBad) {
    if (higherIsBad === null) return '';
    const nums = rawValues.map((v) => safeNum(v)).filter((v) => v != null);
    if (nums.length < 2) return '';
    const val = safeNum(rawValues[idx]);
    if (val == null) return '';
    const best = higherIsBad ? Math.min(...nums) : Math.max(...nums);
    const worst = higherIsBad ? Math.max(...nums) : Math.min(...nums);
    if (val === best) return ` style="background:rgba(0,114,178,0.13);color:${STATUS_COLOR_GOOD};font-weight:600"`;
    if (val === worst) return ` style="background:rgba(213,94,0,0.11);color:${STATUS_COLOR_BAD}"`;
    return '';
  }

  const confBands = compareLGAs.map((l) => {
    const c = confidenceBadge(l.confidence_pct);
    return `${c.emoji} ${c.label}`;
  });

  const headerCells = compareLGAs
    .map((l) => `<th>${escapeHtml(sanitizeText(l.name, 'LGA'))}<br><span class="compare-state-tag">${escapeHtml(sanitizeText(l.state, ''))}</span></th>`)
    .join('');

  const confRow = `
    <tr>
      <td class="metric-name">Data confidence</td>
      ${confBands.map((b) => `<td class="metric-val">${escapeHtml(b)}</td>`).join('')}
    </tr>`;

  const metricRows = metrics.map((metric) => {
    const rawValues = compareLGAs.map((l) =>
      metric.key === 'year' ? l.year : safeNum(l[metric.key])
    );
    const cells = rawValues.map((val, idx) => {
      const style = cellStyle(rawValues, idx, metric.higherIsBad);
      return `<td class="metric-val"${style}>${escapeHtml(metric.format(val))}</td>`;
    }).join('');
    return `<tr><td class="metric-name">${escapeHtml(metric.label)}</td>${cells}</tr>`;
  }).join('');

  const html = `
    <div class="compare-overlay" id="compare-overlay" role="dialog" aria-modal="true" aria-label="LGA comparison">
      <div class="compare-modal">
        <div class="compare-header">
          <h2 class="compare-title">LGA Comparison</h2>
          <button type="button" class="close-btn" id="compare-close-btn" aria-label="Close comparison">×</button>
        </div>
        <div class="compare-legend-note">
          <span style="color:${STATUS_COLOR_GOOD}">■</span> Best on metric &nbsp;
          <span style="color:${STATUS_COLOR_BAD}">■</span> Worst on metric &nbsp;
          <span style="opacity:0.5">■</span> Mid / no data
        </div>
        <div class="compare-body">
          <table class="compare-table">
            <thead><tr><th>Metric</th>${headerCells}</tr></thead>
            <tbody>${confRow}${metricRows}</tbody>
          </table>
        </div>
        <p class="compare-disclaimer">
          Decision-support only. Scores reflect data availability and modelling assumptions, not ground truth.
          Always combine with local field knowledge before planning decisions.
        </p>
        <div class="compare-footer">
          <button type="button" class="dl-btn" id="compare-download-btn">Download CSV</button>
        </div>
      </div>
    </div>
  `;

  const existing = document.getElementById('compare-overlay');
  if (existing) existing.remove();

  document.body.insertAdjacentHTML('beforeend', html);

  document.getElementById('compare-close-btn')?.addEventListener('click', () => {
    document.getElementById('compare-overlay')?.remove();
  });

  document.getElementById('compare-download-btn')?.addEventListener('click', () => {
    const allRows = [
      ['Metric', ...compareLGAs.map((l) => sanitizeText(l.name, 'LGA'))],
      ['Data confidence', ...confBands],
      ...metrics.map((metric) => [
        metric.label,
        ...compareLGAs.map((l) => {
          const val = metric.key === 'year' ? l.year : safeNum(l[metric.key]);
          return metric.format(val);
        }),
      ]),
    ];
    const csv = allRows.map((r) => r.map(csvSafe).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lga_comparison_${currentState}_${currentYear}.csv`;
    a.click();
  });

  openOverlay('compare-overlay');
}

function initMap() {
  if (mapInstance) {
    queueMapResize();
    return;
  }
  if (typeof L === 'undefined') {
    setMapEmptyState('Map library failed to load. Try refreshing.');
    return;
  }

  const mapEl = document.getElementById('map-leaflet');
  if (!mapEl) return;

  const rect = mapEl.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    scheduleMapInit();
    return;
  }

  mapInstance = L.map(mapEl, {
    zoomControl: false,
    attributionControl: true,
    minZoom: 5,
    maxZoom: 12,
  }).setView([9.1, 8.7], 6);

  L.control.zoom({ position: isMobile ? 'bottomright' : 'topleft' }).addTo(mapInstance);

  if (mapInstance.fullscreenControl == null && L.Control.Fullscreen) {
    mapInstance.addControl(new L.Control.Fullscreen({ position: 'topleft' }));
  }

  applyMapModeBaseLayer();
  queueMapResize();
}

function setMapEmptyState(message) {
  const mapArea = document.querySelector('.map-area');
  if (!mapArea) return;

  let empty = document.getElementById('map-empty-state');
  if (!empty) {
    empty = document.createElement('div');
    empty.id = 'map-empty-state';
    empty.className = 'map-empty-state';
    mapArea.appendChild(empty);
  }

  empty.textContent = message || 'Map data unavailable. Try refreshing.';
}

function clearMapEmptyState() {
  document.getElementById('map-empty-state')?.remove();
}

function queueMapResize(delay = 0) {
  if (!mapInstance) return;
  window.setTimeout(() => {
    if (!mapInstance) return;
    try {
      mapInstance.invalidateSize();
    } catch (e) {
      // ignore delayed resize errors
    }
  }, delay);
}

function scheduleMapInit() {
  if (pendingMapInitFrame != null) return;
  pendingMapInitFrame = window.requestAnimationFrame(() => {
    pendingMapInitFrame = null;
    renderMap();
  });
}

function reportBootError(err) {
  console.error('[HDS] boot error:', err);
  const list = document.getElementById('hotspot-list');
  if (list && !list.children.length) {
    const empty = document.createElement('div');
    empty.className = 'hotspot-empty';
    empty.textContent = 'App failed to initialize. Open the iframe console for details.';
    list.replaceChildren(empty);
  }
  setMapEmptyState('App failed to initialize. Open the iframe console for details.');
  scheduleBootOverlayHide(0);
}

function normalizeGeoJson() {
  if (baseGeoJson) return baseGeoJson;

  let gj = null;
  if (injected.map?.geojson) {
    try {
      gj = JSON.parse(injected.map.geojson);
    } catch (e) {
      setMapEmptyState('Map data unavailable. Try refreshing.');
      return null;
    }
  }

  if (!gj || !Array.isArray(gj.features) || !gj.features.length) {
    setMapEmptyState('Map data unavailable. Try refreshing.');
    return null;
  }

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

  clearMapEmptyState();
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

function quantileFromSorted(sorted, q) {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  const next = sorted[base + 1];
  if (next === undefined) return sorted[base];
  return sorted[base] + rest * (next - sorted[base]);
}

function getLayerStats(layer) {
  const key = `stats:${layer}`;
  if (fieldValuesCache.has(key)) return fieldValuesCache.get(key);

  const values = lgas
    .map((l) => valueForLayer(l, layer))
    .filter((v) => v != null && !Number.isNaN(v))
    .sort((a, b) => a - b);

  const stats = {
    min: values.length ? values[0] : 0,
    q25: quantileFromSorted(values, 0.25) ?? 0,
    q50: quantileFromSorted(values, 0.5) ?? 0,
    q75: quantileFromSorted(values, 0.75) ?? 0,
    max: values.length ? values[values.length - 1] : 0,
  };
  fieldValuesCache.set(key, stats);
  return stats;
}

function interpolateBadness(value, bad, mid, good) {
  if (value == null || Number.isNaN(value)) return null;
  const safeBad = Number.isFinite(bad) ? bad : value;
  const safeMid = Number.isFinite(mid) ? Math.max(mid, safeBad + 1e-6) : safeBad + 1;
  const safeGood = Number.isFinite(good) ? Math.max(good, safeMid + 1e-6) : safeMid + 1;

  if (value <= safeBad) return 10;
  if (value >= safeGood) return 0;
  if (value <= safeMid) {
    const t = (value - safeBad) / (safeMid - safeBad);
    return 10 - (t * 4.5);
  }
  const t = (value - safeMid) / (safeGood - safeMid);
  return 5.5 - (t * 5.5);
}

function badnessForLayer(lga, layer = currentLayer) {
  const val = valueForLayer(lga, layer);
  if (val == null || Number.isNaN(val)) return null;

  switch (layer) {
    case 'Facilities':
      // Align map colors with the panel's facility-access semantics.
      return interpolateBadness(val, 0.5, 1.5, 3.0);
    case 'Connectivity':
    case 'Towers': {
      const stats = getLayerStats(layer);
      return interpolateBadness(
        val,
        Math.max(0.25, stats.q25 ?? 0.25),
        Math.max(1.5, stats.q50 ?? 1.5),
        Math.max(5.0, stats.q75 ?? 5.0),
      );
    }
    case 'Risk score':
      return scoreOutOfTen(lga);
    default:
      return null;
  }
}

function colorForLayer(lga, layer = currentLayer) {
  if (!lga) return 'rgba(80,80,90,0.4)';
  if (layer === 'Risk score') return getColor(scoreOutOfTen(lga));

  const badness = badnessForLayer(lga, layer);
  if (badness != null) return getColor(badness);

  const norm = scaledLayerValue(lga.id);
  return riskColorHex(norm);
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
  if (currentLayer === 'Risk score') {
    const score = scoreOutOfTen(lga);
    return score == null ? 'NA' : score.toFixed(2);
  }
  return fmtMetric(val);
}

function tooltipMetricForFocus(props) {
  switch (currentFocus) {
    case 'Child mortality':
      return {
        display: fmtMetric(props.u5mr ?? props.u5_mortality_rate ?? props.u5mr_mean),
        label: 'U5 MORTALITY',
        color: getColor(getFocusColorScore(props, currentFocus)),
      };
    case 'Facility access':
      return {
        display: fmtMetric(props.fac ?? props.facilities_per_10k),
        label: 'FACILITIES / 10K',
        color: getColor(getFocusColorScore(props, currentFocus)),
      };
    case 'Connectivity':
      return {
        display: fmtMetric(props.towers ?? props.towers_per_10k ?? props.connectivity_score ?? props.cov ?? props.coverage_5km),
        label: hasTowerConnectivityData ? 'TOWERS / 10K' : 'CONNECTIVITY',
        color: getColor(getFocusColorScore(props, currentFocus)),
      };
    case '5km coverage':
      return {
        display: fmtMetric(props.cov ?? props.coverage_5km),
        label: '5KM COVERAGE',
        color: getColor(getFocusColorScore(props, currentFocus)),
      };
    case '60-min coverage': {
      const value = safeNum(props.pop_pct_60min);
      let color = '#cbd5e1';
      if (value != null && value < 40) color = '#d73027';
      else if (value != null && value < 70) color = '#ff6b35';
      return {
        display: fmtMetric(props.pop_pct_60min),
        label: '60-MIN COVERAGE',
        color,
      };
    }
    default: {
      const score = Number(scoreOutOfTen(props) || 0);
      return {
        display: score.toFixed(2),
        label: 'RISK SCORE',
        color: score > 5.5 ? '#d73027' : '#2166ac',
      };
    }
  }
}

function buildTooltipHTML(props) {
  const metric = tooltipMetricForFocus(props);
  const driver = sanitizeText(props.worst_driver, '');
  const driverHtml = driver
    ? `<div style="margin-top:5px;font-size:8px;padding:2px 6px;border-radius:3px;
        background:rgba(215,48,39,0.12);color:#fca5a5;display:inline-block;
        letter-spacing:.04em">${escapeHtml(driver)}</div>`
    : '';
  return `
    <div style="font-family:'IBM Plex Mono',monospace;min-width:130px;padding:2px">
      <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:14px;
        color:#e8eaf0;margin-bottom:2px">${escapeHtml(sanitizeText(props.lga_name || props.name, ''))}</div>
      <div style="font-size:9px;color:${TEXT_COLOR_MUTED};letter-spacing:.06em;margin-bottom:6px">
        ${escapeHtml(sanitizeText(props.state_name || props.state, ''))}</div>
      <div style="font-size:28px;font-weight:800;font-family:'Syne',sans-serif;
        color:${metric.color};line-height:1">${escapeHtml(metric.display)}</div>
      <div style="font-size:8px;color:${TEXT_COLOR_DIM};letter-spacing:.1em;
        text-transform:uppercase;margin-top:1px">${escapeHtml(metric.label)}</div>
      ${driverHtml}
    </div>
  `;
}

function ensureLayerTooltip(layer, props) {
  if (!layer) return;
  const tooltipHtml = buildTooltipHTML(props);
  if (typeof layer.getTooltip === 'function' && layer.getTooltip()) {
    layer.setTooltipContent(tooltipHtml);
    return;
  }
  layer.bindTooltip(tooltipHtml, {
    sticky: true,
    direction: 'top',
    className: 'hd-tooltip',
  });
}

function styleForFeatureId(id) {
  const lga = lgaById.get(String(id));
  const fill = fillColorForFeature(lga, currentLayer);
  const selected = selectedLGA && String(selectedLGA.id) === String(id);
  const modeCfg = getMapModeConfig();
  return {
    color: selected ? '#ffffff' : modeCfg.strokeColor,
    weight: selected ? 2.5 : modeCfg.strokeWeight,
    opacity: selected ? 1 : modeCfg.strokeOpacity,
    fillColor: fill,
    fillOpacity: fillOpacityForFeature(lga),
  };
}

function initGeoLayer() {
  if (!mapInstance || geoLayer) return false;

  const gj = normalizeGeoJson();
  if (!gj) return false;

  geoLayer = L.geoJSON(gj, {
    style: (feature) => {
      const id = String(feature.properties?.lga_uid ?? feature.properties?.lga_name ?? '');
      return styleForFeatureId(id);
    },
    onEachFeature: (feature, layer) => {
      const id = String(feature.properties?.lga_uid ?? feature.properties?.lga_name ?? '');
      const props = {
        ...(feature.properties || {}),
        ...(lgaById.get(id) || {}),
      };
      feature.properties = props;

      featureLayerById.set(id, layer);
      layer.on('mouseover', () => {
        const nextProps = {
          ...(feature.properties || {}),
          ...(lgaById.get(id) || {}),
        };
        feature.properties = nextProps;
        ensureLayerTooltip(layer, nextProps);
        if (typeof layer.openTooltip === 'function') {
          layer.openTooltip();
        }
      });
      layer.on('click', () => selectLGA(id));
    },
  }).addTo(mapInstance);

  if (!hasFitBounds) {
    try {
      fitMapToCurrentState();
      hasFitBounds = true;
    } catch (e) {
      // ignore fit errors
    }
  }

  queueMapResize();
  return true;
}

function renderMap() {
  initMap();
  if (!mapInstance) return;

  const layerWasCreated = initGeoLayer();
  if (!geoLayer) return;
  if (layerWasCreated) return;

  featureLayerById.forEach((layer, id) => {
    layer.setStyle(styleForFeatureId(id));
    const feature = layer.feature || {};
    const props = {
      ...(feature.properties || {}),
      ...(lgaById.get(String(id)) || {}),
    };
    feature.properties = props;
    if (typeof layer.getTooltip === 'function' && layer.getTooltip()) {
      layer.setTooltipContent(buildTooltipHTML(feature.properties));
    }
  });

  queueMapResize();
}

function renderMapTable() {
  const tableWrap = document.getElementById('map-table');
  if (!tableWrap) return;
  // Show all LGAs (no artificial limit)
  const rows = lgas;
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
  // Add count indicator
  const countInfo = `<div class="map-table-count">Showing all ${rows.length} LGAs</div>`;
  tableWrap.innerHTML = countInfo + `<table class="map-table-inner">${header}${body}</table>`;
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
  const url = getAppStateBaseUrl();
  if (!url) return `/static/share_preview.html?${params.toString()}`;
  return `${url.origin}/static/share_preview.html?${params.toString()}`;
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

function syncMobileMoreMeta() {
  const target = document.getElementById('mobile-more-meta-text');
  if (!target) return;
  const source = document.getElementById('dataset-meta-text');
  target.textContent = source?.textContent || `${lgas.length} LGAs · ${currentYear}`;
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
  const modelVersion = Array.isArray(meta.model_version) ? meta.model_version.join(', ') : meta.model_version || 'v1.4';
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
    '# Data Sources: DHS 2013, 2018, 2024 · NHFR 2020 · WorldPop 2024 · ORS isochrones · OpenCellID 2019',
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
    { key: 'pop_pct_60min', label: 'pop_pct_within_60min_drive' },
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
    model_version: meta.model_version || 'v1.4',
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
  if (typeof JSZip === 'undefined') {
    try {
      await ensureExternalScript('assets/jszip.min.js', 'JSZip');
    } catch (e) {
      throw new Error('Bundle export is unavailable in this embedded render.');
    }
  }
  if (typeof JSZip === 'undefined') {
    throw new Error('Bundle export is unavailable in this embedded render.');
  }
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
  if (currentExportMode === 'bundle') {
    try {
      payload = await buildBundleExport();
    } catch (e) {
      console.warn('[HDS] bundle export unavailable:', e?.message || e);
      setExportMode('csv');
      payload = buildCsvExport();
    }
  }
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

const TOUR_STEPS = [
  {
    title: 'THE SCORE',
    body: `Each LGA in Nigeria gets a <strong style="color:#f97316">0-10 risk score</strong>.
      Higher means harder to reach care.<br><br>
      <div style="font-family:'Syne',sans-serif;font-size:48px;font-weight:800;
        color:#f97316;line-height:1;margin:12px 0">7.22</div>
      <div style="font-size:10px;color:${TEXT_COLOR_DIM};letter-spacing:.1em">
        TAMBUWAL &middot; SOKOTO &middot; EXAMPLE</div>
      <br>Score is based on facility density, distance to care,
      child mortality, and mobile coverage.`,
    cta: 'See the map ->',
  },
  {
    title: 'THE MAP',
    body: `<strong style="color:${SCORE_COLOR_HIGH}">Warm</strong> means high risk.
      <strong style="color:${SCORE_COLOR_LOW}">Cool</strong> means lower risk.<br><br>
      The northwest cluster - Sokoto, Zamfara, Katsina - is Nigeria's
      hardest-access zone. The South shows better access on average,
      but significant LGA-level variation exists everywhere.<br><br>
      Hover any LGA to see its score. Click to open full details.`,
    cta: 'See the filters ->',
  },
  {
    title: 'FILTER BY STATE',
    body: `Use the <strong style="color:#f97316">STATE dropdown</strong>
      in the header to focus on your state. The map zooms in,
      non-selected LGAs dim, and the list updates to show only
      your state's LGAs ranked by risk.<br><br>
      Use the <strong>Risk dimension chips</strong> to re-rank by
      a single factor: mortality, facilities, or connectivity.`,
    cta: 'See the comparison ->',
  },
  {
    title: 'COMPARE LGAS',
    body: `Select up to three LGAs and run a side-by-side comparison.
      The heatmap shows which LGA scores worst on each dimension -
      green is better, red is worse.<br><br>
      <em style="color:${TEXT_COLOR_DIM};font-size:10px">
      Decision-support only. Always combine with local field knowledge
      before planning decisions.</em><br><br>
      Two LGAs have been pre-loaded for you. Run the comparison now ->`,
    cta: 'Start exploring',
    isLast: true,
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
  if (markComplete) markTourComplete();
}

function renderTourStep() {
  const titleEl = document.getElementById('tour-step-title');
  const bodyEl = document.getElementById('tour-step-body');
  if (!titleEl || !bodyEl) return;
  const step = TOUR_STEPS[tourIndex] || TOUR_STEPS[0];
  titleEl.textContent = step.title;
  bodyEl.innerHTML = step.body;
  const nextBtn = document.getElementById('tour-next-btn');
  if (nextBtn) nextBtn.textContent = step.cta || (step.isLast ? 'Start exploring' : 'Next');
}

function maybeStartTour() {
  if (pwaMode) return;
  const completed = localStorage.getItem(TOUR_STORAGE_KEY) === '1';
  if (!completed) openTour();
}

function syncMobileMoreMeta() {
  const target = document.getElementById('mobile-more-meta-text');
  if (!target) return;
  const count = document.getElementById('lga-count')?.textContent || String(lgas.length);
  target.textContent = `${count} LGAs - ${currentYear}`;
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

  updateLegend();
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
  const next = window.innerWidth <= 768
    || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '');
  document.documentElement.setAttribute('data-mobile', next ? '1' : '0');
  if (next !== isMobile) {
    isMobile = next;
    document.body.classList.toggle('is-mobile', isMobile);
    applyDepthVisibility();
    pushStateToPython({ immediate: true });
  } else {
    document.body.classList.toggle('is-mobile', isMobile);
    applyDepthVisibility();
  }
}

let eventsWired = false;
let supportUiBootstrapped = false;
let lifecycleHandlersBound = false;

function syncStateSelectOptions() {
  const stateSelect = document.getElementById('state-select');
  if (stateSelect) {
    const previousValue = stateSelect.value || currentState;
    stateSelect.replaceChildren();
    stateOptions.forEach((st) => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      stateSelect.appendChild(opt);
    });
    stateSelect.value = stateOptions.includes(currentState) ? currentState : previousValue;
  }

  const stateMobile = document.getElementById('state-select-mobile');
  if (stateMobile) {
    const previousValue = stateMobile.value || currentState;
    stateMobile.replaceChildren();
    stateOptions.forEach((st) => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      stateMobile.appendChild(opt);
    });
    stateMobile.value = stateOptions.includes(currentState) ? currentState : previousValue;
  }

  const yearSelect = document.getElementById('year-select');
  if (yearSelect) yearSelect.value = currentYear;
  const yearMobile = document.getElementById('year-select-mobile');
  if (yearMobile) yearMobile.value = currentYear;
}

function configureSupportLinks() {
  const methodLink = document.getElementById('methodology-link');
  const glossaryLink = document.getElementById('glossary-link');
  const mobileMethodLink = document.getElementById('mobile-more-methodology-link');
  const mobileGlossaryLink = document.getElementById('mobile-more-glossary-link');
  [methodLink, glossaryLink, mobileMethodLink, mobileGlossaryLink].forEach((link) => {
    if (!link) return;

    const rawHref = link.getAttribute('href') || '';
    const parentUrl = getAppStateBaseUrl();
    const baseHref = parentUrl ? `${parentUrl.origin}${parentUrl.pathname}` : 'http://localhost/';
    const parsed = new URL(rawHref, baseHref);
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

function primeDeferredUi() {
  if (supportUiBootstrapped) return;
  supportUiBootstrapped = true;
  applyHelpTooltips();
  configureSupportLinks();
  initBottomSheetDrag();
}

function wireEvents() {
  syncStateSelectOptions();
  if (eventsWired) return;
  eventsWired = true;

  const stateSelect = document.getElementById('state-select');
  const yearSelect = document.getElementById('year-select');
  const stateMobile = document.getElementById('state-select-mobile');
  const yearMobile = document.getElementById('year-select-mobile');

  stateSelect?.addEventListener('change', (e) => {
    handleStateChange(e.target.value);
  });

  yearSelect?.addEventListener('change', (e) => {
    currentYear = e.target.value;
    syncHeader();
    renderHotspots();
    renderMap();
    setApplyStatus(`Year updated to ${currentYear}`, 'updating');
    pushStateToPython();
    queueEvent('filter_change', { year: currentYear });
  });

  stateMobile?.addEventListener('change', (e) => {
    if (stateSelect) stateSelect.value = e.target.value;
    handleStateChange(e.target.value);
  });

  yearMobile?.addEventListener('change', (e) => {
    currentYear = e.target.value;
    if (yearSelect) yearSelect.value = currentYear;
    syncHeader();
    renderHotspots();
    renderMap();
    setApplyStatus(`Year updated to ${currentYear}`, 'updating');
    pushStateToPython();
    queueEvent('filter_change', { year: currentYear });
  });

  const searchInput = document.getElementById('search-input');
  if (searchInput) searchInput.addEventListener('input', renderHotspots);

  document.querySelectorAll('.depth-btn').forEach((btn) => {
    btn.addEventListener('click', () => setDepth(btn.dataset.depth));
  });

  document.querySelectorAll('.focus-section .chip').forEach((btn) => {
    btn.addEventListener('click', () => setFocus(btn.dataset.focus || 'All risk'));
  });

  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchMapMode(btn.dataset.mode || 'polygon'));
  });

  document.querySelectorAll('.layer-btn').forEach((btn) => {
    btn.addEventListener('click', () => toggleLayer(btn.dataset.layer || 'Risk score'));
  });

  document.getElementById('compare-go-btn')?.addEventListener('click', runCompare);
  document.getElementById('compare-add-btn')?.addEventListener('click', addCompareSlot);
  document.getElementById('compare-fab')?.addEventListener('click', () => {
    document.getElementById('compare-strip')?.scrollIntoView({ behavior: 'smooth' });
  });

  document.getElementById('map-table-toggle')?.addEventListener('click', toggleMapTable);
  const mapToggle = document.getElementById('map-table-toggle');
  if (mapToggle) mapToggle.setAttribute('aria-expanded', 'false');

  document.getElementById('detail-drawer')?.addEventListener('click', (e) => {
    if (e.target.closest('#detail-close-btn')) {
      closeDrawer();
      return;
    }
    if (e.target.closest('#download-lga-btn')) {
      downloadLGA();
      return;
    }
    if (e.target.closest('#add-to-compare-btn')) {
      if (!compareLGAs.some((l) => String(l.id) === String(selectedLGA?.id))) {
        addCompareSlot();
      }
      renderDetail();
      return;
    }
  });

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

  document.getElementById('mobile-more-btn')?.addEventListener('click', () => {
    syncMobileMoreMeta();
    openOverlay('mobile-more-drawer');
  });
  document.getElementById('desktop-more-btn')?.addEventListener('click', () => {
    syncMobileMoreMeta();
    openOverlay('mobile-more-drawer');
  });
  document.getElementById('mobile-more-close-btn')?.addEventListener('click', () => closeOverlay('mobile-more-drawer'));
  document.getElementById('mobile-more-tour-btn')?.addEventListener('click', () => {
    closeOverlay('mobile-more-drawer');
    tourIndex = 0;
    openTour();
  });

  document.getElementById('tour-skip-btn')?.addEventListener('click', () => closeTour(true));
  document.getElementById('tour-next-btn')?.addEventListener('click', () => {
    if (tourIndex >= TOUR_STEPS.length - 1) {
      closeTour(true);
      seedCompareFromHotspots();
      return;
    }
    tourIndex += 1;
    renderTourStep();
  });
}

let appBooted = false;

function bootApp({ force = false, skipDeferredWork = false } = {}) {
  if ((!window.__INITIAL_DATA__ && !meta.focus) || bootstrappingLatestYear) return;
  if (appBooted && !force) return;
  try {
    wireEvents();
    detectMobile();
    attachStandaloneMessageHandler();

    if (!lifecycleHandlersBound) {
      lifecycleHandlersBound = true;
      window.addEventListener('resize', () => {
        detectMobile();
        queueMapResize(80);
      });
      window.addEventListener('load', () => queueMapResize(180), { once: true });
    }

    if (!appBooted && testingMode && testSession) {
      pushStateToPython({ immediate: true });
    }

    syncStateSelectOptions();
    syncHeader();
    applyDepthVisibility();
    renderHotspots();
    switchMapMode(currentMapMode);

    if (selectedLGA) {
      openDrawer();
      if (force || skipDeferredWork) {
        renderDetail();
      }
    } else {
      document.getElementById('detail-drawer')?.classList.remove('open');
      document.getElementById('drawer-backdrop')?.remove();
    }

    if (document.getElementById('map-table')?.classList.contains('open')) {
      renderMapTable();
    }

    if (skipDeferredWork) {
      renderCompareSlots();
      syncMobileMoreMeta();
    } else {
      scheduleNonCriticalWork(() => {
        primeDeferredUi();
        renderCompareSlots();
        syncMobileMoreMeta();
        if (selectedLGA) renderDetail();
      });
      if (!appBooted) {
        scheduleNonCriticalWork(() => maybeStartTour(), 1200);
      }
    }

    appBooted = true;
    scheduleBootOverlayHide(force ? 0 : 80);
  } catch (err) {
    reportBootError(err);
  }
}

if (window.__INITIAL_DATA__ && !bootstrappingLatestYear) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootApp, { once: true });
  } else {
    bootApp();
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  closeOverlay('share-drawer');
  closeOverlay('export-drawer');
  closeOverlay('mobile-more-drawer');
  if (isTourOpen()) {
    markTourComplete();
    closeTour();
  }
});
