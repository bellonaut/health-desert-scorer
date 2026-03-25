const SHELL_CACHE = 'hds-shell-v2';
const DATA_CACHE = 'hds-data-v1';
const APP_SHELL = [
  '/',
  '/health_desert_ui.html',
  '/health_desert_ui.css',
  '/health_desert_ui.js',
  '/sw.js',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/assets/leaflet.css',
  '/assets/leaflet.js',
];

async function cacheFirst(request, cacheName = SHELL_CACHE) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request, { ignoreSearch: true });
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(DATA_CACHE);
  const cached = await cache.match(request);
  const networkResponse = await fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);

  return cached || networkResponse || Response.error();
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys
        .filter((key) => ![SHELL_CACHE, DATA_CACHE].includes(key))
        .map((key) => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (url.pathname === '/api/data' || url.pathname.endsWith('/api/data')) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  if (request.mode === 'navigate') {
    event.respondWith(cacheFirst('/health_desert_ui.html'));
    return;
  }

  const isShellAsset = (
    url.pathname.endsWith('.html')
    || url.pathname.endsWith('.css')
    || url.pathname.endsWith('.js')
    || url.pathname.startsWith('/assets/')
    || url.pathname.startsWith('/static/')
  );

  if (isShellAsset) {
    event.respondWith(cacheFirst(request));
  }
});
