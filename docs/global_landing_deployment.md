# Global Landing Deployment Plan

This repo now treats the globe landing page as a static artifact, not a Streamlit page.

## Routing target
- `https://healthdesert.io/` -> global globe (`app/static/global.html`)
- `https://healthdesert.io/ng` -> Nigeria app (standalone PWA served by FastAPI)
- `https://healthdesert.io/ca` -> Canada app (future)

Legacy Streamlit fallback:
- `https://bashir-healthdesert.streamlit.app/` -> global landing
- `https://bashir-healthdesert.streamlit.app/?app=ng` -> Nigeria dashboard

## Why static-first
- Faster first paint and no Python cold start on landing.
- Better reliability for a lightweight visual page.
- Cleaner separation between marketing/global entry and country apps.

## Current implementation notes
- `app/static/global.html` contains the global page.
- `app/api.py` now serves the public same-origin flow directly: `/` for landing, `/ng` for the Nigeria app, and `/api/data` for payload hydration.
- Pulse rings use JS `requestAnimationFrame` + interval fallback by default for Safari/Firefox compatibility.
- Google Fonts CDN remains in place for now.
- Local/self-hosted fonts are intentionally deferred until offline/field deployment hardening.
- Redirect assets included:
  - `app/static/_redirects` for Cloudflare Pages (and Netlify-style hosts) to same-origin standalone routes
  - `app/static/ng/index.html` for GitHub Pages `/ng` redirect fallback to the same-origin standalone route
  - `app/static/ca/index.html` for GitHub Pages `/ca` placeholder fallback

## Deployment guidance
- Deploy the FastAPI app in `app/api.py` as the public host so `/`, `/ng`, `/api/data`, `/static/*`, and `/assets/*` share one origin.
- If you keep a separate static landing deployment, ensure it routes `/ng` to the same-origin standalone app instead of the legacy Streamlit URL.
- Cloudflare Pages will apply `_redirects` automatically.
- GitHub Pages does not use `_redirects`; `/ng` and `/ca` are handled by folder-level `index.html` redirect pages.
- `global.html` CTA now auto-selects `/ng` on custom domains and standalone localhost, while preserving `/?app=ng` only on Streamlit hosts.
