# Global Landing Deployment Plan

This repo now treats the globe landing page as a static artifact, not a Streamlit page.

## Routing target
- `https://healthdesert.io/` -> global globe (`app/static/global.html`)
- `https://healthdesert.io/ng` -> Nigeria app (Streamlit deployment)
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
- Pulse rings use JS `requestAnimationFrame` + interval fallback by default for Safari/Firefox compatibility.
- Google Fonts CDN remains in place for now.
- Local/self-hosted fonts are intentionally deferred until offline/field deployment hardening.
- Redirect assets included:
  - `app/static/_redirects` for Cloudflare Pages (and Netlify-style hosts)
  - `app/static/ng/index.html` for GitHub Pages `/ng` redirect fallback
  - `app/static/ca/index.html` for GitHub Pages `/ca` placeholder fallback

## Deployment guidance
- Host the static landing on Cloudflare Pages or GitHub Pages.
- Ensure your static publish directory includes `app/static/` files at the web root.
- Cloudflare Pages will apply `_redirects` automatically.
- GitHub Pages does not use `_redirects`; `/ng` and `/ca` are handled by folder-level `index.html` redirect pages.
- `global.html` CTA auto-selects `/ng` on custom domain and `/?app=ng` on Streamlit host.
