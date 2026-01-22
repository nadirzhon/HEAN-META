# HEAN React/Vite UI (Dockerized)

This UI is packaged for both live-reload development and static production hosting alongside the HEAN backend.

## Environment
- Copy `.env.example` for local runs or `.env.docker.example` when using compose: `cp .env.docker.example .env`.
- Default websocket path is `/ws` (from `src/hean/api/main.py`).

## Dev (hot reload)
- Start API + UI dev server:
  - `docker compose --profile dev up -d --build api ui-dev`
- Open http://localhost:5173
- The container connects to the `api` service at `http://api:8000` and `ws://api:8000/ws`.
- If you change dependencies, rebuild the image; source edits hot-reload via mounted files.

## Prod (static nginx, default in docker-compose.yml)
- `docker compose up -d --build ui`
- Open http://localhost:3000
- Build args default to `http://localhost:8000` / `ws://localhost:8000/ws`; override with `VITE_API_BASE` / `VITE_WS_URL` env vars when building.
