# Deploy the MVP Demo

This guide helps you deploy the public-facing demo so YC reviewers can try it online without any local setup.

Two deployables:

- Frontend (Vite React app) → Vercel (recommended)
- Backend (Express + WebSocket) → Railway, Render, or Fly.io (pick one)

Make sure the backend is reachable over HTTPS and CORS allows your frontend origin.

---

## 1) Deploy the Backend API

The backend lives in `backend/` and exposes:

- HTTP: `GET /health`, `GET /devices`, `POST /connect`, `POST /run-demo`, `POST /generate-kernel`
- WebSocket: same host/port as HTTP (used for streaming logs)

Default port is `8787` (controlled by `PORT`). CORS is controlled via `CORS_ORIGIN`.

### A. Railway (fastest path)

1. Create a new Railway project, then “Deploy from GitHub”. Select this repo.
2. In the “Root Directory” or “Build” settings, set the service’s root to `backend/`.
3. Railway auto-detects Node and runs `npm install` then `npm start` (uses `backend/package.json`).
4. Add Environment Variables:
   - `PORT=8787` (Railway will map it)
   - `CORS_ORIGIN=https://your-frontend-domain.vercel.app` (or `*` while testing)
5. Deploy. Note the public URL, e.g. `https://uhop-backend-production.up.railway.app`.

### B. Render

1. Create a new “Web Service”.
2. Select your GitHub repo. Set “Root Directory” to `backend/`.
3. Runtime: Node 18+.
4. Build Command: `npm install`
5. Start Command: `npm start`
6. Environment:
   - `PORT` will be provided by Render; no custom value needed
   - `CORS_ORIGIN=https://your-frontend-domain.vercel.app`
7. Deploy and copy the public URL.

### C. Fly.io

1. Install Fly CLI and run `fly launch` from `backend/`.
2. Choose Node runtime, expose port 8787.
3. Set env:
   - `CORS_ORIGIN=https://your-frontend-domain.vercel.app`
4. `fly deploy` then note the app URL.

### Quick local run (optional)

```bash
cd backend
npm install
PORT=8787 CORS_ORIGIN=http://localhost:5173 npm start
```

Test:

- Health: `GET <BACKEND_URL>/health` → `{ ok: true }`
- WebSocket: connect to `ws://<BACKEND_HOST>/` and you should receive JSON `{ t, line }` when actions occur.

---

## 2) Deploy the Frontend (Vercel)

The frontend lives in the `frontend/` folder and uses a single environment variable:

- `VITE_BACKEND_URL`: The base URL of your backend, e.g. `https://uhop-backend-production.up.railway.app`

Steps:

1. Create a new Vercel project and import your GitHub repo.
2. In “Framework Preset” choose Vite.
3. Set “Root Directory” to `frontend/`.
4. Add Environment Variable in Vercel Project Settings:
   - `VITE_BACKEND_URL` → your backend public URL
5. Build & Deploy.

After deploy, open your Vercel URL and:

- Click “Connect Devices” → should show 3 devices as connected.
- Click “Run Demo” → should stream logs via WebSocket and show result `{ time: 8.23, device: GPU }`.
- Click “Generate Kernel” → displays mock CUDA/OpenCL code and logs.

---

## 3) Notes and troubleshooting

- CORS: If requests fail in browser, ensure backend `CORS_ORIGIN` includes your Vercel domain (or `*` temporarily).
- WebSocket: The app connects with `new WebSocket(backendUrl.replace('http', 'ws'))`.
  - Make sure your host supports WebSockets (Railway/Render/Fly do).
- HTTPS: Use the https URL for `VITE_BACKEND_URL` in production to avoid mixed-content issues.
- Environment drift: If you move the backend between providers, update Vercel `VITE_BACKEND_URL` and redeploy.
- Local dev:
   - Backend: `npm run dev` in `backend/` (nodemon), default `http://localhost:8787`
   - Frontend: `npm run dev` in `frontend/`, and set `VITE_BACKEND_URL=http://localhost:8787`

---

## 4) One-line summary for README

“Deploy the demo: backend to Railway/Render/Fly (set CORS_ORIGIN), frontend to Vercel (set VITE_BACKEND_URL). See docs/DEPLOY.md.”

---

## Appendix: Deploy the Frontend on GitHub Pages (alternative)

You can host the frontend on GitHub Pages under the same repo. This deploys a static site to `gh-pages` and serves it at:

- User site: `https://<username>.github.io/` (root repo named `<username>.github.io`)
- Project site (this repo): `https://sevenloops.github.io/uhop/`

Already set up in this repo:

- Vite `base` is configured to `/uhop/` in production.
- SPA fallback `public/404.html` redirects into the project base and preserves deep links.
- GitHub Actions workflow `.github/workflows/deploy-frontend-pages.yml` builds `frontend/` and deploys `dist/` to Pages on pushes to `main`.

Steps:

1. In repo Settings → Pages, set Source to "GitHub Actions".
2. Push to `main`. The workflow publishes `frontend/dist` to `gh-pages`.
3. Visit `https://sevenloops.github.io/uhop/`.

Notes:

- If your repo/fork lives under a different owner or name, update `frontend/vite.config.ts` and `frontend/public/404.html` base path to match (`/your-repo-name/`).
- Set `VITE_BACKEND_URL` in your frontend build environment to your backend URL. For Pages, you can bake it at build time (e.g., in the deploy workflow or locally before pushing artifacts).

### Custom Domain (uhop.dev)

If you’re using a custom domain like `uhop.dev`:

1. DNS: Create the required records (usually CNAME for `www` to `<username>.github.io` and an A/ALIAS/ANAME for apex if supported, or use a redirect to `www`).
2. In GitHub repo Settings → Pages, add the custom domain `uhop.dev` and enforce HTTPS.
3. The workflow creates a `CNAME` file (`dist/CNAME`) with `uhop.dev` so Pages serves the custom domain correctly.
4. Vite base should be `/` in production (already set). `public/404.html` redirects to `/`.
