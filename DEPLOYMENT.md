# SignSync deployment (Vercel frontend + Oracle Cloud API)

## Current deployment and free Render option

The frontend is deployed at https://sign-sync-lemon.vercel.app. The Python
recognition API still needs a public host; website availability alone does not
mean camera or Testing Lab predictions are connected.

`render.yaml` prepares a Docker web service on Render's Free plan. Its CORS
origin matches the Vercel production URL. The Docker image installs CPU-only
PyTorch and listens on all interfaces, using the host-provided `PORT`.

Deploy the current project files from your Git repository using a Render
Blueprint. Keep the service on the Free plan. Free instances have limited
memory and sleep when idle, so verify that `/api/status` returns `ready: true`
and that image/video inference succeeds before considering this backend ready.
Do not upgrade to paid compute to resolve a memory failure without approval.

Once the backend is healthy, set Vercel's production `SIGNSYNC_API_BASE` to
its HTTPS origin (without `/api`) and redeploy the frontend. Check predictions
from the public site, including CORS, before sharing it as a working recognizer.

See [Render free service limits](https://render.com/docs/free).

SignSync's browser app is built as a static Vercel deployment. The existing
Python service, MediaPipe hand tracker, MLP/LSTM checkpoints, and video decoder
run together in a Docker container on an Oracle Cloud VM. The Vercel build
publishes only the frontend files; it does not package Python, model checkpoints,
training data, or repository files as public website content.

## Why the API is hosted separately

Vercel Function request and response bodies are limited to 4.5 MB. SignSync now
limits each test file to 4 MB and sends a batch as separate, sequential requests.
However, this deployment builds a static frontend and sends uploads directly to
the separate Python API; the limit alone does not move inference onto Vercel.
The live J/Z recognizer also keeps an in-memory sequence of camera frames, which
needs a stable backend process. The browser calls that process directly over
HTTPS.

## 1. Push the prepared project to GitHub

Commit and push this project, including `Dockerfile`, `compose.yaml`, `Caddyfile`,
`vercel.json`, `package.json`, `package-lock.json`, `.vercelignore`,
`assets/deployment-config.js`, and `scripts/build-vercel.mjs`, together with the
frontend API URL changes. Do not commit your real `.env` file.
An untracked local file is not available to Vercel's GitHub deployment. Ensure
Vercel deploys the branch containing these changes (currently
`feature/offline-sign-testing`), or merge them into your production branch first.

## 2. Deploy the static frontend to Vercel

1. Import the GitHub repository into Vercel and select **Other** as the
   framework preset. Keep the repository root as the project root. The checked-in
   `vercel.json` explicitly selects `@vercel/static-build` using `package.json`.
   That builder runs `npm run vercel-build` (the Node copy/configuration script)
   and publishes `vercel-dist`. It installs no Python dependencies and creates
   no Python functions. No Python entry point is required.
2. Let the initial deployment finish and note the exact production origin, such
   as `https://signsync-example.vercel.app`. Recognition won't connect until the
   backend is deployed and `SIGNSYNC_API_BASE` is configured in step 3.

### Fixing an existing PYTHON_ENTRYPOINT_NOT_FOUND deployment

1. Commit and push all deployment files above, and confirm they appear in the
   GitHub branch Vercel is deploying. Redeploying an older commit cannot include
   these fixes.
2. Set the Vercel project's Root Directory to the folder containing `index.html`,
   `package.json`, and `vercel.json` (leave it at the repository root if that is
   where they are). Select framework **Other** and remove old Python install,
   build, or entry-point overrides.
3. Redeploy the new commit with the existing build cache disabled. With an
   explicit `builds` configuration, Vercel may warn that dashboard Build and
   Development settings are ignored; that is expected. The builder and output
   directory are defined in the committed configuration.
4. Confirm the log uses `@vercel/static-build` and runs `vercel-build`. A Python
   entry-point error still indicates the wrong commit, root directory, or
   configuration is being used.

The explicit `builds` setting is a supported legacy Vercel configuration. It is
intentional here: it selects only the static frontend builder in this mixed
Python/HTML repository. `.vercelignore` also limits deployment uploads to the
frontend's inputs. Keep `requirements.txt` and the Python files for the backend;
do not rename the persistent server to imitate a serverless entry point.

To check the static build locally, run `npm run vercel-build`. No npm libraries
are required. `SIGNSYNC_API_BASE` must be an HTTPS origin without an `/api` suffix.
This URL is public browser configuration, so never put credentials in it.

## 3. Create and connect the free backend VM

1. In Oracle Cloud, create an Always Free Ampere A1 VM using no more than 2 OCPUs
   and 12 GB RAM after the trial period. Select an Ubuntu image and assign a
   public IPv4 address.
2. In the VM's network security list or NSG, allow inbound TCP ports 22, 80, and
   443. On Ubuntu, allow ports 80 and 443 in the host firewall too.
3. Install Docker Engine with the Docker Compose plugin, then clone this GitHub
   repository onto the VM.
4. Create a DNS name and point its A record to the VM's public IPv4 address.
   The DNS name must resolve before Caddy can obtain its free HTTPS certificate.
5. In the repository folder on the VM, copy `.env.example` to `.env`. Set
   `API_DOMAIN` to that DNS name and set `SIGNSYNC_ALLOWED_ORIGINS` to the exact
   Vercel website origin from step 2, with no trailing slash.
6. Start the API and HTTPS proxy with `docker compose up -d --build`. Check
   `docker compose logs -f` while the Python packages and models initialize.
   Visit `https://YOUR_API_DOMAIN/api/status`; it should return JSON with
   `"ready": true` after initialization.
7. In Vercel Project Settings → Environment Variables, add
   `SIGNSYNC_API_BASE` with the API's HTTPS origin, such as
   `https://signsync-api.example.com`. Apply it to Production and Preview if
   you will use preview deployments, and redeploy.
8. Open the Vercel website. Allow camera access on the HTTPS page. Confirm the
   status reports the models ready, then try live recognition, one static image,
   and one dynamic video.

## Runtime settings and behavior

- Vercel Hobby is free for personal, non-commercial use. If SignSync is used
  commercially, Vercel requires an eligible paid plan.
- `SIGNSYNC_API_BASE`: Vercel build-time environment variable; HTTPS origin of
  the backend, without a trailing slash or `/api` suffix.
- `SIGNSYNC_ALLOWED_ORIGINS`: comma-separated, exact HTTPS website origins the
  API should accept. Add production and any preview origins you use.
- `PORT`: container listener, default `8001`; Docker Compose sets it.
- `HOST`: interface to bind; Docker Compose sets `0.0.0.0` inside the container.
- Caddy terminates HTTPS on ports 80/443 and forwards requests to the Python API.
- Camera access requires HTTPS (Vercel provides this). Camera frames and uploaded
  test files are sent to the hosted API for inference; uploaded video is processed
  in a temporary file and removed after the request.
- The API currently keeps one shared predictor/sequence buffer in one process.
  It rejects overlapping inference with HTTP 429, as the local service does.
  This deployment is suitable for a personal/demo workload; multi-user traffic
  would need per-client sequence state and concurrency controls.
- OCI free capacity may not be available in every region and Oracle can reclaim
  accounts/resources under its current Free Tier terms. Its free allowance is
  not an uptime guarantee.
- The container has not been validated on an Oracle ARM VM. Confirm Linux ARM
  dependency installation and model loading with `/api/status` before launch;
  the checkpoints and recognition algorithms remain the existing ones.
- Check `THIRD_PARTY_NOTICES.md` and confirm rights for the migrated model
  checkpoints and bundled MediaPipe task model before making the service public.

## Vercel build files

`scripts/build-vercel.mjs` copies only browser files and writes the configured
API origin into `vercel-dist/assets/deployment-config.js`. The output folder is
generated during deployment and is not committed. Python source and PyTorch
checkpoints remain on the backend VM.

References: [Vercel function limits](https://vercel.com/docs/functions/limitations),
[explicit Vercel builders](https://vercel.com/docs/project-configuration/vercel-json#builds),
[Oracle Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm).
