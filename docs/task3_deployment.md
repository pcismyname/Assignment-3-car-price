# Task 3 — Deployment & CI/CD record

This documents how the Assignment 3 classifier app is deployed on the shared
`ml-brain.cs.ait.ac.th` server (IP `192.41.170.25`, Ubuntu 24.04) and how the GitHub Actions
pipeline automates it. It mirrors the A2 setup, updated for the new image name `car-price-a3`.

---

## Architecture

The server runs **Traefik** as a reverse proxy on ports 80/443. Student containers join the shared
external Docker network `web`; Traefik reads container labels and routes HTTPS to each container's
internal port. TLS certificates are issued automatically via Let's Encrypt.

```
Browser → HTTPS → Traefik (:443) → web-st127004 container (Streamlit :8501, internal)
```

---

## Objectives 1 & 2 — MLflow (local, per the TA's updated grading)

The course MLflow server is unstable, so the TA made server logging optional and grades Objectives 1
and 2 from screenshots of a **local** MLflow instance. `python run_a3_experiment.py` logs the sweep
to `sqlite:///mlflow.db` (experiment `st127004-a3`, no dataset logged), saves the best model as a
pyfunc in run `best-final-model`, and registers it as `st127004-a3-model` v1 at **Staging**.
Screenshots: `artifacts/mlflow_runs.png`, `mlflow_best_run.png`, `mlflow_model_staging.png`,
`mlflow_model_version.png`. Browse with
`mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000`.

---

## Objective 3 — CI/CD pipeline

`.github/workflows/ci-cd.yml` has two jobs:

1. **`test`** — checkout, set up Python 3.12, `pip install -r requirements.txt`, then
   `pytest tests app/code/tests`. Runs on every push and PR to `main`.
2. **`build-and-deploy`** — `needs: test`, runs only on a push to `main`:
   - `docker/login-action` → Docker Hub
   - `docker/build-push-action` builds `./app` and pushes `<user>/car-price-a3:latest`
   - deployment is **pull-based**: the server's `updater` container pulls `latest` every 5 minutes
     and recreates `web-st127004-a3` if it changed (GitHub runners can't SSH into ml-brain from
     outside the AIT network)

### GitHub Secrets required

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | `pcismyname` |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## Server-side `docker-compose.yaml`

Copied once to `~/a3/docker-compose.yaml` on the server and started with `docker compose up -d`
from `~/a3`; the `updater` service in it keeps the app on the latest image. The full file (with the
updater) is [`docs/server-docker-compose.yaml`](server-docker-compose.yaml); the compose project is
named `st127004-a3` because project names are shared server-wide. A3 uses its **own** subdomain / container / router
(`web-st127004-a3`) so it runs **alongside** the A2 app rather than replacing it.

```yaml
services:
  web-st127004-a3:
    image: pcismyname/car-price-a3:latest
    container_name: web-st127004-a3
    networks:
      - web
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.st127004-a3.rule=Host(`web-st127004-a3.ml.brain.cs.ait.ac.th`)"
      - "traefik.http.routers.st127004-a3.entrypoints=websecure"
      - "traefik.http.routers.st127004-a3.tls=true"
      - "traefik.http.routers.st127004-a3.tls.certresolver=letsencrypt"
      - "traefik.http.services.st127004-a3.loadbalancer.server.port=8501"
    restart: unless-stopped

networks:
  web:
    external: true
```

No host port is published — traffic arrives exclusively through Traefik.

---

## Manual deploy (fallback, as in A2)

```bash
# local machine
docker build -t pcismyname/car-price-a3:latest ./app
docker push pcismyname/car-price-a3:latest

# on the server
ssh -i C:\Users\chids\.ssh\st127004 st127004@ml-brain.cs.ait.ac.th \
  "cd ~/a3 && docker compose pull && docker compose up -d"
```

Live URL (once deployed): **https://web-st127004-a3.ml.brain.cs.ait.ac.th**
