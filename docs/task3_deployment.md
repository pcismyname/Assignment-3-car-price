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

## Objective 1 — MLflow logging (local, server down)

The CSIM MLflow server is currently down (TA notice). The experiment is logged locally:

```python
tracking_uri = "sqlite:///mlflow.db"      # temporary
# tracking_uri = "http://mlflow.ml.brain.cs.ait.ac.th/"   # once the server is restored
mlflow.set_experiment("st127004-a3")
```

Reproduce with `python run_a3_experiment.py`; browse with
`mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000`.

## Objective 2 — Model registry (deferred)

Once the server returns, register the best run as `st127004-a3-model` and move it to *Staging* via
the MLflow **Models** UI (or `mlflow.register_model(...)`).

---

## Objective 3 — CI/CD pipeline

`.github/workflows/ci-cd.yml` has two jobs:

1. **`test`** — checkout, set up Python 3.12, `pip install -r requirements.txt`, then
   `pytest tests app/code/tests`. Runs on every push and PR to `main`.
2. **`build-and-deploy`** — `needs: test`, runs only on a push to `main`:
   - `docker/login-action` → Docker Hub
   - `docker/build-push-action` builds `./app` and pushes `<user>/car-price-a3:latest`
   - `appleboy/ssh-action` SSHes to the server and runs `docker compose pull && docker compose up -d`

### GitHub Secrets required

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | `pcismyname` |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `SSH_HOST` | `ml-brain.cs.ait.ac.th` |
| `SSH_USER` | `st127004` |
| `SSH_PRIVATE_KEY` | Ed25519 private key authorised on the server |
| `SSH_PASSPHRASE` | Passphrase protecting that key |

---

## Server-side `docker-compose.yaml`

The CI/CD deploy step **writes this automatically** to `~/a3/docker-compose.yaml` on the server and
runs `docker compose pull && up -d` from `~/a3`. A3 uses its **own** subdomain / container / router
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
