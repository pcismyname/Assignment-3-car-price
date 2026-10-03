# CI/CD setup — step by step (Objective 3)

This is the exact runbook to get the GitHub Actions pipeline live. The pipeline is defined in
`.github/workflows/ci-cd.yml`:

- **CI (`test` job)** — runs on every push / PR to `main`: installs deps and runs
  `pytest tests app/code/tests`.
- **CD (`build-and-deploy` job)** — runs only on a push to `main` **after CI passes**: builds the
  `app/` Docker image and pushes it to Docker Hub. The server's `updater` container pulls the new
  image within 5 minutes (pull-based deploy — GitHub runners can't SSH into ml-brain).

Run the commands below from the repo root (`Assignment-3-car-price/`) in Git Bash / PowerShell.

---

## Prerequisites (one-time)

| Need | How to get it |
|---|---|
| GitHub CLI logged in | `gh auth status` — already logged in as `pcismyname` |
| Docker Hub **access token** | hub.docker.com → Account Settings → Personal access tokens → *Generate* (Read/Write). This is the value for `DOCKERHUB_TOKEN`. |
| SSH key | `~/.ssh/st127004` — only for the one-time Step 2, from the AIT network |

---

## Step 1 — Create the GitHub repo and push

```bash
gh repo create Assignment-3-car-price --private --source=. --remote=origin --push
```

(`--private` recommended for coursework. Use `--public` if the submission must be publicly viewable.)
This pushes `main` and immediately triggers the **CI** job (tests). CD will not deploy yet — the
secrets aren't set.

## Step 2 — Put the compose file (with the updater) on the server, once

From a machine on the AIT network:

```bash
scp -i ~/.ssh/st127004 docs/server-docker-compose.yaml st127004@ml-brain.cs.ait.ac.th:~/a3/docker-compose.yaml
ssh -i ~/.ssh/st127004 st127004@ml-brain.cs.ait.ac.th "cd ~/a3 && docker compose up -d"
```

This starts `web-st127004-a3` (Traefik-labelled, no host port) and `web-st127004-a3-updater`, which
re-pulls `latest` every 5 minutes. After this, deployment is automatic.

## Step 3 — Set the two GitHub Secrets

```bash
gh secret set DOCKERHUB_USERNAME --body "pcismyname"
gh secret set DOCKERHUB_TOKEN    --body "<docker-hub-token-with-READ-WRITE-scope>"
```

(The old `SSH_*` secrets are no longer used — the server pulls the image itself.)

> The Docker Hub token **must have Read & Write scope** — a read-only token builds but fails to push
> with `unauthorized: access token has insufficient scopes`.

Verify:

```bash
gh secret list
```

You should see both Docker Hub names (no values are ever shown).

## Step 4 — Trigger a deploy and watch it

Any push to `main` now runs the full pipeline. To trigger one and follow it live:

```bash
git commit --allow-empty -m "ci: trigger deploy"
git push
gh run watch
```

When it goes green, the app is live at **https://web-st127004-a3.ml.brain.cs.ait.ac.th**
(allow ~30 s for Traefik to issue the TLS cert on first deploy).

---

## Why not SSH from GitHub?

The CSIM server only accepts SSH from inside the AIT network, so a GitHub-hosted runner's SSH step
times out (`dial tcp 192.41.170.105:22: i/o timeout`). The pipeline therefore stops after pushing the
image, and the server's updater does the deploy. Check it with
`docker logs web-st127004-a3-updater` on the server.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `test` job fails | Reproduce locally: `pytest tests app/code/tests -q`. |
| `docker login` step fails | `DOCKERHUB_TOKEN` wrong/expired — regenerate and `gh secret set` again. |
| New image not live after 5 min | `docker logs web-st127004-a3-updater` on the server; restart with `cd ~/a3 && docker compose up -d`. |
| `unauthorized: access token has insufficient scopes` | Docker Hub token is read-only — regenerate with **Read & Write** and re-set `DOCKERHUB_TOKEN`. |
| Deploys but 502 in browser | Container name/labels must match Traefik; check `docker logs web-st127004-a3` on the server. |
| Only want CI, not auto-deploy | Delete the `build-and-deploy` job, or leave its secrets unset (CI still runs). |
