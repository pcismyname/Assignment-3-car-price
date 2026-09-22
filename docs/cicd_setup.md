# CI/CD setup — step by step (Objective 3)

This is the exact runbook to get the GitHub Actions pipeline live. The pipeline is defined in
`.github/workflows/ci-cd.yml`:

- **CI (`test` job)** — runs on every push / PR to `main`: installs deps and runs
  `pytest tests app/code/tests`.
- **CD (`build-and-deploy` job)** — runs only on a push to `main` **after CI passes**: builds the
  `app/` Docker image, pushes it to Docker Hub, then SSH-deploys it on the CSIM server
  (`docker compose pull && docker compose up -d`).

Run the commands below from the repo root (`Assignment-3-car-price/`) in Git Bash / PowerShell.

---

## Prerequisites (one-time)

| Need | How to get it |
|---|---|
| GitHub CLI logged in | `gh auth status` — already logged in as `pcismyname` |
| Docker Hub **access token** | hub.docker.com → Account Settings → Personal access tokens → *Generate* (Read/Write). This is the value for `DOCKERHUB_TOKEN`. |
| SSH key + passphrase | `~/.ssh/st127004` (already on this machine) and its passphrase |

---

## Step 1 — Create the GitHub repo and push

```bash
gh repo create Assignment-3-car-price --private --source=. --remote=origin --push
```

(`--private` recommended for coursework. Use `--public` if the submission must be publicly viewable.)
This pushes `main` and immediately triggers the **CI** job (tests). CD will not deploy yet — the
secrets aren't set.

## Step 2 — Put the A3 compose file on the server

The deploy step runs `docker compose ...` from the server home dir, so it must reference the **A3**
image. Copy the compose file from `docs/task3_deployment.md` up to the server (overwrites the A2 one):

```bash
scp -i ~/.ssh/st127004 docs/server-docker-compose.yaml \
    st127004@ml-brain.cs.ait.ac.th:~/docker-compose.yaml
```

> `docs/server-docker-compose.yaml` is provided in this repo ready to copy.

## Step 3 — Set the five GitHub Secrets

```bash
gh secret set DOCKERHUB_USERNAME --body "pcismyname"
gh secret set DOCKERHUB_TOKEN    --body "<paste-your-docker-hub-token>"
gh secret set SSH_HOST           --body "ml-brain.cs.ait.ac.th"
gh secret set SSH_USER           --body "st127004"
gh secret set SSH_PASSPHRASE     --body "<your-ssh-key-passphrase>"

# The private key is read straight from the file (no copy/paste):
gh secret set SSH_PRIVATE_KEY < ~/.ssh/st127004
```

Verify:

```bash
gh secret list
```

You should see all six names (no values are ever shown).

## Step 4 — Trigger a deploy and watch it

Any push to `main` now runs the full pipeline. To trigger one and follow it live:

```bash
git commit --allow-empty -m "ci: trigger deploy"
git push
gh run watch
```

When it goes green, the app is live at **https://web-st127004.ml.brain.cs.ait.ac.th**
(allow ~30 s for Traefik to issue the TLS cert on first deploy).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `test` job fails | Reproduce locally: `pytest tests app/code/tests -q`. |
| `docker login` step fails | `DOCKERHUB_TOKEN` wrong/expired — regenerate and `gh secret set` again. |
| SSH step: `ssh: handshake failed` | Wrong `SSH_PASSPHRASE`, or the public key isn't in the server's `~/.ssh/authorized_keys`. |
| Deploys but 502 in browser | Container name/labels must match Traefik; check `docker logs web-st127004` on the server. |
| Only want CI, not auto-deploy | Delete the `build-and-deploy` job, or leave its secrets unset (CI still runs). |
