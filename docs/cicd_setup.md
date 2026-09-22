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

## Step 2 — (nothing to do) the compose file is written automatically

The deploy step SSHes in and writes `~/a3/docker-compose.yaml` on the server itself, then runs
`docker compose pull && up -d` from `~/a3`. A3 uses its **own** subdomain/container/router
(`web-st127004-a3`) so it runs **alongside** the A2 app without replacing it. No manual `scp` is
needed. (`docs/server-docker-compose.yaml` is the same file, kept for reference / manual deploys.)

## Step 3 — Set the six GitHub Secrets

```bash
gh secret set DOCKERHUB_USERNAME --body "pcismyname"
gh secret set DOCKERHUB_TOKEN    --body "<docker-hub-token-with-READ-WRITE-scope>"
gh secret set SSH_HOST           --body "ml-brain.cs.ait.ac.th"
gh secret set SSH_USER           --body "st127004"
gh secret set SSH_PASSPHRASE     --body "<your-ssh-key-passphrase>"

# The private key is read straight from the file (no copy/paste):
gh secret set SSH_PRIVATE_KEY < ~/.ssh/st127004
```

> The Docker Hub token **must have Read & Write scope** — a read-only token builds but fails to push
> with `unauthorized: access token has insufficient scopes`.

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

When it goes green, the app is live at **https://web-st127004-a3.ml.brain.cs.ait.ac.th**
(allow ~30 s for Traefik to issue the TLS cert on first deploy).

---

## Making the deploy step green from GitHub (self-hosted runner)

The CSIM server only accepts SSH from inside the AIT campus network, so a GitHub-hosted runner
cannot reach it (the SSH step times out). To have the **deploy job run from campus and pass**,
register a self-hosted runner on a campus machine:

1. GitHub repo → **Settings → Actions → Runners → New self-hosted runner** → follow the shown
   `./config.cmd --url ... --token ...` and `./run.cmd` steps.
2. In `.github/workflows/ci-cd.yml`, change the deploy job to run on it:
   ```yaml
   build-and-deploy:
     needs: test
     runs-on: [self-hosted]     # was: ubuntu-latest
   ```
   (Keep `test` on `ubuntu-latest` — only the deploy needs campus network access.)
3. With the runner online, every push to `main` builds, pushes, and deploys automatically.

Until then, the deploy is run once from any campus machine with the two commands in the repo README
(`scp` the compose file, then `ssh ... docker compose pull && up -d`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `test` job fails | Reproduce locally: `pytest tests app/code/tests -q`. |
| `docker login` step fails | `DOCKERHUB_TOKEN` wrong/expired — regenerate and `gh secret set` again. |
| SSH step: `ssh: handshake failed` | Wrong `SSH_PASSPHRASE`, or the public key isn't in the server's `~/.ssh/authorized_keys`. |
| `unauthorized: access token has insufficient scopes` | Docker Hub token is read-only — regenerate with **Read & Write** and re-set `DOCKERHUB_TOKEN`. |
| Deploys but 502 in browser | Container name/labels must match Traefik; check `docker logs web-st127004-a3` on the server. |
| Only want CI, not auto-deploy | Delete the `build-and-deploy` job, or leave its secrets unset (CI still runs). |
