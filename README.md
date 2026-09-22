# Assignment 3 — Predicting Car Price (Classification)

**Course:** AT82.03 Machine Learning
**Student:** Chidsanuphong Pengchai — `st127004`

Assignment 3 reuses the A1/A2 car-price dataset but treats it as a **4-class classification**
problem. It implements **multinomial logistic regression from scratch** (based on the class
notebook *02 - Multinomial Logistic Regression*), hand-codes every classification metric, adds an
optional **Ridge (L2) penalty**, runs an **MLflow** experiment, and ships a Dockerised **Streamlit**
app with a **GitHub Actions CI/CD** pipeline.

---

## ⚠️ MLflow server status (please read)

Per the TA's notice, the CSIM MLflow server `mlflow.ml.brain.cs.ait.ac.th` is **currently down**.
As instructed:

- **Objective 1 (logging):** the experiment is logged **locally** to a sqlite store (`mlflow.db`) in
  the meantime. The experiment name is `st127004-a3`. Switching to the server later is a one-line
  change of `tracking_uri` (see `a3_experiment.run_mlflow_experiment` and the notebook).
- **Objective 2 (model registry):** **deferred** until the server is restored, awaiting further
  instructions. The saved runs are ready to be registered as `st127004-a3-model` at *Staging*.
- **Objective 3 (CI/CD):** does **not** depend on the server and is **complete**.

---

## Task completion

| Task | Status |
|---|---|
| **Task 1** — bucket price into 4 classes (`pd.qcut`); from-scratch `accuracy`, per-class `precision`/`recall`/`f1`, `macro_*`, `weighted_*`; compare to sklearn; explain *support* | ✅ Complete |
| **Task 2** — optional Ridge (L2) penalty on the logistic loss (on/off + `lambda_`) | ✅ Complete |
| **Task 3 · Obj 1** — MLflow experiment logged (locally — server down) as `st127004-a3` | ✅ Complete (local) |
| **Task 3 · Obj 2** — register best model to MLflow *Models* at *Staging* | ⏳ Deferred (server down) |
| **Task 3 · Obj 3** — GitHub Actions CI (unit tests) → CD (build, push, deploy) | ✅ Complete |

---

## Key results

- **Dataset:** 6,808 rows after A1/A2 cleaning, split into 4 **balanced** price quartiles
  (~25% each) with edges (INR): `[30k, 250k, 410k, 640k, 10M]`.
- **Best configuration (full 27-run sweep):** `method=batch`, `use_penalty=False`,
  `learning_rate=0.1`.

| Metric | Value |
|---|---:|
| CV accuracy | 0.7385 |
| CV macro-F1 | 0.7340 |
| **Test accuracy** | **0.7173** |
| Test macro-F1 | 0.7137 |
| Test weighted-F1 | 0.7183 |

The extreme classes (Budget, Luxury) are easiest to separate (F1 ≈ 0.84 / 0.81); the middle bands
overlap more, as expected for price quartiles.

---

## Repository structure

| Path | Purpose |
|---|---|
| `car_price_classification_A3.ipynb` | Executed notebook — Tasks 1–3, all outputs inline |
| `car_price_classification_A3.html` | Rendered notebook for quick viewing / PDF export |
| `logistic_regression.py` | **From-scratch** multinomial logistic regression: softmax, cross-entropy, 3 GD flavours, **Ridge (L2) penalty**, and all metrics |
| `a3_experiment.py` | Cleaning, `pd.qcut` bucketing, preprocessing, stratified CV, MLflow helpers |
| `run_a3_experiment.py` | Reproduces the full 27-run sweep, saves artifacts + the model bundle |
| `Cars.csv` | Course dataset (committed so the notebook and CI are self-contained) |
| `model/a3_car_price_classifier.pkl` | Fitted bundle: preprocessor + classifier + class bin edges |
| `artifacts/` | `cv_results.csv`, `experiment_summary.json`, confusion-matrix & loss figures |
| `mlflow.db` | Local MLflow store — all runs with params + metrics |
| `app/` | Dockerised Streamlit app that predicts the price **category** |
| `tests/` | From-scratch model + metric tests (vs scikit-learn) |
| `app/code/tests/` | The two **required** model unit tests (+ extras) |
| `.github/workflows/ci-cd.yml` | CI/CD pipeline |
| `docs/task3_deployment.md` | Deployment record (Traefik, Docker Hub, SSH) |

---

## Reproduce everything

```powershell
python -m pip install -r requirements.txt

# 1) Full experiment + saved model + artifacts
python run_a3_experiment.py

# 2) Run the notebook
jupyter lab car_price_classification_A3.ipynb

# 3) Inspect MLflow runs (experiment: st127004-a3)
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Run the tests

```powershell
pytest tests app/code/tests -q
```

The two unit tests required by the assignment are in `app/code/tests/test_predict.py`:
`test_model_takes_expected_input` (the model accepts the expected input) and
`test_model_output_has_expected_shape` (`predict_proba` returns shape `(n, 4)`).

## Run the web app locally

```powershell
cd app
docker compose up --build       # -> http://localhost:8501
```

The app collects a few car details and predicts one of four price bands
(**Budget / Mid-range / Premium / Luxury**) with its rupee range and the model's confidence.

---

## CI/CD (Objective 3)

`.github/workflows/ci-cd.yml`:

1. **CI** — on every push / PR to `main`, install deps and run `pytest tests app/code/tests`.
2. **CD** — on a push to `main` that passes CI: build the `app/` Docker image, push it to Docker
   Hub, and SSH-deploy it on the CSIM server (`docker compose pull && docker compose up -d`).

### Required GitHub Secrets

Add these under **Settings → Secrets and variables → Actions** for the deploy job to work:

| Secret | Meaning |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username (e.g. `pcismyname`) |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `SSH_HOST` | `ml-brain.cs.ait.ac.th` |
| `SSH_USER` | `st127004` |
| `SSH_PRIVATE_KEY` | Private key authorised on the server (contents of `~/.ssh/st127004`) |
| `SSH_PASSPHRASE` | Passphrase for that key |

**Step-by-step setup (create repo → set secrets → push → watch):** see
[`docs/cicd_setup.md`](docs/cicd_setup.md).

Until the secrets are set, the `test` job still runs on every push (CI works standalone); the
`build-and-deploy` job simply fails at the login/deploy step, which you can enable once the secrets
are in place. See `docs/task3_deployment.md` for the server-side `docker-compose.yaml`.
