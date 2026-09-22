# Handoff — resume Objectives 1 & 2 once the MLflow server is back

The CSIM MLflow server `mlflow.ml.brain.cs.ait.ac.th` is down (TA notice), so **Task 3 Objective 1**
was logged **locally** and **Objective 2 (registry) is deferred**. Everything else is finished. This
document is the exact to-do list to finish those two once the server is restored.

---

## Status snapshot

| Item | State |
|---|---|
| Task 1 (buckets, from-scratch LR, all metrics, sklearn compare, *support*) | ✅ Done |
| Task 2 (Ridge L2 penalty) | ✅ Done |
| Task 3 · Obj 1 — log experiment `st127004-a3` | ⚠️ Logged **locally** (sqlite `mlflow.db`) — needs re-run against the server |
| Task 3 · Obj 2 — register `st127004-a3-model` at *Staging* | ⛔ Pending — **needs the server** |
| Task 3 · Obj 3 — CI/CD | ✅ Done; app live at https://web-st127004-a3.ml.brain.cs.ait.ac.th |

---

## PENDING 1 — Objective 1: log the experiment to the server

The only functional change is the tracking URI. The experiment name is already `st127004-a3`, and we
already **don't** log the dataset (only params + metrics).

1. In **`run_a3_experiment.py`**, change the tracking URI constant:
   ```python
   # from:
   TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
   # to:
   TRACKING_URI = "http://mlflow.ml.brain.cs.ait.ac.th/"
   ```
   (Same one-line change in the notebook's Objective-1 cell if re-running there.)

2. Re-run the full sweep so all runs land on the server:
   ```bash
   python run_a3_experiment.py
   ```

3. Confirm in the server UI: experiment **`st127004-a3`** shows the runs with params + metrics.

> ✅ Objective 1's "Don't log the dataset" is already satisfied — `run_mlflow_experiment` only calls
> `log_params` / `log_metrics`, never logs `Cars.csv`.

---

## PENDING 2 — Objective 1 "Save the model" + Objective 2 registration

**Gap to close:** the runs currently log params/metrics only. To *save* and *register* a model, the
best run must also log a **model artifact**. Our model is a custom bundle (sklearn preprocessor +
from-scratch `LogisticRegression`), so wrap it as an MLflow **pyfunc** model.

### Step A — log the model in the best run (add to `run_a3_experiment.py` after `fit_final_model`)

```python
import mlflow, mlflow.pyfunc

class BundleModel(mlflow.pyfunc.PythonModel):
    """Serve the A3 bundle (preprocessor + from-scratch classifier) via MLflow."""
    def load_context(self, context):
        import joblib
        self.bundle = joblib.load(context.artifacts["bundle"])
    def predict(self, context, model_input):
        from a3_experiment import predict_class   # returns class 0..3
        return predict_class(self.bundle, model_input)

mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st127004-a3")
with mlflow.start_run(run_name="best-final-model") as run:
    mlflow.log_params(best_config)
    mlflow.log_metrics({k: float(v) for k, v in test_metrics.items()})
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=BundleModel(),
        artifacts={"bundle": str(bundle_path)},          # model/a3_car_price_classifier.pkl
        code_paths=["logistic_regression.py", "a3_experiment.py"],
    )
    BEST_RUN_ID = run.info.run_id
    print("model logged under run:", BEST_RUN_ID)
```

### Step B — register it and move to Staging (Objective 2)

Either the **UI Workflow** (Models → Register model → name `st127004-a3-model` → Stage = Staging), or
programmatically:

```python
from mlflow.tracking import MlflowClient

model_uri = f"runs:/{BEST_RUN_ID}/model"
version = mlflow.register_model(model_uri, "st127004-a3-model")

client = MlflowClient()
client.transition_model_version_stage(
    name="st127004-a3-model", version=version.version, stage="Staging"
)
print(f"Registered st127004-a3-model v{version.version} -> Staging")
```

### Step C — verify
- Models page shows **`st127004-a3-model`** with a version in **Staging**.
- Capture screenshots (e.g. `artifacts/mlflow_runs.png`, `artifacts/mlflow_model_staging.png`) as in A2.

---

## After finishing 1 & 2 — docs to flip

- `README.md` → change the two ⏳/⚠️ statuses (Obj 1 "local", Obj 2 "deferred") to ✅ and add the
  server experiment/model links + screenshots.
- Remove the "MLflow server status" warning box (or note it was resolved on <date>).
- The notebook's Objective-1/2 markdown already describes the switch — update its wording to past
  tense once done.

---

## Non-blocking optional (independent of the server)

- **Self-hosted runner** so the GitHub deploy step goes green from campus — guide in
  `docs/cicd_setup.md` ("Making the deploy step green from GitHub").
- **Rotate the Docker Hub token** (one was pasted in chat during setup) and update the
  `DOCKERHUB_TOKEN` secret: `gh secret set DOCKERHUB_TOKEN --body "<new-read-write-token>"`.

---

## Quick reference

| Thing | Value |
|---|---|
| Experiment name | `st127004-a3` |
| Registered model name | `st127004-a3-model` (target stage: Staging) |
| Server tracking URI | `http://mlflow.ml.brain.cs.ait.ac.th/` |
| Local store (current) | `sqlite:///mlflow.db` |
| Model bundle | `model/a3_car_price_classifier.pkl` |
| Reproduce experiment | `python run_a3_experiment.py` |
| Live app | https://web-st127004-a3.ml.brain.cs.ait.ac.th |
