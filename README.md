# PredictaCore

**A hybrid machine-learning system for industrial failure-risk estimation using supervised prediction and unsupervised anomaly detection.**

Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) as part of a study of industrial ML pipelines and the engineering challenges involved in combining supervised and unsupervised approaches.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/TanishqGoyal27/predictacore/actions/workflows/ci.yml/badge.svg)

---

## What this project does

PredictaCore takes six sensor readings from an industrial machine and returns:

- **Failure probability** — from an MLP classifier trained with SMOTE oversampling
- **Anomaly flag** — from an autoencoder trained exclusively on normal-operation samples
- **Fused risk score** — combining both signals (70/30 weighted)
- **Top contributing features** — via leave-one-out sensitivity analysis
- **Maintenance recommendation** — rule-based action derived from the top feature

All of this is surfaced through a Flask API and a real-time web dashboard.

---

## Screenshots

| Dashboard |

> <img width="1901" height="961" alt="image" src="https://github.com/user-attachments/assets/96177f34-6583-4578-9786-dd86553c82cb" />
---

## Architecture

```
predictacore/
├── app.py                     ← Flask application factory
├── run.py                     ← Entry point (trains on first run)
├── render.yaml                ← Render deployment config
├── Procfile                   ← Railway deployment config
│
├── src/
│   ├── preprocessing.py       ← Feature engineering + dataset generation
│   ├── predictor.py           ← MLP classifier (train / predict / save / load)
│   ├── anomaly_detector.py    ← NumPy autoencoder + AnomalyDetector wrapper
│   ├── risk_engine.py         ← Risk scoring, confidence, maintenance actions
│   ├── explainability.py      ← Leave-one-out feature attribution
│   ├── model_manager.py       ← Orchestrates training pipeline + persistence
│   └── api_routes.py          ← Flask Blueprint (/api/*)
│
├── templates/
│   └── index.html             ← Dashboard UI
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── models/                    ← Auto-created on first run (gitignored)
│   ├── predictor.pkl
│   └── anomaly_detector.pkl
│
├── notebooks/
│   └── Honeywell_Predictive_Maintenance.ipynb   ← Research notebook
├── tests/                     ← Unit tests (pytest)
└── docs/                      ← Screenshots and additional documentation
```

### ML pipeline

```
Raw sensor input (6 values)
        │
        ▼
  Feature Engineering          build_features()
  (+ 4 derived features)       preprocessing.py
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  MLP Classifier                          Autoencoder
  (64 → 32 → sigmoid)                   (10 → 32 → 16 → 32 → 10)
  threshold = 0.30                       trained on normal samples only
  predictor.py                           anomaly_detector.py
        │                                          │
        │  failure probability p                   │  reconstruction MSE
        └──────────────────────┬───────────────────┘
                               ▼
                    Risk Score = 0.7p + 0.3 × norm_MSE
                               │
                               ▼
                    risk_engine.py + explainability.py
                    (risk level / action / top features)
```

| Component | Implementation | Notes |
|---|---|---|
| Preprocessing | `StandardScaler` (fitted on SMOTE data) | 4 derived features engineered from 6 raw inputs |
| Class balancing | SMOTE | ai4i2020 has ~3.4% failures — severe imbalance |
| Classifier | `MLPClassifier(64, 32)` | Recall-first threshold (0.30, not 0.50) |
| Anomaly detector | Custom NumPy autoencoder | Trained on normal class only; flags 99th-pct outliers |
| Risk fusion | Weighted average | 70% classifier prob, 30% normalised reconstruction error |
| Attribution | Leave-one-out zeroing | Simple sensitivity, not certified XAI |

---

## Installation

### Requirements

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone
git clone https://github.com/TanishqGoyal27/predictacore.git
cd predictacore

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
python run.py
```

On **first launch**, the system trains all models (~30–60 seconds depending on hardware) and saves them to `models/`. Subsequent launches load from disk in about 1 second.

Open `http://localhost:5000` in your browser.

---

## API reference

All endpoints are prefixed with `/api`.

### `GET /api/status`

Returns current model status.

```json
{"status": "ready", "ready": true}
```

---

### `POST /api/predict`

Run the prediction pipeline on a single machine reading.

**Request body:**

```json
{
  "type":      "M",     // Machine type: "L" (low), "M" (medium), "H" (high)
  "air_temp":  300.5,   // Air temperature in Kelvin   [295.3 – 304.5]
  "proc_temp": 310.5,   // Process temperature in K    [305.7 – 313.8]
  "rpm":       1538,    // Rotational speed in rpm     [1168 – 2886]
  "torque":    40.0,    // Torque in Nm                [3.8 – 76.6]
  "tool_wear": 180      // Tool wear in minutes        [0 – 253]
}
```

**Response:**

```json
{
  "prediction":        "FAILURE",
  "probability":       87.42,
  "risk_level":        "HIGH",
  "confidence":        "HIGH CONFIDENCE",
  "anomaly":           "YES",
  "anomaly_score":     0.089123,
  "anomaly_threshold": 0.052646,
  "final_score":       73.19,
  "root_causes": [
    {"feature": "Wear_rate",    "impact": 0.412},
    {"feature": "Tool wear [min]", "impact": 0.298}
  ],
  "action":        "Replace tool immediately — high wear detected",
  "feature_chart": [{"feature": "Wear_rate", "impact": 100.0}, ...]
}
```

---

### `GET /api/simulate`

Generates 8 random machine readings and returns their predictions. Powers the live monitoring panel in the dashboard.

---

### `POST /api/train`

Triggers a background model retraining. Returns `409` if training is already in progress.

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Deploying to Render

1. Push your code to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com), connecting your repo.
3. Render will auto-detect `render.yaml` and configure the service.
4. Note: on Render's free tier, models are retrained on every cold start because the filesystem is ephemeral. To avoid this, use a paid tier with a persistent disk, or commit pre-trained `.pkl` files.

## Deploying to Railway

```bash
railway login
railway init
railway up
```

`Procfile` is already configured. Set `PORT` as an environment variable in the Railway dashboard if needed.

---

## Honest limitations

This is a student/portfolio ML engineering project. It has real value as a demonstration of combining supervised and unsupervised models in a clean pipeline, but there are important caveats:

- **Synthetic data** — the app trains on a synthetic dataset that approximates the ai4i2020 distribution. It is not trained on real industrial sensor data.
- **Leave-one-out attribution is not rigorous XAI** — zeroing a feature in scaled space is a rough sensitivity proxy. SHAP or LIME would give more reliable attribution.
- **No persistence across Render free-tier deploys** — models are saved to a local `models/` folder, which is ephemeral on free hosting tiers.
- **Autoencoder is NumPy-only** — this was an interesting exercise but PyTorch or scikit-learn's isolation forest would likely be more robust in practice.
- **No authentication or rate limiting** — the API is open. Do not expose this directly to the internet without adding appropriate security.
- **Not benchmarked against the real ai4i2020 dataset** — performance figures (recall, precision) are from the synthetic dataset only.

---

## Potential improvements

These are genuine next steps, not marketing claims:

- Load the real `ai4i2020.csv` dataset and evaluate against its held-out test split
- Replace the NumPy autoencoder with `sklearn.ensemble.IsolationForest` for a fair unsupervised baseline comparison
- Add SHAP values for more reliable feature attribution
- Add a `/api/batch` endpoint for scoring multiple machines at once
- Persist models to S3 or another object store so Render free-tier deploys don't retrain every cold start
- Add model versioning so you can compare performance across retrains

---

## Tech stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3, Flask-CORS |
| ML | scikit-learn (MLP, StandardScaler), imbalanced-learn (SMOTE) |
| Autoencoder | Custom NumPy (no deep-learning framework required) |
| Frontend | Vanilla JS, Chart.js, CSS custom properties |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |
| Deployment | Render / Railway |

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
UCI ML Repository — [link](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

10,000 records, 14 features, 5 failure types (TWF, HDF, PWF, OSF, RNF).
Overall failure rate: ~3.4%.

To use the real dataset, place `ai4i2020.csv` in `data/` and update `model_manager.py` to load it instead of calling `generate_dataset()`.

---

## Research notebook

The full exploratory pipeline — data analysis, SMOTE, Keras autoencoder, evaluation charts — is in [`notebooks/Honeywell_Predictive_Maintenance.ipynb`](notebooks/Honeywell_Predictive_Maintenance.ipynb).

---

## License

MIT — see [LICENSE](LICENSE).
