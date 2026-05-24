# Contributing to PredictaCore

Thank you for taking the time to contribute!

## Getting started

```bash
git clone https://github.com/<your-username>/predictacore.git
cd predictacore
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Development workflow

1. Fork the repo and create a feature branch:
   `git checkout -b feature/your-feature-name`
2. Make your changes. Keep commits small and focused.
3. Format your code:
   ```bash
   black .
   isort .
   ```
4. Lint:
   ```bash
   flake8 .
   ```
5. Run the tests:
   ```bash
   pytest tests/ --cov=src
   ```
6. Open a Pull Request with a clear description of what changed and why.

## Where to contribute

| Area | Module |
|---|---|
| Feature engineering improvements | `src/preprocessing.py` |
| Better anomaly detection | `src/anomaly_detector.py` |
| Alternative classifiers (XGBoost, RF) | `src/predictor.py` |
| Improved risk scoring logic | `src/risk_engine.py` |
| Real dataset support (`ai4i2020.csv`) | `src/preprocessing.py` |
| Frontend / UI improvements | `templates/`, `static/` |
| More unit tests | `tests/` |

## Code style

- Line length: 100 characters (configured in `setup.cfg`)
- Formatting: Black
- Import ordering: isort (Black-compatible profile)
- All public functions should have a docstring

## Reporting bugs

Open a GitHub Issue with:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behaviour
- Any relevant error output
