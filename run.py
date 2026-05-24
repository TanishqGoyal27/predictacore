#!/usr/bin/env python3
"""
run.py
------
Entry point for PredictaCore.

On first run  → trains all models and saves them to models/
Subsequent    → loads models from disk (~1 second)
Then          → starts the Flask dev server on http://localhost:5000
"""

import os
from app import create_app
from src.model_manager import manager

if __name__ == "__main__":
    print("=" * 56)
    print("  PredictaCore — Predictive Maintenance System")
    print("=" * 56)

    print("\n[STEP 1] Initialising models (training on first run)…")
    manager.load_or_train()

    print("\n[STEP 2] Starting web server → http://localhost:5000")
    print("         Press Ctrl+C to stop.\n")

    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
