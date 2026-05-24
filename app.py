"""
app.py
------
Flask application factory.

Keeps the app object creation separate from the entry point (run.py)
so tests can import `create_app()` without starting the server.
"""

from flask import Flask, render_template
from flask_cors import CORS

from src.api_routes import api_bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Register the API blueprint (all routes prefixed with /api)
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
