"""Flask web interface for EmbeddedFinder."""

import os

from flask import Flask, jsonify, render_template, request

from embedded_finder.config import get_api_key
from embedded_finder.embedder import Embedder
from embedded_finder.ranker import rank_results
from embedded_finder.search import SearchEngine
from embedded_finder.store import VectorStore


def create_app(embedder=None, store=None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    def _get_engine():
        nonlocal embedder, store
        if embedder is None:
            api_key = get_api_key()
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set")
            embedder = Embedder(api_key=api_key)
        if store is None:
            store = VectorStore()
        return SearchEngine(embedder=embedder, store=store)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/search", methods=["POST"])
    def api_search():
        data = request.get_json(silent=True) or {}
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        n_results = data.get("n_results", 10)
        min_score = data.get("min_score", 0.0)

        try:
            engine = _get_engine()
            results = engine.search(query, n_results=n_results, min_score=min_score)
            results = rank_results(results, query)

            return jsonify({
                "query": query,
                "results": [
                    {
                        "file_path": r.file_path,
                        "file_name": r.file_name,
                        "score": r.score,
                        "snippet": r.snippet[:300],
                        "file_extension": r.file_extension,
                        "file_size": r.file_size,
                    }
                    for r in results
                ],
                "count": len(results),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/status", methods=["GET"])
    def api_status():
        try:
            s = store if store is not None else VectorStore()
            count = s.count()
            indexed = s.get_indexed_files()
            return jsonify({
                "documents": count,
                "unique_files": len(set(indexed.keys())),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
