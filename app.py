"""ALI anastomosis scorer — Flask backend.

Serves the single-page UI plus three JSON endpoints: /api/meta (feature and
threshold spec), /api/samples (bundled example images) and /api/analyze
(scores an uploaded image).
"""
import json
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for

from ali_core import analyze, meta, PipelineError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

SAMPLES_DIR = Path(app.static_folder) / "samples"

# matplotlib and the CV pipeline aren't thread-safe, so serialize analysis.
_analysis_lock = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    return jsonify(meta())


@app.get("/api/samples")
def api_samples():
    if not SAMPLES_DIR.exists():
        return jsonify([])
    samples = []
    for path in sorted(SAMPLES_DIR.glob("*.png")):
        samples.append({"name": path.stem, "url": url_for("static", filename=f"samples/{path.name}")})
    return jsonify(samples)


def _parse_json_field(name, default):
    raw = request.form.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


@app.post("/api/analyze")
def api_analyze():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image was uploaded."}), 400

    selected = _parse_json_field("features", [])
    if not selected:
        return jsonify({"error": "Select at least one feature to score."}), 400
    overrides = _parse_json_field("overrides", {})

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "The uploaded file was empty."}), 400

    try:
        with _analysis_lock:
            result = analyze(image_bytes, selected, overrides)
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Unexpected error while scoring: {exc}"}), 500

    result["filename"] = file.filename
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
