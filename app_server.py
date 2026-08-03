"""
NutriScore Production Web Application & ML REST API Server
==========================================================
Flask application server for Step 11 & Final Submission Portfolio deployment.
Serves both the interactive web user interface and structured JSON API endpoints.
"""

import time
import os
from flask import Flask, request, jsonify, send_from_directory
from src.api.inference import NutriScoreInferenceEngine
from src.monitoring.logger import get_logger, telemetry

app = Flask(__name__, static_folder=".", static_url_path="")
logger = get_logger("NutriScoreApp")
engine = NutriScoreInferenceEngine()

START_TIME = time.time()


@app.before_request
def start_timer():
    request._start_time = time.time()


@app.after_request
def log_request_telemetry(response):
    if hasattr(request, "_start_time"):
        latency_ms = (time.time() - request._start_time) * 1000.0
        path = request.path
        if path.startswith("/api/v1/"):
            telemetry.log_request(
                path=path,
                status_code=response.status_code,
                latency_ms=latency_ms
            )
    return response


@app.route("/")
def serve_index():
    """
    Serves the simple user interface for interactive NutriScore evaluation.
    """
    return send_from_directory(".", "index.html")


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """
    Kubernetes / Cloud Run readiness & liveness probe endpoint.
    """
    uptime_sec = round(time.time() - START_TIME, 2)
    return jsonify({
        "status": "healthy",
        "service": "nutriscore-api",
        "model_version": engine.model_name,
        "is_ml_model": engine.is_ml_model,
        "uptime_seconds": uptime_sec
    }), 200


@app.route("/api/v1/metrics", methods=["GET"])
def metrics_endpoint():
    """
    Prometheus / Cloud Telemetry monitoring metrics endpoint.
    """
    summary = telemetry.get_metrics_summary()
    summary["model_loaded"] = engine.model_name
    summary["uptime_seconds"] = round(time.time() - START_TIME, 2)
    return jsonify(summary), 200


@app.route("/api/v1/predict", methods=["POST"])
def predict_endpoint():
    """
    REST endpoint to evaluate nutrition facts and return predicted health score & grade.
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON with Content-Type: application/json"
        }), 400

    payload = request.get_json()
    try:
        prediction_res = engine.predict(payload)
        prediction_res["success"] = True
        
        # Log specific prediction metrics
        telemetry.log_request(
            path="/api/v1/predict",
            status_code=200,
            latency_ms=(time.time() - getattr(request, "_start_time", time.time())) * 1000.0,
            prediction=prediction_res.get("score"),
            grade=prediction_res.get("grade")
        )
        return jsonify(prediction_res), 200

    except Exception as e:
        logger.error(f"Error executing prediction: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/v1/barcode/<barcode>", methods=["GET"])
def barcode_lookup_endpoint(barcode):
    """
    REST endpoint integrating Open Food Facts lookup with real-time ML score prediction.
    """
    try:
        res = engine.predict_from_open_food_facts(barcode)
        status_code = 200 if res.get("success", False) else 404
        
        if res.get("success", False):
            telemetry.log_request(
                path=f"/api/v1/barcode/{barcode}",
                status_code=status_code,
                latency_ms=(time.time() - getattr(request, "_start_time", time.time())) * 1000.0,
                prediction=res.get("score"),
                grade=res.get("grade"),
                barcode=barcode
            )
        return jsonify(res), status_code

    except Exception as e:
        logger.error(f"Error in barcode lookup endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting NutriScore Flask production server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
