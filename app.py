"""Serve the local Fake News Detection demo and its prediction endpoint."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

import joblib


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MODEL_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"
HOST = "127.0.0.1"
PORT = 8501
MAX_TEXT_LENGTH = 40_000


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train_baseline.py` first."
        )
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The selected model must support predict_proba().")
    return model


def build_handler(model):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PaperSignals/1.0"

        def send_bytes(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, payload):
            self.send_bytes(
                status,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = (WEB_DIR / "index.html").read_bytes()
                return self.send_bytes(200, page, "text/html; charset=utf-8")
            if self.path == "/styles.css":
                css = (WEB_DIR / "styles.css").read_bytes()
                return self.send_bytes(200, css, "text/css; charset=utf-8")
            if self.path == "/app.js":
                js = (WEB_DIR / "app.js").read_bytes()
                return self.send_bytes(200, js, "text/javascript; charset=utf-8")
            if self.path == "/health":
                return self.send_json(200, {"status": "ready", "model": "tfidf_logistic_regression"})
            return self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/predict":
                return self.send_json(404, {"error": "Not found"})
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                return self.send_json(415, {"error": "Send the article as JSON."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_json(400, {"error": "Invalid request length."})
            if length <= 0 or length > MAX_TEXT_LENGTH:
                return self.send_json(413, {"error": "Text must be between 1 and 40,000 bytes."})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self.send_json(400, {"error": "The request body must be valid JSON."})
            text = body.get("text", "") if isinstance(body, dict) else ""
            if not isinstance(text, str) or not text.strip():
                return self.send_json(400, {"error": "Paste a headline or article first."})
            word_count = len(text.split())
            if word_count < 5:
                return self.send_json(400, {"error": "Add at least five words so the model has some context."})

            try:
                classes = [str(label).lower() for label in model.classes_]
                probabilities = model.predict_proba([text])[0]
                predicted = str(model.predict([text])[0]).lower()
                estimated_probability = float(probabilities[classes.index(predicted)])
            except Exception as exc:  # keep implementation failures out of the server trace
                print(f"Prediction error: {exc}", file=sys.stderr)
                return self.send_json(500, {"error": "The model could not process that text."})

            return self.send_json(
                200,
                {
                    "label": predicted,
                    "confidence": estimated_probability,
                    "probabilities": {
                        label: float(probability)
                        for label, probability in zip(classes, probabilities)
                    },
                    "word_count": word_count,
                },
            )

        def log_message(self, format_string, *args):
            # The input text is never printed to the local terminal.
            print(f"{self.address_string()} - {format_string % args}")

    return Handler


def main():
    model = load_model()
    server = ThreadingHTTPServer((HOST, PORT), build_handler(model))
    print(f"Paper Signals is ready at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the local demo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Paper Signals…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
