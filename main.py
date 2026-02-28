import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# CORS: pozwól na WordPressa (albo "*" na testy)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Bez zgadywania modeli: ustawimy domyślny, a jak nie działa, pokażemy błąd z Google
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")

@app.get("/")
def health():
    return jsonify(ok=True, service="warkan-backend")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify(error="Brak pola 'message'."), 400

    if not GEMINI_API_KEY:
        return jsonify(error="Brak GEMINI_API_KEY w Render → Environment."), 500

    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"
    payload = {"contents": [{"parts": [{"text": message}]}]}

    try:
        r = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        return jsonify(error="Timeout (Gemini nie odpowiedział na czas)."), 504
    except Exception as e:
        return jsonify(error="Request error", details=str(e)), 500

    if not r.ok:
        return jsonify(error="Gemini API error", status=r.status_code, details=r.text), 502

    j = r.json()
    text = (
        j.get("candidates", [{}])[0]
         .get("content", {})
         .get("parts", [{}])[0]
         .get("text")
    )

    if not text:
        return jsonify(error="Brak tekstu w odpowiedzi Gemini.", raw=j), 502

    return jsonify(reply=text)

# Endpoint do sprawdzenia jakie modele widzi Twój klucz
@app.get("/models")
def models():
    if not GEMINI_API_KEY:
        return jsonify(error="Brak GEMINI_API_KEY"), 500

    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": GEMINI_API_KEY},
            timeout=12
        )
    except requests.exceptions.Timeout:
        return jsonify(error="Timeout na ListModels."), 504

    if not r.ok:
        return jsonify(error="ListModels error", status=r.status_code, details=r.text), 502

    j = r.json()
    slim = []
    for m in j.get("models", []):
        slim.append({
            "name": m.get("name"),
            "methods": m.get("supportedGenerationMethods", [])
        })
    return jsonify(models=slim)
