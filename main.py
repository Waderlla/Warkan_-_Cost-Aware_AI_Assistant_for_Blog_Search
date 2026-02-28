import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GEMINI_API_KEY")

@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify(error="Brak wiadomości"), 400

        if not API_KEY:
            return jsonify(error="Brak GEMINI_API_KEY"), 500

        url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

        payload = {
            "contents": [
                {"parts": [{"text": message}]}
            ]
        }

        r = requests.post(
            url,
            params={"key": API_KEY},
            json=payload,
            timeout=30
        )

        if not r.ok:
            return jsonify(error="Gemini API error", status=r.status_code, details=r.text), 502

        j = r.json()
        text = j["candidates"][0]["content"]["parts"][0]["text"]

        return jsonify(reply=text)

    except Exception as e:
        return jsonify(error="Backend exception", details=str(e)), 500

@app.get("/models")
def models():
    if not API_KEY:
        return jsonify(error="Brak GEMINI_API_KEY"), 500

    url = "https://generativelanguage.googleapis.com/v1/models"
    r = requests.get(url, params={"key": API_KEY}, timeout=30)

    if not r.ok:
        return jsonify(error="ListModels error", status=r.status_code, details=r.text), 502

    return jsonify(r.json())
