import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

@app.get("/")
def health():
    return jsonify(ok=True, service="backend alive")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_text = data.get("message", "")

MODEL = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent"    payload = {"contents": [{"parts": [{"text": user_text}]}]}

    r = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=30)
    r.raise_for_status()

    out = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return jsonify(reply=out)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
