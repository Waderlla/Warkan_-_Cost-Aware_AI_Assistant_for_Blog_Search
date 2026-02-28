from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# WordPress domena, która może wysyłać zapytania (CORS)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class Msg(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(m: Msg):
    text = (m.message or "").strip()
    if len(text) < 2:
        return {"error": "Wpisz wiadomość."}
    if not GEMINI_API_KEY:
        return {"error": "Brak GEMINI_API_KEY na serwerze (env)."}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": text}]}]}

    r = requests.post(url, json=payload, timeout=30)
    data = r.json()

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        answer = "Nie udało się odczytać odpowiedzi."

    return {"answer": answer}
