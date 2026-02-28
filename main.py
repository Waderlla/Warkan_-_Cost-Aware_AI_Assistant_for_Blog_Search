from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# Pozwól na zapytania tylko z Twojej domeny WP (albo "*" na czas testów)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class ChatIn(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(payload: ChatIn):
    text = (payload.message or "").strip()
    if len(text) < 2:
        return {"error": "Wpisz wiadomość."}
    if not GEMINI_API_KEY:
        return {"error": "Brak GEMINI_API_KEY w Environment na Render."}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )

    body = {
        "contents": [
            {"parts": [{"text": text}]}
        ]
    }

    r = requests.post(url, json=body, timeout=30)
    data = r.json()

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        # jeśli Gemini zwróci błąd, pokaż go czytelnie
        api_msg = data.get("error", {}).get("message")
        return {"error": api_msg or "Nie udało się odczytać odpowiedzi z Gemini."}

    return {"answer": answer}
