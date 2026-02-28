import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ====== USTAWIENIA (DARMOWY TRYB) ======
# 1) Ograniczamy długość wejścia, żeby nie spalić limitu
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1200"))

# 2) Model z ENV (na Render ustaw: GEMINI_MODEL=models/gemini-2.0-flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")

# 3) Klucz z ENV (na Render ustaw: GEMINI_API_KEY=...)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 4) CORS – na Render ustaw: ALLOWED_ORIGIN=https://olgamironczuk.pl
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})


# ====== POMOCNICZE ======
def _safe_json(resp: requests.Response):
    """Spróbuj zdekodować JSON, jak nie wyjdzie – zwróć tekst."""
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def _looks_like_quota(error_obj) -> bool:
    """Wykryj typowe komunikaty o limicie/quocie."""
    s = str(error_obj).lower()
    keywords = [
        "quota", "resource_exhausted", "rate limit",
        "exceeded", "too many requests", "limit"
    ]
    return any(k in s for k in keywords)


# ====== ENDPOINTY ======
@app.get("/")
def health():
    # To usuwa "Not Found" na głównym adresie backendu
    return jsonify(ok=True, service="warkan-backend")


@app.post("/chat")
def chat():
    # 1) Walidacja wejścia
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify(error="Wpisz wiadomość (pole 'message')."), 400

    # limit znaków, żeby nie przepalać darmowego limitu
    if len(message) > MAX_INPUT_CHARS:
        message = message[:MAX_INPUT_CHARS]

    # 2) Walidacja ENV
    if not GEMINI_API_KEY:
        return jsonify(error="Brak GEMINI_API_KEY w Render → Environment."), 500

    if not GEMINI_MODEL.startswith("models/"):
        return jsonify(
            error="Zły GEMINI_MODEL. Musi zaczynać się od 'models/'.",
            example="models/gemini-2.0-flash"
        ), 500

    # 3) Zapytanie do Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [
            {"parts": [{"text": message}]}
        ]
    }

    try:
        r = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=30
        )
    except requests.exceptions.Timeout:
        return jsonify(error="Timeout: Gemini nie odpowiedział na czas."), 504
    except Exception as e:
        return jsonify(error="Błąd połączenia z Gemini.", details=str(e)), 502

    # 4) Obsługa błędów Google (żeby nie było 500)
    if not r.ok:
        err = _safe_json(r)

        # limit darmowego API / quota
        if r.status_code in (429, 403) and _looks_like_quota(err):
            return jsonify(
                error="Limit darmowego API został wyczerpany. Spróbuj później.",
                status=r.status_code
            ), 429

        # model not found itp.
        return jsonify(
            error="Gemini API error",
            status=r.status_code,
            details=err
        ), 502

    # 5) Parsowanie odpowiedzi
    j = _safe_json(r)
    text = (
        j.get("candidates", [{}])[0]
         .get("content", {})
         .get("parts", [{}])[0]
         .get("text")
    )

    if not text:
        return jsonify(error="Brak tekstu w odpowiedzi Gemini.", raw=j), 502

    return jsonify(reply=text)


@app.get("/models")
def models():
    # Diagnostyka: lista modeli dostępnych dla Twojego klucza
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
    except Exception as e:
        return jsonify(error="Błąd połączenia (ListModels).", details=str(e)), 502

    if not r.ok:
        return jsonify(error="ListModels error", status=r.status_code, details=_safe_json(r)), 502

    j = _safe_json(r)
    slim = []
    for m in j.get("models", []):
        slim.append({
            "name": m.get("name"),
            "methods": m.get("supportedGenerationMethods", [])
        })

    return jsonify(models=slim)


if __name__ == "__main__":
    # Lokalnie (na Render i tak odpalasz przez gunicorn)
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
