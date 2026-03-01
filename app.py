import os
import re
import time
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

# =========================
# ENV
# =========================
WP_BASE_URL = os.getenv("WP_BASE_URL", "").rstrip("/")   # np. https://twojadomena.pl
WP_ORIGIN = os.getenv("WP_ORIGIN", "*")                 # np. https://twojadomena.pl

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")

WP_FETCH_PER_PAGE = int(os.getenv("WP_FETCH_PER_PAGE", "100"))
MAX_POSTS_TO_INDEX = int(os.getenv("MAX_POSTS_TO_INDEX", "500"))
TOP_K = int(os.getenv("TOP_K", "5"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # 6h


# =========================
# APP
# =========================
app = FastAPI(title="Warkan Blog Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[WP_ORIGIN] if WP_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


# =========================
# In-memory state (cache)
# =========================
_state: Dict[str, Any] = {
    "last_index_time": 0,
    "posts": [],
    "vectorizer": None,
    "tfidf_matrix": None,
}


# =========================
# Helpers
# =========================
def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_wp_posts() -> List[Dict[str, Any]]:
    """
    Pobiera wpisy z WordPress REST API.
    """
    if not WP_BASE_URL:
        raise RuntimeError("Brak WP_BASE_URL w zmiennych środowiskowych (Render → Environment).")

    posts: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = f"{WP_BASE_URL}/wp-json/wp/v2/posts"
        params = {
            "per_page": WP_FETCH_PER_PAGE,
            "page": page,
            "_fields": "id,link,title,excerpt,content,modified",
        }

        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 400:
            # zwykle: nie ma kolejnej strony
            break
        r.raise_for_status()

        batch = r.json()
        if not batch:
            break

        for p in batch:
            posts.append({
                "id": p.get("id"),
                "url": p.get("link", ""),
                "title": strip_html((p.get("title") or {}).get("rendered", "")),
                "excerpt": strip_html((p.get("excerpt") or {}).get("rendered", "")),
                "content": strip_html((p.get("content") or {}).get("rendered", "")),
                "modified": p.get("modified", ""),
            })

        if len(posts) >= MAX_POSTS_TO_INDEX:
            break

        page += 1

    return posts


def build_index(posts: List[Dict[str, Any]]) -> None:
    """
    Buduje TF-IDF na bazie title+excerpt+content.
    """
    documents = []
    for p in posts:
        doc = f"{p['title']} {p['excerpt']} {p['content']}"
        documents.append(doc)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=40000,
        ngram_range=(1, 2),
        stop_words=None,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    _state["posts"] = posts
    _state["vectorizer"] = vectorizer
    _state["tfidf_matrix"] = tfidf_matrix
    _state["last_index_time"] = int(time.time())


def ensure_index_fresh() -> None:
    """
    Odświeża indeks, jeśli minął TTL lub indeksu nie ma.
    """
    now = int(time.time())
    if _state["tfidf_matrix"] is None or (now - _state["last_index_time"]) > CACHE_TTL_SECONDS:
        posts = fetch_wp_posts()
        build_index(posts)


def search_posts(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    ensure_index_fresh()

    vectorizer: TfidfVectorizer = _state["vectorizer"]
    tfidf_matrix = _state["tfidf_matrix"]
    posts = _state["posts"]

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    idx_sorted = sims.argsort()[::-1][:top_k]

    results = []
    for idx in idx_sorted:
        score = float(sims[idx])
        if score <= 0:
            continue

        p = posts[idx]
        excerpt = p["excerpt"] or ""
        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "…"

        results.append({
            "title": p["title"],
            "url": p["url"],
            "excerpt": excerpt,
            "score": score,
        })

    return results


def workers_ai_summarize(question: str, results: List[Dict[str, Any]]) -> str:
    """
    Cloudflare Workers AI: tylko do krótkiej, ładnej odpowiedzi na bazie listy wyników.
    Jeśli AI nie działa / limit się skończy -> fallback bez AI.
    """
    # Fallback gdy brak tokena / brak konfiguracji
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return (
            "Znalazłam pasujące wpisy poniżej."
            if results else
            "Nie znalazłam nic pewnego. Doprecyzuj pytanie (np. „CV ATS”, „portfolio”, „rekrutacja”)."
        )

    bullets = "\n".join(
        [f"- {r['title']} ({r['url']}): {r['excerpt']}" for r in results[:5]]
    ) or "Brak wyników."

    prompt = (
        "Jesteś asystentem bloga. Polecasz wpisy z listy.\n"
        "Zasady:\n"
        "1) Nie wymyślaj tytułów ani linków.\n"
        "2) Jeśli brak wyników, poproś o doprecyzowanie.\n"
        "3) Odpowiedź krótka (max 6 zdań).\n\n"
        f"Pytanie: {question}\n\n"
        f"Wyniki:\n{bullets}\n\n"
        "Napisz odpowiedź i wskaż 1–3 najtrafniejsze wpisy (tytuł + link)."
    )

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(url, headers=headers, json={"prompt": prompt}, timeout=30)

        # Free-tier / quota – najczęściej 429
        if r.status_code == 429:
            return (
                "Dziś wyczerpał się limit AI, ale poniżej masz najlepiej pasujące wpisy."
                if results else
                "Dziś wyczerpał się limit AI. Spróbuj jutro lub doprecyzuj pytanie."
            )

        r.raise_for_status()
        data = r.json()

        # Typowa odpowiedź: {"success":true, "result":{"response":"..."}}
        if isinstance(data, dict) and data.get("success") is False:
            return "AI nie odpowiedziało. Poniżej masz pasujące wpisy."

        text = ""
        if isinstance(data, dict):
            result = data.get("result", {})
            if isinstance(result, dict):
                text = result.get("response", "") or ""

        return text.strip() or "Znalazłam pasujące wpisy poniżej."

    except Exception:
        return "Coś poszło nie tak po stronie AI. Poniżej masz pasujące wpisy."


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(payload: AskRequest):
    q = (payload.question or "").strip()
    if not q:
        return {"answer": "Napisz proszę pytanie.", "results": []}

    results = search_posts(q, top_k=TOP_K)
    answer = workers_ai_summarize(q, results)

    return {"answer": answer, "results": results}
