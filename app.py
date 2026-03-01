import os
import re
import time
from typing import List, Dict, Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

WP_BASE_URL = os.getenv("WP_BASE_URL", "").rstrip("/")  # np. https://twojadomena.pl
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Jak dużo wpisów pobieramy i ile wyników zwracamy
WP_FETCH_PER_PAGE = int(os.getenv("WP_FETCH_PER_PAGE", "100"))
MAX_POSTS_TO_INDEX = int(os.getenv("MAX_POSTS_TO_INDEX", "500"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Cache w pamięci (na start wystarczy)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # 6h

app = FastAPI(title="Warkan Blog Assistant API")

# CORS: pozwól WordPressowi pytać Render (ustaw domenę!)
allowed_origins = [os.getenv("WP_ORIGIN", "*")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class SearchResult(BaseModel):
    title: str
    url: str
    excerpt: str
    score: float


# ---- Prosty magazyn w RAM ----
_state: Dict[str, Any] = {
    "last_index_time": 0,
    "posts": [],
    "vectorizer": None,
    "tfidf_matrix": None,
}


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_wp_posts() -> List[Dict[str, Any]]:
    """
    Pobiera wpisy z WP REST API.
    Wymaga: WP_BASE_URL
    """
    if not WP_BASE_URL:
        raise RuntimeError("Brak WP_BASE_URL w zmiennych środowiskowych.")

    posts: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = f"{WP_BASE_URL}/wp-json/wp/v2/posts"
        params = {
            "per_page": WP_FETCH_PER_PAGE,
            "page": page,
            "_fields": "id,link,title,excerpt,content,modified"
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 400:
            # zazwyczaj oznacza brak kolejnej strony
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        for p in batch:
            posts.append({
                "id": p.get("id"),
                "url": p.get("link", ""),
                "title": strip_html(p.get("title", {}).get("rendered", "")),
                "excerpt": strip_html(p.get("excerpt", {}).get("rendered", "")),
                "content": strip_html(p.get("content", {}).get("rendered", "")),
                "modified": p.get("modified", ""),
            })

        if len(posts) >= MAX_POSTS_TO_INDEX:
            break

        page += 1

    return posts


def build_index(posts: List[Dict[str, Any]]):
    """
    TF-IDF na bazie (title + excerpt + content).
    """
    documents = []
    for p in posts:
        doc = f"{p['title']} {p['excerpt']} {p['content']}"
        documents.append(doc)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=40000,
        ngram_range=(1, 2),
        stop_words=None  # dla PL możesz kiedyś dodać listę stopwords
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    _state["posts"] = posts
    _state["vectorizer"] = vectorizer
    _state["tfidf_matrix"] = tfidf_matrix
    _state["last_index_time"] = int(time.time())


def ensure_index_fresh():
    """
    Odświeża indeks, jeśli minął TTL albo indeksu nie ma.
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

    # wybierz top_k
    idx_sorted = sims.argsort()[::-1][:top_k]

    results = []
    for idx in idx_sorted:
        score = float(sims[idx])
        if score <= 0:
            continue
        p = posts[idx]
        results.append({
            "title": p["title"],
            "url": p["url"],
            "excerpt": (p["excerpt"][:240] + "…") if len(p["excerpt"]) > 240 else p["excerpt"],
            "score": score
        })
    return results


def gemini_summarize(question: str, results: List[Dict[str, Any]]) -> str:
    """
    Gemini tylko 'upiększa' odpowiedź na podstawie listy wyników.
    """
    if not GEMINI_API_KEY:
        # fallback: bez LLM
        return "Znalazłam pasujące wpisy poniżej. Jeśli chcesz, doprecyzuj pytanie (np. CV w IT, ATS, portfolio)."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}

    # Minimalny, bezpieczny prompt: model ma tylko opisać linki
    bullets = "\n".join(
        [f"- {r['title']} ({r['url']}): {r['excerpt']}" for r in results]
    ) or "Brak wyników."

    prompt = (
        "Jesteś asystentem bloga. Twoim zadaniem jest polecić wpisy z listy.\n"
        "Zasady:\n"
        "1) Nie wymyślaj linków ani tytułów.\n"
        "2) Jeśli brak wyników, poproś o doprecyzowanie.\n"
        "3) Odpowiedz po polsku, krótko (max 6 zdań).\n\n"
        f"Pytanie użytkownika: {question}\n\n"
        f"Lista wyników:\n{bullets}\n\n"
        "Napisz odpowiedź i wskaż 1–3 najbardziej trafne wpisy (tytuł + link)."
    )

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4}
    }

    r = requests.post(url, headers=headers, params={"key": GEMINI_API_KEY}, json=data, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["candidates"][0]["content"]["parts"][0]["text"]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(payload: AskRequest):
    q = payload.question.strip()
    if not q:
        return {"answer": "Napisz proszę pytanie.", "results": []}

    results = search_posts(q, top_k=TOP_K)
    answer = gemini_summarize(q, results)

    return {
        "answer": answer,
        "results": results
    }
