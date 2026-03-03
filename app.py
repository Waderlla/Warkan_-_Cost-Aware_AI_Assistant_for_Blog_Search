import os
import re
import time
import requests
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ==============================
# Environment Configuration
# ==============================

WP_BASE_URL = os.getenv("WP_BASE_URL", "")

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")

WP_FETCH_PER_PAGE = int(os.getenv("WP_FETCH_PER_PAGE", "100"))
MAX_POSTS_TO_INDEX = int(os.getenv("MAX_POSTS_TO_INDEX", "500"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.02"))

WP_CATEGORY_ID = int(os.getenv("WP_CATEGORY_ID", "23"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "604800"))

# ==============================
# FastAPI Application
# ==============================

app = FastAPI(title="Warkan")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[WP_BASE_URL],
    allow_methods=["GET", "POST"],
)

class AskRequest(BaseModel):
    question: str

# ==============================
# In-Memory Application State
# ==============================

_state = {
    "last_index_time": 0,
    "posts": [],
    "vectorizer": None,
    "tfidf_matrix": None,
}

# ==============================
# Utility Functions
# ==============================

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_wp_posts():
    posts = []
    page = 1

    while True:
        url = f"{WP_BASE_URL}/wp-json/wp/v2/posts"
        params = {
            "per_page": WP_FETCH_PER_PAGE,
            "page": page,
            "categories": WP_CATEGORY_ID,
            "_fields": "link,title,excerpt,content",
        }

        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 400:
            break

        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        for p in batch:
            posts.append({
                "url": p.get("link", ""),
                "title": strip_html((p.get("title") or {}).get("rendered", "")),
                "excerpt": strip_html((p.get("excerpt") or {}).get("rendered", "")),
                "content": strip_html((p.get("content") or {}).get("rendered", "")),
            })

        if len(posts) >= MAX_POSTS_TO_INDEX:
            break

        page += 1

    return posts


def build_index(posts):
    documents = [f"{p['title']} {p['excerpt']} {p['content']}" for p in posts]

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    _state["posts"] = posts
    _state["vectorizer"] = vectorizer
    _state["tfidf_matrix"] = tfidf_matrix
    _state["last_index_time"] = int(time.time())


def ensure_index_fresh():
    now = int(time.time())

    if (
        _state["tfidf_matrix"] is None
        or (now - _state["last_index_time"]) > CACHE_TTL_SECONDS
    ):
        posts = fetch_wp_posts()
        build_index(posts)


def extract_context_around_keyword(text: str, query: str, window: int = 400) -> str:
    if not text:
        return ""

    text_lower = text.lower()
    query_lower = query.lower()

    pos = text_lower.find(query_lower)

    if pos == -1:
        return text[: window * 2].strip()

    start = max(pos - window, 0)
    end = min(pos + len(query) + window, len(text))

    snippet = text[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def search_posts(question: str, top_k: int = TOP_K):
    ensure_index_fresh()

    vectorizer: TfidfVectorizer = _state["vectorizer"]
    tfidf_matrix = _state["tfidf_matrix"]
    posts = _state["posts"]

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    idx_sorted = sims.argsort()[::-1][:top_k]
    best = float(sims[idx_sorted[0]]) if len(idx_sorted) else 0.0

    if best < MIN_SIMILARITY:
        return []

    results = []

    for idx in idx_sorted:
        score = float(sims[idx])
        if score < MIN_SIMILARITY:
            continue

        p = posts[idx]
        excerpt = p["excerpt"] or ""
        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "…"

        context_snippet = extract_context_around_keyword(
            p["content"],
            question,
            window=400,
        )

        results.append({
            "title": p["title"],
            "url": p["url"],
            "excerpt": excerpt,
            "snippet": context_snippet,
            "score": score,
        })

    return results


def workers_ai_summarize(question: str, results):
    if not results:
        return "Nie znalazłem pasujących wpisów. Spróbuj użyć innych słów lub doprecyzować pytanie."

    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return "Znalazłem pasujące wpisy poniżej."

    items = "\n".join(
        [
            f"{i+1}. TYTUŁ: {r['title']}\n"
            f"   LINK: {r['url']}\n"
            f"   TEKST:\n{r.get('snippet','')}"
            for i, r in enumerate(results[:3])
        ]
    )
    n = min(len(results), 3)

    prompt = (
        "Jesteś asystentem mojego bloga.\n"
        "Dostajesz wyniki wyszukiwania z bloga. To są jedyne źródła, z których możesz korzystać.\n"
        "Nie dodawaj żadnych innych tytułów ani linków.\n"
        "Odpowiadaj wyłącznie po polsku.\n\n"
        f"Masz dokładnie {n} wynik(ów). Opisz dokładnie {n} wynik(ów), ani mniej, ani więcej.\n"
        "Maksymalnie 3 zdania na jeden wynik.\n"
        "Odpowiedź ma być naturalna, krótka i konkretna.\n\n"
        f"Pytanie użytkownika: {question}\n\n"
        f"WYNIKI:\n{items}\n"
    )

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        r = requests.post(url, headers=headers, json={"prompt": prompt}, timeout=30)

        if r.status_code == 429:
            return "Znalazłem pasujące wpisy poniżej."

        r.raise_for_status()
        data = r.json()
        result = data.get("result", {})

        return result.get("response", "").strip() or "Znalazłem pasujące wpisy poniżej."

    except Exception:
        return "Znalazłem pasujące wpisy poniżej."


# ==============================
# API Routes
# ==============================

@app.api_route("/health", methods=["GET", "HEAD"])
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
