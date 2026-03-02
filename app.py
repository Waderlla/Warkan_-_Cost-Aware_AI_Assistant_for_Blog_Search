import os                                                       # dostęp do zmiennych środowiskowych (np. klucze API, URL-e)
import re                                                       # wyrażenia regularne – czyszczenie HTML i tekstu
import time                                                     # operacje na czasie (np. TTL cache)
import requests                                                 # wykonywanie zapytań HTTP (WordPress API, Cloudflare AI)
from dotenv import load_dotenv                                  # ładowanie zmiennych z pliku .env

from fastapi import FastAPI                                     # framework do budowy API (endpointy HTTP)
from fastapi.middleware.cors import CORSMiddleware              # obsługa CORS (pozwala przeglądarce łączyć się z backendem)
from pydantic import BaseModel                                  # walidacja danych wejściowych (model requestu)
from sklearn.feature_extraction.text import TfidfVectorizer     # zamiana tekstu na wektory (TF-IDF)
from sklearn.metrics.pairwise import cosine_similarity          # obliczanie podobieństwa między tekstami
from threading import Thread, Lock

load_dotenv()

# __________ ENV __________

WP_BASE_URL = os.getenv("WP_BASE_URL", "")

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")

WP_FETCH_PER_PAGE = int(os.getenv("WP_FETCH_PER_PAGE", "100"))      # ile wpisów pobieramy jednorazowo z bloga
MAX_POSTS_TO_INDEX = int(os.getenv("MAX_POSTS_TO_INDEX", "500"))    # maksymalna liczba wpisów, które zapisujemy w pamięci do przeszukiwania
TOP_K = int(os.getenv("TOP_K", "5"))                                # nie wiem czy potrzebne
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.045"))

WP_CATEGORY_ID = int(os.getenv("WP_CATEGORY_ID", "23"))             # kategoria "Kartka z pamiętnika" z ID = 23

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "604800"))   # maksymalny czas ważności indeksu w pamięci (7 dni) - po tym czasie przy kolejnym zapytaniu dane zostaną pobrane ponownie


# __________ APP __________ 

app = FastAPI(title="Warkan")       # tworzy "silnik" backendu - od tego momentu aplikacja może przyjmować zapytania z internetu

app.add_middleware(                 # dodajemy zasadę, która będzie sprawdzana przy KAŻDYM zapytaniu do aplikacji (taki wrapper)
    CORSMiddleware,                 # to mechanizm, który mówi przeglądarce: "tak, możesz rozmawiać z tym backendem"
    allow_origins=[WP_BASE_URL],    # tylko ta konkretna strona internetowa może korzystać z tego backendu
    allow_methods=["GET", "POST"],  # backend zgadza się tylko na pobieranie danych (GET) i wysyłanie pytań (POST)
    allow_headers=["*"],            # przeglądarka może wysyłać dowolne dodatkowe informacje techniczne
)


class AskRequest(BaseModel):        # model danych wejściowych – określa, jak ma wyglądać zapytanie do API
    question: str                    # pole wymagane w JSON-ie; użytkownik musi wysłać tekst pod nazwą "question"


# __________ Pamięć aplikacji (tymczasowy schowek w RAM serwera) __________

_state = {
    "last_index_time": 0,   # kiedy ostatnio odświeżyliśmy wpisy z bloga
    "posts": [],            # lista pobranych wpisów (kopiujemy blog do pamięci)
    "vectorizer": None,     # przygotowany mechanizm do porównywania tekstów
    "tfidf_matrix": None,   # "numeryczna mapa" wszystkich wpisów do szybkiego wyszukiwania
}

_refreshing = False         # Flaga informująca, czy trwa aktualizacja indeksu w tle.
                            # Ustawiana na True w momencie rozpoczęcia odświeżania,
                            # aby kolejne zapytania nie uruchomiły równoległej aktualizacji.

_refresh_lock = Lock()      # Blokada zabezpieczająca sekcję odpowiedzialną za start odświeżania.
                            # Gwarantuje, że tylko jeden wątek naraz może rozpocząć proces aktualizacji indeksu,
                            # co chroni przed wielokrotnym równoczesnym pobieraniem danych i nadpisywaniem pamięci.


# __________ Funkcje pomocnicze (przetwarzanie danych i budowa indeksu) __________

def strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    # Usuwamy z tekstu wszystkie znaczniki HTML (np. <p>, <strong>, <a>),
    # ponieważ WordPress zwraca treść w formacie HTML,
    # a do wyszukiwania potrzebujemy zwykłego tekstu.
    # Każdy znaleziony znacznik zamieniamy na spację.

    text = re.sub(r"\s+", " ", text).strip()
    # Po usunięciu znaczników może zostać dużo niepotrzebnych spacji
    # albo pustych linii. Zamieniamy więc wiele spacji / enterów
    # na jedną zwykłą spację i usuwamy spacje z początku i końca tekstu.

    return text             # Zwracamy oczyszczony, czytelny tekst gotowy do analizy.


def fetch_wp_posts():       # pobiera wpisy z WordPress REST API (tylko z wybranej kategorii).

    posts = []
    page = 1

    while True:
        url = f"{WP_BASE_URL}/wp-json/wp/v2/posts"
        params = {
            "per_page": WP_FETCH_PER_PAGE,                          # ile wpisów pobieramy jednorazowo z bloga
            "page": page,                                           # numer strony z wynikami
            "categories": WP_CATEGORY_ID,                           # kategoria „Kartka z pamiętnika” z ID = 23.
            "_fields": "link,title,excerpt,content",                # jakie informacje chcemy dostać z każdego wpisu: adres strony, tytuł, skrót i pełny tekst
        }

        r = requests.get(url, params=params, timeout=30)            # prosi blog o wpisy i nie czeka dłużej niż 30 sekund na odpowiedź

        if r.status_code == 400:            # jeśli WordPress zwróci błąd (nie ma już kolejnej strony), przerywamy dalsze pobieranie
            break

        r.raise_for_status()                # sprawdzamy, czy odpowiedź jest poprawna; jeśli nie, przerywamy i zgłaszamy błąd; metoda obiektu zwróconego przez bibliotekę requests

        batch = r.json()                    # zamieniamy odpowiedź z WordPressa na dane, które Python potrafi dalej przetwarzać
        if not batch:                       # nie ma już kolejnej strony wyników, więc przestajemy dalej pytać
            break


        for p in batch:
            posts.append({                  # dla każdego pobranego wpisu zapisujemy tylko potrzebne informacje:
                "url": p.get("link", ""),   # link, tytuł, skrót i pełną treść (po oczyszczeniu z HTML)
                "title": strip_html((p.get("title") or {}).get("rendered", "")),
                "excerpt": strip_html((p.get("excerpt") or {}).get("rendered", "")),
                "content": strip_html((p.get("content") or {}).get("rendered", "")),
            })

        if len(posts) >= MAX_POSTS_TO_INDEX:    # jeśli zebraliśmy już maksymalną liczbę wpisów, przestajemy pobierać kolejne
            break

        page += 1                               # rzechodzimy do następnej "strony" wpisów, żeby pobrać kolejną paczkę artykułów

    print(f"Indexed {len(posts)} posts from category {WP_CATEGORY_ID}")     # wyświetlamy ile wpisów zostało zapisanych
    return posts                                                            # zwracamy listę pobranych wpisów, żeby mogły zostać dalej przetworzone


def build_index(posts):                         # buduje TF-IDF na bazie title+excerpt+content, zamienia wszystkie wpisy blogowe na formę "numeryczną"
    documents = []                              # lista wszystkich tekstów z bloga, każdy wpis stanie się jednym dużym tekstem

    for p in posts:                             # dla każdego wpisu łączymy tytuł, skrót i pełną treść w jeden wspólny tekst
        doc = f"{p['title']} {p['excerpt']} {p['content']}"
        documents.append(doc)                   # dodajemy gotowy tekst do listy "documents"

    vectorizer = TfidfVectorizer(               # TfidfVectorizer to narzędzie, które zamienia tekst na liczby. Dzięki temu wyszukiwarka „rozumie”, które teksty są najbardziej pasujące.

        max_features=40000,                     # ograniczamy maksymalną liczbę słów, które model zapamięta (ochrona pamięci serwera)

        ngram_range=(1, 2),                     # bierzemy pod uwagę od pojedynczych słów do pary słów
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    # Tutaj dzieje się magia:
    # Każdy wpis zamieniany jest na wektor liczb.
    # To taka "numeryczna mapa tekstu",
    # dzięki której komputer może porównywać teksty matematycznie.

    _state["posts"] = posts
    # Zapisujemy oryginalne wpisy w pamięci aplikacji.

    _state["vectorizer"] = vectorizer
    # Zapisujemy "mechanizm tłumaczący tekst na liczby",
    # żeby później móc zamienić pytanie użytkownika w ten sam sposób.

    _state["tfidf_matrix"] = tfidf_matrix
    # Zapisujemy gotową macierz liczbową wszystkich wpisów.
    # To właśnie na niej będziemy liczyć podobieństwo do pytania.

    _state["last_index_time"] = int(time.time())
    # Zapamiętujemy moment stworzenia indeksu,
    # żeby wiedzieć, kiedy trzeba go odświeżyć (np. po 7 dniach).


def ensure_index_fresh() -> None:
    """
    Odświeża indeks, jeśli minął TTL lub indeksu nie ma.
    """
    now = int(time.time())
    if _state["tfidf_matrix"] is None or (now - _state["last_index_time"]) > CACHE_TTL_SECONDS:
        posts = fetch_wp_posts()
        build_index(posts)


def extract_context_around_keyword(text: str, query: str, window: int = 400) -> str:
    """
    Zwraca fragment tekstu wokół pierwszego wystąpienia słowa z zapytania.
    Jeśli nie znajdzie dopasowania, zwraca początek tekstu.
    """

    if not text:
        return ""

    text_lower = text.lower()
    query_lower = query.lower()

    pos = text_lower.find(query_lower)

    # Jeśli nie znaleziono słowa — zwracamy początek tekstu
    if pos == -1:
        return text[:window * 2].strip()

    start = max(pos - window, 0)
    end = min(pos + len(query) + window, len(text))

    snippet = text[start:end].strip()

    # Dodajemy "..." jeśli ucięliśmy początek lub koniec
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet
    

def search_posts(question, top_k = TOP_K):
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
            window=400
        )

        results.append({
            "title": p["title"],
            "url": p["url"],
            "excerpt": excerpt,
            "snippet": context_snippet,
            "score": score,
        })

    return results


def workers_ai_summarize(question: str, results: List[Dict[str, Any]]) -> str:
    # 1) Jeśli wyszukiwarka nie znalazła żadnych wpisów, nie pytamy AI.
    #    Model językowy ma tendencję do "dopowiadania" ogólników, więc robimy twardy fallback.
    if not results:
        return "Nie znalazłam wpisów pasujących do tego tematu. Spróbuj użyć innych słów lub doprecyzuj pytanie."

    # 2) Jeśli nie mamy dostępu do AI (brak tokenów), nadal zwracamy same wyniki wyszukiwania.
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return "Znalazłam pasujące wpisy poniżej."

    # 3) Budujemy czytelną, numerowaną listę wyników dla AI (max 3),
    #    żeby model wiedział ile ma linków i nie dopisywał kolejnych.
    items = "\n".join(
         [
            f"{i+1}. TYTUŁ: {r['title']}\n"
            f"   LINK: {r['url']}\n"
            f"   TEKST:\n{r.get('snippet','')}"
               for i, r in enumerate(results[:3])
         ]
    )
    n = min(len(results), 3)

    # 4) Prompt: naturalnie, ale twardo trzymamy się liczby wyników i zakazu nowych linków.
    prompt = (
        "Jesteś asystentem mojego bloga.\n"
        "Dostajesz listę wyników wyszukiwania z bloga. To są jedyne wpisy, na które możesz się powołać.\n"
        "Nie dodawaj żadnych innych wpisów, tytułów ani linków.\n\n"
        f"Masz dokładnie {n} wynik(ów). Opisz dokładnie {n} wpis(ów), ani mniej, ani więcej.\n"
        "Odpowiedź ma brzmieć naturalnie, krótko i konkretnie.\n"
        "Dla każdego wpisu podaj do max 3 zdań opisu oparte wyłącznie na OPISIE z listy.\n"
        "Nie pisz osobnej sekcji 'Linki:' i nie dodawaj dodatkowych propozycji.\n\n"
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
            return "Limit AI wyczerpany. Poniżej masz pasujące wpisy."

        r.raise_for_status()
        data = r.json()

        result = data.get("result", {})
        return result.get("response", "").strip() or "Znalazłam pasujące wpisy poniżej."

    except Exception:
        return "Błąd po stronie AI. Poniżej masz pasujące wpisy."


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
