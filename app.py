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

load_dotenv()

# __________ ENV __________

WP_BASE_URL = os.getenv("WP_BASE_URL", "")

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")

WP_FETCH_PER_PAGE = int(os.getenv("WP_FETCH_PER_PAGE", "100"))      # ile wpisów pobieramy jednorazowo z bloga
MAX_POSTS_TO_INDEX = int(os.getenv("MAX_POSTS_TO_INDEX", "500"))    # maksymalna liczba wpisów, które zapisujemy w pamięci do przeszukiwania
TOP_K = int(os.getenv("TOP_K", "5"))                                
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.02"))

WP_CATEGORY_ID = int(os.getenv("WP_CATEGORY_ID", "23"))             # kategoria "Kartka z pamiętnika" z ID = 23

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "604800"))   # maksymalny czas ważności indeksu w pamięci (7 dni) - po tym czasie przy kolejnym zapytaniu dane zostaną pobrane ponownie


# __________ APP __________ 

app = FastAPI(title="Warkan")       # tworzy "silnik" backendu - od tego momentu aplikacja może przyjmować zapytania z internetu

app.add_middleware(                 # dodajemy zasadę, która będzie sprawdzana przy KAŻDYM zapytaniu do aplikacji (taki wrapper)
    CORSMiddleware,                 # to mechanizm, który mówi przeglądarce: "tak, możesz rozmawiać z tym backendem"
    allow_origins=[WP_BASE_URL],    # tylko ta konkretna strona internetowa może korzystać z tego backendu
    allow_methods=["GET", "POST"],  # backend zgadza się tylko na pobieranie danych (GET) i wysyłanie pytań (POST)
)


class AskRequest(BaseModel):        # model danych wejściowych – określa, jak ma wyglądać zapytanie do API
    question: str                   # pole wymagane w JSON-ie; użytkownik musi wysłać tekst pod nazwą "question"


# __________ Pamięć aplikacji (tymczasowy schowek w RAM serwera) __________

_state = {
    "last_index_time": 0,   # kiedy ostatnio odświeżyliśmy wpisy z bloga
    "posts": [],            # lista pobranych wpisów (kopiujemy blog do pamięci)
    "vectorizer": None,     # przygotowany mechanizm do porównywania tekstów
    "tfidf_matrix": None,   # "numeryczna mapa" wszystkich wpisów do szybkiego wyszukiwania
}


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

    vectorizer = TfidfVectorizer(               # TfidfVectorizer to narzędzie, które zamienia tekst na liczby. Dzięki temu wyszukiwarka „rozumie”, które teksty są najbardziej pasujące
        max_features=40000,                     # ograniczamy maksymalną liczbę słów, które model zapamięta (ochrona pamięci serwera)
        ngram_range=(1, 2),                     # bierzemy pod uwagę od pojedynczych słów do pary słów
    )

    tfidf_matrix = vectorizer.fit_transform(documents)  # każdy wpis zamieniany jest na wektor liczb

    _state["posts"] = posts                             # zapisujemy oryginalne wpisy w pamięci aplikacji
    _state["vectorizer"] = vectorizer                   # zapisujemy "mechanizm tłumaczący tekst na liczby", żeby później zamienić pytanie użytkownika w ten sam sposób
    _state["tfidf_matrix"] = tfidf_matrix               # zapisujemy gotową macierz liczbową wszystkich wpisów (na niej będziemy liczyć podobieństwo)
    _state["last_index_time"] = int(time.time())        # zapamiętujemy moment stworzenia indeksu, żeby odświeżyć po 7 dniach


def ensure_index_fresh():                               # odświeża indeks, jeśli minął TTL lub indeksu nie ma
    now = int(time.time())
    if _state["tfidf_matrix"] is None or (now - _state["last_index_time"]) > CACHE_TTL_SECONDS:
        posts = fetch_wp_posts()
        build_index(posts)


def extract_context_around_keyword(text, query, window = 400):      # funkcja wycina fragment artykułu wokół słowa, o które pyta użytkownik, jeśli tego słowa nie znajdzie, zwraca po prostu początek tekstu
    if not text:                                                    # jeśli tekst jest pusty zwracamy pusty wynik
        return ""

    text_lower = text.lower()
    query_lower = query.lower()

    pos = text_lower.find(query_lower)                  # szukamy pozycji, w której w tekście pojawia się słowo z zapytania (jeśli nie ma będzie -1)

    if pos == -1:
        return text[:window * 2].strip()                # gdy słowa nie ma pokazujemy początek tekstu jako ogólny podgląd wpisu (pierwsze 800 znaków -> window = 400)

    start = max(pos - window, 0)                        # ustalamy, od którego miejsca zacząć wycinanie fragmentu (400 znaków wcześniej) lub 0 jeśli słowo jest blisko początku
    end = min(pos + len(query) + window, len(text))     # ustalamy, gdzie zakończyć wycinanie fragmentu (400 znaków po słowie)

    snippet = text[start:end].strip()                   # wycinamy wyliczony fragment z oryginalnego tekstu

    
    if start > 0:                                       # dodajemy "..." jeśli ucięliśmy początek lub koniec
        snippet = "..." + snippet


    if end < len(text):
        snippet = snippet + "..."

    return snippet                                      # zwracamy gotowy fragment tekstu, który będzie pokazany jako kontekst dopasowania
    

def search_posts(question, top_k = TOP_K):
    ensure_index_fresh()                                            # upewniamy się, że mamy aktualny indeks bloga

    vectorizer: TfidfVectorizer = _state["vectorizer"]              # pobieramy z pamięci vectorizer,tfidf_matrix, listę wpisów blogowych
    tfidf_matrix = _state["tfidf_matrix"]
    posts = _state["posts"]

    q_vec = vectorizer.transform([question])                        # zamieniamy pytanie użytkownika na formę liczbową, żeby komputer mógł je porównać z wpisami

    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()         # liczymy, jak bardzo pytanie jest podobne do każdego wpisu

    idx_sorted = sims.argsort()[::-1][:top_k]                       # sortujemy wpisy od najbardziej podobnych do najmniej i bierzemy top

    best = float(sims[idx_sorted[0]]) if len(idx_sorted) else 0.0   # sprawdzamy, jaki jest najlepszy wynik dopasowania

    if best < MIN_SIMILARITY:               # jeśli nawet najlepszy wynik jest zbyt słaby, uznajemy, że nie znaleziono sensownych wpisów
        return []


    results = []                            # tutaj będziemy zbierać gotowe wyniki do pokazania użytkownikowi

    for idx in idx_sorted:                  # pomijamy wpisy, które są zbyt słabo dopasowane
        score = float(sims[idx])
        if score < MIN_SIMILARITY:
            continue

        p = posts[idx]                      # pobieramy konkretny wpis z listy

        excerpt = p["excerpt"] or ""        # bierzemy krótki opis wpisu

        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "…"   # jeśli opis jest długi, skracamy go

        context_snippet = extract_context_around_keyword(           # wycinamy fragment artykułu wokół słowa z pytania, żeby pokazać kontekst dopasowania.
            p["content"],
            question,
            window=400
        )
        

        results.append({                    # dodajemy gotowy wpis do listy wyników
            "title": p["title"],
            "url": p["url"],
            "excerpt": excerpt,
            "snippet": context_snippet,
            "score": score,
        })

    return results                          # zwracamy listę najlepiej dopasowanych wpisów

def workers_ai_summarize(question, results):
    # 1) jeśli wyszukiwarka nie znalazła żadnych wpisów, nie pytamy AI
    if not results:
        return "Nie znalazłem wpisów pasujących do tego tematu. Spróbuj użyć innych słów lub doprecyzuj pytanie."

    # 2) jeśli nie mamy dostępu do AI, nadal zwracamy same wyniki wyszukiwania.
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return "Znalazłem pasujące wpisy poniżej."

    # 3) Budujemy czytelną, numerowaną listę wyników dla AI (max 3), żeby model wiedział ile ma linków i nie dopisywał kolejnych
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
        "Dostajesz listę wyników wyszukiwania z bloga. To są jedyne wpisy, na które możesz się powołać.\n"
        "Nie dodawaj żadnych innych wpisów, tytułów ani linków.\n\n"
        f"Masz dokładnie {n} wynik(ów). Opisz dokładnie {n} wpis(ów), ani mniej, ani więcej.\n"
        "Odpowiedź ma brzmieć naturalnie, krótko i konkretnie.\n"
        "Dla każdego wpisu podaj do max 3 zdań opisu oparte wyłącznie na tekscie z listy.\n"
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
            return "Znalazłem pasujące wpisy poniżej."

        r.raise_for_status()
        data = r.json()

        result = data.get("result", {})
        return result.get("response", "").strip() or "Znalazłem pasujące wpisy poniżej."

    except Exception:
        return "Znalazłem pasujące wpisy poniżej."


# __________ Routes __________

@app.api_route("/health", methods=["GET", "HEAD"])      # endpoint kontrolny, służy do sprawdzenia, czy backend działa + aktywacja uptimerobot
def health():
    return {"ok": True}


@app.post("/ask")                   # główny endpoint czatu, przyjmuje zapytanie typu POST z JSON-em zawierającym pole "question"
def ask(payload: AskRequest):

    q = (payload.question or "").strip()        # pobieramy pytanie użytkownika

    if not q:                                   # jeśli po oczyszczeniu pytanie jest puste, zwracamy komunikat zamiast próbować wyszukiwać
        return {"answer": "Napisz proszę pytanie.", "results": []}

    results = search_posts(q, top_k=TOP_K)      # szukamy w blogu wpisów najbardziej pasujących do pytania, zwracana jest lista najlepszych dopasowań.

    answer = workers_ai_summarize(q, results)   # na podstawie znalezionych wpisów generujemy odpowiedź

    return {"answer": answer, "results": results}
    # Zwracamy odpowiedź w formacie JSON:
    # - "answer" → tekstowa odpowiedź dla użytkownika
    # - "results" → lista dopasowanych wpisów (tytuł, link, fragment itd.)
