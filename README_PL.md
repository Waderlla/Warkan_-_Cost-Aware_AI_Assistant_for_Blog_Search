# 🐺 Warkan -- Świadomy kosztowo asystent AI dla bloga

Warkan to autorski system typu Retrieval-Augmented Generation (RAG),
zintegrowany z blogiem: 🔗 https://olgamironczuk.pl/warkan/

Projekt został zaprojektowany jako lekki, kontrolowany kosztowo i
świadomie uproszczony architektonicznie asystent AI, którego celem jest
wyszukiwanie i streszczanie treści blogowych bez generowania stałych
kosztów infrastrukturalnych.

------------------------------------------------------------------------

## 🎯 Cel projektu

Celem było stworzenie systemu, który:

-   wyszukuje wpisy z określonej kategorii WordPress,
-   buduje lokalny indeks tekstowy,
-   znajduje najbardziej dopasowane artykuły,
-   generuje krótką odpowiedź wyłącznie na podstawie znalezionych
    wyników,
-   działa bez bazy danych,
-   nie przechowuje historii rozmowy,
-   minimalizuje zużycie modelu językowego,
-   nie generuje stałych kosztów utrzymania.

Projekt powstał jako część budowy portfolio technicznego łączącego
analizę danych, architekturę backendową oraz systemy AI.

------------------------------------------------------------------------

## 🧠 Architektura systemu

User → WordPress UI → FastAPI (Render)\
→ TF-IDF retrieval\
→ snippet extraction\
→ Cloudflare Workers AI\
→ JSON response → UI

System składa się z czterech głównych warstw.

### 1. Warstwa interfejsu

Minimalny interfejs czatu osadzony w WordPress.\
Jego zadaniem jest wyłącznie wysyłanie zapytania do backendu i
wyświetlanie odpowiedzi JSON.\
Projekt skupia się na architekturze backendowej, dlatego warstwa
frontendowa jest celowo uproszczona.

### 2. Backend (Render Web Service)

Backend został napisany w Pythonie z użyciem FastAPI.\
Odpowiada za:

-   pobieranie wpisów z WordPress REST API,
-   czyszczenie HTML,
-   budowę indeksu TF-IDF,
-   wyszukiwanie podobieństwa,
-   ekstrakcję kontekstu,
-   wywołanie modelu językowego,
-   kontrolę kosztów i fallbacki.

Aplikacja działa w modelu stateless.

### 3. Warstwa Retrieval

To kluczowy element systemu.

Zamiast wysyłać wszystkie treści do modelu:

-   wpisy są pobierane z WordPress,
-   czyszczone z HTML,
-   łączone w dokument (tytuł + excerpt + content),
-   indeksowane przy użyciu TF-IDF,
-   porównywane z zapytaniem użytkownika przy użyciu cosine similarity.

Dzięki temu model językowy otrzymuje wyłącznie najbardziej istotne
fragmenty.

### 4. Warstwa Generative AI

Model językowy (Cloudflare Workers AI, Llama) jest wywoływany tylko
wtedy, gdy:

-   wyszukiwarka znalazła istotne dopasowania,
-   similarity przekracza minimalny próg.

Model otrzymuje:

-   maksymalnie 3 wyniki,
-   wycięte fragmenty zamiast całych artykułów,
-   precyzyjny prompt ograniczający zakres odpowiedzi,
-   zakaz dodawania nowych źródeł.

------------------------------------------------------------------------

## 🛠 Tech Stack

### Backend

-   Python
-   FastAPI
-   Uvicorn
-   scikit-learn (TF-IDF + cosine similarity)
-   NumPy (obliczenia macierzowe)
-   Requests (HTTP API)
-   python-dotenv

### Infrastruktura

-   Render (Web Service)
-   Cloudflare Workers AI (LLM)
-   WordPress REST API (źródło danych)
-   UptimeRobot (utrzymywanie aktywności serwisu)
-   GitHub (repozytorium)

------------------------------------------------------------------------

## ⚙️ Jak działa system krok po kroku

### 1️⃣ Pobieranie wpisów

Backend pobiera wpisy z wybranej kategorii WordPress przez REST API.\
Czyszczenie HTML odbywa się przy użyciu wyrażeń regularnych.

Do indeksu trafiają:

-   tytuł
-   excerpt
-   pełna treść

### 2️⃣ Budowa indeksu TF-IDF

Wykorzystano:

-   max_features=40000 (kontrola pamięci)
-   ngram_range=(1, 2)

Indeks przechowywany jest w pamięci RAM serwera.\
Nie istnieje zewnętrzny vector store.

### 3️⃣ Cache z TTL

Indeks posiada TTL = 7 dni.

Jeżeli:

-   indeks nie istnieje,
-   minął czas ważności,

następuje ponowne pobranie danych i przebudowa indeksu.

Brak trwałego storage.\
Brak bazy danych.

### 4️⃣ Wyszukiwanie

Zapytanie użytkownika:

1.  Zamieniane jest na wektor TF-IDF.
2.  Obliczane jest cosine similarity.
3.  Wyniki są sortowane.
4.  Wybierane jest TOP_K dopasowań.
5.  Jeżeli najlepszy wynik \< MIN_SIMILARITY → model nie jest
    wywoływany.

### 5️⃣ Ekstrakcja kontekstu

Zamiast wysyłać pełne artykuły:

-   wycinany jest fragment wokół słowa kluczowego,
-   ograniczana jest długość kontekstu,
-   dodawane są wielokropki przy przycięciu tekstu.

Redukuje to:

-   zużycie tokenów,
-   koszty inferencji,
-   ryzyko halucynacji.

### 6️⃣ Generowanie odpowiedzi

Model otrzymuje:

-   maksymalnie 3 wpisy,
-   ściśle sformatowaną listę,
-   ograniczenie długości odpowiedzi,
-   instrukcję, aby nie dodawał innych linków.

Fallback:

-   brak wyników → brak wywołania AI,
-   brak klucza API → system działa jako wyszukiwarka,
-   rate limit → fallback bez AI.

------------------------------------------------------------------------

## 💰 Architektura zaprojektowana z myślą o kontroli kosztów

System został zaprojektowany z myślą o minimalizacji kosztów
operacyjnych.

### Kluczowe decyzje

#### Stateless backend

-   brak pamięci rozmowy,
-   brak sesji,
-   brak rosnącego zużycia tokenów,
-   brak przechowywania danych użytkownika.

#### Brak embeddingów

-   brak generowania embeddingów,
-   brak vector database,
-   brak dodatkowych kosztów modelowych.

#### Ograniczony kontekst

-   max 3 wpisy,
-   fragment zamiast całego artykułu,
-   kontrola długości odpowiedzi.

#### AI wywoływane warunkowo

-   brak wyników → brak wywołania modelu,
-   minimalny próg podobieństwa,
-   kontrola promptu.

#### Brak stałych kosztów infrastrukturalnych

-   brak bazy danych,
-   brak storage,
-   brak płatnych usług,
-   darmowa warstwa Render,
-   darmowy monitoring UptimeRobot.

Projekt może działać bez generowania stałych kosztów utrzymania.

------------------------------------------------------------------------

## 🔐 Bezpieczeństwo

-   Tokeny API przechowywane jako zmienne środowiskowe.
-   Brak kluczy API i danych uwierzytelniających w repozytorium.
-   Ograniczenie CORS do konkretnego WP_BASE_URL.
-   Stateless design → brak przechowywania danych użytkowników.

------------------------------------------------------------------------

## 📈 Ograniczenia

-   Brak pamięci rozmowy.
-   Wyszukiwanie oparte na słowach kluczowych.
-   Indeks przechowywany w RAM.
-   Brak panelu administracyjnego.

Ograniczenia wynikają ze świadomych decyzji architektonicznych.

------------------------------------------------------------------------

## 🧠 Wnioski z projektu

### 🔹 Najpierw wyszukiwanie, potem generowanie

Jakość odpowiedzi modelu zależy od jakości kontekstu bardziej niż od
samego modelu.

TF-IDF + kontrolowany snippet znacząco redukują:

-   koszty,
-   halucynacje,
-   niekontrolowane generowanie.

Dobrze zaprojektowana warstwa retrieval jest kluczowa.

### 🔹 Stateless architecture upraszcza system

Brak pamięci rozmowy:

-   upraszcza backend,
-   eliminuje potrzebę sesji,
-   zapobiega narastaniu kosztów tokenów,
-   redukuje ryzyko prywatności.

Prostota może być przewagą.

### 🔹 Koszt to element architektury

Optymalizacja kosztów była elementem projektowania, nie kompromisem.

Decyzje:

-   brak embeddingów,
-   brak vector store,
-   kontrolowany prompt,
-   warunkowe wywołanie AI.

To pokazuje, że AI może być używane odpowiedzialnie.

### 🔹 Cache w RAM często wystarcza

Zamiast budować złożony system storage:

-   użyto TTL,
-   indeks w RAM,
-   odświeżanie przy wygaśnięciu.

To uprościło architekturę i operacyjność.

### 🔹 Kontrola promptu jest elementem bezpieczeństwa

Ograniczenie liczby wyników i długości odpowiedzi:

-   redukuje halucynacje,
-   zwiększa przewidywalność,
-   utrzymuje spójność systemu.

### 🔹 RAG bez frameworków jest możliwy

System został zbudowany bez LangChain i bez zewnętrznych orkiestratorów.

Pozwoliło to:

-   zrozumieć każdy element przepływu,
-   zachować pełną kontrolę,
-   uniknąć nadmiarowej abstrakcji.

### 🔹 Monitoring jest częścią systemu

Endpoint /health + UptimeRobot:

-   zapobiega cold start,
-   poprawia doświadczenie użytkownika,
-   zwiększa stabilność.

System to nie tylko kod, ale również operacyjność.

------------------------------------------------------------------------

## 🏗 Filozofia projektu

Warkan nie jest próbą zbudowania najbardziej zaawansowanego chatbota.

Jest próbą zbudowania:

-   przemyślanego,
-   kontrolowanego,
-   ekonomicznego,
-   transparentnego systemu RAG,
-   możliwego do utrzymania bez zespołu i bez budżetu.

Projekt pokazuje, że AI może być używane w sposób świadomy i
architektonicznie odpowiedzialny.
