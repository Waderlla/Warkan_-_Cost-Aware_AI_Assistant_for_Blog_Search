# 🐺 Warkan -- Cost-Aware AI Blog Assistant

Warkan is a custom Retrieval-Augmented Generation (RAG) system
integrated with a blog:

🔗 https://olgamironczuk.pl/warkan/

The project was designed as a lightweight, cost-aware and
architecturally simplified AI assistant whose purpose is to search and
summarize blog content without generating fixed infrastructure costs.

This is not a classic chatbot built purely around a language model.\
It is a system where the retrieval layer is as important as the
generative layer.

------------------------------------------------------------------------

## 🎯 Project Goal

The objective was to build a system that:

-   searches posts from a specific WordPress category,
-   builds a local text index,
-   finds the most relevant articles,
-   generates a short response based exclusively on retrieved results,
-   operates without a database,
-   does not store conversation history,
-   minimizes language model usage,
-   avoids fixed operational costs.

The project was created as part of building a technical portfolio
combining data analysis, backend architecture, and AI systems.

------------------------------------------------------------------------

## 🧠 System Architecture

User → WordPress UI → FastAPI (Render)\
→ TF-IDF retrieval\
→ snippet extraction\
→ Cloudflare Workers AI\
→ JSON response → UI

The system consists of four main layers.

### 1. Interface Layer

A minimal chat interface embedded in WordPress.\
Its sole responsibility is sending user queries to the backend and
displaying the JSON response.

The project focuses on backend architecture, therefore the frontend
layer is intentionally simple.

### 2. Backend (Render Web Service)

The backend is written in Python using FastAPI.\
It is responsible for:

-   fetching posts from the WordPress REST API,
-   cleaning HTML content,
-   building the TF-IDF index,
-   computing similarity,
-   extracting contextual snippets,
-   conditionally invoking the language model,
-   handling cost control and fallback mechanisms.

The application operates in a stateless model.

### 3. Retrieval Layer

This is the core of the system.

Instead of sending full articles to the model:

-   posts are fetched from WordPress,
-   cleaned from HTML,
-   combined into documents (title + excerpt + content),
-   indexed using TF-IDF,
-   compared against the user query using cosine similarity.

As a result, the language model receives only the most relevant
fragments.

### 4. Generative AI Layer

The language model (Cloudflare Workers AI, Llama) is invoked only when:

-   the retrieval layer finds relevant matches,
-   similarity exceeds a defined minimum threshold.

The model receives:

-   a maximum of 3 results,
-   extracted snippets instead of full articles,
-   a tightly controlled prompt,
-   explicit instruction not to add new sources.

------------------------------------------------------------------------

## 🛠 Tech Stack

### Backend

-   Python\
-   FastAPI\
-   Uvicorn\
-   scikit-learn (TF-IDF + cosine similarity)\
-   NumPy (matrix computations)\
-   Requests (HTTP API)\
-   python-dotenv

### Infrastructure

-   Render (Web Service)\
-   Cloudflare Workers AI (LLM)\
-   WordPress REST API (data source)\
-   UptimeRobot (service activity monitoring)\
-   GitHub (repository)

------------------------------------------------------------------------

## ⚙️ How the System Works Step by Step

### 1️⃣ Fetching Posts

The backend retrieves posts from a selected WordPress category via REST
API.\
HTML is cleaned using regular expressions.

Indexed content includes:

-   title
-   excerpt
-   full content

### 2️⃣ Building the TF-IDF Index

Configuration includes:

-   max_features=40000 (memory control)
-   ngram_range=(1, 2)

The index is stored in server RAM.\
No external vector store is used.

### 3️⃣ Cache with TTL

The index has a TTL of 7 days.

If:

-   the index does not exist,
-   the TTL expires,

the system refetches data and rebuilds the index.

No persistent storage.\
No database.

### 4️⃣ Retrieval and Minimum Similarity Threshold

The user query:

1.  Is transformed into a TF-IDF vector.\
2.  Cosine similarity is computed.\
3.  Results are sorted.\
4.  TOP_K matches are selected.\
5.  If the best score \< MIN_SIMILARITY → the model is not invoked.

### 5️⃣ Snippet Extraction

Instead of sending full articles:

-   a fragment around the keyword is extracted,
-   context length is limited,
-   ellipses are added if truncated.

This reduces:

-   token usage,
-   inference cost,
-   hallucination risk.

### 6️⃣ Response Generation

The model receives:

-   a maximum of 3 entries,
-   a strictly formatted list,
-   response length constraints,
-   instructions not to add additional links.

Fallback mechanisms:

-   no retrieval results → no AI call,
-   missing API key → retrieval-only mode,
-   rate limit → fallback without AI.

------------------------------------------------------------------------

## 💰 Cost-Aware Architecture

The system was intentionally designed to minimize operational costs.

### Key Design Decisions

#### Stateless Backend

-   no conversation memory,
-   no session management,
-   no growing token usage,
-   no user data storage.

#### No Embeddings

-   no embedding generation,
-   no vector database,
-   no additional model costs.

#### Limited Context

-   maximum of 3 results,
-   snippets instead of full articles,
-   response length control.

#### Conditional AI Invocation

-   no results → no model call,
-   minimum similarity threshold,
-   strict prompt control.

#### No Fixed Infrastructure Costs

-   no database,
-   no storage layer,
-   no paid services,
-   free Render tier,
-   free UptimeRobot monitoring.

The project can operate without generating recurring infrastructure
costs.

------------------------------------------------------------------------

## 🔐 Security

-   API tokens are stored as environment variables.
-   No API keys or credentials are included in the repository.
-   CORS is restricted to a specific WP_BASE_URL.
-   Stateless design → no user data storage.

------------------------------------------------------------------------

## 📈 Limitations

-   No conversation memory.
-   Keyword-based retrieval (not semantic embeddings).
-   Index stored in RAM.
-   No administrative panel.

These limitations are conscious architectural decisions.

------------------------------------------------------------------------

## 🧠 Lessons Learned

### 🔹 Retrieval Before Generation

The quality of model responses depends more on context quality than on
model size.

TF-IDF combined with controlled snippet extraction significantly
reduces:

-   costs,
-   hallucinations,
-   uncontrolled generation.

A well-designed retrieval layer is essential.

### 🔹 Stateless Architecture Simplifies the System

No conversation memory:

-   simplifies the backend,
-   eliminates session complexity,
-   prevents token growth,
-   improves privacy.

Simplicity can be an advantage.

### 🔹 Cost Is an Architectural Element

Cost optimization was part of the design, not a compromise.

Decisions include:

-   no embeddings,
-   no external vector store,
-   controlled prompts,
-   conditional AI invocation.

AI can be implemented responsibly.

### 🔹 RAM Cache Is Often Enough

Instead of building complex storage:

-   TTL-based refresh,
-   in-memory index,
-   automatic rebuild.

This simplifies both architecture and operations.

### 🔹 Prompt Control Is a Security Mechanism

Limiting results and response length:

-   reduces hallucinations,
-   improves predictability,
-   maintains system consistency.

### 🔹 RAG Without Frameworks Is Possible

The system was built without LangChain or external orchestration
frameworks.

This allowed:

-   full understanding of the pipeline,
-   complete control,
-   avoidance of unnecessary abstraction.

### 🔹 Monitoring Is Part of the System

Endpoint `/health` + UptimeRobot:

-   prevents cold start,
-   improves user experience,
-   increases stability.

A system is not only code, but also operations.

------------------------------------------------------------------------

## 🏗 Project Philosophy

Warkan is not an attempt to build the most advanced chatbot.

It is an attempt to build:

-   a thoughtful,
-   controlled,
-   economical,
-   transparent RAG system,
-   maintainable without a team or budget.

The project demonstrates that AI can be implemented in a conscious and
architecturally responsible way.
