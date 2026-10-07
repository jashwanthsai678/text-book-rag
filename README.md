# Textbook RAG

A retrieval pipeline over the [EduTeach](https://edu-teach-textbook-api-interface.onrender.com/) textbook API: it ingests published textbooks (text + images), indexes them in Qdrant, and exposes a semantic retrieval API so downstream tools (lesson/prep-material generators, test generators, etc.) can pull grounded textbook content for a given topic instead of relying on an LLM's own knowledge.

This repo covers the **retrieval layer only** — given a query, it returns the real textbook passages and images relevant to it. Turning that into generated lesson material, worksheets, or tests is a separate, not-yet-built layer that consumes this API.

## How it works

```
EduTeach Textbook API
        |
        v
  Ingestion (app/ingestion)
   - discover every published book (/published/books)
   - discover each book's chapters (/published/books/{id}/chapters)
   - fetch each chapter's content (/published/books/{id}/chapters/{n})
   - parse markdown content: split on <!-- page N --> markers and
     ### headings, resolve <img id="..."/> tags against the images[] array
        |
        v
  data/processed/{book_id}.json   (pages/sections + linked images, page-accurate)
        |
        v
  Embeddings (app/embeddings)
   - BAAI/bge-small-en-v1.5 (local, via sentence-transformers)
   - text chunks embedded directly; images embedded via their captions
     (no vision model in v1 - captions are already descriptive)
        |
        v
  Qdrant (app/vectorstore)
   - textbook_text collection   (one point per section/page chunk)
   - textbook_images collection (one point per image, caption-embedded)
        |
        v
  Retrieval API (app/retrieval, app/main.py)
   POST /retrieve-content
   {book_id, chapter?, query, top_k_text, top_k_images}
   -> relevant text chunks + relevant images, filtered by book/chapter
     metadata before semantic search
        |
        v
  Minimal web UI (app/static/index.html) at GET /
   - query form + rendered results, for manually inspecting retrieval quality
```

No LangChain/LlamaIndex — ingestion, chunking, embeddings, and retrieval are all plain code so the pipeline stays easy to reason about and migrate later.

## Project structure

```
app/
  ingestion/
    api_client.py   # list_books, list_chapters, fetch_chapter (with retry/backoff)
    parser.py       # markdown -> page-anchored, heading-anchored sections
    models.py       # pydantic models: raw API response + parsed internal format
  embeddings/
    embedder.py     # sentence-transformers wrapper
  vectorstore/
    qdrant_store.py # Qdrant client, collection setup, upsert
  retrieval/
    search.py       # retrieve_content(): embeds query, filters + searches both collections
  static/
    index.html      # manual-testing UI
  main.py           # FastAPI app: GET / (UI), POST /retrieve-content
scripts/
  ingest_book.py        # discovers every published book and chapter, writes data/processed/*.json
  build_index.py        # embeds + upserts every processed book into Qdrant
  refresh_image_urls.py # payload-only refresh of image URLs (see "Image URL expiry" below)
data/
  processed/        # ingestion output (gitignored, regenerate via ingest_book.py)
Dockerfile           # CPU-only torch, model pre-downloaded at build time (~2.3GB image)
```

## Setup

### 1. Python environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Qdrant

**Production / default: Qdrant Cloud.** Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) and put its URL + API key in `.env` (see below). This is what the deployed app uses - data survives independently of any one machine.

**Local dev fallback:** a local Docker Qdrant also works (no API key needed) if you want to develop without touching the cloud cluster:

```powershell
docker run -d `
  --name textbook-qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v textbook_qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

Dashboard: http://localhost:6333/dashboard

### 3. Environment variables

Copy `.env.example` to `.env` and fill in `QDRANT_URL` (+ `QDRANT_API_KEY` if using Qdrant Cloud). The `*_FILTER` vars narrow ingestion to a subset of the catalog (by board/grade/subject/language) if you don't want to ingest every published book.

## Usage

```powershell
# 1. Ingest every published book from the EduTeach API into data/processed/
python scripts\ingest_book.py

# 2. Embed + index everything in data/processed/ into Qdrant
python scripts\build_index.py

# 3. Refresh image URLs (needed periodically - see "Image URL expiry" below)
python scripts\refresh_image_urls.py

# 4. Run the API + UI
uvicorn app.main:app --reload --port 8000
```

### Docker

```powershell
docker build -t textbook-retrieval .
docker run -d -p 8000:8000 --env-file .env textbook-retrieval
```

Open http://localhost:8000 for the manual test UI, or call the API directly:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/retrieve-content" -Method Post -ContentType "application/json" -Body '{"book_id":"ts_scert_class5_environmental_studies_en","chapter":2,"query":"agricultural tools used for cultivation","top_k_text":3,"top_k_images":2}'
```

Response shape:

```json
{
  "text": [
    {"chunk_id": "...", "page_number": 27, "section_title": "2.2 Agricultural equipment/ tools", "text": "...", "score": 0.83}
  ],
  "images": [
    {"image_id": "img_c5ch2_11", "page_number": 27, "caption": "...", "url": "...", "score": 0.81}
  ]
}
```

## Production status

- ✅ **Qdrant Cloud** — the live index runs on a managed Qdrant Cloud cluster (AWS, eu-central-1), not local Docker. Local Docker Qdrant still works as a dev-only fallback (no API key needed).
- ✅ **Image URL expiry handled.** The textbook API returns Supabase signed URLs valid for ~6 hours. `scripts/refresh_image_urls.py` does a payload-only update (no re-embedding) of every indexed image's URL by re-fetching its chapter. Run it on a schedule (every 3-4 hours) in production so served URLs are never stale.
- ✅ **Dockerized.** `Dockerfile` builds a ~2.3GB image (CPU-only torch, embedding model pre-downloaded at build time so containers start fast with no Hugging Face access needed at runtime).
- ⬜ **Not yet deployed.** The API still only runs via `uvicorn` on a local/manual machine - no public URL yet.
- ⬜ **Not every published book is indexed.** The catalog currently has 8 published books; only `ts_scert_class5_environmental_studies_en` has been run through `build_index.py`. `ingest_book.py` already discovers and can ingest all of them - it just hasn't been run against the full catalog yet.
- ⬜ **No generation layer yet.** This repo stops at retrieval. Prep material / worksheet / test generation (feeding retrieved content into an LLM) is a separate, not-yet-built consumer of `/retrieve-content`.
- ⬜ **Ingestion is still manual.** `ingest_book.py` / `build_index.py` / `refresh_image_urls.py` are run by hand. Production needs these on a schedule (cron / scheduled job).
- ℹ️ **EduTeach API uptime.** It runs on a free Render tier and cold-starts slowly (sometimes causing request timeouts) — `api_client.py` retries on timeout/5xx to absorb this. Some chapters also contain images with a null `url` (upstream data issue) - these are skipped rather than breaking the whole chapter.
