from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.retrieval.search import RetrieveRequest, RetrieveResponse, retrieve_content

app = FastAPI(title="Textbook Retrieval API")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/retrieve-content", response_model=RetrieveResponse)
def retrieve_content_endpoint(request: RetrieveRequest) -> RetrieveResponse:
    return retrieve_content(request)
