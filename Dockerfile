FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first - plain `pip install torch` pulls full CUDA/GPU
# packages (several GB) we don't need for running a small embedding model.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so containers start fast
# and don't need internet access to Hugging Face at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
