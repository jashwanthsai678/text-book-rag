FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000

# Shell form (not exec array) so $PORT expands - most PaaS platforms
# (Render included) assign a port at runtime rather than always using 8000.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
