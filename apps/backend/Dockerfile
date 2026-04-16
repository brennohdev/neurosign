FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv sync --no-dev

COPY neurosign_backend/ neurosign_backend/

EXPOSE 8000

CMD ["uvicorn", "neurosign_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
