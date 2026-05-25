# --- build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- runtime stage ---
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY model/       model/
COPY serving/     serving/
COPY monitoring/  monitoring/
COPY src/         src/
COPY configs/     configs/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
