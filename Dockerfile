# Dockerfile (minimal)
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --upgrade pip
RUN pip install numpy openai

CMD ["python", "examples/demo.py"]
