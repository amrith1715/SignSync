FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY serve_with_cors.py recognition_service.py ./
COPY recognition/ ./recognition/
COPY models/mlp.pt models/lstm_jz.pt models/hand_landmarker.task ./models/
COPY assets/asl/manifest.json ./assets/asl/manifest.json

EXPOSE 8001
CMD ["python", "serve_with_cors.py"]
