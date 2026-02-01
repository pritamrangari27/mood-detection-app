FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Reduce TensorFlow memory usage
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

# Use single worker to reduce memory, preload app to share memory
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120"]