# Image chạy API trên CPU. Model KHÔNG nằm trong image (500 MB, đổi model không cần build lại):
# mount thư mục models/ chỉ-đọc lúc chạy.
#   docker build -t phobert-sentiment-api:1.0 .
#   docker run --rm -p 8000:8000 -v "${PWD}/models:/models:ro" phobert-sentiment-api:1.0
#   (bật int8: thêm -e QUANTIZE=int8; đổi checkpoint: -e MODEL_DIR=/models/<tên thư mục>)
FROM python:3.11-slim

# HF_HUB_OFFLINE: model nạp từ thư mục mount, không bao giờ cần gọi mạng tới Hugging Face
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_DIR=/models/phobert_len128_cwbalanced_s42 \
    QUANTIZE=none

WORKDIR /app

# torch bản CPU từ index chính thức của PyTorch, nhỏ hơn nhiều so với bản kèm CUDA.
# Layer nặng nhất và ít đổi nhất đặt trước: sửa code thì Docker dùng lại cache, không tải lại torch
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Chỉ chép phần API cần: app/ và src/dataset.py (tách từ + Collator dùng chung với lúc train)
COPY src/__init__.py src/dataset.py src/
COPY app/ app/

# Không chạy bằng root: nếu app bị khai thác, kẻ tấn công không có quyền root trong container
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
# Dùng Python thay curl (image slim không có curl). /health trả 503 khi model chưa sẵn sàng -> urlopen lỗi -> unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Một worker: mỗi worker nạp riêng một bản model (~1 GB RAM). Cần nhiều thông lượng hơn thì chạy thêm container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
