"""API phân loại cảm xúc câu phản hồi tiếng Việt (negative / neutral / positive) bằng PhoBERT đã fine-tune.

Chạy: uvicorn app.main:app --host 127.0.0.1 --port 8000
Biến môi trường: MODEL_DIR = thư mục checkpoint dạng save_pretrained (trọng số, tokenizer, id2label);
QUANTIZE = "none" (fp32, mặc định) hoặc "int8" (lượng tử hoá động, xem results/quantization.json).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import Collator, segment

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models" / "phobert_len128_cwbalanced_s42"))
QUANTIZE = os.getenv("QUANTIZE", "none")
BASE_MODEL = "vinai/phobert-base"
MAX_LEN = 128     # phải bằng --max_len lúc train: khác đi thì câu dài bị cắt khác, model thấy input lạ
MAX_TEXTS = 64    # chặn một request chiếm CPU quá lâu, làm các request khác phải chờ
MAX_CHARS = 1000  # 128 token chỉ cỡ vài trăm ký tự; câu dài hơn chỉ làm tách từ chậm mà phần đuôi vẫn bị cắt

log = logging.getLogger("uvicorn.error")  # dùng logger của uvicorn để log hiện cùng chỗ với log server


class SentimentClassifier:
    """Gói tokenizer + model + tiền xử lý. Chỉ tạo MỘT lần lúc khởi động rồi dùng chung cho mọi request:
    nạp 500 MB trọng số mất vài giây, làm lại mỗi request thì latency tăng vọt."""

    def __init__(self, model_dir: Path, quantize: str = QUANTIZE):
        if quantize not in ("none", "int8"):  # kiểm tra trước khi nạp 500 MB: gõ sai thì báo lỗi ngay
            raise ValueError(f"QUANTIZE phải là 'none' hoặc 'int8', nhận được {quantize!r}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()  # tắt dropout: cùng một câu luôn ra cùng một kết quả
        if quantize == "int8":
            # Lượng tử hoá động: trọng số các lớp nn.Linear lưu dạng int8, activation được lượng tử hoá ngay
            # lúc chạy nên không cần dữ liệu hiệu chỉnh. Embedding và LayerNorm giữ fp32.
            # Đánh đổi đo được ở results/quantization.json. API này đã deprecated (hướng thay thế: torchao)
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.quantization = quantize
        # Train/serve skew: nếu tiền xử lý lúc serve khác lúc train (tách từ khác, max_len khác...),
        # model nhận input khác phân phối đã học và đoán kém đi mà không báo lỗi gì.
        # Vì vậy API gọi lại đúng segment() và Collator đã dùng khi train, không viết lại bản thứ hai.
        self.collate = Collator(self.tokenizer, MAX_LEN)
        config = self.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]  # thứ tự nhãn lưu trong checkpoint

    def tokenize(self, segmented: list[str]) -> dict:
        # Collator lúc train cần cặp (câu, nhãn); API không có nhãn nên truyền 0 rồi bỏ qua cột labels
        batch = self.collate([(text, 0) for text in segmented])
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

    @torch.inference_mode()  # như no_grad nhưng chặt hơn: không ghi lại gì cho autograd nên nhanh hơn một chút
    def forward(self, batch: dict) -> torch.Tensor:
        logits = self.model(**batch).logits
        return torch.softmax(logits.float(), dim=-1)  # xác suất, mỗi hàng cộng lại bằng 1

    def predict_proba(self, texts: list[str]) -> torch.Tensor:
        # Pad động: Collator chỉ pad tới câu dài nhất trong request, câu ngắn không tốn tính toán cho 128 token
        return self.forward(self.tokenize(segment(texts)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy một lần trước khi server nhận request. MODEL_DIR sai thì server dừng ngay (fail fast)
    thay vì chạy lên rồi request nào cũng lỗi."""
    t0 = time.perf_counter()
    classifier = SentimentClassifier(MODEL_DIR)
    classifier.predict_proba(["khởi động"])  # chạy thử: nạp sẵn model tách từ của underthesea (nạp lười ở lần gọi đầu)
    app.state.load_seconds = round(time.perf_counter() - t0, 2)
    app.state.classifier = classifier
    log.info("Đã nạp %s trong %.2f s (torch threads = %d)", MODEL_DIR.name, app.state.load_seconds,
             torch.get_num_threads())
    yield


app = FastAPI(title="Vietnamese sentiment (PhoBERT)", version="1.0.0", lifespan=lifespan)


def not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("câu không được chỉ gồm khoảng trắng")
    return text  # trả nguyên văn, không strip: tiền xử lý phải giống hệt lúc train


Text = Annotated[str, Field(min_length=1, max_length=MAX_CHARS), AfterValidator(not_blank)]


class PredictRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_TEXTS, examples=[["giảng viên dạy rất nhiệt tình"]])


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    name: str
    base_model: str
    max_len: int
    quantization: str


class PredictResponse(BaseModel):
    model: ModelInfo
    predictions: list[Prediction]  # cùng thứ tự với texts trong request


@app.get("/health")
def health():
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:  # 503 để Docker HEALTHCHECK / load balancer chưa chuyển request tới
        return JSONResponse(status_code=503, content={"status": "loading", "model_loaded": False})
    return {"status": "ok", "model_loaded": True, "model": MODEL_DIR.name,
            "quantization": classifier.quantization, "load_seconds": app.state.load_seconds}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Hàm thường (không async): suy luận là việc nặng CPU. FastAPI chạy hàm thường trong threadpool,
    # nên event loop không bị chặn và /health vẫn trả lời trong lúc model đang tính
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        return JSONResponse(status_code=503, content={"detail": "model chưa sẵn sàng"})
    proba = classifier.predict_proba(request.texts)
    predictions = [
        Prediction(label=classifier.labels[int(row.argmax())],
                   probabilities=dict(zip(classifier.labels, row.tolist())))
        for row in proba
    ]
    info = ModelInfo(name=MODEL_DIR.name, base_model=BASE_MODEL, max_len=MAX_LEN,
                     quantization=classifier.quantization)
    return PredictResponse(model=info, predictions=predictions)
