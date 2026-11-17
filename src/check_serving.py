"""Kiểm tra train/serve skew: cùng một tập câu, đi qua API và đi qua đường đánh giá offline (lúc train)
phải ra cùng nhãn và gần như cùng xác suất. Lệch là dấu hiệu tiền xử lý lúc serve khác lúc train.

Dùng validation (không đụng test). Không cần chạy server: TestClient gọi thẳng app trong tiến trình này.
python -X utf8 -m src.check_serving
"""

import numpy as np
import torch
from fastapi.testclient import TestClient
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.main import MAX_LEN, MAX_TEXTS, MODEL_DIR, app
from src.data import load_split
from src.dataset import Collator, SentimentDataset, segment
from src.evaluate import predict

N_SENTENCES = 300
API_CHUNK = 50  # cố ý khác batch offline (64): pad động khác nhau mà kết quả vẫn phải giống
TOLERANCE = 1e-4


def offline_proba(texts, labels):
    """Đường đánh giá lúc train (giống evaluate_on_test_once trong src/train.py): nạp checkpoint từ đĩa,
    tách từ, Collator, predict() trên CPU, không dùng fp16."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    loader = DataLoader(SentimentDataset(segment(texts), labels), batch_size=64, shuffle=False,
                        collate_fn=Collator(tokenizer, MAX_LEN))
    _, proba = predict(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), use_amp=False)
    return proba


def api_proba(texts):
    """Đường API: gửi JSON tới /predict, đúng như client thật (gồm validate Pydantic và serialize JSON)."""
    assert API_CHUNK <= MAX_TEXTS
    rows = []
    with TestClient(app) as client:  # "with" để chạy lifespan: app tự nạp model như khi chạy server
        labels = None
        for i in range(0, len(texts), API_CHUNK):
            r = client.post("/predict", json={"texts": texts[i:i + API_CHUNK]})
            r.raise_for_status()
            for p in r.json()["predictions"]:
                labels = labels or list(p["probabilities"])
                rows.append([p["probabilities"][name] for name in labels])
    return np.array(rows), labels


def main():
    val = load_split("validation").sample(n=N_SENTENCES, random_state=42)  # mẫu cố định, đủ cả 3 lớp
    texts, y_true = val.text.tolist(), val.label.to_numpy()

    p_offline = offline_proba(texts, y_true)
    p_api, label_order = api_proba(texts)
    assert label_order == ["negative", "neutral", "positive"], label_order  # cùng thứ tự cột với offline

    same_label = p_offline.argmax(axis=1) == p_api.argmax(axis=1)
    max_diff = float(np.abs(p_offline - p_api).max())
    print(f"{N_SENTENCES} câu validation | nhãn trùng: {same_label.sum()}/{N_SENTENCES} | "
          f"chênh xác suất lớn nhất: {max_diff:.2e} (ngưỡng {TOLERANCE:.0e})")
    print(f"Accuracy trên mẫu này: offline {np.mean(p_offline.argmax(1) == y_true):.4f} | "
          f"API {np.mean(p_api.argmax(1) == y_true):.4f}")
    assert same_label.all(), f"{(~same_label).sum()} câu khác nhãn giữa API và offline"
    assert max_diff < TOLERANCE, f"xác suất lệch {max_diff:.2e} >= {TOLERANCE:.0e}"
    print("OK: API cho kết quả giống đường đánh giá offline")


if __name__ == "__main__":
    main()
