"""Test API bằng TestClient: gọi thẳng app trong cùng tiến trình, không cần chạy server.
Dùng checkpoint thật; máy chưa có models/ thì bỏ qua cả file (models/ không nằm trong git)."""

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_CHARS, MAX_TEXTS, MODEL_DIR, SentimentClassifier, app

if not (MODEL_DIR / "config.json").exists():
    pytest.skip(f"không thấy checkpoint ở {MODEL_DIR}", allow_module_level=True)

LABELS = {"negative", "neutral", "positive"}
POSITIVE = "giảng viên dạy rất nhiệt tình và dễ hiểu"
NEGATIVE = "thầy dạy quá nhanh , không hiểu gì cả"


@pytest.fixture(scope="module")
def client():
    # scope="module": nạp model một lần cho cả file. "with" để chạy lifespan (nạp model lúc khởi động)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True
    assert r.json()["quantization"] in {"none", "int8"}


def test_unknown_quantize_value_fails_fast():
    with pytest.raises(ValueError):
        SentimentClassifier(MODEL_DIR, quantize="fp16")


def test_predict_single(client):
    r = client.post("/predict", json={"texts": [POSITIVE]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["max_len"] == 128
    [pred] = body["predictions"]
    assert set(pred["probabilities"]) == LABELS
    assert sum(pred["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert pred["label"] == max(pred["probabilities"], key=pred["probabilities"].get)  # nhãn = lớp xác suất cao nhất
    assert pred["label"] == "positive"  # câu khen rõ ràng: kiểm tra thứ tự nhãn không bị đảo


def test_predict_batch_keeps_order_and_matches_single(client):
    texts = [POSITIVE, NEGATIVE, "môn học có nhiều bài tập về nhà"]
    body = client.post("/predict", json={"texts": texts}).json()
    batch = body["predictions"]
    assert [p["label"] for p in batch][:2] == ["positive", "negative"]
    if body["model"]["quantization"] != "none":
        # int8 động tính thang lượng tử hoá activation theo min/max của cả batch, nên kết quả một câu
        # phụ thuộc các câu đi cùng request (số đo: results/quantization.json, mục batch_dependence)
        pytest.skip("int8 động không cho cùng kết quả giữa gửi riêng và gửi theo batch")
    # Pad động: câu ngắn bị pad thêm khi đi cùng câu dài, nhưng attention_mask làm model bỏ qua pad,
    # nên xác suất phải gần như bằng lúc gửi riêng từng câu
    for text, p_batch in zip(texts, batch):
        p_single = client.post("/predict", json={"texts": [text]}).json()["predictions"][0]
        assert p_single["label"] == p_batch["label"]
        for label in LABELS:
            assert p_single["probabilities"][label] == pytest.approx(p_batch["probabilities"][label], abs=1e-4)


@pytest.mark.parametrize("body", [
    {},                                   # thiếu trường texts
    {"texts": []},                        # danh sách rỗng
    {"texts": [""]},                      # câu rỗng
    {"texts": ["   "]},                   # chỉ có khoảng trắng
    {"texts": ["a" * (MAX_CHARS + 1)]},   # câu quá dài
    {"texts": ["hay"] * (MAX_TEXTS + 1)}, # quá nhiều câu trong một request
    {"texts": "giảng viên dạy hay"},      # chuỗi thay vì danh sách
])
def test_invalid_input_rejected(client, body):
    assert client.post("/predict", json=body).status_code == 422
