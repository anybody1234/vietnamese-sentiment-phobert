"""So sánh fp32 với int8 động (QUANTIZE=int8 trong app/main.py) trên cùng máy, cùng điều kiện.

Bước 1 (mặc định): kích thước model, metric trên VALIDATION, tỉ lệ câu đổi nhãn, kết quả có đổi theo batch
không, latency trong tiến trình.
Quyết định có dùng int8 hay không chỉ dựa trên bước này.
    python -X utf8 -m src.quantization
Bước 2 (chạy MỘT lần, sau bước 1): chấm TEST cho bản int8 để báo cáo, vì đây là một bản triển khai riêng.
fp32 không chấm lại test: lấy số đã có trong results/phobert_len128_cwbalanced_s42.json.
    python -X utf8 -m src.quantization --test
"""

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from app.main import MODEL_DIR, SentimentClassifier
from src.bench_latency import N1, N16, N_WARMUP, environment, make_requests, on_ac_power, summarize, time_request
from src.data import LABEL_NAMES, load_split
from src.dataset import segment
from src.evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quantization.json"
FP32_RUN = ROOT / "results" / "phobert_len128_cwbalanced_s42.json"
VARIANTS = {"fp32": "none", "int8": "int8"}  # tên trong báo cáo -> giá trị QUANTIZE


def state_dict_mb(model):
    """Kích thước file khi torch.save(state_dict): đo fp32 và int8 theo cùng một cách."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state_dict.pt"
        torch.save(model.state_dict(), path)
        return round(path.stat().st_size / 2**20, 1)


def predict_all(clf, segmented, batch_size=64):
    """Xác suất cho cả một tập, qua đúng tokenize + forward của API (câu đã tách từ sẵn một lần)."""
    chunks = [clf.forward(clf.tokenize(segmented[i:i + batch_size])) for i in range(0, len(segmented), batch_size)]
    return torch.cat(chunks).numpy()


def brief(m):
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "f1_per_class": {n: round(m["per_class"][n]["f1-score"], 4) for n in LABEL_NAMES},
            "confusion_matrix": m["confusion_matrix"]}


def bench_interleaved(clfs, requests, warmup):
    """Đo xen kẽ hai bản trên cùng request, đảo thứ tự sau mỗi request: CPU nóng lên hay tải nền thay đổi
    theo thời gian thì ảnh hưởng đều cả hai, không dồn vào bản đo sau."""
    out = {name: {} for name in clfs}
    for bs, reqs in requests.items():
        for texts in warmup[bs]:
            for clf in clfs.values():
                time_request(clf, texts)
        rows = {name: [] for name in clfs}
        for i, texts in enumerate(reqs):
            for name in (list(clfs) if i % 2 == 0 else list(clfs)[::-1]):
                rows[name].append(time_request(clfs[name], texts))
        for name, r in rows.items():
            res = summarize([x["total"] for x in r], bs)
            res["model_only_ms"] = summarize([x["model"] for x in r], bs)["per_request_ms"]  # int8 chỉ đổi chặng này
            out[name][f"batch_{bs}"] = res
    return out


def load_result():
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save(data):
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compare_on_validation():
    data = load_result()  # giữ phần "test" nếu đã có
    data.update(model_dir=MODEL_DIR.name, method="torch.ao.quantization.quantize_dynamic({nn.Linear}, qint8)",
                measured_at=datetime.now().isoformat(timespec="seconds"), on_ac_power=on_ac_power(),
                environment=environment())
    val = load_split("validation")
    val_seg = segment(val.text)
    clfs = {name: SentimentClassifier(MODEL_DIR, q) for name, q in VARIANTS.items()}

    # 1) Kích thước: checkpoint safetensors trên đĩa chỉ có bản fp32; int8 được tạo lúc khởi động
    data["size_mb"] = {name: state_dict_mb(clf.model) for name, clf in clfs.items()}
    data["size_mb"]["fp32_checkpoint_safetensors"] = round((MODEL_DIR / "model.safetensors").stat().st_size / 2**20, 1)

    # 2) Chất lượng trên validation và số câu đổi nhãn
    proba = {name: predict_all(clf, val_seg) for name, clf in clfs.items()}
    pred = {name: p.argmax(axis=1) for name, p in proba.items()}
    changed = pred["fp32"] != pred["int8"]
    transitions = Counter(f"{LABEL_NAMES[a]} -> {LABEL_NAMES[b]}" for a, b in zip(pred["fp32"][changed], pred["int8"][changed]))
    reported = json.loads(FP32_RUN.read_text(encoding="utf-8"))["best_validation"]["macro_f1"]
    data["validation"] = {
        "n_sentences": len(val),
        **{name: brief(compute_metrics(val.label, p)) for name, p in pred.items()},
        "changed_predictions": {"count": int(changed.sum()), "fraction": round(float(changed.mean()), 4),
                                "fp32_to_int8": dict(transitions)},
        "max_abs_prob_diff": round(float(np.abs(proba["fp32"] - proba["int8"]).max()), 4),
        "fp32_macro_f1_reported_in_training": reported,  # Kaggle T4, fp16: đối chiếu với fp32 trên CPU ở trên
    }
    save(data)  # lưu ngay, phòng bị ngắt giữa chừng ở các bước sau
    print(json.dumps({k: data["validation"][k] for k in ("fp32", "int8", "changed_predictions")}, ensure_ascii=False))

    # 3) Kết quả một câu có đổi theo các câu đi cùng request không: gửi riêng từng câu so với batch 64.
    # fp32 chỉ lệch do làm tròn số thực. int8 động tính thang lượng tử hoá activation theo min/max của cả batch,
    # nên cùng một câu có thể ra xác suất (thậm chí nhãn) khác tuỳ request -> khó tái lập, khó debug
    single = {name: predict_all(clf, val_seg, batch_size=1) for name, clf in clfs.items()}
    data["batch_dependence"] = {
        name: {"label_changes": int((single[name].argmax(1) != pred[name]).sum()),
               "max_abs_prob_diff": float(np.abs(single[name] - proba[name]).max()),
               "compared": "batch 1 vs batch 64, toàn bộ validation"}
        for name in clfs}
    save(data)
    print(json.dumps(data["batch_dependence"], ensure_ascii=False))

    # 4) Latency trong tiến trình, cùng request với src/bench_latency.py
    requests, warmup = make_requests(N1, N16, N_WARMUP)
    data["latency_inprocess"] = bench_interleaved(clfs, requests, warmup)
    save(data)
    for name, res in data["latency_inprocess"].items():
        for key, r in res.items():
            print(f"{name} {key}: p50 {r['per_request_ms']['p50']} ms, p95 {r['per_request_ms']['p95']} ms "
                  f"(riêng model p50 {r['model_only_ms']['p50']} ms)")


def evaluate_int8_on_test_once():
    data = load_result()
    if "validation" not in data:
        sys.exit("Chạy bước 1 (không có --test) trước")
    if "test" in data:
        sys.exit(f"Test của bản int8 đã chấm lúc {data['test']['evaluated_at']}; không chấm lại")
    test = load_split("test")
    t0 = time.perf_counter()
    proba = predict_all(SentimentClassifier(MODEL_DIR, "int8"), segment(test.text))
    fp32 = json.loads(FP32_RUN.read_text(encoding="utf-8"))["test"]
    data["test"] = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "n_sentences": len(test),
        "int8": brief(compute_metrics(test.label, proba.argmax(axis=1))),
        "fp32_reference": {**brief(fp32), "source": FP32_RUN.name + " (Kaggle T4, fp16, không chấm lại)"},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    save(data)
    print(json.dumps(data["test"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh fp32 và int8 động")
    parser.add_argument("--test", action="store_true", help="chấm test cho bản int8 (chỉ một lần)")
    evaluate_int8_on_test_once() if parser.parse_args().test else compare_on_validation()
