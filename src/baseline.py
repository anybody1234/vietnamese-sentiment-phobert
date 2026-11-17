"""Bước 3: baseline Count/TF-IDF + Logistic Regression cho 3 lớp cảm xúc.

Mọi lựa chọn (vectorizer, n-gram, C, class_weight) dựa trên macro-F1 của validation.
Test chỉ được dùng đúng một lần ở cuối, trong `evaluate_on_test_once`.
Chạy từ thư mục project: python -X utf8 -m src.baseline
"""

import itertools
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # chỉ lưu ảnh ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline

from src.data import LABEL_NAMES, load_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RESULTS, MODELS = ROOT / "results", ROOT / "models"
FIGURES = RESULTS / "figures"

# Lưới nhỏ: 2 vectorizer x 2 n-gram x 2 class_weight x 6 giá trị C = 48 cấu hình
VECTORIZERS = {"count": CountVectorizer, "tfidf": TfidfVectorizer}
GRID = list(itertools.product(VECTORIZERS, [(1, 1), (1, 2)], [None, "balanced"], [0.25, 0.5, 1, 2, 4, 8]))


def build_pipeline(vectorizer, ngram_range, class_weight, C):
    """Gói vectorizer + model vào một Pipeline: gọi fit() trên train thì từ điển và IDF
    chỉ học từ train, nên val/test không bao giờ rò rỉ vào bước tạo đặc trưng."""
    return Pipeline([
        # Câu đã cách nhau bằng dấu cách theo từng âm tiết. Pattern mặc định (?u)\b\w\w+\b bỏ token
        # dưới 2 ký tự, tức mất âm tiết như "ở", "ý" và dấu câu "!", "?". \S+ giữ mọi cụm không chứa dấu cách.
        ("vec", VECTORIZERS[vectorizer](ngram_range=ngram_range, token_pattern=r"\S+")),
        # C nhỏ = phạt L2 mạnh = model đơn giản hơn. max_iter lớn để lbfgs hội tụ cả khi C lớn
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=SEED)),
    ])


def select_on_validation(train, val):
    """Fit từng cấu hình trên train, chấm trên validation. Không dùng GridSearchCV vì bộ dữ liệu
    đã có sẵn tập validation riêng, và vòng lặp tường minh thì dễ kiểm tra hơn."""
    grid = []
    for vec, ngram, cw, C in GRID:
        pred = build_pipeline(vec, ngram, cw, C).fit(train.text, train.label).predict(val.text)
        p, r, f, _ = precision_recall_fscore_support(val.label, pred, zero_division=0)
        grid.append({
            "vectorizer": vec, "ngram_range": ngram, "class_weight": cw, "C": C,
            "val_macro_f1": round(f.mean(), 4),  # trung bình F1 các lớp, lớp nhỏ nặng ngang lớp lớn
            "val_accuracy": round(accuracy_score(val.label, pred), 4),
            "val_f1_per_class": dict(zip(LABEL_NAMES, f.round(4).tolist())),
            "val_neutral_precision": round(float(p[1]), 4),  # index 1 = neutral
            "val_neutral_recall": round(float(r[1]), 4),
        })
        print(f"{vec:<6} {str(ngram):<7} {str(cw):<9} C={C:<5} val macro-F1 = {f.mean():.4f}")
    return grid


def evaluate_on_test_once(model, test):
    """Đánh giá test ĐÚNG MỘT LẦN, sau khi cấu hình đã chốt trên validation.
    Nếu xem điểm test rồi quay lại chỉnh tham số, test trở thành validation thứ hai và con số
    báo cáo sẽ lạc quan hơn thực tế, không còn ước lượng đúng hiệu năng trên dữ liệu mới."""
    t0 = time.perf_counter()
    proba = model.predict_proba(test.text)  # cả tập test trong một lần gọi
    batch_seconds = time.perf_counter() - t0
    pred = model.classes_[proba.argmax(axis=1)]  # classes_ = [0, 1, 2] nên cột thứ i là nhãn i
    report = classification_report(test.label, pred, target_names=LABEL_NAMES, output_dict=True)
    metrics = {
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {n: {k: round(v, 4) for k, v in report[n].items()} for n in LABEL_NAMES},
        "confusion_matrix": confusion_matrix(test.label, pred).tolist(),  # hàng = nhãn thật, cột = dự đoán
        "ms_per_sentence_full_test": round(1000 * batch_seconds / len(test), 4),
    }
    return metrics, pred, proba


def time_batch1(model, texts, n=200):
    """Đo độ trễ khi API nhận từng câu một (batch = 1) trên CPU. Chỉ đo thời gian, không dùng nhãn."""
    model.predict_proba(texts[:1])  # chạy nháp một lần để bỏ chi phí khởi động
    ms = []
    for s in texts[:n]:
        t0 = time.perf_counter()
        model.predict_proba([s])
        ms.append(1000 * (time.perf_counter() - t0))
    return {"n": n, "mean_ms": round(np.mean(ms), 4), "p50_ms": round(np.percentile(ms, 50), 4),
            "p95_ms": round(np.percentile(ms, 95), 4)}


def top_ngrams(model, k=15):
    """Mỗi lớp lấy k n-gram có hệ số dương lớn nhất: xuất hiện là đẩy xác suất về phía lớp đó."""
    names = model.named_steps["vec"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_  # shape (3 lớp, số n-gram)
    return {label: [[names[j], round(float(coef[i, j]), 4)] for j in np.argsort(coef[i])[::-1][:k]]
            for i, label in enumerate(LABEL_NAMES)}


def save_confusion_matrix(cm, title, path):
    disp = ConfusionMatrixDisplay(np.array(cm), display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues", values_format="d", colorbar=False)  # một màu, càng đậm càng nhiều câu
    disp.ax_.set(title=title, xlabel="Predicted label", ylabel="True label")
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=150)
    plt.close(disp.figure_)


def main():
    train, val, test = (load_split(s) for s in ("train", "validation", "test"))
    FIGURES.mkdir(parents=True, exist_ok=True)  # exist_ok: không xoá file có sẵn trong thư mục
    MODELS.mkdir(parents=True, exist_ok=True)

    # 1) Chọn cấu hình theo macro-F1 trên validation
    grid = select_on_validation(train, val)
    best = max(grid, key=lambda g: g["val_macro_f1"])
    config = {k: best[k] for k in ("vectorizer", "ngram_range", "class_weight", "C")}

    # Cấu hình tốt nhất khi không / có class_weight="balanced", để xem tác động lên lớp neutral (~4% dữ liệu)
    cw_effect = {str(cw): max((g for g in grid if g["class_weight"] == cw), key=lambda g: g["val_macro_f1"])
                 for cw in (None, "balanced")}

    # 2) Fit lại cấu hình tốt nhất, vẫn chỉ trên train
    t0 = time.perf_counter()
    model = build_pipeline(**config).fit(train.text, train.label)
    fit_seconds = time.perf_counter() - t0

    # 3) Test: đúng một lần
    test_metrics, pred, proba = evaluate_on_test_once(model, test)
    test_metrics["latency_batch1"] = time_batch1(model, test.text.tolist())

    # 4) Lưu model, kết quả, hình và các câu đoán sai
    joblib.dump(model, MODELS / "baseline.joblib")
    title = f"{config['vectorizer']} {config['ngram_range']} + LogReg: test macro-F1 = {test_metrics['macro_f1']:.3f}"
    save_confusion_matrix(test_metrics["confusion_matrix"], title, FIGURES / "cm_baseline.png")
    labels = np.array(LABEL_NAMES)
    wrong = pred != test.label.to_numpy()
    errors = pd.DataFrame({"text": test.text, "true_label": labels[test.label], "pred_label": labels[pred],
                           "pred_proba": proba.max(axis=1).round(4)})[wrong]
    # Câu sai mà model tự tin nhất lên đầu (đáng đọc nhất). utf-8-sig để Excel hiện đúng tiếng Việt
    errors.sort_values("pred_proba", ascending=False).to_csv(
        RESULTS / "errors_baseline.csv", index=False, encoding="utf-8-sig")
    with open(RESULTS / "baseline_top_ngrams.json", "w", encoding="utf-8") as f:
        json.dump(top_ngrams(model), f, ensure_ascii=False, indent=2)

    result = {
        "model": "baseline", "seed": SEED, "token_pattern": r"\S+", "selection_metric": "validation macro-F1",
        "grid": grid, "best_config": config, "best_validation": best,
        "class_weight_effect_on_validation": cw_effect, "fit_seconds": round(fit_seconds, 3),
        "model_size_mb": round((MODELS / "baseline.joblib").stat().st_size / 2**20, 2),
        "test": test_metrics, "n_test_errors": int(wrong.sum()),
    }
    with open(RESULTS / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 5) Tóm tắt
    print(f"\nCấu hình tốt nhất (theo val): {config} | val macro-F1 = {best['val_macro_f1']:.4f}")
    print(f"Test macro-F1 = {test_metrics['macro_f1']:.4f} | accuracy = {test_metrics['accuracy']:.4f}")
    print("Test F1 từng lớp:", {n: test_metrics["per_class"][n]["f1-score"] for n in LABEL_NAMES})
    for cw, g in cw_effect.items():
        print(f"class_weight={cw:<9} best val macro-F1 = {g['val_macro_f1']:.4f}, "
              f"val neutral F1 = {g['val_f1_per_class']['neutral']:.4f} "
              f"(precision {g['val_neutral_precision']:.4f}, recall {g['val_neutral_recall']:.4f})")
    lat = test_metrics["latency_batch1"]
    print(f"Latency CPU batch=1: mean {lat['mean_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms | "
          f"cả tập test: {test_metrics['ms_per_sentence_full_test']:.4f} ms/câu | fit {fit_seconds:.1f} s")


if __name__ == "__main__":
    main()
