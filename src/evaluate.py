"""Dự đoán trên một tập, metric, confusion matrix, đường cong train và file câu đoán sai."""

import matplotlib
matplotlib.use("Agg")  # chỉ lưu ảnh ra file, không cần mở cửa sổ (Kaggle không có màn hình)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.data import LABEL_NAMES

LABELS = list(range(len(LABEL_NAMES)))  # [0, 1, 2]
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"  # xanh = train, cam = validation, giữ nguyên ở mọi hình


@torch.no_grad()  # không lưu đồ thị tính gradient: tốn ít bộ nhớ hơn và chạy nhanh hơn
def predict(model, loader, loss_fn, device, use_amp):
    """Chạy model trên cả một tập (validation hoặc test). Trả về loss trung bình và xác suất từng lớp (n câu x 3)."""
    model.eval()  # tắt dropout để cùng một câu luôn ra cùng một dự đoán
    total_loss, n, probs = 0.0, 0, []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        total_loss += loss_fn(logits.float(), batch["labels"]).item() * len(batch["labels"])
        n += len(batch["labels"])
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())  # logits -> xác suất, tổng mỗi hàng = 1
    return total_loss / n, torch.cat(probs).numpy()


def compute_metrics(y_true, y_pred):
    """macro-F1 là metric chính: trung bình F1 của 3 lớp, nên lớp neutral (~4%) nặng ngang lớp lớn.
    labels=LABELS để kết quả luôn đủ 3 lớp, kể cả khi một tập nhỏ (chạy --subset) thiếu một lớp."""
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=LABEL_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "accuracy": round(float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {n: {k: round(v, 4) for k, v in report[n].items()} for n in LABEL_NAMES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),  # hàng = thật, cột = dự đoán
    }


def save_confusion_matrix(cm, title, path):
    disp = ConfusionMatrixDisplay(np.array(cm), display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues", values_format="d", colorbar=False)  # một màu, càng đậm càng nhiều câu
    disp.ax_.set(title=title, xlabel="Predicted label", ylabel="True label")
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=150)
    plt.close(disp.figure_)


def save_curves(epoch_log, best_epoch, title, path):
    """Trái: train loss và val loss theo epoch (val loss tăng trong khi train loss giảm = overfit).
    Phải: val macro-F1, metric dùng để chọn checkpoint. Loss và F1 khác thang đo nên tách 2 hình."""
    epochs = [e["epoch"] for e in epoch_log]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [e["train_loss"] for e in epoch_log], "o-", color=TRAIN_COLOR, lw=2, ms=6, label="train loss")
    ax_loss.plot(epochs, [e["val_loss"] for e in epoch_log], "o-", color=VAL_COLOR, lw=2, ms=6, label="val loss")
    ax_loss.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax_loss.legend(frameon=False)
    ax_f1.plot(epochs, [e["val_macro_f1"] for e in epoch_log], "o-", color=VAL_COLOR, lw=2, ms=6)
    ax_f1.set(xlabel="Epoch", ylabel="Macro-F1", title=f"Validation macro-F1 (best epoch {best_epoch})")
    for ax in (ax_loss, ax_f1):
        ax.axvline(best_epoch, color="0.6", ls="--", lw=1)  # nét đứt = epoch có checkpoint được giữ lại
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_errors(texts, segmented, y_true, proba, path):
    """Lưu các câu đoán sai kèm xác suất từng lớp. Câu sai mà model tự tin nhất lên đầu (đáng đọc nhất).
    Giữ cả câu đã tách từ để thấy lỗi nào do underthesea tách sai. utf-8-sig để Excel hiện đúng tiếng Việt."""
    labels, y_true = np.array(LABEL_NAMES), np.asarray(y_true)
    pred = proba.argmax(axis=1)
    df = pd.DataFrame({"text": list(texts), "text_segmented": list(segmented),
                       "true_label": labels[y_true], "pred_label": labels[pred],
                       "pred_proba": proba.max(axis=1).round(4)})
    for i, name in enumerate(LABEL_NAMES):
        df[f"p_{name}"] = proba[:, i].round(4)
    errors = df[pred != y_true].sort_values("pred_proba", ascending=False)
    errors.to_csv(path, index=False, encoding="utf-8-sig")
    return len(errors)
