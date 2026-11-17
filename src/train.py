"""Bước 4–5: fine-tune PhoBERT-base phân loại cảm xúc 3 lớp bằng vòng lặp PyTorch tự viết.

Epoch tốt nhất được chọn theo macro-F1 trên validation. Test chỉ dùng đúng một lần ở cuối,
trong `evaluate_on_test_once`. Chạy thật trên Kaggle T4 (xem kaggle/). Chạy thử trên CPU:
python -X utf8 -m src.train --max_len 64 --class_weights none --subset 64 --epochs 1 --batch_size 8
"""

import argparse
import json
import math
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import LABEL_NAMES, load_split
from src.dataset import Collator, SentimentDataset, segment, take_subset
from src.evaluate import compute_metrics, predict, save_confusion_matrix, save_curves, save_errors

MODEL_NAME = "vinai/phobert-base"
ROOT = Path(__file__).resolve().parent.parent
RESULTS, MODELS = ROOT / "results", ROOT / "models"
FIGURES = RESULTS / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune PhoBERT-base trên UIT-VSFC (3 lớp cảm xúc)")
    p.add_argument("--max_len", type=int, required=True, help="số token tối đa mỗi câu, tính cả <s> và </s>")
    p.add_argument("--class_weights", choices=["none", "balanced"], required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=None, help="chỉ lấy khoảng N câu mỗi tập để chạy thử")
    p.add_argument("--log_every", type=int, default=50, help="in train loss trung bình sau mỗi N bước")
    p.add_argument("--output_dir", default=None, help="thư mục lưu checkpoint tốt nhất")
    args = p.parse_args(argv)
    if not 2 < args.max_len <= 256:  # PhoBERT chỉ có vị trí embedding cho tối đa 256 token
        p.error("--max_len phải nằm trong khoảng (2, 256]")
    return args


def make_run_name(args):
    """Tên run ghi các lựa chọn chính, dùng đặt tên file kết quả và checkpoint."""
    name = f"len{args.max_len}_cw{args.class_weights}_s{args.seed}"
    return f"smoke{args.subset}_{name}" if args.subset else name  # chạy thử: tên khác hẳn lần chạy thật


def set_seed(seed):
    """Cố định mọi nguồn ngẫu nhiên: khởi tạo lớp phân loại mới, dropout, thứ tự xáo dữ liệu."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # cuDNN dùng thuật toán tất định
    torch.backends.cudnn.benchmark = False     # không tự dò thuật toán nhanh nhất (lựa chọn có thể đổi giữa các lần)


def balanced_class_weights(labels, n_classes=3):
    """Cùng công thức với sklearn class_weight="balanced": w_c = số câu / (số lớp * số câu của lớp c).
    Lớp càng ít câu thì đoán sai câu của lớp đó càng bị phạt nặng."""
    counts = np.bincount(labels, minlength=n_classes)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float)


def peak_vram_mb(device):
    """Bộ nhớ GPU lớn nhất mà tensor đã chiếm. nvidia-smi sẽ thấy nhiều hơn vì còn CUDA context và cache."""
    return round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, use_amp, step, log_every, step_log):
    """Một lượt qua toàn bộ train. Trả về loss của từng bước và số bước đã chạy tính từ đầu."""
    model.train()  # bật dropout
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()  # xoá gradient bước trước, nếu không PyTorch sẽ cộng dồn vào
        # fp16 trên GPU: nhân ma trận nhanh hơn và tốn ít VRAM hơn. T4 không hỗ trợ bf16 nên dùng fp16
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        # Tự tính loss từ logits (không đưa labels vào model) để chủ động chọn trọng số lớp. Tính ở fp32 cho ổn định
        loss = loss_fn(logits.float(), batch["labels"])
        scaler.scale(loss).backward()  # nhân loss lên trước khi backward để gradient nhỏ không bị fp16 làm tròn về 0
        scaler.unscale_(optimizer)  # chia ngược về gradient thật, nếu không ngưỡng cắt 1.0 bên dưới sẽ sai
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # chặn bước cập nhật quá lớn
        scaler.step(optimizer)  # tự bỏ qua bước này nếu gradient có inf/NaN do fp16 tràn số
        scaler.update()         # chỉnh hệ số nhân cho bước sau
        scheduler.step()        # learning rate đi theo lịch warmup rồi giảm tuyến tính
        step += 1
        losses.append(loss.item())
        if step == 1:  # lớp phân loại mới khởi tạo đoán gần đều 3 lớp, nên loss phải gần -ln(1/3)
            print(f"Loss bước đầu tiên = {losses[0]:.4f} (kỳ vọng gần ln 3 = {math.log(3):.4f})")
        if step % log_every == 0:
            mean = float(np.mean(losses[-log_every:]))
            step_log.append({"step": step, "train_loss": round(mean, 4), "lr": scheduler.get_last_lr()[0]})
            print(f"  bước {step:>5} | train loss {log_every} bước gần nhất = {mean:.4f} | lr = {scheduler.get_last_lr()[0]:.2e}")
    return losses, step


def evaluate_on_test_once(ckpt_dir, test_df, args, loss_fn, device, use_amp):
    """Đánh giá test ĐÚNG MỘT LẦN, bằng checkpoint đã chọn theo validation (nạp lại từ đĩa, tức đúng
    model mà API sẽ dùng). Nếu xem điểm test rồi quay lại chỉnh tham số, test trở thành validation thứ hai
    và con số báo cáo sẽ lạc quan hơn thực tế, không còn ước lượng đúng hiệu năng trên dữ liệu mới."""
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    test_seg = segment(test_df.text)
    loader = DataLoader(SentimentDataset(test_seg, test_df.label), batch_size=args.batch_size * 2,
                        shuffle=False, collate_fn=Collator(tokenizer, args.max_len))
    t0 = time.perf_counter()
    _, proba = predict(model, loader, loss_fn, device, use_amp)
    metrics = compute_metrics(test_df.label, proba.argmax(axis=1))
    metrics["inference_seconds"] = round(time.perf_counter() - t0, 2)  # cả tập test, gồm tokenize, chưa gồm tách từ
    return metrics, proba, test_seg


def main(args):
    run_name = make_run_name(args)
    ckpt_dir = Path(args.output_dir) if args.output_dir else MODELS / f"phobert_{run_name}"
    set_seed(args.seed)  # đặt trước khi tạo model, vì lớp phân loại mới được khởi tạo ngẫu nhiên
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision chỉ có lợi trên GPU
    gpu_name = torch.cuda.get_device_name(0) if use_amp else None
    print(f"Run {run_name} | device = {gpu_name or 'cpu'} | args = {vars(args)}")

    # 1) Dữ liệu: tải, lấy mẫu nhỏ nếu chạy thử, tách từ. Test chưa đụng tới cho đến bước 4
    train_df, val_df, test_df = (load_split(s) for s in ("train", "validation", "test"))
    if args.subset:
        train_df, val_df, test_df = (take_subset(df, args.subset, args.seed) for df in (train_df, val_df, test_df))
    t0 = time.perf_counter()
    train_seg, val_seg = segment(train_df.text), segment(val_df.text)
    print(f"Tách từ train + val: {time.perf_counter() - t0:.1f} s")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = Collator(tokenizer, args.max_len)
    # num_workers=0: tokenize một batch câu ngắn rất nhanh, không cần thêm tiến trình con
    train_loader = DataLoader(SentimentDataset(train_seg, train_df.label), batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)  # thứ tự xáo cố định
    # Khi đánh giá không lưu gradient nên batch gấp đôi vẫn đủ bộ nhớ
    val_loader = DataLoader(SentimentDataset(val_seg, val_df.label), batch_size=args.batch_size * 2, collate_fn=collate)

    # 2) Model: thân PhoBERT đã pretrain + lớp phân loại 3 lớp mới. id2label lưu vào config để API trả tên nhãn
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES), id2label=dict(enumerate(LABEL_NAMES)),
        label2id={name: i for i, name in enumerate(LABEL_NAMES)}).to(device)
    weights = None
    if args.class_weights == "balanced":
        weights = balanced_class_weights(train_df.label.to_numpy()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # val loss dùng cùng hàm loss để so được với train loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% bước đầu tăng dần lr từ 0: lớp phân loại mới còn ngẫu nhiên,
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)  # gradient lớn dễ phá trọng số pretrain
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)  # tắt trên CPU: mọi lệnh scaler thành lệnh thường

    # 3) Vòng lặp train: sau mỗi epoch chấm validation, chỉ lưu checkpoint khi val macro-F1 tăng
    epoch_log, step_log, step, best_epoch, best_val, first_step_loss = [], [], 0, 0, None, None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        losses, step = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, device, use_amp,
                                       step, args.log_every, step_log)
        if epoch == 1:
            first_step_loss = losses[0]
        val_loss, val_proba = predict(model, val_loader, loss_fn, device, use_amp)
        val_m = compute_metrics(val_df.label, val_proba.argmax(axis=1))
        improved = best_val is None or val_m["macro_f1"] > best_val["macro_f1"]
        epoch_log.append({
            "epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val_loss": round(val_loss, 4),
            "val_macro_f1": val_m["macro_f1"], "val_accuracy": val_m["accuracy"],
            "val_f1_per_class": {n: val_m["per_class"][n]["f1-score"] for n in LABEL_NAMES},
            "lr_end": scheduler.get_last_lr()[0], "epoch_seconds": round(time.perf_counter() - t0, 1),
            "peak_vram_mb": peak_vram_mb(device), "saved_checkpoint": improved})
        if improved:
            best_epoch, best_val = epoch, val_m
            model.save_pretrained(ckpt_dir)  # trọng số + config.json
            tokenizer.save_pretrained(ckpt_dir)  # lưu kèm tokenizer để nạp lại không cần mạng
        print(f"Epoch {epoch}: train loss {epoch_log[-1]['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {val_m['macro_f1']:.4f} | {epoch_log[-1]['epoch_seconds']} s"
              + (" | lưu checkpoint" if improved else ""))
    train_seconds = time.perf_counter() - train_start

    # 4) Test: giải phóng model đang train và AdamW (giữ 2 bản trạng thái cỡ bằng model) rồi nạp checkpoint tốt nhất.
    # Xoá cả scheduler vì nó vẫn giữ tham chiếu tới optimizer
    del model, optimizer, scheduler
    if use_amp:
        torch.cuda.empty_cache()
    test_m, test_proba, test_seg = evaluate_on_test_once(ckpt_dir, test_df, args, loss_fn, device, use_amp)

    # 5) Lưu kết quả, hình và các câu đoán sai
    FIGURES.mkdir(parents=True, exist_ok=True)  # exist_ok: không xoá file có sẵn
    title = f"PhoBERT-base ({run_name})"
    save_confusion_matrix(test_m["confusion_matrix"], f"{title}: test macro-F1 = {test_m['macro_f1']:.3f}",
                          FIGURES / f"cm_phobert_{run_name}.png")
    save_curves(epoch_log, best_epoch, title, FIGURES / f"curves_phobert_{run_name}.png")
    n_errors = save_errors(test_df.text, test_seg, test_df.label, test_proba, RESULTS / f"errors_phobert_{run_name}.csv")
    ckpt = ckpt_dir.resolve()
    result = {
        "model": MODEL_NAME, "run_name": run_name, "seed": args.seed, "args": vars(args),
        "segmenter": "underthesea.word_tokenize(text, format='text')", "selection_metric": "validation macro-F1",
        "versions": {"python": platform.python_version(),
                     **{pkg: version(pkg) for pkg in ("torch", "transformers", "tokenizers", "underthesea",
                                                      "numpy", "scikit-learn")}},
        "device": device.type, "gpu_name": gpu_name, "mixed_precision_fp16": use_amp,
        "n_sentences": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "class_weights": weights.tolist() if weights is not None else None,
        "steps": {"per_epoch": len(train_loader), "total": total_steps, "warmup": warmup_steps},
        "first_step_loss": round(first_step_loss, 4), "expected_first_step_loss_ln3": round(math.log(3), 4),
        "epochs": epoch_log, "step_log": step_log, "best_epoch": best_epoch, "best_validation": best_val,
        "train_seconds": round(train_seconds, 1), "peak_vram_mb": peak_vram_mb(device),
        "checkpoint_dir": ckpt.relative_to(ROOT).as_posix() if ckpt.is_relative_to(ROOT) else str(ckpt_dir),
        "checkpoint_size_mb": round(sum(f.stat().st_size for f in ckpt.iterdir()) / 2**20, 1),
        "test": test_m, "n_test_errors": n_errors,
    }
    with open(RESULTS / f"phobert_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nBest epoch {best_epoch}: val macro-F1 = {best_val['macro_f1']:.4f} | train {train_seconds / 60:.1f} phút"
          f" | peak VRAM = {result['peak_vram_mb']} MB")
    print(f"Test macro-F1 = {test_m['macro_f1']:.4f} | accuracy = {test_m['accuracy']:.4f} | "
          f"F1 từng lớp: { {n: test_m['per_class'][n]['f1-score'] for n in LABEL_NAMES} }")


if __name__ == "__main__":
    main(parse_args())
