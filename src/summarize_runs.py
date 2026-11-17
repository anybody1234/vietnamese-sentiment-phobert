"""Tổng hợp các lần train PhoBERT chính thức (results/phobert_len128_*.json) và chọn model.

Quy tắc chọn được chốt TRƯỚC khi xem kết quả và chỉ dựa trên VALIDATION:
1) Cấu hình loss (class_weights none hay balanced): chọn cấu hình có trung bình val macro-F1
   (tại epoch tốt nhất của từng seed) cao hơn trên 3 seed. Bằng nhau thì chọn "none" vì đơn giản hơn.
2) Checkpoint đại diện: trong cấu hình đã chọn, lấy run có val macro-F1 cao nhất.
Điểm test của cả hai cấu hình được báo cáo nhưng không bao giờ dùng để chọn: nếu chọn theo test,
test trở thành validation thứ hai và con số báo cáo sẽ lạc quan hơn thực tế.
Chạy từ thư mục project: python -X utf8 -m src.summarize_runs
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
PATTERN = "phobert_len128_*.json"  # chỉ run chính thức; file chạy thử bắt đầu bằng phobert_smoke


def load_runs():
    runs = []
    for path in sorted(RESULTS.glob(PATTERN)):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        val_losses = [e["val_loss"] for e in r["epochs"]]
        runs.append({
            "run_name": r["run_name"], "class_weights": r["args"]["class_weights"], "seed": r["seed"],
            "best_epoch": r["best_epoch"], "val_macro_f1": r["best_validation"]["macro_f1"],
            # Epoch có val loss thấp nhất: nếu sớm hơn best_epoch nhiều thì model bắt đầu quá tự tin (overfit)
            "val_loss_min_epoch": int(np.argmin(val_losses)) + 1,
            "test_macro_f1": r["test"]["macro_f1"], "test_accuracy": r["test"]["accuracy"],
            "test_neutral_f1": r["test"]["per_class"]["neutral"]["f1-score"],
            "train_minutes": round(r["train_seconds"] / 60, 1), "peak_vram_mb": r["peak_vram_mb"],
            "first_step_loss": r["first_step_loss"],
        })
    return runs


def mean_std(values):
    """std mẫu (ddof=1) vì 3 seed chỉ là một mẫu nhỏ trong mọi seed có thể."""
    return {"mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None, "n": len(values)}


def main():
    runs = load_runs()
    configs = {}
    for cw in ("none", "balanced"):
        group = [r for r in runs if r["class_weights"] == cw]
        if len(group) != 3:
            print(f"Cảnh báo: class_weights={cw} có {len(group)} run, kỳ vọng 3")
        if group:
            configs[cw] = {key: mean_std([r[key] for r in group])
                           for key in ("val_macro_f1", "test_macro_f1", "test_accuracy", "test_neutral_f1")}

    # Bước 1: chọn cấu hình theo trung bình val macro-F1. max() giữ phần tử đầu khi bằng nhau, nên "none" thắng hoà
    chosen_cw = max(configs, key=lambda cw: configs[cw]["val_macro_f1"]["mean"])
    # Bước 2: trong cấu hình đó, chọn run có val macro-F1 cao nhất
    chosen = max((r for r in runs if r["class_weights"] == chosen_cw), key=lambda r: r["val_macro_f1"])

    with open(RESULTS / "baseline.json", encoding="utf-8") as f:
        baseline = json.load(f)["test"]
    summary = {
        "selection_rule": "cấu hình: trung bình val macro-F1 cao hơn trên 3 seed (hoà thì chọn none); "
                          "checkpoint: run có val macro-F1 cao nhất trong cấu hình đã chọn. Không dùng test để chọn.",
        "runs": runs, "per_config_mean_std": configs,
        "chosen_class_weights": chosen_cw, "chosen_run": chosen["run_name"],
        "chosen_run_metrics": chosen,
        "baseline_test": {"macro_f1": baseline["macro_f1"], "accuracy": baseline["accuracy"]},
    }
    with open(RESULTS / "phobert_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"{'run':<22} {'best ep':>7} {'val F1':>7} {'test F1':>7} {'test acc':>8} {'neu F1':>7} {'phút':>5} {'VRAM MB':>8}")
    for r in runs:
        print(f"{r['run_name']:<22} {r['best_epoch']:>7} {r['val_macro_f1']:>7.4f} {r['test_macro_f1']:>7.4f} "
              f"{r['test_accuracy']:>8.4f} {r['test_neutral_f1']:>7.4f} {r['train_minutes']:>5} {r['peak_vram_mb']:>8}")
    for cw, m in configs.items():
        print(f"class_weights={cw:<9} " + " | ".join(f"{k} {v['mean']:.4f} ± {v['std']}" for k, v in m.items()))
    print(f"Chọn class_weights={chosen_cw}, checkpoint {chosen['run_name']} (val macro-F1 {chosen['val_macro_f1']:.4f})")


if __name__ == "__main__":
    main()
