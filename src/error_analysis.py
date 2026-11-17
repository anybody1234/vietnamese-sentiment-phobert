"""Bước 6: phân tích lỗi trên test của model đã chọn, so với baseline và với các seed khác.

Chỉ đọc file đã lưu (results/errors_*.csv và results/*.json), không chạy lại model.
File errors_*.csv không có cột chỉ số dòng, nên ghép với test theo nguyên văn câu. Cách này chỉ
đúng khi test không có 2 câu trùng nhau, nên script kiểm tra điều đó trước. Sau khi ghép, script
dựng lại confusion matrix của từng run và so với confusion matrix đã lưu trong JSON: khớp thì
việc ghép là đúng.

Phần phân loại lỗi thủ công nằm ở results/error_categories.csv. Chưa có file thì script tạo khung
(cột category và note để trống). Có rồi thì script chỉ đọc để tổng hợp, không ghi đè.

Giấy phép của UIT-VSFC chưa rõ nên câu gốc của bộ dữ liệu không nằm trong file được commit:
results/error_analysis.json chỉ chứa số liệu; câu ví dụ ghi riêng vào results/error_examples.json
(cùng error_categories.csv và errors_*.csv nằm trong .gitignore).
Chạy từ thư mục project: python -X utf8 -m src.error_analysis
"""

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LABEL_NAMES, load_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
CATEGORIES_CSV = RESULTS / "error_categories.csv"
EXAMPLES_JSON = RESULTS / "error_examples.json"  # chứa câu của bộ dữ liệu: không commit
EXAMPLES_POINTER = "xem results/error_examples.json (không commit)"
SEEDS = (42, 43, 44)
RUNS = {cw: [f"len128_cw{cw}_s{s}" for s in SEEDS] for cw in ("balanced", "none")}
ANNOTATION_STATUS = "1 người gán (có AI hỗ trợ), chưa được chủ dự án kiểm tra lại"

# Từ phủ định, so khớp theo âm tiết (dữ liệu đã tách dấu câu bằng dấu cách).
# Bỏ "không khí", "không gian" vì ở đó "không" không mang nghĩa phủ định.
NEGATION_WORDS = {"không", "chưa", "chẳng", "chả", "đừng"}
NOT_NEGATION_BIGRAMS = {("không", "khí"), ("không", "gian")}

# Placeholder của UIT-VSFC, cùng regex với notebooks/01_eda.ipynb. "doubledot" phải xét trước "dot"
PLACEHOLDER_FAMILIES = [
    ("wzjwz<số>", r"wzjwz\d+"),
    ("colon...", r"^colon"),
    ("doubledot", r"doubledot"),
    ("dot", r"dot"),
    ("fraction", r"^fraction$"),
]
# Có ít nhất 1 chữ cái mang dấu tiếng Việt. Câu không có chữ nào như vậy thường là tiếng Anh (suy đoán)
VIETNAMESE_CHARS = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]")
LENGTH_BINS = [(1, 3), (4, 6), (7, 10), (11, 20), (21, 40), (41, None)]
# Định nghĩa các nhóm lỗi dùng khi đọc tay (cột category của error_categories.csv)
CATEGORY_DEFINITIONS = {
    "Nhãn có vẻ không nhất quán": "câu có từ khen/chê rõ ràng trái với nhãn gốc, hoặc câu gần giống nhau mà nhãn khác nhau",
    "Ranh giới trung tính (mô tả, khen/chê nhẹ, câu cụt)": "câu mô tả sự việc, khen/chê nhẹ ('ổn', 'đầy đủ') hoặc cụm từ thiếu ngữ cảnh",
    "Góp ý, đề xuất, mong muốn": "câu dạng 'nên', 'cần', 'mong', 'hy vọng', 'thêm' hoặc góp ý ngầm không có các từ đó",
    "Khen và chê trong cùng câu": "câu vừa khen vừa chê, thường nối bằng 'nhưng', 'tuy nhiên', 'mặc dù'",
    "Phủ định, hàm ý, mỉa mai": "nghĩa của câu khác nghĩa bề mặt: phủ định kép, câu hỏi tu từ, chê ngầm, mỉa mai, từ tiêu cực nằm trong mệnh đề phụ",
    "Ngoài lề (nói về bản thân, câu hỏi, lời chúc)": "câu không đánh giá giảng viên/môn học",
    "Tiếng Anh, placeholder, lỗi gõ": "câu tiếng Anh, nhiều placeholder/tên riêng, hoặc lỗi chính tả làm đổi nghĩa",
    "Câu rõ nghĩa, model vẫn sai": "câu rõ nghĩa, nhãn hợp lý, không thuộc các nhóm trên",
}
# Quy ước gán nhãn, kiểm tra trên TRAIN: câu cảm ơn ngắn và câu bắt đầu bằng từ góp ý
THANKS_PATTERN = r"\b(?:cảm ơn|cám ơn)\b"
SUGGESTION_START = r"^(?:(?:thầy|cô|giảng viên) )?(?:nên|cần|mong|hy vọng|hi vọng|thêm) "
SMALL_GROUP = 30  # nhóm ít hơn 30 câu: tỉ lệ lỗi dao động mạnh, chỉ để tham khảo


def load_test():
    test = load_split("test").reset_index(drop=True)
    if test.text.duplicated().any():
        raise ValueError("test có câu trùng nhau, không ghép theo nguyên văn câu được")
    return test


def load_predictions(errors_csv, test, saved_cm):
    """Dựng lại nhãn dự đoán cho toàn bộ test từ file câu sai: câu không có trong file là đoán đúng.
    Trả về (mảng nhãn dự đoán, DataFrame câu sai có thêm cột idx = vị trí trong test)."""
    errors = pd.read_csv(errors_csv, encoding="utf-8-sig")
    pos = pd.Series(test.index, index=test.text)
    missing = ~errors.text.isin(pos.index)
    if missing.any():
        raise ValueError(f"{errors_csv.name}: {missing.sum()} câu không tìm thấy trong test")
    errors["idx"] = pos[errors.text].to_numpy()
    name_to_id = {n: i for i, n in enumerate(LABEL_NAMES)}
    true_ids = errors.true_label.map(name_to_id).to_numpy()
    if (true_ids != test.label.to_numpy()[errors.idx]).any():
        raise ValueError(f"{errors_csv.name}: true_label không khớp nhãn của test")
    pred = test.label.to_numpy().copy()
    pred[errors.idx] = errors.pred_label.map(name_to_id).to_numpy()
    cm = confusion(test.label.to_numpy(), pred)
    if cm != saved_cm:
        raise ValueError(f"{errors_csv.name}: confusion matrix dựng lại khác bản đã lưu: {cm} vs {saved_cm}")
    return pred, errors


def confusion(y_true, y_pred):
    k = len(LABEL_NAMES)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)  # hàng = nhãn thật, cột = nhãn dự đoán
    return cm.tolist()


def wilson_ci(k, n, z=1.96):
    """Khoảng tin cậy 95% cho tỉ lệ k/n (Wilson). Nhóm càng nhỏ thì khoảng càng rộng."""
    if n == 0:
        return None
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, float(center - half)), 4), round(min(1.0, float(center + half)), 4)]


def rate(mask, wrong):
    """Số câu, số câu sai và tỉ lệ sai trong một nhóm (mask = các câu thuộc nhóm)."""
    n, k = int(mask.sum()), int((mask & wrong).sum())
    return {"n": n, "n_errors": k, "error_rate": round(k / n, 4) if n else None,
            "ci95": wilson_ci(k, n), "small_group": n < SMALL_GROUP}


def has_negation(text):
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in NEGATION_WORDS and tuple(tokens[i:i + 2]) not in NOT_NEGATION_BIGRAMS:
            return True
    return False


def placeholder_families(text):
    found = set()
    for tok in text.split():
        for family, pattern in PLACEHOLDER_FAMILIES:
            if re.search(pattern, tok):
                found.add(family)
                break
    return found


def normalize_loose(s):
    """Chuẩn hoá giống EDA (mức "lower + bỏ dấu câu") để tìm câu test gần trùng câu train."""
    return s.str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.split().str.join(" ")


def examples(test, idx, preds, k):
    """Lấy k ví dụ theo thứ tự idx đã sắp, kèm nhãn thật và dự đoán của từng model."""
    rows = []
    for i in list(idx)[:k]:
        row = {"text": test.text[i], "true_label": LABEL_NAMES[test.label[i]]}
        row.update({f"pred_{name}": LABEL_NAMES[p[i]] for name, p in preds.items()})
        rows.append(row)
    return rows


def chosen_run_section(test, pred, errors):
    y = test.label.to_numpy()
    pairs = (errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False))
    cm = np.array(confusion(y, pred))
    neu = LABEL_NAMES.index("neutral")
    precision = cm[neu, neu] / cm[:, neu].sum()
    recall = cm[neu, neu] / cm[neu, :].sum()
    p = errors.pred_proba
    confidence = {"<0.5": int((p < 0.5).sum()), "0.5-0.7": int(((p >= 0.5) & (p < 0.7)).sum()),
                  "0.7-0.9": int(((p >= 0.7) & (p < 0.9)).sum()), ">=0.9": int((p >= 0.9).sum())}
    return {
        "n_errors": int(len(errors)), "n_test": int(len(test)), "error_rate": round(len(errors) / len(test), 4),
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "confusion_pairs": [{"true": t, "pred": p, "count": int(c), "pct_of_errors": round(c / len(errors), 4)}
                            for (t, p), c in pairs.items()],
        "neutral": {"support": int(cm[neu, :].sum()), "predicted_as_neutral": int(cm[:, neu].sum()),
                    "correct": int(cm[neu, neu]), "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(2 * precision * recall / (precision + recall)), 4)},
        # Câu sai mà model rất tự tin (>= 0,9) đáng đọc trước: hoặc khó thật, hoặc nhãn có vấn đề
        "errors_by_confidence": confidence,
    }


def baseline_vs_chosen(test, base_pred, phobert_pred, neg_mask):
    """Trả về (số liệu, câu ví dụ). Câu ví dụ được ghi ra file riêng, không vào error_analysis.json."""
    y = test.label.to_numpy()
    b_wrong, p_wrong = base_pred != y, phobert_pred != y
    fixed, broken, both = b_wrong & ~p_wrong, ~b_wrong & p_wrong, b_wrong & p_wrong
    preds = {"baseline": base_pred, "phobert": phobert_pred}
    base_err = pd.read_csv(RESULTS / "errors_baseline.csv", encoding="utf-8-sig")
    base_conf = pd.Series(base_err.pred_proba.to_numpy(), index=base_err.text)
    # Câu baseline sai với độ tự tin cao nhất lên đầu (sắp xếp cố định để chạy lại ra cùng ví dụ)
    order_by_base = test.index[np.argsort(-test.text.map(base_conf).fillna(0).to_numpy(), kind="stable")]

    def by_true_label(mask):
        return {LABEL_NAMES[c]: int((mask & (y == c)).sum()) for c in range(len(LABEL_NAMES))}

    example_sets = {
        "fixed_with_negation": examples(test, [i for i in order_by_base if fixed[i] and neg_mask[i]], preds, 10),
        "fixed_other": examples(test, [i for i in order_by_base if fixed[i] and not neg_mask[i]], preds, 5),
        "broken": examples(test, [i for i in test.index if broken[i]], preds, 8),
        "both_wrong": examples(test, [i for i in order_by_base if both[i]], preds, 5),
    }
    stats = {
        "both_correct": int((~b_wrong & ~p_wrong).sum()),
        "baseline_wrong_phobert_right": int(fixed.sum()),
        "phobert_wrong_baseline_right": int(broken.sum()),
        "both_wrong": int(both.sum()),
        "both_wrong_same_prediction": int((both & (base_pred == phobert_pred)).sum()),
        "fixed_by_true_label": by_true_label(fixed),
        "broken_by_true_label": by_true_label(broken),
        "negation_subset": {
            "definition": f"câu có ít nhất 1 từ trong {sorted(NEGATION_WORDS)}, trừ 'không khí', 'không gian'",
            "n": int(neg_mask.sum()),
            "baseline_errors": int((neg_mask & b_wrong).sum()), "phobert_errors": int((neg_mask & p_wrong).sum()),
            "fixed": int((neg_mask & fixed).sum()), "broken": int((neg_mask & broken).sum()),
            "both_wrong": int((neg_mask & both).sum()),
        },
        "examples": EXAMPLES_POINTER,
    }
    return stats, example_sets


def stability(test, run_preds, base_pred, chosen):
    y = test.label.to_numpy()
    wrong = {r: p != y for r, p in run_preds.items()}

    def wrong_counts(runs):
        counts = np.sum([wrong[r] for r in runs], axis=0)
        return counts, {str(k): int((counts == k).sum()) for k in range(len(runs) + 1)}

    bal_counts, bal_dist = wrong_counts(RUNS["balanced"])
    all_runs = RUNS["balanced"] + RUNS["none"]
    all_counts, all_dist = wrong_counts(all_runs)
    stable_bal = bal_counts == len(RUNS["balanced"])
    same_pred_bal = stable_bal & np.all([run_preds[r] == run_preds[RUNS["balanced"][0]] for r in RUNS["balanced"]], axis=0)
    stable_all = all_counts == len(all_runs)
    chosen_wrong = wrong[chosen]
    return {
        "errors_per_run": {r: int(wrong[r].sum()) for r in all_runs},
        "balanced_3_runs": {
            "distribution_n_runs_wrong": bal_dist,
            "wrong_in_any": int((bal_counts > 0).sum()), "wrong_in_all_3_stable": int(stable_bal.sum()),
            "stable_with_same_wrong_label": int(same_pred_bal.sum()),
            "wrong_in_only_some": int(((bal_counts > 0) & ~stable_bal).sum()),
            "pairwise_overlap": {f"{a}&{b}": int((wrong[a] & wrong[b]).sum())
                                 for a, b in combinations(RUNS["balanced"], 2)},
        },
        "all_6_runs": {
            "distribution_n_runs_wrong": all_dist,
            "wrong_in_any": int((all_counts > 0).sum()), "wrong_in_all_6_stable": int(stable_all.sum()),
            "wrong_in_only_some": int(((all_counts > 0) & ~stable_all).sum()),
        },
        "wrong_in_all_6_runs_and_baseline": int((stable_all & (base_pred != y)).sum()),
        "chosen_run_errors": {
            "n": int(chosen_wrong.sum()),
            "stable_across_balanced_seeds": int((chosen_wrong & stable_bal).sum()),
            "stable_across_all_6": int((chosen_wrong & stable_all).sum()),
        },
    }, stable_bal


def group_rates(test, phobert_pred, base_pred, neg_mask):
    y = test.label.to_numpy()
    p_wrong, b_wrong = phobert_pred != y, base_pred != y
    n_tokens = test.text.str.split().str.len().to_numpy()
    families = test.text.map(placeholder_families)

    def both_models(mask):
        return {"phobert": rate(mask, p_wrong), "baseline": rate(mask, b_wrong)}

    length = {}
    for lo, hi in LENGTH_BINS:
        mask = (n_tokens >= lo) & ((n_tokens <= hi) if hi else True)
        length[f"{lo}-{hi}" if hi else f"{lo}+"] = both_models(mask)

    placeholder = {fam: both_models(families.map(lambda f, fam=fam: fam in f).to_numpy())
                   for fam, _ in PLACEHOLDER_FAMILIES}
    any_ph = families.map(bool).to_numpy()
    placeholder["any_placeholder"] = both_models(any_ph)
    placeholder["no_placeholder"] = both_models(~any_ph)

    # Câu test gần trùng câu train (bỏ dấu câu, chữ thường): nhãn có giống train không
    train = load_split("train")
    train_labels = train.groupby(normalize_loose(train.text))["label"].agg(set)
    key = normalize_loose(test.text)
    in_train = key.isin(train_labels.index).to_numpy()
    same_label = np.array([in_train[i] and test.label[i] in train_labels[key[i]] for i in test.index])

    return {
        "overall": both_models(np.ones(len(test), dtype=bool)),
        "length_note": "độ dài = số phần tử tách theo dấu cách (gần bằng số âm tiết, dấu câu cũng tính 1), giống EDA",
        "by_length": length,
        "by_placeholder": placeholder,
        "negation": both_models(neg_mask),
        "no_vietnamese_diacritics": {
            "note": "câu không có chữ cái nào mang dấu tiếng Việt, thường là tiếng Anh (heuristic, chưa đọc từng câu)",
            **both_models(~test.text.str.contains(VIETNAMESE_CHARS).to_numpy())},
        "near_duplicate_of_train": {
            "note": "câu test trùng 1 câu train sau khi chữ thường + bỏ dấu câu",
            "same_label_as_train": both_models(in_train & same_label),
            "different_label_from_train": both_models(in_train & ~same_label),
        },
    }


def train_conventions():
    """Nhãn của 2 kiểu câu hay gặp trong lỗi, đếm trên train (không dùng test)."""
    train = load_split("train")
    names = dict(enumerate(LABEL_NAMES))

    def label_counts(mask):
        return {"n": int(mask.sum()), **{k: int(v) for k, v in train[mask].label.map(names).value_counts().items()}}

    n_tokens = train.text.str.split().str.len()
    return {
        "short_thanks_max8_tokens": label_counts(train.text.str.contains(THANKS_PATTERN) & (n_tokens <= 8)),
        "starts_with_suggestion_word": {"pattern": SUGGESTION_START,
                                        **label_counts(train.text.str.match(SUGGESTION_START))},
    }


def manual_categories(errors, stable_bal):
    """Tạo khung file phân loại thủ công nếu chưa có; nếu có thì đọc và tổng hợp."""
    frame = pd.DataFrame({
        "text": errors.text, "true_label": errors.true_label, "pred_label": errors.pred_label,
        "pred_proba": errors.pred_proba, "stable_across_balanced_seeds": stable_bal[errors.idx],
        "category": "", "note": "", "annotation_status": ANNOTATION_STATUS,
    })
    if not CATEGORIES_CSV.exists():
        frame.to_csv(CATEGORIES_CSV, index=False, encoding="utf-8-sig")
        print(f"Đã tạo khung {CATEGORIES_CSV.name}: điền cột category và note rồi chạy lại")
        return None

    cat = pd.read_csv(CATEGORIES_CSV, encoding="utf-8-sig", keep_default_na=False)
    if set(cat.text) != set(frame.text) or len(cat) != len(frame):
        raise ValueError(f"{CATEGORIES_CSV.name} không khớp danh sách câu sai của run đã chọn")
    merged = frame.drop(columns=["category", "note"]).merge(cat[["text", "category", "note"]], on="text")
    if (merged.category == "").any():
        print(f"Cảnh báo: {(merged.category == '').sum()} câu chưa có category")
    merged["pair"] = merged.true_label + "→" + merged.pred_label
    by_cat = merged.groupby("category").agg(count=("text", "size"),
                                            stable_across_balanced_seeds=("stable_across_balanced_seeds", "sum"))
    by_cat = by_cat.sort_values("count", ascending=False)
    pair_table = pd.crosstab(merged.category, merged.pair)
    pair_order = merged.pair.value_counts().index
    unknown = set(merged.category) - set(CATEGORY_DEFINITIONS) - {""}
    if unknown:
        print(f"Cảnh báo: category không có trong CATEGORY_DEFINITIONS: {unknown}")
    return {
        "status": ANNOTATION_STATUS + ". Mỗi câu sai được gán đúng 1 nhóm chính theo cách đọc của người gán; "
                  "nhóm 'nhãn có vẻ không nhất quán' chỉ là nghi vấn, không khẳng định nhãn gốc sai. "
                  "Cần chủ dự án đọc lại một phần trước khi dùng các con số này.",
        "file": CATEGORIES_CSV.name,
        "definitions": CATEGORY_DEFINITIONS,
        "n": int(len(merged)),
        "by_category": {c: {"count": int(r["count"]), "pct": round(r["count"] / len(merged), 4),
                            "stable_across_balanced_seeds": int(r["stable_across_balanced_seeds"])}
                        for c, r in by_cat.iterrows()},
        "by_category_and_pair": {c: {p: int(v) for p, v in row.items() if v}
                                 for c, row in pair_table.loc[by_cat.index].iterrows()},
        "by_pair_and_category": {p: {c: int(v) for c, v in pair_table[p].sort_values(ascending=False).items() if v}
                                 for p in pair_order},
    }


def load_saved_cm(run_name):
    with open(RESULTS / f"phobert_{run_name}.json", encoding="utf-8") as f:
        return json.load(f)["test"]["confusion_matrix"]


def main():
    with open(RESULTS / "phobert_summary.json", encoding="utf-8") as f:
        chosen = json.load(f)["chosen_run"]
    with open(RESULTS / "baseline.json", encoding="utf-8") as f:
        baseline_cm = json.load(f)["test"]["confusion_matrix"]

    test = load_test()
    run_preds, run_errors = {}, {}
    for run in RUNS["balanced"] + RUNS["none"]:
        run_preds[run], run_errors[run] = load_predictions(
            RESULTS / f"errors_phobert_{run}.csv", test, load_saved_cm(run))
    base_pred, _ = load_predictions(RESULTS / "errors_baseline.csv", test, baseline_cm)
    neg_mask = test.text.map(has_negation).to_numpy()

    stab, stable_bal = stability(test, run_preds, base_pred, chosen)
    bv_stats, bv_examples = baseline_vs_chosen(test, base_pred, run_preds[chosen], neg_mask)
    out = {
        "chosen_run": chosen,
        "note": "Mọi số liệu ở đây tính lại từ results/errors_*.csv (câu sai) và nhãn test; confusion matrix "
                "dựng lại đã được kiểm tra khớp với bản lưu trong JSON của từng run. File này chỉ chứa số liệu; "
                "câu ví dụ của bộ dữ liệu: " + EXAMPLES_POINTER,
        "chosen_run_errors": chosen_run_section(test, run_preds[chosen], run_errors[chosen]),
        "baseline_vs_chosen": bv_stats,
        "stability_across_seeds": stab,
        "error_rate_by_group": group_rates(test, run_preds[chosen], base_pred, neg_mask),
        "label_conventions_in_train": train_conventions(),
        "manual_categorization": manual_categories(run_errors[chosen], stable_bal),
    }
    with open(RESULTS / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(EXAMPLES_JSON, "w", encoding="utf-8") as f:
        json.dump({"note": "Câu gốc của UIT-VSFC (giấy phép chưa rõ): chỉ dùng trên máy, không commit.",
                   "chosen_run": chosen, "baseline_vs_chosen": bv_examples}, f, ensure_ascii=False, indent=2)

    c, bv, s = out["chosen_run_errors"], out["baseline_vs_chosen"], out["stability_across_seeds"]
    print(f"{chosen}: {c['n_errors']}/{c['n_test']} câu sai ({c['error_rate']:.2%})")
    for p in c["confusion_pairs"]:
        print(f"  {p['true']:>8} → {p['pred']:<8} {p['count']:>4} ({p['pct_of_errors']:.0%})")
    n = c["neutral"]
    print(f"  neutral: precision {n['precision']:.3f}, recall {n['recall']:.3f} ({n['correct']}/{n['support']})")
    print(f"Baseline sai → PhoBERT đúng: {bv['baseline_wrong_phobert_right']} | PhoBERT sai → baseline đúng: "
          f"{bv['phobert_wrong_baseline_right']} | cả hai sai: {bv['both_wrong']}")
    b3 = s["balanced_3_runs"]
    print(f"3 seed balanced: sai ở cả 3 = {b3['wrong_in_all_3_stable']}, chỉ sai ở 1–2 run = {b3['wrong_in_only_some']}"
          f" | 6 run: sai ở cả 6 = {s['all_6_runs']['wrong_in_all_6_stable']}")
    print("Tỉ lệ sai theo độ dài (PhoBERT):")
    for b, r in out["error_rate_by_group"]["by_length"].items():
        print(f"  {b:>6}: {r['phobert']['n_errors']:>3}/{r['phobert']['n']:<5} = {r['phobert']['error_rate']:.2%}")
    m = out["manual_categorization"]
    if m:
        print(f"Phân loại thủ công ({ANNOTATION_STATUS}):")
        for cat, r in m["by_category"].items():
            print(f"  {r['count']:>3} ({r['pct']:.0%}) {cat}")
    print(f"Đã lưu {RESULTS / 'error_analysis.json'} (chỉ số liệu) và {EXAMPLES_JSON.name} (câu ví dụ, không commit)")


if __name__ == "__main__":
    main()
