"""Bước 1: tải bộ UIT-VSFC (bản Parquet) và đọc thành DataFrame.

Vì sao không gọi `datasets.load_dataset("uitnlp/vietnamese_students_feedback")`?
Repo gốc chỉ có một loading script (.py). Thư viện `datasets` từ bản 4.0 không chạy
script nữa (chạy code lạ tải từ internet không an toàn). Hugging Face tự chuyển bộ dữ liệu
sang Parquet, là file chỉ chứa dữ liệu dạng bảng nén theo cột, nên ta đọc thẳng file đó.
"""

from pathlib import Path
import urllib.request

import pandas as pd

BASE_URL = "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default"
SPLITS = ("train", "validation", "test")
LABEL_NAMES = ["negative", "neutral", "positive"]  # cột sentiment: 0, 1, 2 (theo schema của dataset)

# Đường dẫn tính từ vị trí file này, nên chạy lệnh từ thư mục nào cũng đúng
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_split(split: str) -> Path:
    """Tải một tập về data/raw/. Lần sau thấy file đã có thì không tải lại (cache)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{split}.parquet"
    if not path.exists():
        url = f"{BASE_URL}/{split}/0.parquet"
        print(f"Đang tải {split}: {url}")
        urllib.request.urlretrieve(url, path)  # tự đi theo redirect sang CDN
    return path


def load_split(split: str) -> pd.DataFrame:
    """Trả về DataFrame 2 cột: text (câu phản hồi) và label (0/1/2)."""
    df = pd.read_parquet(download_split(split))
    df = df.rename(columns={"sentence": "text", "sentiment": "label"})
    return df[["text", "label"]]  # bỏ cột topic vì project chỉ làm nhãn cảm xúc


if __name__ == "__main__":
    for split in SPLITS:
        df = load_split(split)
        first = df.iloc[0]
        print(f"{split:<10} {len(df):>6} câu | ví dụ: {first.text!r} -> {LABEL_NAMES[first.label]}")
