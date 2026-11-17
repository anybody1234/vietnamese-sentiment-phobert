"""Bước 4: tách từ, Dataset và hàm gom batch (collate) cho PhoBERT.

PhoBERT được pretrain trên văn bản đã tách từ: các âm tiết của một từ nối bằng "_"
(vd "giảng_viên"), nên câu đưa vào lúc fine-tune cũng phải ở đúng dạng đó.
"""

import torch
from torch.utils.data import Dataset
from underthesea import word_tokenize


def take_subset(df, n, seed):
    """Chỉ dùng khi chạy thử (--subset). Lấy mẫu phân tầng khoảng n câu: mỗi lớp giữ tỉ lệ cũ,
    để tập nhỏ vẫn có đủ cả 3 lớp."""
    return df.groupby("label").sample(frac=min(1.0, n / len(df)), random_state=seed).reset_index(drop=True)


def segment(texts):
    """Tách từ từng câu: "giảng viên dạy hay" -> "giảng_viên dạy hay".
    format="text" trả về một chuỗi (từ nhiều âm tiết nối bằng "_"), đúng dạng tokenizer PhoBERT cần."""
    return [word_tokenize(text, format="text") for text in texts]


class SentimentDataset(Dataset):
    """Giữ câu đã tách từ và nhãn. Không tokenize sẵn ở đây: để Collator tokenize theo từng batch,
    nhờ vậy mỗi batch chỉ pad tới câu dài nhất của chính batch đó."""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]  # int thuần để torch.tensor nhận ngay

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class Collator:
    """DataLoader gọi đối tượng này với một list (câu, nhãn) để ghép thành một batch tensor.

    Dynamic padding: pad tới câu dài nhất trong batch chứ không pad cố định tới max_len.
    Câu trung vị chỉ khoảng 11 token, nên pad cố định tới 128 làm phần lớn phép tính là trên token rỗng.
    Viết thành class (thay vì hàm lồng) để pickle được nếu sau này dùng num_workers > 0.
    """

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts, labels = zip(*batch)
        enc = self.tokenizer(
            list(texts),
            truncation=True,          # câu dài hơn max_len token (tính cả <s> và </s>) bị cắt phần đuôi
            max_length=self.max_len,
            padding="longest",        # dynamic padding
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],  # 1 = token thật, 0 = pad: attention bỏ qua vị trí pad
            "labels": torch.tensor(labels, dtype=torch.long),  # CrossEntropyLoss cần nhãn kiểu long
        }
