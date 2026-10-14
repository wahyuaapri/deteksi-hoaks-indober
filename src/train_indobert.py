"""
Fine-Tuning IndoBERT untuk Klasifikasi Berita Hoaks
Model pre-trained: indobenchmark/indobert-base-p1
Framework: PyTorch & Hugging Face Transformers
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Pastikan folder src terbaca saat dieksekusi dari root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

from evaluate import calculate_metrics, save_confusion_matrix
from preprocessing import prepare_and_split_data


class HoaxDataset(Dataset):
    """Custom PyTorch Dataset untuk tokenisasi batch."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_indobert(
    model_name: str = "indobenchmark/indobert-base-p1",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Menggunakan perangkat komputasi: {device}")

    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"

    # Prioritaskan dataset riil 12k jika sudah ada
    raw_12k = "data/raw/indonesian_hoax_news_12k.csv"
    raw_sample = "data/raw/sample_hoax_dataset.csv"
    chosen_raw = raw_12k if os.path.exists(raw_12k) else raw_sample

    print(f"[*] Menyiapkan dan memproses dataset: {chosen_raw}...")
    prepare_and_split_data(chosen_raw)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"[*] Ukuran data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Inisialisasi Tokenizer & Model
    print(f"[*] Mengunduh / memuat pre-trained model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Buat DataLoader
    train_dataset = HoaxDataset(train_df["cleaned_text"], train_df["label"], tokenizer, max_length)
    val_dataset = HoaxDataset(val_df["cleaned_text"], val_df["label"], tokenizer, max_length)
    test_dataset = HoaxDataset(test_df["cleaned_text"], test_df["label"], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    save_dir = "models/indobert_hoax"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("           MEMULAI FINE-TUNING INDOBERT")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Evaluasi Validasi
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Simpan checkpoint terbaik
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[*] Checkpoint terbaik disimpan di {save_dir}")

    # Pengujian pada Data Test
    print("\n[*] Menguji performa model IndoBERT pada data uji (Test Set)...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    calculate_metrics(all_labels, all_preds)
    save_confusion_matrix(
        all_labels,
        all_preds,
        output_path="reports/confusion_matrix_indobert.png",
        title="Confusion Matrix - IndoBERT"
    )
    print(f"\n[OK] Proses Fine-Tuning Selesai. Model tersimpan di '{save_dir}'.")


if __name__ == "__main__":
    train_indobert()
