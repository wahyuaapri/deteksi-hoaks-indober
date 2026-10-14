"""
Modul Preprocessing Teks Berita untuk Deteksi Hoaks
Mencakup:
1. Pembersihan teks (URL, Mention, Simbol, Spasi berlebih)
2. Case folding (Konversi ke huruf kecil)
3. Pembagian dataset (Train, Validation, Test) dengan Stratified Split
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Membersihkan teks dari URL, tanda baca berlebih, dan karakter non-alfabet."""
    if not isinstance(text, str):
        return ""
    
    # Konversi ke huruf kecil
    text = text.lower()
    
    # Hapus URL (http/https/www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Hapus mention (@username) dan hashtag (#tag)
    text = re.sub(r'[@#]\S+', '', text)
    
    # Hapus karakter selain huruf alfabet dan spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def prepare_and_split_data(
    raw_csv_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """Membaca data mentah, membersihkan, membagi ke train/val/test, dan menyimpan hasilnya."""
    print(f"[*] Membaca data dari: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # Validasi kolom
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset wajib memiliki kolom 'text' dan 'label'!")

    print(f"[*] Total baris data mentah: {len(df)}")
    
    # Hapus duplikat dan baris kosong
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    
    print("[*] Melakukan pembersihan teks (cleaning & case folding)...")
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    # Filter baris yang menjadi kosong setelah cleaning
    df = df[df["cleaned_text"].str.strip() != ""]

    print(f"[*] Total baris setelah cleaning: {len(df)}")
    print(f"[*] Distribusi kelas:\n{df['label'].value_counts()}")

    # Stratified Split: Train + (Val + Test)
    total_test_val = test_size + val_size
    train_df, temp_df = train_test_split(
        df,
        test_size=total_test_val,
        stratify=df["label"] if df["label"].value_counts().min() >= 2 else None,
        random_state=random_state
    )

    # Split temp_df menjadi Validation dan Test
    relative_val_ratio = val_size / total_test_val
    stratify_temp = temp_df["label"] if temp_df["label"].value_counts().min() >= 2 else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_ratio),
        stratify=stratify_temp,
        random_state=random_state
    )

    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Data tersimpan di '{output_dir}':")
    print(f"    - Train : {len(train_df)} sampel -> {train_path}")
    print(f"    - Val   : {len(val_df)} sampel -> {val_path}")
    print(f"    - Test  : {len(test_df)} sampel -> {test_path}")

    return train_path, val_path, test_path


if __name__ == "__main__":
    # Test menjalankan preprocessing
    raw_path = "data/raw/sample_hoax_dataset.csv"
    if os.path.exists(raw_path):
        prepare_and_split_data(raw_path)
    else:
        print(f"[!] File {raw_path} tidak ditemukan.")
