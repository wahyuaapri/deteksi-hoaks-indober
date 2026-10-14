"""
Skrip Otomatis Pengunduh Dataset Berita Hoaks Indonesia
Sumber: Indonesian Hoax Detection Dataset (TurnBackHoax & Portal Berita Nasional)
Total Data: ~12.150 Berita (Seimbang antara Fakta dan Hoaks)
"""

import os
import urllib.request
import pandas as pd
from tqdm import tqdm


DATASET_URL = "https://huggingface.co/datasets/mzuama/IndonesianHoaxdetection/resolve/main/datasetnew.csv"
RAW_DIR = "data/raw"
OUTPUT_FILE = os.path.join(RAW_DIR, "indonesian_hoax_news_12k.csv")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_prepare_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("      PENGUNDUH DATASET BERITA HOAKS INDONESIA (12.000+ DATA)")
    print("=" * 60)
    print(f"[*] Mengunduh dataset dari Hugging Face Hub...")
    print(f"[*] URL: {DATASET_URL}")

    temp_download_path = os.path.join(RAW_DIR, "temp_download.csv")

    try:
        # Request dengan User-Agent agar tidak dibatasi Hugging Face
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 65536  # 64KB chunks
            
            with open(temp_download_path, "wb") as out_file, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Mengunduh Dataset"
            ) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))

        print("\n[*] Mengonversi dan menstandardisasi format dataset...")
        df = pd.read_csv(temp_download_path)

        # Standarisasi nama kolom ke 'text' dan 'label'
        # Dataset asli memiliki kolom 'Label' dan 'Text'
        rename_map = {}
        for col in df.columns:
            if col.lower() == "text":
                rename_map[col] = "text"
            elif col.lower() == "label":
                rename_map[col] = "label"

        df = df.rename(columns=rename_map)
        
        # Validasi kolom wajib
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)

        # Simpan ke file akhir
        df.to_csv(OUTPUT_FILE, index=False)

        # Hapus temp file
        if os.path.exists(temp_download_path):
            os.remove(temp_download_path)

        print("\n" + "=" * 60)
        print("                 DATASET BERHASIL DIUNDUH")
        print("=" * 60)
        print(f"[OK] File tersimpan di: {OUTPUT_FILE}")
        print(f"[OK] Total baris data : {len(df):,}")
        print(f"[OK] Distribusi Label :")
        print(f"     - Label 0 (Fakta / Valid) : {(df['label'] == 0).sum():,} berita")
        print(f"     - Label 1 (Hoaks / Palsu) : {(df['label'] == 1).sum():,} berita")
        print("=" * 60)

        return OUTPUT_FILE

    except Exception as e:
        print(f"\n[!] Terjadi kesalahan saat mengunduh: {e}")
        if os.path.exists(temp_download_path):
            os.remove(temp_download_path)
        return None


if __name__ == "__main__":
    download_and_prepare_dataset()
