# Proyek Skripsi: Deteksi Berita Hoaks Berbahasa Indonesia

Proyek penelitian tugas akhir (skripsi) untuk klasifikasi dan deteksi berita hoaks berbahasa Indonesia menggunakan komparasi model **Pre-trained Language Model (IndoBERT)** dan **Machine Learning Baseline (Support Vector Machine / Logistic Regression)**.

---

## 📁 Struktur Direktori

```text
bismillah skripsi/
├── data/
│   ├── raw/                 # Dataset mentah (CSV dari MAFINDO / Kaggle)
│   └── processed/           # Dataset hasil preprocessing & pembagian train/val/test
├── models/                  # Bobot model tersimpan (.pt / .joblib)
├── notebooks/               # Eksperimen Google Colab / Jupyter
├── reports/                 # Draf naskah skripsi (Bab 1, Bab 2, Bab 3, dll.)
│   └── PROPOSAL_BAB_1.md    # Draf usulan Bab 1 (Latar Belakang s/d Manfaat)
├── src/                     # Source code modular
│   ├── __init__.py
│   ├── preprocessing.py     # Pembersihan teks & tokenisasi
│   ├── train_baseline.py    # Training model baseline (TF-IDF + SVM)
│   ├── train_indobert.py    # Fine-tuning IndoBERT (Transformers + PyTorch)
│   ├── evaluate.py          # Metrik evaluasi, Confusion Matrix, ROC-AUC
│   ├── explainability.py    # Explainable AI (XAI) Feature Attribution
│   ├── url_scraper.py       # Live Web Scraper untuk Tautan/Link Berita
│   └── download_dataset.py  # Pengunduh dataset 12.000+ berita hoaks
├── app.py                   # Demo Web Interaktif (URL Scraper + XAI Visualizer)
└── README.md                # Dokumentasi proyek
```

---

## 🎯 Target Eksperimen & Komparasi

| Model | Ekstraksi Fitur / Representasi | Arsitektur | Keterangan |
| :--- | :--- | :--- | :--- |
| **Baseline** | TF-IDF (N-gram 1-2) | Support Vector Machine (Linear/RBF) | Pembanding metode klasik |
| **Proposed** | Contextual Word Embedding | Fine-Tuned IndoBERT (`indobenchmark/indobert-base-p1`) | Model utama berbasis Transformer |

---

## 🚀 Panduan Memulai

### 1. Persiapan Environment
Gunakan virtual environment Python (disarankan versi 3.10 atau 3.11):
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Baseline (SVM)
```bash
python src/train_baseline.py
```

### 3. Menjalankan Fine-Tuning IndoBERT
*(Disarankan menggunakan GPU atau upload ke Google Colab jika menggunakan dataset ribuan data)*
```bash
python src/train_indobert.py
```
