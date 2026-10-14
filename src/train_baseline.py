"""
Pelatihan Model Baseline: TF-IDF + Support Vector Machine (SVM)
Digunakan sebagai tolak ukur (benchmark) performa sebelum dibandingkan dengan IndoBERT.
"""

import os
import sys
import pandas as pd
import joblib

# Pastikan folder src terbaca saat dieksekusi dari root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from preprocessing import prepare_and_split_data
from evaluate import calculate_metrics, save_confusion_matrix


def train_baseline():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    # Prioritaskan dataset riil 12k jika sudah ada
    raw_12k = "data/raw/indonesian_hoax_news_12k.csv"
    raw_sample = "data/raw/sample_hoax_dataset.csv"
    chosen_raw = raw_12k if os.path.exists(raw_12k) else raw_sample

    print(f"[*] Menyiapkan dan memproses dataset: {chosen_raw}...")
    prepare_and_split_data(chosen_raw)

    print("[*] Memuat data train dan test...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["cleaned_text"].astype(str)
    y_train = train_df["label"]

    X_test = test_df["cleaned_text"].astype(str)
    y_test = test_df["label"]

    print(f"[*] Jumlah data latih : {len(X_train)}")
    print(f"[*] Jumlah data uji   : {len(X_test)}")

    # Pipeline: TF-IDF (Unigram & Bigram) + Linear Support Vector Classifier
    print("[*] Membangun pipeline TF-IDF + LinearSVC...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", LinearSVC(
            C=1.0,
            random_state=42,
            dual="auto"
        ))
    ])

    print("[*] Melatih model baseline...")
    pipeline.fit(X_train, y_train)

    print("[*] Melakukan prediksi pada data uji...")
    y_pred = pipeline.predict(X_test)

    # Evaluasi
    calculate_metrics(y_test, y_pred)
    save_confusion_matrix(
        y_test,
        y_pred,
        output_path="reports/confusion_matrix_baseline.png",
        title="Confusion Matrix - Baseline (TF-IDF + SVM)"
    )

    # Simpan model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/baseline_svm.joblib"
    joblib.dump(pipeline, model_save_path)
    print(f"[OK] Model baseline berhasil disimpan di: {model_save_path}")


if __name__ == "__main__":
    train_baseline()
