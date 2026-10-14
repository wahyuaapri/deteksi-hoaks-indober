"""
Modul Evaluasi Model Klasifikasi Teks
Mencakup:
1. Perhitungan metrik: Accuracy, Precision, Recall, Macro F1-Score
2. Ekspor Classification Report
3. Visualisasi & Penyimpanan Confusion Matrix (Seaborn / Matplotlib)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, target_names=("Fakta (0)", "Hoaks (1)")):
    """Menghitung metrik performa klasifikasi secara komprehensif."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "=" * 55)
    print("                 HASIL EVALUASI MODEL")
    print("=" * 55)
    print(f"Accuracy        : {acc * 100:.2f}%")
    print(f"Macro Precision : {prec_macro * 100:.2f}%")
    print(f"Macro Recall    : {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score  : {f1_macro * 100:.2f}%")
    print("-" * 55)
    print(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")
    print("=" * 55)
    
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    print("\nDetail Per Kelas:")
    print(report)

    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "report": report
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path: str = "reports/confusion_matrix.png",
    class_labels=("Fakta (0)", "Hoaks (1)"),
    title: str = "Confusion Matrix"
):
    """Membuat dan menyimpan plot Confusion Matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=False
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Prediksi Model", fontsize=11)
    plt.ylabel("Label Sebenarnya", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"[OK] Confusion matrix tersimpan di: {output_path}")
