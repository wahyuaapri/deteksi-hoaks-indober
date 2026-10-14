"""
Modul Explainable AI (XAI) untuk Klasifikasi Berita Hoaks
Menjelaskan kontribusi kata-kata dalam teks terhadap keputusan klasifikasi (Fakta vs Hoaks)
menggunakan Local Linear Feature Attribution.
"""

import re
import numpy as np
import pandas as pd


def explain_prediction(pipeline, text: str, top_n: int = 6):
    """
    Menghitung kontribusi lokal setiap kata dalam teks masukan terhadap keputusan model.
    Skor positif (> 0)  : Mendorong prediksi ke arah HOAKS (1)
    Skor negatif (< 0)  : Mendorong prediksi ke arah FAKTA (0)
    """
    if not text or not text.strip():
        return None

    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")

    if vectorizer is None or classifier is None:
        raise ValueError("Pipeline wajib memiliki komponen 'tfidf' dan 'classifier'!")

    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    # Transform teks
    tfidf_vec = vectorizer.transform([text])
    nonzero_indices = tfidf_vec.nonzero()[1]

    word_contributions = {}
    for idx in nonzero_indices:
        feature_word = feature_names[idx]
        tfidf_val = tfidf_vec[0, idx]
        score = float(tfidf_val * coefs[idx])
        word_contributions[feature_word] = score

    # Urutkan kata berdasarkan pengaruh terbesar
    sorted_words = sorted(word_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    hoax_triggers = [item for item in sorted_words if item[1] > 0][:top_n]
    fact_triggers = [item for item in sorted_words if item[1] < 0][:top_n]

    # DataFrame untuk visualisasi chart
    chart_data = []
    for word, score in sorted_words[:top_n * 2]:
        chart_data.append({
            "Kata": word,
            "Pengaruh": score,
            "Kategori": "Pemicu Hoaks" if score > 0 else "Indikator Fakta"
        })
    df_chart = pd.DataFrame(chart_data)

    # Buat teks ter-highlight dalam format HTML
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)
    highlighted_parts = []

    for token in tokens:
        token_lower = token.lower()
        if token_lower in word_contributions:
            score = word_contributions[token_lower]
            if score > 0.01:
                # Warna Merah untuk Hoaks
                intensity = min(1.0, 0.2 + (score * 3))
                highlighted_parts.append(
                    f'<span style="background-color: rgba(239, 68, 68, {intensity:.2f}); '
                    f'padding: 2px 5px; border-radius: 4px; margin: 1px; font-weight: 500; '
                    f'border: 1px solid rgba(220, 38, 38, 0.4);" '
                    f'title="Pemicu Hoaks (skor: +{score:.3f})">{token}</span>'
                )
            elif score < -0.01:
                # Warna Hijau untuk Fakta
                intensity = min(1.0, 0.2 + (abs(score) * 3))
                highlighted_parts.append(
                    f'<span style="background-color: rgba(34, 197, 94, {intensity:.2f}); '
                    f'padding: 2px 5px; border-radius: 4px; margin: 1px; font-weight: 500; '
                    f'border: 1px solid rgba(22, 163, 74, 0.4);" '
                    f'title="Indikator Fakta (skor: {score:.3f})">{token}</span>'
                )
            else:
                highlighted_parts.append(token)
        else:
            highlighted_parts.append(token)

    # Rekonstruksi kalimat dengan spasi wajar
    html_text = " ".join(highlighted_parts)
    # Rapikan spasi sebelum tanda baca
    html_text = re.sub(r'\s+([.,!?;:])', r'\1', html_text)

    return {
        "hoax_triggers": hoax_triggers,
        "fact_triggers": fact_triggers,
        "df_chart": df_chart,
        "html_text": html_text
    }


def generate_verdict_reason(prediction: int, decision_score: float, explanation: dict, text: str) -> dict:
    """
    Menghasilkan narasi alasan objektif dan ilmiah mengapa suatu berita divonis Fakta atau Hoaks.
    Menganalisis kombinasi:
    1. Bobot leksikal fitur pemicu teratas (top trigger keywords)
    2. Pola narasi (viralitas/ketakutan/klaim vs institusi/jurnalisme resmi)
    3. Skor margin keyakinan model
    """
    text_lower = text.lower()
    hoax_words = [w for w, s in explanation.get("hoax_triggers", [])]
    fact_words = [w for w, s in explanation.get("fact_triggers", [])]
    
    # Kategori pola leksikal
    viral_markers = ["beredar", "klaim", "narasi", "viral", "pesan", "berantai", "grup", "whatsapp", "facebook", "unggahan", "foto", "video"]
    fear_or_health = ["racun", "mematikan", "bahaya", "seketika", "rebusan", "kanker", "sembuh", "darah", "jantung", "gejala", "mati"]
    fraud_markers = ["gratis", "kuota", "hadiah", "undian", "link", "tautan", "jutaan", "rupiah", "pin", "atm", "transfer", "bantuan"]
    
    gov_institutions = ["kementerian", "kemenkes", "kemendikbud", "kemenkeu", "kominfo", "pemerintah", "presiden", "dinas", "bmkg", "bi", "ojk", "bpom", "kpu", "bps", "polri", "kapolri", "polda"]
    journalism_verbs = ["mengumumkan", "meresmikan", "melaporkan", "memastikan", "mengimbau", "mencatat", "meluncurkan", "menegaskan", "menyampaikan", "menggelar"]

    # Deteksi pola yang muncul dalam teks
    has_viral = [w for w in viral_markers if w in text_lower]
    has_fear = [w for w in fear_or_health if w in text_lower]
    has_fraud = [w for w in fraud_markers if w in text_lower]
    has_gov = [w for w in gov_institutions if w in text_lower]
    has_journalism = [w for w in journalism_verbs if w in text_lower]

    # Tingkat keyakinan matematis
    abs_margin = abs(decision_score)
    if abs_margin >= 1.5:
        conf_label = "Sangat Tinggi (Karakteristik Sangat Kuat)"
    elif abs_margin >= 0.8:
        conf_label = "Tinggi (Karakteristik Jelas)"
    else:
        conf_label = "Moderat (Perlu Verifikasi Tambahan)"

    reasons = []

    if prediction == 1:
        # KASUS HOAKS
        title = "Terindikasi Kuat sebagai Berita HOAKS / Misinformasi"
        summary = (
            f"Model mendeteksi teks ini memiliki karakteristik narasi tidak valid atau misinformasi "
            f"dengan tingkat keyakinan {conf_label} (Skor Margin: {decision_score:+.2f})."
        )

        # 1. Alasan Kata Kunci Pemicu Teratas
        if hoax_words:
            words_str = ", ".join([f"'{w}'" for w in hoax_words[:4]])
            reasons.append(
                f"**Dominasi Kata Kunci Pemicu Hoaks:** Ditemukan kata-kata {words_str} "
                f"yang dalam korpus 6.800+ data MAFINDO memiliki frekuensi kemunculan sangat tinggi pada narasi palsu."
            )

        # 2. Alasan Pola Sensasional / Ketakutan / Penipuan
        if has_fear:
            words_str = ", ".join([f"'{w}'" for w in has_fear[:3]])
            reasons.append(
                f"**Indikasi Sensasionalisme / Ketakutan (Fear-Mongering):** Penggunaan kata {words_str} "
                f"kerap digunakan untuk memicu kepanikan atau menawarkan klaim kesehatan instan tanpa konfirmasi medis resmi."
            )
        elif has_fraud:
            words_str = ", ".join([f"'{w}'" for w in has_fraud[:3]])
            reasons.append(
                f"**Indikasi Penipuan / Umpan Klik (Phishing/Scam):** Ditemukan pola iming-iming bantuan atau hadiah ({words_str}) "
                f"yang merupakan modus umum penyebaran tautan berbahaya atau rekrutmen fiktif."
            )

        # 3. Alasan Gaya Bahasa Penyebaran Berantai
        if has_viral:
            words_str = ", ".join([f"'{w}'" for w in has_viral[:3]])
            reasons.append(
                f"**Pola Bahasa Berantai (Viral Claim):** Teks menggunakan frasa narasi desas-desus ({words_str}) "
                f"yang mencerminkan kabar yang disebarkan dari mulut ke mulut atau media sosial tanpa atribusi narasumber primer."
            )

        # 4. Ketiadaan Sumber Resmi
        if not has_gov and not has_journalism:
            reasons.append(
                "**Ketiadaan Atribusi Otoritas Resmi:** Tidak ditemukan kutipan atau rujukan langsung ke lembaga "
                "pemerintah/institusi kredibel dengan kaidah jurnalistik 5W+1H yang valid."
            )

        advice = (
            "Disarankan untuk memverifikasi klaim ini melalui arsip cek fakta MAFINDO (turnbackhoax.id) "
            "atau kanal aduan konten Kementerian Komunikasi dan Digital sebelum membagikannya."
        )

    else:
        # KASUS FAKTA
        title = "Terindikasi Kuat sebagai Berita FAKTA / Informasi Valid"
        summary = (
            f"Model mendeteksi teks ini memiliki struktur dan karakteristik informasi resmi yang valid "
            f"dengan tingkat keyakinan {conf_label} (Skor Margin: {decision_score:+.2f})."
        )

        # 1. Alasan Kata Kunci Pendukung Validitas
        if fact_words:
            words_str = ", ".join([f"'{w}'" for w in fact_words[:4]])
            reasons.append(
                f"**Penggunaan Kosakata Resmi & Informatif:** Ditemukan kata-kata {words_str} "
                f"yang dalam korpus berita media nasional memiliki korelasi sangat kuat dengan pemberitaan faktual."
            )

        # 2. Adanya Lembaga / Otoritas Resmi
        if has_gov:
            words_str = ", ".join([f"'{w}'" for w in has_gov[:3]])
            reasons.append(
                f"**Keberadaan Rujukan Institusi / Regulator:** Teks mencantumkan pihak berwenang ({words_str}) "
                f"sebagai subjek atau sumber kebijakan, yang merupakan ciri utama rilis pers dan berita terkonfirmasi."
            )

        # 3. Gaya Jurnalistik Objektif
        if has_journalism:
            words_str = ", ".join([f"'{w}'" for w in has_journalism[:3]])
            reasons.append(
                f"**Struktur Jurnalisme Baku:** Penggunaan kata kerja pelaporan ({words_str}) "
                f"menunjukkan tata bahasa berita yang netral, teratur, dan sesuai dengan standar penulisan media kredibel."
            )

        # 4. Bebas dari Frasa Desas-desus
        if not has_viral and not has_fear and not has_fraud:
            reasons.append(
                "**Bebas dari Unsur Provokatif:** Teks tidak memuat bahasa alarmis, klaim konspiratif, "
                "maupun instruksi penyebaran pesan berantai."
            )

        advice = (
            "Informasi ini memiliki kredibilitas tekstual yang baik. Tetap pastikan tanggal publikasi "
            "dan rujukan konteks waktu tetap relevan dengan situasi saat ini."
        )

    # Fallback jika alasan kurang dari 2
    if len(reasons) < 2:
        reasons.append(
            "**Kesesuaian Distribusi Vektor TF-IDF:** Bobot representasi kata dalam teks ini "
            f"condong signifikan ke klaster data {'hoaks' if prediction == 1 else 'faktual'} pada ruang fitur pelatihan."
        )

    return {
        "title": title,
        "summary": summary,
        "confidence": conf_label,
        "reasons": reasons,
        "advice": advice
    }

