"""
Aplikasi Demo Interaktif Deteksi Berita Hoaks Berbahasa Indonesia
Fitur:
1. Akses Internet Langsung (Ekstraksi Konten via URL/Link Berita)
2. Input Teks Manual
3. Klasifikasi NLP Biner (Fakta vs Hoaks)
4. Transparansi Keputusan Berbasis Explainable AI (XAI)
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# Pastikan folder src dapat diakses
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from explainability import explain_prediction, generate_verdict_reason
from url_scraper import extract_article_from_url

st.set_page_config(
    page_title="Deteksi Berita Hoaks & URL Scraper",
    page_icon="🛡️",
    layout="wide"
)

# Header & Informasi
st.title("🛡️ Deteksi Berita Hoaks Berbahasa Indonesia")
st.caption("Sistem Klasifikasi Berita & Akses Internet Langsung Disertai **Explainable AI (XAI)**")

MODEL_PATH = "models/baseline_svm.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

if pipeline is None:
    st.warning("⚠️ Model belum ditemukan! Jalankan `python src/train_baseline.py` terlebih dahulu.")
    st.stop()


def render_prediction_and_xai(text_to_analyze: str):
    """Fungsi pembantu untuk memproses klasifikasi dan Explainable AI."""
    # 1. Prediksi Kelas
    prediction = pipeline.predict([text_to_analyze])[0]
    
    # Skor Margin / Keputusan
    if hasattr(pipeline.named_steps["classifier"], "decision_function"):
        decision = float(pipeline.decision_function([text_to_analyze])[0])
    else:
        decision = 0.0

    st.markdown("---")
    
    # 2. Tampilkan Hasil Prediksi
    res_col1, res_col2 = st.columns([6, 4])

    with res_col1:
        if prediction == 1:
            st.error("### 🚨 HASIL: BERPOTENSI HOAKS (PALSU)")
            st.markdown(
                "Teks ini terdeteksi memiliki karakteristik narasi tidak valid, klaim berlebihan, "
                "atau misinformasi berdasarkan analisis pola korpus MAFINDO."
            )
        else:
            st.success("### 🟢 HASIL: BERITA FAKTA (VALID)")
            st.markdown(
                "Teks ini terindikasi merupakan informasi kredibel atau bersumber resmi "
                "berdasarkan korpus berita nasional terverifikasi."
            )

    with res_col2:
        st.metric(
            label="Skor Keputusan Model (Margin Score)",
            value=f"{decision:+.3f}",
            help="Skor positif menunjukkan kecenderungan Hoaks, sedangkan skor negatif menunjukkan kecenderungan Fakta."
        )

    explanation = explain_prediction(pipeline, text_to_analyze)
    reason_data = generate_verdict_reason(prediction, decision, explanation, text_to_analyze)

    # 3. Fitur Utama: ALASAN KEPUTUSAN MODEL
    st.markdown("### 📋 Mengapa Berita Ini Dinyatakan " + ("HOAKS?" if prediction == 1 else "FAKTA?"))
    st.info(reason_data["summary"])

    st.markdown("#### Poin Analisis Penyebab:")
    for reason in reason_data["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown(f"> 💡 **Saran Verifikasi:** {reason_data['advice']}")

    # 4. Fitur Explainable AI (XAI)
    st.markdown("### 🧠 Bukti Matematis & Transparansi Model (Explainable AI)")
    st.markdown(
        "Pendekatan *Local Feature Attribution* untuk memvalidasi kata-kata spesifik yang "
        "mendasari keputusan model secara objektif:"
    )

    if explanation:
        # A. Penyorotan Kata Kunci
        st.markdown("#### 1. Penyorotan Kata Kunci pada Kalimat")
        st.markdown(
            "> **Panduan Warna:** "
            "<span style='background-color: rgba(239, 68, 68, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Merah = Pemicu Hoaks</span> | "
            "<span style='background-color: rgba(34, 197, 94, 0.4); padding: 2px 6px; border-radius: 4px; font-weight: bold;'>Hijau = Indikator Fakta</span>",
            unsafe_allow_html=True
        )
        
        # Batasi panjang tampilan teks yang disorot jika berita sangat panjang
        display_html = explanation['html_text']
        if len(display_html) > 3000:
            display_html = display_html[:3000] + "... *(teks dipotong untuk efisiensi tampilan)*"

        st.markdown(
            f"""
            <div style="background-color: rgba(255, 255, 255, 0.05); padding: 18px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.15); line-height: 2.0; font-size: 1.05rem;">
                {display_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # B. Grafik Kontribusi Kata
        st.markdown("#### 2. Grafik Peringkat Kontribusi Kata")
        df_chart = explanation["df_chart"]

        if not df_chart.empty:
            df_sorted = df_chart.sort_values(by="Pengaruh", ascending=True)
            
            fig = px.bar(
                df_sorted,
                x="Pengaruh",
                y="Kata",
                color="Kategori",
                orientation="h",
                color_discrete_map={
                    "Pemicu Hoaks": "#ef4444",
                    "Indikator Fakta": "#22c55e"
                },
                labels={"Pengaruh": "Skor Kontribusi Fitur", "Kata": "Kata Kunci"},
                title="Tingkat Pengaruh Kata terhadap Keputusan Model"
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=35, b=10),
                height=min(450, max(220, len(df_chart) * 26)),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kata-kata dalam teks ini berada di luar kosakata representasi model.")

        # Penjelasan Metodologi XAI
        with st.expander("🔍 Penjelasan Metodologi XAI"):
            st.markdown(
                """
                **Bagaimana XAI ini bekerja?**
                * Menggunakan **Linear Feature Attribution**: $\\text{Impact}_i = \\text{TF-IDF}(w_i) \\times \\beta_i$.
                * Fitur bernilai positif secara signifikan menaikkan probabilitas vonis hoaks.
                * Memenuhi kaidah transparansi algoritma (*accountability*) untuk moderasi konten otomatis.
                """
            )

tab_url, tab_manual = st.tabs(["🌐 Cek Langsung dari Link Berita (Akses Internet)", "✍️ Cek dari Teks Manual"])

# TAB 1: SCRAPING DARI URL
with tab_url:
    st.markdown("#### Masukkan Tautan (URL) Berita Online")
    st.write(
        "Sistem akan mengakses internet, mengunduh artikel berita, mengekstrak judul dan narasinya, "
        "lalu melakukan klasifikasi secara instan."
    )

    url_col1, url_col2 = st.columns([8, 2])
    with url_col1:
        input_url = st.text_input(
            "URL Berita:",
            placeholder="Contoh: https://turnbackhoax.id/... atau https://news.detik.com/...",
            key="url_input_box"
        )
    with url_col2:
        st.write("")
        st.write("")
        btn_scrape = st.button("🌐 Ambil & Analisis", type="primary", use_container_width=True)

    # Contoh link cepat
    with st.expander("💡 Contoh Tautan untuk Diuji"):
        st.markdown(
            """
            * Cek Fakta MAFINDO: `https://turnbackhoax.id`
            * Berita Nasional: `https://www.kompas.com`
            * Berita Detik: `https://news.detik.com`
            *(Anda juga bisa menyalin link artikel berita apa pun dari browser lalu menempelkannya di atas).*
            """
        )

    if btn_scrape:
        if not input_url.strip():
            st.error("Silakan masukkan tautan URL terlebih dahulu.")
        else:
            with st.spinner("Sedang menghubungi server berita dan mengekstrak isi artikel..."):
                article_data = extract_article_from_url(input_url)

            if not article_data["success"]:
                st.error(f"❌ {article_data['error']}")
            else:
                st.success(f"✅ Berhasil mengekstrak artikel dari domain: **{article_data['domain']}**")
                
                # Info Artikel
                with st.expander("📄 Lihat Konten yang Diekstrak dari Halaman Web", expanded=True):
                    st.markdown(f"**Judul:** {article_data['title']}")
                    st.markdown(f"**Jumlah Paragraf Ditemukan:** {article_data['paragraph_count']}")
                    preview_text = article_data['content'][:400] + ("..." if len(article_data['content']) > 400 else "")
                    st.markdown(f"**Cuplikan Teks:** *\"{preview_text}\"*")

                # Analisis teks yang diekstrak
                render_prediction_and_xai(article_data["full_text"])


# TAB 2: INPUT TEKS MANUAL
with tab_manual:
    col_input, col_info = st.columns([7, 3])

    with col_info:
        st.markdown("### 📋 Contoh Cepat")
        sample_options = [
            "Pilih contoh untuk dicoba...",
            "Beredar narasi bahwa minum air rebusan pare hangat dapat membunuh virus seketika tanpa obat.",
            "Kementerian Kesehatan resmi mengumumkan jadwal vaksinasi booster kedua gratis di puskesmas.",
            "Informasi mengklaim pembagian kuota internet gratis 100 GB cukup dengan membagikan ke 10 grup WhatsApp.",
            "Bank Indonesia meluncurkan standar kode respon cepat QRIS untuk mempermudah transaksi digital."
        ]
        selected_sample = st.selectbox("Pilih teks berita:", sample_options, index=0, key="manual_sample_select")

    with col_input:
        default_text = "" if selected_sample == sample_options[0] else selected_sample
        text_input = st.text_area(
            "Masukkan Narasi atau Judul Berita:",
            value=default_text,
            placeholder="Ketik atau tempel teks berita di sini...",
            height=130,
            key="manual_text_box"
        )

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            check_button = st.button("🔍 Analisis Teks", type="primary", use_container_width=True, key="manual_check_btn")
        with btn_col2:
            if st.button("🗑️ Reset", use_container_width=True, key="manual_reset_btn"):
                st.rerun()

    if check_button and text_input.strip():
        render_prediction_and_xai(text_input)


# Sidebar Informasi Peneliti
st.sidebar.title("ℹ️ Informasi Penelitian")
st.sidebar.markdown(
    """
    **Judul Penelitian:**  
    Deteksi Berita Hoaks Berbahasa Indonesia Menggunakan Komparasi Machine Learning & Explainable AI (XAI)

    **Fitur Sistem:**
    * 🌐 **Live Web Scraper:** Ekstraksi otomatis dari tautan berita online
    * 🤖 **Machine Learning:** Dilatih pada 12.150 dataset berita nasional
    * 🧠 **Explainable AI:** Transparansi kata kunci pemicu keputusan

    **Sumber Dataset:**  
    Arsip MAFINDO (TurnBackHoax) & Media Berita Nasional
    """
)
