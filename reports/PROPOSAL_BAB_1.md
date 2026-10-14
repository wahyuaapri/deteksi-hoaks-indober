# DRAFT PROPOSAL TUGAS AKHIR / SKRIPSI

## USULAN JUDUL

* **Pilihan 1 (Rekomendasi Utama - Sangat Bernilai Tambah):**
  > **"Deteksi Berita Hoaks Berbahasa Indonesia Menggunakan Komparasi IndoBERT dan Support Vector Machine Berbasis Explainable AI (XAI)"**
* **Pilihan 2 (Fokus Klasifikasi & Transparansi Model):**
  > **"Analisis Komparasi Kinerja Algoritma IndoBERT dan SVM dalam Klasifikasi Berita Hoaks Disertai Pendekatan Explainable AI"**
* **Pilihan 3 (Fokus Deep Learning & Transformer):**
  > **"Fine-Tuning IndoBERT untuk Deteksi Berita Hoaks Berbahasa Indonesia dengan Interpretasi Model Menggunakan Feature Attribution"**

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Perkembangan teknologi informasi dan penetrasi internet di Indonesia telah membawa perubahan besar pada cara masyarakat mengonsumsi informasi. Berdasarkan laporan Asosiasi Penyelenggara Jasa Internet Indonesia (APJII), pengguna internet di Indonesia terus meningkat secara signifikan setiap tahunnya. Namun, kemudahan akses informasi ini juga diiringi dengan ancaman serius berupa maraknya penyebaran informasi palsu, misinformasi, disinformasi, dan berita bohong (*hoaks*).

Data dari Masyarakat Anti Fitnah Indonesia (MAFINDO) dan Kementerian Komunikasi dan Informatika (Kominfo) menunjukkan ribuan temuan isu hoaks beredar setiap tahunnya, mencakup sektor politik, kesehatan, sosial, hingga penipuan daring. Paparan berita hoaks dapat memicu keresahan sosial, polarisasi masyarakat, hingga kerugian material. 

Upaya verifikasi fakta yang dilakukan secara manual oleh manusia (*fact-checkers*) menghadapi kendala skalabilitas karena volume berita dan narasi yang beredar di media digital sangat masif dan cepat. Oleh karena itu, diperlukan sistem berbasis *Natural Language Processing* (NLP) dan *Machine Learning* yang mampu mengklasifikasikan keaslian teks berita secara otomatis, cepat, dan presisi.

Metode klasifikasi teks konvensional seperti *Support Vector Machine* (SVM) dengan pembobotan *Term Frequency-Inverse Document Frequency* (TF-IDF) sering digunakan pada penelitian sebelumnya karena efisiensinya. Namun, pendekatan ini memiliki kelemahan mendasar: TF-IDF mengabaikan konteks semantik, susunan gramatikal, dan makna kontekstual antar-kata (*bag-of-words assumption*).

Perkembangan arsitektur *Transformer* melalui model *Bidirectional Encoder Representations from Transformers* (BERT) membawa lompatan besar dalam pemrosesan bahasa alami. Model ini mampu memahami representasi makna kata dua arah (*bidirectional*) berdasarkan konteks kalimat di sekitarnya. Untuk konteks bahasa Indonesia, telah dikembangkan model **IndoBERT** (*Indo4B / IndoBenchmark*), yaitu model bahasa pre-trained yang dilatih pada korpus bahasa Indonesia berskala miliaran kata.

Di sisi lain, penerapan model klasifikasi cerdas sering menghadapi tantangan *Black-Box Problem*, di mana praktisi dan verifikator fakta kesulitan memahami alasan di balik vonis klasifikasi yang dihasilkan algoritma. Tanpa adanya transparansi, adopsi sistem kecerdasan buatan dalam proses moderasi konten dan verifikasi jurnalisme menjadi rentan bias. Oleh karena itu, integrasi **Explainable AI (XAI)** menggunakan metode atribusi fitur lokal (*Local Feature Attribution*) menjadi esensial untuk membedah kata-kata kunci apa saja yang menjadi pemicu label hoaks atau indikator fakta.

Berdasarkan latar belakang tersebut, penelitian ini bermaksud untuk menguji dan membandingkan efektivitas model **IndoBERT** terhadap algoritma baseline (**SVM**), sekaligus menerapkan pendekatan **Explainable AI (XAI)** untuk menghasilkan deteksi berita hoaks berbahasa Indonesia yang tidak hanya akurat, tetapi juga transparan dan dapat dipertanggungjawabkan secara ilmiah.

---

### 1.2 Rumusan Masalah
Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana merancang tahapan pra-pemrosesan teks (*text preprocessing*) dan penyesuaian (*fine-tuning*) model IndoBERT untuk dataset berita hoaks berbahasa Indonesia?
2. Bagaimana kinerja model IndoBERT jika dibandingkan dengan algoritma baseline (Support Vector Machine) dalam mengklasifikasikan berita hoaks?
3. Sejauh mana nilai akurasi, *precision*, *recall*, dan *macro F1-score* yang dihasilkan oleh kedua model tersebut?
4. Bagaimana menerapkan metode *Explainable AI* (XAI) untuk mengungkap dan memvisualisasikan fitur kata kunci (*feature attribution*) yang mendasari keputusan klasifikasi berita hoaks?

---

### 1.3 Batasan Masalah
Agar penelitian ini terarah dan mendalam, batasan masalah yang ditetapkan adalah:
1. Data teks yang digunakan adalah artikel/klaim berita berbahasa Indonesia yang bersumber dari arsip klarifikasi fakta (seperti MAFINDO / TurnBackHoax / portal berita nasional).
2. Klasifikasi bersifat biner (*binary classification*), yaitu mengategorikan teks ke dalam dua kelas: **Fakta (Valid)** dan **Hoaks (Palsu)**.
3. Fitur yang dianalisis berfokus pada konten tekstual (teks judul dan/atau isi narasi berita), tidak mencakup analisis multimedia (gambar, video, metadata waktu, atau graf jejaring sosial).
4. Model pre-trained bahasa Indonesia yang digunakan adalah arsitektur **IndoBERT** (`indobenchmark/indobert-base-p1`).
5. Analisis transparansi model dibatasi pada metode atribusi fitur lokal (*Local Feature Attribution*) untuk menyorot kata-kata pemicu keputusan model.

---

### 1.4 Tujuan Penelitian
Tujuan dari pelaksanaan penelitian ini adalah:
1. Mengimplementasikan alur pra-pemrosesan teks dan proses *fine-tuning* model IndoBERT untuk tugas klasifikasi berita hoaks.
2. Membandingkan performa model IndoBERT terhadap algoritma Support Vector Machine (SVM) berbasis TF-IDF.
3. Menganalisis hasil evaluasi berdasarkan metrik *Accuracy*, *Precision*, *Recall*, *F1-score*, dan *Confusion Matrix*.
4. Mengimplementasikan pendekatan *Explainable AI* (XAI) untuk memberikan visualisasi kontribusi kata kunci (*feature attribution*), sehingga keputusan sistem dapat diinterpretasikan secara transparan.

---

### 1.5 Manfaat Penelitian

#### 1.5.1 Manfaat Teoretis (Akademis)
* Memberikan kontribusi empiris pada literatur kecerdasan buatan dan *Natural Language Processing* (NLP) mengenai efektivitas model *Transformer* (khususnya IndoBERT) dalam penanganan konten disinformasi berbahasa Indonesia.
* Memperkaya kajian akademis di bidang *Explainable AI* (XAI) pada pemrosesan teks bahasa lokal (Bahasa Indonesia).

#### 1.5.2 Manfaat Praktis
* **Bagi Tim Cek Fakta / Verifikator Berita:** Memberikan alat bantu otomasi verifikasi awal yang tidak hanya memvonis hoaks, tetapi juga memperlihatkan kata-kata pemicu utama di dalam teks.
* **Bagi Masyarakat & Pengembang:** Membantu ekosistem digital dalam meminimalkan penyebaran konten berita yang menyesatkan secara transparan dan akuntabel.
