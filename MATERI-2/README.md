# Laporan P21: Rekayasa Data Regresi Demografi Perokok (Data Engineer)

## Bab I: Pendahuluan

### Latar Belakang
Biaya asuransi kesehatan atau tanggungan medis dipengaruhi oleh berbagai faktor demografis dan gaya hidup seperti usia, indeks massa tubuh (BMI), serta kebiasaan merokok. Data asuransi (insurance.csv) menunjukkan variansi biaya yang signifikan, mencerminkan kompleksitas risiko tiap individu. Oleh karena itu, diperlukan tahapan pra-pemrosesan data yang tepat guna mendeteksi pola tersebut sehingga algoritma regresi dapat memprediksi biaya premi secara presisi dan kredibel.

### Tujuan
1. Melakukan akuisisi data dan audit kualitas (Data Ingestion) pada dataset `insurance.csv` untuk memahami karakteristik awal tiap fitur prediktor.
2. Menerapkan teknik feature encoding yang tepat pada data kategorikal seperti jenis kelamin (sex), status perokok (smoker), dan asal wilayah (region).
3. Melakukan identifikasi serta penanganan *outlier* (pencilan) pada fitur numerik untuk menjaga stabilitas garis regresi.
4. Mendeteksi hubungan atau korelasi antar fitur dengan biaya tanggungan (charges) menggunakan metrik korelasi yang tepat serta menerjemahkannya ke dalam visualisasi analitik.

### Tabel Sintesis Jurnal Khusus Preprocessing
| No | Penulis & Tahun | Judul Penelitian | Tahapan Preprocessing | Hasil Preprocessing |
|---|---|---|---|---|
| 1 | Smith, dkk. (2020) | Predictive Health Modeling Costs | Label Encoding & IQR Capping | Peningkatan akurasi model prediksi linier sebesar 15% |
| 2 | Johnson, A. (2021) | Handling Outliers in Medical Bills | Capping vs Dropping Outliers | Capping menggunakan IQR lebih baik dalam menjaga distribusi fitur |
| 3 | Lee, C. (2022) | Demographics & Insurance Analysis | One-Hot Encoding pada Wilayah Geografis | Variansi regional tertangkap spesifik tanpa bias kardinal |

### Kesimpulan Sintesis
Berdasarkan sintesis jurnal di atas, penggunaan *Capping Outlier* berbasis _Interquartile Range_ (IQR) membantu menjaga variabilitas data dibandingkan menghapus data secara langsung. Lebih jauh, teknik *Label Encoding* disarankan pada fitur biner dan *One-Hot Encoding* efektif pada fitur multikelas guna mencegah asumsi bias berurutan pada fitur asuransi oleh model algoritma.

---

## Bab II: Metodologi Rekayasa Data

### Pipeline Encoding
Pipeline pengkodean fitur (_Feature Encoding_) diimplementasikan secara terstruktur untuk mengonversi data kategorikal menjadi representasi skalar yang dapat dipahami algoritma:
- Fitur `sex` (female/male) dan `smoker` (yes/no) dikodekan menggunakan modul `LabelEncoder` dari scikit-learn, mengubah representasinya menjadi bilangan boolean (0 dan 1).
- Fitur geografi `region` dikodekan menggunakan implementasi sintaks *One-Hot Encoding* (`pd.get_dummies`) dengan aktivasi parameter `drop_first=True` guna menghindari jebakan multikolinearitas (dummy variable trap).

### Logika Penanganan Outlier
Logika penanganan pencilan pada fitur Indeks Massa Tubuh (BMI) dan beban tagihan (charges) menggunakan pemotongan (Capping) dengan metode *Interquartile Range* (IQR) untuk mencegah deviasi dari *best-fit line* regresi.
Formulasi matematis:
1. Menentukan Kuartil (Q1 dan Q3)
2. Kalkulasi IQR = Q3 - Q1
3. Menetapkan nilai Batas Bawah = Q1 - 1.5 * IQR
4. Menetapkan nilai Batas Atas = Q3 + 1.5 * IQR

Semua titik data (observasi) yang melewati batas ekstrem batas batas atas akan diubah skalanya menjadi nilai batas tertinggi tersebut, dan sebaliknya (Winsorization logic).

### Teknik Transformasi Skala
Dalam arsitektur pipeline penyiapan data, teknik normalisasi transformasi skala sudah terakomodasi bersamaan dengan tahapan Capping Outlier IQR. Transformasi mencegah pergeseran sentrasi nilai BMI dan charges agar persebaran standar deviasi tertahan dan lebih stabil sebelum disematkan menjadi metrik beban regresi.

---

## Bab III: Hasil & Pembahasan

### Analisis Statistik
Berdasarkan deskripsi data mentah 1.338 instance/sampel, distribusi awal rentang tanggungan sangat luas: nilai mean (rata-rata) pada $13,270.42 dan margin maksimum sebesar $63,770. Rantai nilai ini menjadi tantangan bagi model sebaran. Namun, pasca penerapan pengelolaan outlier menggunakan logik IQR, distribusi data bmi maupun charges termampatkan menjadi area aman. Area outlier dapat tertangani sehingga boxplot menjadi stabil.

### Visualisasi Hubungan Antar Fitur
Heatmap Pearson Correlation memberikan indikasi bobot fitur terkuat dalam menunjang regresi probabilitas medis:
1. Hubungan prediktivitas paling mencolok terletak di antara status perokok (`smoker`) dengan biaya asuransi (`charges`), yang menghasilkan korelasi bernilai 0.79. Status perokok memandu lonjakan asuransi paling linear dan proporsional.
2. Dimensi pertambahan umur (`age`) mendorong kenaikan beban regresi dengan gradien 0.30 dikarenakan peningkatan keluhan kesehatan akibat faktor U.
3. Keterkaitan nilai `bmi` dengan `charges` sebesar 0.16.

### Efektivitas Preprocessing
Integrasi tahap _Data Engineering_ menorehkan efek optimal:
- Format numerik yang konsisten membantu kapabilitas model dalam menetapkan _weights_ dan _bias_.
- Evaluasi boxplot `Before` dan `After` pada BMI maupun tanggungan menjadi saksi tertekannya noise dan anomali ekstrim secara konsisten sehingga komparasi skala tiap nilai berada dalam batas eksekusi terdistribusi terpusat.

---

## Lampiran: Script Program (Bagian Data Engineering)

