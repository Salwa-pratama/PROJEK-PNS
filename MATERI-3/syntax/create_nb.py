import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

# Cell 1: Import
m1 = """# Tugas Prediksi Menggunakan Algoritma ARIMA
**Perhatian**: Teks tugas sebelumnya menyebutkan Dataset Serangan Jantung dengan algoritma klasifikasi. Namun, karena instruksi Anda adalah murni menggunakan **Materi 3 (Dataset daily-website-visitors) menggunakan algoritma ARIMA** beserta waktu training yang cepat dan penjelasan kode yang detail, maka kode di bawah ini dikhususkan untuk memprediksi trafik *website* menggunakan model *Time Series* ARIMA."""
cell_1 = """# Mengimpor pustaka yang diperlukan
import pandas as pd # Untuk manipulasi dan analisis data (seperti membaca file CSV)
import matplotlib.pyplot as plt # Untuk visualisasi data berupa grafik
from statsmodels.tsa.arima.model import ARIMA # Mengimpor algoritma ARIMA untuk time series forecasting
import warnings # Mengimpor modul peringatan untuk penanganan warnings
warnings.filterwarnings('ignore') # Mengabaikan semua peringatan (warnings) agar output terlihat konsisten dan rapi
"""

# Cell 2
cell_2 = """# Menentukan path (lokasi) dari dataset website visitors yang ada pada folder dataset
file_path = '../dataset/daily-website-visitors.csv' 

# Membaca file CSV menjadi objek DataFrame menggunakan pustaka pandas
df = pd.read_csv(file_path) 

# Menampilkan 5 baris pertama dari dataframe untuk melihat bentuk dan struktur data
display(df.head())
"""

# Cell 3
cell_3 = """# Mengonversi tipe data kolom 'Date' menjadi datetime agar sesuai untuk analisis runtun waktu (time series)
df['Date'] = pd.to_datetime(df['Date'])

# Menghilangkan tanda koma (,) pada angka di kolom 'Page.Loads' (misal: "2,146" diubah menjadi "2146")
df['Page.Loads'] = df['Page.Loads'].str.replace(',', '')

# Mengonversi tipe data kolom 'Page.Loads' dari text/string menjadi integer (bilangan bulat) agar bisa diolah secara matematis
df['Page.Loads'] = df['Page.Loads'].astype(int)

# Mengurutkan data (sorting) berdasarkan tanggal dari yang paling lama hingga yang terbaru
df = df.sort_values('Date')

# Menjadikan kolom 'Date' sebagai index dataframe. Hal ini bersifat wajib dalam analisis Time Series dengan statsmodels
df = df.set_index('Date')

# Menampilkan preview sebagian data setelah dilakukan proses pra-pemrosesan (preprocessing)
display(df.head())
"""

# Cell 4
cell_4 = """# Mengambil kolom 'Page.Loads' sebagai data target (data yang nilainya akan diprediksi)
ts_data = df['Page.Loads']

# Menyaring (slicing) data yang akan digunakan untuk proses training.
# Agar proses training TIDAK MEMAKAN WAKTU BANYAK (seperti yang Anda minta), 
# kita hanya menggunakan sampel dari 200 hari terakhir untuk proses latih.
train_data = ts_data.tail(200) 

# Menentukan parameter (p, d, q) untuk model ARIMA. 
# p=1 (AutoRegressive), d=1 (Integrated/differencing), q=1 (Moving Average).
# Penggunaan parameter yang rendah atau sederhana memastikan kecepatan optimasi/fitting yang sangat singkat.
model = ARIMA(train_data, order=(1, 1, 1))

# Melatih (fitting) model ARIMA dengan data training `train_data`
# Fungsi `fit()` akan mencari bobot algoritma yang optimal. Kecepatannya tinggi karena sampel data kita perkecil
fitted_model = model.fit()

# Menampilkan ringkasan (summary) hasil pelatihan model
print(fitted_model.summary())
"""

# Cell 5
cell_5 = """# Melakukan prediksi (forecast) nilai 'Page.Loads' untuk 30 hari (steps) ke depan
forecast = fitted_model.forecast(steps=30)

# Membuat kerangka/kanvas grafik menggunakan matplotlib dengan ukuran gambar (panjang 12 x lebar 6) inci
plt.figure(figsize=(12, 6))

# Mengeplot grafik dari data aktual (historis) yang digunakan untuk model training dengan warna default
plt.plot(train_data.index, train_data.values, label='Data Aktual (Historis)')

# Mengeplot grafik dari hasil prediksi / forecast untuk 30 hari ke depan
# Grafik prediksi ini diberi warna merah ('red') dan bentuk garis putus-putus ('--')
plt.plot(forecast.index, forecast.values, color='red', linestyle='--', label='Prediksi ARIMA (30 Hari Berikutnya)')

# Menambahkan elemen dekoratif pada grafik seperti judul dan penamaan label sumbu
plt.title('Prediksi Jumlah Page Loads Website Menggunakan Model ARIMA') # Label Judul Atas
plt.xlabel('Tanggal') # Label sumbu X
plt.ylabel('Total Page Loads') # Label Sumbu Y

# Menampilkan kotak legenda grafik untuk membedakan antara Data Aktual vs Data Prediksi
plt.legend() 

# Mengaktifkan garis bantu (grid lines) di belakang grafik untuk memudahkan pembacaan titik nilai
plt.grid(True) 

# Menyajikan atau menampilkan grafik final secara utuh ke layar (output cell)
plt.show() 
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(m1),
    nbf.v4.new_code_cell(cell_1),
    nbf.v4.new_code_cell(cell_2),
    nbf.v4.new_code_cell(cell_3),
    nbf.v4.new_code_cell(cell_4),
    nbf.v4.new_code_cell(cell_5)
]

with open('/media/pratama/Data/Other/SELURUH FILE PERKULIAHAN/MATERI SMESTER 6/Pemodelan dan simulasi/projek/KODING-KASUS/MATERI-3/syntax/main.ipynb', 'w') as f:
    nbf.write(nb, f)
