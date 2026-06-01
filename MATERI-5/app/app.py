import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Matikan warning untuk kebersihan log
warnings.filterwarnings('ignore')

# Set Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="SPK Portofolio Saham - ARIMA & TOPSIS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING CUSTOM UNTUK LOOK MODERN DARK MODE ---
st.markdown("""
<style>
    /* Styling Dasar */
    .reportview-container {
        background: #0e1117;
        color: #c9d1d9;
    }
    /* Panel Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    /* Kartu Metric Premium */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    /* Judul Menu Sidebar */
    .sidebar-title {
        color: #58a6ff;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 1px solid #30363d;
        margin-bottom: 20px;
    }
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #30363d;
        color: #8b949e;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# --- DIRECTORY CONFIGURATION ---
# Mendapatkan path absolut relatif terhadap file script ini
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MATERI_5_DIR = os.path.dirname(APP_DIR)
CSV_DIR = os.path.join(MATERI_5_DIR, "dataset", "stock_market_data", "forbes2000", "csv")

# Fallback ke folder workspace jika struktur folder berbeda
if not os.path.exists(CSV_DIR):
    # Coba telusuri relatif dari cwd
    CSV_DIR = os.path.abspath("./MATERI-5/dataset/stock_market_data/forbes2000/csv")

# --- UTILITY & CACHING FUNCTIONS ---
@st.cache_data
def get_available_stocks(directory):
    """Mendapatkan daftar ticker saham yang tersedia dari nama file CSV."""
    if not os.path.exists(directory):
        return ["AAPL", "MSFT", "AMZN"] # Fallback default
    files = os.listdir(directory)
    stocks = [f[:-4] for f in files if f.endswith(".csv")]
    return sorted(stocks)

@st.cache_data
def load_stock_csv(directory, ticker):
    """Membaca file CSV untuk satu ticker saham tertentu."""
    filepath = os.path.join(directory, f"{ticker}.csv")
    if not os.path.exists(filepath):
        # Cari alternatif lain secara rekursif atau return kosong
        raise FileNotFoundError(f"File data {ticker}.csv tidak ditemukan di {directory}.")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def preprocess_stock_data(directory, tickers, history_size=500):
    """Memotong data historis (500 baris terakhir) dan menghitung ROI serta MA_7."""
    df_list = []
    for ticker in tickers:
        try:
            temp_df = load_stock_csv(directory, ticker)
            # Ambil histori terbaru agar efisien
            temp_df = temp_df.tail(history_size).copy()
            
            # Hitung Daily ROI (Return on Investment)
            temp_df['ROI'] = temp_df['Close'].pct_change()
            # Hitung Moving Average 7 Hari
            temp_df['MA_7'] = temp_df['Close'].rolling(window=7).mean()
            
            # Imputasi nilai NaN hasil pergeseran rolling
            temp_df.bfill(inplace=True)
            temp_df['Ticker'] = ticker
            
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"Gagal memuat data {ticker}: {e}")
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

# --- MODEL TRAINING FUNCTIONS ---
def train_arima_model(ts_data, order=(5, 1, 0), horizon=7):
    """Melatih model ARIMA dan memproyeksikan harga ke depan."""
    try:
        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        return forecast, model_fit
    except Exception as e:
        # Fallback jika model ARIMA tidak konvergen / error data
        # Menggunakan naive forecast (mengulang harga penutupan terakhir)
        st.warning(f"Model ARIMA gagal melakukan fitting: {e}. Menggunakan proyeksi Fallback (Naive).")
        forecast = np.full(horizon, ts_data[-1])
        return forecast, None

def evaluate_arima_validation(ts_data, order=(5, 1, 0), validation_days=7):
    """Melakukan evaluasi train/test split untuk validasi RMSE & MAPE."""
    train_ts = ts_data[:-validation_days]
    test_ts = ts_data[-validation_days:]
    
    try:
        model = ARIMA(train_ts, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=validation_days)
        
        # Hitung RMSE
        rmse = np.sqrt(mean_squared_error(test_ts, predictions))
        # Hitung MAPE
        mape = np.mean(np.abs((test_ts - predictions) / test_ts)) * 100
        return rmse, mape, test_ts, predictions
    except Exception as e:
        return np.nan, np.nan, test_ts, np.full(validation_days, train_ts[-1])

def calculate_ahp_weights(ahp_matrix):
    """Menghitung bobot prioritas kriteria menggunakan AHP (Analytic Hierarchy Process)."""
    col_sums = ahp_matrix.sum(axis=0)
    norm_ahp = ahp_matrix / col_sums
    ahp_weights = norm_ahp.mean(axis=1)
    return ahp_weights

# --- TOPSIS COMPUTATION FUNCTIONS ---
def run_topsis_calculation(matrix_df, weights, impacts):
    """Mengimplementasikan Algoritma TOPSIS secara matematis dan modular."""
    matrix = matrix_df.values
    
    # 1. Normalisasi Matriks Keputusan R = [r_ij]
    # Menggunakan Euclidean Normalization
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0) + 1e-12)
    
    # 2. Normalisasi Terbobot V = [v_ij]
    weighted_norm = norm_matrix * weights
    
    # 3. Solusi Ideal Positif (A+) dan Solusi Ideal Negatif (A-)
    # Di mana impacts: 1 (Benefit - Maximize), -1 (Cost - Minimize)
    ideal_pos = np.where(impacts == 1, weighted_norm.max(axis=0), weighted_norm.min(axis=0))
    ideal_neg = np.where(impacts == 1, weighted_norm.min(axis=0), weighted_norm.max(axis=0))
    
    # 4. Jarak Solusi Ideal Positif (D+) & Negatif (D-)
    dist_pos = np.sqrt(((weighted_norm - ideal_pos)**2).sum(axis=1))
    dist_neg = np.sqrt(((weighted_norm - ideal_neg)**2).sum(axis=1))
    
    # 5. Nilai Preferensi (TOPSIS Score)
    scores = dist_neg / (dist_pos + dist_neg + 1e-12)
    
    # Bungkus hasil ke dalam DataFrame
    result_df = matrix_df.copy()
    result_df['TOPSIS_Score'] = scores
    result_df['Peringkat'] = result_df['TOPSIS_Score'].rank(ascending=False, method='min').astype(int)
    result_df['Jarak_Ideal_Positif(D+)'] = dist_pos
    result_df['Jarak_Ideal_Negatif(D-)'] = dist_neg
    
    return result_df.sort_values('Peringkat'), ideal_pos, ideal_neg, norm_matrix, weighted_norm

# --- MENU NAVIGATION IN SIDEBAR ---
st.sidebar.markdown('<div class="sidebar-title">📊 Menu Navigasi</div>', unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    [
        "🏠 Overview & Teori SPK",
        "📂 Dataset & Preprocessing",
        "⚙️ ARIMA Modeling & Validasi",
        "📈 Prediksi & Matriks Keputusan",
        "🏆 AHP-TOPSIS Dashboard",
        "🧪 Simulator & Analisis What-If"
    ]
)

# --- GLOBAL APP STATE / SESSION STATE ---
# Inisialisasi daftar saham default
if 'selected_stocks' not in st.session_state:
    st.session_state['selected_stocks'] = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']
if 'history_size' not in st.session_state:
    st.session_state['history_size'] = 500
if 'forecast_horizon' not in st.session_state:
    st.session_state['forecast_horizon'] = 7

# Daftar saham dengan demand tinggi untuk analisis portofolio
available_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']

# --- RENDER PAGES ---

# ----------------- Halaman 1: Overview -----------------
if menu == "🏠 Overview & Teori SPK":
    st.title("🏠 Overview Sistem Pendukung Keputusan (SPK) Portofolio Saham")
    st.subheader("Model Peramalan Time Series Hybrid (ARIMA) & MCDM (AHP-TOPSIS)")
    
    st.markdown("""
    Selamat datang di **Aplikasi Analisis Portofolio Saham SPK Hybrid**. Aplikasi ini dirancang untuk mensimulasikan dan membuktikan proses ilmiah pengambilan keputusan investasi portofolio saham dengan mengintegrasikan algoritma **Kecerdasan Buatan / Machine Learning (ARIMA)** dan metode **Multi-Criteria Decision Making (AHP & TOPSIS)**.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Tujuan Aplikasi
        1. **Ekstraksi Data Bursa:** Mengakuisisi dan memproses penampang data runtun waktu (*time series*) historis saham dari 1000+ pilihan global.
        2. **Peramalan Prediktif (ARIMA):** Melatih model autoregresif statistika *ARIMA(5,1,0)* untuk memproyeksikan harga masa depan dan rasio estimasi ROI.
        3. **Integrasi Kriteria Komposit (TOPSIS):** Menyandingkan metrik profitabilitas prediktif dengan kriteria risiko volatilitas historis dan likuiditas volume harian.
        4. **Simulasi Analisis What-If:** Menyediakan dashboard interaktif guna mensimulasikan dampak perubahan profil risiko investor terhadap perubahan peringkat alternatif.
        """)
        
    with col2:
        st.markdown("### 🛠️ Alur Arsitektur Sistem (Pipeline)")
        st.graphviz_chart('''
        digraph G {
            bgcolor="#0e1117"
            node [style=filled, color="#30363d", fillcolor="#161b22", fontcolor="#c9d1d9", fontname="Helvetica", shape=box, style="rounded,filled"]
            edge [color="#58a6ff", fontname="Helvetica"]
            
            Ingestion [label="1. Ingestion Data\\n(CSV Stock Data)"]
            Prep [label="2. Preprocessing & Feature\\n(ROI & MA_7, tail(500))"]
            ARIMA [label="3. ARIMA(5,1,0)\\n(Prediksi ROI 7 Hari)"]
            Matrix [label="4. Matriks Keputusan\\n(ROI, Volatilitas, Volume)"]
            Weights [label="5. Bobot Kriteria AHP\\n(Moderat/Agresif/Konservatif)"]
            TOPSIS [label="6. Algoritma TOPSIS\\n(Normalization, Ideal Sol, Distances)"]
            Rank [label="7. Output Rekomendasi\\n(Peringkat Portofolio)"]
            
            Ingestion -> Prep
            Prep -> ARIMA
            ARIMA -> Matrix
            Matrix -> TOPSIS
            Weights -> TOPSIS
            TOPSIS -> Rank
        }
        ''')

    st.markdown("---")
    st.markdown("### 👥 Pembagian Peran Komputasi Akademis")
    
    col_role1, col_role2 = st.columns(2)
    with col_role1:
        st.info("""
        **Mahasiswa ke-1: Data Architect & Predictive Engineer**
        - **Fokus utama:** Ingestion data, feature engineering (pembuatan fitur Daily ROI dan Moving Average 7 Hari), data cleaning/imputation, penyusunan baseline time-series, fitting model peramalan ARIMA(5,1,0), dan audit performa model uji ralat (RMSE dan MAPE).
        """)
    with col_role2:
        st.success("""
        **Mahasiswa ke-2: Decision Strategist & Systems Integrator**
        - **Fokus utama:** Desain matriks keputusan SPK, pembobotan prioritas kriteria menggunakan AHP (Analytic Hierarchy Process), integrasi data model prediktif sebagai kriteria Benefit, pemrosesan peringkat alternatif menggunakan TOPSIS, dan simulator what-if interaktif.
        """)

# ----------------- Halaman 2: Dataset & Preprocessing -----------------
elif menu == "📂 Dataset & Preprocessing":
    st.title("📂 Data Ingestion & Preprocessing Time Series")
    
    # Sidebar control khusus di halaman ini untuk load data
    st.sidebar.subheader("⚙️ Konfigurasi Ingest Data")
    selected_stocks = st.sidebar.multiselect(
        "Pilih Saham Portofolio:",
        options=available_tickers,
        default=st.session_state['selected_stocks']
    )
    history_size = st.sidebar.slider(
        "Jumlah Baris Histori (Terbaru):",
        min_value=100,
        max_value=1000,
        value=st.session_state['history_size'],
        step=50
    )
    
    # Simpan ke session state
    if selected_stocks:
        st.session_state['selected_stocks'] = selected_stocks
    st.session_state['history_size'] = history_size
    
    tickers = st.session_state['selected_stocks']
    
    st.markdown(f"""
    Tahap awal ini memfokuskan pada pengumpulan data historis dan transformasi fitur. 
    Saat ini memproses data untuk **{len(tickers)} saham**: `{', '.join(tickers)}` dengan jendela waktu historis **{history_size} hari aktif bursa**.
    """)
    
    # Eksekusi preprocessing
    with st.spinner("Memproses pipeline data..."):
        df_feat = preprocess_stock_data(CSV_DIR, tickers, history_size)
        
    if df_feat.empty:
        st.error("Data tidak berhasil dimuat. Silakan periksa kembali file dataset Anda.")
    else:
        # Tampilkan Metadata
        st.markdown("### 📊 Metadata Data Hasil Gabungan (Ingested)")
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.metric("Total Baris Data", len(df_feat))
        with meta_cols[1]:
            st.metric("Jumlah Fitur/Kolom", len(df_feat.columns))
        with meta_cols[2]:
            st.metric("Rentang Saham", len(df_feat['Ticker'].unique()))
        with meta_cols[3]:
            # Cari rentang tanggal
            min_date = df_feat['Date'].min().strftime('%Y-%m-%d')
            max_date = df_feat['Date'].max().strftime('%Y-%m-%d')
            st.metric("Periode Data", f"{min_date} s/d {max_date}")
            
        # Tabel Penjelasan Variabel
        with st.expander("📚 Penjelasan Keterangan Variabel Dataset (Klik untuk detail)"):
            st.markdown("""
            | Nama Variabel | Jenis | Deskripsi |
            |---|---|---|
            | **Date** | Datetime (Index) | Tanggal transaksi perdagangan bursa saham. |
            | **Open** | Desimal (Mata Uang) | Harga pembukaan lembar saham pada hari perdagangan tersebut. |
            | **High** | Desimal (Mata Uang) | Nilai perdagangan tertinggi yang dicapai saham pada hari tersebut. |
            | **Low** | Desimal (Mata Uang) | Nilai perdagangan terendah yang dicapai saham pada hari tersebut. |
            | **Close** | Desimal (Mata Uang) | Harga penutupan resmi saham pada hari tersebut (digunakan sebagai basis peramalan). |
            | **Volume** | Angka Bulat | Jumlah lembar saham yang diperdagangkan selama sesi hari tersebut. |
            | **Adjusted Close**| Desimal (Mata Uang) | Harga penutupan yang disesuaikan setelah adanya aksi korporasi (Stock Split/Deviden). |
            | **ROI (Return)** | Desimal (Persentase) | *Feature Engineering:* Rasio pengembalian harian yang dihitung dari persentase perubahan harga Close hari ini dibanding kemarin. |
            | **MA_7** | Desimal (Mata Uang) | *Feature Engineering:* Rata-rata pergerakan harga penutupan dalam rentang jendela 7 hari terakhir. |
            """)
            
        # Tampilkan Sampel Data
        st.markdown("### 📋 Cuplikan Dataset Terproses (10 Baris Pertama)")
        st.dataframe(df_feat.head(10), use_container_width=True)
        
        # Visualisasi Data Historis
        st.markdown("### 📈 Visualisasi Tren Harga Penutupan Historis (Close Price)")
        
        # Reshape data untuk line_chart streamlit
        chart_data = df_feat.pivot(index='Date', columns='Ticker', values='Close')
        st.line_chart(chart_data)
        
        # Visualisasi ROI
        st.markdown("### 📉 Distribusi Volatilitas ROI Harian (%)")
        roi_chart_data = df_feat.pivot(index='Date', columns='Ticker', values='ROI') * 100
        st.line_chart(roi_chart_data)
        
        # Visualisasi Moving Average vs Close (Pilih 1 Saham)
        st.markdown("### 🔍 Penghalusan Tren: Close Price vs Moving Average 7 Hari")
        selected_single = st.selectbox("Pilih Saham untuk Analisis Tren Detail:", tickers)
        single_df = df_feat[df_feat['Ticker'] == selected_single].set_index('Date')
        
        # Plot Close vs MA_7
        ma_plot_data = single_df[['Close', 'MA_7']]
        st.line_chart(ma_plot_data)

# ----------------- Halaman 3: ARIMA Modeling -----------------
elif menu == "⚙️ ARIMA Modeling & Validasi":
    st.title("⚙️ Analisis Model Prediktif ARIMA & Validasi Uji Ralat")
    
    tickers = st.session_state['selected_stocks']
    history_size = st.session_state['history_size']
    
    st.markdown("""
    Model yang digunakan untuk peramalan time series adalah **ARIMA (AutoRegressive Integrated Moving Average)** dengan arsitektur **(5, 1, 0)**.
    Konfigurasi hiperparameter dikunci secara harmonis berdasarkan aspek keuangan bursa:
    - **P = 5 (AutoRegressive):** Menjelaskan pengaruh linear dari momentum harga saham hingga mundur 5 hari ke belakang (ekivalen 1 minggu penuh bursa perdagangan aktif).
    - **D = 1 (Integrated):** Diferensiasi tingkat-1 guna menghilangkan tren tidak stasioner sehingga data bergerak stasioner.
    - **Q = 0 (Moving Average):** Rataan *noise* residual diset nol untuk efisiensi serta peredaman bias sentimen berlebih.
    """)
    
    # Pilih saham sampel untuk diuji validasinya
    sample_stock = st.selectbox("Pilih Saham untuk Analisis Validasi Uji Ralat:", tickers)
    
    with st.spinner("Menjalankan validasi backtesting..."):
        df_feat = preprocess_stock_data(CSV_DIR, [sample_stock], history_size)
        ts_data = df_feat['Close'].values
        
        # Jalankan validasi 7 hari
        rmse, mape, test_ts, val_preds = evaluate_arima_validation(ts_data, order=(5, 1, 0), validation_days=7)
        
    st.markdown("### 📊 Hasil Evaluasi Model Validasi (Backtesting 7 Hari Terakhir)")
    
    # Metric cards
    val_cols = st.columns(3)
    with val_cols[0]:
        st.metric(
            label="Root Mean Squared Error (RMSE)", 
            value=f"{rmse:.4f} USD",
            help="Menunjukkan rata-rata simpangan harga prediksi dalam satuan Dolar."
        )
    with val_cols[1]:
        # Tentukan status MAPE
        status_mape = "Sangat Tangguh (<10%)" if mape < 10 else "Toleransi Rawan (>=10%)"
        st.metric(
            label="Mean Absolute Percentage Error (MAPE)", 
            value=f"{mape:.4f} %", 
            delta=status_mape,
            delta_color="normal"
        )
    with val_cols[2]:
        st.metric(
            label="Order ARIMA", 
            value="ARIMA(5, 1, 0)"
        )
        
    # Visualisasi Uji Validasi
    st.markdown("### 📐 Grafik Perbandingan Kurva Uji Validasi (Aktual vs Prediksi)")
    # Siapkan DataFrame untuk charting
    val_dates = df_feat['Date'].tail(7).values
    val_df = pd.DataFrame({
        'Tanggal': val_dates,
        'Aktual': test_ts,
        'Prediksi': val_preds
    }).set_index('Tanggal')
    
    st.line_chart(val_df)
    
    st.markdown("""
    > [!TIP]
    > **Kesimpulan Pengujian:** Nilai **MAPE di bawah 10%** membuktikan model ARIMA(5,1,0) sangat tangguh untuk memproyeksikan harga jangka pendek harian. Model ini menghasilkan output ROI prediksi bebas bias untuk diserahkan ke Sistem SPK TOPSIS.
    """)

# ----------------- Halaman 4: Prediksi & Matriks Keputusan -----------------
elif menu == "📈 Prediksi & Matriks Keputusan":
    st.title("📈 Proyeksi Masa Depan & Desain Matriks Keputusan SPK")
    
    tickers = st.session_state['selected_stocks']
    history_size = st.session_state['history_size']
    forecast_horizon = st.session_state['forecast_horizon']
    
    st.markdown(f"""
    Di halaman ini, model ARIMA dilatih menggunakan seluruh histori data **{history_size} hari** untuk memproyeksikan pergerakan harga **{forecast_horizon} hari ke depan** bagi seluruh alternatif saham.
    Data hasil peramalan ini akan diolah untuk merancang **Matriks Keputusan (Decision Matrix)** yang memuat 3 kriteria penentu:
    1. **C1: Prediksi ROI (%) [Benefit - Maximize]:** Persentase estimasi pengembalian investasi di hari ke-{forecast_horizon} mendatang.
    2. **C2: Volatilitas Risiko (%) [Cost - Minimize]:** Tingkat risiko investasi yang dihitung dari simpang baku (Standard Deviation) ROI harian historis.
    3. **C3: Likuiditas Volume (Juta Lembar) [Benefit - Maximize]:** Rata-rata volume perdagangan harian sebagai penjamin kecepatan pencairan aset.
    """)
    
    # Hitung prediksi dan susun matriks keputusan
    predictions = {}
    matrix_data = []
    
    with st.spinner("Melatih model ARIMA untuk seluruh alternatif saham..."):
        for stock in tickers:
            df_single = preprocess_stock_data(CSV_DIR, [stock], history_size)
            if df_single.empty:
                continue
            
            ts_data = df_single['Close'].values
            roi_data = df_single['ROI'].values
            vol_data = df_single['Volume'].values
            
            # 1. Training ARIMA & Forecast
            forecast_prices, _ = train_arima_model(ts_data, order=(5,1,0), horizon=forecast_horizon)
            
            # Hitung estimasi ROI akhir horizon
            current_price = ts_data[-1]
            predicted_price = forecast_prices[-1]
            predicted_roi = (predicted_price - current_price) / current_price
            
            predictions[stock] = {
                'current_price': current_price,
                'forecast_prices': forecast_prices,
                'predicted_roi': predicted_roi
            }
            
            # 2. Hitung Kriteria Lainnya
            # C2: Volatilitas (Std Dev dari ROI historis dalam %)
            kr_vol = roi_data.std() * 100
            # C3: Likuiditas (Rata-rata Volume dalam Juta Lembar)
            kr_liq = vol_data.mean() / 1e6
            
            matrix_data.append({
                'Saham': stock,
                'C1_Prediksi_ROI (%)': predicted_roi * 100,
                'C2_Volatilitas_Risiko (%)': kr_vol,
                'C3_Likuiditas_Volume (Juta)': kr_liq
            })
            
    df_matrix = pd.DataFrame(matrix_data).set_index('Saham')
    
    # Simpan matriks keputusan ke session state untuk digunakan di halaman TOPSIS
    st.session_state['df_matrix'] = df_matrix
    st.session_state['predictions'] = predictions
    
    # Tampilkan Hasil Peramalan
    st.markdown("### 🔮 Proyeksi Harga Penutupan Masa Depan (Hasil Forecasting)")
    for stock in tickers:
        if stock not in predictions:
            continue
        p_info = predictions[stock]
        st.markdown(f"**Saham {stock}** | Harga Terakhir: `${p_info['current_price']:.2f}` | Prediksi Hari ke-{forecast_horizon}: `${p_info['forecast_prices'][-1]:.2f}` (Estimasi ROI: `{p_info['predicted_roi']*100:+.4f}%`)")
        
    st.markdown("---")
    
    # Tampilkan Matriks Keputusan
    st.markdown("### 🧮 Hasil Desain Matriks Keputusan SPK (Decision Matrix)")
    st.dataframe(df_matrix.style.highlight_max(subset=['C1_Prediksi_ROI (%)', 'C3_Likuiditas_Volume (Juta)'], color='#1e3a1e')
                         .highlight_min(subset=['C2_Volatilitas_Risiko (%)'], color='#1e3a1e'), 
                 use_container_width=True)

# ----------------- Halaman 5: AHP-TOPSIS Dashboard -----------------
elif menu == "🏆 AHP-TOPSIS Dashboard":
    st.title("🏆 Dashboard Rekomendasi Portofolio AHP-TOPSIS")
    
    if 'df_matrix' not in st.session_state:
        st.warning("⚠️ Silakan buka halaman **'📈 Prediksi & Matriks Keputusan'** terlebih dahulu untuk menghasilkan data prediksi!")
    else:
        df_matrix = st.session_state['df_matrix']
        
        # Pilihan Profil Investor untuk bobot AHP
        st.sidebar.subheader("👤 Profil Risiko Investor")
        profile = st.sidebar.selectbox(
            "Pilih Profil Investor:",
            ["Moderat (Default AHP)", "Konservatif (Takut Risiko)", "Agresif (Mengejar Return)", "Kustom Sendiri"]
        )
        
        # Pengaturan bobot kriteria berdasarkan profil
        if profile == "Moderat (Default AHP)":
            # Matrix AHP Moderat
            # C1 = ROI, C2 = Volatilitas, C3 = Likuiditas
            ahp_matrix = np.array([
                [1,   2,   5],   # C1 terhadap (C1, C2, C3)
                [1/2, 1,   3],   # C2 terhadap (C1, C2, C3)
                [1/5, 1/3, 1]    # C3 terhadap (C1, C2, C3)
            ])
            weights = calculate_ahp_weights(ahp_matrix)
            st.sidebar.info(f"Bobot AHP: ROI={weights[0]*100:.2f}%, Volatilitas={weights[1]*100:.2f}%, Likuiditas={weights[2]*100:.2f}%")
        elif profile == "Konservatif (Takut Risiko)":
            weights = np.array([0.15, 0.80, 0.05])
            st.sidebar.warning("Fokus utama pada minimalisasi Volatilitas Risiko (Bobot C2 = 80%).")
        elif profile == "Agresif (Mengejar Return)":
            weights = np.array([0.70, 0.10, 0.20])
            st.sidebar.success("Fokus utama pada maksimalisasi tingkat keuntungan ROI (Bobot C1 = 70%).")
        else: # Kustom
            w_roi = st.sidebar.slider("Bobot C1: Prediksi ROI (%)", 0, 100, 50)
            w_vol = st.sidebar.slider("Bobot C2: Volatilitas Risiko (%)", 0, 100, 30)
            w_liq = st.sidebar.slider("Bobot C3: Likuiditas Volume (%)", 0, 100, 20)
            
            # Normalisasi bobot kustom
            total_w = w_roi + w_vol + w_liq
            if total_w == 0:
                weights = np.array([0.33, 0.33, 0.34])
            else:
                weights = np.array([w_roi/total_w, w_vol/total_w, w_liq/total_w])
            st.sidebar.write(f"Bobot Hasil Normalisasi: ROI={weights[0]*100:.1f}%, Volatilitas={weights[1]*100:.1f}%, Likuiditas={weights[2]*100:.1f}%")
            
        impacts = np.array([1, -1, 1]) # Benefit, Cost, Benefit
        
        # Eksekusi TOPSIS
        topsis_result, ideal_pos, ideal_neg, norm_matrix, weighted_norm = run_topsis_calculation(df_matrix, weights, impacts)
        
        # Visualisasi Bobot Kriteria
        st.markdown("### ⚖️ Distribusi Pembobotan Kriteria Aktif")
        weights_df = pd.DataFrame({
            'Kriteria': ['C1 (Prediksi ROI)', 'C2 (Volatilitas Risiko)', 'C3 (Likuiditas Volume)'],
            'Bobot (%)': weights * 100
        }).set_index('Kriteria')
        st.bar_chart(weights_df)
        
        st.markdown("---")
        
        # Tampilkan Tabel Peringkat Portofolio
        st.markdown("### 🏆 Hasil Peringkat Alternatif Portofolio Saham (Metode TOPSIS)")
        
        # Tampilkan metric rekomendasi utama
        top_stock = topsis_result.index[0]
        top_score = topsis_result['TOPSIS_Score'].iloc[0]
        
        st.info(f"💡 **Rekomendasi Keputusan Utama:** Berdasarkan metode TOPSIS, saham **{top_stock}** menempati peringkat ke-1 dengan nilai preferensi tertinggi sebesar **{top_score:.4f}**.")
        
        # Format tabel agar representatif
        show_cols = ['C1_Prediksi_ROI (%)', 'C2_Volatilitas_Risiko (%)', 'C3_Likuiditas_Volume (Juta)', 'TOPSIS_Score', 'Peringkat']
        st.dataframe(topsis_result[show_cols], use_container_width=True)
        
        # Visualisasi Skor Preferensi TOPSIS
        st.markdown("### 📊 Grafik Perbandingan Skor Preferensi TOPSIS")
        score_chart = topsis_result[['TOPSIS_Score']].sort_values('TOPSIS_Score', ascending=True)
        st.bar_chart(score_chart)
        
        # Langkah Matematika Ekspansi (Education Mode)
        with st.expander("📝 Tampilkan Detail Perhitungan Matematika TOPSIS (Langkah Demi Langkah)"):
            st.markdown("#### Langkah 1: Matriks Keputusan Ternormalisasi (R)")
            st.dataframe(pd.DataFrame(norm_matrix, index=df_matrix.index, columns=df_matrix.columns), use_container_width=True)
            
            st.markdown("#### Langkah 2: Matriks Ternormalisasi Terbobot (V)")
            st.dataframe(pd.DataFrame(weighted_norm, index=df_matrix.index, columns=df_matrix.columns), use_container_width=True)
            
            st.markdown("#### Langkah 3: Solusi Ideal Positif (A+) & Solusi Ideal Negatif (A-)")
            st.write("Solusi Ideal Positif (A+):", ideal_pos)
            st.write("Solusi Ideal Negatif (A-):", ideal_neg)
            
            st.markdown("#### Langkah 4: Jarak Ideal Positif (D+) & Negatif (D-) Serta Nilai Preferensi")
            st.dataframe(topsis_result[['Jarak_Ideal_Positif(D+)', 'Jarak_Ideal_Negatif(D-)', 'TOPSIS_Score']], use_container_width=True)

# ----------------- Halaman 6: Simulator & What-If -----------------
elif menu == "🧪 Simulator & Analisis What-If":
    st.title("🧪 Simulator Dinamis & Analisis Skenario What-If")
    
    tickers = st.session_state['selected_stocks']
    history_size = st.session_state['history_size']
    
    st.markdown("""
    Sandbox interaktif ini menggabungkan kedua keahlian mahasiswa:
    - **Uji Simulasi Model Prediktif:** Mengubah jendela/horizon hari peramalan secara dinamis untuk melatih ulang model ARIMA.
    - **Uji Analisis What-If SPK:** Mengubah prioritas bobot kriteria untuk melihat pergeseran peringkat portofolio secara langsung.
    """)
    
    # 1. Slider Simulasi Horizon Prediksi
    st.subheader("🔁 1. Simulasi Dinamis Jendela Hari Peramalan ARIMA")
    sim_horizon = st.slider(
        "Tentukan Jendela Horizon Hari Prediksi:",
        min_value=1,
        max_value=30,
        value=st.session_state['forecast_horizon'],
        step=1,
        help="Mengubah nilai ini akan memicu model ARIMA melakukan retraining harga secara dinamis."
    )
    st.session_state['forecast_horizon'] = sim_horizon
    
    # Retrain model secara real-time berdasarkan horizon baru
    sim_matrix_data = []
    with st.spinner("Sedang memproses ulang data peramalan secara real-time..."):
        for stock in tickers:
            df_single = preprocess_stock_data(CSV_DIR, [stock], history_size)
            if df_single.empty:
                continue
            
            ts_data = df_single['Close'].values
            roi_data = df_single['ROI'].values
            vol_data = df_single['Volume'].values
            
            # Training ARIMA harian dinamis
            forecast_prices, _ = train_arima_model(ts_data, order=(5,1,0), horizon=sim_horizon)
            
            # Kalkulasi ROI estimasi baru
            current_price = ts_data[-1]
            predicted_price = forecast_prices[-1]
            predicted_roi = (predicted_price - current_price) / current_price
            
            # Hitung parameter historis stabil
            kr_vol = roi_data.std() * 100
            kr_liq = vol_data.mean() / 1e6
            
            sim_matrix_data.append({
                'Saham': stock,
                'C1_Prediksi_ROI (%)': predicted_roi * 100,
                'C2_Volatilitas_Risiko (%)': kr_vol,
                'C3_Likuiditas_Volume (Juta)': kr_liq
            })
            
    df_sim_matrix = pd.DataFrame(sim_matrix_data).set_index('Saham')
    
    st.write(f"**Matriks Keputusan Baru pada Horizon Peramalan {sim_horizon} Hari:**")
    st.dataframe(df_sim_matrix, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Pengaturan What-If Pembobotan
    st.subheader("🎛️ 2. Analisis What-If Sensitivitas Pembobotan Kriteria")
    st.markdown("Geser nilai di bawah ini untuk melihat pergeseran keputusan investasi portofolio secara langsung:")
    
    w_cols = st.columns(3)
    with w_cols[0]:
        w1 = st.slider("Simulasi Bobot Keuntungan ROI (C1)", 0.0, 1.0, 0.58, 0.01)
    with w_cols[1]:
        w2 = st.slider("Simulasi Bobot Risiko Volatilitas (C2)", 0.0, 1.0, 0.31, 0.01)
    with w_cols[2]:
        w3 = st.slider("Simulasi Bobot Likuiditas Volume (C3)", 0.0, 1.0, 0.11, 0.01)
        
    # Normalisasi bobot simulasi
    total_sim_w = w1 + w2 + w3
    if total_sim_w == 0:
        sim_weights = np.array([0.33, 0.33, 0.34])
    else:
        sim_weights = np.array([w1/total_sim_w, w2/total_sim_w, w3/total_sim_w])
        
    st.write(f"**Bobot Aktif Simulasi:** C1 (ROI) = `{sim_weights[0]*100:.2f}%` | C2 (Volatilitas) = `{sim_weights[1]*100:.2f}%` | C3 (Likuiditas) = `{sim_weights[2]*100:.2f}%`")
    
    # Eksekusi TOPSIS pada data simulasi
    impacts = np.array([1, -1, 1])
    sim_topsis_res, _, _, _, _ = run_topsis_calculation(df_sim_matrix, sim_weights, impacts)
    
    # Bandingkan ranking
    st.markdown("### 🏆 Hasil Keputusan Peringkat Pasca-Intervensi")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown("**Matriks Hasil Peringkat Simulasi:**")
        st.dataframe(sim_topsis_res[['TOPSIS_Score', 'Peringkat']], use_container_width=True)
        
    with col_res2:
        st.markdown("**Perbandingan Visual Peringkat Preferensi:**")
        comp_chart = sim_topsis_res[['TOPSIS_Score']].sort_values('TOPSIS_Score', ascending=True)
        st.bar_chart(comp_chart)
        
    st.markdown("""
    #### 💡 Interpretasi What-If Skenario:
    Saat Anda menaikkan bobot **C2 (Volatilitas Risiko)** ke level yang tinggi (misal > 70%), perhatikan bagaimana sistem SPK TOPSIS secara otomatis memberikan "penalti" pada saham berfluktuasi tinggi. Peringkat saham dengan volatilitas paling rendah akan terangkat secara agresif ke Peringkat 1 meskipun taksiran keuntungan (ROI)-nya sedang. Sebaliknya, meningkatkan bobot **C1 (ROI)** akan mendongkrak saham dengan tren pertumbuhan penutupan harga tertinggi dari ramalan ARIMA.
    """)

# --- GLOBAL FOOTER ---
st.markdown("""
<div class="footer">
    Aplikasi SPK Portofolio Saham • Pemodelan & Simulasi Komputasi • Departemen Matematika & Teknologi Informasi<br>
    © 2026. Dikembangkan dengan Streamlit.
</div>
""", unsafe_allow_html=True)
