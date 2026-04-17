import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(text):
        notebook['cells'].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [t + "\n" if i < len(text)-1 else t for i, t in enumerate(text)]
        })

    def add_code(text):
        notebook['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [t + "\n" if i < len(text)-1 else t for i, t in enumerate(text)]
        })

    add_md([
        "# Tugas Kelompok Materi ke-3 (Model Regresi Time Series)",
        "Notebook ini disusun untuk menyelesaikan seluruh instruksi yang terbagi atas *Job Desc Mahasiswa 1* dan *Mahasiswa 2* berdasarkan dataset `daily-website-visitors.csv`."
    ])

    add_code([
        "# Import Semua Pustaka yang Diperlukan",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from statsmodels.tsa.stattools import adfuller",
        "from statsmodels.tsa.seasonal import seasonal_decompose",
        "from statsmodels.tsa.arima.model import ARIMA",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error",
        "from sklearn.ensemble import RandomForestRegressor",
        "import shap",
        "import itertools",
        "import warnings",
        "warnings.filterwarnings('ignore')"
    ])

    add_md(["## --- TUGAS MAHASISWA 1: Data Architect & Engineer ---", "### 1. Data Ingestion & Reshaping"])
    
    add_code([
        "# Membaca dataset dari folder dataset",
        "df = pd.read_csv('../dataset/daily-website-visitors.csv')",
        "",
        "# Membersihkan data numerik yang mengandung koma",
        "cols_to_clean = ['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']",
        "for col in cols_to_clean:",
        "    df[col] = df[col].astype(str).str.replace(',', '').astype(float)",
        "",
        "# Formatting tanggal",
        "df['Date'] = pd.to_datetime(df['Date'])",
        "",
        "# Menggunakan Melt ('Reshaping') dari format standar ke format long per Metrik",
        "# Hal ini mensimulasikan proses 'melt' yang ada di requirement Kaggle Wide Format",
        "df_melt = pd.melt(df, id_vars=['Date'], value_vars=cols_to_clean, var_name='Metric', value_name='Visits')",
        "print(\"Bentuk data setelah fungsi Melt:\")",
        "display(df_melt.head())",
        "",
        "# Untuk kelanjutan Time Series regresi ini, kita fokus pada satu parameter target yaitu 'Page.Loads'",
        "target_df = df[['Date', 'Page.Loads']].copy()",
        "target_df.set_index('Date', inplace=True)",
        "target_df = target_df.sort_index()"
    ])

    add_md(["### 2. Missing Value Management"])
    add_code([
        "# Mengecek apakah ada nilai null / hari yang hilang (missing values)",
        "print(\"Total Nilai Null dari Page.Loads:\", target_df['Page.Loads'].isnull().sum())",
        "",
        "# Menerapkan imputasi cerdas menggunakan metode 'time' (interpolasi berdasarkan tanggal terdekat)",
        "# apabila kebetulan ada loncatan hari yang hilang, atau mengganti nilai 0 ke nilai rata-rata historis",
        "target_df['Page.Loads'] = target_df['Page.Loads'].replace(0, np.nan)",
        "target_df['Page.Loads'] = target_df['Page.Loads'].interpolate(method='time')"
    ])

    add_md(["### 3. Stationarity Audit (ADF Test)"])
    add_code([
        "# Melakukan uji stasioneritas Dickey-Fuller pada trafik dataset penuh",
        "result = adfuller(target_df['Page.Loads'].dropna())",
        "print('ADF Statistic (Nilai Statistik):', result[0])",
        "print('p-value:', result[1])",
        "",
        "if result[1] < 0.05:",
        "    print('Kesimpulan: p-value < 0.05, maka Dataset memiliki sifat STASIONER.')",
        "else:",
        "    print('Kesimpulan: p-value > 0.05, maka Dataset BERSIFAT TIDAK STASIONER (Terdapat Tren / Musiman).')"
    ])

    add_md(["### 4. Feature Engineering (Lagging)"])
    add_code([
        "# Menciptakan fitur masa lalu Lag-1 hingga Lag-7 (mendapatkan data pengunjung seminggu lalu)",
        "# Hal ini sangat krusial untuk XAI SHAP dan LSTM/Regresi nantinya",
        "for i in range(1, 8):",
        "    target_df[f'Lag_{i}'] = target_df['Page.Loads'].shift(i)",
        "",
        "# Membuang beberapa baris awal yang memuat nilai NaN hasil dari shift/lag",
        "target_df.dropna(inplace=True)",
        "print(\"Dataset dengan fitur Lag:\")",
        "display(target_df.head(2))"
    ])

    add_md(["### 5. Dekomposisi Temporal"])
    add_code([
        "# Menggunakan fungsi seasonal_decompose. Diambil subset dari 90 hari terakhir untuk kejelasan grafik.",
        "decompose_data = target_df['Page.Loads'].tail(90)",
        "decomposition = seasonal_decompose(decompose_data, model='additive', period=7) # Perioda 7 hari (Mingguan)",
        "",
        "# Menampilkan plot Trend, Seasonal, dan Residual",
        "fig = decomposition.plot()",
        "fig.set_size_inches(10, 8)",
        "plt.tight_layout()",
        "plt.show()"
    ])

    add_md(["## --- TUGAS MAHASISWA 2: Model Analyst & Strategist ---", "### 1 & 2. Forecasting Implementation + Hyperparameter Tuning"])
    add_code([
        "# Untuk proses training cepat (seperti yang diminta), kita akan menggunakan hanya sebagian data",
        "data_fast = target_df['Page.Loads'].tail(150)",
        "train_size = int(len(data_fast) * 0.8)",
        "train, test = data_fast[:train_size], data_fast[train_size:]",
        "",
        "print(f\"Banyak data train: {len(train)}, data test: {len(test)}\")",
        "",
        "# Simple Grid Search (Hyperparameter Tuning ARIMA) untuk P,D,Q",
        "best_aic = float('inf')",
        "best_order = None",
        "best_model_fit = None",
        "",
        "p_values = range(0, 3) # Komponen AutoRegressive",
        "d_values = range(0, 2) # Komponen Differencing",
        "q_values = range(0, 3) # Komponen Moving Average",
        "",
        "for order in itertools.product(p_values, d_values, q_values):",
        "    try:",
        "        model = ARIMA(train, order=order)",
        "        res = model.fit()",
        "        if res.aic < best_aic:",
        "            best_aic = res.aic",
        "            best_order = order",
        "            best_model_fit = res",
        "    except:",
        "        continue",
        "",
        "print(f\"\\nParamater ARIMA Terbaik: order={best_order} dengan AIC: {best_aic}\")",
        "",
        "# Forecasting",
        "forecast = best_model_fit.forecast(steps=len(test))"
    ])

    add_md(["### 3. Performance Audit"])
    add_code([
        "# Membuat fungsi Evaluasi SMAPE",
        "def smape_score(y_true, y_pred):",
        "    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100",
        "",
        "# Menghitung error metrik (MAE, RMSE, SMAPE)",
        "mae = mean_absolute_error(test, forecast)",
        "rmse = np.sqrt(mean_squared_error(test, forecast))",
        "smape = smape_score(test.values, forecast.values)",
        "",
        "print(f\"Mean Absolute Error (MAE)   : {mae:.2f}\")",
        "print(f\"Root Mean Square (RMSE)     : {rmse:.2f}\")",
        "print(f\"Symmetric MAPE (SMAPE)      : {smape:.2f}%\")",
        "",
        "# Visualisasi Prediksi VS Aktual",
        "plt.figure(figsize=(10, 4))",
        "plt.plot(train.index, train.values, label='Training Data')",
        "plt.plot(test.index, test.values, label='Actual Test Data')",
        "plt.plot(test.index, forecast.values, color='red', linestyle='--', label='ARIMA Forecast')",
        "plt.title('Evaluasi Hasil Prediksi Data')",
        "plt.legend()",
        "plt.show()"
    ])

    add_md(["### 4. Explainable AI (XAI / SHAP)"])
    add_code([
        "# Karena ekstraksi standar SHAP sulit dikaitkan ke model ARIMA, kita mengadopsi Random Forest pada fitur Lagged (1-7)",
        "# Hal ini secara spesifik bertugas menganalisis \"pengaruh pola kunjungan minggu lalu terhadap prediksi hari ini\"",
        "features = [f'Lag_{i}' for i in range(1, 8)]",
        "X = target_df[features].tail(150)",
        "y = target_df['Page.Loads'].tail(150)",
        "",
        "# Pelatihan Tree Explainer",
        "rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)",
        "rf_model.fit(X, y)",
        "",
        "# Melakukan kalkulasi SHAP values",
        "explainer = shap.TreeExplainer(rf_model)",
        "shap_values = explainer.shap_values(X)",
        "",
        "# Menampilkan bentuk pengaruh variabel terhadap prediksi. ",
        "# Semakin atas, fitur (contoh: Lag_7=Minggu lalu) tersebut makin punya 'Contribution' besar terhadap ramalan besok",
        "print(\"Grafik SHAP Summary:\")",
        "shap.summary_plot(shap_values, X, plot_type='bar')"
    ])

    add_md(["### 5. What-If Simulation: Viral Page / Spike Load Traffic"])
    add_code([
        "# \"Apa yang terjadi jika halaman tiba-tiba viral (naik 1000%)?\"",
        "# Pembuatan Sintesis: menaikkan trafik secara ekstrim di index masa lalu terdekat",
        "spike_data = list(train.values)",
        "spike_idx = len(spike_data) - 10 # 10 hari sebelum akhir masa training",
        "spike_data[spike_idx] = spike_data[spike_idx] * 10 # 1000% jump",
        "",
        "plt.figure(figsize=(10,4))",
        "plt.plot(train.index, spike_data, label='Data Aktual dengan ANOMALI VIRAL', color='orange')",
        "",
        "# Melakukan Refitting ARIMA untuk melihat simulator merespons",
        "# Kita menggunakan order tuning terbaik sebelumnya atas spike_data",
        "anomaly_model = ARIMA(spike_data, order=best_order).fit()",
        "anomaly_forecast = anomaly_model.forecast(steps=len(test))",
        "",
        "plt.plot(test.index, anomaly_forecast, label='Forecast Adaptif Pasca-Viral', color='red')",
        "plt.title('Simulasi Simulator/ARIMA pada Trafik yang Mendadak Loncat 1000%')",
        "plt.legend()",
        "plt.show()"
    ])

    with open('./MATERI-3/syntax/main.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

create_notebook()
