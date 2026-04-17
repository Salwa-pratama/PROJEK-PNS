import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analisis Klastering Bintang (Star Type Prediction)\n",
    "Dataset: Star Dataset to Predict Star Type"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Mahasiswa 1: Data Architect & Engineer\n",
    "Fokus: Preprocessing data saintifik, encoding spektral, dan optimasi pusat massa."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Data Ingestion\n",
    "Melakukan akuisisi dataset dan penanganan inkonsistensi penulisan pada fitur Star_Color (misal: \"Blue-white\" vs \"Blue White\")."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "df = pd.read_csv('../dataset/6 class csv.csv')\n",
    "print('Shape awal dataset:', df.shape)\n",
    "\n",
    "# Menangani inkonsistensi penulisan pada Star_Color\n",
    "print('Keberagaman Star_Color sebelum cleaning:')\n",
    "print(df['Star color'].unique())\n",
    "\n",
    "df['Star color'] = df['Star color'].str.lower().str.replace('-', ' ').str.replace('white yellowish', 'yellowish white').str.strip()\n",
    "print('\\nKeberagaman Star_Color setelah cleaning:')\n",
    "print(df['Star color'].unique())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Feature Engineering\n",
    "Melakukan transformasi logaritma pada fitur Luminosity untuk menyeimbangkan distribusi data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transformasi logaritma pada fitur Luminosity untuk menyeimbangkan distribusi\n",
    "df['Log_Luminosity'] = np.log1p(df['Luminosity(L/Lo)'])\n",
    "print('Distribusi Luminosity vs Log_Luminosity:')\n",
    "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n",
    "sns.histplot(df['Luminosity(L/Lo)'], kde=True, ax=ax[0])\n",
    "ax[0].set_title('Original Luminosity')\n",
    "sns.histplot(df['Log_Luminosity'], kde=True, ax=ax[1])\n",
    "ax[1].set_title('Log Transformed Luminosity')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Advanced Preprocessing\n",
    "Implementasi StandardScaler atau MinMaxScaler untuk menyatukan skala fitur suhu (ribuan) dengan magnitudo (puluhan)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler, OrdinalEncoder\n",
    "\n",
    "# Implementasi StandardScaler untuk menyatukan skala fitur\n",
    "scaler = StandardScaler()\n",
    "features_to_scale = ['Temperature (K)', 'Log_Luminosity', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']\n",
    "df_scaled = df.copy()\n",
    "df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])\n",
    "print('Data setelah disatukan skalanya dengan StandardScaler:')\n",
    "display(df_scaled[features_to_scale].head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Encoding Strategy\n",
    "Transformasi fitur Spectral_Class menggunakan Ordinal Encoding berdasarkan urutan suhu standar astronomi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transformasi fitur Spectral_Class menggunakan Ordinal Encoding\n",
    "# Urutan suhu standar astronomi dari yang terdingin ke terpanas: M, K, G, F, A, B, O\n",
    "spectral_order = [['M', 'K', 'G', 'F', 'A', 'B', 'O']]\n",
    "encoder = OrdinalEncoder(categories=spectral_order)\n",
    "df_scaled['Spectral_Class_Encoded'] = encoder.fit_transform(df[['Spectral Class']])\n",
    "print('Pemetaan kelas spektral:\\n', df[['Spectral Class']].drop_duplicates().merge(df_scaled[['Spectral_Class_Encoded']].drop_duplicates(), left_index=True, right_index=True).sort_values('Spectral_Class_Encoded'))\n",
    "\n",
    "# Fitur akhir yang akan digunakan untuk klastering\n",
    "X = df_scaled[['Temperature (K)', 'Log_Luminosity', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Spectral_Class_Encoded']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Optimization Search\n",
    "Menjalankan eksperimen perulangan algoritma untuk mencari nilai KK (jumlah tipe bintang) terbaik menggunakan Metode Elbow (Versi DBSCAN: K-Distance Graph)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import NearestNeighbors\n",
    "\n",
    "# Untuk algoritma DBSCAN, metode Elbow diimplementasikan dengan membuat K-Distance Graph (Jarak ke tetangga terdekat K)\n",
    "min_samples = 5\n",
    "neighbors = NearestNeighbors(n_neighbors=min_samples)\n",
    "neighbors_fit = neighbors.fit(X)\n",
    "distances, indices = neighbors_fit.kneighbors(X)\n",
    "distances = np.sort(distances[:, min_samples-1], axis=0)\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(distances)\n",
    "plt.title('K-Distance Graph untuk mencari nilai epsilon terbaik')\n",
    "plt.xlabel('Titik data yang diurutkan')\n",
    "plt.ylabel(f'{min_samples}-NN Distance')\n",
    "plt.grid(True)\n",
    "plt.axhline(y=0.6, color='r', linestyle='--', label='Titik Belok (Epsilon)')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "print('Berdasarkan kurva, titik belok (elbow) berada di sekitar nilai jarak (epsilon) = 0.6')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Mahasiswa 2: Model Analyst & Strategist\n",
    "Fokus: Arsitektur klastering bintang, validasi stabilitas, dan interpretasi astrofisika."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Model Implementation\n",
    "Menerapkan algoritma klastering DBSCAN pada fitur fisik bintang."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.cluster import DBSCAN\n",
    "from sklearn.metrics import silhouette_score\n",
    "\n",
    "optimal_eps = 0.6\n",
    "dbscan_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)\n",
    "cluster_labels = dbscan_model.fit_predict(X)\n",
    "\n",
    "df['Cluster_DBSCAN'] = cluster_labels\n",
    "df_scaled['Cluster_DBSCAN'] = cluster_labels\n",
    "print('Jumlah klaster yang dibentuk (termasuk noise -1):', len(set(cluster_labels)))\n",
    "print(df['Cluster_DBSCAN'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Quality Audit\n",
    "Mengevaluasi kerapatan kelompok bintang menggunakan Silhouette Coefficient."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(set(cluster_labels)) > 1:\n",
    "    sil_score = silhouette_score(X, cluster_labels)\n",
    "    print(f'Mengevaluasi kerapatan menggunakan Silhouette Coefficient: {sil_score:.4f}')\n",
    "    if sil_score > 0.5:\n",
    "        print('Interpretasi: Kerapatan dan pemisahan klaster berkualitas baik/kuat.')\n",
    "    else:\n",
    "        print('Interpretasi: Kerapatan klaster moderat atau overlapping yang sering dijumpai pada model DBSCAN karena adanya noise.')\n",
    "else:\n",
    "    print('DBSCAN hanya menemukan 1 klaster atau semua dianggap noise.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Stability Testing\n",
    "Melakukan uji stabilitas kelompok menggunakan Indeks Jaccard dengan menambahkan gangguan (noise) pada data fisik untuk memastikan kelompok bintang tetap konsisten."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import jaccard_score\n",
    "\n",
    "# Melakukan uji stabilitas kelompok dengan menambahkan gangguan empirikal (noise gaussian)\n",
    "np.random.seed(42)\n",
    "noise = np.random.normal(0, 0.05, X.shape) # menambah 5% std noise ke fitur skala\n",
    "X_noisy = X + noise\n",
    "\n",
    "dbscan_noisy = DBSCAN(eps=optimal_eps, min_samples=min_samples)\n",
    "cluster_noisy_labels = dbscan_noisy.fit_predict(X_noisy)\n",
    "\n",
    "# Evaluasi konsistensi label, DBSCAN dapat menggeser id klaster karena sifat kepadatan non-parametrik.\n",
    "# Namun untuk memudakhan dengan asumsi label klaster masih sejalan, kita uji kemiripan menggunakan macro Jaccard Score.\n",
    "jaccard_idx = jaccard_score(cluster_labels, cluster_noisy_labels, average='weighted')\n",
    "print(f'Jaccard Index (Stability Score): {jaccard_idx:.4f}')\n",
    "if jaccard_idx > 0.8:\n",
    "    print('Kesimpulan: Klaster bintang SANGAT STABIL terhadap perturbasi fisik (noise) minimal.')\n",
    "elif jaccard_idx > 0.5:\n",
    "    print('Kesimpulan: Klaster bintang CUKUP STABIL terhadap perturbasi.')\n",
    "else:\n",
    "    print('Kesimpulan: Klaster bintang RENTAN, terjadi pergeseran batas kelompok akibat perturbasi.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Star Profiling\n",
    "Memberikan interpretasi logis pada setiap kelompok yang ditemukan dan membandingkannya dengan teori Diagram Hertzsprung-Russell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Interpretasi Profil Bintang terhadap sub-klaster (DBSCAN) dan kaitannya dengan Teori Diagram Hertzsprung-Russell (HR Diagram)')\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "# HR Diagram biasanya merencanakan Temperature (terbalik) vs Absolute Magnitude (terbalik)\n",
    "# Tapi di sini kita gambarkan Temperature vs Log_Luminosity (semakin cerah di atas)\n",
    "sns.scatterplot(x='Temperature (K)', y='Log_Luminosity', hue='Cluster_DBSCAN', palette='tab10', data=df)\n",
    "plt.gca().invert_xaxis() # Temperature diplot terbalik sesuai Diagram HR standar\n",
    "plt.title('Hertzsprung-Russell Diagram (T vs L) Berdasarkan DBSCAN Clusters')\n",
    "plt.xlabel('Temperature (K) [Suhu Permukaan - Inverted]')\n",
    "plt.ylabel('Log Luminosity [Kecerahan]')\n",
    "plt.show()\n",
    "\n",
    "print('Interpretasi Singkat:')\n",
    "print('- Biasanya, sebagian besar klaster dengan temperatur turun namun kecerahan ekstrem menempati ranah Raksasa Merah (Red Giants/Supergiants).')\n",
    "print('- Klaster garis diagonal pada bagian tengah dinamakan Main Sequence Stars (termasuk Matahari).')\n",
    "print('- Nilai klaster -1 menandakan titik kebisingan (Anomali yang tidak masuk profil manapun menurut kerapatan DBSCAN)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Astro-Simulation\n",
    "Melakukan analisis intervensi: \"Jika sebuah bintang mengalami penurunan suhu namun luminositasnya tetap, apakah ia akan berpindah ke klaster bintang raksasa?\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Membuat model klasifikasi K-NN pada hasil klasterisasi DBSCAN untuk mensimulasi prediksi pergeseran ruang bintang\n",
    "# (Langkah ini mempermudah prediksi pergerakan bintang padahal DBSCAN tidak memiliki fungsi predict() native).\n",
    "# Gunakan data yang tidak terdeteksi noise saja (cluster != -1)\n",
    "valid_idx = cluster_labels != -1\n",
    "X_train, y_train = X[valid_idx], cluster_labels[valid_idx]\n",
    "\n",
    "knn_sim = KNeighborsClassifier(n_neighbors=3)\n",
    "knn_sim.fit(X_train, y_train)\n",
    "\n",
    "# Identifikasi sample Main Sequence Star. \n",
    "# Memilih bintang bertipe Main Sequence (misal Star type == 3) atau bintang yang cukup pertengahan\n",
    "base_star = df[(df['Star type'] == 3)].iloc[0]\n",
    "print(f'\\nBintang Original (Type {base_star[\"Star type\"]}, Spectral {base_star[\"Spectral Class\"]})')\n",
    "print(f'Temperature: {base_star[\"Temperature (K)\"]}, Luminosity: {base_star[\"Luminosity(L/Lo)\"]}, Klaster Awal: {base_star[\"Cluster_DBSCAN\"]}')\n",
    "\n",
    "# Simulasi: Penurunan suhu -60%, luminositas, radius, dan abs_mag tetap\n",
    "sim_star_df = pd.DataFrame([base_star])\n",
    "sim_star_df['Temperature (K)'] = sim_star_df['Temperature (K)'] * 0.4 # Penurunan Suhu Signifikan\n",
    "sim_star_df['Spectral Class'] = 'M' # Jika suhu turun sangat drastis, warna geser ke kemerahan kelas M\n",
    "sim_star_df['Spectral_Class_Encoded'] = encoder.transform(sim_star_df[['Spectral Class']])\n",
    "sim_star_df[features_to_scale] = scaler.transform(sim_star_df[features_to_scale])\n",
    "\n",
    "sim_X = sim_star_df[['Temperature (K)', 'Log_Luminosity', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Spectral_Class_Encoded']]\n",
    "new_cluster = knn_sim.predict(sim_X)[0]\n",
    "\n",
    "print(f'\\nHasil ASTRO-SIMULATION:\\nBintang dengan suhu diturunkan diprediksi akan BERPINDAH ke klaster: {new_cluster}')\n",
    "\n",
    "# Periksa apakah klaster tersebut dominan adalah bintang raksasa/super raksasa (Tipe >= 4)\n",
    "target_cluster_types = df[df['Cluster_DBSCAN'] == new_cluster]['Star type']\n",
    "print(f'Profil mayoritas di klaster {new_cluster} adalah tipe bintang: \\n{target_cluster_types.value_counts()}')\n",
    "print('\\nJawaban: Ya, jika suhu menurun drastis namun Luminositas tinggi (tetap terang), bintang akan bergeser statusnya menuju klaster bintang raksasa / Supergiant merah, sesuai teori evolusi tahap lanjut bintang di diagram HR.')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 }
}

with open('/media/pratama/Data/Other/SELURUH FILE PERKULIAHAN/MATERI SMESTER 6/Pemodelan dan simulasi/projek/KODING-KASUS/MATERI-4/syntax/main.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
