import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
from glob import glob

# Fungsi untuk memuat data dari file Excel
@st.cache_data
def load_data(file):
    try:
        data = pd.read_excel(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Fungsi untuk mengekstrak bulan dan tahun dari nama file
def extract_month_year_from_filename(filename):
    base_name = os.path.basename(filename)
    month_year = base_name.split('.')[0][-7:]  # Mengambil bagian bulan-tahun dari nama file
    return month_year

# Fungsi untuk memuat dan menggabungkan data dari beberapa file dalam folder
def load_data_from_folder(folder_path):
    all_files = glob(os.path.join(folder_path, "*.xlsx"))
    combined_data = pd.DataFrame()
    
    for file in all_files:
        data = load_data(file)
        if data is not None:
            month_year = extract_month_year_from_filename(file)
            data['Bulan_Tahun'] = month_year  # Menambahkan kolom 'Bulan_Tahun' berdasarkan nama file
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    return combined_data

# Fungsi untuk menyimpan hasil prediksi ke file CSV
def save_results(predictions_df, filename):
    predictions_df.to_csv(filename, index=False)
    st.write(f"Hasil prediksi telah disimpan ke file {filename}.")

# Fungsi untuk memvisualisasikan perbandingan antara prediksi dan nilai aktual
def visualize_predictions(y_test, y_pred):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color='red', ax=ax)
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Nilai Prediksi")
    ax.set_title("Perbandingan Nilai Aktual dan Prediksi Penjualan")
    st.pyplot(fig)

# Antarmuka Streamlit
st.title('Prediksi Penjualan Produk dengan MLP')
st.write("Pilih mode prediksi: dari satu file atau dari beberapa file dalam folder.")

# Pilih mode prediksi
prediction_mode = st.radio("Mode Prediksi:", ["Prediksi dari satu file", "Prediksi dari folder uji (08-2023 hingga 03-2024)"])

data = None  # Inisialisasi variabel data

if prediction_mode == "Prediksi dari satu file":
    # Upload file Excel tunggal
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")

    if uploaded_file is not None:
        # Memuat data dari file yang dipilih
        data = load_data(uploaded_file)
        
        if data is not None:
            st.write("Data yang dimuat dari file Excel:")
            st.write(data)
elif prediction_mode == "Prediksi dari folder uji (08-2023 hingga 03-2024)":
    # Menentukan path folder uji
    folder_path = os.path.join(os.getcwd(), "uji")
    st.write(f"Menggunakan data dari folder: {folder_path}")
    
    # Memuat dan menggabungkan data dari folder
    data = load_data_from_folder(folder_path)
    
    if not data.empty:
        st.write("Data yang dimuat dari folder uji:")
        st.write(data)
    else:
        st.error("Tidak ada data yang ditemukan dalam folder uji atau semua file kosong/tidak valid.")
        st.stop()

# Melanjutkan proses hanya jika data tersedia
if data is not None and not data.empty:
    # Pastikan nama produk ikut disertakan
    if 'Nama Produk' not in data.columns:
        st.error("Kolom 'Nama Produk' tidak ditemukan dalam data.")
        st.stop()

    # Memilih kolom-kolom yang akan digunakan sebagai fitur
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.error("Data tidak memiliki kolom numerik yang dapat digunakan sebagai fitur.")
        st.stop()
    
    st.write("Pilih kolom fitur yang ingin digunakan untuk pelatihan:")
    selected_features = st.multiselect("Fitur:", numeric_columns, default=numeric_columns[:5])
    X = data[selected_features]
    
    # Tampilkan tabel data setelah preprocessing hanya dengan kolom fitur yang dipilih
    st.write("Data setelah preprocessing (hanya kolom fitur yang digunakan):")
    st.write(X)

    # Mendefinisikan kolom target yang akan diprediksi
    target_column = 'Produk (Pesanan Siap Dikirim)'
    if target_column not in data.columns:
        st.error(f"Kolom target '{target_column}' tidak ditemukan dalam data.")
        st.stop()
    y = data[target_column]

    # Membagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standarisasi data agar model lebih stabil
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Inisialisasi MLP Regressor dengan parameter yang dipilih
    hidden_layers_option = st.radio("Pilih Hidden Layers:", [(64, 32), (128, 64), (32, 16)])
    activation_option = st.selectbox("Pilih Activation Function:", ['relu', 'tanh', 'logistic'])
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers_option, activation=activation_option, solver='adam', max_iter=500, random_state=42)

    # Melatih model MLP dengan data pelatihan
    mlp.fit(X_train, y_train)

    # Menyimpan model MLP yang telah dilatih ke file
    model_filename = 'mlp_model.pkl'
    joblib.dump(mlp, model_filename)
    st.write(f"Model telah dilatih dan disimpan sebagai {model_filename}.")

    # Melakukan prediksi pada data uji
    y_pred = mlp.predict(X_test)

    # Evaluasi model menggunakan MAE, MSE, dan R² Score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Evaluasi Model:**")
    st.write(f"- Mean Absolute Error (MAE): {mae}")
    st.write(f"- Mean Squared Error (MSE): {mse}")
    st.write(f"- R² Score: {r2}")

    # Visualisasi perbandingan antara prediksi dan nilai aktual
    visualize_predictions(y_test, y_pred)

    # Menampilkan prediksi berdasarkan data uji
    st.write("**Prediksi Penjualan pada Data Uji:**")
    predictions_df = pd.DataFrame({
        'Kode Produk': data['Kode Produk'][y_test.index],
        'Nama Produk': data['Nama Produk'][y_test.index],
        'Prediksi Penjualan Bulan Depan': y_pred
    })
    
    # Mengurutkan prediksi dari yang tertinggi ke terendah
    predictions_df = predictions_df.sort_values(by='Prediksi Penjualan Bulan Depan', ascending=False)
    
    st.write(predictions_df)

    # Menyimpan prediksi ke file CSV jika tombol ditekan
    if st.button('Simpan Prediksi ke CSV'):
        save_results(predictions_df, 'prediksi_penjualan.csv')

# Memuat model yang sudah disimpan
if st.button('Muat Model'):
    if os.path.exists('mlp_model.pkl'):
        loaded_mlp = joblib.load('mlp_model.pkl')
        st.write("Model berhasil dimuat dari file.")
    else:
        st.error("Model tidak ditemukan. Pastikan model telah disimpan dengan benar.")
