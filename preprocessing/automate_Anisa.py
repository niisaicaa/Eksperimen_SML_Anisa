import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocessing_pipeline(csv_path=None, plot_boxplot=True):
    """
    Fungsi untuk melakukan preprocessing dataset Banknote Authentication
    Langkah-langkah:
    1. Pisahkan fitur & target
    2. Hapus missing values
    3. Hapus duplikasi
    4. Hapus outliers menggunakan IQR threshold
    5. Scaling/normalisasi fitur
    6. Cek missing value & duplikasi
    7. Simpan dataset hasil preprocessing
    """

    # === Tentukan path otomatis jika tidak diberikan ===
    if csv_path is None:
        # BASE_DIR = folder Eksperimen_SML_Anisa
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(BASE_DIR, "Banknote-Authentication_raw", 
                                "banknote+authentication", "data_banknote_authentication.txt")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

    # === Load dataset ===
    data = pd.read_csv(csv_path, header=None,
                       names=['variance','skewness','curtosis','entropy','class'])

    # 1. Pisahkan fitur & target
    features = ['variance','skewness','curtosis','entropy']
    X = data[features]
    y = data['class']

    # 2. Hapus missing values
    data_clean = data.dropna()

    # 3. Hapus duplikasi
    data_clean = data_clean.drop_duplicates()

    # 4. Hapus outliers per fitur menggunakan IQR
    Q1 = data_clean[features].quantile(0.25)
    Q3 = data_clean[features].quantile(0.75)
    IQR = Q3 - Q1
    threshold = 1.0  # Sama seperti notebook
    mask = ~((data_clean[features] < (Q1 - threshold * IQR)) |
             (data_clean[features] > (Q3 + threshold * IQR))).any(axis=1)
    data_clean = data_clean[mask]

    # Boxplot untuk cek outlier
    if plot_boxplot:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=data_clean[features])
        plt.title('Boxplot per Feature Setelah Cleaning Ketat')
        plt.show()

    # 5. Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_clean[features])
    y_clean = data_clean['class']

    # Gabungkan lagi dengan target
    data_processed = pd.DataFrame(np.column_stack((X_scaled, y_clean)),
                                  columns=features + ['class'])

    # 6. Cek missing value & duplikasi
    print("=== Missing Values ===")
    print(data_processed.isnull().sum())
    print("\n=== Duplicated Rows ===")
    print(data_processed.duplicated().sum())

    # 7. Simpan CSV di folder yang sama dengan script (preprocessing/)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_banknote_preprocessing.csv")
    data_processed.to_csv(output_path, index=False)
    print(f"\nDataset preprocessing tersimpan di: {output_path}")



    return data_processed

# Jalankan otomatis jika file ini dieksekusi langsung
if __name__ == "__main__":
    df_ready = preprocessing_pipeline()