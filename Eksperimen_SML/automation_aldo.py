import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import os

def load_data(filepath):
    """Memuat dataset dari file lokal"""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Menghapus kolom yang tidak relevan, nilai kosong, dan duplikat"""
    if "Time_spent_Alone" in df.columns:
        df = df.drop(columns="Time_spent_Alone")
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def encode_and_scale(df):
    """Memisahkan fitur dan target, melakukan encoding dan scaling"""
    X = df.drop(columns='Personality')
    y = df['Personality']

    # Encoding target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # One-hot encoding untuk fitur kategorikal
    X_encoded = pd.get_dummies(X, drop_first=True).astype(int)

    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return pd.DataFrame(X_scaled, columns=X_encoded.columns), y_encoded, label_encoder, scaler

def apply_smote(X, y):
    """Mengatasi ketidakseimbangan kelas dengan SMOTE"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def save_preprocessed_data(X, y, output_path):
    """Menyimpan data yang telah diproses ke CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(X)
    df['Personality'] = y
    df.to_csv(output_path, index=False)
    print(f"Dataset tersimpan di {output_path}")

def main():
    filepath = "dataset/personality_dataset.csv"  # Ganti dengan path dataset Anda
    output_path = "preprocessing/preprocessed_dataset.csv"

    df = load_data(filepath)
    df = clean_data(df)
    X_scaled, y_encoded, _, _ = encode_and_scale(df)
    X_resampled, y_resampled = apply_smote(X_scaled, y_encoded)
    save_preprocessed_data(X_resampled, y_resampled, output_path)

if __name__ == "__main__":
    main()
