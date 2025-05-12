import os
import sys
import joblib
import pandas as pd
import numpy as np

from extractFeature import run_extract
from preprocessingFeature import run_preprocessing
from sklearn.preprocessing import LabelEncoder

THRESHOLD = 0.5

LDA_TARGETS = [
    "cbMOTPintra",
    "dpcm_sf_probabilities",
    "fa_sfmotp_inter",
    "fa_sfmotp_intra",
    "sect_len_probabilities",
    "sect_len_motp"
]

def apply_column_filtering(df: pd.DataFrame, name: str) -> pd.DataFrame:
    filtering_columns = joblib.load("trained_voting/filtering_columns.pkl")
    if name not in filtering_columns:
        return df
    target_cols = filtering_columns[name]
    for col in target_cols:
        if col not in df.columns:
            return df
    return df[["file"] + target_cols]

def apply_lda_transform(feature_folder):
    for name in LDA_TARGETS:
        raw_csv_path = os.path.join(feature_folder, f"{name}.csv")
        lda_model_path = os.path.join("trained_voting", f"lda_{name}.pkl")

        if not os.path.exists(raw_csv_path) or not os.path.exists(lda_model_path):
            continue

        df = pd.read_csv(raw_csv_path)
        if "file" not in df.columns:
            continue

        file_col = df["file"]

        lda_bundle = joblib.load(lda_model_path)
        lda_model = lda_bundle["model"]
        lda_columns = lda_bundle["columns"]

        try:
            X = df[lda_columns].astype(np.float32)
        except KeyError as e:
            continue

        X_lda = lda_model.transform(X)
        lda_cols = [f"LDA{i+1}" for i in range(X_lda.shape[1])]
        df_lda = pd.DataFrame(X_lda, columns=lda_cols)
        df_lda.insert(0, "file", file_col)

        out_path = os.path.join(feature_folder, f"{name}_lda.csv")
        df_lda.to_csv(out_path, index=False)
        
def merge_csvs(csv_dir):
    exclude_files = {
        "cbMOTPintra.csv",
        "dpcm_sf_probabilities.csv",
        "fa_sfmotp_inter.csv",
        "fa_sfmotp_intra.csv",
        "sect_len_probabilities.csv",
        "sect_len_motp.csv"
    }

    all_files = [
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith(".csv") and f not in exclude_files
    ]

    merged_df = None
    for csv_path in all_files:
        df = pd.read_csv(csv_path)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="file", suffixes=("", f"_{os.path.basename(csv_path)}"))
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    return merged_df

def classify_audio_voting():
    if len(sys.argv) > 1:
        audio_folder = sys.argv[1]
    else:
        audio_folder = "test"

    extracted_csv = run_extract(audio_folder)
    feature_folder = run_preprocessing(extracted_csv)

    filtering_targets = [
        "cbMOTPintra",
        "dpcm_sf_probabilities",
        "fa_sfmotp_inter",
        "fa_sfmotp_intra",
        "num_sec_probabilities",
        "sect_len_probabilities",
        "sect_len_motp"
    ]
    for name in filtering_targets:
        csv_path = os.path.join(feature_folder, f"{name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, float_precision="round_trip")
            df_filtered = apply_column_filtering(df, name)
            df_filtered.to_csv(csv_path, index=False)
            
    apply_lda_transform(feature_folder)

    merged_data = merge_csvs(feature_folder)
    file_list = merged_data["file"].values
    merged_data = merged_data.drop(columns=["file"], errors="ignore")

    used_columns = joblib.load("trained_voting/used_columns.pkl")
    for col in used_columns:
        if col not in merged_data.columns:
            merged_data[col] = 0.0
    merged_data = merged_data[used_columns]
    X_cpu = merged_data.values

    model = joblib.load("trained_voting/trained_voting.pkl")
    scaler = joblib.load("trained_voting/scaler.pkl")
    labels = joblib.load("trained_voting/labels.pkl")

    X_cpu = np.nan_to_num(X_cpu, nan=0.0)
    X_scaled = scaler.transform(X_cpu)
    probs = model.predict_proba(X_scaled)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = labels

    for idx, row in enumerate(probs):
        max_prob = row.max()
        best_idx = row.argmax()
        file_name = file_list[idx]

        print(f"\n[Sample {idx}] File: {file_name}")

        if max_prob < THRESHOLD:
            print("Predicted label: Unknown")
        else:
            label_name = label_encoder.inverse_transform([best_idx])[0]
            print(f"Predicted label: {label_name}, Max probability: {max_prob}")

if __name__ == "__main__":
    classify_audio_voting()
