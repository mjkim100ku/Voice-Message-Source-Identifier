import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import LabelEncoder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.extract_feature import run_extract
from preprocessing.preprocessing_feature import run_preprocessing
THRESHOLD = 0.1
EXPORT_PROB_CSV = True
PROB_CSV_DECIMALS = 12
PROB_CSV_FLOAT_FORMAT = f"%.{PROB_CSV_DECIMALS}f"
SESSION_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

LDA_TARGETS = [
    "cbMOTPintra",
    "dpcm_sf_probabilities",
    "fa_sfmotp_inter",
    "fa_sfmotp_intra",
    "sect_len_probabilities",
    "sect_len_motp",
]


def get_resource_path(*parts):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = PROJECT_ROOT
    return os.path.join(base, *parts)


def apply_column_filtering(df: pd.DataFrame, name: str) -> pd.DataFrame:
    # filtering_columns = joblib.load("models/filtering_columns.pkl")
    filtering_columns = joblib.load(get_resource_path("models", "filtering_columns.pkl"))
    if name not in filtering_columns:
        return df
    target_cols = filtering_columns[name]
    for col in target_cols:
        if col not in df.columns:
            return df
    return df[["file"] + target_cols]

def apply_lda_transform(feature_folder: str):
    for name in LDA_TARGETS:
        raw_csv_path = os.path.join(feature_folder, f"{name}.csv")
        lda_model_path = os.path.join("models", f"lda_{name}.pkl")
        joblib.load(get_resource_path("models", f"lda_{name}.pkl"))
        if not os.path.exists(raw_csv_path) or not os.path.exists(lda_model_path):
            continue

        df = pd.read_csv(raw_csv_path)
        if "file" not in df.columns:
            continue

        file_col = df["file"]

        # lda_bundle = joblib.load(lda_model_path)
        lda_bundle = joblib.load(get_resource_path("models", f"lda_{name}.pkl"))
        lda_model = lda_bundle["model"]
        lda_columns = lda_bundle["columns"]

        try:
            X = df[lda_columns].astype(np.float32)
        except KeyError:
            continue

        X_lda = lda_model.transform(X)
        lda_cols = [f"LDA{i+1}" for i in range(X_lda.shape[1])]
        df_lda = pd.DataFrame(X_lda, columns=lda_cols)
        df_lda.insert(0, "file", file_col)

        out_path = os.path.join(feature_folder, f"{name}_lda.csv")
        df_lda.to_csv(out_path, index=False)

def apply_lda_transform_in_memory(frames: dict) -> dict:
    for name in LDA_TARGETS:
        if name not in frames:
            continue

        lda_model_path = os.path.join("models", f"lda_{name}.pkl")
        if not os.path.exists(lda_model_path):
            continue

        df = frames[name]
        if "file" not in df.columns:
            continue

        file_col = df["file"]

        # lda_bundle = joblib.load(lda_model_path)
        lda_bundle = joblib.load(get_resource_path("models", f"lda_{name}.pkl"))
        lda_model = lda_bundle["model"]
        lda_columns = lda_bundle["columns"]

        try:
            X = df[lda_columns].astype(np.float32)
        except KeyError:
            continue

        X_lda = lda_model.transform(X)
        lda_cols = [f"LDA{i+1}" for i in range(X_lda.shape[1])]
        df_lda = pd.DataFrame(X_lda, columns=lda_cols)
        df_lda.insert(0, "file", file_col)

        frames[f"{name}_lda"] = df_lda

    return frames

def merge_csvs(csv_dir: str) -> pd.DataFrame:
    exclude_files = {
        "cbMOTPintra.csv",
        "dpcm_sf_probabilities.csv",
        "fa_sfmotp_inter.csv",
        "fa_sfmotp_intra.csv",
        "sect_len_probabilities.csv",
        "sect_len_motp.csv",
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
            merged_df = pd.merge(
                merged_df, df, on="file", suffixes=("", f"_{os.path.basename(csv_path)}")
            )
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    return merged_df

def merge_frames(frames: dict) -> pd.DataFrame:
    exclude_raw = {
        "cbMOTPintra",
        "dpcm_sf_probabilities",
        "fa_sfmotp_inter",
        "fa_sfmotp_intra",
        "sect_len_probabilities",
        "sect_len_motp",
    }

    merged_df = None

    for key in sorted(frames.keys()):
        df = frames[key]
        if "file" not in df.columns:
            continue

        if key in exclude_raw and f"{key}_lda" in frames:
            continue

        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pd.merge(
                merged_df, df, on="file", suffixes=("", f"_{key}")
            )
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    return merged_df

def normalize_frame_types(frames: dict) -> dict:
    normalized = {}
    for name, df in frames.items():
        if "file" not in df.columns:
            normalized[name] = df
            continue
        df = df.copy()
        file_col = df["file"]
        feature_df = df.drop(columns=["file"])
        for col in feature_df.columns:
            if feature_df[col].dtype == object:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        normalized[name] = pd.concat([file_col, feature_df], axis=1)
    return normalized

def classify_audio_voting(audio_path: str, verbose: bool = False):
    extracted_csv = run_extract(audio_path)
    feature_folder = run_preprocessing(extracted_csv)
    frames = None

    filtering_targets = [
        "cbMOTPintra",
        "dpcm_sf_probabilities",
        "fa_sfmotp_inter",
        "fa_sfmotp_intra",
        "num_sec_probabilities",
        "sect_len_probabilities",
        "sect_len_motp",
    ]

    for name in filtering_targets:
        csv_path = os.path.join(feature_folder, f"{name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, float_precision="round_trip")
            df_filtered = apply_column_filtering(df, name)
            df_filtered.to_csv(csv_path, index=False)

    if feature_folder is not None:
        apply_lda_transform(feature_folder)
    else:
        frames = apply_lda_transform_in_memory(frames)

    if feature_folder is not None:
        merged_data = merge_csvs(feature_folder)
    else:
        merged_data = merge_frames(frames)

    if merged_data is None or merged_data.empty:
        raise RuntimeError("No features to classify. Check input path and preprocessing pipeline.")

    file_list = merged_data["file"].values
    merged_data = merged_data.drop(columns=["file"], errors="ignore")

    # used_columns = joblib.load("models/used_columns.pkl")
    used_columns = joblib.load(get_resource_path("models", "used_columns.pkl"))
    for col in used_columns:
        if col not in merged_data.columns:
            merged_data[col] = 0.0
    merged_data = merged_data[used_columns]
    X_cpu = merged_data.values

    # model = joblib.load("models/trained_voting.pkl")
    # scaler = joblib.load("models/scaler.pkl")
    # labels = joblib.load("models/labels.pkl")
    model = joblib.load(get_resource_path("models", "trained_voting.pkl"))
    scaler = joblib.load(get_resource_path("models", "scaler.pkl"))
    labels = joblib.load(get_resource_path("models", "labels.pkl"))

    X_cpu = np.nan_to_num(X_cpu, nan=0.0)
    X_scaled = scaler.transform(X_cpu)
    probs = model.predict_proba(X_scaled)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(labels)
    model_classes = getattr(model, "classes_", None)

    if model_classes is not None:
        model_classes_arr = np.asarray(model_classes)
        if np.issubdtype(model_classes_arr.dtype, np.integer):
            class_names = label_encoder.inverse_transform(model_classes_arr.astype(int))
        else:
            class_names = np.array([str(c) for c in model_classes_arr])
    else:
        class_names = np.asarray(label_encoder.classes_)

    if EXPORT_PROB_CSV:
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{SESSION_TIME}.csv")
        probs_df = pd.DataFrame(probs, columns=class_names)
        probs_df.insert(0, "file", file_list)
        write_header = not os.path.exists(out_path)
        probs_df.to_csv(
            out_path,
            index=False,
            float_format=PROB_CSV_FLOAT_FORMAT,
            mode="a",
            header=write_header,
        )

    label_name, max_prob = None, None

    for idx, raw_row in enumerate(probs):
        row = np.asarray(raw_row).ravel()
        best_idx = int(np.argmax(row))
        max_prob_val = float(row[best_idx])
        file_name = file_list[idx]

        if max_prob_val < THRESHOLD:
            pred_label = "Unknown"
        else:
            encoded_label = model_classes[best_idx] if model_classes is not None else best_idx
            if isinstance(encoded_label, (np.integer, int)):
                pred_label = label_encoder.inverse_transform([int(encoded_label)])[0]
            else:
                pred_label = str(encoded_label)

        if verbose:
            print(f"\n[Sample {idx}] File: {file_name}")
            if max_prob_val < THRESHOLD:
                print("Predicted label: Unknown")
            else:
                print(f"Predicted label: {pred_label}, Max probability: {max_prob_val}")

        label_name, max_prob = pred_label, max_prob_val

    return label_name, max_prob


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test"
    is_dir = os.path.isdir(path)
    classify_audio_voting(path)
