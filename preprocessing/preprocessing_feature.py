import pandas as pd
import ast
import os, sys
from datetime import datetime
import numpy as np
import joblib
import warnings
from pathlib import Path
from pandas.errors import PerformanceWarning

warnings.simplefilter(action='ignore', category=PerformanceWarning)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def get_resource_path(*parts):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = PROJECT_ROOT
    return os.path.join(base, *parts)

def tokenize_text(text):
    if not isinstance(text, str):
        text = str(text)
    replaced = text.replace(" ", "_")
    return [replaced]

def text_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens if token in vocab]

def preprocess_text_feature(column, column_data):
    all_tokens = [tokenize_text(str(text)) for text in column_data]

    vocab_path = os.path.join("models", "vocab.pkl")
    # vocab = joblib.load(vocab_path)
    vocab = joblib.load(get_resource_path("models", "vocab.pkl"))


    numeric_data = []
    if column not in vocab:
        # print(f"[WARN] Column '{column}' not in vocab.pkl. Skipping...")
        return np.empty((len(column_data), 0))
    for tokens in all_tokens:
        indices = text_to_indices(tokens, vocab[column])
        numeric_data.append(indices)

    max_length = max(len(row) for row in numeric_data)
    numeric_data = [row + [None] * (max_length - len(row)) for row in numeric_data]

    return np.array(numeric_data)

def preprocess_text_feature_count(column_data):
    all_tokens = []
    for val in column_data.fillna(""):
        tokens = str(val).split()
        all_tokens.extend(tokens)

    unique_tokens = sorted(set(all_tokens))

    result_matrix = []
    for val in column_data.fillna(""):
        tokens = str(val).split()
        token_count = {}
        for t in tokens:
            token_count[t] = token_count.get(t, 0) + 1

        row_vec = []
        for t in unique_tokens:
            row_vec.append(token_count.get(t, 0))
        result_matrix.append(row_vec)

    col_name = column_data.name
    new_col_names = [f"{col_name}_cat_{t}" for t in unique_tokens]
    df_count = pd.DataFrame(result_matrix, columns=new_col_names, index=column_data.index)
    return df_count

def preprocess_numeric_feature(column_data):
    return column_data.fillna(0).astype(float)

def ensure_dict(x):
    if not x or pd.isna(x):
        return {}
    try:
        return ast.literal_eval(x)
    except:
        return {}

def expand_1d_probability(row_dict, key_range):
    return [row_dict.get(k, 0.0) for k in key_range]

def expand_2d_probability(row_dict, key_range):
    result = []
    for m in key_range:
        for n in key_range:
            result.append(row_dict.get((m, n), 0.0))
    return result

def main(input_csv: str, out_dir: str = "./output"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    if "cb" in df.columns:
        meta_cols = df.columns[1: df.columns.get_loc("cb")]
    else:
        meta_cols = df.columns[1:]

    exclude_keywords = ["_time", "duration", "reserved", "flags", "pre_defined", "predefined", "sample_count", "stts/@entries", "stsz/@entries", "samples_per_chunk"]
    meta_cols = [col for col in meta_cols if not any(kw in col for kw in exclude_keywords)]

    metadata = pd.DataFrame()

    for col in meta_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            processed = preprocess_numeric_feature(df[col])
            metadata[col] = processed
        else:
            processed = preprocess_text_feature(col, df[col])
            num_tokens = processed.shape[1]
            for i in range(num_tokens):
                metadata[f"{col}_idx{i}"] = processed[:, i]

    metadata.insert(0, "file", df["file"])

    metadata_path = os.path.join(out_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    
    max_num_sec = 1
    max_sect_len = 1

    for _, row in df.iterrows():
        ns_dict = ensure_dict(row.get("num_sec_probabilities", "{}"))
        sl_dict = ensure_dict(row.get("sect_len_probabilities", "{}"))
        if ns_dict:
            tmp_max_ns = max(map(int, ns_dict.keys()))
            if tmp_max_ns > max_num_sec:
                max_num_sec = tmp_max_ns
        if sl_dict:
            tmp_max_sl = max(map(int, sl_dict.keys()))
            if tmp_max_sl > max_sect_len:
                max_sect_len = tmp_max_sl

    cb_range = range(0, 12)
    dpcm_range_raw = range(0, 121)
    fa_sfmotp_range_raw = range(0, 121)
    num_sec_range = range(1, max_num_sec + 1)
    sect_len_range = range(1, max_sect_len + 1)

    result_cb = []
    result_cbMOTPintra = []
    result_dpcm_sf_raw = []
    result_fa_sfmotp_inter_raw = []
    result_fa_sfmotp_intra_raw = []
    result_num_sec_raw = []
    result_sect_len_raw = []
    result_sect_len_motp_raw = []

    for _, row in df.iterrows():
        file_name = row["file"]

        cb_dict = ensure_dict(row.get("cb", "{}"))
        cb_probs = expand_1d_probability(cb_dict, cb_range)
        result_cb.append([file_name] + cb_probs)

        cbMOTPintra_dict = ensure_dict(row.get("cbMOTPintra", "{}"))
        cbMOTPintra_probs = expand_2d_probability(cbMOTPintra_dict, cb_range)
        result_cbMOTPintra.append([file_name] + cbMOTPintra_probs)

        dpcm_dict = ensure_dict(row.get("dpcm_sf_probabilities", "{}"))
        dpcm_probs = expand_1d_probability(dpcm_dict, dpcm_range_raw)
        result_dpcm_sf_raw.append([file_name] + dpcm_probs)

        fa_inter_dict = ensure_dict(row.get("fa_sfmotp_inter", "{}"))
        fa_inter_probs = expand_2d_probability(fa_inter_dict, fa_sfmotp_range_raw)
        result_fa_sfmotp_inter_raw.append([file_name] + fa_inter_probs)

        fa_intra_dict = ensure_dict(row.get("fa_sfmotp_intra", "{}"))
        fa_intra_probs = expand_2d_probability(fa_intra_dict, fa_sfmotp_range_raw)
        result_fa_sfmotp_intra_raw.append([file_name] + fa_intra_probs)

        ns_dict = ensure_dict(row.get("num_sec_probabilities", "{}"))
        ns_probs = expand_1d_probability(ns_dict, num_sec_range)
        result_num_sec_raw.append([file_name] + ns_probs)

        sl_dict = ensure_dict(row.get("sect_len_probabilities", "{}"))
        sl_probs = expand_1d_probability(sl_dict, sect_len_range)
        result_sect_len_raw.append([file_name] + sl_probs)

        sl_motp_dict = ensure_dict(row.get("sect_len_motp", "{}"))
        sl_motp_probs = expand_2d_probability(sl_motp_dict, sect_len_range)
        result_sect_len_motp_raw.append([file_name] + sl_motp_probs)

    cb_pca_path = os.path.join(out_dir, "cb.csv")
    pd.DataFrame(result_cb, columns=["file"] + [str(i) for i in cb_range]).to_csv(cb_pca_path, index=False)

    cbMOTPintra_path = os.path.join(out_dir, "cbMOTPintra.csv")
    pd.DataFrame(result_cbMOTPintra, columns=["file"] + [f"({m},{n})" for m in cb_range for n in cb_range]).to_csv(cbMOTPintra_path, index=False)

    dpcm_sf_path = os.path.join(out_dir, "dpcm_sf_probabilities.csv")
    pd.DataFrame(result_dpcm_sf_raw, columns=["file"] + [str(i) for i in dpcm_range_raw]).to_csv(dpcm_sf_path, index=False)

    fa_sfmotp_inter_path = os.path.join(out_dir, "fa_sfmotp_inter.csv")
    pd.DataFrame(result_fa_sfmotp_inter_raw, columns=["file"] + [f"({m},{n})" for m in fa_sfmotp_range_raw for n in fa_sfmotp_range_raw]).to_csv(fa_sfmotp_inter_path, index=False)

    fa_sfmotp_intra_path = os.path.join(out_dir, "fa_sfmotp_intra.csv")
    pd.DataFrame(result_fa_sfmotp_intra_raw, columns=["file"] + [f"({m},{n})" for m in fa_sfmotp_range_raw for n in fa_sfmotp_range_raw]).to_csv(fa_sfmotp_intra_path, index=False)

    num_sec_path = os.path.join(out_dir, "num_sec_probabilities.csv")
    pd.DataFrame(result_num_sec_raw, columns=["file"] + [str(i) for i in num_sec_range]).to_csv(num_sec_path, index=False)

    sect_len_path = os.path.join(out_dir, "sect_len_probabilities.csv")
    pd.DataFrame(result_sect_len_raw, columns=["file"] + [str(i) for i in sect_len_range]).to_csv(sect_len_path, index=False)

    sect_len_motp_path = os.path.join(out_dir, "sect_len_motp.csv")
    pd.DataFrame(result_sect_len_motp_raw, columns=["file"] + [f"({m},{n})" for m in sect_len_range for n in sect_len_range]).to_csv(sect_len_motp_path, index=False)


    # print("\n=== Processing complete. ===")
    # print(f"metadata.csv => {metadata_path}")
    # print(f"cb.csv => {cb_pca_path}")
    # print(f"cbMOTPintra.csv => {cbMOTPintra_path}")
    # print(f"dpcm_sf_probabilities.csv => {dpcm_sf_path}")
    # print(f"fa_sfmotp_inter.csv => {fa_sfmotp_inter_path}")
    # print(f"fa_sfmotp_intra.csv => {fa_sfmotp_intra_path}")
    # print(f"num_sec_probabilities.csv => {num_sec_path}")
    # print(f"sect_len_probabilities.csv => {sect_len_path}")
    # print(f"sect_len_motp.csv => {sect_len_motp_path}")

def run_preprocessing(file_path):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir_path = f"{Path(file_path).parent.name + os.path.sep}preprocessing_{current_time}"
    main(file_path, output_dir_path)
    
    return output_dir_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_csv_path = sys.argv[1]
    else:
        input_csv_path = "audio_dataset_info_20250415_215714.csv"

    run_preprocessing(input_csv_path)