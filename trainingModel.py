import os
import sys
import pandas as pd
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.linear_model import LogisticRegression
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score as cuml_accuracy
from cuml.metrics import confusion_matrix as cuml_confusion_matrix

from sklearn.metrics import accuracy_score as sk_accuracy
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import (
    VotingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier as skRandomForest
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.neighbors import KNeighborsClassifier as skKNN
from sklearn.svm import SVC as skSVC
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.model_selection import train_test_split as SklearnTrainTestSplit
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.inspection import permutation_importance

model_results = []

def extract_labels(file_name):
    parts = file_name.split("_")
    os_label = parts[0]
    messenger_label = parts[1]
    if os_label.startswith("and"):
        os_label = "and"
    elif os_label.startswith("ios"):
        os_label = "ios"
    return os_label, messenger_label

def merge_csvs(csv_dir):
    all_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    merged_df = None
    for file in all_files:
        df = pd.read_csv(file)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="file", suffixes=("", f"_{os.path.basename(file)}"))
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    return merged_df

def cross_validate_model(model, X, y, folds=5, gpu=False, unique_labels=None, random_state=42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    accuracies = []
    fold_idx = 1

    for train_idx, val_idx in kf.split(X):
        if gpu:
            X_train = cp.asarray(X[train_idx], dtype=cp.float32)
            X_val   = cp.asarray(X[val_idx],   dtype=cp.float32)
            y_train = cp.asarray(y[train_idx], dtype=cp.int32)
            y_val   = cp.asarray(y[val_idx],   dtype=cp.int32)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            new_model = model.__class__(**model.get_params())
            new_model.fit(X_train, y_train)
            y_pred = new_model.predict(X_val)

            fold_acc = cuml_accuracy(y_val, y_pred)
            fold_acc = float(fold_acc)
            accuracies.append(fold_acc)
            print(f"[Fold {fold_idx}] GPU Accuracy = {fold_acc:.5f}")

            cp.get_default_memory_pool().free_all_blocks()

        else:
            X_train = X[train_idx]
            X_val   = X[val_idx]
            y_train = y[train_idx]
            y_val   = y[val_idx]

            scaler = SklearnStandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            new_model = clone(model)
            new_model.fit(X_train, y_train)
            y_pred = new_model.predict(X_val)

            fold_acc = sk_accuracy(y_val, y_pred)
            accuracies.append(fold_acc)
            print(f"[Fold {fold_idx}] CPU Accuracy = {fold_acc:.5f}")

        fold_idx += 1

    return float(np.mean(accuracies))

def save_confusion_matrix(cm, labels, model_name, output_folder="confusionMatrix", use_percent=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.figure(figsize=(12, 10))
    if use_percent:
        sns.heatmap(
            cm, annot=True, cmap='Blues', fmt='.2f',
            xticklabels=labels, yticklabels=labels,
            vmin=0.0, vmax=1.0, annot_kws={"size": 9}
        )
    else:
        sns.heatmap(
            cm, annot=True, cmap='Blues', fmt='g',
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 9}
        )

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14)

    save_path = os.path.join(output_folder, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_train, X_test, y_train, y_test, unique_labels, gpu=False, feature_names=None):
    model_name = model.__class__.__name__
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    print("Predicting on test data...")
    y_pred = model.predict(X_test)

    if gpu:
        acc = cuml_accuracy(y_test, y_pred)
        conf_matrix_count = cuml_confusion_matrix(y_test, y_pred)
        conf_matrix_norm  = cuml_confusion_matrix(y_test, y_pred, normalize='true')

        acc = float(acc)
        conf_matrix_count = cp.asnumpy(conf_matrix_count)
        conf_matrix_norm  = cp.asnumpy(conf_matrix_norm)
        y_test_cpu = cp.asnumpy(y_test)
        y_pred_cpu = cp.asnumpy(y_pred)
    else:
        acc = sk_accuracy(y_test, y_pred)
        conf_matrix_count = sk_confusion_matrix(y_test, y_pred)
        conf_matrix_norm  = sk_confusion_matrix(y_test, y_pred, normalize='true')
        y_test_cpu = y_test
        y_pred_cpu = y_pred

    print(f"Accuracy: {acc:.5f}")
    print("Confusion Matrix (Count):")
    print(conf_matrix_count)
    print("Confusion Matrix (Normalized):")
    print(conf_matrix_norm)
    print("Classification Report:")
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=unique_labels, digits=5))

    save_confusion_matrix(
        conf_matrix_count, unique_labels, f"{model_name}_count",
        "confusionMatrix_count_training_20250501_162409", use_percent=False
    )
    save_confusion_matrix(
        conf_matrix_norm,  unique_labels, f"{model_name}_norm",
        "confusionMatrix_norm_training_20250501_162409", use_percent=True
    )

    if feature_names is not None:
        if hasattr(model, "feature_importances_"):
            print("\nFeature Importances:")
            fi = model.feature_importances_
            for feat, val in zip(feature_names, fi):
                print(f"{feat}: {val:.5f}")
        elif hasattr(model, "coef_"):
            print("\nFeature Coefficients:")
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            for feat, cval in zip(feature_names, coef):
                print(f"{feat}: {cval:.5f}")
        else:
            print(f"\nNo feature_importances_ or coef_ in {model_name} (GPU={gpu}).")
            print("If needed, use permutation_importance or a CPU-converted model for feature importance.")

    model_results.append({
        "model": model,
        "model_name": model_name,
        "accuracy": acc,
        "gpu": gpu,
        "cv_accuracy": None
    })
    return acc

def train_all_models(csv_dir):
    print("Merging CSV files...")
    data = merge_csvs(csv_dir)

    print("Extracting OS/Messenger labels...")
    data["os"], data["messenger"] = zip(*data["file"].apply(extract_labels))
    data["combined_label"] = data["os"] + "_" + data["messenger"]

    print("Checking missing values...")
    if data.isnull().sum().sum() > 0:
        print("Filling missing values with column means...")
        data.fillna(data.mean(), inplace=True)

    print("Splitting features and labels...")
    X = data.drop(columns=["file", "os", "messenger", "combined_label"])
    y_combined = data["combined_label"]
    y_combined, unique_labels = pd.factorize(y_combined)

    print("\nConverting data to GPU arrays...")
    X_cupy = cp.asarray(X.values, dtype=cp.float32)
    y_cupy = cp.asarray(y_combined, dtype=cp.int32)

    print("Split (GPU)...")
    X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = train_test_split(
        X_cupy, y_cupy, test_size=0.2, random_state=42
    )
    scaler_gpu = StandardScaler()
    X_train_gpu = scaler_gpu.fit_transform(X_train_gpu)
    X_test_gpu = scaler_gpu.transform(X_test_gpu)

    gpu_models = [
        SVC(kernel="linear", C=1.0, gamma="scale"),
        RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42),
        KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        LogisticRegression()
    ]

    for model in gpu_models:
        name = model.__class__.__name__
        print(f"\n===== GPU Model: {name} =====")
        test_acc = evaluate_model(
            model,
            X_train_gpu, X_test_gpu,
            y_train_gpu, y_test_gpu,
            unique_labels, gpu=True
        )

        print(f"\nPerforming k-Fold CV on {name} (GPU)...")
        cv_acc = cross_validate_model(
            model=model,
            X=X_cupy,
            y=y_cupy,
            folds=5,
            gpu=True,
            unique_labels=unique_labels
        )
        print(f"[{name}] Mean CV Accuracy = {cv_acc:.5f}")

        for info in model_results:
            if info["model_name"] == name and info["gpu"] and info["cv_accuracy"] is None:
                info["cv_accuracy"] = cv_acc
                break

        print(f"\nConverting GPU model '{name}' to CPU model for feature importance...")
        if name == "SVC":
            cpu_model = skSVC(kernel="linear", C=1.0, gamma="scale", probability=False)
        elif name == "RandomForestClassifier":
            cpu_model = skRandomForest(n_estimators=100, max_depth=16, random_state=42)
        elif name == "KNeighborsClassifier":
            print("KNN does not support feature importance. Skipping CPU conversion.")
            continue
        elif name == "LogisticRegression":
            cpu_model = skLogisticRegression()
        else:
            print("No CPU conversion available for this GPU model.")
            continue

        X_train_gpu_cpu = cp.asnumpy(X_train_gpu)
        y_train_gpu_cpu = cp.asnumpy(y_train_gpu)
        cpu_model.fit(X_train_gpu_cpu, y_train_gpu_cpu)

        if hasattr(cpu_model, "feature_importances_"):
            print("\nFeature Importances (converted CPU model):")
            for feat, val in zip(X.columns, cpu_model.feature_importances_):
                print(f"{feat}: {val:.5f}")
        elif hasattr(cpu_model, "coef_"):
            print("\nFeature Coefficients (converted CPU model):")
            coef_vals = cpu_model.coef_
            if coef_vals.ndim > 1:
                coef_vals = np.mean(np.abs(coef_vals), axis=0)
            for feat, cval in zip(X.columns, coef_vals):
                print(f"{feat}: {cval:.5f}")
        else:
            print("This CPU model does not provide feature_importances_ or coef_.")

    print("\nSplitting data (CPU)...")
    X_cpu = X.values
    y_cpu = y_combined

    X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = SklearnTrainTestSplit(
        X_cpu, y_cpu, test_size=0.2, random_state=42
    )
    scaler_cpu = SklearnStandardScaler()
    X_train_cpu = scaler_cpu.fit_transform(X_train_cpu)
    X_test_cpu  = scaler_cpu.transform(X_test_cpu)

    cpu_models = [
        ExtraTreesClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        BernoulliNB()
    ]

    for model in cpu_models:
        name = model.__class__.__name__
        print(f"\n===== CPU Model: {name} =====")
        test_acc = evaluate_model(
            model,
            X_train_cpu, X_test_cpu,
            y_train_cpu, y_test_cpu,
            unique_labels, gpu=False,
            feature_names=X.columns
        )

        print(f"\nPerforming k-Fold CV on {name} (CPU)...")
        cv_acc = cross_validate_model(
            model=model,
            X=X_cpu,
            y=y_cpu,
            folds=5,
            gpu=False,
            unique_labels=unique_labels
        )
        print(f"[{name}] Mean CV Accuracy = {cv_acc:.5f}")

        for info in model_results:
            if info["model_name"] == name and not info["gpu"] and info["cv_accuracy"] is None:
                info["cv_accuracy"] = cv_acc
                break

    print("\nSelecting Top 3 Models by single train-test accuracy...")
    sorted_results = sorted(model_results, key=lambda x: x["accuracy"], reverse=True)
    top_3 = sorted_results[:3]
    print("Top 3 Models:")
    for i, info in enumerate(top_3, 1):
        print(f"{i}. {info['model_name']} - Accuracy: {info['accuracy']:.5f} (CV: {info['cv_accuracy']})")

    print("\nEnsemble with Top 3 Models (VotingClassifier)...")
    estimators = []
    for idx, info in enumerate(top_3):
        is_gpu = info["gpu"]
        m_name = info["model_name"]
        old_model = info["model"]

        if is_gpu:
            if "SVC" in m_name:
                new_model = skSVC(kernel="linear", C=1.0, gamma="scale", probability=False)
            elif "RandomForestClassifier" in m_name:
                new_model = skRandomForest(n_estimators=100, max_depth=16, random_state=42)
            elif "KNeighborsClassifier" in m_name:
                new_model = skKNN(n_neighbors=5, metric="euclidean")
            elif "LogisticRegression" in m_name:
                new_model = skLogisticRegression()
            else:
                raise ValueError(f"Cannot convert GPU model '{m_name}' to CPU.")
        else:
            new_model = clone(old_model)

        estimators.append((f"model_{idx}", new_model))

    voting_clf = VotingClassifier(estimators=estimators, voting='hard')
    voting_clf.fit(X_train_cpu, y_train_cpu)
    y_pred_ens = voting_clf.predict(X_test_cpu)

    ens_acc = sk_accuracy(y_test_cpu, y_pred_ens)
    cm_ens = sk_confusion_matrix(y_test_cpu, y_pred_ens)
    cm_ens_norm = sk_confusion_matrix(y_test_cpu, y_pred_ens, normalize='true')

    print(f"Ensemble Accuracy: {ens_acc:.5f}")
    print("Confusion Matrix (Count):")
    print(cm_ens)
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test_cpu, y_pred_ens, target_names=unique_labels, digits=5))

    model_results.append({
        "model": voting_clf,
        "model_name": "VotingClassifier_Ensemble",
        "accuracy": ens_acc,
        "gpu": False,
        "cv_accuracy": None
    })

    save_confusion_matrix(
        cm_ens, unique_labels,
        model_name="VotingClassifier_Ensemble_count",
        output_folder="confusionMatrix_count_training_20250501_162409",
        use_percent=False
    )
    save_confusion_matrix(
        cm_ens_norm, unique_labels,
        model_name="VotingClassifier_Ensemble_norm",
        output_folder="confusionMatrix_norm_training_20250501_162409",
        use_percent=True
    )

    print("\nCross Validation (CPU) for Voting Ensemble...")
    ens_cv_acc = cross_validate_model(
        model=voting_clf,
        X=X_cpu,
        y=y_cpu,
        folds=5,
        gpu=False,
        unique_labels=unique_labels
    )
    print(f"[VotingClassifier Ensemble] Mean CV Accuracy = {ens_cv_acc:.5f}")
    for info in model_results:
        if info["model_name"] == "VotingClassifier_Ensemble" and info["cv_accuracy"] is None:
            info["cv_accuracy"] = ens_cv_acc
            break

def main():
    csv_dir = "./preprocessing_20250501_162409"
    with open("result_training_20250501_162409.txt", "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        train_all_models(csv_dir)
        sys.stdout = original_stdout

    print("All results have been saved to 'result_training_20250501_162409.txt'.")
    print("Confusion matrix images are saved in 'confusionMatrix_count_20250501_162409' (count) "
          "and 'confusionMatrix_norm_20250501_162409' (normalized).")

if __name__ == "__main__":
    main()
