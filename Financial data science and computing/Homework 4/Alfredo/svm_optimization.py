import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ============================================================
# 0) Load data
# ============================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

news = pd.read_csv(os.path.join(BASE_PATH, "news.csv"))
news_test = pd.read_csv(os.path.join(BASE_PATH, "news_test.csv"))
news_topics = pd.read_csv(os.path.join(BASE_PATH, "news_topics.csv"))

news["article"] = news["article"].fillna("").astype(str).str.strip()
news_test["article"] = news_test["article"].fillna("").astype(str).str.strip()
news_topics["cat"] = news_topics["cat"].astype(str).str.strip()

TOP_CODES = ["CCAT", "ECAT", "GCAT", "MCAT"]

# ============================================================
# 1) Build Task 1a labels
# ============================================================
def make_multilabel_matrix(article_ids, mapping_df, labels):
    y = pd.DataFrame({"id": article_ids})

    tmp = (
        mapping_df.loc[
            mapping_df["id"].isin(article_ids) & mapping_df["cat"].isin(labels),
            ["id", "cat"]
        ]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="id", columns="cat", values="value")
        .fillna(0)
        .reset_index()
    )

    out = y.merge(tmp, on="id", how="left").fillna(0)

    for lab in labels:
        if lab not in out.columns:
            out[lab] = 0

    out[labels] = out[labels].astype(int)
    return out

train_y_top = make_multilabel_matrix(news["id"], news_topics, TOP_CODES)
test_y_top = make_multilabel_matrix(news_test["id"], news_topics, TOP_CODES)

train_df = news.merge(train_y_top, on="id", how="left")
test_df = news_test.merge(test_y_top, on="id", how="left")

# ============================================================
# 2) Train / validation split
# ============================================================
train_sub, val_sub = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["CCAT"]
)

print("Train subset shape:", train_sub.shape)
print("Validation subset shape:", val_sub.shape)
print("Test shape:", test_df.shape)

# ============================================================
# 3) Metrics
# ============================================================
def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

def overall_from_per_label(df):
    macro_accuracy = df["Accuracy"].mean()
    macro_precision = df["Precision"].mean()
    macro_recall = df["Recall"].mean()
    macro_f1 = df["F1"].mean()

    tp = df["TP"].sum()
    tn = df["TN"].sum()
    fp = df["FP"].sum()
    fn = df["FN"].sum()

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if pd.notna(micro_precision) and pd.notna(micro_recall) and (micro_precision + micro_recall) > 0
        else np.nan
    )
    micro_accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "Macro_Accuracy": macro_accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Micro_Accuracy": micro_accuracy,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Micro_F1": micro_f1
    }

# ============================================================
# 4) Threshold helper
# ============================================================
def choose_best_threshold(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    unique_scores = np.unique(scores)

    if len(unique_scores) > 200:
        thresholds = np.quantile(unique_scores, np.linspace(0.05, 0.95, 101))
        thresholds = np.unique(thresholds)
    else:
        thresholds = unique_scores

    best_threshold = 0.0
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, best_f1

# ============================================================
# 5) Representation builders
# ============================================================
def build_tfidf(train_text, other_text, min_df=2):
    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        token_pattern=r"(?u)\b\w+\b"
    )

    X_train = vectorizer.fit_transform(train_text)
    X_other = vectorizer.transform(other_text)
    return X_train, X_other

def build_tfidf_pca(train_text, other_text, n_components, min_df=2):
    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        token_pattern=r"(?u)\b\w+\b"
    )

    X_train_tfidf = vectorizer.fit_transform(train_text)
    X_other_tfidf = vectorizer.transform(other_text)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train = svd.fit_transform(X_train_tfidf)
    X_other = svd.transform(X_other_tfidf)
    return X_train, X_other

# ============================================================
# 6) Tune one SVM representation on validation
# ============================================================
def tune_svm_pipeline(representation_name):
    validation_rows = []

    if representation_name == "TFIDF":
        param_grid = [
            {"Representation": "TFIDF", "C": c, "Class_Weight": cw, "Components": 0}
            for c in [0.1, 1.0, 5.0, 10.0]
            for cw in [None, "balanced"]
        ]
    elif representation_name == "TFIDF+PCA":
        param_grid = [
            {"Representation": "TFIDF+PCA", "C": c, "Class_Weight": cw, "Components": comp}
            for comp in [100, 200, 300]
            for c in [0.1, 1.0, 5.0, 10.0]
            for cw in [None, "balanced"]
        ]
    else:
        raise ValueError("Unknown representation")

    for params in param_grid:
        print("\nTuning with:", params)

        if params["Representation"] == "TFIDF":
            X_train_rep, X_val_rep = build_tfidf(
                train_sub["article"],
                val_sub["article"]
            )
        else:
            X_train_rep, X_val_rep = build_tfidf_pca(
                train_sub["article"],
                val_sub["article"],
                params["Components"]
            )

        per_label_rows = []

        for lab in TOP_CODES:
            y_train = train_sub[lab].to_numpy()
            y_val = val_sub[lab].to_numpy()

            clf = LinearSVC(
                C=params["C"],
                class_weight=params["Class_Weight"],
                random_state=42
            )

            clf.fit(X_train_rep, y_train)
            val_scores = clf.decision_function(X_val_rep)
            val_pred = (val_scores >= 0.0).astype(int)

            row = binary_metrics(y_val, val_pred)
            row["Label"] = lab
            per_label_rows.append(row)

        per_label_df = pd.DataFrame(per_label_rows)
        overall = overall_from_per_label(per_label_df)

        validation_rows.append({
            "Model": "SVM",
            "Representation": params["Representation"],
            "Components": params["Components"],
            "C": params["C"],
            "Class_Weight": str(params["Class_Weight"]),
            **overall
        })

    tuning_df = pd.DataFrame(validation_rows).sort_values(
        by=["Macro_F1", "Micro_F1"],
        ascending=False
    )

    print(f"\nValidation tuning results for {representation_name}:")
    print(tuning_df)

    return tuning_df

# ============================================================
# 7) Tune TFIDF and TFIDF+PCA SVM
# ============================================================
svm_tuning_tfidf = tune_svm_pipeline("TFIDF")
svm_tuning_pca = tune_svm_pipeline("TFIDF+PCA")

svm_tuning_all = pd.concat([svm_tuning_tfidf, svm_tuning_pca], ignore_index=True)
svm_tuning_all = svm_tuning_all.sort_values(
    by=["Macro_F1", "Micro_F1"],
    ascending=False
)

print("\nCombined SVM tuning results:")
print(svm_tuning_all)

svm_tuning_all.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_tuning_results.csv"),
    index=False
)

best_tfidf = svm_tuning_tfidf.iloc[0].to_dict()
best_pca = svm_tuning_pca.iloc[0].to_dict()

print("\nBest TFIDF SVM setting:")
print(best_tfidf)

print("\nBest TFIDF+PCA SVM setting:")
print(best_pca)

# ============================================================
# 8) Threshold tuning for best TFIDF and best PCA SVM
# ============================================================
def tune_thresholds(best_params):
    representation = best_params["Representation"]
    best_C = float(best_params["C"])
    best_class_weight = None if best_params["Class_Weight"] == "None" else "balanced"
    best_components = int(best_params["Components"])

    if representation == "TFIDF":
        X_train_rep, X_val_rep = build_tfidf(
            train_sub["article"],
            val_sub["article"]
        )
    else:
        X_train_rep, X_val_rep = build_tfidf_pca(
            train_sub["article"],
            val_sub["article"],
            best_components
        )

    threshold_rows = []

    for lab in TOP_CODES:
        print(f"Tuning threshold for {representation} - {lab}")

        y_train = train_sub[lab].to_numpy()
        y_val = val_sub[lab].to_numpy()

        clf = LinearSVC(
            C=best_C,
            class_weight=best_class_weight,
            random_state=42
        )

        clf.fit(X_train_rep, y_train)
        val_scores = clf.decision_function(X_val_rep)

        best_threshold, best_val_f1 = choose_best_threshold(y_val, val_scores)

        threshold_rows.append({
            "Representation": representation,
            "Label": lab,
            "Threshold": best_threshold,
            "Validation_F1_At_Threshold": best_val_f1
        })

    return pd.DataFrame(threshold_rows)

thresholds_tfidf = tune_thresholds(best_tfidf)
thresholds_pca = tune_thresholds(best_pca)
thresholds_all = pd.concat([thresholds_tfidf, thresholds_pca], ignore_index=True)

print("\nThreshold tables:")
print(thresholds_all)

thresholds_all.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_thresholds.csv"),
    index=False
)

# ============================================================
# 9) Final test evaluation for both tuned SVM pipelines
# ============================================================
def final_test_eval(best_params, threshold_table):
    representation = best_params["Representation"]
    best_C = float(best_params["C"])
    best_class_weight = None if best_params["Class_Weight"] == "None" else "balanced"
    best_components = int(best_params["Components"])

    if representation == "TFIDF":
        X_train_rep, X_test_rep = build_tfidf(
            train_df["article"],
            test_df["article"]
        )
    else:
        X_train_rep, X_test_rep = build_tfidf_pca(
            train_df["article"],
            test_df["article"],
            best_components
        )

    threshold_map = dict(zip(threshold_table["Label"], threshold_table["Threshold"]))

    per_label_rows = []

    for lab in TOP_CODES:
        print(f"Final tuned SVM on test for {representation} - {lab}")

        y_train = train_df[lab].to_numpy()
        y_test = test_df[lab].to_numpy()

        clf = LinearSVC(
            C=best_C,
            class_weight=best_class_weight,
            random_state=42
        )

        clf.fit(X_train_rep, y_train)
        test_scores = clf.decision_function(X_test_rep)

        threshold = threshold_map[lab]
        y_pred = (test_scores >= threshold).astype(int)

        row = binary_metrics(y_test, y_pred)
        row["Model"] = "SVM"
        row["Representation"] = representation
        row["Components"] = best_components
        row["C"] = best_C
        row["Class_Weight"] = str(best_class_weight)
        row["Label"] = lab
        row["Threshold"] = threshold
        per_label_rows.append(row)

    per_label_df = pd.DataFrame(per_label_rows)
    overall = overall_from_per_label(per_label_df)
    overall["Model"] = "SVM"
    overall["Representation"] = representation
    overall["Components"] = best_components
    overall["C"] = best_C
    overall["Class_Weight"] = str(best_class_weight)

    return per_label_df, pd.DataFrame([overall])

tfidf_per_label, tfidf_overall = final_test_eval(best_tfidf, thresholds_tfidf)
pca_per_label, pca_overall = final_test_eval(best_pca, thresholds_pca)

final_svm_per_label = pd.concat([tfidf_per_label, pca_per_label], ignore_index=True)
final_svm_overall = pd.concat([tfidf_overall, pca_overall], ignore_index=True)
final_svm_overall = final_svm_overall.sort_values(
    by=["Macro_F1", "Micro_F1"],
    ascending=False
)

print("\nFinal tuned SVM comparison on test:")
print(final_svm_overall)

print("\nFinal tuned SVM per-label results:")
print(final_svm_per_label)

# ============================================================
# 10) Save outputs
# ============================================================
final_svm_overall.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_final_overall.csv"),
    index=False
)

final_svm_per_label.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_final_per_label.csv"),
    index=False
)

print("\nSaved:")
print(" - task1a_svm_tfidf_vs_pca_tuning_results.csv")
print(" - task1a_svm_tfidf_vs_pca_thresholds.csv")
print(" - task1a_svm_tfidf_vs_pca_final_overall.csv")
print(" - task1a_svm_tfidf_vs_pca_final_per_label.csv")