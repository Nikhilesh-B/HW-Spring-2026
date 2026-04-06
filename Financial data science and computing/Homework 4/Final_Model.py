import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ============================================================
# 0) Final selected model settings
# ============================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

FINAL_SVM_C = 1.0
FINAL_SVM_CLASS_WEIGHT = None
MIN_DF = 2
MIN_TRAIN_POS_TASK1B = 10

# ============================================================
# 1) Load data
# ============================================================
def read_topics_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 4:
            parts = parts + [""] * (4 - len(parts))

        rows.append({
            "id_cat": parts[0].strip(),
            "parent": parts[1].strip(),
            "child": parts[2].strip(),
            "description": ",".join(parts[3:]).strip()
        })

    return pd.DataFrame(rows)

news = pd.read_csv(os.path.join(BASE_PATH, "news.csv"))
news_test = pd.read_csv(os.path.join(BASE_PATH, "news_test.csv"))
news_topics = pd.read_csv(os.path.join(BASE_PATH, "news_topics.csv"))
topics = read_topics_csv(os.path.join(BASE_PATH, "topics.csv"))

news["article"] = news["article"].fillna("").astype(str).str.strip()
news_test["article"] = news_test["article"].fillna("").astype(str).str.strip()
news_topics["cat"] = news_topics["cat"].astype(str).str.strip()
topics["parent"] = topics["parent"].astype(str).str.strip()
topics["child"] = topics["child"].astype(str).str.strip()

print("Loaded files.")
print("news shape:", news.shape)
print("news_test shape:", news_test.shape)
print("news_topics shape:", news_topics.shape)
print("topics shape:", topics.shape)

# ============================================================
# 2) Helpers
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


def build_tfidf(train_text, other_text, min_df=MIN_DF):
    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        token_pattern=r"(?u)\b\w+\b"
    )

    X_train = vectorizer.fit_transform(train_text)
    X_other = vectorizer.transform(other_text)
    return X_train, X_other

# ============================================================
# 3) Build leaf-topic labels
# ============================================================
valid_parents = set(topics.loc[~topics["parent"].isin(["", "None", "Root"]), "parent"])
valid_children = set(topics.loc[~topics["child"].isin(["", "None", "Root"]), "child"])
leaf_topics_all = sorted(valid_children - valid_parents)
leaf_topics_in_data = sorted(set(news_topics["cat"]).intersection(leaf_topics_all))

print("\nLeaf topics in hierarchy:", len(leaf_topics_all))
print("Leaf topics appearing in data:", len(leaf_topics_in_data))

train_y_leaf = make_multilabel_matrix(news["id"], news_topics, leaf_topics_in_data)
test_y_leaf = make_multilabel_matrix(news_test["id"], news_topics, leaf_topics_in_data)

train_df_leaf = news.merge(train_y_leaf, on="id", how="left")
test_df_leaf = news_test.merge(test_y_leaf, on="id", how="left")

# ============================================================
# 4) Train/validation split for threshold tuning
# ============================================================
train_sub_leaf, val_sub_leaf = train_test_split(
    train_df_leaf,
    test_size=0.2,
    random_state=42
)

X_train_sub_leaf, X_val_leaf = build_tfidf(train_sub_leaf["article"], val_sub_leaf["article"])
X_train_full_leaf, X_test_leaf = build_tfidf(train_df_leaf["article"], test_df_leaf["article"])

# ============================================================
# 5) Tune thresholds per leaf label
# ============================================================
threshold_rows_leaf = []
skipped_info_leaf = []

for lab in leaf_topics_in_data:
    y_train = train_sub_leaf[lab].to_numpy()
    y_val = val_sub_leaf[lab].to_numpy()

    n_pos_train = int(y_train.sum())
    n_pos_val = int(y_val.sum())

    print(f"Tuning threshold for {lab} | train positives = {n_pos_train} | val positives = {n_pos_val}")

    if n_pos_train < MIN_TRAIN_POS_TASK1B:
        skipped_info_leaf.append({
            "Label": lab,
            "Train_Pos": n_pos_train,
            "Validation_Pos": n_pos_val,
            "Reason": f"Skipped: fewer than {MIN_TRAIN_POS_TASK1B} training positives"
        })
        continue

    clf = LinearSVC(
        C=FINAL_SVM_C,
        class_weight=FINAL_SVM_CLASS_WEIGHT,
        random_state=42
    )
    clf.fit(X_train_sub_leaf, y_train)

    val_scores = clf.decision_function(X_val_leaf)
    threshold, val_f1 = choose_best_threshold(y_val, val_scores)

    threshold_rows_leaf.append({
        "Label": lab,
        "Threshold": threshold,
        "Validation_F1_At_Threshold": val_f1,
        "Train_Pos": n_pos_train,
        "Validation_Pos": n_pos_val
    })

threshold_table_leaf = pd.DataFrame(threshold_rows_leaf)
skipped_df_leaf = pd.DataFrame(skipped_info_leaf)

print("\nTask 1b threshold table head:")
print(threshold_table_leaf.head())

if len(skipped_df_leaf) > 0:
    print("\nTask 1b skipped labels:")
    print(skipped_df_leaf)

threshold_map_leaf = dict(zip(threshold_table_leaf["Label"], threshold_table_leaf["Threshold"]))

# ============================================================
# 6) Final Task 1b evaluation
# ============================================================
task1b_per_label_rows = []

for lab in leaf_topics_in_data:
    if lab not in threshold_map_leaf:
        continue

    y_train = train_df_leaf[lab].to_numpy()
    y_test = test_df_leaf[lab].to_numpy()

    n_pos_train = int(y_train.sum())
    n_pos_test = int(y_test.sum())

    print(f"Final Task 1b test run for {lab} | train positives = {n_pos_train} | test positives = {n_pos_test}")

    clf = LinearSVC(
        C=FINAL_SVM_C,
        class_weight=FINAL_SVM_CLASS_WEIGHT,
        random_state=42
    )
    clf.fit(X_train_full_leaf, y_train)

    test_scores = clf.decision_function(X_test_leaf)
    threshold = threshold_map_leaf[lab]
    y_pred = (test_scores >= threshold).astype(int)

    row = binary_metrics(y_test, y_pred)
    row["Model"] = "SVM"
    row["Representation"] = "TFIDF"
    row["Components"] = 0
    row["C"] = FINAL_SVM_C
    row["Class_Weight"] = str(FINAL_SVM_CLASS_WEIGHT)
    row["Label"] = lab
    row["Threshold"] = threshold
    row["Train_Pos"] = n_pos_train
    row["Test_Pos"] = n_pos_test
    task1b_per_label_rows.append(row)

task1b_final_per_label = pd.DataFrame(task1b_per_label_rows)
task1b_final_overall = pd.DataFrame([{
    "Model": "SVM",
    "Representation": "TFIDF",
    "Components": 0,
    "C": FINAL_SVM_C,
    "Class_Weight": str(FINAL_SVM_CLASS_WEIGHT),
    "Labels_Evaluated": len(task1b_final_per_label),
    "Labels_Skipped": len(skipped_df_leaf),
    **overall_from_per_label(task1b_final_per_label)
}])

task1b_best = task1b_final_per_label.sort_values(
    by=["F1", "Recall", "Precision"],
    ascending=[False, False, False]
)

task1b_worst = task1b_final_per_label.sort_values(
    by=["F1", "Recall", "Precision"],
    ascending=[True, True, True]
)

print("\nTask 1b final overall:")
print(task1b_final_overall)

print("\nTop 15 Task 1b labels:")
print(task1b_best[["Label", "Train_Pos", "Test_Pos", "Precision", "Recall", "F1"]].head(15))

print("\nBottom 15 Task 1b labels:")
print(task1b_worst[["Label", "Train_Pos", "Test_Pos", "Precision", "Recall", "F1"]].head(15))

# ============================================================
# 7) Save outputs
# ============================================================
threshold_table_leaf.to_csv(
    os.path.join(BASE_PATH, "task1b_svm_thresholds.csv"),
    index=False
)
task1b_final_per_label.to_csv(
    os.path.join(BASE_PATH, "task1b_svm_final_per_label.csv"),
    index=False
)
task1b_final_overall.to_csv(
    os.path.join(BASE_PATH, "task1b_svm_final_overall.csv"),
    index=False
)
task1b_best.to_csv(
    os.path.join(BASE_PATH, "task1b_svm_best_labels.csv"),
    index=False
)
task1b_worst.to_csv(
    os.path.join(BASE_PATH, "task1b_svm_worst_labels.csv"),
    index=False
)
if len(skipped_df_leaf) > 0:
    skipped_df_leaf.to_csv(
        os.path.join(BASE_PATH, "task1b_svm_skipped_labels.csv"),
        index=False
    )

print("\nSaved final outputs.")