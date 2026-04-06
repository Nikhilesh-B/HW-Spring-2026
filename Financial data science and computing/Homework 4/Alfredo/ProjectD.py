import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# ============================================================
# 0) CONFIG
# ============================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TOP_CODES = ["CCAT", "ECAT", "GCAT", "MCAT"]

# Final chosen SVM settings from Task 1a tuning
FINAL_SVM_C = 1.0
FINAL_SVM_CLASS_WEIGHT = None
FINAL_SVM_USE_PCA = False
FINAL_SVM_PCA_COMPONENTS = 300   # only used if FINAL_SVM_USE_PCA = True

MIN_DF = 2
MIN_TRAIN_POS_TASK1B = 10


# ============================================================
# 1) DATA LOADING
# ============================================================
def read_plain_csv(path):
    return pd.read_csv(path)

def read_topics_csv(path):
    # topics.csv may contain commas in description, so parse manually
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header = lines[0]
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 4:
            parts = parts + [""] * (4 - len(parts))
        row = {
            "id_cat": parts[0].strip(),
            "parent": parts[1].strip(),
            "child": parts[2].strip(),
            "description": ",".join(parts[3:]).strip()
        }
        rows.append(row)

    return pd.DataFrame(rows)

news = read_plain_csv(os.path.join(BASE_PATH, "news.csv"))
news_test = read_plain_csv(os.path.join(BASE_PATH, "news_test.csv"))
news_topics = read_plain_csv(os.path.join(BASE_PATH, "news_topics.csv"))
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
# 2) HELPERS
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


def build_tfidf_pca(train_text, other_text, n_components, min_df=MIN_DF):
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


def get_models():
    return {
        "GLMNET": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=2000,
            random_state=42
        ),
        "SVM": LinearSVC(
            C=1.0,
            random_state=42
        ),
        "MAXENT": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
            random_state=42
        ),
        "BOOSTING": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }


# ============================================================
# 3) TASK 1A LABELS
# ============================================================
train_y_top = make_multilabel_matrix(news["id"], news_topics, TOP_CODES)
test_y_top = make_multilabel_matrix(news_test["id"], news_topics, TOP_CODES)

train_df_top = news.merge(train_y_top, on="id", how="left")
test_df_top = news_test.merge(test_y_top, on="id", how="left")

print("\nTask 1a label counts in train:")
print(train_df_top[TOP_CODES].sum())

print("\nTask 1a label counts in test:")
print(test_df_top[TOP_CODES].sum())


# ============================================================
# 4) TASK 1A: BASE MODEL COMPARISON
#    Compare 4 models under:
#    - TFIDF
#    - TFIDF + PCA
# ============================================================
print("\n" + "=" * 70)
print("TASK 1A - BASE MODEL COMPARISON")
print("=" * 70)

X_train_tfidf, X_test_tfidf = build_tfidf(news["article"], news_test["article"])
X_train_pca, X_test_pca = build_tfidf_pca(news["article"], news_test["article"], 300)

representations = {
    "TFIDF": (X_train_tfidf, X_test_tfidf, 0),
    "TFIDF+PCA(300)": (X_train_pca, X_test_pca, 300)
}

comparison_per_label_rows = []
comparison_overall_rows = []

for rep_name, (X_train_rep, X_test_rep, comps) in representations.items():
    print(f"\nRepresentation: {rep_name}")

    for model_name, model in get_models().items():
        print(f"  Running {model_name}")
        per_label_rows = []

        for lab in TOP_CODES:
            y_train = train_y_top[lab].to_numpy()
            y_test = test_y_top[lab].to_numpy()

            model.fit(X_train_rep, y_train)
            y_pred = model.predict(X_test_rep)

            row = binary_metrics(y_test, y_pred)
            row["Model"] = model_name
            row["Label"] = lab
            row["Representation"] = rep_name
            row["Components"] = comps
            per_label_rows.append(row)
            comparison_per_label_rows.append(row)

        per_label_df = pd.DataFrame(per_label_rows)
        overall = overall_from_per_label(per_label_df)
        overall["Model"] = model_name
        overall["Representation"] = rep_name
        overall["Components"] = comps
        comparison_overall_rows.append(overall)

comparison_per_label = pd.DataFrame(comparison_per_label_rows)
comparison_overall = pd.DataFrame(comparison_overall_rows)
comparison_overall = comparison_overall.sort_values(
    by=["Representation", "Macro_F1", "Micro_F1"],
    ascending=[True, False, False]
)

print("\nTask 1a overall comparison:")
print(comparison_overall)

comparison_overall.to_csv(
    os.path.join(BASE_PATH, "task1a_pca_vs_nopca_model_comparison_overall.csv"),
    index=False
)
comparison_per_label.to_csv(
    os.path.join(BASE_PATH, "task1a_pca_vs_nopca_model_comparison_per_label.csv"),
    index=False
)


# ============================================================
# 5) TASK 1A: TUNE SVM WITH AND WITHOUT PCA
#    Tune:
#    - C
#    - class_weight
#    - thresholds
# ============================================================
print("\n" + "=" * 70)
print("TASK 1A - SVM TUNING")
print("=" * 70)

train_sub_top, val_sub_top = train_test_split(
    train_df_top,
    test_size=0.2,
    random_state=42,
    stratify=train_df_top["CCAT"]
)

def tune_svm_pipeline(representation_name):
    rows = []

    if representation_name == "TFIDF":
        param_grid = [
            {"Representation": "TFIDF", "C": c, "Class_Weight": cw, "Components": 0}
            for c in [0.1, 1.0, 5.0, 10.0]
            for cw in [None, "balanced"]
        ]
    else:
        param_grid = [
            {"Representation": "TFIDF+PCA", "C": c, "Class_Weight": cw, "Components": comp}
            for comp in [100, 200, 300]
            for c in [0.1, 1.0, 5.0, 10.0]
            for cw in [None, "balanced"]
        ]

    for params in param_grid:
        print("Tuning with:", params)

        if params["Representation"] == "TFIDF":
            X_train_rep, X_val_rep = build_tfidf(train_sub_top["article"], val_sub_top["article"])
        else:
            X_train_rep, X_val_rep = build_tfidf_pca(
                train_sub_top["article"], val_sub_top["article"], params["Components"]
            )

        per_label_rows = []
        for lab in TOP_CODES:
            y_train = train_sub_top[lab].to_numpy()
            y_val = val_sub_top[lab].to_numpy()

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
        overall["Model"] = "SVM"
        overall["Representation"] = params["Representation"]
        overall["Components"] = params["Components"]
        overall["C"] = params["C"]
        overall["Class_Weight"] = str(params["Class_Weight"])
        rows.append(overall)

    tuning_df = pd.DataFrame(rows).sort_values(
        by=["Macro_F1", "Micro_F1"],
        ascending=False
    )
    return tuning_df

svm_tuning_tfidf = tune_svm_pipeline("TFIDF")
svm_tuning_pca = tune_svm_pipeline("TFIDF+PCA")
svm_tuning_all = pd.concat([svm_tuning_tfidf, svm_tuning_pca], ignore_index=True)
svm_tuning_all = svm_tuning_all.sort_values(by=["Macro_F1", "Micro_F1"], ascending=False)

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

def tune_thresholds(best_params):
    representation = best_params["Representation"]
    best_C = float(best_params["C"])
    best_class_weight = None if best_params["Class_Weight"] == "None" else "balanced"
    best_components = int(best_params["Components"])

    if representation == "TFIDF":
        X_train_rep, X_val_rep = build_tfidf(train_sub_top["article"], val_sub_top["article"])
    else:
        X_train_rep, X_val_rep = build_tfidf_pca(
            train_sub_top["article"], val_sub_top["article"], best_components
        )

    rows = []
    for lab in TOP_CODES:
        y_train = train_sub_top[lab].to_numpy()
        y_val = val_sub_top[lab].to_numpy()

        clf = LinearSVC(
            C=best_C,
            class_weight=best_class_weight,
            random_state=42
        )
        clf.fit(X_train_rep, y_train)
        val_scores = clf.decision_function(X_val_rep)

        threshold, val_f1 = choose_best_threshold(y_val, val_scores)
        rows.append({
            "Representation": representation,
            "Label": lab,
            "Threshold": threshold,
            "Validation_F1_At_Threshold": val_f1
        })

    return pd.DataFrame(rows)

thresholds_tfidf = tune_thresholds(best_tfidf)
thresholds_pca = tune_thresholds(best_pca)
thresholds_all = pd.concat([thresholds_tfidf, thresholds_pca], ignore_index=True)

print("\nThreshold tables:")
print(thresholds_all)

thresholds_all.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_thresholds.csv"),
    index=False
)

def final_test_eval(best_params, threshold_table):
    representation = best_params["Representation"]
    best_C = float(best_params["C"])
    best_class_weight = None if best_params["Class_Weight"] == "None" else "balanced"
    best_components = int(best_params["Components"])

    if representation == "TFIDF":
        X_train_rep, X_test_rep = build_tfidf(train_df_top["article"], test_df_top["article"])
    else:
        X_train_rep, X_test_rep = build_tfidf_pca(
            train_df_top["article"], test_df_top["article"], best_components
        )

    threshold_map = dict(zip(threshold_table["Label"], threshold_table["Threshold"]))

    per_label_rows = []
    for lab in TOP_CODES:
        y_train = train_df_top[lab].to_numpy()
        y_test = test_df_top[lab].to_numpy()

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
final_svm_overall = final_svm_overall.sort_values(by=["Macro_F1", "Micro_F1"], ascending=False)

print("\nFinal tuned SVM comparison on test:")
print(final_svm_overall)

final_svm_overall.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_final_overall.csv"),
    index=False
)
final_svm_per_label.to_csv(
    os.path.join(BASE_PATH, "task1a_svm_tfidf_vs_pca_final_per_label.csv"),
    index=False
)


# ============================================================
# 6) SELECT FINAL MODEL
#    Based on Task 1a tuning, we use:
#    - SVM
#    - TFIDF
#    - no PCA
#    - C = 1.0
#    - class_weight = None
#    - per-label thresholds
# ============================================================
print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)

print("Selected final model for Task 1b:")
print("Model = SVM")
print("Representation = TFIDF")
print("Use PCA = False")
print(f"C = {FINAL_SVM_C}")
print(f"class_weight = {FINAL_SVM_CLASS_WEIGHT}")


# ============================================================
# 7) TASK 1B LABELS (LEAF TOPICS)
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
# 8) TASK 1B - FINAL SELECTED MODEL
#    Apply selected non-PCA SVM to the leaf topics
# ============================================================
print("\n" + "=" * 70)
print("TASK 1B - FINAL SELECTED SVM")
print("=" * 70)

train_sub_leaf, val_sub_leaf = train_test_split(
    train_df_leaf,
    test_size=0.2,
    random_state=42
)

X_train_sub_leaf, X_val_leaf = build_tfidf(train_sub_leaf["article"], val_sub_leaf["article"])
X_train_full_leaf, X_test_leaf = build_tfidf(train_df_leaf["article"], test_df_leaf["article"])

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