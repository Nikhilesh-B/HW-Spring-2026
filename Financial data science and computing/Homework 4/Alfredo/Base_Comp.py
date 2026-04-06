import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

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

# ============================================================
# 2) Metrics
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

    return pd.DataFrame([{
        "Macro_Accuracy": macro_accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Micro_Accuracy": micro_accuracy,
        "Micro_Precision": micro_precision,
        "Micro_Recall": micro_recall,
        "Micro_F1": micro_f1
    }])

# ============================================================
# 3) Build representations
# ============================================================
vectorizer = TfidfVectorizer(
    lowercase=False,
    min_df=2,
    token_pattern=r"(?u)\b\w+\b"
)

X_train_tfidf = vectorizer.fit_transform(news["article"])
X_test_tfidf = vectorizer.transform(news_test["article"])

n_components = 300
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_train_pca = svd.fit_transform(X_train_tfidf)
X_test_pca = svd.transform(X_test_tfidf)

representations = {
    "TFIDF": (X_train_tfidf, X_test_tfidf, 0),
    "TFIDF+PCA(300)": (X_train_pca, X_test_pca, 300)
}

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape :", X_test_tfidf.shape)
print("PCA train shape   :", X_train_pca.shape)
print("PCA test shape    :", X_test_pca.shape)

# ============================================================
# 4) Define models
# ============================================================
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
# 5) Run base comparison
# ============================================================
all_per_label = []
all_overall = []

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
            all_per_label.append(row)

        per_label_df = pd.DataFrame(per_label_rows)
        overall = overall_from_per_label(per_label_df)
        overall["Model"] = model_name
        overall["Representation"] = rep_name
        overall["Components"] = comps
        all_overall.append(overall)

comparison_per_label = pd.DataFrame(all_per_label)
comparison_overall = pd.concat(all_overall, ignore_index=True)

comparison_overall = comparison_overall.sort_values(
    by=["Representation", "Macro_F1", "Micro_F1"],
    ascending=[True, False, False]
)

print("\nOverall comparison:")
print(comparison_overall)

print("\nPer-label comparison:")
print(comparison_per_label)

# ============================================================
# 6) Save outputs
# ============================================================
comparison_overall.to_csv(
    os.path.join(BASE_PATH, "task1a_pca_vs_nopca_model_comparison_overall.csv"),
    index=False
)

comparison_per_label.to_csv(
    os.path.join(BASE_PATH, "task1a_pca_vs_nopca_model_comparison_per_label.csv"),
    index=False
)

print("\nSaved:")
print(" - task1a_pca_vs_nopca_model_comparison_overall.csv")
print(" - task1a_pca_vs_nopca_model_comparison_per_label.csv")