# Full-tree hierarchical router (Project D)

Linear **multi-label binary relevance** on the RCV1-style taxonomy: one **TF-IDF + `LinearSVC`** per branching edge (`parent → child`), with optional **depth-specific `C`** (see `hierarchical_projectD_depthC_consolidated.ipynb`).

## Files you need

| File | Columns / role |
|------|------------------|
| `topics.csv` | Taxonomy (in `--base` directory) |
| `news_topics.csv` | `id`, `cat` — multi-label gold for every id in train (and optional in-pool test) |
| `articles.csv` | `id`, `article` (or set `--article-column`) |
| `train_ids.csv` | `id` (or first column) — training document ids |
| `test_ids.csv` | Optional — in-pool test ids; omit for train-only |

`make_multilabel_binary_pool_data_from_files` keeps only rows whose `id` is in **train ∪ test** (smaller RAM if `articles.csv` is a superset). Duplicate `id` rows in `articles.csv` are deduplicated (**first row kept**).

## Full corpus (`news.csv` + `news_test.csv`)

If your texts live in the usual **split files** (not one combined `articles.csv` + id lists), build the training layout first, then train.

**Layout:** `news.csv` = training split, `news_test.csv` = test split (both need `id` + `article`). Defaults resolve paths under `--base`. If `news_test.csv` is missing, you get a train-only pool.

**Default training ids:** `full_corpus_train_ids.csv` lists the **union** of ids from `news.csv` and `news_test.csv` (after overlap handling), so **edge fitting uses every row in the merged file**. A separate `full_corpus_test_ids.csv` is **not** written in that mode. Use **`--news-only-training`** on `prepare-full-corpus` / `train-from-news-split` for the legacy layout (train ids = `news.csv` only, test ids = `news_test.csv` for a held-out eval split).

**1) Build merged articles + id lists only** (streaming; safe for large files):

```bash
cd "/path/to/Homework 4"
python3 full_tree_router_cli.py prepare-full-corpus \
  --base . \
  --output-dir artifacts/full_corpus_build
```

Writes `articles_merged.csv`, `full_corpus_train_ids.csv`, `full_corpus_bundle_manifest.json`, and prints a JSON summary (manifest includes `pool_train_ids`, `n_news_ids`, etc.). Overrides: `--news`, `--news-test`, `--article-column`, `--chunksize`.

By default, **ids that appear in both** `news.csv` and `news_test.csv` are **dropped from the test** side in the merged file (use `--keep-overlapping-test-rows` to keep them).

**2) Train in one step** (merge if needed, then fit + save):

```bash
python3 full_tree_router_cli.py train-from-news-split \
  --base . \
  --build-dir artifacts/full_corpus_build \
  --out artifacts/full_tree_router_full_corpus
```

Add `--reuse-build` to skip re-merging when `articles_merged.csv` already exists in `--build-dir`.

**Validate after `prepare-full-corpus` (default union mode):**

```bash
python3 full_tree_router_cli.py validate \
  --base . \
  --articles artifacts/full_corpus_build/articles_merged.csv \
  --train-ids artifacts/full_corpus_build/full_corpus_train_ids.csv
```

If you used **`--news-only-training`**, pass **`--test-ids artifacts/full_corpus_build/full_corpus_test_ids.csv`** as well.

**Python:** `hierarchical_training_data.prepare_news_split_corpus_bundle(..., pool_train_ids="union" | "news_only")`.

## Validate inputs (before training)

```bash
cd "/path/to/Homework 4"
python3 full_tree_router_cli.py validate \
  --base . \
  --articles articles_leaf_min60_cap200.csv \
  --train-ids articles_leaf_min60_cap200_train_ids.csv \
  --test-ids articles_leaf_min60_cap200_test_ids.csv
```

Checks: paths exist, `topics` loads, articles header, every pool id appears in `articles.csv`, and `news_topics` coverage (warns if any pool id has no labels).

## Train and save

```bash
python3 full_tree_router_cli.py train \
  --base . \
  --articles articles_leaf_min60_cap200.csv \
  --train-ids articles_leaf_min60_cap200_train_ids.csv \
  --test-ids articles_leaf_min60_cap200_test_ids.csv \
  --out artifacts/full_tree_router
```

Optional: `--no-restrict` (global pool), `--depth-c-json '{"0":0.1,"1":10,"2":10,"3":10}'`, `--news-topics`, `--topics`, `--max-features`.

**Outputs:** `tree_meta.pkl`, `edge_models.pkl`, `train_meta.json`.

## Predict on unseen CSV

Same **Python + scikit-learn** version as training. Input CSV must include the text column (default `article`).

```bash
python3 full_tree_router_cli.py predict \
  --router artifacts/full_tree_router \
  --input holdout.csv \
  --output predictions.csv
```

Output adds `predicted_leaves` (leaf codes, `;`-separated) and `n_predicted_leaves`.

## Python API

- Pool: `hierarchical_training_data.make_multilabel_binary_pool_data_from_files`
- Fit: `hierarchical_summary_metrics.fit_all_binary_edges`
- Save / load: `hierarchical_classifier.save_multilabel_router`, `load_multilabel_router`
- Batch scores: `apply_router_to_dataframe`, `predict_reached_leaves_batch`

## Scale notes

Large train sets (e.g. hundreds of thousands of rows) mean **high RAM** and **long wall time** (many edges × vectorization + SVM). Pickled models are **not** portable across sklearn/Python versions.
