#!/usr/bin/env python3
"""
Train a full-tree :class:`MultilabelHierarchyRouter` on large article pools (e.g. ~400k train
rows), save it to disk, and/or run inference on a CSV of unseen documents.

Examples::

  # Train from explicit articles + id lists (see FULL_TREE_ROUTER.md)
  python full_tree_router_cli.py train \\
    --base /path/to/homework4 \\
    --articles /path/articles.csv \\
    --train-ids /path/train_ids.csv \\
    --out /path/artifacts/full_router

  # Build merged CSV + id lists from news.csv + news_test.csv, then train (full corpus)
  python full_tree_router_cli.py train-from-news-split \\
    --base /path/to/homework4 \\
    --build-dir /path/to/homework4/artifacts/full_corpus_build \\
    --out /path/to/artifacts/full_router

  # Only build merged articles + train/test id CSVs (no training)
  python full_tree_router_cli.py prepare-full-corpus \\
    --base /path/to/homework4 \\
    --output-dir /path/to/homework4/artifacts/full_corpus_build

  python full_tree_router_cli.py validate --base ... --articles ... --train-ids ...

  python full_tree_router_cli.py predict \\
    --router /path/artifacts/full_router \\
    --input /path/holdout.csv \\
    --output /path/predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _add_shared_train_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base", required=True, help="Directory with topics.csv (and news_topics.csv)")
    p.add_argument("--out", required=True, help="Output directory for saved router")
    p.add_argument("--news-topics", default=None, help="Override path to news_topics.csv")
    p.add_argument("--topics", default=None, help="Override path to topics.csv")
    p.add_argument("--article-column", default="article", help="Text column name in articles CSV")
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--max-iter", type=int, default=8000)
    p.add_argument("--default-c", type=float, default=1.0, help="LinearSVC C for depths not in map")
    p.add_argument(
        "--depth-c-json",
        default=None,
        help='JSON object mapping depth to C, e.g. \'{"0":0.1,"1":10,"2":10,"3":10}\'',
    )
    p.add_argument(
        "--no-restrict",
        action="store_true",
        help="Train on global pool (legacy); default is subtree-restricted positives",
    )


def _train_core(
    pool: Any,
    out: Path,
    args: argparse.Namespace,
    *,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    from hierarchical_classifier import (
        MultilabelHierarchyRouter,
        binary_edge_factory,
        linear_svc_estimator_by_depth,
        save_multilabel_router,
    )
    from hierarchical_summary_metrics import fit_all_binary_edges

    if args.depth_c_json:
        raw = json.loads(args.depth_c_json)
        depth_map = {int(k): float(v) for k, v in raw.items()}
    else:
        depth_map = {0: 0.1, 1: 10.0, 2: 10.0, 3: 10.0}
    est = linear_svc_estimator_by_depth(
        depth_map,
        default_c=float(args.default_c),
        dual=False,
        max_iter=int(args.max_iter),
    )
    factory = binary_edge_factory(
        tfidf_kw=dict(min_df=2, max_features=int(args.max_features)),
        estimator=est,
        text_vectorizer="tfidf",
    )

    router = MultilabelHierarchyRouter(pool.tree, factory)
    print("Fitting all binary edges...", flush=True)
    t1 = time.perf_counter()
    fit_all_binary_edges(
        router,
        pool,
        verbose=True,
        restrict_to_parent_subtree=not args.no_restrict,
    )
    print(f"Fit done in {time.perf_counter() - t1:.1f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    save_multilabel_router(router, out)
    meta: Dict[str, Any] = {
        "base_path": str(Path(args.base).resolve()),
        "depth_c_map": depth_map,
        "default_c": float(args.default_c),
        "max_features": int(args.max_features),
        "restrict_to_parent_subtree": not args.no_restrict,
        "train_n": len(pool.train_ids()),
        "test_n": len(pool.test_ids()),
    }
    if extra_meta:
        meta["corpus_prep"] = extra_meta
    with open(out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved router to {out}", flush=True)


def _cmd_validate(args: argparse.Namespace) -> int:
    from hierarchical_training_data import validate_router_training_inputs

    ok, lines = validate_router_training_inputs(
        base_path=str(Path(args.base).resolve()),
        articles_csv=str(Path(args.articles).resolve()),
        train_ids_csv=str(Path(args.train_ids).resolve()),
        test_ids_csv=str(Path(args.test_ids).resolve()) if args.test_ids else None,
        news_topics_csv=str(Path(args.news_topics).resolve()) if args.news_topics else None,
        topics_csv=str(Path(args.topics).resolve()) if args.topics else None,
        article_column=args.article_column,
    )
    for ln in lines:
        print(ln, flush=True)
    return 0 if ok else 1


def _cmd_train(args: argparse.Namespace) -> int:
    from hierarchical_training_data import make_multilabel_binary_pool_data_from_files

    print("Loading pool (this may take a while for large CSVs)...", flush=True)
    t0 = time.perf_counter()
    pool = make_multilabel_binary_pool_data_from_files(
        base_path=str(Path(args.base).resolve()),
        articles_csv=str(Path(args.articles).resolve()),
        train_ids_csv=str(Path(args.train_ids).resolve()),
        test_ids_csv=str(Path(args.test_ids).resolve()) if args.test_ids else None,
        news_topics_csv=str(Path(args.news_topics).resolve()) if args.news_topics else None,
        topics_csv=str(Path(args.topics).resolve()) if args.topics else None,
        article_column=args.article_column,
    )
    print(
        f"  train ids={len(pool.train_ids())}  test ids={len(pool.test_ids())}  "
        f"load_wall={time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    meta_paths = {
        "articles_csv": str(Path(args.articles).resolve()),
        "train_ids_csv": str(Path(args.train_ids).resolve()),
        "test_ids_csv": str(Path(args.test_ids).resolve()) if args.test_ids else None,
    }
    _train_core(pool, Path(args.out).resolve(), args, extra_meta=meta_paths)
    return 0


def _cmd_prepare_full_corpus(args: argparse.Namespace) -> int:
    from hierarchical_training_data import prepare_news_split_corpus_bundle

    info = prepare_news_split_corpus_bundle(
        base_path=str(Path(args.base).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        news_csv=str(Path(args.news).resolve()) if args.news else None,
        news_test_csv=str(Path(args.news_test).resolve()) if args.news_test else None,
        article_column=args.article_column,
        merged_filename=args.merged_filename,
        train_ids_filename=args.train_ids_filename,
        test_ids_filename=args.test_ids_filename,
        chunksize=int(args.chunksize),
        exclude_test_ids_already_in_train=not args.keep_overlapping_test_rows,
        pool_train_ids="news_only" if args.news_only_training else "union",
    )
    print(json.dumps(info, indent=2), flush=True)
    manifest_path = Path(args.output_dir) / "full_corpus_bundle_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote manifest to {manifest_path}", flush=True)
    return 0


def _cmd_train_from_news_split(args: argparse.Namespace) -> int:
    from hierarchical_training_data import (
        make_multilabel_binary_pool_data_from_files,
        prepare_news_split_corpus_bundle,
    )

    build_dir = Path(args.build_dir).resolve()
    merged = build_dir / args.merged_filename
    train_ids_p = build_dir / args.train_ids_filename
    test_ids_p = build_dir / args.test_ids_filename

    if args.reuse_build and merged.is_file() and train_ids_p.is_file():
        use_test = test_ids_p.is_file()
        info = {
            "articles_merged": str(merged),
            "train_ids": str(train_ids_p),
            "test_ids": str(test_ids_p) if use_test else None,
            "reused_build": True,
        }
        print("Reusing existing build in", build_dir, flush=True)
    else:
        print("Building merged corpus + id lists (streaming)...", flush=True)
        t0 = time.perf_counter()
        info = prepare_news_split_corpus_bundle(
            base_path=str(Path(args.base).resolve()),
            output_dir=str(build_dir),
            news_csv=str(Path(args.news).resolve()) if args.news else None,
            news_test_csv=str(Path(args.news_test).resolve()) if args.news_test else None,
            article_column=args.article_column,
            merged_filename=args.merged_filename,
            train_ids_filename=args.train_ids_filename,
            test_ids_filename=args.test_ids_filename,
            chunksize=int(args.chunksize),
            exclude_test_ids_already_in_train=not args.keep_overlapping_test_rows,
            pool_train_ids="news_only" if args.news_only_training else "union",
        )
        print(f"  build_wall={time.perf_counter() - t0:.1f}s", flush=True)
        manifest_path = build_dir / "full_corpus_bundle_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(info, f, indent=2)

    print("Loading pool into memory...", flush=True)
    t0 = time.perf_counter()
    pool = make_multilabel_binary_pool_data_from_files(
        base_path=str(Path(args.base).resolve()),
        articles_csv=info["articles_merged"],
        train_ids_csv=info["train_ids"],
        test_ids_csv=info.get("test_ids"),
        news_topics_csv=str(Path(args.news_topics).resolve()) if args.news_topics else None,
        topics_csv=str(Path(args.topics).resolve()) if args.topics else None,
        article_column=args.article_column,
    )
    print(
        f"  train ids={len(pool.train_ids())}  test ids={len(pool.test_ids())}  "
        f"load_wall={time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    _train_core(pool, Path(args.out).resolve(), args, extra_meta=info)
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import pandas as pd

    from hierarchical_classifier import apply_router_to_dataframe, load_multilabel_router

    router = load_multilabel_router(Path(args.router).resolve())
    df = pd.read_csv(Path(args.input).resolve())
    if args.text_column not in df.columns:
        print(f"Column {args.text_column!r} not in CSV: {df.columns.tolist()}", file=sys.stderr)
        return 1
    out_df = apply_router_to_dataframe(
        router,
        df,
        text_column=args.text_column,
        leaves_sep=args.leaves_sep,
    )
    outp = Path(args.output).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outp, index=False)
    print(f"Wrote {len(out_df)} rows to {outp}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Full-tree hierarchical router: validate / prepare / train / predict"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser(
        "validate",
        help="Check topics, news_topics, articles, and id coverage before train",
    )
    pv.add_argument("--base", required=True, help="Directory with topics.csv and news_topics.csv")
    pv.add_argument("--articles", required=True, help="CSV with id + article text")
    pv.add_argument("--train-ids", required=True, help="CSV of training ids")
    pv.add_argument("--test-ids", default=None, help="Optional CSV of test ids")
    pv.add_argument("--news-topics", default=None, help="Override path to news_topics.csv")
    pv.add_argument("--topics", default=None, help="Override path to topics.csv")
    pv.add_argument("--article-column", default="article")
    pv.set_defaults(func=_cmd_validate)

    pprep = sub.add_parser(
        "prepare-full-corpus",
        help="Merge news.csv (+ optional news_test.csv) into one articles CSV + id lists for training",
    )
    pprep.add_argument("--base", required=True, help="Directory containing news.csv (default names)")
    pprep.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write articles_merged.csv, id lists, and manifest JSON",
    )
    pprep.add_argument("--news", default=None, help="Override path to training articles CSV")
    pprep.add_argument("--news-test", default=None, dest="news_test", help="Override path to test articles CSV")
    pprep.add_argument("--article-column", default="article")
    pprep.add_argument("--merged-filename", default="articles_merged.csv")
    pprep.add_argument("--train-ids-filename", default="full_corpus_train_ids.csv")
    pprep.add_argument("--test-ids-filename", default="full_corpus_test_ids.csv")
    pprep.add_argument("--chunksize", type=int, default=200_000)
    pprep.add_argument(
        "--keep-overlapping-test-rows",
        action="store_true",
        help="If an id appears in both news and news_test, keep test row (default: drop from test)",
    )
    pprep.add_argument(
        "--news-only-training",
        action="store_true",
        help="Train ids = news.csv only; write a separate test id list from news_test (legacy). "
        "Default: train ids = union of both files so the router fits on the full merged corpus.",
    )
    pprep.set_defaults(func=_cmd_prepare_full_corpus)

    pt = sub.add_parser("train", help="Fit all edges using explicit articles + train id CSVs")
    _add_shared_train_flags(pt)
    pt.add_argument("--articles", required=True, help="CSV with id + article text")
    pt.add_argument("--train-ids", required=True, help="CSV of training ids")
    pt.add_argument("--test-ids", default=None, help="Optional CSV of test ids")
    pt.set_defaults(func=_cmd_train)

    pfs = sub.add_parser(
        "train-from-news-split",
        help="Build (or reuse) merged corpus from news.csv + news_test.csv, then train full tree",
    )
    _add_shared_train_flags(pfs)
    pfs.add_argument(
        "--build-dir",
        required=True,
        help="Where to write/read articles_merged.csv and id lists",
    )
    pfs.add_argument(
        "--reuse-build",
        action="store_true",
        help="Skip streaming merge if merged file already exists in --build-dir",
    )
    pfs.add_argument("--news", default=None, help="Override path to training articles CSV")
    pfs.add_argument("--news-test", default=None, dest="news_test", help="Override path to test articles CSV")
    pfs.add_argument("--merged-filename", default="articles_merged.csv")
    pfs.add_argument("--train-ids-filename", default="full_corpus_train_ids.csv")
    pfs.add_argument("--test-ids-filename", default="full_corpus_test_ids.csv")
    pfs.add_argument("--chunksize", type=int, default=200_000)
    pfs.add_argument(
        "--keep-overlapping-test-rows",
        action="store_true",
        help="Keep test rows whose id also appears in news (default: exclude)",
    )
    pfs.add_argument(
        "--news-only-training",
        action="store_true",
        help="Same as prepare-full-corpus: fit only on news.csv ids (legacy split for eval on news_test). "
        "Default: fit on all ids from news ∪ news_test.",
    )
    pfs.set_defaults(func=_cmd_train_from_news_split)

    ppr = sub.add_parser("predict", help="Load saved router and label a CSV")
    ppr.add_argument("--router", required=True, help="Directory with tree_meta.pkl + edge_models.pkl")
    ppr.add_argument("--input", required=True, help="Input CSV")
    ppr.add_argument("--output", required=True, help="Output CSV with predicted_leaves")
    ppr.add_argument("--text-column", default="article")
    ppr.add_argument("--leaves-sep", default=";", dest="leaves_sep")
    ppr.set_defaults(func=_cmd_predict)

    ns = p.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
