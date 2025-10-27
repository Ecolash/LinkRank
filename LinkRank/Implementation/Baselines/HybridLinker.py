
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import logging
import time


dataset_paths = [
    "Add your file path here",

]
output_dir = "Add path for the directory to save results"


Path(output_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("baseline_ic_links")

REQUIRED_COLS = [
    "Issue ID","Issue Date","Title","Description","Labels","Comments",
    "Commit ID","Commit Date","Message","Diff Summary","File Changes","Full Diff","Output"
]

def check_columns(df, path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {missing}")

def build_text(df: pd.DataFrame):
    """
    Simple baseline: bag-of-words over all textual fields (issue + commit sides).
    """
    parts = [
        df["Title"].fillna(""),
        df["Description"].fillna(""),
        df["Labels"].fillna(""),
        df["Comments"].fillna(""),
        df["Message"].fillna(""),
        df["Diff Summary"].fillna(""),
        df["File Changes"].fillna(""),
        df["Full Diff"].fillna(""),
    ]
    return (" ".join(cols) for cols in zip(*parts))

def split_by_issue_id(df: pd.DataFrame, seed=42, test_ratio=0.2):
    rng = np.random.default_rng(seed)
    issue_ids = df["Issue ID"].drop_duplicates().tolist()
    rng.shuffle(issue_ids)
    n = len(issue_ids)
    n_train = int((1.0 - test_ratio) * n)
    train_ids = set(issue_ids[:n_train])
    test_ids  = set(issue_ids[n_train:])
    return df[df["Issue ID"].isin(train_ids)].copy(), df[df["Issue ID"].isin(test_ids)].copy()

def per_issue_metrics(test_df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for iid, g in test_df.groupby("Issue ID"):
        true_set = set(g.loc[g["Output"] == 1, "Commit ID"])
        pred_set = set(g.loc[g["Predicted"] == 1, "Commit ID"])

        inter = len(true_set & pred_set)
        upred = len(pred_set)
        utrue = len(true_set)
        union = len(true_set | pred_set)

        precision = inter / upred if upred > 0 else 0.0
        recall    = inter / utrue if utrue > 0 else 0.0
        f1        = (2 * inter) / (upred + utrue) if (upred + utrue) > 0 else 0.0
        jaccard   = inter / union if union > 0 else 0.0

        rows.append({
            "Issue ID": iid,
            "True_Pos_Commits": utrue,
            "Pred_Pos_Commits": upred,
            "Inter": inter,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })
    return pd.DataFrame(rows)

def train_and_eval_one(file_path: str, out_dir: str, vectorizer_max_features=3000, seed=42):
    t0 = time.perf_counter()
    ds_name = os.path.splitext(os.path.basename(file_path))[0]
    log.info(f"▶ Processing {ds_name} ...")

    df = pd.read_csv(file_path)
    check_columns(df, file_path)

    df["Output"] = df["Output"].astype(int)

    train_df, test_df = split_by_issue_id(df, seed=seed, test_ratio=0.2)
    log.info(f"[{ds_name}] Train issues={train_df['Issue ID'].nunique()} rows={len(train_df):,} | "
             f"Test issues={test_df['Issue ID'].nunique()} rows={len(test_df):,}")

    vectorizer = TfidfVectorizer(max_features=vectorizer_max_features)
    X_train = vectorizer.fit_transform(list(build_text(train_df)))
    y_train = train_df["Output"].values
    X_test  = vectorizer.transform(list(build_text(test_df)))

    tree_method = "hist"
    try:
        if os.environ.get("XGB_USE_GPU", "0") == "1":
            tree_method = "gpu_hist"
    except Exception:
        pass

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": tree_method,
        "max_depth": 4,
        "learning_rate": 0.05,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "lambda": 1.0,
        "alpha": 0.1,
        "random_state": seed,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=300)

    test_df = test_df.copy()
    y_prob = model.predict(dtest)
    test_df["Predicted_Prob"] = y_prob
    test_df["Predicted"] = (y_prob >= 0.5).astype(int)

    per_issue = per_issue_metrics(test_df)
    macro = {
        "Dataset": ds_name,
        "Issues": len(per_issue),
        "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
        "Mean_Recall": per_issue["Recall"].mean() if len(per_issue) else 0.0,
        "Mean_F1": per_issue["F1"].mean() if len(per_issue) else 0.0,
    }

    per_issue_path = os.path.join(out_dir, f"{ds_name}_per_issue_metrics.csv")
    preds_path     = os.path.join(out_dir, f"{ds_name}_test_predictions.csv")
    per_issue.to_csv(per_issue_path, index=False)
    test_df.to_csv(preds_path, index=False)

    log.info(f"[{ds_name}] Saved per-issue metrics → {per_issue_path}")
    log.info(f"[{ds_name}] Saved test predictions  → {preds_path}")
    log.info(f"[{ds_name}] Macro: "
             f"P={macro['Mean_Precision']:.4f} R={macro['Mean_Recall']:.4f} "
             f"F1={macro['Mean_F1']:.4f} "
             f"({time.perf_counter()-t0:.2f}s)")

    return macro

all_summaries = []
for path in dataset_paths:
    try:
        summary = train_and_eval_one(path, output_dir)
        all_summaries.append(summary)
    except Exception as e:
        log.error(f"Failed on {path}: {e}")

if all_summaries:
    summary_df = pd.DataFrame(all_summaries)

    for col in ["Mean_Precision", "Mean_Recall", "Mean_F1"]:
        summary_df[col] = (summary_df[col] * 100).round(2)

    summary_csv = os.path.join(output_dir, "Hybrid_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    log.info(f"✔ All-dataset summary saved → {summary_csv}")

    with pd.option_context("display.max_columns", None):
        print("\n=== Macro Summary Across Datasets (values are in %) ===")
        print(summary_df.to_string(index=False))
else:
    log.warning("No summaries produced. Please check dataset paths and schema.")

