import os
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


train_paths = [
    "Add path of the training files here",
]
test_paths = [
    "Add path of the testing files here",
]
output_dir = "Add your output directory here"

VECTORIZER_MAX_FEATURES = 3000
SEED = 42
PRED_THRESHOLD = 0.50


Path(output_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("xgb_crossproj_baseline")

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
    Bag-of-words over issue + commit text fields.
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

def per_issue_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iid, g in test_df.groupby("Issue ID"):
        true_set = set(g.loc[g["Output"] == 1, "Commit ID"])
        pred_set = set(g.loc[g["Predicted"] == 1, "Commit ID"])

        inter = len(true_set & pred_set)
        upred = len(pred_set)
        utrue = len(true_set)

        precision = inter / upred if upred > 0 else 0.0
        recall    = inter / utrue if utrue > 0 else 0.0
        f1        = (2 * inter) / (upred + utrue) if (upred + utrue) > 0 else 0.0

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

def fit_vectorizer_and_train(train_paths, max_features=3000, seed=42):
    t0 = time.perf_counter()
    texts = []
    labels = []
    n_rows = 0

    for p in train_paths:
        df = pd.read_csv(p)
        check_columns(df, p)
        df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).astype(int)
        batch_texts = list(build_text(df))
        texts.extend(batch_texts)
        labels.extend(df["Output"].tolist())
        n_rows += len(df)
        log.info(f"[TRAIN LOAD] {Path(p).stem}: {len(df):,} rows")

    log.info(f"[TRAIN] Total rows across train_paths: {n_rows:,}")

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(texts)
    y_train = np.asarray(labels, dtype=np.int32)
    log.info(f"[TRAIN] TF-IDF fitted. Matrix shape: {X_train.shape}")


    tree_method = "gpu_hist" if os.environ.get("XGB_USE_GPU", "0") == "1" else "hist"
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
    model = xgb.train(params, dtrain, num_boost_round=300)
    log.info(f"[TRAIN] Model trained in {time.perf_counter()-t0:.2f}s (tree_method={tree_method})")

    return vectorizer, model

def evaluate_one_test(test_csv, vectorizer, model, out_dir, thresh=PRED_THRESHOLD):
    t0 = time.perf_counter()
    ds_name = Path(test_csv).stem
    log.info(f"▶ Testing {ds_name}")

    df = pd.read_csv(test_csv)
    check_columns(df, test_csv)
    df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).astype(int)

    X_test = vectorizer.transform(list(build_text(df)))
    dtest  = xgb.DMatrix(X_test)

    y_prob = model.predict(dtest)
    df = df.copy()
    df["Predicted_Prob"] = y_prob
    df["Predicted"] = (y_prob >= thresh).astype(int)

    per_issue = per_issue_metrics(df)
    macro = {
        "Dataset": ds_name,
        "Issues": int(per_issue["Issue ID"].nunique()) if len(per_issue) else 0,
        "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
        "Mean_Recall":    per_issue["Recall"].mean() if len(per_issue) else 0.0,
        "Mean_F1":        per_issue["F1"].mean() if len(per_issue) else 0.0,
    }

    per_issue_path = os.path.join(out_dir, f"{ds_name}_per_issue_metrics.csv")
    preds_path     = os.path.join(out_dir, f"{ds_name}_test_predictions.csv")
    per_issue.to_csv(per_issue_path, index=False)
    df.to_csv(preds_path, index=False)

    log.info(f"[{ds_name}] P={macro['Mean_Precision']:.4f} R={macro['Mean_Recall']:.4f} "
             f"F1={macro['Mean_F1']:.4f} ({time.perf_counter()-t0:.2f}s)")

    return macro


if __name__ == "__main__":

    try:
        vectorizer, model = fit_vectorizer_and_train(train_paths, max_features=VECTORIZER_MAX_FEATURES, seed=SEED)
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    summaries = []
    for p in test_paths:
        try:
            summaries.append(evaluate_one_test(p, vectorizer, model, output_dir))
        except Exception as e:
            log.error(f" Failed on {p}: {e}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        for col in ["Mean_Precision","Mean_Recall","Mean_F1"]:
            summary_df[col] = (summary_df[col] * 100.0).round(2)
        summary_csv = os.path.join(output_dir, "XGB_cross_project_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        print("\n=== Cross-Project Macro Summary (values in %) ===")
        print(summary_df.to_string(index=False))
        log.info(f"✔ Summary saved → {summary_csv}")
    else:
        log.warning("No summaries produced. Check paths and schema.")

