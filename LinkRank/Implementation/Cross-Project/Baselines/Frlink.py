import os
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


USE_GPU = False
try:
    import cupy as cp
    from cupyx.scipy import sparse as cpx_sparse
    _ = cp.cuda.runtime.getDeviceCount()  
    USE_GPU = True
except Exception:
    USE_GPU = False


train_paths = [
    "Add path of the training files here",
]
test_paths = [
    "Add path of the testing files here",
]
output_dir = "Add your output directory here"

SIM_THRESHOLD = 0.50       
SEED = 42
TFIDF_MAX_FEATURES = 1000  


Path(output_dir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("frlink_crossproj")


REQUIRED_COLS = [
    "Issue ID","Issue Date","Title","Description","Labels","Comments",
    "Commit ID","Commit Date","Message","Diff Summary","File Changes","Full Diff","Output"
]

def check_columns(df, path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {missing}")


def build_text(df: pd.DataFrame):
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


def max_sim_cpu(test_matrix, train_matrix, batch_size=2000):
    n = test_matrix.shape[0]
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = test_matrix[s:e]            
        dp = batch @ train_matrix.T         
        mx = dp.max(axis=1)                 
        out[s:e] = np.asarray(mx.toarray()).ravel().astype(np.float32)
    return out

def max_sim_gpu(test_matrix, train_matrix, batch_size=5000):
    test_gpu  = cpx_sparse.csr_matrix(test_matrix)
    train_gpu = cpx_sparse.csr_matrix(train_matrix)
    n = test_gpu.shape[0]
    out = cp.empty(n, dtype=cp.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = test_gpu[s:e]
        dp = batch @ train_gpu.T            
        mx = dp.max(axis=1)                 
        out[s:e] = cp.asarray(mx.A.ravel()) 
    return cp.asnumpy(out)


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

def to_percent_2(x):
    return round(float(x) * 100.0, 2)

def fit_vectorizer_on_train(train_paths):
    texts = []
    n_rows = 0
    for p in train_paths:
        df = pd.read_csv(p)
        check_columns(df, p)
        n_rows += len(df)
        texts.extend(build_text(df))
        log.info(f"[TRAIN LOAD] {Path(p).stem}: {len(df):,} rows")
    log.info(f"[TRAIN] Total rows across train_paths: {n_rows:,}")

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, norm='l2')
    X_train = vectorizer.fit_transform(texts)
    log.info(f"[TRAIN] TF-IDF fitted. Matrix shape: {X_train.shape}")
    return vectorizer, X_train


def run_one_test(file_path: str, vectorizer: TfidfVectorizer, X_train) -> dict:
    t0 = time.perf_counter()
    ds_name = Path(file_path).stem
    log.info(f"Testing {ds_name} (GPU={USE_GPU})")

    df = pd.read_csv(file_path)
    check_columns(df, file_path)
    df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).astype(int)

    X_test = vectorizer.transform(list(build_text(df)))

    sims = max_sim_gpu(X_test, X_train) if USE_GPU else max_sim_cpu(X_test, X_train)
    df = df.copy()
    df["Max_Sim"]   = sims
    df["Predicted"] = (df["Max_Sim"] >= SIM_THRESHOLD).astype(int)

    per_issue = per_issue_metrics(df)
    macro = {
        "Dataset": ds_name,
        "Issues": len(per_issue),
        "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
        "Mean_Recall":    per_issue["Recall"].mean() if len(per_issue) else 0.0,
        "Mean_F1":        per_issue["F1"].mean() if len(per_issue) else 0.0,
    }

    preds_path = os.path.join(output_dir, f"{ds_name}_test_predictions.csv")
    df.to_csv(preds_path, index=False)

    per_issue_pct = per_issue.copy()
    for c in ["Precision","Recall","F1"]:
        per_issue_pct[c] = per_issue_pct[c].map(to_percent_2)
    per_issue_path = os.path.join(output_dir, f"{ds_name}_per_issue_metrics.csv")
    per_issue_pct.to_csv(per_issue_path, index=False)

    macro_pct = {k: (to_percent_2(v) if k.startswith("Mean_") else v) for k, v in macro.items()}
    log.info(f"[{ds_name}] Macro (%): "
             f"P={macro_pct['Mean_Precision']:.2f} R={macro_pct['Mean_Recall']:.2f} "
             f"F1={macro_pct['Mean_F1']:.2f} ({time.perf_counter()-t0:.2f}s)")

    return macro  

if __name__ == "__main__":
    if USE_GPU:
        try:
            ng = cp.cuda.runtime.getDeviceCount()
            log.info(f" GPU available: {ng} device(s)")
        except Exception:
            log.info(" GPU check failed; continuing with CPU.")
    else:
        log.info(" Using CPU (CuPy/CUDA not available)")

    try:
        vectorizer, X_train = fit_vectorizer_on_train(train_paths)
    except Exception as e:
        log.error(f" Failed during training vectorizer: {e}")
        raise

    summaries = []
    for p in test_paths:
        try:
            summaries.append(run_one_test(p, vectorizer, X_train))
        except Exception as e:
            log.error(f" Failed on {p}: {e}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        for col in ["Mean_Precision","Mean_Recall","Mean_F1"]:
            summary_df[col] = (summary_df[col] * 100.0).round(2)
        out_sum = os.path.join(output_dir, "FRLINK_cross_project_summary.csv")
        summary_df.to_csv(out_sum, index=False)

        print("\n=== Cross-Project Macro Summary (values in %) ===")
        print(summary_df.to_string(index=False))
        log.info(f" Summary saved → {out_sum}")
    else:
        log.warning("No summaries produced.")
