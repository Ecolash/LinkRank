import os, re, time, logging, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import psutil
    def mem_gb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
except Exception:
    psutil = None
    def mem_gb():
        return float('nan')

try:
    import torch
    TORCH_HAS_CUDA = torch.cuda.is_available()
except Exception:
    TORCH_HAS_CUDA = False


USE_CUML_SVD = False
try:
    from cuml.decomposition import TruncatedSVD as cuML_TruncatedSVD
    USE_CUML_SVD = True
except Exception:
    from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD
    USE_CUML_SVD = False


train_paths = [
    "Add path of the training files here",
]
test_paths = [
    "Add path of the testing files here",
]

OUT_ROOT = Path("Add your output directory here")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = OUT_ROOT / "summaries"; SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = SUMMARY_DIR / "summary_results_cross_nocodebert.csv"
LOG_PATH    = OUT_ROOT / "run.log"
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
SVD_DIM = 256
USE_TIME_FEATURE = True
TIME_TAU_DAYS = 7.0
TOPK_ISSUES_PER_COMMIT = 5           
TUNE_OBJECTIVE = "F1"               
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
LOG_EVERY_N_ITERS = 10
FEATURE_COLS = ["feat_text", "feat_file", "feat_time"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger("linkrank_nocodebert_cross")


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def cos_sim(a, b):
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def parse_files(x):
    if pd.isna(x): return set()
    return set([p.strip() for p in str(x).split(",") if p.strip()])

def time_prox(issue_date, commit_date, tau_days=14.0):
    if pd.isna(issue_date) or pd.isna(commit_date): return 0.0
    days = abs((commit_date - issue_date).days)
    return float(np.exp(-days / tau_days))

def issue_text(d):
    g = d.groupby("Issue ID").agg(
        title=("Title","first"),
        desc =("Description","first"),
        comm =("Comments","first"),
    )
    txt = (g["title"].fillna("") + " " + g["desc"].fillna("") + " " + g["comm"].fillna(""))
    out = txt.reset_index(); out.columns = ["Issue ID", "text"]
    return out

def commit_text(d):
    g = d.groupby("Commit ID").agg(
        msg =("Message","first"),
        dif =("Diff Summary","first"),
        files=("File Changes","first"),
        full =("Full Diff","first"),
    )
    txt = (g["msg"].fillna("") + " " + g["dif"].fillna("") + " " + g["files"].fillna("") + " " + g["full"].fillna(""))
    out = txt.reset_index(); out.columns = ["Commit ID", "text"]
    return out

def issue_file_union_all(d):
    if "File Changes" not in d.columns:
        return {}
    t = d.groupby("Issue ID")["File Changes"].apply(
        lambda s: set().union(*[parse_files(x) for x in s])
    )
    return t.to_dict()

def macro_percent(df):
    """Return (P,R,F1) in percent."""
    p = round(100.0 * df["Precision"].mean(), 2) if len(df) else 0.0
    r = round(100.0 * df["Recall"].mean(),    2) if len(df) else 0.0
    f = round(100.0 * df["F1"].mean(),        2) if len(df) else 0.0
    return p, r, f

def stage(name):
    class _Stage:
        def __enter__(self):
            self.name = name
            self.t0 = time.perf_counter()
            m = f"{mem_gb():.2f} GB" if psutil else "n/a"
            log.info(f"▶ START: {self.name} | mem={m}")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            m = f"{mem_gb():.2f} GB" if psutil else "n/a"
            if exc_type is None:
                log.info(f"✔ END:   {self.name} | {dt:.2f}s | mem={m}")
            else:
                log.error(f"✖ FAIL:  {self.name} | {dt:.2f}s | mem={m} | {exc_type.__name__}: {exc}")
            return False
    return _Stage()

def ensure_nonempty_valid(Xtr, ytr, gtr, Xdv, ydv, gdv):
    """If dev is empty, carve out a tiny slice from train as dev (keeps flow intact)."""
    if len(Xdv) > 0 and len(gdv) > 0:
        return Xtr, ytr, gtr, Xdv, ydv, gdv
    n = len(ytr)
    if n < 2:
        return Xtr, ytr, gtr, Xtr, ytr, gtr
    cut = max(1, n // 10)
    return Xtr[cut:], ytr[cut:], gtr, Xtr[:cut], ytr[:cut], gtr

def prep_rank(df_feat, group_col):
    df_s = df_feat.sort_values([group_col]).reset_index(drop=True)
    X = df_s[FEATURE_COLS].values
    y = df_s["Output"].astype(int).values
    groups = df_s.groupby(group_col).size().tolist()
    return df_s, X, y, groups

def train_lambdamart(tag, Xtr, ytr, gtr, Xdv, ydv, gdv):
    Xtr, ytr, gtr, Xdv, ydv, gdv = ensure_nonempty_valid(Xtr, ytr, gtr, Xdv, ydv, gdv)
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEATURE_COLS)
    dvalid = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain, feature_name=FEATURE_COLS)
    params = dict(
        objective="lambdarank",
        metric=["ndcg"],
        ndcg_eval_at=[1,3,5],
        learning_rate=0.05,
        num_leaves=47,
        min_data_in_leaf=25,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=2.0,
        verbose=-1,
        num_threads=os.cpu_count() or 1,
    )
    if TORCH_HAS_CUDA:
        params.update(dict(device_type="gpu", gpu_platform_id=0, gpu_device_id=0))
        log.info(f"[{tag}] LightGBM: GPU enabled.")
    else:
        log.info(f"[{tag}] LightGBM: CPU.")
    model = lgb.train(
        params, dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=2000,
        callbacks=[
            lgb.log_evaluation(LOG_EVERY_N_ITERS),
            lgb.early_stopping(stopping_rounds=150, verbose=True)
        ]
    )
    log.info(f"[{tag}] Best iteration: {model.best_iteration}")
    return model

def build_features(d, issue_idx_tr, commit_idx_tr, issue_idx_te, commit_idx_te,
                   E_issue_tr, E_commit_tr, X_te_issue, X_te_commit,
                   issue_files_union, split, ds_tag):
    rows = []
    it = tqdm(d.iterrows(), total=len(d), desc=f"build_features[{ds_tag}][{split}]")
    for _, row in it:
        iid = row["Issue ID"]; cid = row["Commit ID"]
        if split == "train":
            if iid in issue_idx_tr and cid in commit_idx_tr:
                v_i = E_issue_tr[issue_idx_tr[iid]]
                v_c = E_commit_tr[commit_idx_tr[cid]]
                f_text = cos_sim(v_i, v_c)
            else:
                f_text = 0.0
        else:
            if iid in issue_idx_te and cid in commit_idx_te:
                v_i = X_te_issue[issue_idx_te[iid]]
                v_c = X_te_commit[commit_idx_te[cid]]
                f_text = cos_sim(v_i, v_c)
            else:
                f_text = 0.0

        files_i = issue_files_union.get(iid, set())
        files_c = parse_files(row.get("File Changes"))
        inter = len(files_i & files_c); union = len(files_i | files_c)
        f_file = (inter/union) if union>0 else 0.0
        f_time = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS) if USE_TIME_FEATURE else 0.0
        rows.append([iid, cid, f_text, f_file, f_time, int(row["Output"])])
    return pd.DataFrame(rows, columns=["Issue ID","Commit ID",*FEATURE_COLS,"Output"])

def score_pool(model, pool_df, use_mm=True):
    X = pool_df[FEATURE_COLS].values
    s = model.predict(X, num_iteration=model.best_iteration)
    out = pool_df.copy()
    out["score_raw"] = s
    if use_mm:
        if len(out) > 1:
            mn, mx = float(out["score_raw"].min()), float(out["score_raw"].max())
            out["score_mm"] = 1.0 if mx==mn else (out["score_raw"] - mn) / (mx - mn)
        else:
            out["score_mm"] = 1.0
    return out.sort_values("score_raw", ascending=False).reset_index(drop=True)

def metrics_for_issue(pred_set, true_set):
    inter = len(pred_set & true_set)
    p = inter / max(len(pred_set), 1)
    r = inter / max(len(true_set), 1)
    f1 = (2*inter) / max(len(pred_set)+len(true_set), 1)
    allc = int(pred_set == true_set)
    half = int(inter >= int(np.ceil(len(true_set)/2)))
    return dict(Precision=p, Recall=r, F1=f1, AllCorrect=allc, HalfCorrect=half)

def restrict_pool_by_A2_shortlist(df_issue_pool, shortlist_map, fallback_to_full=True):
    if df_issue_pool.empty:
        return df_issue_pool
    iid = df_issue_pool["Issue ID"].iloc[0]
    mask = df_issue_pool["Commit ID"].map(lambda cid: iid in shortlist_map.get(cid, set()))
    subset = df_issue_pool[mask]
    if subset.empty and fallback_to_full:
        return df_issue_pool
    return subset

def topk_issues_per_commit(df_split, model_A2, K=TOPK_ISSUES_PER_COMMIT):
    res = {}
    by_commit = {cid: df_split[df_split["Commit ID"]==cid].reset_index(drop=True)
                 for cid in df_split["Commit ID"].drop_duplicates().tolist()}
    for cid, pool in by_commit.items():
        ranked = score_pool(model_A2, pool, use_mm=True)
        res[cid] = set(ranked["Issue ID"].iloc[:K].tolist())
    return res

REQUIRED_COLS = ["Repository","Issue ID","Issue Date","Title","Description","Labels","Comments",
                 "Commit ID","Commit Date","Message","Diff Summary","File Changes","Full Diff","Output"]

def read_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).stem}: Missing columns: {missing}")
    for c in ["Issue Date","Commit Date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).clip(0,1).astype(int)
    # Make cross-project IDs unique: repo#id
    df["Issue ID"]  = df["Repository"].astype(str) + "#" + df["Issue ID"].astype(str)
    df["Commit ID"] = df["Repository"].astype(str) + "#" + df["Commit ID"].astype(str)
    return df

def main():
    log.info(f"CUDA available: {TORCH_HAS_CUDA}")
    log.info("cuML TruncatedSVD: " + ("ENABLED (GPU)" if USE_CUML_SVD else "Not available, using sklearn CPU SVD"))

    with stage("[LOAD] Train/Test datasets"):
        train_dfs = [read_and_prepare(p) for p in train_paths]
        test_dfs  = [(Path(p).stem, read_and_prepare(p)) for p in test_paths]
        train_df = pd.concat(train_dfs, ignore_index=True)
        log.info(f"[TRAIN] rows={len(train_df):,}, issues={train_df['Issue ID'].nunique():,}, commits={train_df['Commit ID'].nunique():,}")

    train_wall_start = time.perf_counter()

    with stage("[TRAIN] TF-IDF/SVD (fit on TRAIN only)"):
        issue_txt_tr  = issue_text(train_df)
        commit_txt_tr = commit_text(train_df)

        tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
        X_tr = tfidf.fit_transform(pd.concat([issue_txt_tr["text"], commit_txt_tr["text"]], axis=0).fillna(""))

        if USE_CUML_SVD:
            svd = cuML_TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr)
        else:
            svd = SkTruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr)

        E_issue_tr  = Xr_tr[:len(issue_txt_tr)]
        E_commit_tr = Xr_tr[len(issue_txt_tr):]

        issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr["Issue ID"].tolist())}
        commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr["Commit ID"].tolist())}

    with stage("[TRAIN] Build file unions & features"):
        train_issue_files = issue_file_union_all(train_df)
        train_feat = build_features(train_df, issue_idx_tr, commit_idx_tr, None, None,
                                    E_issue_tr, E_commit_tr, None, None,
                                    train_issue_files, split="train", ds_tag="TRAIN")
        log.info(f"[TRAIN] train_feat rows={len(train_feat):,}")

    with stage("[TRAIN] Train A2 (commit→issues)"):
        uniq_train_commits = train_feat["Commit ID"].drop_duplicates().tolist()
        rng.shuffle(uniq_train_commits)
        n_dev_A2 = max(1, int(0.20 * len(uniq_train_commits)))
        dev_cids_A2 = set(uniq_train_commits[:n_dev_A2])

        trn_A2 = train_feat[~train_feat["Commit ID"].isin(dev_cids_A2)].copy()
        dev_A2 = train_feat[ train_feat["Commit ID"].isin(dev_cids_A2)].copy()

        trn_s_A2, Xtr_A2, ytr_A2, gtr_A2 = prep_rank(trn_A2, group_col="Commit ID")
        dev_s_A2, Xdv_A2, ydv_A2, gdv_A2 = prep_rank(dev_A2, group_col="Commit ID")

        model_A2 = train_lambdamart("A2", Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2)

    with stage("[TRAIN] Train A1 (issue→commits) + tune thresholds"):
        uniq_train_issues = train_feat["Issue ID"].drop_duplicates().tolist()
        rng.shuffle(uniq_train_issues)
        n_dev_A1 = max(1, int(0.20 * len(uniq_train_issues)))
        dev_iids_A1 = set(uniq_train_issues[:n_dev_A1])

        trn_A1 = train_feat[~train_feat["Issue ID"].isin(dev_iids_A1)].copy()
        dev_A1 = train_feat[ train_feat["Issue ID"].isin(dev_iids_A1)].copy()

        trn_s_A1, Xtr_A1, ytr_A1, gtr_A1 = prep_rank(trn_A1, group_col="Issue ID")
        dev_s_A1, Xdv_A1, ydv_A1, gdv_A1 = prep_rank(dev_A1, group_col="Issue ID")

        model_A1 = train_lambdamart("A1", Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1)

        dev_shortlist_like = topk_issues_per_commit(dev_A1, model_A2, K=TOPK_ISSUES_PER_COMMIT)
        true_by_issue_dev = dev_s_A1[dev_s_A1["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_dev = sorted(true_by_issue_dev.keys())
        dev_rows_by_issue = {iid: dev_s_A1[dev_s_A1["Issue ID"]==iid].reset_index(drop=True)
                             for iid in issues_dev}

        def aggregate(dfm):
            obj = (TUNE_OBJECTIVE or "F1").lower()
            if obj == "allcorrect":
                return dfm["AllCorrect"].mean()
            return dfm["F1"].mean()

        def eval_abs_threshold_A1(tau):
            rows = []
            for iid in issues_dev:
                pool_full = dev_rows_by_issue[iid]
                pool = restrict_pool_by_A2_shortlist(pool_full, dev_shortlist_like, fallback_to_full=True)
                ranked = score_pool(model_A1, pool, use_mm=True)
                chosen = ranked[ranked["score_mm"] >= float(tau)]["Commit ID"].tolist()
                rows.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
            dfm = pd.DataFrame(rows); return aggregate(dfm), dfm

        def eval_rel_threshold_A1(gamma):
            rows = []
            for iid in issues_dev:
                pool_full = dev_rows_by_issue[iid]
                pool = restrict_pool_by_A2_shortlist(pool_full, dev_shortlist_like, fallback_to_full=True)
                ranked = score_pool(model_A1, pool, use_mm=True)
                if len(ranked)==0:
                    rows.append(metrics_for_issue(set(), true_by_issue_dev[iid])); continue
                best = float(ranked["score_raw"].iloc[0])
                chosen = ranked[ranked["score_raw"] >= float(gamma) * best]["Commit ID"].tolist()
                rows.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
            dfm = pd.DataFrame(rows); return aggregate(dfm), dfm

        taus   = [x/100 for x in range(10, 96, 2)]   
        gammas = [x/100 for x in range(30, 96, 2)]   

        best_abs = (-1, None)
        for t in taus:
            sc, _ = eval_abs_threshold_A1(t)
            if sc > best_abs[0]: best_abs = (sc, t)

        best_rel = (-1, None)
        for g in gammas:
            sc, _ = eval_rel_threshold_A1(g)
            if sc > best_rel[0]: best_rel = (sc, g)

        log.info(f"[DEV][A1 No-K] objective={TUNE_OBJECTIVE} | ABS τ={best_abs[1]:.2f} | REL γ={best_rel[1]:.2f}")

    train_wall_end = time.perf_counter()
    total_train_seconds = round(train_wall_end - train_wall_start, 2)

    with stage("[SAVE] Train artifacts"):
        TA = OUT_ROOT / "train_artifacts"; TA.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"param":["SVD_DIM","TOPK_ISSUES_PER_COMMIT","TUNE_OBJECTIVE","USE_TIME_FEATURE","TIME_TAU_DAYS"],
                      "value":[SVD_DIM, TOPK_ISSUES_PER_COMMIT, TUNE_OBJECTIVE, USE_TIME_FEATURE, TIME_TAU_DAYS]}).to_csv(TA/"config.csv", index=False)
        model_A2.save_model(str(TA / "model_A2_commit2issues.txt"))
        model_A1.save_model(str(TA / "model_A1_issue2commits.txt"))
        train_df.to_csv(TA / "train_rows.csv", index=False)

    all_summary_rows, all_timing_rows = [], []

    for name, test_df in test_dfs:
        with stage(f"[{name}] TF-IDF/SVD transform (TEST only)"):
            issue_txt_te  = issue_text(test_df)
            commit_txt_te = commit_text(test_df)
            X_te_issue  = svd.transform(tfidf.transform(issue_txt_te["text"].fillna("")))
            X_te_commit = svd.transform(tfidf.transform(commit_txt_te["text"].fillna("")))
            issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te["Issue ID"].tolist())}
            commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te["Commit ID"].tolist())}

        with stage(f"[{name}] Build file unions & features (TEST)"):
            test_issue_files  = issue_file_union_all(test_df)
            test_feat  = build_features(test_df, None, None, issue_idx_te, commit_idx_te,
                                        None, None, X_te_issue, X_te_commit,
                                        test_issue_files, split="test", ds_tag=name)
            log.info(f"[{name}] test_feat rows={len(test_feat):,}")

        test_start = time.perf_counter()

        with stage(f"[{name}] Build A2 shortlist for TEST"):
            test_shortlist = topk_issues_per_commit(test_feat, model_A2, K=TOPK_ISSUES_PER_COMMIT)

        with stage(f"[{name}] Evaluate (A1 No-K + Oracle-K)"):
            ds_out = OUT_ROOT / f"eval_{name}"; ds_out.mkdir(parents=True, exist_ok=True)

            def prep_rank_issue(df_feat):
                df_s = df_feat.sort_values(["Issue ID"]).reset_index(drop=True)
                return df_s
            tst_s_A1 = prep_rank_issue(test_feat)

            tst_rows_by_issue = {iid: tst_s_A1[tst_s_A1["Issue ID"]==iid].reset_index(drop=True)
                                 for iid in tst_s_A1["Issue ID"].drop_duplicates().tolist()}
            true_by_issue_test = tst_s_A1[tst_s_A1["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
            issues_test = sorted(true_by_issue_test.keys())

            def eval_policy_on_test_A1_NoK(kind, val, tag, shortlist_map):
                rows = []; rank_dump = []
                for iid in issues_test:
                    pool_full = tst_rows_by_issue[iid]
                    pool = restrict_pool_by_A2_shortlist(pool_full, shortlist_map, fallback_to_full=True)
                    ranked = score_pool(model_A1, pool, use_mm=True)
                    if len(ranked) == 0:
                        pred_set = set()
                    else:
                        if kind == "ABS":
                            chosen = ranked[ranked["score_mm"] >= float(val)]["Commit ID"].tolist()
                        else:
                            best = float(ranked["score_raw"].iloc[0])
                            chosen = ranked[ranked["score_raw"] >= float(val) * best]["Commit ID"].tolist()
                        pred_set = set(chosen)
                    true_set = true_by_issue_test.get(iid, set())
                    m = metrics_for_issue(pred_set, true_set)
                    rows.append(dict(Issue=iid, **m, Predicted=len(pred_set), TrueCount=len(true_set)))
                    tmp = ranked[["Commit ID","score_raw","score_mm","Output"]].copy()
                    tmp["Issue ID"] = iid; tmp["Rank"] = np.arange(1, len(tmp)+1)
                    rank_dump.append(tmp)
                res_df = pd.DataFrame(rows)
                rank_df = pd.concat(rank_dump, ignore_index=True) if rank_dump else pd.DataFrame()
                res_df.to_csv(ds_out / f"hybrid_NoK_{tag}.csv", index=False)
                rank_df.to_csv(ds_out / f"hybrid_NoK_ranked_{tag}.csv", index=False)
                log.info(f"[{name}] No-K {tag} ,  P={res_df['Precision'].mean():.4f} R={res_df['Recall'].mean():.4f} F1={res_df['F1'].mean():.4f}")
                return res_df

            def eval_test_A1_OracleK(shortlist_map, tag="OracleK_topK_after_A2top5"):
                rows = []; rank_dump = []
                for iid in issues_test:
                    pool_full = tst_rows_by_issue[iid]
                    pool = restrict_pool_by_A2_shortlist(pool_full, shortlist_map, fallback_to_full=True)
                    ranked = score_pool(model_A1, pool, use_mm=True)
                    true_set = true_by_issue_test.get(iid, set())
                    K = len(true_set)
                    chosen = ranked["Commit ID"].iloc[:K].tolist() if K>0 else []
                    pred_set = set(chosen)
                    m = metrics_for_issue(pred_set, true_set)
                    rows.append(dict(Issue=iid, **m, Predicted=len(pred_set), TrueCount=len(true_set)))
                    tmp = ranked[["Commit ID","score_raw","score_mm","Output"]].copy()
                    tmp["Issue ID"] = iid; tmp["Rank"] = np.arange(1, len(tmp)+1)
                    rank_dump.append(tmp)
                res_df = pd.DataFrame(rows)
                rank_df = pd.concat(rank_dump, ignore_index=True) if rank_dump else pd.DataFrame()
                res_df.to_csv(ds_out / f"hybrid_{tag}.csv", index=False)
                rank_df.to_csv(ds_out / f"hybrid_ranked_{tag}.csv", index=False)
                log.info(f"[{name}] {tag} ,  P={res_df['Precision'].mean():.4f} R={res_df['Recall'].mean():.4f} F1={res_df['F1'].mean():.4f}")
                return res_df

            tag_abs = f"ABSmm_tau{best_abs[1]:.2f}_{TUNE_OBJECTIVE}"
            tag_rel = f"REL_gamma{best_rel[1]:.2f}_{TUNE_OBJECTIVE}"

            res_abs = eval_policy_on_test_A1_NoK("ABS", best_abs[1], tag_abs, test_shortlist)
            res_rel = eval_policy_on_test_A1_NoK("REL", best_rel[1], tag_rel, test_shortlist)
            res_orc = eval_test_A1_OracleK(test_shortlist, tag="OracleK_topK_after_A2top5")

        test_end = time.perf_counter()
        test_secs = round(test_end - test_start, 2)

        pABS, rABS, fABS = macro_percent(res_abs)
        pREL, rREL, fREL = macro_percent(res_rel)
        pOK,  rOK,  fOK  = macro_percent(res_orc)

        all_summary_rows.extend([
            [name, "No-K (ABS-mm)", pABS, rABS, fABS],
            [name, "No-K (REL)",    pREL, rREL, fREL],
            [name, "Oracle-K",      pOK,  rOK,  fOK ],
        ])
        all_timing_rows.append([name, total_train_seconds, test_secs])


    with stage("[SAVE] Rollup CSVs"):
        summary_df = pd.DataFrame(all_summary_rows, columns=["Dataset","Setting","Precision","Recall","F1"])
        summary_df.to_csv(SUMMARY_CSV, index=False)

        print("\nSummary written to:", SUMMARY_CSV)
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()

