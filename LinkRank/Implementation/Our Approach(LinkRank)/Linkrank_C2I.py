import os, re, time, logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

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

# ============ CONFIG ============
dataset_paths = [
    "Add your file paths here",

]

OUT_ROOT = Path("Add your output directory here")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH    = OUT_ROOT / "run_c2i_cv5.log"

RANDOM_SEED = 42
K_FOLDS     = 5
SVD_DIM     = 256
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
log = logging.getLogger("linkrank_c2i_nocodebert_cv5")

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

rng = np.random.default_rng(RANDOM_SEED)
log.info(f"CUDA available: {TORCH_HAS_CUDA}")
log.info("cuML TruncatedSVD: " + ("ENABLED (GPU)" if USE_CUML_SVD else "Not available, using sklearn CPU SVD"))


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

def prep_rank(df_feat, group_col):
    df_s = df_feat.sort_values([group_col]).reset_index(drop=True)
    X = df_s[FEATURE_COLS].values
    y = df_s["Output"].astype(int).values
    groups = df_s.groupby(group_col).size().tolist()
    return df_s, X, y, groups

def ensure_nonempty_valid(Xtr, ytr, gtr, Xdv, ydv, gdv):
    """If dev is empty, carve out a tiny slice from train as dev (keeps flow intact)."""
    if len(Xdv) > 0 and len(gdv) > 0:
        return Xtr, ytr, gtr, Xdv, ydv, gdv
    n = len(ytr)
    if n < 2:
        return Xtr, ytr, gtr, Xtr, ytr, gtr
    cut = max(1, n // 10)
    return Xtr[cut:], ytr[cut:], gtr, Xtr[:cut], ytr[:cut], gtr

def train_lambdamart(ds_name, tag, Xtr, ytr, gtr, Xdv, ydv, gdv):
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
        log.info(f"[{ds_name}][{tag}] LightGBM: GPU enabled.")
    else:
        log.info(f"[{ds_name}][{tag}] LightGBM: CPU.")
    model = lgb.train(
        params, dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=2000,
        callbacks=[
            lgb.log_evaluation(LOG_EVERY_N_ITERS),
            lgb.early_stopping(stopping_rounds=150, verbose=True)
        ]
    )
    log.info(f"[{ds_name}][{tag}] Best iteration: {model.best_iteration}")
    return model

def build_features(d, issue_idx_tr, commit_idx_tr, issue_idx_te, commit_idx_te,
                   E_issue_tr, E_commit_tr, X_te_issue, X_te_commit,
                   issue_files_union, split, ds_name):
    rows = []
    it = tqdm(d.iterrows(), total=len(d), desc=f"build_features[{ds_name}][{split}]")
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
        rows.append([iid, cid, f_text, f_file, f_time, row["Output"]])
    return pd.DataFrame(rows, columns=["Issue ID","Commit ID",*FEATURE_COLS,"Output"])


def run_one_fold(ds_name, full_df, train_ids, test_ids, fold_dir):
    fold_dir.mkdir(parents=True, exist_ok=True)


    train_df = full_df[full_df["Issue ID"].isin(train_ids)].copy()
    test_df  = full_df[full_df["Issue ID"].isin(test_ids)].copy()


    with stage(f"[{ds_name}] TF-IDF/SVD (fit on TRAIN; transform TEST)"):
        issue_txt_tr = issue_text(train_df); commit_txt_tr = commit_text(train_df)
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

        issue_txt_te = issue_text(test_df); commit_txt_te = commit_text(test_df)
        X_te_issue  = svd.transform(tfidf.transform(issue_txt_te["text"].fillna("")))
        X_te_commit = svd.transform(tfidf.transform(commit_txt_te["text"].fillna("")))

        issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr["Issue ID"].tolist())}
        commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr["Commit ID"].tolist())}
        issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te["Issue ID"].tolist())}
        commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te["Commit ID"].tolist())}


    with stage(f"[{ds_name}] Build file unions & features (TRAIN/TEST)"):
        train_issue_files = issue_file_union_all(train_df)
        test_issue_files  = issue_file_union_all(test_df)

        train_feat = build_features(train_df, issue_idx_tr, commit_idx_tr, None, None,
                                    E_issue_tr, E_commit_tr, None, None,
                                    train_issue_files, split="train", ds_name=ds_name)

        test_feat  = build_features(test_df,  None, None, issue_idx_te, commit_idx_te,
                                    None, None, X_te_issue, X_te_commit,
                                    test_issue_files,  split="test",  ds_name=ds_name)

    def prep_rank_commit(df_feat):
        return prep_rank(df_feat, group_col="Commit ID")

    with stage(f"[{ds_name}] Train A2 (commit→issues)"):
        uniq_train_commits = train_feat["Commit ID"].drop_duplicates().tolist()
        rng.shuffle(uniq_train_commits)
        n_dev_A2 = max(1, int(0.20 * len(uniq_train_commits)))
        dev_cids_A2 = set(uniq_train_commits[:n_dev_A2])

        trn_A2 = train_feat[~train_feat["Commit ID"].isin(dev_cids_A2)].copy()
        dev_A2 = train_feat[ train_feat["Commit ID"].isin(dev_cids_A2)].copy()

        trn_s_A2, Xtr_A2, ytr_A2, gtr_A2 = prep_rank_commit(trn_A2)
        dev_s_A2, Xdv_A2, ydv_A2, gdv_A2 = prep_rank_commit(dev_A2)

        Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2 = ensure_nonempty_valid(
            Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2
        )

        model_A2 = train_lambdamart(ds_name, "A2", Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2)

    def score_pool_A2(pool_df):
        X = pool_df[FEATURE_COLS].values
        s = model_A2.predict(X, num_iteration=model_A2.best_iteration)
        out = pool_df.copy()
        out["score_raw"] = s
        if len(out) > 1:
            mn, mx = float(out["score_raw"].min()), float(out["score_raw"].max())
            out["score_mm"] = 1.0 if mx==mn else (out["score_raw"] - mn) / (mx - mn)
        else:
            out["score_mm"] = 1.0
        return out.sort_values("score_raw", ascending=False).reset_index(drop=True)

    def topk_issues_per_commit(df_split, K=TOPK_ISSUES_PER_COMMIT):
        res = {}
        by_commit = {cid: df_split[df_split["Commit ID"]==cid].reset_index(drop=True)
                     for cid in df_split["Commit ID"].drop_duplicates().tolist()}
        for cid, pool in by_commit.items():
            ranked = score_pool_A2(pool)
            res[cid] = set(ranked["Issue ID"].iloc[:K].tolist())
        return res

    with stage(f"[{ds_name}] Build A2 shortlist on DEV and TEST"):

        dev_shortlist_map = topk_issues_per_commit(train_feat[train_feat["Commit ID"].isin(dev_A2["Commit ID"].unique())])

        test_shortlist_map = topk_issues_per_commit(test_feat)

    def prep_rank_issue(df_feat):
        return prep_rank(df_feat, group_col="Issue ID")

    with stage(f"[{ds_name}] Train A1 (issue→commits)"):
        uniq_train_issues = train_feat["Issue ID"].drop_duplicates().tolist()
        rng.shuffle(uniq_train_issues)
        n_dev_A1 = max(1, int(0.20 * len(uniq_train_issues)))
        dev_iids_A1 = set(uniq_train_issues[:n_dev_A1])

        trn_A1 = train_feat[~train_feat["Issue ID"].isin(dev_iids_A1)].copy()
        dev_A1 = train_feat[ train_feat["Issue ID"].isin(dev_iids_A1)].copy()

        trn_s_A1, Xtr_A1, ytr_A1, gtr_A1 = prep_rank_issue(trn_A1)
        dev_s_A1, Xdv_A1, ydv_A1, gdv_A1 = prep_rank_issue(dev_A1)
        tst_s_A1, Xte_A1, yte_A1, gte_A1 = prep_rank_issue(test_feat)

        Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1 = ensure_nonempty_valid(
            Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1
        )

        model_A1 = train_lambdamart(ds_name, "A1", Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1)

    def score_pool_A1(pool_df):
        X = pool_df[FEATURE_COLS].values
        s = model_A1.predict(X, num_iteration=model_A1.best_iteration)
        out = pool_df.copy()
        if len(out) > 1:
            mn, mx = float(s.min()), float(s.max())
            out["score_raw"] = s
            out["score_mm"] = 1.0 if mx==mn else (out["score_raw"] - mn) / (mx - mn)
        else:
            out["score_raw"] = s
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

    with stage(f"[{ds_name}] No-K tuning on DEV (A1, gated by A2 DEV shortlist)"):
        dev_rows_by_issue = {iid: dev_s_A1[dev_s_A1["Issue ID"]==iid].reset_index(drop=True)
                             for iid in dev_s_A1["Issue ID"].drop_duplicates().tolist()}
        true_by_issue_dev = dev_s_A1[dev_s_A1["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_dev = sorted(true_by_issue_dev.keys())

        def aggregate(dfm):
            if TUNE_OBJECTIVE.lower() == "allcorrect":
                return dfm["AllCorrect"].mean()
            return dfm["F1"].mean()

        def eval_abs_threshold_A1(tau):
            rows = []
            for iid in issues_dev:
                pool_full = dev_rows_by_issue[iid]
                pool = restrict_pool_by_A2_shortlist(pool_full, dev_shortlist_map, fallback_to_full=True)
                ranked = score_pool_A1(pool)
                chosen = ranked[ranked["score_mm"] >= float(tau)]["Commit ID"].tolist()
                rows.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
            dfm = pd.DataFrame(rows); return aggregate(dfm)

        def eval_rel_threshold_A1(gamma):
            rows = []
            for iid in issues_dev:
                pool_full = dev_rows_by_issue[iid]
                pool = restrict_pool_by_A2_shortlist(pool_full, dev_shortlist_map, fallback_to_full=True)
                ranked = score_pool_A1(pool)
                if len(ranked)==0:
                    rows.append(metrics_for_issue(set(), true_by_issue_dev[iid])); continue
                best = float(ranked["score_raw"].iloc[0])
                chosen = ranked[ranked["score_raw"] >= float(gamma) * best]["Commit ID"].tolist()
                rows.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
            dfm = pd.DataFrame(rows); return aggregate(dfm)

        taus   = [x/100 for x in range(10, 96, 2)]   # 0.10..0.95
        gammas = [x/100 for x in range(30, 96, 2)]   # 0.30..0.95

        best_abs = (-1, None)
        for t in taus:
            sc = eval_abs_threshold_A1(t)
            if sc > best_abs[0]: best_abs = (sc, t)

        best_rel = (-1, None)
        for g in gammas:
            sc = eval_rel_threshold_A1(g)
            if sc > best_rel[0]: best_rel = (sc, g)

        log.info(f"[{ds_name}][DEV][A1 No-K] objective={TUNE_OBJECTIVE} | ABS τ={best_abs[1]:.2f} | REL γ={best_rel[1]:.2f}")


    with stage(f"[{ds_name}] Evaluate on TEST (A1 gated by A2 TEST shortlist)"):
        tst_rows_by_issue = {iid: tst_s_A1[tst_s_A1["Issue ID"]==iid].reset_index(drop=True)
                             for iid in tst_s_A1["Issue ID"].drop_duplicates().tolist()}
        true_by_issue_test = tst_s_A1[tst_s_A1["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_test = sorted(true_by_issue_test.keys())

        def eval_policy_on_test_A1_NoK(kind, val, tag, shortlist_map):
            rows = []; rank_dump = []
            for iid in issues_test:
                pool_full = tst_rows_by_issue[iid]
                pool = restrict_pool_by_A2_shortlist(pool_full, shortlist_map, fallback_to_full=True)
                ranked = score_pool_A1(pool)
                if len(ranked) == 0:
                    pred_set = set()
                else:
                    if kind == "ABS":
                        chosen = ranked[ranked["score_mm"] >= float(val)]["Commit ID"].tolist()
                    else:
                        best = float(ranked["score_raw"].iloc[0])
                        chosen = ranked[ranked["score_raw"] >= float(val) * best]["Commit ID"].tolist()
                    pred_set = set(chosen)
                true_set = true_by_issue_test[iid]
                m = metrics_for_issue(pred_set, true_set)
                rows.append(dict(Issue=iid, **m, Predicted=len(pred_set), TrueCount=len(true_set)))
                tmp = ranked[["Commit ID","score_raw","score_mm","Output"]].copy()
                tmp["Issue ID"] = iid; tmp["Rank"] = np.arange(1, len(tmp)+1)
                rank_dump.append(tmp)
            res_df = pd.DataFrame(rows)
            rank_df = pd.concat(rank_dump, ignore_index=True) if rank_dump else pd.DataFrame()
            res_df.to_csv(fold_dir / f"hybrid_NoK_{tag}.csv", index=False)
            rank_df.to_csv(fold_dir / f"hybrid_NoK_ranked_{tag}.csv", index=False)
            return res_df

        def eval_test_A1_OracleK(shortlist_map, tag="OracleK_topK_after_A2top5"):
            rows = []; rank_dump = []
            for iid in issues_test:
                pool_full = tst_rows_by_issue[iid]
                pool = restrict_pool_by_A2_shortlist(pool_full, shortlist_map, fallback_to_full=True)
                ranked = score_pool_A1(pool)
                true_set = true_by_issue_test[iid]
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
            res_df.to_csv(fold_dir / f"hybrid_{tag}.csv", index=False)
            rank_df.to_csv(fold_dir / f"hybrid_ranked_{tag}.csv", index=False)
            return res_df

        tag_abs = f"ABSmm_tau{best_abs[1]:.2f}_{TUNE_OBJECTIVE}"
        tag_rel = f"REL_gamma{best_rel[1]:.2f}_{TUNE_OBJECTIVE}"

        res_abs = eval_policy_on_test_A1_NoK("ABS", best_abs[1], tag_abs, test_shortlist_map)
        res_rel = eval_policy_on_test_A1_NoK("REL", best_rel[1], tag_rel, test_shortlist_map)
        res_ok  = eval_test_A1_OracleK(test_shortlist_map, tag="OracleK_topK_after_A2top5")


    def pct3(df): return macro_percent(df)

    pABS, rABS, fABS = pct3(res_abs)
    pREL, rREL, fREL = pct3(res_rel)
    pOK,  rOK,  fOK  = pct3(res_ok)

    summary_rows = [
        ["No-K (ABS-mm)", pABS, rABS, fABS],
        ["No-K (REL)",    pREL, rREL, fREL],
        ["Oracle-K",      pOK,  rOK,  fOK ],
    ]
    sum_df = pd.DataFrame(summary_rows, columns=["Setting","Precision","Recall","F1"])
    return sum_df

if __name__ == "__main__":
    all_means = []  

    for path in dataset_paths:
        ds_name = Path(path).stem
        print(f"Running 5-fold CV for dataset: {ds_name}")


        with stage(f"[{ds_name}] Load CSV"):
            df = pd.read_csv(path)
            for c in ["Issue Date","Commit Date"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            issue_ids = sorted(df["Issue ID"].drop_duplicates().tolist())

        DS_OUT = OUT_ROOT / ds_name / "cv5"
        DS_OUT.mkdir(parents=True, exist_ok=True)

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        fold_summaries = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(issue_ids), start=1):
            fold_dir = DS_OUT / f"fold{fold_idx}"
            train_ids = set([issue_ids[i] for i in train_idx])
            test_ids  = set([issue_ids[i] for i in test_idx])

            log.info(f"[{ds_name}] ===== CV fold{fold_idx}: train_issues={len(train_ids)}, test_issues={len(test_ids)} =====")
            sum_df = run_one_fold(ds_name, df, train_ids, test_ids, fold_dir)

            (DS_OUT / f"summary_fold{fold_idx}.csv").write_text(sum_df.to_csv(index=False))
            fold_summaries.append(sum_df.assign(Fold=fold_idx))


        all_summ = pd.concat(fold_summaries, ignore_index=True)


        for col in ["Precision", "Recall", "F1"]:
            all_summ[col] = pd.to_numeric(all_summ[col], errors="coerce")


        setting_order = ["No-K (ABS-mm)", "No-K (REL)", "Oracle-K"]


        mean_df = (
            all_summ
            .groupby("Setting")[["Precision", "Recall", "F1"]]
            .mean()
            .reindex(setting_order)         
            .reset_index()
            .round(2)
        )

        mean_df.to_csv(DS_OUT / "summary_k5_mean.csv", index=False)

        mean_df_with_ds = mean_df.copy()
        mean_df_with_ds.insert(0, "Dataset", ds_name)
        all_means.append(mean_df_with_ds)

        print(f"[DONE] {ds_name}: wrote 5 per-fold summaries + summary_k5_mean.csv in {DS_OUT}")

    if all_means:
        master_df = pd.concat(all_means, ignore_index=True)
        master_out = OUT_ROOT / "summary_k5_mean_all_datasets.csv"
        master_df.to_csv(master_out, index=False)
        print(f"[MASTER] Wrote stacked means to: {master_out}")
