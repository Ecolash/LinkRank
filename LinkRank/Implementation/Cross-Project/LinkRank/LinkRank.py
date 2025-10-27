
import os, re, time, logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import psutil
    def mem_gb(): return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
except Exception:
    psutil = None
    def mem_gb(): return float('nan')

try:
    import torch
    TORCH_HAS_CUDA = torch.cuda.is_available()
    torch_device = torch.device("cuda" if TORCH_HAS_CUDA else "cpu")
except Exception:
    TORCH_HAS_CUDA = False
    class _DummyDevice:
        def __str__(self): return "cpu (torch not installed)"
    torch_device = _DummyDevice()

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
SUMMARY_CSV = OUT_ROOT / "summaries.csv"
TIMINGS_CSV = OUT_ROOT / "timings.csv"
LOG_PATH    = OUT_ROOT / "run.log"
RANDOM_SEED = 42
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
SVD_DIM      = 256
USE_TIME_FEATURE = True
TIME_TAU_DAYS    = 7.0
USE_ITERATION_KNOWNK  = True
USE_ITERATION_NOK_REL = True
ALPHA = 0.7   
BETA  = 0.3   
LOG_EVERY_N_ITERS = 10
TUNE_OBJECTIVE = "F1"    
USE_CAP = False
MAX_RELATIVE = 0.75

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger("approach1_no_jaccard_xproj")

def stage(name):
    class _Stage:
        def __enter__(self):
            self.name = name; self.t0 = time.perf_counter()
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


log.info(f"Using device: {torch_device}")
log.info("cuML TruncatedSVD: " + ("ENABLED (GPU)" if USE_CUML_SVD else "not available, using sklearn CPU SVD"))


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
def tokenize(s: str):
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def cos_sim(a, b):
    if a is None or b is None: return 0.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

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
    out = (g["title"].fillna("") + " " + g["desc"].fillna("") + " " + g["comm"].fillna("")).reset_index(name="text")
    return out

def commit_text(d):
    g = d.groupby("Commit ID").agg(
        msg   =("Message","first"),
        dif   =("Diff Summary","first"),
        files =("File Changes","first"),
        full  =("Full Diff","first"),
    )
    out = (g["msg"].fillna("") + " " + g["dif"].fillna("") + " " + g["files"].fillna("") + " " + g["full"].fillna("")).reset_index(name="text")
    return out


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

    df["Issue ID"]  = df["Repository"].astype(str) + "#" + df["Issue ID"].astype(str)
    df["Commit ID"] = df["Repository"].astype(str) + "#" + df["Commit ID"].astype(str)
    return df

def main():

    with stage("[LOAD] Train/Test datasets"):
        train_dfs = [read_and_prepare(p) for p in train_paths]
        test_dfs  = [(Path(p).stem, read_and_prepare(p)) for p in test_paths]
        train_df = pd.concat(train_dfs, ignore_index=True)
        log.info(f"[TRAIN] rows={len(train_df):,} | issues={train_df['Issue ID'].nunique():,} | commits={train_df['Commit ID'].nunique():,}")


    with stage("[TRAIN] Build train texts"):
        issue_txt_tr  = issue_text(train_df)
        commit_txt_tr = commit_text(train_df)

    with stage("[TRAIN] TF-IDF fit + transform; SVD fit + transform"):
        tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
        X_tr_cat = tfidf.fit_transform(pd.concat([issue_txt_tr["text"], commit_txt_tr["text"]], axis=0).fillna(""))
        if USE_CUML_SVD:
            svd = cuML_TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr_cat)
        else:
            svd = SkTruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr_cat)
        E_issue_tr  = Xr_tr[:len(issue_txt_tr)]
        E_commit_tr = Xr_tr[len(issue_txt_tr):]
        issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr["Issue ID"].tolist())}
        commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr["Commit ID"].tolist())}


    FEATURES = ["feat_text","feat_time"]

    with stage("[TRAIN] Build pair features"):
        def build_features_train(d):
            rows = []
            it = tqdm(d.iterrows(), total=len(d), desc="build_features[TRAIN]")
            for _, row in it:
                iid = row["Issue ID"]; cid = row["Commit ID"]
                v_i = E_issue_tr[issue_idx_tr[iid]]  if iid in issue_idx_tr  else None
                v_c = E_commit_tr[commit_idx_tr[cid]] if cid in commit_idx_tr else None
                f_text = cos_sim(v_i, v_c)
                f_time = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS) if USE_TIME_FEATURE else 0.0
                rows.append([iid, cid, f_text, f_time, row["Output"]])
            return pd.DataFrame(rows, columns=["Issue ID","Commit ID",*FEATURES,"Output"])

        train_feat = build_features_train(train_df)


    def prep_rank(df_feat):
        df_s = df_feat.sort_values(["Issue ID"]).reset_index(drop=True)
        X = df_s[FEATURES].values
        y = df_s["Output"].astype(int).values
        groups = df_s.groupby("Issue ID").size().tolist()
        return df_s, X, y, groups

    with stage("[TRAIN] Split into train/dev; train LambdaMART"):
        uniq_train_issues = train_feat["Issue ID"].drop_duplicates().tolist()
        rng.shuffle(uniq_train_issues)
        n_dev = max(1, int(0.20 * len(uniq_train_issues)))
        dev_ids = set(uniq_train_issues[:n_dev])

        trn_feat = train_feat[~train_feat["Issue ID"].isin(dev_ids)].copy()
        dev_feat = train_feat[ train_feat["Issue ID"].isin(dev_ids)].copy()


        def _filter_groups_with_labels(df_feat, require_neg=True):
            grp = df_feat.groupby("Issue ID")["Output"]
            pos = grp.sum(); cnt = grp.count()
            ok = (pos >= 1) & ((cnt - pos) >= 1) if require_neg else (pos >= 1)
            keep = set(pos[ok].index.tolist())
            return df_feat[df_feat["Issue ID"].isin(keep)].copy()

        trn_feat = _filter_groups_with_labels(trn_feat, require_neg=True)
        dev_feat = _filter_groups_with_labels(dev_feat, require_neg=True)

        trn_s, Xtr, ytr, gtr = prep_rank(trn_feat)
        dev_s, Xdv, ydv, gdv = prep_rank(dev_feat)

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
            log.info("[TRAIN] LightGBM: GPU enabled.")
        else:
            log.info("[TRAIN] LightGBM: using CPU.")

        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEATURES)
        dvalid = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain, feature_name=FEATURES)

        model = lgb.train(
            params, dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=2000,
            callbacks=[
                lgb.log_evaluation(LOG_EVERY_N_ITERS),
                lgb.early_stopping(stopping_rounds=150, verbose=True)
            ]
        )
        log.info(f"[TRAIN] Best iteration: {model.best_iteration}")


    def score_pool(pool_df):
        if len(pool_df)==0:
            return pool_df.assign(score=[], score_mm=[], score_zn=[])
        X = pool_df[FEATURES].values
        s = model.predict(X, num_iteration=model.best_iteration)
        out = pool_df.copy()
        out["score"] = s
        if len(out) <= 1:
            out["score_mm"] = 1.0; out["score_zn"] = 0.0
        else:
            mn, mx = float(out["score"].min()), float(out["score"].max())
            out["score_mm"] = 1.0 if mx==mn else (out["score"] - mn) / (mx - mn)
            mu, sd = float(out["score"].mean()), float(out["score"].std(ddof=0) + 1e-9)
            out["score_zn"] = (out["score"] - mu) / sd
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    def metrics_for_issue(pred_set, true_set):
        inter = len(pred_set & true_set)
        p = inter / max(len(pred_set), 1)
        r = inter / max(len(true_set), 1)
        f1 = (2*inter) / max(len(pred_set)+len(true_set), 1)
        allc = int(pred_set == true_set)
        half = int(inter >= int(np.ceil(len(true_set)/2)))
        return dict(Precision=p, Recall=r, F1=f1, AllCorrect=allc, HalfCorrect=half)

    def apply_cap(chosen_df):
        if not USE_CAP or len(chosen_df) == 0:
            return chosen_df
        top = float(chosen_df["score"].iloc[0])
        return chosen_df[chosen_df["score"] >= MAX_RELATIVE * top]

    with stage("[DEV] Tune No-K thresholds"):
        dev_rows_by_issue = {iid: dev_s[dev_s["Issue ID"]==iid].reset_index(drop=True)
                             for iid in dev_s["Issue ID"].drop_duplicates().tolist()}
        ranked_dev = {iid: score_pool(df) for iid, df in dev_rows_by_issue.items()}
        true_by_issue_dev = dev_s[dev_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_dev = sorted(true_by_issue_dev.keys())


        taus_mm = [x/100 for x in range(10, 96, 2)]
        best_abs_mm = (-1, None)
        for t in taus_mm:
            rows=[]
            for iid in issues_dev:
                ranked = ranked_dev[iid]
                chosen = apply_cap(ranked[ranked["score_mm"] >= float(t)].copy())
                pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_dev[iid]
                rows.append(metrics_for_issue(pred, true))
            dfm = pd.DataFrame(rows)
            sc = dfm["AllCorrect"].mean() if TUNE_OBJECTIVE.lower()=="allcorrect" else dfm["F1"].mean()
            if sc > best_abs_mm[0]: best_abs_mm = (sc, t)


        gammas  = [x/100 for x in range(30, 96, 2)]
        best_rel = (-1, None)
        for g in gammas:
            rows=[]
            for iid in issues_dev:
                ranked = ranked_dev[iid]
                if len(ranked)==0:
                    rows.append(metrics_for_issue(set(), true_by_issue_dev[iid])); continue
                best = float(ranked["score"].iloc[0])
                chosen = apply_cap(ranked[ranked["score"] >= float(g)*best].copy())
                pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_dev[iid]
                rows.append(metrics_for_issue(pred, true))
            dfm = pd.DataFrame(rows)
            sc = dfm["AllCorrect"].mean() if TUNE_OBJECTIVE.lower()=="allcorrect" else dfm["F1"].mean()
            if sc > best_rel[0]: best_rel = (sc, g)

        log.info(f"[DEV] best ABS-mm τ={best_abs_mm[1]:.2f} | best REL γ={best_rel[1]:.2f}")


    with stage("[SAVE] Training artifacts"):
        TA = OUT_ROOT / "train_artifacts"; TA.mkdir(parents=True, exist_ok=True)
        model.save_model(str(TA / "lambdamart_issue2commits.txt"))
        train_df.to_csv(TA / "train_rows.csv", index=False)


    all_summary_rows, all_timing_rows = [], []

    for name, test_df in test_dfs:
        ds_out = OUT_ROOT / f"eval_{name}"
        ds_out.mkdir(parents=True, exist_ok=True)

        with stage(f"[{name}] Build texts + transform with TRAIN tfidf/svd"):
            issue_txt_te  = issue_text(test_df)
            commit_txt_te = commit_text(test_df)
            X_te_issue  = svd.transform(tfidf.transform(issue_txt_te["text"].fillna("")))
            X_te_commit = svd.transform(tfidf.transform(commit_txt_te["text"].fillna("")))
            issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te["Issue ID"].tolist())}
            commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te["Commit ID"].tolist())}

        with stage(f"[{name}] Build TEST features"):
            def build_features_test(d):
                rows = []
                it = tqdm(d.iterrows(), total=len(d), desc=f"build_features[{name}][TEST]")
                for _, row in it:
                    iid = row["Issue ID"]; cid = row["Commit ID"]
                    v_i = X_te_issue[issue_idx_te[iid]]  if iid in issue_idx_te  else None
                    v_c = X_te_commit[commit_idx_te[cid]] if cid in commit_idx_te else None
                    f_text = cos_sim(v_i, v_c)
                    f_time = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS) if USE_TIME_FEATURE else 0.0
                    rows.append([iid, cid, f_text, f_time, row["Output"]])
                return pd.DataFrame(rows, columns=["Issue ID","Commit ID",*FEATURES,"Output"])

            test_feat = build_features_test(test_df)

        with stage(f"[{name}] Prep rank frames"):
            tst_s = test_feat.sort_values(["Issue ID"]).reset_index(drop=True)
            tst_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                                 for iid in tst_s["Issue ID"].drop_duplicates().tolist()}
            true_by_issue = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
            issues_test = sorted(true_by_issue.keys())


        issue_vec_state = {}
        def init_issue_state(iid):
            v = X_te_issue[ issue_idx_te[iid] ].copy() if iid in issue_idx_te else None
            if v is None: v = np.zeros((SVD_DIM,), dtype=np.float32)
            issue_vec_state[iid] = v / (np.linalg.norm(v) + 1e-12)

        def update_issue_state(iid, cid):
            v_issue = issue_vec_state[iid]
            v_commit = X_te_commit[ commit_idx_te[cid] ] if cid in commit_idx_te else np.zeros_like(v_issue)
            v_new = ALPHA * v_issue + BETA * v_commit
            issue_vec_state[iid] = v_new / (np.linalg.norm(v_new) + 1e-12)

        def refresh_features_for_issue_pool(iid, pool_df):
            if len(pool_df)==0: return pool_df
            v_issue = issue_vec_state[iid]
            new_rows = []
            for _, r in pool_df.iterrows():
                cid = r["Commit ID"]
                v_commit = X_te_commit[ commit_idx_te[cid] ] if cid in commit_idx_te else None
                f_text = cos_sim(v_issue, v_commit)
                new_rows.append([r["Issue ID"], r["Commit ID"], f_text, r["feat_time"], r["Output"]])
            return pd.DataFrame(new_rows, columns=["Issue ID","Commit ID","feat_text","feat_time","Output"])


        def score_pool_local(pool_df):
            if len(pool_df)==0:
                return pool_df.assign(score=[], score_mm=[], score_zn=[])
            X = pool_df[FEATURES].values
            s = model.predict(X, num_iteration=model.best_iteration)
            out = pool_df.copy()
            out["score"] = s
            if len(out) <= 1:
                out["score_mm"] = 1.0; out["score_zn"] = 0.0
            else:
                mn, mx = float(out["score"].min()), float(out["score"].max())
                out["score_mm"] = 1.0 if mx==mn else (out["score"] - mn) / (mx - mn)
                mu, sd = float(out["score"].mean()), float(out["score"].std(ddof=0) + 1e-9)
                out["score_zn"] = (out["score"] - mu) / sd
            return out.sort_values("score", ascending=False).reset_index(drop=True)


        t_test_start = time.perf_counter()
        with stage(f"[{name}] Evaluate Known-K (iterative)"):
            rowsK = []
            for iid in tqdm(issues_test, desc=f"eval[{name}][Known-K-ITER]"):
                true_set = true_by_issue[iid]
                K = len(true_set)
                pool = tst_rows_by_issue[iid].copy()
                init_issue_state(iid)
                picks = []
                for _ in range(K):
                    ranked = score_pool_local(pool)
                    if len(ranked)==0: break
                    cid = ranked.iloc[0]["Commit ID"]
                    picks.append(cid)
                    if USE_ITERATION_KNOWNK:
                        update_issue_state(iid, cid)
                        pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
                        if len(pool):
                            pool = refresh_features_for_issue_pool(iid, pool)
                    else:
                        pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
                pred_set = set(picks)
                inter = len(pred_set & true_set)
                prec = inter / max(len(pred_set), 1)
                rec  = inter / max(K, 1)
                f1   = (2*inter) / max(len(pred_set)+K, 1)
                rowsK.append(dict(Issue=iid, K=K, Predicted=len(pred_set),
                                  Inter=inter, AllCorrect=int(inter==K),
                                  HalfCorrect=int(inter >= int(np.ceil(K/2))),
                                  Precision=prec, Recall=rec, F1=f1))
            res_knownK = pd.DataFrame(rowsK)
            res_knownK.to_csv(ds_out / "results_KnownK_iter.csv", index=False)


        with stage(f"[{name}] Evaluate No-K (ABS-mm non-iter, REL iterative)"):
            ranked_tst = {iid: score_pool_local(df) for iid, df in tst_rows_by_issue.items()}

            def eval_abs_mm_on_test(tau, tag):
                rows = []; dumps=[]
                for iid in issues_test:
                    ranked = ranked_tst[iid]
                    chosen = apply_cap(ranked[ranked["score_mm"] >= float(tau)].copy())
                    pred = set(chosen["Commit ID"].tolist()); true = true_by_issue[iid]
                    m = metrics_for_issue(pred, true)
                    rows.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true)))
                    tmp = ranked[["Issue ID","Commit ID","score","score_mm","score_zn","Output"]].copy()
                    tmp["Rank"] = np.arange(1, len(tmp)+1); dumps.append(tmp)
                res_df = pd.DataFrame(rows)
                res_df.to_csv(ds_out / f"results_NoK_{tag}.csv", index=False)
                pd.concat(dumps, ignore_index=True).to_csv(ds_out / f"ranked_NoK_{tag}.csv", index=False)
                return res_df

            def eval_rel_on_test_iterative(gamma, tag):
                rows = []; dumps=[]
                for iid in issues_test:
                    init_issue_state(iid)
                    pool_orig = ranked_tst[iid].copy()
                    if len(pool_orig)==0:
                        true = true_by_issue[iid]
                        rows.append(dict(Issue=iid, **metrics_for_issue(set(), true), Predicted=0, TrueCount=len(true)))
                        continue
                    best0 = float(pool_orig["score"].iloc[0])  # fixed anchor
                    accepted=[]; pool = pool_orig.copy()
                    while len(pool):
                        top = pool.iloc[0]
                        if float(top["score"]) >= gamma * best0:
                            cid = top["Commit ID"]
                            accepted.append(cid)
                            if USE_ITERATION_NOK_REL:
                                update_issue_state(iid, cid)
                                pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
                                if len(pool):
                                    pool = refresh_features_for_issue_pool(iid, pool)
                                    pool = score_pool_local(pool)
                            else:
                                pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
                        else:
                            break
                    pred = set(accepted); true = true_by_issue[iid]
                    m = metrics_for_issue(pred, true)
                    rows.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true)))
                    tmp = pool[["Issue ID","Commit ID","score","score_mm","score_zn","Output"]].copy()
                    tmp["Rank"] = np.arange(1, len(tmp)+1); dumps.append(tmp)
                res_df = pd.DataFrame(rows)
                res_df.to_csv(ds_out / f"results_NoK_{tag}_ITER.csv", index=False)
                if dumps:
                    pd.concat(dumps, ignore_index=True).to_csv(ds_out / f"ranked_NoK_{tag}_ITER.csv", index=False)
                return res_df

            tag_abs = f"ABSmm_tau{best_abs_mm[1]:.2f}_{TUNE_OBJECTIVE}"
            tag_rel = f"REL_gamma{best_rel[1]:.2f}_{TUNE_OBJECTIVE}"
            res_absmm = eval_abs_mm_on_test(best_abs_mm[1], tag_abs)
            res_rel   = eval_rel_on_test_iterative(best_rel[1], tag_rel)

        t_test_end = time.perf_counter()
        test_secs = round(t_test_end - t_test_start, 3)

        def macro_percent(df):
            p = round(100.0 * df["Precision"].mean(), 2) if len(df) else 0.0
            r = round(100.0 * df["Recall"].mean(),    2) if len(df) else 0.0
            f = round(100.0 * df["F1"].mean(),        2) if len(df) else 0.0
            return p, r, f

        pK, rK, fK         = macro_percent(res_knownK)
        pABS, rABS, fABS   = macro_percent(res_absmm)
        pREL, rREL, fREL   = macro_percent(res_rel)

        all_summary_rows += [
            [name, "Known-K",        pK,   rK,   fK],
            [name, "No-K (ABS-mm)",  pABS, rABS, fABS],
            [name, "No-K (REL)",     pREL, rREL, fREL],
        ]
        all_timing_rows.append([name, float('nan'), test_secs]) 


    with stage("[SAVE] Rollup CSVs"):
        summary_df = pd.DataFrame(all_summary_rows, columns=["Dataset","Setting","Precision","Recall","F1"])
        timings_df = pd.DataFrame(all_timing_rows, columns=["Dataset","TrainSeconds","TestSeconds"])
        summary_df.to_csv(SUMMARY_CSV, index=False)
        timings_df.to_csv(TIMINGS_CSV, index=False)
        print("\nSummary written to:", SUMMARY_CSV)
        print(summary_df.to_string(index=False))
        print("\nTimings written to:", TIMINGS_CSV)
        print(timings_df.to_string(index=False))

if __name__ == "__main__":
    main()

