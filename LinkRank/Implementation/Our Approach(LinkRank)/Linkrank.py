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


dataset_paths = [
    "Add your file path here",

]
OUT_ROOT     = Path("Add your output path here")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
K_FOLDS = 5  


TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
SVD_DIM      = 256


USE_TIME_FEATURE = True
TIME_TAU_DAYS    = 7


USE_ITERATION_KNOWNK  = True
USE_ITERATION_NOK_REL = True
ALPHA = 0.7   
BETA  = 0.3  


LOG_EVERY_N_ITERS = 10
TUNE_OBJECTIVE = "F1"  


USE_CAP = False
MAX_RELATIVE = 0.75



log_path = OUT_ROOT / "run_cv5.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger("linkrank_no_cb_cv5")

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

def macro_percent(df):
    p = round(100.0 * df["Precision"].mean(), 2) if len(df) else 0.0
    r = round(100.0 * df["Recall"].mean(),    2) if len(df) else 0.0
    f = round(100.0 * df["F1"].mean(),        2) if len(df) else 0.0
    return p, r, f

def run_one_split(ds_name, df, train_ids, test_ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)


    for c in ["Issue Date","Commit Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    with stage(f"[{ds_name}] Build train/test dataframes for provided split"):
        train_df = df[df["Issue ID"].isin(train_ids)].copy()
        test_df  = df[df["Issue ID"].isin(test_ids)].copy()
        log.info(f"[{ds_name}] Train issues={len(train_ids)}, rows={len(train_df):,}")
        log.info(f"[{ds_name}] Test  issues={len(test_ids)}, rows={len(test_df):,}")


    with stage(f"[{ds_name}] Build train text (issue/commit)"):
        issue_txt_tr = issue_text(train_df)
        commit_txt_tr = commit_text(train_df)

    with stage(f"[{ds_name}] TF-IDF fit_transform (train only)"):
        tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
        X_tr = tfidf.fit_transform(pd.concat([issue_txt_tr["text"], commit_txt_tr["text"]], axis=0).fillna(""))
        log.info(f"[{ds_name}] TF-IDF: shape={X_tr.shape}, vocab={len(tfidf.vocabulary_):,}")

    with stage(f"[{ds_name}] SVD fit_transform (train only)"):
        if USE_CUML_SVD:
            svd = cuML_TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr)
        else:
            svd = SkTruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr)
        E_issue_tr  = Xr_tr[:len(issue_txt_tr)]
        E_commit_tr = Xr_tr[len(issue_txt_tr):]
        log.info(f"[{ds_name}] SVD: comp={SVD_DIM} | issue_emb={E_issue_tr.shape} | commit_emb={E_commit_tr.shape}")

    with stage(f"[{ds_name}] Build test text and transform"):
        issue_txt_te = issue_text(test_df)
        commit_txt_te = commit_text(test_df)
        X_te_issue  = svd.transform(tfidf.transform(issue_txt_te["text"].fillna("")))
        X_te_commit = svd.transform(tfidf.transform(commit_txt_te["text"].fillna("")))
        log.info(f"[{ds_name}] X_te_issue={X_te_issue.shape} | X_te_commit={X_te_commit.shape}")


    issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr["Issue ID"].tolist())}
    commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr["Commit ID"].tolist())}
    issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te["Issue ID"].tolist())}
    commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te["Commit ID"].tolist())}


    def build_features(d, split="train"):
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


            f_time = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS) if USE_TIME_FEATURE else 0.0
            rows.append([iid, cid, f_text, f_time, row["Output"]])

        return pd.DataFrame(rows, columns=["Issue ID","Commit ID","feat_text","feat_time","Output"])

    with stage(f"[{ds_name}] Build features: TRAIN"):
        train_feat = build_features(train_df, split="train")

    with stage(f"[{ds_name}] Build features: TEST"):
        test_feat  = build_features(test_df,  split="test")


    FEATURES = ["feat_text","feat_time"]

    def prep_rank(df_feat):
        df_s = df_feat.sort_values(["Issue ID"]).reset_index(drop=True)
        X = df_s[FEATURES].values
        y = df_s["Output"].astype(int).values
        groups = df_s.groupby("Issue ID").size().tolist()
        return df_s, X, y, groups

    with stage(f"[{ds_name}] Prepare rank datasets (train/dev/test)"):
        uniq_train_issues = train_feat["Issue ID"].drop_duplicates().tolist()
        rng.shuffle(uniq_train_issues)
        n_dev = max(1, int(0.20*len(uniq_train_issues)))
        dev_ids = set(uniq_train_issues[:n_dev])

        trn_feat = train_feat[~train_feat["Issue ID"].isin(dev_ids)].copy()
        dev_feat = train_feat[ train_feat["Issue ID"].isin(dev_ids)].copy()

        def _filter_groups_with_labels(df_feat, require_neg=True):
            grp = df_feat.groupby("Issue ID")["Output"]
            pos = grp.sum()
            cnt = grp.count()
            ok = (pos >= 1) & ((cnt - pos) >= 1) if require_neg else (pos >= 1)
            keep = set(pos[ok].index.tolist())
            return df_feat[df_feat["Issue ID"].isin(keep)].copy()

        trn_feat = _filter_groups_with_labels(trn_feat, require_neg=True)
        dev_feat = _filter_groups_with_labels(dev_feat, require_neg=True)
        test_feat = _filter_groups_with_labels(test_feat, require_neg=False)

        trn_s, Xtr, ytr, gtr = prep_rank(trn_feat)
        dev_s, Xdv, ydv, gdv = prep_rank(dev_feat)
        tst_s, Xte, yte, gte = prep_rank(test_feat)


    with stage(f"[{ds_name}] Train LambdaMART"):
        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEATURES)
        dvalid = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain, feature_name=FEATURES)

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
            log.info(f"[{ds_name}] LightGBM: GPU enabled.")
        else:
            log.info(f"[{ds_name}] LightGBM: using CPU.")

        model = lgb.train(
            params, dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=2000,
            callbacks=[
                lgb.log_evaluation(LOG_EVERY_N_ITERS),
                lgb.early_stopping(stopping_rounds=150, verbose=True)
            ]
        )
        log.info(f"[{ds_name}] Best iteration: {model.best_iteration}")

    def score_pool(pool_df):
        if len(pool_df)==0:
            return pool_df.assign(score=[])
        X = pool_df[FEATURES].values
        s = model.predict(X, num_iteration=model.best_iteration)
        out = pool_df.copy()
        out["score"] = s
        if len(out) <= 1:
            out["score_mm"] = 1.0
            out["score_zn"] = 0.0
        else:
            mn, mx = float(out["score"].min()), float(out["score"].max())
            out["score_mm"] = 1.0 if mx==mn else (out["score"] - mn) / (mx - mn)
            mu, sd = float(out["score"].mean()), float(out["score"].std(ddof=0) + 1e-9)
            out["score_zn"] = (out["score"] - mu) / sd
        return out.sort_values("score", ascending=False).reset_index(drop=True)


    issue_vec_state = {}
    def init_issue_state(iid):
        v = X_te_issue[ issue_idx_te[iid] ].copy()
        issue_vec_state[iid] = v / (np.linalg.norm(v) + 1e-12)

    def update_issue_state(iid, cid):
        v_issue = issue_vec_state[iid]
        v_commit = X_te_commit[ commit_idx_te[cid] ]
        v_new = ALPHA * v_issue + BETA * v_commit
        issue_vec_state[iid] = v_new / (np.linalg.norm(v_new) + 1e-12)

    def refresh_features_for_issue_pool(iid, pool_df):
        if len(pool_df)==0: return pool_df
        v_issue = issue_vec_state[iid]
        new_rows = []
        for _, r in pool_df.iterrows():
            cid = r["Commit ID"]
            v_commit = X_te_commit[ commit_idx_te[cid] ]
            f_text = cos_sim(v_issue, v_commit)
            new_rows.append([r["Issue ID"], r["Commit ID"], f_text, r["feat_time"], r["Output"]])
        return pd.DataFrame(new_rows, columns=["Issue ID","Commit ID","feat_text","feat_time","Output"])


    with stage(f"[{ds_name}] Evaluate on test (Known-K, iterative)"):
        test_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                              for iid in tst_s["Issue ID"].drop_duplicates().tolist()}
        true_by_issue = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_test = sorted(true_by_issue.keys())

        rowsK = []
        for iid in tqdm(issues_test, desc=f"eval[{ds_name}][Known-K-ITER]"):
            true_set = true_by_issue[iid]
            K = len(true_set)
            pool = test_rows_by_issue[iid].copy()
            init_issue_state(iid)
            picks = []
            for _ in range(K):
                ranked = score_pool(pool)
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
            rowsK.append(dict(
                Issue=iid, K=K, Predicted=len(pred_set),
                Inter=inter,
                AllCorrect=int(inter==K),
                HalfCorrect=int(inter >= int(np.ceil(K/2))),
                Precision=prec, Recall=rec, F1=f1
            ))
        res_knownK = pd.DataFrame(rowsK)
        res_knownK.to_csv(out_dir / "approach1_results_per_issue_KnownK_iter.csv", index=False)

    with stage(f"[{ds_name}] No-K: tune thresholds on dev and evaluate on test"):
        trn_s_full = trn_s  
        dev_rows_by_issue = {iid: trn_s_full[trn_s_full["Issue ID"]==iid].reset_index(drop=True)
                             for iid in trn_s_full["Issue ID"].drop_duplicates().tolist()}
        tst_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                             for iid in tst_s["Issue ID"].drop_duplicates().tolist()}
        true_by_issue_dev = trn_s_full[trn_s_full["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        true_by_issue_tst = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
        issues_dev = sorted(true_by_issue_dev.keys())
        issues_tst = sorted(true_by_issue_tst.keys())

        def metrics_for_issue(pred_set, true_set):
            inter = len(pred_set & true_set)
            p = inter / max(len(pred_set), 1)
            r = inter / max(len(true_set), 1)
            f1 = (2*inter) / max(len(pred_set)+len(true_set), 1)
            allc = int(pred_set == true_set)
            half = int(inter >= int(np.ceil(len(true_set)/2)))
            return dict(Precision=p, Recall=r, F1=f1, AllCorrect=allc, HalfCorrect=half)

        def aggregate(dfm):
            obj = TUNE_OBJECTIVE.lower()
            if obj == "allcorrect": return dfm["AllCorrect"].mean()
            return dfm["F1"].mean()

        def rank_full(pool): return score_pool(pool)

        ranked_dev = {iid: rank_full(df) for iid, df in dev_rows_by_issue.items()}
        ranked_tst = {iid: rank_full(df) for iid, df in tst_rows_by_issue.items()}

        def apply_cap(chosen_df):
            if not USE_CAP or len(chosen_df) == 0:
                return chosen_df
            top = float(chosen_df["score"].iloc[0])
            return chosen_df[chosen_df["score"] >= MAX_RELATIVE * top]


        taus_mm = [x/100 for x in range(10, 96, 2)]
        best_abs_mm = (-1, None)
        for t in taus_mm:
            rows = []
            for iid in issues_dev:
                ranked = ranked_dev[iid]
                chosen = ranked[ranked["score_mm"] >= float(t)].copy()
                chosen = apply_cap(chosen)
                pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_dev[iid]
                rows.append(metrics_for_issue(pred, true))
            sc = aggregate(pd.DataFrame(rows))
            if sc > best_abs_mm[0]: best_abs_mm = (sc, t)

        gammas  = [x/100 for x in range(30, 96, 2)]
        best_rel = (-1, None)
        for g in gammas:
            rows = []
            for iid in issues_dev:
                ranked = ranked_dev[iid]
                if len(ranked)==0:
                    rows.append(metrics_for_issue(set(), true_by_issue_dev[iid])); continue
                best = float(ranked["score"].iloc[0])
                chosen = ranked[ranked["score"] >= float(g)*best].copy()
                chosen = apply_cap(chosen)
                pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_dev[iid]
                rows.append(metrics_for_issue(pred, true))
            sc = aggregate(pd.DataFrame(rows))
            if sc > best_rel[0]: best_rel = (sc, g)

        def eval_abs_mm_on_test(tau, tag):
            rows = []; dumps = []
            for iid in issues_tst:
                ranked = ranked_tst[iid]
                chosen = ranked[ranked["score_mm"] >= float(tau)].copy()
                chosen = apply_cap(chosen)
                pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_tst[iid]
                m = metrics_for_issue(pred, true)
                rows.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true)))
                tmp = ranked[["Issue ID","Commit ID","score","score_mm","score_zn","Output"]].copy()
                tmp["Rank"] = np.arange(1, len(tmp)+1); dumps.append(tmp)
            res_df = pd.DataFrame(rows)
            res_df.to_csv(out_dir / f"approach1_results_per_issue_NoK_{tag}.csv", index=False)
            pd.concat(dumps, ignore_index=True).to_csv(out_dir / f"approach1_ranked_commits_NoK_{tag}.csv", index=False)
            return res_df

        def eval_rel_on_test_iterative(gamma, tag):
            rows = []; dumps = []
            for iid in issues_tst:
                init_issue_state(iid)
                pool_orig = ranked_tst[iid].copy()
                if len(pool_orig) == 0:
                    true = true_by_issue_tst[iid]
                    m = metrics_for_issue(set(), true)
                    rows.append(dict(Issue=iid, **m, Predicted=0, TrueCount=len(true)))
                    continue

                best0 = float(pool_orig["score"].iloc[0])   # FIXED anchor
                accepted = []
                pool = pool_orig.copy()

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
                                pool = score_pool(pool)
                        else:
                            pool = pool[pool["Commit ID"] != cid].reset_index(drop=True)
                    else:
                        break

                pred = set(accepted)
                true = true_by_issue_tst[iid]
                m = metrics_for_issue(pred, true)
                rows.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true)))

                tmp = pool[["Issue ID","Commit ID","score","score_mm","score_zn","Output"]].copy()
                tmp["Rank"] = np.arange(1, len(tmp)+1); dumps.append(tmp)

            res_df = pd.DataFrame(rows)
            res_df.to_csv(out_dir / f"approach1_results_per_issue_NoK_{tag}_ITER.csv", index=False)
            if dumps:
                pd.concat(dumps, ignore_index=True).to_csv(out_dir / f"approach1_ranked_commits_NoK_{tag}_ITER.csv", index=False)
            return res_df

        res_absmm = eval_abs_mm_on_test(best_abs_mm[1], f"ABSmm_tau{best_abs_mm[1]:.2f}_{TUNE_OBJECTIVE}")
        res_rel   = eval_rel_on_test_iterative(best_rel[1], f"REL_gamma{best_rel[1]:.2f}_{TUNE_OBJECTIVE}")


    with stage(f"[{ds_name}] Save split artifacts (rows)"):
        (out_dir / "artifacts").mkdir(exist_ok=True, parents=True)


    pK, rK, fK         = macro_percent(res_knownK)
    pABS, rABS, fABS   = macro_percent(res_absmm)
    pREL, rREL, fREL   = macro_percent(res_rel)

    summary_rows = [
        ["Known-K",        pK,   rK,   fK],
        ["No-K (ABS-mm)",  pABS, rABS, fABS],
        ["No-K (REL)",     pREL, rREL, fREL],
    ]
    sum_df = pd.DataFrame(summary_rows, columns=["Setting","Precision","Recall","F1"])
    return sum_df


if __name__ == "__main__":
    for path in dataset_paths:
        ds_name = Path(path).stem
        print(f"Running 5-fold CV for dataset: {ds_name}")


        with stage(f"[{ds_name}] Load CSV"):
            df = pd.read_csv(path)
            for c in ["Issue Date","Commit Date"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            issue_ids = df["Issue ID"].drop_duplicates().tolist()
            issue_ids_sorted = sorted(issue_ids) 


        DS_OUT = OUT_ROOT / ds_name / "cv5"
        DS_OUT.mkdir(parents=True, exist_ok=True)

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        fold_summaries = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(issue_ids_sorted), start=1):
            fold_tag = f"fold{fold_idx}"
            fold_dir = DS_OUT / fold_tag
            train_ids = set([issue_ids_sorted[i] for i in train_idx])
            test_ids  = set([issue_ids_sorted[i] for i in test_idx])

            log.info(f"[{ds_name}] ===== CV {fold_tag}: train_issues={len(train_ids)}, test_issues={len(test_ids)} =====")
            sum_df = run_one_split(ds_name, df, train_ids, test_ids, fold_dir)

            sum_path = DS_OUT / f"summary_{fold_tag}.csv"
            sum_df.to_csv(sum_path, index=False)
            fold_summaries.append(sum_df.assign(Fold=fold_idx))

        all_summ = pd.concat(fold_summaries, ignore_index=True)
        mean_df = (all_summ
                   .groupby("Setting", as_index=False)[["Precision","Recall","F1"]]
                   .mean()
                   .round(2))
        mean_df.to_csv(DS_OUT / "summary_k5_mean.csv", index=False)

        print(f"[DONE] {ds_name}: wrote 5 per-fold summaries + summary_k5_mean.csv in {DS_OUT}")
