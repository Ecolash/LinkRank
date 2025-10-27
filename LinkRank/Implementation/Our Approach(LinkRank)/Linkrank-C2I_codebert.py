import os, time, math, logging, re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    TORCH_HAS_CUDA = torch.cuda.is_available()
    TORCH_DEVICE = torch.device("cuda" if TORCH_HAS_CUDA else "cpu")
except Exception:
    TORCH_HAS_CUDA = False
    class _D:
        def __str__(self): return "cpu (torch not available)"
    TORCH_DEVICE = _D()

USE_CUML_SVD = False
try:
    from cuml.decomposition import TruncatedSVD as cuML_TruncatedSVD
    USE_CUML_SVD = True
except Exception:
    from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD
    USE_CUML_SVD = False


HAVE_BM25 = False
try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False


dataset_paths = [
    "Add your file path here",

]

OUT_ROOT     = Path("Add your output path here")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
K_FOLDS = 5
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
SVD_DIM = 256
CODEBERT_MODEL = "microsoft/codebert-base"
CB_MAX_LEN = 256
CB_BATCH   = 32  
USE_TIME_FEATURE = True
TIME_TAU_DAYS    = 7.0
TOPK_ISSUES_PER_COMMIT = 6  
BM25_M_ISSUES = 4            
BM25_M_COMMITS = 6           


TUNE_OBJECTIVE = "F1"

LOG_PATH = OUT_ROOT / "run_codebert_c2i_cv5.log"



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger("codebert_c2i_cv5")

def stage(name):
    class _Stage:
        def __enter__(self):
            self.name = name
            self.t0 = time.perf_counter()
            log.info(f"▶ START: {self.name}")
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            if exc_type is None:
                log.info(f"✔ END:   {self.name} | {dt:.2f}s")
            else:
                log.error(f"✖ FAIL:  {self.name} | {dt:.2f}s | {exc_type.__name__}: {exc}")
            return False
    return _Stage()


C_REPO = "Repository"
C_IID  = "Issue ID"
C_IDT  = "Issue Date"
C_ITTL = "Title"
C_IDSC = "Description"
C_ILBL = "Labels"
C_ICMT = "Comments"
C_CID  = "Commit ID"
C_CDT  = "Commit Date"
C_CMSG = "Message"
C_CSUM = "Diff Summary"
C_CFLS = "File Changes"
C_CFUL = "Full Diff"
C_Y    = "Output"

REQUIRED_COLS = [C_REPO,C_IID,C_IDT,C_ITTL,C_IDSC,C_ILBL,C_ICMT,
                 C_CID,C_CDT,C_CMSG,C_CSUM,C_CFLS,C_CFUL,C_Y]


def cos_sim(a, b):
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def time_prox(issue_date, commit_date, tau_days=14.0):
    if pd.isna(issue_date) or pd.isna(commit_date): return 0.0
    days = abs((commit_date - issue_date).days)
    return float(np.exp(-days / tau_days))

def simple_tokenize(s): 
    return re.findall(r"[A-Za-z0-9_]+", str(s).lower())

def bm25_scores(query_text, doc_texts):
    if not HAVE_BM25 or len(doc_texts)==0:
        return np.zeros(len(doc_texts), dtype=float)
    bm = BM25Okapi([simple_tokenize(t or "") for t in doc_texts])
    return bm.get_scores(simple_tokenize(query_text or ""))

def percent2(x):
    return float(np.round(100.0 * float(x), 2))

def split_by_group_robust(df, group_col, dev_frac=0.20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    groups = df[group_col].drop_duplicates().tolist()
    if len(groups) <= 1:
        return df.copy(), df.iloc[0:0].copy()
    rng.shuffle(groups)
    n = len(groups)
    n_dev = max(1, int(round(dev_frac * n)))
    n_dev = min(n_dev, n - 1)
    dev_ids = set(groups[:n_dev])
    trn = df[~df[group_col].isin(dev_ids)].copy()
    dev = df[df[group_col].isin(dev_ids)].copy()
    if trn.empty:
        trn, dev = dev, trn
    return trn, dev

def ensure_nonempty_train_valid(Xtr, ytr, gtr, Xdv, ydv, gdv):
    if len(Xdv)==0 or len(gdv)==0:
        cut = max(1, min(100, len(ytr)//10)) if len(ytr) else 1
        return Xtr[cut:], ytr[cut:], gtr, Xtr[:cut], ytr[:cut], gtr
    return Xtr, ytr, gtr, Xdv, ydv, gdv

def train_lambdamart(tag, feat_names, Xtr, ytr, gtr, Xdv, ydv, gdv):
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feat_names)
    dvalid = lgb.Dataset(Xdv, label=ydv, group=gdv, reference=dtrain, feature_name=feat_names)
    params = dict(
        objective="lambdarank",
        metric=["ndcg"],
        ndcg_eval_at=[1,3,5],
        learning_rate=0.06,
        num_leaves=47,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        num_threads=os.cpu_count() or 1,
    )
    if TORCH_HAS_CUDA:
        params.update(dict(device_type="gpu", gpu_platform_id=0, gpu_device_id=0))
        log.info(f"[{tag}] LightGBM: GPU enabled")
    else:
        log.info(f"[{tag}] LightGBM: CPU")
    model = lgb.train(
        params, dtrain, valid_sets=[dtrain, dvalid],
        num_boost_round=1200,
        callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)]
    )
    log.info(f"[{tag}] Best iteration: {model.best_iteration}")
    return model

# ---------- Text builders ----------
def issue_text(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(C_IID).agg(
        title=(C_ITTL,"first"),
        desc =(C_IDSC,"first"),
        comm =(C_ICMT,"first"),
    ).fillna("")
    txt = (g["title"] + " " + g["desc"] + " " + g["comm"])
    out = txt.reset_index()
    out.columns = [C_IID, "text"]
    return out

def commit_text(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(C_CID).agg(
        msg =(C_CMSG,"first"),
        dif =(C_CSUM,"first"),
        files=(C_CFLS,"first"),
        full=(C_CFUL,"first"),
    ).fillna("")
    txt = (g["msg"] + " " + g["dif"] + " " + g["files"] + " " + g["full"])
    out = txt.reset_index()
    out.columns = [C_CID, "text"]
    return out

def issue_fields(d: pd.DataFrame):
    g = d.groupby(C_IID).agg(
        title=(C_ITTL,"first"),
        desc =(C_IDSC,"first"),
        comm =(C_ICMT,"first"),
    ).fillna("")
    return pd.DataFrame({
        C_IID: g.index,
        "i_td": (g["title"] + " " + g["desc"]).values,
        "i_comm": g["comm"].values
    })

def commit_fields(d: pd.DataFrame):
    g = d.groupby(C_CID).agg(
        msg =(C_CMSG,"first"),
        dif =(C_CSUM,"first"),
        files=(C_CFLS,"first"),
        full=(C_CFUL,"first"),
    ).fillna("")
    return pd.DataFrame({
        C_CID: g.index,
        "c_msg":  g["msg"].values,
        "c_df":   (g["dif"] + " " + g["files"]).values,
        "c_full": g["full"].values
    })


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def load_sem_model():

    model_dtype = torch.float16 if TORCH_HAS_CUDA else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model_sem = AutoModel.from_pretrained(
        CODEBERT_MODEL,
        dtype=model_dtype,
        low_cpu_mem_usage=True
    ).to(TORCH_DEVICE)
    model_sem.eval()
    return tokenizer, model_sem

@torch.inference_mode()
def encode_texts(tokenizer, model_sem, ids, texts, batch=CB_BATCH):
    vecs = {}
    for i in tqdm(range(0, len(texts), batch), desc="encode_sem", leave=False):
        chunk_ids = ids[i:i+batch]
        chunk_txt = texts[i:i+batch]
        enc = tokenizer(chunk_txt, padding=True, truncation=True, max_length=CB_MAX_LEN, return_tensors="pt").to(TORCH_DEVICE)
        out = model_sem(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        arr = emb.detach().cpu().numpy()
        for j, _id in enumerate(chunk_ids):
            vecs[_id] = arr[j]
        if TORCH_HAS_CUDA:
            torch.cuda.empty_cache()
    return vecs

def cos_from_maps(a_map, a_id, b_map, b_id):
    va = a_map.get(a_id); vb = b_map.get(b_id)
    if va is None or vb is None: return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-12))


FEATURES = [
    "feat_text","feat_time",
    "feat_sem","feat_sem_td_msg","feat_sem_td_df","feat_sem_comm_full"
]

def build_features(d: pd.DataFrame,
                   VI, VC, i_idx, c_idx,
                   issue_cb, commit_cb, E_i_td, E_i_comm, E_c_msg, E_c_df, E_c_full):
    rows = []
    it = tqdm(d.iterrows(), total=len(d), desc="build_features", leave=False)
    for _, row in it:
        iid = row[C_IID]; cid = row[C_CID]

        if (iid in i_idx) and (cid in c_idx):
            f_text = cos_sim(VI[i_idx[iid]], VC[c_idx[cid]])
        else:
            f_text = 0.0

        f_time = time_prox(row.get(C_IDT), row.get(C_CDT), TIME_TAU_DAYS) if USE_TIME_FEATURE else 0.0

        f_sem_global    = cos_from_maps(issue_cb, iid, commit_cb, cid)
        f_sem_td_msg    = cos_from_maps(E_i_td,   iid, E_c_msg,  cid)
        f_sem_td_df     = cos_from_maps(E_i_td,   iid, E_c_df,   cid)
        f_sem_comm_full = cos_from_maps(E_i_comm, iid, E_c_full, cid)
        rows.append([iid, cid, f_text, f_time,
                     f_sem_global, f_sem_td_msg, f_sem_td_df, f_sem_comm_full,
                     int(row[C_Y])])
    return pd.DataFrame(rows, columns=[
        C_IID, C_CID,
        "feat_text","feat_time",
        "feat_sem","feat_sem_td_msg","feat_sem_td_df","feat_sem_comm_full",
        C_Y
    ])

def score_pool_with_model(model, pool_df):
    if pool_df.empty:
        return pool_df.assign(score=[], score_mm=[], score_zn=[])
    X = pool_df[FEATURES].values.astype(np.float32)
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

def aggregate_dev(dfm):
    if (TUNE_OBJECTIVE or "F1").lower() == "allcorrect":
        return dfm["AllCorrect"].mean()
    return dfm["F1"].mean()

def restrict_pool_by_A2_and_BM25(iid, df_issue_pool, shortlist_map, issue_text_map, commit_text_map):
    if df_issue_pool.empty: return df_issue_pool

    mask = df_issue_pool[C_CID].map(lambda cid: iid in shortlist_map.get(cid, set()))
    subset = df_issue_pool[mask].copy()

    if HAVE_BM25 and BM25_M_COMMITS > 0:
        commits = df_issue_pool[C_CID].tolist()
        docs = [commit_text_map.get(c, "") for c in commits]
        q = issue_text_map.get(iid, "")
        scores = bm25_scores(q, docs)
        order = np.argsort(-scores)[:BM25_M_COMMITS]
        keep = set([commits[i] for i in order])
        subset = pd.concat([subset, df_issue_pool[df_issue_pool[C_CID].isin(keep)]],
                           ignore_index=True).drop_duplicates([C_IID, C_CID])
    if subset.empty:
        return df_issue_pool
    return subset

def topk_issues_per_commit(df_split, model_A2, issue_text_map, commit_text_map):
    res = {}
    by_commit = {cid: df_split[df_split[C_CID]==cid].reset_index(drop=True)
                 for cid in df_split[C_CID].drop_duplicates().tolist()}
    for cid, pool in by_commit.items():
        ranked = score_pool_with_model(model_A2, pool)
        top_ranked = set(ranked[C_IID].iloc[:TOPK_ISSUES_PER_COMMIT].tolist())
        if HAVE_BM25 and BM25_M_ISSUES > 0:
            issues = pool[C_IID].tolist()
            docs = [issue_text_map.get(i, "") for i in issues]
            q = commit_text_map.get(cid, "")
            scores = bm25_scores(q, docs)
            order = np.argsort(-scores)[:BM25_M_ISSUES]
            bm25_top = {issues[i] for i in order}
        else:
            bm25_top = set()
        res[cid] = top_ranked | bm25_top
    return res


def run_one_fold(ds_name, full_df, train_ids, test_ids, fold_dir, tokenizer, model_sem):
    fold_dir.mkdir(parents=True, exist_ok=True)


    train_df = full_df[full_df[C_IID].isin(train_ids)].copy()
    test_df  = full_df[full_df[C_IID].isin(test_ids)].copy()


    issue_txt_tr = issue_text(train_df); commit_txt_tr = commit_text(train_df)
    tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
    X_tr = tfidf.fit_transform(pd.concat([issue_txt_tr["text"], commit_txt_tr["text"]], axis=0).fillna(""))

    if USE_CUML_SVD:
        with stage(f"[{ds_name}] cuML TruncatedSVD (GPU)"):
            svd = cuML_TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr) 
    else:
        with stage(f"[{ds_name}] sklearn TruncatedSVD (CPU)"):
            svd = SkTruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
            Xr_tr = svd.fit_transform(X_tr)

    E_issue_tr  = Xr_tr[:len(issue_txt_tr)]
    E_commit_tr = Xr_tr[len(issue_txt_tr):]

    issue_txt_te = issue_text(test_df); commit_txt_te = commit_text(test_df)
    X_te_issue  = svd.transform(tfidf.transform(issue_txt_te["text"].fillna("")))
    X_te_commit = svd.transform(tfidf.transform(commit_txt_te["text"].fillna("")))


    issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr[C_IID].tolist())}
    commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr[C_CID].tolist())}
    issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te[C_IID].tolist())}
    commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te[C_CID].tolist())}

    issue_text_map  = pd.concat([issue_txt_tr,  issue_txt_te]).drop_duplicates(C_IID).set_index(C_IID)["text"].to_dict()
    commit_text_map = pd.concat([commit_txt_tr, commit_txt_te]).drop_duplicates(C_CID).set_index(C_CID)["text"].to_dict()

    with stage(f"[{ds_name}] Encode CodeBERT (GPU)"):
        E_issue_sem_tr  = encode_texts(tokenizer, model_sem, issue_txt_tr[C_IID].tolist(),  issue_txt_tr["text"].fillna("").tolist())
        E_commit_sem_tr = encode_texts(tokenizer, model_sem, commit_txt_tr[C_CID].tolist(), commit_txt_tr["text"].fillna("").tolist())
        E_issue_sem_te  = encode_texts(tokenizer, model_sem, issue_txt_te[C_IID].tolist(),  issue_txt_te["text"].fillna("").tolist())
        E_commit_sem_te = encode_texts(tokenizer, model_sem, commit_txt_te[C_CID].tolist(), commit_txt_te["text"].fillna("").tolist())

        i_fields_tr = issue_fields(train_df); i_fields_te = issue_fields(test_df)
        c_fields_tr = commit_fields(train_df); c_fields_te = commit_fields(test_df)
        i_fields_all = pd.concat([i_fields_tr, i_fields_te], ignore_index=True).drop_duplicates(C_IID)
        c_fields_all = pd.concat([c_fields_tr, c_fields_te], ignore_index=True).drop_duplicates(C_CID)

        E_i_td    = encode_texts(tokenizer, model_sem, i_fields_all[C_IID].tolist(), i_fields_all["i_td"].tolist())
        E_i_comm  = encode_texts(tokenizer, model_sem, i_fields_all[C_IID].tolist(), i_fields_all["i_comm"].tolist())
        E_c_msg   = encode_texts(tokenizer, model_sem, c_fields_all[C_CID].tolist(), c_fields_all["c_msg"].tolist())
        E_c_df    = encode_texts(tokenizer, model_sem, c_fields_all[C_CID].tolist(), c_fields_all["c_df"].tolist())
        E_c_full  = encode_texts(tokenizer, model_sem, c_fields_all[C_CID].tolist(), c_fields_all["c_full"].tolist())


    train_feat = build_features(
        train_df, E_issue_tr, E_commit_tr, issue_idx_tr, commit_idx_tr,
        E_issue_sem_tr, E_commit_sem_tr, E_i_td, {}, E_c_msg, E_c_df, E_c_full
    )
    test_feat  = build_features(
        test_df,  X_te_issue,  X_te_commit,  issue_idx_te,  commit_idx_te,
        E_issue_sem_te, E_commit_sem_te, E_i_td, {}, E_c_msg, E_c_df, E_c_full
    )

    trn_A2, dev_A2 = split_by_group_robust(train_feat, group_col=C_CID, dev_frac=0.20, rng=rng)
    def prep_rank_commit(df_feat):
        df_s = df_feat.sort_values([C_CID]).reset_index(drop=True)
        X = df_s[FEATURES].values.astype(np.float32)
        y = df_s[C_Y].astype(int).values
        groups = df_s.groupby(C_CID).size().tolist()
        return df_s, X, y, groups
    trn_s_A2, Xtr_A2, ytr_A2, gtr_A2 = prep_rank_commit(trn_A2)
    dev_s_A2, Xdv_A2, ydv_A2, gdv_A2 = prep_rank_commit(dev_A2)
    Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2 = ensure_nonempty_train_valid(Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2)
    model_A2 = train_lambdamart(f"{ds_name} A2", FEATURES, Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2)


    test_shortlist = topk_issues_per_commit(test_feat, model_A2, issue_text_map, commit_text_map)


    def prep_rank_issue(df_feat):
        df_s = df_feat.sort_values([C_IID]).reset_index(drop=True)
        X = df_s[FEATURES].values.astype(np.float32)
        y = df_s[C_Y].astype(int).values
        groups = df_s.groupby(C_IID).size().tolist()
        return df_s, X, y, groups

    trn_A1, dev_A1 = split_by_group_robust(train_feat, group_col=C_IID, dev_frac=0.20, rng=rng)
    trn_s_A1, Xtr_A1, ytr_A1, gtr_A1 = prep_rank_issue(trn_A1)
    dev_s_A1, Xdv_A1, ydv_A1, gdv_A1 = prep_rank_issue(dev_A1)
    tst_s_A1, Xte_A1, yte_A1, gte_A1 = prep_rank_issue(test_feat)
    Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1 = ensure_nonempty_train_valid(Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1)
    model_A1 = train_lambdamart(f"{ds_name} A1", FEATURES, Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1)


    dev_rows_by_issue = {iid: dev_s_A1[dev_s_A1[C_IID]==iid].reset_index(drop=True)
                         for iid in dev_s_A1[C_IID].drop_duplicates().tolist()}
    true_by_issue_dev = dev_s_A1[dev_s_A1[C_Y]==1].groupby(C_IID)[C_CID].apply(set).to_dict()
    issues_dev = sorted(true_by_issue_dev.keys())
    dev_shortlist_like = topk_issues_per_commit(dev_A1, model_A2, issue_text_map, commit_text_map)

    def eval_abs_mm_on_dev(tau):
        rows=[]
        for iid in issues_dev:
            pool_full = dev_rows_by_issue[iid]
            pool = restrict_pool_by_A2_and_BM25(iid, pool_full, dev_shortlist_like, issue_text_map, commit_text_map)
            ranked = score_pool_with_model(model_A1, pool)
            chosen = ranked[ranked["score_mm"] >= float(tau)][C_CID].tolist()
            rows.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
        dfm = pd.DataFrame(rows); return aggregate_dev(dfm)

    def eval_rel_on_dev(gamma):
        rows=[]
        for iid in issues_dev:
            pool_full = dev_rows_by_issue[iid]
            pool = restrict_pool_by_A2_and_BM25(iid, pool_full, dev_shortlist_like, issue_text_map, commit_text_map)
            ranked = score_pool_with_model(model_A1, pool)
            if len(ranked)==0:
                rows.append(metrics_for_issue(set(), true_by_issue_dev[iid])); continue
            best = float(ranked["score"].iloc[0])
            chosen = ranked[ranked["score"] >= float(gamma) * best][C_CID].tolist()
            rows.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
        dfm = pd.DataFrame(rows); return aggregate_dev(dfm)

    taus   = [x/100 for x in range(20, 96, 2)]
    gammas = [x/100 for x in range(30, 96, 2)]
    best_abs = (-1, None); best_rel = (-1, None)
    for t in taus:
        sc = eval_abs_mm_on_dev(t)
        if sc > best_abs[0]: best_abs = (sc, t)
    for g in gammas:
        sc = eval_rel_on_dev(g)
        if sc > best_rel[0]: best_rel = (sc, g)
    log.info(f"[{ds_name}] DEV best ABS-mm τ={best_abs[1]:.2f}")
    log.info(f"[{ds_name}] DEV best REL   γ={best_rel[1]:.2f}")


    tst_rows_by_issue = {iid: tst_s_A1[tst_s_A1[C_IID]==iid].reset_index(drop=True)
                         for iid in tst_s_A1[C_IID].drop_duplicates().tolist()}
    true_by_issue_test = tst_s_A1[tst_s_A1[C_Y]==1].groupby(C_IID)[C_CID].apply(set).to_dict()
    issues_test = sorted(true_by_issue_test.keys())

    def eval_policy_on_test(kind, val, tag):
        rows=[]
        for iid in issues_test:
            pool_full = tst_rows_by_issue[iid]
            pool = restrict_pool_by_A2_and_BM25(iid, pool_full, test_shortlist, issue_text_map, commit_text_map)
            ranked = score_pool_with_model(model_A1, pool)
            if len(ranked)==0:
                pred=set()
            else:
                if kind=="ABS":
                    chosen = ranked[ranked["score_mm"] >= float(val)][C_CID].tolist()
                else:
                    best = float(ranked["score"].iloc[0])
                    chosen = ranked[ranked["score"] >= float(val)*best][C_CID].tolist()
                pred=set(chosen)
            true = true_by_issue_test[iid]
            m = metrics_for_issue(pred, true)
            rows.append(m)
        return pd.DataFrame(rows)

    def eval_test_oracleK():
        rows=[]
        for iid in issues_test:
            pool_full = tst_rows_by_issue[iid]
            pool = restrict_pool_by_A2_and_BM25(iid, pool_full, test_shortlist, issue_text_map, commit_text_map)
            ranked = score_pool_with_model(model_A1, pool)
            true = true_by_issue_test[iid]
            K = len(true)
            chosen = ranked[C_CID].iloc[:K].tolist() if K>0 else []
            pred=set(chosen)
            rows.append(metrics_for_issue(pred, true))
        return pd.DataFrame(rows)

    tag_abs = f"ABSmm_tau{best_abs[1]:.2f}_{TUNE_OBJECTIVE}"
    tag_rel = f"REL_gamma{best_rel[1]:.2f}_{TUNE_OBJECTIVE}"
    res_abs = eval_policy_on_test("ABS", best_abs[1], tag_abs)
    res_rel = eval_policy_on_test("REL", best_rel[1], tag_rel)
    res_ok  = eval_test_oracleK()


    def macro_pct(dfm):
        return (
            percent2(dfm["Precision"].mean()),
            percent2(dfm["Recall"].mean()),
            percent2(dfm["F1"].mean())
        )

    pABS, rABS, fABS = macro_pct(res_abs)
    pREL, rREL, fREL = macro_pct(res_rel)
    pOK,  rOK,  fOK  = macro_pct(res_ok)

    fold_sum = pd.DataFrame(
        [["No-K (ABS-mm)", pABS, rABS, fABS],
         ["No-K (REL)",    pREL, rREL, fREL],
         ["Oracle-K",      pOK,  rOK,  fOK]],
        columns=["Setting","Precision","Recall","F1"]
    )
    return fold_sum


if __name__ == "__main__":
    log.info(f"CUDA available: {TORCH_HAS_CUDA} | Device: {TORCH_DEVICE}")
    log.info("cuML TruncatedSVD: " + ("ENABLED (GPU)" if USE_CUML_SVD else "not available, using sklearn CPU SVD"))
    log.info("BM25: " + ("ENABLED" if HAVE_BM25 else "disabled"))

    all_means = [] 

    for path in dataset_paths:
        ds_name = Path(path).stem
        print(f"\n=== Running 5-fold CV for dataset: {ds_name} ===")


        with stage(f"[{ds_name}] Load CSV"):
            df = pd.read_csv(path)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{ds_name}: Missing columns: {missing}")
            for c in [C_IDT, C_CDT]:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            df[C_Y] = pd.to_numeric(df[C_Y], errors="coerce").fillna(0).clip(0,1).astype(int)
            issue_ids = sorted(df[C_IID].drop_duplicates().tolist())

        DS_OUT = OUT_ROOT / ds_name / "cv5"
        DS_OUT.mkdir(parents=True, exist_ok=True)


        with stage("Load CodeBERT (GPU)"):
            tokenizer, model_sem = load_sem_model()

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_summaries = []

        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(issue_ids), start=1):
            fold_dir = DS_OUT / f"fold{fold_idx}"
            train_ids = set([issue_ids[i] for i in tr_idx])
            test_ids  = set([issue_ids[i] for i in te_idx])

            log.info(f"[{ds_name}] ===== CV fold{fold_idx}: train_issues={len(train_ids)}, test_issues={len(test_ids)} =====")
            fold_sum = run_one_fold(ds_name, df, train_ids, test_ids, fold_dir, tokenizer, model_sem)


            (DS_OUT / f"summary_fold{fold_idx}.csv").write_text(fold_sum.to_csv(index=False))
            fold_summaries.append(fold_sum.assign(Fold=fold_idx))


        all_summ = pd.concat(fold_summaries, ignore_index=True) if fold_summaries else pd.DataFrame(columns=["Setting","Precision","Recall","F1","Fold"])
        setting_order = ["No-K (ABS-mm)", "No-K (REL)", "Oracle-K"]


        mean_df = (all_summ.groupby("Setting", as_index=False)[["Precision","Recall","F1"]]
                            .mean()
                            .round(2))
        mean_df["Setting"] = pd.Categorical(mean_df["Setting"], categories=setting_order, ordered=True)
        mean_df = mean_df.sort_values("Setting").reset_index(drop=True)

        mean_out = DS_OUT / "summary_k5_mean.csv"
        mean_df.to_csv(mean_out, index=False)


        mean_df_with_ds = mean_df.copy()
        mean_df_with_ds.insert(0, "Dataset", ds_name)
        all_means.append(mean_df_with_ds)

        print(f"[DONE] {ds_name}: wrote 5 per-fold summaries + summary_k5_mean.csv in {DS_OUT}")

    if all_means:
        master_df = pd.concat(all_means, ignore_index=True)
        master_out = OUT_ROOT / "summary_k5_mean_all_datasets.csv"
        master_df.to_csv(master_out, index=False)
        print(f"[MASTER] Wrote stacked means to: {master_out}")
