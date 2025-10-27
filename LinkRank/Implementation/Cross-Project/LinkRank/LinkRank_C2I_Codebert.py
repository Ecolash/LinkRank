import os, time, math, logging, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD
from transformers import AutoTokenizer, AutoModel

try:
    import torch
    TORCH_HAS_CUDA = torch.cuda.is_available()
    TORCH_DEVICE = torch.device("cuda" if TORCH_HAS_CUDA else "cpu")
except Exception:
    TORCH_HAS_CUDA = False
    class _D:
        def __str__(self): return "cpu (torch not available)"
    TORCH_DEVICE = _D()


HAVE_BM25 = False
try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

train_paths = [
    "Add path of the training files here",
    
]
test_paths = [
    "Add path of the testing files here",

]

SUMMARY_ROOT = Path("Add your output summary directory here")
OUT_ROOT     = Path("Add your output directory here")   

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
SVD_DIM = 256
CODEBERT_MODEL = "microsoft/codebert-base"
CB_MAX_LEN = 256
CB_BATCH   = 16
USE_TIME_FEATURE = True
TIME_TAU_DAYS    = 7.0
TOPK_ISSUES_PER_COMMIT = 6  
BM25_M_ISSUES = 4           
BM25_M_COMMITS = 6         
TUNE_OBJECTIVE = "F1"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
log_path = OUT_ROOT / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger("hybrid_codebert_c2i_cross")

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

def simple_tokenize(s): return str(s).lower().split()

def bm25_scores(query_text, doc_texts):
    if not HAVE_BM25 or len(doc_texts)==0:
        return np.zeros(len(doc_texts), dtype=float)
    bm = BM25Okapi([simple_tokenize(t or "") for t in doc_texts])
    return bm.get_scores(simple_tokenize(query_text or ""))

def percent2(x):
    return float(np.round(100.0 * float(x), 2))

def split_by_group_robust(df, group_col, dev_frac=0.20, min_train_groups=1, min_dev_groups=1, rng=None):
    """Ensure both train/dev have at least one group."""
    if rng is None:
        rng = np.random.default_rng(42)
    groups = df[group_col].drop_duplicates().tolist()
    if len(groups) == 0:
        return df.copy(), df.iloc[0:0].copy()
    rng.shuffle(groups)
    n = len(groups)
    n_dev = max(min_dev_groups, int(round(dev_frac * n)))
    n_dev = min(n_dev, n - min_train_groups) if n > min_train_groups else 0
    dev_ids = set(groups[:n_dev])
    trn = df[~df[group_col].isin(dev_ids)].copy()
    dev = df[df[group_col].isin(dev_ids)].copy()
    if trn.empty and not dev.empty:
        mv = dev[group_col].iloc[0]
        trn = df[df[group_col] == mv].copy()
        dev = df[df[group_col] != mv].copy()
    if dev.empty and not trn.empty:
        mv = trn[group_col].iloc[0]
        dev = df[df[group_col] == mv].copy()
        trn = df[df[group_col] != mv].copy()
    if trn.empty and dev.empty:
        trn = df.copy()
        dev = df.iloc[0:0].copy()
    return trn, dev

def ensure_nonempty_train_valid(Xtr, ytr, gtr, Xdv, ydv, gdv):
    """Swap/slice to keep LightGBM happy."""
    if len(Xtr) > 0 and sum(gtr) > 0:
        if len(Xdv) == 0 or sum(gdv) == 0:
            cut = max(1, min(100, len(ytr)//10))
            return Xtr[cut:], ytr[cut:], gtr, Xtr[:cut], ytr[:cut], gtr
        return Xtr, ytr, gtr, Xdv, ydv, gdv
    if len(Xdv) > 0 and sum(gdv) > 0:
        return Xdv, ydv, gdv, Xtr, ytr, gtr
    raise ValueError("Both train and dev are empty.")

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
        log.info(f"[{tag}] LightGBM: GPU enabled.")
    else:
        log.info(f"[{tag}] LightGBM: CPU.")
    model = lgb.train(
        params, dtrain, valid_sets=[dtrain, dvalid],
        num_boost_round=1200,
        callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)]
    )
    log.info(f"[{tag}] Best iteration: {model.best_iteration}")
    return model

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
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(1) / mask.sum(1).clamp(min=1e-9)

def encode_texts(df, key_col, text_col, cache_path):
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=True).item()
    tok = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    mdl = AutoModel.from_pretrained(CODEBERT_MODEL).to(TORCH_DEVICE)
    mdl.eval()
    ids = df[key_col].tolist()
    texts = df[text_col].fillna("").tolist()
    embs = {}
    for i in tqdm(range(0, len(texts), CB_BATCH), desc=f"CodeBERT[{text_col}]"):
        batch_ids = ids[i:i+CB_BATCH]
        batch_txt = texts[i:i+CB_BATCH]
        with torch.no_grad():
            t = tok(batch_txt, padding=True, truncation=True, max_length=CB_MAX_LEN, return_tensors="pt")
            t = {k: v.to(TORCH_DEVICE) for k, v in t.items()}
            out = mdl(**t)
            pooled = mean_pool(out.last_hidden_state, t["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for j, _id in enumerate(batch_ids):
                embs[_id] = pooled[j].detach().cpu().numpy()
        if TORCH_HAS_CUDA:
            torch.cuda.empty_cache()
    np.save(cache_path, embs, allow_pickle=True)
    return embs

def cos_from_maps(a_map, a_id, b_map, b_id):
    va = a_map.get(a_id); vb = b_map.get(b_id)
    if va is None or vb is None: return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-12))

def build_features(d: pd.DataFrame,
                   VI, VC, i_idx, c_idx,
                   issue_cb, commit_cb, E_i_td, E_i_comm, E_c_msg, E_c_df, E_c_full):
    rows = []
    for _, row in tqdm(d.iterrows(), total=len(d), desc="build_features"):
        iid = row[C_IID]; cid = row[C_CID]

        f_text = 0.0
        if (iid in i_idx) and (cid in c_idx):
            f_text = cos_sim(VI[i_idx[iid]], VC[c_idx[cid]])

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

FEATURES = ["feat_text","feat_time",
            "feat_sem","feat_sem_td_msg","feat_sem_td_df","feat_sem_comm_full"]

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
    o = (TUNE_OBJECTIVE or "F1").lower()
    if o == "allcorrect":
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


def read_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).stem}: Missing columns: {missing}")
    for c in [C_IDT, C_CDT]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df[C_Y] = pd.to_numeric(df[C_Y], errors="coerce").fillna(0).clip(0,1).astype(int)
    df[C_IID] = df[C_REPO].astype(str) + "#" + df[C_IID].astype(str)
    df[C_CID] = df[C_REPO].astype(str) + "#" + df[C_CID].astype(str)
    return df

def hash_id_list(ids) -> str:
    h = hashlib.sha1(("|".join(sorted(map(str, ids)))).encode("utf-8")).hexdigest()[:10]
    return h

def main():
    print(f"CUDA available: {TORCH_HAS_CUDA} | BM25: {'ENABLED' if HAVE_BM25 else 'disabled'} | Device: {TORCH_DEVICE}")

    with stage("[LOAD] Train/Test datasets"):
        train_dfs = [read_and_prepare(p) for p in train_paths]
        test_dfs  = [(Path(p).stem, read_and_prepare(p)) for p in test_paths]

        train_df = pd.concat(train_dfs, ignore_index=True)
        log.info(f"[TRAIN] rows={len(train_df)}, issues={train_df[C_IID].nunique()}, commits={train_df[C_CID].nunique()}")


    train_wall_start = time.perf_counter()

    with stage("[TRAIN] TFIDF+SVD (fit on TRAIN only)"):
        issue_txt_tr  = issue_text(train_df)
        commit_txt_tr = commit_text(train_df)

        tfidf = TfidfVectorizer(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
        X_tr = tfidf.fit_transform(pd.concat([issue_txt_tr["text"], commit_txt_tr["text"]], axis=0).fillna(""))

        svd = SkTruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
        Xr_tr = svd.fit_transform(X_tr)
        E_issue_tr  = Xr_tr[:len(issue_txt_tr)]
        E_commit_tr = Xr_tr[len(issue_txt_tr):]

        issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr[C_IID].tolist())}
        commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr[C_CID].tolist())}

    with stage("[TRAIN] CodeBERT encode (train+all tests for caching)"):
        all_issue_txt_parts = [issue_txt_tr]
        all_commit_txt_parts = [commit_txt_tr]
        for name, df_te in test_dfs:
            all_issue_txt_parts.append(issue_text(df_te))
            all_commit_txt_parts.append(commit_text(df_te))
        all_issue_txt = pd.concat(all_issue_txt_parts, ignore_index=True).drop_duplicates(C_IID)
        all_commit_txt = pd.concat(all_commit_txt_parts, ignore_index=True).drop_duplicates(C_CID)

        cache_dir = OUT_ROOT / "cb_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ihash = hash_id_list(all_issue_txt[C_IID].tolist())
        chash = hash_id_list(all_commit_txt[C_CID].tolist())

        issue_cb  = encode_texts(all_issue_txt,  C_IID, "text", cache_dir / f"cb_issue_{ihash}.npy")
        commit_cb = encode_texts(all_commit_txt, C_CID, "text", cache_dir / f"cb_commit_{chash}.npy")

        i_fields_tr = issue_fields(train_df)
        c_fields_tr = commit_fields(train_df)
        i_fields_parts = [i_fields_tr]
        c_fields_parts = [c_fields_tr]
        for _, df_te in test_dfs:
            i_fields_parts.append(issue_fields(df_te))
            c_fields_parts.append(commit_fields(df_te))
        i_fields_all = pd.concat(i_fields_parts, ignore_index=True).drop_duplicates(C_IID)
        c_fields_all = pd.concat(c_fields_parts, ignore_index=True).drop_duplicates(C_CID)

        fihash = hash_id_list(i_fields_all[C_IID].tolist())
        fchash = hash_id_list(c_fields_all[C_CID].tolist())

        E_i_td   = encode_texts(i_fields_all, C_IID, "i_td",   cache_dir / f"cb_i_td_{fihash}.npy")
        E_i_comm = encode_texts(i_fields_all, C_IID, "i_comm", cache_dir / f"cb_i_comm_{fihash}.npy")
        E_c_msg  = encode_texts(c_fields_all, C_CID, "c_msg",  cache_dir / f"cb_c_msg_{fchash}.npy")
        E_c_df   = encode_texts(c_fields_all, C_CID, "c_df",   cache_dir / f"cb_c_df_{fchash}.npy")
        E_c_full = encode_texts(c_fields_all, C_CID, "c_full", cache_dir / f"cb_c_full_{fchash}.npy")

        issue_text_map  = all_issue_txt.set_index(C_IID)["text"].to_dict()
        commit_text_map = all_commit_txt.set_index(C_CID)["text"].to_dict()

    with stage("[TRAIN] Build features (TRAIN)"):
        train_feat = build_features(train_df,
                                    E_issue_tr, E_commit_tr, issue_idx_tr, commit_idx_tr,
                                    issue_cb, commit_cb, E_i_td, E_i_comm, E_c_msg, E_c_df, E_c_full)
        log.info(f"[TRAIN] train_feat rows={len(train_feat)}")

    with stage("[TRAIN] Train A2 (commit→issues)"):
        trn_A2, dev_A2 = split_by_group_robust(train_feat, group_col=C_CID, dev_frac=0.20, rng=rng)
        def prep_rank_commit(df_feat):
            df_s = df_feat.sort_values([C_CID]).reset_index(drop=True)
            X = df_s[FEATURES].values.astype(np.float32)
            y = df_s[C_Y].astype(int).values
            groups = df_s.groupby(C_CID).size().tolist()
            return df_s, X, y, groups
        trn_s_A2, Xtr_A2, ytr_A2, gtr_A2 = prep_rank_commit(trn_A2)
        dev_s_A2, Xdv_A2, ydv_A2, gdv_A2 = prep_rank_commit(dev_A2)
        log.info(f"[A2] Train rows={len(Xtr_A2)}, groups={len(gtr_A2)} | Dev rows={len(Xdv_A2)}, groups={len(gdv_A2)}")
        Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2 = ensure_nonempty_train_valid(Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2)
        model_A2 = train_lambdamart("A2", FEATURES, Xtr_A2, ytr_A2, gtr_A2, Xdv_A2, ydv_A2, gdv_A2)

    with stage("[TRAIN] Train A1 (issue→commits) + tune thresholds"):
        trn_A1, dev_A1 = split_by_group_robust(train_feat, group_col=C_IID, dev_frac=0.20, rng=rng)
        def prep_rank_issue(df_feat):
            df_s = df_feat.sort_values([C_IID]).reset_index(drop=True)
            X = df_s[FEATURES].values.astype(np.float32)
            y = df_s[C_Y].astype(int).values
            groups = df_s.groupby(C_IID).size().tolist()
            return df_s, X, y, groups
        trn_s_A1, Xtr_A1, ytr_A1, gtr_A1 = prep_rank_issue(trn_A1)
        dev_s_A1, Xdv_A1, ydv_A1, gdv_A1 = prep_rank_issue(dev_A1)
        log.info(f"[A1] Train rows={len(Xtr_A1)}, groups={len(gtr_A1)} | Dev rows={len(Xdv_A1)}, groups={len(gdv_A1)}")
        Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1 = ensure_nonempty_train_valid(Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1)
        model_A1 = train_lambdamart("A1", FEATURES, Xtr_A1, ytr_A1, gtr_A1, Xdv_A1, ydv_A1, gdv_A1)

        dev_shortlist_like = topk_issues_per_commit(dev_A1, model_A2, issue_text_map, commit_text_map)
        true_by_issue_dev = dev_s_A1[dev_s_A1[C_Y]==1].groupby(C_IID)[C_CID].apply(set).to_dict()
        issues_dev = sorted(true_by_issue_dev.keys())
        dev_rows_by_issue = {iid: dev_s_A1[dev_s_A1[C_IID]==iid].reset_index(drop=True)
                             for iid in issues_dev}

        def eval_abs_mm_on_dev(tau):
            res=[]
            for iid in issues_dev:
                pool_full = dev_rows_by_issue[iid]
                pool = restrict_pool_by_A2_and_BM25(iid, pool_full, dev_shortlist_like, issue_text_map, commit_text_map)
                ranked = score_pool_with_model(model_A1, pool)
                chosen = ranked[ranked["score_mm"] >= float(tau)][C_CID].tolist()
                res.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
            dfm = pd.DataFrame(res); return aggregate_dev(dfm), dfm

        def eval_rel_on_dev(gamma):
            res=[]
            for iid in issues_dev:
                pool_full = dev_rows_by_issue[iid]
                pool = restrict_pool_by_A2_and_BM25(iid, pool_full, dev_shortlist_like, issue_text_map, commit_text_map)
                ranked = score_pool_with_model(model_A1, pool)
                if len(ranked)==0:
                    res.append(metrics_for_issue(set(), true_by_issue_dev[iid])); continue
                best = float(ranked["score"].iloc[0])
                chosen = ranked[ranked["score"] >= float(gamma) * best][C_CID].tolist()
                res.append(metrics_for_issue(set(chosen), true_by_issue_dev[iid]))
            dfm = pd.DataFrame(res); return aggregate_dev(dfm), dfm

        taus   = [x/100 for x in range(20, 96, 2)]
        gammas = [x/100 for x in range(30, 96, 2)]
        best_abs = (-1, None); best_rel = (-1, None)
        for t in taus:
            sc,_ = eval_abs_mm_on_dev(t)
            if sc > best_abs[0]: best_abs = (sc, t)
        for g in gammas:
            sc,_ = eval_rel_on_dev(g)
            if sc > best_rel[0]: best_rel = (sc, g)
        log.info(f"[DEV] best ABS-mm τ={best_abs[1]:.2f}, score={best_abs[0]:.4f}")
        log.info(f"[DEV] best REL   γ={best_rel[1]:.2f}, score={best_rel[0]:.4f}")

    train_wall_end = time.perf_counter()
    total_train_seconds = round(train_wall_end - train_wall_start, 2)

    with stage("[SAVE] Train artifacts"):
        (OUT_ROOT / "train_artifacts").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "summaries").mkdir(parents=True, exist_ok=True)
        ta = OUT_ROOT / "train_artifacts"
        model_A2.save_model(str(ta / "model_A2_commit2issues.txt"))
        model_A1.save_model(str(ta / "model_A1_issue2commits.txt"))
        train_df.to_csv(ta / "train_rows.csv", index=False)

    all_summary = []
    all_timings = []
    for name, test_df in test_dfs:
        with stage(f"[{name}] TFIDF+SVD transform (TEST only)"):
            issue_txt_te  = issue_text(test_df)
            commit_txt_te = commit_text(test_df)
            X_te_issue  = svd.transform(tfidf.transform(issue_txt_te["text"].fillna("")))
            X_te_commit = svd.transform(tfidf.transform(commit_txt_te["text"].fillna("")))
            issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te[C_IID].tolist())}
            commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te[C_CID].tolist())}

        with stage(f"[{name}] Build features (TEST)"):
            test_feat  = build_features(test_df,
                                        X_te_issue, X_te_commit, issue_idx_te, commit_idx_te,
                                        issue_cb, commit_cb, E_i_td, E_i_comm, E_c_msg, E_c_df, E_c_full)
            log.info(f"[{name}] test_feat rows={len(test_feat)}")

        test_start = time.perf_counter()

        with stage(f"[{name}] Build A2 shortlist for TEST"):
            test_shortlist = topk_issues_per_commit(test_feat, model_A2, issue_text_map, commit_text_map)

        with stage(f"[{name}] Evaluate"):
            ds_out = OUT_ROOT / f"eval_{name}"
            ds_out.mkdir(parents=True, exist_ok=True)

            def prep_rank_issue(df_feat):
                df_s = df_feat.sort_values([C_IID]).reset_index(drop=True)
                return df_s
            tst_s_A1 = prep_rank_issue(test_feat)

            tst_rows_by_issue = {iid: tst_s_A1[tst_s_A1[C_IID]==iid].reset_index(drop=True)
                                 for iid in tst_s_A1[C_IID].drop_duplicates().tolist()}
            true_by_issue_test = tst_s_A1[tst_s_A1[C_Y]==1].groupby(C_IID)[C_CID].apply(set).to_dict()
            issues_test = sorted(true_by_issue_test.keys())

            def eval_policy_on_test(kind, val, tag):
                rows=[]; dumps=[]
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
                    true = true_by_issue_test.get(iid, set())
                    m = metrics_for_issue(pred, true)
                    rows.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true)))
                    tmp = ranked[[C_CID,"score","score_mm","score_zn",C_Y]].copy()
                    tmp[C_IID]=iid; tmp["Rank"]=np.arange(1, len(tmp)+1); dumps.append(tmp)
                res_df = pd.DataFrame(rows)
                res_df.to_csv(ds_out / f"hybrid_NoK_{tag}.csv", index=False)
                if dumps:
                    pd.concat(dumps, ignore_index=True).to_csv(ds_out / f"hybrid_NoK_ranked_{tag}.csv", index=False)
                log.info(f"[{name}] No-K {tag} — P={res_df['Precision'].mean():.4f} R={res_df['Recall'].mean():.4f} F1={res_df['F1'].mean():.4f}")
                return res_df

            def eval_test_oracleK(tag="OracleK_topK_after_A2BM25"):
                rows=[]; dumps=[]
                for iid in issues_test:
                    pool_full = tst_rows_by_issue[iid]
                    pool = restrict_pool_by_A2_and_BM25(iid, pool_full, test_shortlist, issue_text_map, commit_text_map)
                    ranked = score_pool_with_model(model_A1, pool)
                    true = true_by_issue_test.get(iid, set())
                    K = len(true)
                    chosen = ranked[C_CID].iloc[:K].tolist() if K>0 else []
                    pred=set(chosen)
                    m = metrics_for_issue(pred, true)
                    rows.append(dict(Issue=iid, **m, Predicted=len(pred), TrueCount=len(true)))
                    tmp = ranked[[C_CID,"score","score_mm","score_zn",C_Y]].copy()
                    tmp[C_IID]=iid; tmp["Rank"]=np.arange(1, len(tmp)+1); dumps.append(tmp)
                res_df = pd.DataFrame(rows)
                res_df.to_csv(ds_out / f"hybrid_{tag}.csv", index=False)
                if dumps:
                    pd.concat(dumps, ignore_index=True).to_csv(ds_out / f"hybrid_ranked_{tag}.csv", index=False)
                log.info(f"[{name}] {tag} — P={res_df['Precision'].mean():.4f} R={res_df['Recall'].mean():.4f} F1={res_df['F1'].mean():.4f}")
                return res_df

            tag_abs = f"ABSmm_tau{best_abs[1]:.2f}_{TUNE_OBJECTIVE}"
            tag_rel = f"REL_gamma{best_rel[1]:.2f}_{TUNE_OBJECTIVE}"
            res_abs = eval_policy_on_test("ABS", best_abs[1], tag_abs)
            res_rel = eval_policy_on_test("REL", best_rel[1], tag_rel)
            res_ok  = eval_test_oracleK()

        test_end = time.perf_counter()
        test_seconds = round(test_end - test_start, 2)

        def macro_pct(dfm):
            return dict(
                Precision = percent2(dfm["Precision"].mean()),
                Recall    = percent2(dfm["Recall"].mean()),
                F1        = percent2(dfm["F1"].mean()),
            )
        s_abs = {"Dataset": name, "Setting":"No-K (ABS-mm)", **macro_pct(res_abs)}
        s_rel = {"Dataset": name, "Setting":"No-K (REL)",    **macro_pct(res_rel)}
        s_ok  = {"Dataset": name, "Setting":"Oracle-K",      **macro_pct(res_ok)}
        all_summary.extend([s_abs, s_rel, s_ok])

        all_timings.append({"Dataset": name, "TrainSeconds": total_train_seconds, "TestSeconds": test_seconds})

    with stage("[SAVE] Rollup CSVs"):
        summary_df = pd.DataFrame(all_summary, columns=["Dataset","Setting","Precision","Recall","F1"])
        timings_df = pd.DataFrame(all_timings,  columns=["Dataset","TrainSeconds","TestSeconds"])

        summary_csv = OUT_ROOT / "summaries" / "summary_results_cross.csv"
        timings_csv = OUT_ROOT / "summaries" / "timings_cross.csv"
        summary_df.to_csv(summary_csv, index=False)
        timings_df.to_csv(timings_csv, index=False)

        print(f"\nSummary written to: {summary_csv}")
        print(summary_df.to_string(index=False))
        print(f"\nTimings written to: {timings_csv}")
        print(timings_df.to_string(index=False))

if __name__ == "__main__":
    main()


