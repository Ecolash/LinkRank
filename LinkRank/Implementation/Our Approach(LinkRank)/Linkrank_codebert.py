import os, re, time, logging, math, hashlib, json
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf


from transformers import AutoTokenizer, AutoModel

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
    if TORCH_HAS_CUDA:
        # speed knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
except Exception:
    TORCH_HAS_CUDA = False
    class _DummyDevice:
        def __str__(self): return "cpu (torch not installed)"
    torch_device = _DummyDevice()

USE_CUML_SVD   = True  
USE_CUML_TFIDF = False  

try:
    from cuml.decomposition import TruncatedSVD as cuSVD
    HAS_CUML_SVD = True
except Exception:
    HAS_CUML_SVD = False
    from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD

if USE_CUML_TFIDF:
    try:
        import cudf
        from cuml.feature_extraction.text import TfidfVectorizer as cuTfidf
        HAS_CUML_TFIDF = True
    except Exception:
        HAS_CUML_TFIDF = False
else:
    HAS_CUML_TFIDF = False




dataset_paths = [
    "Ass your file paths here",

]

OUT_ROOT = Path("Add your output directory path here")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_ROOT / "run_codebert_cv5_gpu.log"
RANDOM_SEED = 42
K_FOLDS = 5
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAMS = (1, 2)
SVD_DIM = 256
USE_TIME_FEATURE = True
TIME_TAU_DAYS = 7.0
SEM_MODEL_NAME = "microsoft/codebert-base"
SEM_BATCH_SIZE = 128           
MAX_LEN = 256
FP16 = True                     
USE_ITERATION_KNOWNK = True
USE_ITERATION_NOK_REL = True
ALPHA = 0.7
BETA  = 0.3
UPDATE_FILES = True
LOG_EVERY_N_ITERS = 10
TUNE_OBJECTIVE = "F1"   
USE_CAP = False
MAX_RELATIVE = 0.75



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger("codebert_linkrank_cv5_gpu")

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
log.info(f"cuML SVD available: {HAS_CUML_SVD} (requested={USE_CUML_SVD}) | cuML TF-IDF available: {HAS_CUML_TFIDF} (requested={USE_CUML_TFIDF})")


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def cos_sim(a, b):
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den

def tokenize(s: str):
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def parse_files(x):
    if pd.isna(x): return set()
    return set([p.strip() for p in str(x).split(",") if p.strip()])

def time_prox(issue_date, commit_date, tau_days=14.0):
    if pd.isna(issue_date) or pd.isna(commit_date): return 0.0
    days = abs((commit_date - issue_date).days)
    return float(np.exp(-days / tau_days))

def issue_text(d):
    g = d.groupby("Issue ID").agg(
        title=("Title","first"), desc=("Description","first"), comm=("Comments","first"),
    )
    out = (g["title"].fillna("") + " " + g["desc"].fillna("") + " " + g["comm"].fillna("")).reset_index(name="text")
    return out

def commit_text(d):
    g = d.groupby("Commit ID").agg(
        msg=("Message","first"), dif=("Diff Summary","first"),
        files=("File Changes","first"), full=("Full Diff","first"),
    )
    out = (g["msg"].fillna("") + " " + g["dif"].fillna("") + " " + g["files"].fillna("") + " " + g["full"].fillna("")).reset_index(name="text")
    return out

def issue_fields(d):
    g = d.groupby("Issue ID").agg(title=("Title","first"), desc=("Description","first"), comm=("Comments","first")).fillna("")
    return pd.DataFrame({"Issue ID": g.index, "i_td": (g["title"] + " " + g["desc"]).values, "i_comm": g["comm"].values})

def commit_fields(d):
    g = d.groupby("Commit ID").agg(msg=("Message","first"), dif=("Diff Summary","first"), files=("File Changes","first"), full=("Full Diff","first")).fillna("")
    return pd.DataFrame({
        "Commit ID": g.index,
        "c_msg": g["msg"].values,
        "c_difffiles": (g["dif"] + " " + g["files"]).values,
        "c_full": g["full"].values
    })

def issue_file_union_all(d):
    if "File Changes" not in d.columns:
        return {}
    t = d.groupby("Issue ID")["File Changes"].apply(lambda s: set().union(*[parse_files(x) for x in s]))
    return t.to_dict()

def macro_percent(df):
    p = round(100.0 * df["Precision"].mean(), 2) if len(df) else 0.0
    r = round(100.0 * df["Recall"].mean(),    2) if len(df) else 0.0
    f = round(100.0 * df["F1"].mean(),        2) if len(df) else 0.0
    return p, r, f


def load_sem_model():
    with stage("Load CodeBERT (GPU)" if TORCH_HAS_CUDA else "Load CodeBERT (CPU)"):
        tokenizer = AutoTokenizer.from_pretrained(SEM_MODEL_NAME, use_fast=True)
        if TORCH_HAS_CUDA:
            model_sem = AutoModel.from_pretrained(
                SEM_MODEL_NAME,
                dtype=(torch.float16 if FP16 else torch.float32)
            ).to("cuda")
        else:
            model_sem = AutoModel.from_pretrained(SEM_MODEL_NAME)
        model_sem.eval()
    return tokenizer, model_sem

def _hash_list_of_texts(name, texts, extra=None):
    h = hashlib.sha1()
    h.update(name.encode("utf-8"))
    for t in texts:
        h.update((t or "").encode("utf-8"))
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return h.hexdigest()

@torch.inference_mode()
def encode_texts_cached(tokenizer, model_sem, texts, cache_dir, tag):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(model=SEM_MODEL_NAME, fp16=FP16, max_len=MAX_LEN, batch=SEM_BATCH_SIZE)
    key  = _hash_list_of_texts(tag, texts, extra=meta)
    npy_path = cache_dir / f"{tag}.{key}.npy"
    if npy_path.exists():
        return np.load(npy_path)

    vecs = []
    device = "cuda" if TORCH_HAS_CUDA else "cpu"
    for i in tqdm(range(0, len(texts), SEM_BATCH_SIZE), desc=f"encode[{tag}]", leave=False):
        batch = texts[i:i+SEM_BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        out = model_sem(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vecs.append(emb.detach().cpu().numpy())
    arr = np.vstack(vecs) if vecs else np.zeros((0, model_sem.config.hidden_size), dtype=np.float32)
    np.save(npy_path, arr)
    return arr


def run_one_fold(ds_name, full_df, train_ids, test_ids, fold_dir, tokenizer, model_sem, cache_dir):
    fold_dir.mkdir(parents=True, exist_ok=True)


    train_df = full_df[full_df["Issue ID"].isin(train_ids)].copy()
    test_df  = full_df[full_df["Issue ID"].isin(test_ids)].copy()


    try:
        tmp = test_df.groupby("Commit ID")["File Changes"].first().fillna("")
        commit_files_te = {cid: parse_files(fc) for cid, fc in tmp.items()}
    except KeyError:
        commit_files_te = {}


    with stage(f"[{ds_name}] TF-IDF/SVD (fit on TRAIN; transform TEST)"):
        issue_txt_tr = issue_text(train_df); commit_txt_tr = commit_text(train_df)
        train_texts = pd.concat([issue_txt_tr["text"], commit_txt_tr["text"]], axis=0).fillna("")

        if HAS_CUML_TFIDF:
            tr_series = cudf.Series(train_texts.tolist())
            tfidf = cuTfidf(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
            X_tr = tfidf.fit_transform(tr_series)  # GPU
            issue_txt_te = issue_text(test_df); commit_txt_te = commit_text(test_df)
            X_te_issue  = tfidf.transform(cudf.Series(issue_txt_te["text"].fillna("").tolist()))
            X_te_commit = tfidf.transform(cudf.Series(commit_txt_te["text"].fillna("").tolist()))
        else:
            tfidf = SkTfidf(min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, ngram_range=TFIDF_NGRAMS)
            X_tr = tfidf.fit_transform(train_texts)
            issue_txt_te = issue_text(test_df); commit_txt_te = commit_text(test_df)
            X_te_issue  = tfidf.transform(issue_txt_te["text"].fillna(""))
            X_te_commit = tfidf.transform(commit_txt_te["text"].fillna(""))

        if USE_CUML_SVD and HAS_CUML_SVD:
            with stage(f"[{ds_name}] cuML TruncatedSVD (GPU)"):
                svd = cuSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
                Xr_tr = svd.fit_transform(X_tr)
                Er_issue_tr  = Xr_tr[:len(issue_txt_tr)]
                Er_commit_tr = Xr_tr[len(issue_txt_tr):]
                Xr_te_issue  = svd.transform(X_te_issue)
                Xr_te_commit = svd.transform(X_te_commit)

                E_issue_tr  = Er_issue_tr.get()
                E_commit_tr = Er_commit_tr.get()
                X_te_issue  = Xr_te_issue.get()
                X_te_commit = Xr_te_commit.get()
        else:
            with stage(f"[{ds_name}] sklearn TruncatedSVD (CPU)"):
                svd = SkTruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
                Xr_tr = svd.fit_transform(X_tr)
                E_issue_tr  = Xr_tr[:len(issue_txt_tr)]
                E_commit_tr = Xr_tr[len(issue_txt_tr):]
                X_te_issue  = svd.transform(X_te_issue)
                X_te_commit = svd.transform(X_te_commit)


        issue_idx_tr  = {iid: i for i, iid in enumerate(issue_txt_tr["Issue ID"].tolist())}
        commit_idx_tr = {cid: i for i, cid in enumerate(commit_txt_tr["Commit ID"].tolist())}
        issue_idx_te  = {iid: i for i, iid in enumerate(issue_txt_te["Issue ID"].tolist())}
        commit_idx_te = {cid: i for i, cid in enumerate(commit_txt_te["Commit ID"].tolist())}


    with stage(f"[{ds_name}] Encode CodeBERT (global TR/TE)"):
        E_issue_sem_tr  = encode_texts_cached(tokenizer, model_sem, issue_txt_tr["text"].fillna("").tolist(), cache_dir, "issue_sem_tr")
        E_commit_sem_tr = encode_texts_cached(tokenizer, model_sem, commit_txt_tr["text"].fillna("").tolist(), cache_dir, "commit_sem_tr")
        E_issue_sem_te  = encode_texts_cached(tokenizer, model_sem, issue_txt_te["text"].fillna("").tolist(),  cache_dir, "issue_sem_te")
        E_commit_sem_te = encode_texts_cached(tokenizer, model_sem, commit_txt_te["text"].fillna("").tolist(), cache_dir, "commit_sem_te")

    with stage(f"[{ds_name}] Encode CodeBERT (fields TR/TE)"):
        i_fields_tr = issue_fields(train_df); c_fields_tr = commit_fields(train_df)
        i_fields_te = issue_fields(test_df);  c_fields_te = commit_fields(test_df)

        i_td_idx_tr   = {iid:i for i, iid in enumerate(i_fields_tr["Issue ID"].tolist())}
        i_td_idx_te   = {iid:i for i, iid in enumerate(i_fields_te["Issue ID"].tolist())}
        i_comm_idx_tr = i_td_idx_tr
        i_comm_idx_te = i_td_idx_te

        c_msg_idx_tr  = {cid:i for i, cid in enumerate(c_fields_tr["Commit ID"].tolist())}
        c_msg_idx_te  = {cid:i for i, cid in enumerate(c_fields_te["Commit ID"].tolist())}
        c_df_idx_tr   = c_msg_idx_tr
        c_df_idx_te   = c_msg_idx_te
        c_full_idx_tr = c_msg_idx_tr
        c_full_idx_te = c_msg_idx_te

        E_i_td_tr    = encode_texts_cached(tokenizer, model_sem, i_fields_tr["i_td"].tolist(),    cache_dir, "i_td_tr")
        E_i_comm_tr  = encode_texts_cached(tokenizer, model_sem, i_fields_tr["i_comm"].tolist(),  cache_dir, "i_comm_tr")
        E_c_msg_tr   = encode_texts_cached(tokenizer, model_sem, c_fields_tr["c_msg"].tolist(),   cache_dir, "c_msg_tr")
        E_c_df_tr    = encode_texts_cached(tokenizer, model_sem, c_fields_tr["c_difffiles"].tolist(), cache_dir, "c_df_tr")
        E_c_full_tr  = encode_texts_cached(tokenizer, model_sem, c_fields_tr["c_full"].tolist(),  cache_dir, "c_full_tr")

        E_i_td_te    = encode_texts_cached(tokenizer, model_sem, i_fields_te["i_td"].tolist(),    cache_dir, "i_td_te")
        E_i_comm_te  = encode_texts_cached(tokenizer, model_sem, i_fields_te["i_comm"].tolist(),  cache_dir, "i_comm_te")
        E_c_msg_te   = encode_texts_cached(tokenizer, model_sem, c_fields_te["c_msg"].tolist(),   cache_dir, "c_msg_te")
        E_c_df_te    = encode_texts_cached(tokenizer, model_sem, c_fields_te["c_difffiles"].tolist(), cache_dir, "c_df_te")
        E_c_full_te  = encode_texts_cached(tokenizer, model_sem, c_fields_te["c_full"].tolist(),  cache_dir, "c_full_te")


    with stage(f"[{ds_name}] BM25 prep"):
        commit_docs_tr = {cid: tokenize(txt) for cid, txt in zip(commit_txt_tr["Commit ID"], commit_txt_tr["text"])}
        commit_docs_te = {cid: tokenize(txt) for cid, txt in zip(commit_txt_te["Commit ID"], commit_txt_te["text"])}
        N = len(commit_docs_tr)
        df_t = {}; doc_len = {}
        for cid, toks in commit_docs_tr.items():
            doc_len[cid] = len(toks)
            for t in set(toks):
                df_t[t] = df_t.get(t, 0) + 1
        avgdl = (sum(doc_len.values()) / max(N, 1)) if N > 0 else 0.0
        idf = {t: math.log((N - v + 0.5) / (v + 0.5) + 1.0) for t, v in df_t.items()}

        def _tf_dict(toks):
            d = {}
            for w in toks: d[w] = d.get(w, 0) + 1
            return d
        tf_tr = {cid: _tf_dict(toks) for cid, toks in commit_docs_tr.items()}
        tf_te = {cid: _tf_dict(toks) for cid, toks in commit_docs_te.items()}
        doc_len_te = {cid: len(toks) for cid, toks in commit_docs_te.items()}

        def bm25_score(query_tokens, doc_tf, doc_len_val, idf_map, avgdl, k1=1.5, b=0.75):
            score = 0.0
            for t in query_tokens:
                if t not in idf_map: continue
                tf = doc_tf.get(t, 0)
                if tf == 0: continue
                denom = tf + k1 * (1 - b + b * (doc_len_val / (avgdl + 1e-9)))
                score += idf_map[t] * (tf * (k1 + 1)) / (denom + 1e-12)
            return float(score)

        issue_docs_tr = {iid: tokenize(txt) for iid, txt in zip(issue_txt_tr["Issue ID"], issue_txt_tr["text"])}
        issue_docs_te = {iid: tokenize(txt) for iid, txt in zip(issue_txt_te["Issue ID"], issue_txt_te["text"])}


    train_issue_files = issue_file_union_all(train_df)
    test_issue_files  = issue_file_union_all(test_df)


    FEATURE_COLS = [
        "feat_text",
        "feat_sem","feat_sem_td_msg","feat_sem_td_df","feat_sem_comm_full",
        "feat_bm25","feat_file","feat_time"
    ]

    def codebert_cos(iid, cid, split):
        if split == "train":
            ii = issue_idx_tr.get(iid); cc = commit_idx_tr.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_issue_sem_tr[ii], E_commit_sem_tr[cc])
        else:
            ii = issue_idx_te.get(iid); cc = commit_idx_te.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_issue_sem_te[ii], E_commit_sem_te[cc])

    def codebert_cos_iTD_cMSG(iid, cid, split):
        if split == "train":
            ii = i_td_idx_tr.get(iid); cc = c_msg_idx_tr.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_i_td_tr[ii], E_c_msg_tr[cc])
        else:
            ii = i_td_idx_te.get(iid); cc = c_msg_idx_te.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_i_td_te[ii], E_c_msg_te[cc])

    def codebert_cos_iTD_cDF(iid, cid, split):
        if split == "train":
            ii = i_td_idx_tr.get(iid); cc = c_df_idx_tr.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_i_td_tr[ii], E_c_df_tr[cc])
        else:
            ii = i_td_idx_te.get(iid); cc = c_df_idx_te.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_i_td_te[ii], E_c_df_te[cc])

    def codebert_cos_iCOMM_cFULL(iid, cid, split):
        if split == "train":
            ii = i_comm_idx_tr.get(iid); cc = c_full_idx_tr.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_i_comm_tr[ii], E_c_full_tr[cc])
        else:
            ii = i_comm_idx_te.get(iid); cc = c_full_idx_te.get(cid)
            if ii is None or cc is None: return 0.0
            return cos_sim(E_i_comm_te[ii], E_c_full_te[cc])

    def build_features(d, issue_files_union, split="train"):
        rows = []
        it = tqdm(d.iterrows(), total=len(d), desc=f"build_features[{ds_name}][{split}]")
        for _, row in it:
            iid = row["Issue ID"]; cid = row["Commit ID"]

            if split == "train":
                if iid in issue_idx_tr and cid in commit_idx_tr:
                    f_text = cos_sim(E_issue_tr[issue_idx_tr[iid]], E_commit_tr[commit_idx_tr[cid]])
                else:
                    f_text = 0.0
            else:
                if iid in issue_idx_te and cid in commit_idx_te:
                    f_text = cos_sim(X_te_issue[issue_idx_te[iid]], X_te_commit[commit_idx_te[cid]])
                else:
                    f_text = 0.0

            files_i = issue_files_union.get(iid, set())
            files_c = parse_files(row.get("File Changes"))
            inter = len(files_i & files_c); union = len(files_i | files_c)
            f_file = (inter/union) if union > 0 else 0.0

            f_time = time_prox(row.get("Issue Date"), row.get("Commit Date"), TIME_TAU_DAYS) if USE_TIME_FEATURE else 0.0

            f_sem_global    = codebert_cos(iid, cid, split)
            f_sem_td_msg    = codebert_cos_iTD_cMSG(iid, cid, split)
            f_sem_td_df     = codebert_cos_iTD_cDF(iid, cid, split)
            f_sem_comm_full = codebert_cos_iCOMM_cFULL(iid, cid, split)

            if split == "train":
                q_toks = issue_docs_tr.get(iid, []); doc_tf = tf_tr.get(cid, {}); dlen = doc_len.get(cid, 0)
            else:
                q_toks = issue_docs_te.get(iid, []); doc_tf = tf_te.get(cid, {}); dlen = doc_len_te.get(cid, 0)
            f_bm25 = 0.0 if len(q_toks)==0 else bm25_score(q_toks, doc_tf, dlen, idf, avgdl)

            rows.append([iid, cid, f_text,
                         f_sem_global, f_sem_td_msg, f_sem_td_df, f_sem_comm_full,
                         f_bm25, f_file, f_time, row["Output"]])
        return pd.DataFrame(rows, columns=["Issue ID","Commit ID","feat_text",
                                           "feat_sem","feat_sem_td_msg","feat_sem_td_df","feat_sem_comm_full",
                                           "feat_bm25","feat_file","feat_time","Output"])

    with stage(f"[{ds_name}] Build features (TRAIN/TEST)"):
        train_feat = build_features(train_df, train_issue_files, split="train")
        test_feat  = build_features(test_df,  test_issue_files,  split="test")


    def prep_rank(df_feat):
        df_s = df_feat.sort_values(["Issue ID"]).reset_index(drop=True)
        X = df_s[FEATURE_COLS].values
        y = df_s["Output"].astype(int).values
        groups = df_s.groupby("Issue ID").size().tolist()
        return df_s, X, y, groups

    uniq_train_issues = train_feat["Issue ID"].drop_duplicates().tolist()
    rng.shuffle(uniq_train_issues)
    n_dev = max(1, int(0.20*len(uniq_train_issues)))
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
    test_feat = _filter_groups_with_labels(test_feat, require_neg=False)

    trn_s, Xtr, ytr, gtr = prep_rank(trn_feat)
    dev_s, Xdv, ydv, gdv = prep_rank(dev_feat)
    tst_s, Xte, yte, gte = prep_rank(test_feat)


    with stage(f"[{ds_name}] Train LambdaMART"):
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
            log.info(f"[{ds_name}] LightGBM: GPU enabled.")
        else:
            params.update(dict(device_type="cpu"))
            log.info(f"[{ds_name}] LightGBM: CPU.")

        model = lgb.train(
            params, dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=2000,
            callbacks=[
                lgb.log_evaluation(LOG_EVERY_N_ITERS),
                lgb.early_stopping(stopping_rounds=150, verbose=True)
            ]
        )


    def score_pool(pool_df):
        if len(pool_df) == 0: return pool_df.assign(score=[])
        X = pool_df[FEATURE_COLS].values
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


    issue_sem_state = {}
    issue_file_state = {}

    def init_issue_state(iid):
        issue_sem_state[iid] = {
            "global":  E_issue_sem_te[ issue_idx_te[iid] ],
            "td":      E_i_td_te[     i_td_idx_te[iid]   ],
            "comm":    E_i_comm_te[   i_comm_idx_te[iid] ],
        }
        issue_file_state[iid] = set(test_issue_files.get(iid, set()))

    def update_issue_state(iid, cid):
        st = issue_sem_state[iid]
        st["global"] = ALPHA*st["global"] + BETA*E_commit_sem_te[ commit_idx_te[cid] ]
        st["td"]     = ALPHA*st["td"]     + BETA*E_c_msg_te[      c_msg_idx_te[cid]  ]
        st["comm"]   = ALPHA*st["comm"]   + BETA*E_c_full_te[     c_full_idx_te[cid] ]
        for k in st:
            v = st[k]; st[k] = v / (np.linalg.norm(v) + 1e-12)
        if UPDATE_FILES:
            issue_file_state[iid] |= commit_files_te.get(cid, set())

    def refresh_features_for_issue_pool(iid, pool_df):
        if len(pool_df)==0: return pool_df
        st = issue_sem_state[iid]; files_i = issue_file_state[iid]
        new = []
        for _, r in pool_df.iterrows():
            cid = r["Commit ID"]
            sem_global    = cos_sim(st["global"], E_commit_sem_te[ commit_idx_te[cid] ])
            sem_td_msg    = cos_sim(st["td"],     E_c_msg_te[      c_msg_idx_te[cid]  ])
            sem_td_df     = cos_sim(st["td"],     E_c_df_te[       c_df_idx_te[cid]   ])
            sem_comm_full = cos_sim(st["comm"],   E_c_full_te[     c_full_idx_te[cid] ])
            files_c = commit_files_te.get(cid, set())
            inter = len(files_i & files_c); union = len(files_i | files_c)
            f_file = (inter/union) if union>0 else 0.0
            new.append([r["Issue ID"], r["Commit ID"], r["feat_text"],
                        sem_global, sem_td_msg, sem_td_df, sem_comm_full,
                        r["feat_bm25"], f_file, r["feat_time"], r["Output"]])
        cols = ["Issue ID","Commit ID","feat_text",
                "feat_sem","feat_sem_td_msg","feat_sem_td_df","feat_sem_comm_full",
                "feat_bm25","feat_file","feat_time","Output"]
        return pd.DataFrame(new, columns=cols)


    test_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                          for iid in tst_s["Issue ID"].drop_duplicates().tolist()}
    true_by_issue = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    issues_test = sorted(true_by_issue.keys())

    rowsK = []
    for iid in issues_test:
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
        rowsK.append(dict(Precision=prec, Recall=rec, F1=f1))
    res_knownK = pd.DataFrame(rowsK)

    dev_rows_by_issue = {iid: dev_s[dev_s["Issue ID"]==iid].reset_index(drop=True)
                         for iid in dev_s["Issue ID"].drop_duplicates().tolist()}
    tst_rows_by_issue = {iid: tst_s[tst_s["Issue ID"]==iid].reset_index(drop=True)
                         for iid in tst_s["Issue ID"].drop_duplicates().tolist()}
    true_by_issue_dev = dev_s[dev_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    true_by_issue_tst = tst_s[tst_s["Output"]==1].groupby("Issue ID")["Commit ID"].apply(set).to_dict()
    issues_dev = sorted(true_by_issue_dev.keys())
    issues_tst = sorted(true_by_issue_tst.keys())

    def metrics_for_issue(pred_set, true_set):
        inter = len(pred_set & true_set)
        p = inter / max(len(pred_set), 1)
        r = inter / max(len(true_set), 1)
        f1 = (2*inter) / max(len(pred_set)+len(true_set), 1)
        return dict(Precision=p, Recall=r, F1=f1)

    def aggregate(dfm):
        return dfm["AllCorrect"].mean() if TUNE_OBJECTIVE.lower()=="allcorrect" else dfm["F1"].mean()

    def apply_cap(chosen_df):
        if not USE_CAP or len(chosen_df) == 0:
            return chosen_df
        top = float(chosen_df["score"].iloc[0])
        return chosen_df[chosen_df["score"] >= MAX_RELATIVE * top]

    ranked_dev = {iid: score_pool(df) for iid, df in dev_rows_by_issue.items()}
    ranked_tst = {iid: score_pool(df) for iid, df in tst_rows_by_issue.items()}

    taus_mm = [x/100 for x in range(10, 96, 2)]
    gammas  = [x/100 for x in range(30, 96, 2)]
    best_abs_mm = (-1, None)
    for t in taus_mm:
        rows = []
        for iid in issues_dev:
            ranked = ranked_dev[iid]
            chosen = apply_cap(ranked[ranked["score_mm"] >= float(t)].copy())
            pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_dev[iid]
            m = metrics_for_issue(pred, true); m["AllCorrect"] = int(pred==true)
            rows.append(m)
        sc = aggregate(pd.DataFrame(rows))
        if sc > best_abs_mm[0]: best_abs_mm = (sc, t)

    best_rel = (-1, None)
    for g in gammas:
        rows = []
        for iid in issues_dev:
            ranked = ranked_dev[iid]
            if len(ranked)==0:
                rows.append(dict(Precision=0.0, Recall=0.0, F1=0.0, AllCorrect=0)); continue
            best = float(ranked["score"].iloc[0])
            chosen = apply_cap(ranked[ranked["score"] >= float(g)*best].copy())
            pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_dev[iid]
            m = metrics_for_issue(pred, true); m["AllCorrect"] = int(pred==true)
            rows.append(m)
        sc = aggregate(pd.DataFrame(rows))
        if sc > best_rel[0]: best_rel = (sc, g)

    rowsABS = []
    for iid in issues_tst:
        ranked = ranked_tst[iid]
        chosen = apply_cap(ranked[ranked["score_mm"] >= float(best_abs_mm[1])].copy())
        pred = set(chosen["Commit ID"].tolist()); true = true_by_issue_tst[iid]
        rowsABS.append(metrics_for_issue(pred, true))
    res_absmm = pd.DataFrame(rowsABS)

    rowsREL = []
    for iid in issues_tst:
        init_issue_state(iid)
        pool_orig = ranked_tst[iid].copy()
        if len(pool_orig) == 0:
            rowsREL.append(dict(Precision=0.0, Recall=0.0, F1=0.0)); continue
        best0 = float(pool_orig["score"].iloc[0])
        accepted = []; pool = pool_orig.copy()
        while len(pool):
            top = pool.iloc[0]
            if float(top["score"]) >= best_rel[1] * best0:
                cid = top["Commit ID"]; accepted.append(cid)
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
        pred = set(accepted); true = true_by_issue_tst[iid]
        rowsREL.append(metrics_for_issue(pred, true))
    res_rel = pd.DataFrame(rowsREL)


    def pct3(df): return macro_percent(df)
    pK, rK, fK       = pct3(res_knownK)
    pABS, rABS, fABS = pct3(res_absmm)
    pREL, rREL, fREL = pct3(res_rel)

    fold_sum = pd.DataFrame(
        [["Known-K", pK, rK, fK],
         ["No-K (ABS-mm)", pABS, rABS, fABS],
         ["No-K (REL)",    pREL, rREL, fREL]],
        columns=["Setting","Precision","Recall","F1"]
    )
    return fold_sum


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
        CACHE_DIR = DS_OUT / "_cache"


        tokenizer, model_sem = load_sem_model()

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_summaries = []

        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(issue_ids), start=1):
            fold_dir = DS_OUT / f"fold{fold_idx}"
            train_ids = set([issue_ids[i] for i in tr_idx])
            test_ids  = set([issue_ids[i] for i in te_idx])

            log.info(f"[{ds_name}] ===== CV fold{fold_idx}: train_issues={len(train_ids)}, test_issues={len(test_ids)} =====")
            fold_sum = run_one_fold(ds_name, df, train_ids, test_ids, fold_dir, tokenizer, model_sem, CACHE_DIR)


            fold_csv = DS_OUT / f"summary_fold{fold_idx}.csv"
            fold_sum.to_csv(fold_csv, index=False)
            fold_summaries.append(fold_sum.assign(Fold=fold_idx))


        all_summ = pd.concat(fold_summaries, ignore_index=True)
        setting_order = ["Known-K", "No-K (ABS-mm)", "No-K (REL)"]

        mean_df = (all_summ
                   .groupby("Setting")[["Precision","Recall","F1"]]
                   .mean()
                   .reindex(setting_order)          
                   .reset_index()
                   .round(2))


        mean_csv = DS_OUT / "summary_k5_mean.csv"
        mean_df.to_csv(mean_csv, index=False)


        mean_df_with_ds = mean_df.copy()
        mean_df_with_ds.insert(0, "Dataset", ds_name)
        all_means.append(mean_df_with_ds)

        print(f"[DONE] {ds_name}: wrote 5 per-fold summaries + summary_k5_mean.csv in {DS_OUT}")


    if all_means:
        master_df = pd.concat(all_means, ignore_index=True)
        master_out = OUT_ROOT / "summary_k5_mean_all_datasets.csv"
        master_df.to_csv(master_out, index=False)
        print(f"[MASTER] Wrote stacked means to: {master_out}")
