import os, time, logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
import nltk


train_paths = [
    "Add path of the training files here",
    
]
test_paths = [
    "Add path of the testing files here",

]

output_dir = "Add your output directory here"

Path(output_dir).mkdir(parents=True, exist_ok=True)

SEED          = 42
VECTOR_SIZE   = 100
HIDDEN_SIZE   = 64
BATCH_SIZE    = 32
EPOCHS        = 5
LEARNING_RATE = 1e-3
MAX_SEQ_LEN   = 50
THRESH        = 0.5  

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("DeepLink-LSTM-XProject")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")


np.random.seed(SEED)
torch.manual_seed(SEED)


try:
    nltk.download('punkt', quiet=True)
except Exception:
    pass


REQUIRED_COLS = [
    "Repository","Issue ID","Issue Date","Title","Description","Labels","Comments",
    "Commit ID","Commit Date","Message","Diff Summary","File Changes","Full Diff","Output"
]

def check_columns(df, path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {missing}")


_rng = np.random.default_rng(SEED)

def random_token_embedding(tokens, vector_size=VECTOR_SIZE):
    return [_rng.uniform(-1, 1, vector_size) for _ in tokens]

def tokenize_and_embed(text):
    tokens = word_tokenize(str(text).lower())
    if not tokens:
        return [np.zeros(VECTOR_SIZE, dtype=np.float32)]
    return random_token_embedding(tokens, VECTOR_SIZE)


class TripleTextDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        x1, x2, x3, y = self.triples[idx]
        return (torch.tensor(x1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32),
                torch.tensor(x3, dtype=torch.float32),
                torch.tensor(y,  dtype=torch.float32))

def collate_fn(batch, max_len=MAX_SEQ_LEN):
    x1, x2, x3, y = zip(*batch)
    x1 = nn.utils.rnn.pad_sequence([torch.tensor(seq[:max_len]) for seq in x1], batch_first=True)
    x2 = nn.utils.rnn.pad_sequence([torch.tensor(seq[:max_len]) for seq in x2], batch_first=True)
    x3 = nn.utils.rnn.pad_sequence([torch.tensor(seq[:max_len]) for seq in x3], batch_first=True)
    y  = torch.tensor(y, dtype=torch.float32)
    return x1, x2, x3, y


class DeepLinkModel(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm1 = nn.LSTM(VECTOR_SIZE, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(VECTOR_SIZE, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(VECTOR_SIZE, hidden_size, batch_first=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2, x3):
        _, (h1, _) = self.lstm1(x1)
        _, (h2, _) = self.lstm2(x2)
        _, (h3, _) = self.lstm3(x3)
        score12 = torch.cosine_similarity(h1[-1], h2[-1], dim=1)
        score13 = torch.cosine_similarity(h1[-1], h3[-1], dim=1)
        combined = torch.maximum(score12, score13).unsqueeze(1)
        return self.sigmoid(combined)  


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for x1, x2, x3, y in loader:
        x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
        y = y.unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        out = model(x1, x2, x3)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def predict(model, loader, df, thresh=THRESH):
    model.eval()
    preds, probs = [], []
    for x1, x2, x3, _ in loader:
        x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
        out = model(x1, x2, x3).squeeze(1).cpu().numpy()
        probs.extend(out.tolist())
        preds.extend((out >= thresh).astype(int).tolist())
    out_df = df.copy()
    out_df["Predicted_Prob"] = probs
    out_df["Predicted"] = preds
    return out_df


def per_issue_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iid, g in test_df.groupby("Issue ID"):
        true_set = set(g.loc[g["Output"] == 1, "Commit ID"])
        pred_set = set(g.loc[g["Predicted"] == 1, "Commit ID"])
        inter = len(true_set & pred_set)
        upred = len(pred_set); utrue = len(true_set)
        precision = inter / upred if upred > 0 else 0.0
        recall    = inter / utrue if utrue > 0 else 0.0
        f1        = (2 * inter) / (upred + utrue) if (upred + utrue) > 0 else 0.0
        rows.append({
            "Issue ID": iid,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })
    return pd.DataFrame(rows)

def pct2(x: float) -> float:
    return round(float(x) * 100.0, 2)


def build_triples_from_df(df: pd.DataFrame):

    for c in ["Title","Description","Labels","Comments","Message","Diff Summary","File Changes"]:
        df[c] = df[c].fillna("")
    triples = [
        (
            tokenize_and_embed(r["Message"]),
            tokenize_and_embed(" ".join([r["Title"], r["Description"], r["Labels"], r["Comments"]])),
            tokenize_and_embed(" ".join([r["Diff Summary"], r["File Changes"]])),
            int(r["Output"])
        )
        for _, r in df.iterrows()
    ]
    return triples

def train_on_train_paths(train_paths):
    t0 = time.perf_counter()
    all_triples = []
    total_rows = 0
    for p in train_paths:
        df = pd.read_csv(p)
        check_columns(df, p)
        df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).clip(0,1).astype(int)
        triples = build_triples_from_df(df)
        all_triples.extend(triples)
        total_rows += len(df)
        log.info(f"[TRAIN LOAD] {Path(p).stem}: {len(df):,} rows")

    log.info(f"[TRAIN] Total rows across train_paths: {total_rows:,}")
    train_loader = DataLoader(TripleTextDataset(all_triples),
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = DeepLinkModel(HIDDEN_SIZE).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        log.info(f"[TRAIN] Epoch {ep+1}/{EPOCHS} – loss={loss:.4f}")

    train_secs = time.perf_counter() - t0
    log.info(f"[TRAIN] Completed in {train_secs:.2f}s")
    return model, train_secs


def evaluate_on_test_csv(model, test_csv):
    t0 = time.perf_counter()
    dsname = Path(test_csv).stem
    log.info(f"▶ Testing {dsname}")

    df = pd.read_csv(test_csv)
    check_columns(df, test_csv)
    df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).clip(0,1).astype(int)


    triples = build_triples_from_df(df)
    loader  = DataLoader(TripleTextDataset(triples),
                         batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    pred_df = predict(model, loader, df, thresh=THRESH)

    per_issue = per_issue_metrics(pred_df)
    macro = {
        "Dataset": dsname,
        "Issues": len(per_issue),
        "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
        "Mean_Recall":    per_issue["Recall"].mean()    if len(per_issue) else 0.0,
        "Mean_F1":        per_issue["F1"].mean()        if len(per_issue) else 0.0,
    }

    pred_out = os.path.join(output_dir, f"{dsname}_test_predictions.csv")
    pred_df.to_csv(pred_out, index=False)

    per_issue_pct = per_issue.copy()
    for c in ["Precision","Recall","F1"]:
        per_issue_pct[c] = per_issue_pct[c].map(pct2)
    per_issue_out = os.path.join(output_dir, f"{dsname}_per_issue_metrics.csv")
    per_issue_pct.to_csv(per_issue_out, index=False)

    secs = time.perf_counter() - t0
    log.info(f"[{dsname}] P={macro['Mean_Precision']:.4f} R={macro['Mean_Recall']:.4f} F1={macro['Mean_F1']:.4f} ({secs:.2f}s)")
    log.info(f"[{dsname}] Saved per-issue metrics → {per_issue_out}")
    log.info(f"[{dsname}] Saved test predictions  → {pred_out}")

    return macro, secs

if __name__ == "__main__":
    summaries = []
    timings   = []

    try:
        model, train_seconds = train_on_train_paths(train_paths)
    except Exception as e:
        log.error(f" Training failed: {e}")
        raise

    for p in test_paths:
        try:
            macro, test_secs = evaluate_on_test_csv(model, p)
            summaries.append(macro)
            timings.append({"Dataset": macro["Dataset"], "TestSeconds": round(test_secs, 2)})
        except Exception as e:
            log.error(f" Failed on {p}: {e}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        for col in ["Mean_Precision","Mean_Recall","Mean_F1"]:
            summary_df[col] = (summary_df[col] * 100.0).round(2)
        summary_csv = os.path.join(output_dir, "DeepLink_LSTM_cross_project_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        print("\n=== Cross-Project Macro Summary (values in %) ===")
        print(summary_df.to_string(index=False))
        log.info(f"✔ Summary saved → {summary_csv}")
    else:
        log.warning("No summaries produced.")


    if timings:
        tdf = pd.DataFrame(timings)
        tdf.loc[len(tdf)] = {"Dataset": "TRAIN(all)", "TestSeconds": round(train_seconds, 2)}
        timings_csv = os.path.join(output_dir, "DeepLink_LSTM_cross_project_timings.csv")
        tdf.to_csv(timings_csv, index=False)
        log.info(f"✔ Timings saved → {timings_csv}")
