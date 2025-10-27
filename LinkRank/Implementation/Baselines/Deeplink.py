
import os, time, logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk


dataset_paths = [
    "Add your file path here",

]
output_dir = "Add path for the directory to save results"

Path(output_dir).mkdir(parents=True, exist_ok=True)

SEED        = 42
TEST_RATIO  = 0.20
VECTOR_SIZE = 100
HIDDEN_SIZE = 64
BATCH_SIZE  = 32
EPOCHS      = 5
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 50
THRESH      = 0.5  


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("DeepLink-LSTM")
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

def split_by_issue_id(df: pd.DataFrame, seed=SEED, test_ratio=TEST_RATIO):
    rng = np.random.default_rng(seed)
    issue_ids = df["Issue ID"].drop_duplicates().tolist()
    rng.shuffle(issue_ids)
    n = len(issue_ids)
    n_train = int((1.0 - test_ratio) * n)
    train_ids = set(issue_ids[:n_train])
    test_ids  = set(issue_ids[n_train:])
    return df[df["Issue ID"].isin(train_ids)].copy(), df[df["Issue ID"].isin(test_ids)].copy()


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
        return self.sigmoid(combined)  # (B,1) in [0,1]


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
    preds = []
    for x1, x2, x3, _ in loader:
        x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
        out = model(x1, x2, x3).squeeze(1).cpu().numpy()
        preds.extend((out >= thresh).astype(int).tolist())
    out_df = df.copy()
    out_df["Predicted"] = preds
    return out_df


def per_issue_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iid, g in test_df.groupby("Issue ID"):
        true_set = set(g.loc[g["Output"] == 1, "Commit ID"])
        pred_set = set(g.loc[g["Predicted"] == 1, "Commit ID"])
        inter = len(true_set & pred_set)
        upred = len(pred_set); utrue = len(true_set)
        union = len(true_set | pred_set)
        precision = inter / upred if upred > 0 else 0.0
        recall    = inter / utrue if utrue > 0 else 0.0
        f1        = (2 * inter) / (upred + utrue) if (upred + utrue) > 0 else 0.0
        jaccard   = inter / union if union > 0 else 0.0
        rows.append({
            "Issue ID": iid,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Jaccard": jaccard
        })
    return pd.DataFrame(rows)

def pct2(x: float) -> float:
    return round(float(x) * 100.0, 2)


summaries, timings = [], []

for dataset_path in dataset_paths:
    try:
        ds_t0  = time.perf_counter()
        dsname = os.path.splitext(os.path.basename(dataset_path))[0]
        log.info(f"▶ Processing {dsname}")

        df = pd.read_csv(dataset_path)
        check_columns(df, dataset_path)

        df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).clip(0,1).astype(int)

        for c in ["Title","Description","Labels","Comments","Message","Diff Summary","File Changes"]:
            df[c] = df[c].fillna("")


        train_df, test_df = split_by_issue_id(df, seed=SEED, test_ratio=TEST_RATIO)

        t_train0 = time.perf_counter()
        train_triples = [
            (
                tokenize_and_embed(r["Message"]),
                tokenize_and_embed(" ".join([r["Title"], r["Description"], r["Labels"], r["Comments"]])),
                tokenize_and_embed(" ".join([r["Diff Summary"], r["File Changes"]])),
                int(r["Output"])
            )
            for _, r in train_df.iterrows()
        ]
        train_loader = DataLoader(TripleTextDataset(train_triples),
                                  batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

        model = DeepLinkModel(HIDDEN_SIZE).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for ep in range(EPOCHS):
            loss = train_epoch(model, train_loader, criterion, optimizer)
            log.info(f"[{dsname}] Epoch {ep+1}/{EPOCHS} – loss={loss:.4f}")
        train_sec = time.perf_counter() - t_train0


        t_test0 = time.perf_counter()
        test_triples = [
            (
                tokenize_and_embed(r["Message"]),
                tokenize_and_embed(" ".join([r["Title"], r["Description"], r["Labels"], r["Comments"]])),
                tokenize_and_embed(" ".join([r["Diff Summary"], r["File Changes"]])),
                int(r["Output"])
            )
            for _, r in test_df.iterrows()
        ]
        test_loader = DataLoader(TripleTextDataset(test_triples),
                                 batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        test_pred = predict(model, test_loader, test_df, thresh=THRESH)
        test_sec  = time.perf_counter() - t_test0


        per_issue = per_issue_metrics(test_pred)
        macro = {
            "Dataset": dsname,
            "Issues": len(per_issue),
            "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
            "Mean_Recall":    per_issue["Recall"].mean()    if len(per_issue) else 0.0,
            "Mean_F1":        per_issue["F1"].mean()        if len(per_issue) else 0.0,
            "Mean_Jaccard":   per_issue["Jaccard"].mean()   if len(per_issue) else 0.0,
        }
        summaries.append(macro)


        pred_out = os.path.join(output_dir, f"{dsname}_test_predictions.csv")
        test_pred.to_csv(pred_out, index=False)

        per_issue_pct = per_issue.copy()
        for c in ["Precision","Recall","F1","Jaccard"]:
            per_issue_pct[c] = per_issue_pct[c].map(pct2)
        per_issue_out = os.path.join(output_dir, f"{dsname}_per_issue_metrics.csv")
        per_issue_pct.to_csv(per_issue_out, index=False)


        timings.append({
            "Dataset": dsname,
            "TrainSeconds": round(train_sec, 4),
            "TestSeconds":  round(test_sec, 4),
            "TotalSeconds": round(time.perf_counter() - ds_t0, 4),
        })
        log.info(f"[{dsname}] Time: train={train_sec:.2f}s | test={test_sec:.2f}s")

    except Exception as e:
        log.error(f" Error on {dataset_path}: {e}")


if summaries:
    summary_df = pd.DataFrame(summaries)
    for col in ["Mean_Precision","Mean_Recall","Mean_F1","Mean_Jaccard"]:
        summary_df[col] = (summary_df[col] * 100.0).round(2)
    summary_out = os.path.join(output_dir, "DeepLink_LSTM_summary.csv")
    summary_df.to_csv(summary_out, index=False)
    print("\n=== Macro Summary Across Datasets (values in %) ===")
    print(summary_df.to_string(index=False))
    print(f"\n✔ Summary saved → {summary_out}")
else:
    print("No summaries produced.")


if timings:
    tdf = pd.DataFrame(timings)
    avg = {
        "Dataset": "Average",
        "TrainSeconds": round(tdf["TrainSeconds"].mean(), 4),
        "TestSeconds":  round(tdf["TestSeconds"].mean(), 4),
        "TotalSeconds": round(tdf["TotalSeconds"].mean(), 4),
    }
    tdf = pd.concat([tdf, pd.DataFrame([avg])], ignore_index=True)
    t_out = os.path.join(output_dir, "DeepLink_LSTM_timings.csv")
    tdf.to_csv(t_out, index=False)
    print("\n=== Train/Test Timings (seconds) — DeepLink (LSTM) ===")
    print(tdf.to_string(index=False))
    print(f"\n✔ Timings saved → {t_out}")

