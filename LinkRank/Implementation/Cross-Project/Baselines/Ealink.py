import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer


train_paths = [
    "Add path of the training files here",
    
]
test_paths = [
    "Add path of the testing files here",

]

output_dir = "Add your output directory here"
Path(output_dir).mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5
WD = 1e-5
MAX_SEQ_LEN = 128
ACCUM_STEPS = 2
PRED_THRESH = 0.50
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Using device: {device} | AMP: {USE_AMP}")


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

def batch_tokenize_texts(texts, max_length=MAX_SEQ_LEN):
    enc = tokenizer(
        list(texts), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


class IssueCommitDataset(Dataset):
    def __init__(self, issue_tokens, commit_tokens, labels=None):
        self.issue_tokens = issue_tokens  # list of (input_ids, attn_mask)
        self.commit_tokens = commit_tokens
        self.labels = labels

    def __len__(self):
        return len(self.issue_tokens)

    def __getitem__(self, idx):
        issue_input_ids, issue_attn = self.issue_tokens[idx]
        commit_input_ids, commit_attn = self.commit_tokens[idx]
        if self.labels is None:
            label = -1
        else:
            label = self.labels[idx]
        return issue_input_ids, issue_attn, commit_input_ids, commit_attn, label

class EALink(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.fc1 = nn.Linear(768 * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, issue_input, issue_mask, commit_input, commit_mask):
        issue_out = self.codebert(input_ids=issue_input, attention_mask=issue_mask).pooler_output
        commit_out = self.codebert(input_ids=commit_input, attention_mask=commit_mask).pooler_output
        x = torch.cat([issue_out, commit_out], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(self.dropout(x))
        logits = self.fc2(x)
        return logits  


REQUIRED_COLS = [
    "Issue ID","Issue Date","Title","Description","Labels","Comments",
    "Commit ID","Commit Date","Message","Diff Summary","File Changes","Full Diff","Output"
]

def check_columns(df, path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {missing}")

def per_issue_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    """P/R/F1 per issue (set-based)."""
    rows = []
    for iid, g in test_df.groupby("Issue ID"):
        true_set = set(g.loc[g["Output"] == 1, "Commit ID"])
        pred_set = set(g.loc[g["Predicted"] == 1, "Commit ID"])
        inter = len(true_set & pred_set)
        upred = len(pred_set); utrue = len(true_set)
        precision = inter / upred if upred > 0 else 0.0
        recall    = inter / utrue if utrue > 0 else 0.0
        f1        = (2 * inter) / (upred + utrue) if (upred + utrue) > 0 else 0.0
        rows.append({"Issue ID": iid, "Precision": precision, "Recall": recall, "F1": f1})
    return pd.DataFrame(rows)

def pct2(x: float) -> float:
    return round(float(x) * 100.0, 2)

def build_views(df: pd.DataFrame):

    for col in ["Title","Description","Message","File Changes"]:
        df[col] = df[col].fillna("")
    issue_text  = (df["Title"] + " " + df["Description"]).astype(str)
    commit_text = (df["Message"] + " " + df["File Changes"]).astype(str)
    return issue_text, commit_text


def train_on_train_paths(train_paths):
    all_issue_texts = []
    all_commit_texts = []
    all_labels = []

    for p in train_paths:
        df = pd.read_csv(p)
        check_columns(df, p)
        df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).clip(0,1).astype(int)
        itxt, ctxt = build_views(df)
        all_issue_texts.extend(itxt.tolist())
        all_commit_texts.extend(ctxt.tolist())
        all_labels.extend(df["Output"].tolist())
        print(f"[TRAIN LOAD] {Path(p).stem}: {len(df):,} rows")

    print(f"[TRAIN] Total rows across train_paths: {len(all_labels):,}")


    i_ids, i_mask = batch_tokenize_texts(all_issue_texts, MAX_SEQ_LEN)
    c_ids, c_mask = batch_tokenize_texts(all_commit_texts, MAX_SEQ_LEN)
    train_ds = IssueCommitDataset(
        list(zip(i_ids, i_mask)),
        list(zip(c_ids, c_mask)),
        np.asarray(all_labels, dtype=np.int64)
    )
    pin_mem = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)

    model = EALink().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    model.train()
    for epoch in range(EPOCHS):
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, unit="batch", leave=False, desc=f"Train Epoch {epoch+1}/{EPOCHS}")
        for i, (ii, im, ci, cm, y) in enumerate(pbar):
            ii, im, ci, cm = ii.to(device), im.to(device), ci.to(device), cm.to(device)
            y = y.float().view(-1, 1).to(device)
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(ii, im, ci, cm)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                if (i + 1) % ACCUM_STEPS == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                logits = model(ii, im, ci, cm)
                loss = criterion(logits, y)
                loss.backward()
                if (i + 1) % ACCUM_STEPS == 0:
                    optimizer.step(); optimizer.zero_grad()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(i+1):.4f}")
        if (i + 1) % ACCUM_STEPS != 0:
            if USE_AMP:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        print(f"[TRAIN] Epoch {epoch+1}/{EPOCHS} avg_loss={running/max(1,i+1):.4f}")

    return model

@torch.no_grad()
def evaluate_on_test_csv(model, test_csv):
    dsname = Path(test_csv).stem
    print(f"▶ Testing {dsname}")
    df = pd.read_csv(test_csv)
    check_columns(df, test_csv)
    df["Output"] = pd.to_numeric(df["Output"], errors="coerce").fillna(0).clip(0,1).astype(int)

    itxt, ctxt = build_views(df)
    i_ids, i_mask = batch_tokenize_texts(itxt, MAX_SEQ_LEN)
    c_ids, c_mask = batch_tokenize_texts(ctxt, MAX_SEQ_LEN)

    pin_mem = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0
    ds = IssueCommitDataset(list(zip(i_ids, i_mask)), list(zip(c_ids, c_mask)), df["Output"].values)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_mem)

    model.eval()
    probs = []
    for ii, im, ci, cm, _ in tqdm(loader, unit="batch", leave=False, desc=f"Predict {dsname}"):
        ii, im, ci, cm = ii.to(device), im.to(device), ci.to(device), cm.to(device)
        logits = model(ii, im, ci, cm)
        p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        probs.extend(p.tolist())

    out = df.copy()
    out["Predicted_Prob"] = probs
    out["Predicted"] = (out["Predicted_Prob"] >= PRED_THRESH).astype(int)

    per_issue = per_issue_metrics(out)
    macro = {
        "Dataset": dsname,
        "Issues": int(per_issue["Issue ID"].nunique()) if len(per_issue) else 0,
        "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
        "Mean_Recall":    per_issue["Recall"].mean() if len(per_issue) else 0.0,
        "Mean_F1":        per_issue["F1"].mean() if len(per_issue) else 0.0,
    }

    out.to_csv(os.path.join(output_dir, f"{dsname}_test_predictions.csv"), index=False)
    per_issue_pct = per_issue.copy()
    for c in ["Precision","Recall","F1"]:
        per_issue_pct[c] = per_issue_pct[c].map(pct2)
    per_issue_pct.to_csv(os.path.join(output_dir, f"{dsname}_per_issue_metrics.csv"), index=False)

    print(f"[{dsname}] P={macro['Mean_Precision']:.4f} R={macro['Mean_Recall']:.4f} F1={macro['Mean_F1']:.4f}")
    return macro

if __name__ == "__main__":

    model = train_on_train_paths(train_paths)


    summaries = []
    for p in test_paths:
        try:
            summaries.append(evaluate_on_test_csv(model, p))
        except Exception as e:
            print(f" Failed on {p}: {e}")


    if summaries:
        summary_df = pd.DataFrame(summaries)
        for col in ["Mean_Precision","Mean_Recall","Mean_F1"]:
            summary_df[col] = (summary_df[col] * 100.0).round(2)
        summary_csv = os.path.join(output_dir, "EALink_CodeBERT_cross_project_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print("\n=== Cross-Project Macro Summary (values in %) ===")
        print(summary_df.to_string(index=False))
        print(f"\n✔ Summary saved → {summary_csv}")
    else:
        print("No summaries produced.")


