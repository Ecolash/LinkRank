
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import RobertaModel, RobertaTokenizer


dataset_paths = [
    "Add your file path here",

]
output_dir = "Add path for the directory to save results"
Path(output_dir).mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_RATIO = 0.20
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-5
WD = 1e-5
MAX_SEQ_LEN = 128
ACCUM_STEPS = 4

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
        label = -1 if self.labels is None else self.labels[idx]
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
        return logits  # raw logits


def train_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps=ACCUM_STEPS):
    model.train()
    running = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, unit="batch", leave=False)
    for i, (ii, im, ci, cm, y) in enumerate(pbar):
        ii, im, ci, cm = ii.to(device), im.to(device), ci.to(device), cm.to(device)
        y = y.to(device).float().view(-1, 1)

        if USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(ii, im, ci, cm)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(ii, im, ci, cm)
            loss = criterion(logits, y)
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        running += loss.item()
        pbar.set_postfix(loss=f"{running/(i+1):.4f}")

    if (i + 1) % accumulation_steps != 0:
        if USE_AMP:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

def predict(model, loader, df):
    model.eval()
    preds = []
    with torch.no_grad():
        for ii, im, ci, cm, _ in tqdm(loader, unit="batch", leave=False):
            ii, im, ci, cm = ii.to(device), im.to(device), ci.to(device), cm.to(device)
            logits = model(ii, im, ci, cm)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            pred = (probs > 0.5).astype(int)
            preds.extend(pred.tolist())
    out = df.copy()
    out["Predicted"] = preds
    return out


REQUIRED_COLS = [
    "Issue ID","Issue Date","Title","Description","Labels","Comments",
    "Commit ID","Commit Date","Message","Diff Summary","File Changes","Full Diff","Output"
]

def check_columns(df, path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{path}] Missing required columns: {missing}")

def split_by_issue_id(df, seed=SEED, test_ratio=TEST_RATIO):
    rng = np.random.default_rng(seed)
    issue_ids = df["Issue ID"].drop_duplicates().tolist()
    rng.shuffle(issue_ids)
    n = len(issue_ids)
    n_train = int((1.0 - test_ratio) * n)
    train_ids = set(issue_ids[:n_train])
    test_ids  = set(issue_ids[n_train:])
    return df[df["Issue ID"].isin(train_ids)].copy(), df[df["Issue ID"].isin(test_ids)].copy()

def per_issue_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iid, g in test_df.groupby("Issue ID"):
        true_set = set(g.loc[g["Output"] == 1, "Commit ID"])
        pred_set = set(g.loc[g["Predicted"] == 1, "Commit ID"])
        inter = len(true_set & pred_set)
        upred = len(pred_set)
        utrue = len(true_set)
        union = len(true_set | pred_set)
        precision = inter / upred if upred > 0 else 0.0
        recall    = inter / utrue if utrue > 0 else 0.0
        f1        = (2 * inter) / (upred + utrue) if (upred + utrue) > 0 else 0.0
        jaccard   = inter / union if union > 0 else 0.0
        rows.append({
            "Issue ID": iid,
            "True_Pos_Commits": utrue,
            "Pred_Pos_Commits": upred,
            "Inter": inter,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Jaccard": jaccard
        })
    return pd.DataFrame(rows)

def pct2(x: float) -> float:
    return round(float(x) * 100.0, 2)


summaries = []
for dataset_path in dataset_paths:
    try:
        ds_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"\n▶ Processing dataset: {ds_name}")

        df = pd.read_csv(dataset_path)
        check_columns(df, dataset_path)
        df["Output"] = df["Output"].astype(int)


        train_df, test_df = split_by_issue_id(df, seed=SEED, test_ratio=TEST_RATIO)

        for col in ["Title","Description","Message","File Changes"]:
            train_df[col] = train_df[col].fillna("")
            test_df[col]  = test_df[col].fillna("")

        train_issue_text  = (train_df["Title"] + " " + train_df["Description"]).astype(str)
        train_commit_text = (train_df["Message"] + " " + train_df["File Changes"]).astype(str)

        test_issue_text   = (test_df["Title"] + " " + test_df["Description"]).astype(str)
        test_commit_text  = (test_df["Message"] + " " + test_df["File Changes"]).astype(str)

        # ---- Tokenize
        tri_ids, tri_mask = batch_tokenize_texts(train_issue_text, MAX_SEQ_LEN)
        trc_ids, trc_mask = batch_tokenize_texts(train_commit_text, MAX_SEQ_LEN)

        tei_ids, tei_mask = batch_tokenize_texts(test_issue_text, MAX_SEQ_LEN)
        tec_ids, tec_mask = batch_tokenize_texts(test_commit_text, MAX_SEQ_LEN)

        # ---- Datasets & loaders
        train_dataset = IssueCommitDataset(
            list(zip(tri_ids, tri_mask)),
            list(zip(trc_ids, trc_mask)),
            train_df["Output"].values
        )
        test_dataset = IssueCommitDataset(
            list(zip(tei_ids, tei_mask)),
            list(zip(tec_ids, tec_mask)),
            test_df["Output"].values
        )

        pin_mem = device.type == "cuda"
        num_workers = 4 if device.type == "cuda" else 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_mem)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_mem)


        model = EALink().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer, scaler, accumulation_steps=ACCUM_STEPS)
            print(f"Epoch {epoch+1}/{EPOCHS} done.")


        test_df_pred = predict(model, test_loader, test_df)


        per_issue = per_issue_metrics(test_df_pred)
        macro = {
            "Dataset": ds_name,
            "Issues": len(per_issue),
            "Mean_Precision": per_issue["Precision"].mean() if len(per_issue) else 0.0,
            "Mean_Recall":    per_issue["Recall"].mean() if len(per_issue) else 0.0,
            "Mean_F1":        per_issue["F1"].mean() if len(per_issue) else 0.0,
            "Mean_Jaccard":   per_issue["Jaccard"].mean() if len(per_issue) else 0.0,
        }
        summaries.append(macro)


        pred_out = os.path.join(output_dir, f"{ds_name}_test_predictions.csv")
        test_df_pred.to_csv(pred_out, index=False)
        print(f" Predictions saved → {pred_out}")


        per_issue_pct = per_issue.copy()
        for c in ["Precision","Recall","F1","Jaccard"]:
            per_issue_pct[c] = per_issue_pct[c].map(pct2)
        per_issue_out = os.path.join(output_dir, f"{ds_name}_per_issue_metrics.csv")
        per_issue_pct.to_csv(per_issue_out, index=False)
        print(f" Per-issue metrics saved → {per_issue_out}")

    except Exception as e:
        print(f" Error processing {dataset_path}: {e}")


if summaries:
    summary_df = pd.DataFrame(summaries)
    for col in ["Mean_Precision","Mean_Recall","Mean_F1","Mean_Jaccard"]:
        summary_df[col] = (summary_df[col] * 100.0).round(2)
    summary_out = os.path.join(output_dir, "EALink_summary.csv")
    summary_df.to_csv(summary_out, index=False)
    print("\n=== Macro Summary Across Datasets (values in %) ===")
    print(summary_df.to_string(index=False))
    print(f"\n✔ All-dataset summary saved → {summary_out}")
else:
    print("No summaries produced.")

