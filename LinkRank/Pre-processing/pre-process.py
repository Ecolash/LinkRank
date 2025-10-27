import pandas as pd
import re
from pathlib import Path
import numpy as np


in_path = Path("Add your path file here") 
out_path = Path("Add your output path file here")


df = pd.read_csv(in_path)

issue_cols = ["Title", "Description", "Comments"]
commit_cols = ["Message", "Diff Summary", "File Changes", "Full Diff"]


for c in issue_cols + commit_cols:
    if c not in df.columns:
        df[c] = ""


URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HASH_RE  = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
MD_LINK  = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")  
HTML_TAG = re.compile(r"<[^>]+>")
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)  

def normalize_ws(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)       
    s = re.sub(r"[ \t]+\n", "\n", s)       
    s = re.sub(r"\n[ \t]+", "\n", s)        
    s = re.sub(r"[ \t]{2,}", " ", s)        
    return s.strip()

def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = CODE_FENCE.sub(" ", s)             
    s = MD_LINK.sub(r"\1", s)               
    s = HTML_TAG.sub(" ", s)               
    s = URL_RE.sub("<URL>", s)             
    s = EMAIL_RE.sub("<EMAIL>", s)         
    s = HASH_RE.sub("<HASH>", s)            
    s = normalize_ws(s)
    return s

def clean_file_changes(x: str) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return ""
    parts = [p.strip() for p in str(x).split(",") if p.strip()]
    return ", ".join(parts)

def clean_diff(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = URL_RE.sub("<URL>", s)
    s = EMAIL_RE.sub("<EMAIL>", s)
    s = HASH_RE.sub("<HASH>", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{4,}", "\n\n", s)
    return s.strip()


for c in issue_cols:
    df[c] = df[c].apply(clean_text)

df["Message"] = df["Message"].apply(clean_text)
df["Diff Summary"] = df["Diff Summary"].apply(clean_text)
df["File Changes"] = df["File Changes"].apply(clean_file_changes)
df["Full Diff"] = df["Full Diff"].apply(clean_diff)


for c in commit_cols:
    df[c] = df[c].apply(lambda v: "NULL" if (pd.isna(v) or str(v).strip() == "") else v)


drop_mask = df["Issue ID"].isna() | df["Commit ID"].isna()
if drop_mask.any():
    df = df[~drop_mask].reset_index(drop=True)


df.to_csv(out_path, index=False)
print(f" Cleaned file saved to {out_path}")
