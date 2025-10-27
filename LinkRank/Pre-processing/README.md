## Pre-processing for LinkRank

This folder contains a small preprocessing script used to clean and normalize issue and commit data before running the LinkRank experiments.

File: `pre-process.py`

## Purpose

`pre-process.py` reads an input CSV containing issue and commit pairs, applies a set of text cleaning rules, normalizes several fields, drops invalid rows, and writes a cleaned CSV to the specified output path.

Use this script before running experiments or model training so downstream code receives consistent, de-duplicated, and privacy-preserving text fields.

## Python requirements

- Python 3.8+
- pandas
- numpy

Install dependencies with:

```bash
pip install pandas numpy
```


## Configuration

Open `Pre-processing/pre-process.py` and change the two variables near the top:

- `in_path = Path("Add your path file here")` — set this to the path of your input CSV.
- `out_path = Path("Add your output path file here")` — set this to where you want the cleaned CSV saved.

Example (in the script):

```python
from pathlib import Path

in_path = Path("/path/to/raw_issues_commits.csv")
out_path = Path("/path/to/cleaned_issues_commits.csv")
```

## Example usage

1. Edit `in_path` and `out_path` in `pre-process.py`.
2. Run the script from the repository root:

```bash
python Pre-processing/pre-process.py
```

On success you should see:

```
 Cleaned file saved to /path/to/cleaned_issues_commits.csv
```


## Edge cases handled

- Missing columns: script creates empty column(s) when needed.
- Empty or NaN values: converted to `""` or `NULL` depending on column type.
- Rows missing `Issue ID` or `Commit ID` are dropped.

## Suggested improvements

- Accept `in_path` and `out_path` as command-line arguments (argparse) to avoid editing the script.
- Add a small test fixture and unit tests for cleaning functions.
- Support processing compressed CSVs (gzip) and streaming for large files.

## Troubleshooting

- If you see encoding errors, ensure your CSV is UTF-8 or open it with the correct encoding and re-save as UTF-8.
- If required columns are named differently in your dataset, adjust column names before running or modify the script.

## Contact

If you need changes to the preprocessing rules, update the helper functions in `pre-process.py`:
- `clean_text`, `clean_file_changes`, `clean_diff`, and `normalize_ws`.

Thank you — this README should help you get started with cleaning your data for LinkRank experiments.
