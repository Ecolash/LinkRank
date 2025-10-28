# LinkRank ,  Data collection

This folder contains the data collection helper used to build the one-to-many
issue--commit datasets used by the LinkRank experiments.

## Pre-built datasets
We publish the processed datasets on Zenodo: https://zenodo.org/records/17310871

## Using the Dataset.py to create a Dataset (quick start)
1. Install dependencies (recommended in a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install requests tqdm
```

2. Configure the collector in `Dataset.py`:
	- Set `GITHUB_TOKEN` to a personal access token with repo access
	- Edit `REPOSITORIES` to list dictionaries of `{"owner": "ORG", "name": "repo"}`
	- Tune `ISSUE_COUNT`, `MIN_COMMITS`, `MAX_COMMITS`, and output directories `DATA_DIR` / `LOGS_DIR`.

3. Run the collector (from repository root):

```bash
python LinkRank/Data/Dataset.py
```

### Output

- CSV files are written to the configured `DATA_DIR` (defaults to `/kaggle/working/data` in the script). 
- Filenames include the repository and the target issue count.
- Logs are written to `LOGS_DIR` (defaults to `/kaggle/working/logs`).
------
### Notes

- The collector respects GitHub rate limits but may still pause between requests ,  be patient for large runs.
- If you plan to run at scale, consider using multiple tokens and sharding repositories, or export data incrementally.

