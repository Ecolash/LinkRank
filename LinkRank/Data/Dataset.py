"""
Dataset.py (LinkRank)
Utility to collect issue-commit pairs from GitHub for the LinkRank project.

Usage (quick):
    - Install requirements: pip install requests tqdm
    - Edit the GITHUB_TOKEN constant below or export GITHUB_TOKEN as an env var.
    - Adjust REPOSITORIES to list repositories you want to collect from.
    - Optionally change DATA_DIR / LOGS_DIR and other settings.
    - Run: python LinkRank/Data/Dataset.py
    - Check DATA_DIR for the resulting CSV file(s).

Hyperparameters:
    - ISSUE_COUNT: Number of issues to collect per repository.
    - MIN_COMMITS / MAX_COMMITS: Commit count range for PRs to be included.
    - PR_PAGE_SIZE: Number of PRs to fetch per GraphQL query page.

What the script does:
    - Iterates merged pull requests in the specified repo(s):
    - It collects PRs that:
        > reference exactly one issue 
        > have at least MIN_COMMITS 
        > have at most MAX_COMMITS commits.

    - Fetches per-commit details (message, files changed, diffs) via the REST API.
    - Builds true (positive) pairs and constructs matched false (negative) pairs from commits sampled from PRs that don't meet the inclusion criteria.
    - Writes a balanced CSV to DATA_DIR with columns for issue, commit, diffs, and a binary Output column (1 = true, 0 = false).

"""

import os
import sys
import csv
import time
import random
import logging
import requests
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

GITHUB_TOKEN = "GITHUB-PERSONAL-ACCESS-TOKEN"
REPOSITORIES = [
    {"owner": "rust-lang", "name": "rust"},
    # {"owner": "apache", "name": "mxnet"},
    # {"owner": "pytorch", "name": "pytorch"},
]

PR_PAGE_SIZE = 100
ISSUE_COUNT = 5000
MIN_COMMITS = 2
MAX_COMMITS = 20

# Directories for logs and data (Kaggle environment)
LOGS_DIR = "/kaggle/working/logs"
DATA_DIR = "/kaggle/working/data"

GRAPHQL_API_URL = "https://api.github.com/graphql"
REST_API_URL = "https://api.github.com"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json"
}

RATE_LIMIT_BUFFER = 100
DEFAULT_SLEEP_TIME = 1
RATE_LIMIT_RESET_BUFFER = 60

def setup_directories():
    for directory in [LOGS_DIR, DATA_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def setup_logging(repo_owner, repo_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOGS_DIR, f"github_collector_{repo_owner}_{repo_name}_{timestamp}.log")

    logger = logging.getLogger(f"{repo_owner}_{repo_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    return logger

class RateLimiter:
    def __init__(self, logger):
        self.logger = logger
        self.remaining_requests = None
        self.reset_time = None
        self.last_check = 0

    def check_rate_limit(self):
        try:
            response = requests.get(f"{REST_API_URL}/rate_limit", headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                core_limit = data['resources']['core']
                self.remaining_requests = core_limit['remaining']
                self.reset_time = core_limit['reset']

                self.logger.info(f"Rate limit status: {self.remaining_requests} requests remaining")
                return True
            else:
                self.logger.error(f"Failed to check rate limit: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            return False

    def wait_if_needed(self):
        current_time = time.time()
        if (self.remaining_requests is None or
            self.remaining_requests < RATE_LIMIT_BUFFER or
            current_time - self.last_check > 60):

            if not self.check_rate_limit():
                # If we can't check rate limit, be conservative
                time.sleep(DEFAULT_SLEEP_TIME)
                return

            self.last_check = current_time

        # If we're low on requests, wait until reset
        if self.remaining_requests < RATE_LIMIT_BUFFER:
            if self.reset_time:
                wait_time = max(0, self.reset_time - current_time + RATE_LIMIT_RESET_BUFFER)
                if wait_time > 0:
                    self.logger.warning(f"Rate limit approaching. Waiting {wait_time:.1f} seconds until reset.")
                    time.sleep(wait_time)
                    self.check_rate_limit()
            else:
                self.logger.warning("Rate limit low but no reset time available. Waiting 60 seconds.")
                time.sleep(60)

        # Always add a small delay between requests
        time.sleep(DEFAULT_SLEEP_TIME)

class ProgressTracker:
    def __init__(self, target_issues=ISSUE_COUNT, repo_name=""):
        self.target_issues = target_issues
        self.repo_name = repo_name
        self.collected_issues = 0
        self.processed_prs = 0
        self.api_calls = 0
        self.skipped_prs = 0
        self.commit_failures = 0
        self.rejected_commits = 0
        self.start_time = time.time()

        self.pbar = tqdm(
            total=target_issues,
            desc=f"Collecting Issues ({repo_name})",
            unit="issue",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.pbar.set_postfix(prs=0, api=0, skip=0, rej_commits=0)

    def update_issues(self, count=1):
        self.collected_issues += count
        self.pbar.update(count)
        self.update_stats()

    def update_stats(self):
        self.pbar.set_postfix(
            prs=self.processed_prs, api=self.api_calls,
            skip=self.skipped_prs, rej_commits=self.rejected_commits
        )

    def increment_prs(self):
        self.processed_prs += 1
        self.update_stats()

    def increment_api_calls(self):
        self.api_calls += 1
        self.update_stats()

    def increment_skipped(self):
        self.skipped_prs += 1
        self.update_stats()

    def increment_failures(self):
        self.commit_failures += 1

    def increment_rejected_commits(self, count=1):
        self.rejected_commits += count
        self.update_stats()

    def close(self):
        self.pbar.close()
        elapsed_time = time.time() - self.start_time
        return {
            'elapsed_time': elapsed_time,
            'collected_issues': self.collected_issues,
            'processed_prs': self.processed_prs,
            'api_calls': self.api_calls,
            'skipped_prs': self.skipped_prs,
            'rejected_commits': self.rejected_commits
        }

def run_graphql_query(query, variables, rate_limiter, progress, logger, max_retries=5, backoff_factor=2):
    for attempt in range(1, max_retries + 1):
        try:
            rate_limiter.wait_if_needed()
            progress.increment_api_calls()

            response = requests.post(
                GRAPHQL_API_URL,
                json={"query": query, "variables": variables},
                headers=HEADERS,
                timeout=30
            )

            # Will raise for HTTP errors (like 502, 500, 403, etc.)
            response.raise_for_status()
            result = response.json()

            if 'X-RateLimit-Remaining' in response.headers:
                rate_limiter.remaining_requests = int(response.headers['X-RateLimit-Remaining'])

            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                # GraphQL errors are usually not transient ,  no retry
                raise Exception(f"GraphQL errors: {result['errors']}")

            return result

        except requests.exceptions.RequestException as e:
            # Retry only on transient errors
            if attempt < max_retries and isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                sleep_time = backoff_factor ** (attempt - 1)
                logger.warning(f"Attempt {attempt} failed ({e}), retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            elif attempt < max_retries and response is not None and 500 <= response.status_code < 600:
                # Retry on 5xx errors
                sleep_time = backoff_factor ** (attempt - 1)
                logger.warning(f"Attempt {attempt} got {response.status_code}, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                logger.error(f"GraphQL query failed after {attempt} attempts: {e}")
                raise

def get_commit_details(owner, repo, sha, rate_limiter, progress, logger):
    try:
        rate_limiter.wait_if_needed()
        url = f"{REST_API_URL}/repos/{owner}/{repo}/commits/{sha}"
        progress.increment_api_calls()

        response = requests.get(url, headers=HEADERS, timeout=30)

        # Update rate limiter info from response headers
        if 'X-RateLimit-Remaining' in response.headers:
            rate_limiter.remaining_requests = int(response.headers['X-RateLimit-Remaining'])

        if response.status_code == 200:
            data = response.json()
            files = data.get("files", [])
            return {
                "commit_id": data["sha"],
                "commit_date": data["commit"]["committer"]["date"],
                "message": data["commit"]["message"],
                "files_changed": [f.get("filename") for f in files],
                "diff_summary": f"+{data['stats']['additions']}/-{data['stats']['deletions']} across {len(files)} files",
                "patch_data": "\n".join([f.get("patch", "") for f in files if f.get("patch")])
            }
        elif response.status_code == 404:
            logger.warning(f"Commit {sha[:8]} not found (404). Skipping.")
            progress.increment_failures()
        else:
            logger.error(f"Failed to fetch commit {sha[:8]}: {response.status_code} - {response.text}")
            progress.increment_failures()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching commit {sha[:8]}: {e}")
        progress.increment_failures()
    return None

# ------------------- DATA COLLECTION -------------------
def collect_true_and_rejected_shas(repo_owner, repo_name, rate_limiter, progress, logger):
    """Collect true and rejected SHAs for a repository."""
    true_rows = []
    rejected_shas = []
    has_next_page = True
    cursor = None

    logger.info(f"Starting data collection for {repo_owner}/{repo_name}")

    while has_next_page and progress.collected_issues < ISSUE_COUNT:
        query = """
        query($owner: String!, $name: String!, $cursor: String, $pageSize: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequests(first: $pageSize, states: MERGED, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
              pageInfo { hasNextPage endCursor }
              nodes {
                number
                closingIssuesReferences(first: 5) {
                  nodes {
                    number title body createdAt
                    labels(first: 10) { nodes { name } }
                    comments(first: 10) { nodes { bodyText } }
                  }
                }
                commits(first: 100) {
                  totalCount
                  nodes { commit { oid } }
                }
              }
            }
          }
        }
        """
        variables = {"owner": repo_owner, "name": repo_name, "cursor": cursor, "pageSize": PR_PAGE_SIZE}

        try:
            result = run_graphql_query(query, variables, rate_limiter, progress, logger)
        except Exception as e:
            logger.error(f"Failed to run GraphQL query: {e}")
            break

        pr_nodes = result["data"]["repository"]["pullRequests"]["nodes"]

        for pr in pr_nodes:
            progress.increment_prs()
            issues = pr["closingIssuesReferences"]["nodes"]
            commits = pr["commits"]["nodes"]
            total_commits = pr["commits"]["totalCount"]

            meets_criteria = (len(issues) == 1 and MIN_COMMITS <= total_commits <= MAX_COMMITS)

            if meets_criteria and progress.collected_issues < ISSUE_COUNT:
                logger.info(f"Accepted PR #{pr['number']} with {total_commits} commits and {len(issues)} linked issue(s).")
                issue = issues[0]
                issue_commit_rows = []

                for commit_node in commits:
                    sha = commit_node["commit"]["oid"]
                    commit_details = get_commit_details(repo_owner, repo_name, sha, rate_limiter, progress, logger)
                    if commit_details:
                        issue_commit_rows.append({
                            "Repository": f"{repo_owner}/{repo_name}",

                            "Issue ID": issue["number"],
                            "Issue Date": issue["createdAt"],
                            "Title": issue["title"],
                            "Description": issue["body"] or "",
                            "Labels": ", ".join(label["name"] for label in issue["labels"]["nodes"]),
                            "Comments": " | ".join(comment["bodyText"] for comment in issue["comments"]["nodes"]),

                            "Commit ID": commit_details["commit_id"],
                            "Commit Date": commit_details["commit_date"],
                            "Message": commit_details["message"],
                            "Diff Summary": commit_details["diff_summary"],
                            "File Changes": ", ".join(commit_details["files_changed"]),
                            "Full Diff": commit_details["patch_data"],
                            "Output": 1
                        })
                true_rows.extend(issue_commit_rows)
                progress.update_issues()
            else:
                logger.debug(f"Rejected PR #{pr['number']} (commits: {total_commits}, issues: {len(issues)})")
                progress.increment_skipped()
                for commit_node in commits:
                    rejected_shas.append(commit_node["commit"]["oid"])
                    progress.increment_rejected_commits()

        page_info = result["data"]["repository"]["pullRequests"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        cursor = page_info["endCursor"]

    logger.info(f"Collected {len(true_rows)} true links and {len(rejected_shas)} rejected SHAs")
    return true_rows, rejected_shas

# ------------------- FALSE LINK GENERATION -------------------
def build_false_links(true_rows, rejected_shas, repo_owner, repo_name, rate_limiter, progress, logger):
    """Build false links from rejected commits."""
    issues_to_links = defaultdict(list)
    for row in true_rows:
        issues_to_links[row["Issue ID"]].append(row)

    needed_false_commits = sum(len(rows) for rows in issues_to_links.values())
    logger.info(f"Need {needed_false_commits} false commits for {len(issues_to_links)} issues")

    if len(rejected_shas) > needed_false_commits:
        rejected_shas = random.sample(rejected_shas, needed_false_commits)
    random.shuffle(rejected_shas)

    unique_rejected_commits = []
    seen_shas = set()

    for sha in rejected_shas:
        if sha not in seen_shas:
            commit_details = get_commit_details(repo_owner, repo_name, sha, rate_limiter, progress, logger)
            if commit_details:
                unique_rejected_commits.append(commit_details)
            seen_shas.add(sha)

    false_rows = []
    idx = 0

    for issue_id, linked_rows in issues_to_links.items():
        num_true_links = len(linked_rows)
        commits_for_false = unique_rejected_commits[idx: idx + num_true_links]
        idx += num_true_links

        logger.debug(f"Creating false links for Issue #{issue_id} with {len(linked_rows)} true commits")
        template = {
            "Repository": f"{repo_owner}/{repo_name}",
            "Issue ID": linked_rows[0]["Issue ID"],
            "Issue Date": linked_rows[0]["Issue Date"],
            "Title": linked_rows[0]["Title"],
            "Description": linked_rows[0]["Description"],
            "Labels": linked_rows[0]["Labels"],
            "Comments": linked_rows[0]["Comments"]
        }

        for commit_data in commits_for_false:
            false_row = template.copy()
            false_row.update({
                "Commit ID": commit_data["commit_id"],
                "Commit Date": commit_data["commit_date"],
                "Message": commit_data["message"],
                "Diff Summary": commit_data["diff_summary"],
                "File Changes": ", ".join(commit_data["files_changed"]),
                "Full Diff": commit_data["patch_data"],
                "Output": 0
            })
            false_rows.append(false_row)

    logger.info(f"Generated {len(false_rows)} false links")
    return false_rows

# ------------------- CSV WRITING -------------------
def write_csv(all_rows, repo_owner, repo_name, logger):
    """Write data to CSV file."""
    csv_filename = os.path.join(DATA_DIR, f"{repo_owner}_{repo_name}_{ISSUE_COUNT}-balanced-{MIN_COMMITS}-{MAX_COMMITS}.csv")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(all_rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"✅ CSV '{csv_filename}' created with {len(all_rows)} entries.")
    return csv_filename

def process_repository(repo_config):
    repo_owner = repo_config["owner"]
    repo_name = repo_config["name"]

    logger = setup_logging(repo_owner, repo_name)
    rate_limiter = RateLimiter(logger)
    progress = ProgressTracker(ISSUE_COUNT, f"{repo_owner}/{repo_name}")

    try:
        logger.info(f"Starting collection for {repo_owner}/{repo_name}")

        # Check rate limit at the start
        if not rate_limiter.check_rate_limit():
            logger.warning("Could not check initial rate limit. Proceeding with caution.")

        true_rows, rejected_shas = collect_true_and_rejected_shas(
            repo_owner, repo_name, rate_limiter, progress, logger
        )

        if not true_rows:
           logger.warning("No true links found. Proceeding with only false links.")

        if not rejected_shas:
           logger.warning("No rejected commits found. Proceeding without false links.")

        # Build false links
        false_rows = build_false_links(
            true_rows, rejected_shas, repo_owner, repo_name, rate_limiter, progress, logger
        )

        all_rows = true_rows + false_rows
        true_ids = {r["Commit ID"] for r in true_rows}
        false_ids = {r["Commit ID"] for r in false_rows}

        overlap = true_ids & false_ids
        if overlap:
            logger.warning(f"Removing {len(overlap)} overlapping commits")
            false_rows = [r for r in false_rows if r["Commit ID"] not in overlap]

        csv_file = write_csv(all_rows, repo_owner, repo_name, logger)
        stats = progress.close()

        logger.info(f"\nCollection completed for {repo_owner}/{repo_name} in {stats['elapsed_time']:.1f}s")
        logger.info(f"Final Stats: {stats['collected_issues']} issues, "
                    f"{stats['processed_prs']} PRs, {stats['api_calls']} API calls, "
                    f"{stats['skipped_prs']} skipped, {stats['rejected_commits']} rejected commits")

        return {
            "success": True,
            "csv_file": csv_file,
            "stats": stats,
            "true_links": len(true_rows),
            "false_links": len(false_rows)
        }

    except Exception as e:
        logger.error(f"Error processing repository {repo_owner}/{repo_name}: {e}", exc_info=True)
        progress.close()
        return {"success": False, "error": str(e)}

# ------------------- MAIN FUNCTION -------------------
def main():
    if GITHUB_TOKEN == "YOUR_PERSONAL_ACCESS_TOKEN":
        print("❌ Please set your GitHub token in the GITHUB_TOKEN variable.")
        return

    setup_directories()

    main_logger = setup_logging("main", "collector")
    main_logger.info(f"Starting collection for {len(REPOSITORIES)} repositories")
    main_logger.info(f"Target: {ISSUE_COUNT} issues per repository")

    results = []
    total_start_time = time.time()

    for i, repo_config in enumerate(REPOSITORIES, 1):
        repo_owner = repo_config["owner"]
        repo_name = repo_config["name"]

        print(f"\n{'='*60}")
        print(f"Processing repository {i}/{len(REPOSITORIES)}: {repo_owner}/{repo_name}")
        print(f"{'='*60}")

        result = process_repository(repo_config)
        result["repo"] = f"{repo_owner}/{repo_name}"
        results.append(result)

        if result["success"]: main_logger.info(f"✅ Successfully processed {repo_owner}/{repo_name}")
        else: main_logger.error(f"❌ Failed to process {repo_owner}/{repo_name}: {result['error']}")

        # Add delay between repositories to be respectful
        if i < len(REPOSITORIES):
            main_logger.info("Waiting 10 seconds before next repository...")
            time.sleep(10)

    # Final summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Repositories processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['repo']}")
        if result["success"]:
            print(f"   - CSV: {result['csv_file']}")
            print(f"   - True links: {result['true_links']}, False links: {result['false_links']}")

    main_logger.info(f"Collection complete. {successful}/{len(results)} repositories processed successfully.")

if __name__ == "__main__": main()