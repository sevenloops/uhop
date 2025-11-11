#!/usr/bin/env python3
"""
Sync GitHub Issues from markdown files in a directory.

- Each .md file becomes an issue with:
  - title: first H1 (# ...) line, or file stem if missing
  - body: full file contents + a hidden marker with file path
  - label: provided via --label (default: auto:roadmap)
- Updates existing issues (by hidden marker); creates new ones when absent
- Optionally closes open issues with the label that are not represented by any file (--close-missing)

Requires environment variables:
- GITHUB_TOKEN (provided automatically by GitHub Actions)
- REPO (owner/repo, provided by workflow)
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

API = "https://api.github.com"

MARKER_START = "<!-- ISSUE-SOURCE: "
MARKER_END = " -->"


@dataclass
class LocalIssue:
    path: Path
    title: str
    body: str
    label: str

    @property
    def marker(self) -> str:
        return f"{MARKER_START}{self.path.as_posix()}{MARKER_END}"

    def body_with_marker(self) -> str:
        if self.marker in self.body:
            return self.body
        return f"{self.body}\n\n{self.marker}\n"


def get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(2)
    return val


def parse_markdown_issue(md_path: Path, default_label: str) -> LocalIssue:
    text = md_path.read_text(encoding="utf-8")
    # Title = first H1 line
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_path.stem.replace("_", " ")
    return LocalIssue(path=md_path, title=title, body=text.strip(), label=default_label)


def gh_request(session: requests.Session, method: str, url: str, **kwargs):
    r = session.request(method, url, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text}")
    return r


def list_open_issues(session: requests.Session, repo: str, label: str) -> List[dict]:
    issues = []
    page = 1
    while True:
        url = f"{API}/repos/{repo}/issues?state=open&labels={requests.utils.quote(label)}&per_page=100&page={page}"
        r = gh_request(session, "GET", url)
        batch = r.json()
        if not batch:
            break
        # Filter out PRs (PRs have a 'pull_request' key)
        batch = [i for i in batch if "pull_request" not in i]
        issues.extend(batch)
        page += 1
    return issues


def find_issue_by_marker(issues: List[dict], marker: str) -> Optional[dict]:
    for i in issues:
        if i.get("body") and marker in i["body"]:
            return i
    return None


def ensure_label_exists(session: requests.Session, repo: str, label: str):
    # Try to create, ignore if already exists
    url = f"{API}/repos/{repo}/labels"
    try:
        gh_request(session, "POST", url, json={"name": label, "color": "0ea5e9", "description": "Synced from /issues"})
    except RuntimeError:
        # If it already exists, GitHub returns 422. Ignore.
        pass


def sync_issues(directory: Path, repo: str, token: str, label: str, close_missing: bool):
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "uhop-issues-sync",
        }
    )

    ensure_label_exists(session, repo, label)

    local_files = sorted([p for p in directory.glob("*.md") if p.is_file()])
    locals_map: Dict[str, LocalIssue] = {}
    for p in local_files:
        li = parse_markdown_issue(p, label)
        locals_map[li.marker] = li

    open_issues = list_open_issues(session, repo, label)

    # Create or update
    for marker, li in locals_map.items():
        existing = find_issue_by_marker(open_issues, marker)
        if existing is None:
            # Create new
            url = f"{API}/repos/{repo}/issues"
            print(f"Creating issue: {li.title}")
            gh_request(
                session,
                "POST",
                url,
                json={
                    "title": li.title,
                    "body": li.body_with_marker(),
                    "labels": [li.label],
                },
            )
        else:
            # Update title/body if changed
            needs_update = False
            new_title = li.title
            new_body = li.body_with_marker()
            if existing.get("title") != new_title:
                needs_update = True
            if (existing.get("body") or "") != new_body:
                needs_update = True
            if needs_update:
                url = f"{API}/repos/{repo}/issues/{existing['number']}"
                print(f"Updating issue: #{existing['number']} {existing['title']} -> {new_title}")
                gh_request(
                    session,
                    "PATCH",
                    url,
                    json={
                        "title": new_title,
                        "body": new_body,
                    },
                )

    if close_missing:
        markers_local = set(locals_map.keys())
        for i in open_issues:
            body = i.get("body") or ""
            # Find marker in body
            marker_match = re.search(r"<!-- ISSUE-SOURCE: .*? -->", body)
            if not marker_match:
                # Skip issues we didn't create
                continue
            marker = marker_match.group(0)
            if marker not in markers_local:
                url = f"{API}/repos/{repo}/issues/{i['number']}"
                print(f"Closing missing issue: #{i['number']} {i['title']}")
                gh_request(session, "PATCH", url, json={"state": "closed"})


def main():
    parser = argparse.ArgumentParser(description="Sync GitHub Issues from /issues directory")
    parser.add_argument("--dir", default="issues", help="Directory with markdown issue files")
    parser.add_argument("--label", default="auto:roadmap", help="Label to apply to synced issues")
    parser.add_argument("--close-missing", action="store_true", help="Close open labeled issues not present as files")
    args = parser.parse_args()

    token = get_env("GITHUB_TOKEN")
    repo = get_env("REPO")

    directory = Path(args.dir)
    if not directory.exists():
        print(f"Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    sync_issues(directory=directory, repo=repo, token=token, label=args.label, close_missing=args.close_missing)


if __name__ == "__main__":
    main()
