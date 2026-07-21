import re
from collections.abc import AsyncIterator

import httpx

from app.schemas import Issue, IssueComment


class GitHubClient:
    def __init__(self, token: str | None = None, client: httpx.AsyncClient | None = None):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url="https://api.github.com", headers=headers, timeout=30
        )

    @staticmethod
    def _parse_issue(record: dict) -> Issue:
        if "pull_request" in record:
            raise ValueError(f"#{record['number']} is a pull request, not an issue")
        return Issue(
            number=record["number"],
            title=record["title"],
            body=record.get("body") or "",
            labels=[label["name"] for label in record.get("labels", [])],
            state=record["state"],
            html_url=record["html_url"],
            created_at=record["created_at"],
        )

    async def iter_issues(self, repository: str, limit: int) -> AsyncIterator[Issue]:
        page = 1
        yielded = 0
        while yielded < limit:
            response = await self.client.get(
                f"/repos/{repository}/issues",
                params={"state": "all", "per_page": 100, "page": page},
            )
            response.raise_for_status()
            records = response.json()
            if not records:
                break
            for record in records:
                # GitHub's issues endpoint also returns pull requests.
                if "pull_request" in record:
                    continue
                yield self._parse_issue(record)
                yielded += 1
                if yielded >= limit:
                    break
            page += 1

    async def iter_issue_comments(
        self, repository: str, issue_number: int
    ) -> AsyncIterator[IssueComment]:
        page = 1
        while True:
            response = await self.client.get(
                f"/repos/{repository}/issues/{issue_number}/comments",
                params={"per_page": 100, "page": page},
            )
            response.raise_for_status()
            records = response.json()
            if not records:
                break
            for record in records:
                user = record.get("user") or {}
                yield IssueComment(
                    body=record.get("body") or "",
                    author=user.get("login"),
                    created_at=record["created_at"],
                    html_url=record["html_url"],
                    issue_number=issue_number,
                )
            page += 1

    async def iter_repository_comments(
        self, repository: str, limit: int = 10_000
    ) -> AsyncIterator[IssueComment]:
        """Yield newest repository issue comments in bulk, newest first."""
        page = 1
        yielded = 0
        while yielded < limit:
            response = await self.client.get(
                f"/repos/{repository}/issues/comments",
                params={
                    "sort": "created",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            response.raise_for_status()
            records = response.json()
            if not records:
                break
            for record in records:
                match = re.search(r"/issues/(?P<number>\d+)$", record["issue_url"])
                if match is None:
                    continue
                user = record.get("user") or {}
                yield IssueComment(
                    body=record.get("body") or "",
                    author=user.get("login"),
                    created_at=record["created_at"],
                    html_url=record["html_url"],
                    issue_number=int(match.group("number")),
                )
                yielded += 1
                if yielded >= limit:
                    break
            page += 1

    async def get_issue(self, repository: str, issue_number: int) -> Issue:
        """Download one issue by number."""
        response = await self.client.get(
            f"/repos/{repository}/issues/{issue_number}"
        )
        response.raise_for_status()
        return self._parse_issue(response.json())

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
