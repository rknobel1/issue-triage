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
        async for issue in self._iter_issues(repository, limit, state="all"):
            yield issue

    async def get_issues_batch(
        self, repository: str, issue_numbers: list[int]
    ) -> list[Issue]:
        """Fetch up to 50 issue numbers in one GraphQL request.

        Repository.issue returns null for pull request numbers, allowing corpus
        sampling to skip them without one REST request per number.
        """
        if len(issue_numbers) > 50:
            raise ValueError("A GraphQL issue batch cannot contain more than 50 numbers")
        owner, name = repository.split("/", maxsplit=1)
        fields = "\n".join(
            f"""
            issue{position}: issue(number: {number}) {{
              number
              title
              body
              state
              url
              createdAt
              labels(first: 100) {{ nodes {{ name }} }}
            }}
            """
            for position, number in enumerate(issue_numbers)
        )
        response = await self.client.post(
            "https://api.github.com/graphql",
            json={
                "query": (
                    "query($owner: String!, $name: String!) {"
                    " repository(owner: $owner, name: $name) {"
                    f"{fields}"
                    " }"
                    "}"
                ),
                "variables": {"owner": owner, "name": name},
            },
        )
        response.raise_for_status()
        payload = response.json()
        unexpected_errors = [
            error
            for error in payload.get("errors", [])
            if error.get("type") != "NOT_FOUND"
            or len(error.get("path", [])) != 2
            or error["path"][0] != "repository"
            or not str(error["path"][1]).startswith("issue")
        ]
        if unexpected_errors:
            raise ValueError(f"GitHub GraphQL error: {unexpected_errors}")
        records = (payload.get("data") or {}).get("repository") or {}
        issues = []
        for position in range(len(issue_numbers)):
            record = records.get(f"issue{position}")
            if record is None:
                continue
            issues.append(
                Issue(
                    number=record["number"],
                    title=record["title"],
                    body=record.get("body") or "",
                    labels=[
                        label["name"]
                        for label in (record.get("labels") or {}).get("nodes", [])
                    ],
                    state=record["state"].lower(),
                    html_url=record["url"],
                    created_at=record["createdAt"],
                )
            )
        return issues

    async def iter_labeled_issues(
        self,
        repository: str,
        label: str,
        limit: int,
        state: str = "closed",
    ) -> AsyncIterator[Issue]:
        """Yield issues carrying an exact label, oldest first for temporal diversity."""
        async for issue in self._iter_issues(
            repository,
            limit,
            state=state,
            labels=label,
            sort="created",
            direction="asc",
        ):
            yield issue

    async def _iter_issues(
        self,
        repository: str,
        limit: int,
        **filters: str,
    ) -> AsyncIterator[Issue]:
        page = 1
        yielded = 0
        while yielded < limit:
            response = await self.client.get(
                f"/repos/{repository}/issues",
                params={**filters, "per_page": 100, "page": page},
            )
            response.raise_for_status()
            records = response.json()
            if not records:
                break
            for record in records:
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
                    author_association=record.get("author_association"),
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
                    author_association=record.get("author_association"),
                    created_at=record["created_at"],
                    html_url=record["html_url"],
                    issue_number=int(match.group("number")),
                )
                yielded += 1
                if yielded >= limit:
                    break
            page += 1

    async def get_issue(self, repository: str, issue_number: int) -> Issue:
        response = await self.client.get(f"/repos/{repository}/issues/{issue_number}")
        response.raise_for_status()
        return self._parse_issue(response.json())

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
