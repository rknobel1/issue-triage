from collections.abc import AsyncIterator

import httpx

from app.schemas import Issue


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
                yield Issue(
                    number=record["number"],
                    title=record["title"],
                    body=record.get("body") or "",
                    labels=[label["name"] for label in record.get("labels", [])],
                    state=record["state"],
                    html_url=record["html_url"],
                    created_at=record["created_at"],
                )
                yielded += 1
                if yielded >= limit:
                    break
            page += 1

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

