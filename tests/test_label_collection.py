import asyncio

import httpx

from app.github import GitHubClient
from app.schemas import Issue
from evaluation.collect_duplicates import candidate_confidence


def issue_record(number: int, *, pull_request: bool = False) -> dict:
    record = {
        "number": number,
        "title": f"Issue {number}",
        "body": "",
        "labels": [{"name": "r: duplicate"}],
        "state": "closed",
        "html_url": f"https://github.com/flutter/flutter/issues/{number}",
        "created_at": "2020-01-01T00:00:00Z",
    }
    if pull_request:
        record["pull_request"] = {}
    return record


def test_iter_labeled_issues_filters_pull_requests():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = request.url.params["page"]
        return httpx.Response(
            200,
            json=[issue_record(1), issue_record(2, pull_request=True)] if page == "1" else [],
        )

    async def run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.github.com"
        )
        github = GitHubClient(client=client)
        issues = [
            issue
            async for issue in github.iter_labeled_issues(
                "flutter/flutter", "r: duplicate", limit=10
            )
        ]
        await client.aclose()
        return issues

    issues = asyncio.run(run())
    assert [issue.number for issue in issues] == [1]
    assert requests[0].url.params["labels"] == "r: duplicate"
    assert requests[0].url.params["state"] == "closed"


def test_trusted_labeled_evidence_gets_highest_review_priority():
    issue = Issue(
        number=10,
        title="Duplicate",
        labels=["r: duplicate"],
        state="closed",
        html_url="https://github.com/flutter/flutter/issues/10",
        created_at="2020-01-02T00:00:00Z",
    )

    assert candidate_confidence(issue, "issue_comment", "MEMBER") == 0.99
    assert candidate_confidence(issue, "issue_comment", "NONE") == 0.98

