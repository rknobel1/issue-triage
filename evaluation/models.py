from typing import Literal

from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    repository: str
    query_issue: int
    duplicate_issue: int
    source: Literal["issue_body", "issue_comment"]
    evidence: str
    confidence: float = Field(ge=0, le=1)
    actor: str | None = None
    evidence_url: str | None = None
    discovered_at: str
    review_status: Literal["pending", "approved", "rejected"] = "pending"

