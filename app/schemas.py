from pydantic import BaseModel, Field


class Issue(BaseModel):
    number: int
    title: str
    body: str = ""
    labels: list[str] = Field(default_factory=list)
    state: str
    html_url: str
    created_at: str

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


class IssueComment(BaseModel):
    body: str
    author: str | None = None
    created_at: str
    html_url: str
    issue_number: int | None = None


class ImportRequest(BaseModel):
    repository: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    limit: int = Field(default=500, ge=1, le=2000)


class SearchRequest(BaseModel):
    repository: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    title: str = Field(min_length=3, max_length=500)
    body: str = Field(default="", max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    issue: Issue
    similarity: float


class RepositorySummary(BaseModel):
    repository: str
    issue_count: int
