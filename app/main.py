from fastapi import Depends, FastAPI, HTTPException

from app.config import settings
from app.embeddings import get_embedder
from app.github import GitHubClient
from app.schemas import ImportRequest, RepositorySummary, SearchRequest, SearchResult
from app.service import TriageService
from app.store import RepositoryStore

app = FastAPI(title="Issue Triage AI", version="0.1.0")


def get_service() -> TriageService:
    return TriageService(RepositoryStore(settings.data_dir), get_embedder())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/repositories/import", response_model=RepositorySummary)
async def import_repository(
    request: ImportRequest, service: TriageService = Depends(get_service)
) -> RepositorySummary:
    github = GitHubClient(settings.github_token)
    try:
        count = await service.import_repository(request.repository, request.limit, github)
        return RepositorySummary(repository=request.repository, issue_count=count)
    except Exception as exc:
        status = 404 if "404" in str(exc) else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    finally:
        await github.close()


@app.post("/search/duplicates", response_model=list[SearchResult])
def search_duplicates(
    request: SearchRequest, service: TriageService = Depends(get_service)
) -> list[SearchResult]:
    try:
        return service.search(
            request.repository, request.title, request.body, request.top_k
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

