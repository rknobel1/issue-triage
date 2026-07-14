import asyncio

import typer

from app.config import settings
from app.embeddings import get_embedder
from app.github import GitHubClient
from app.service import TriageService
from app.store import RepositoryStore

app = typer.Typer(help="Import repositories and search for duplicate GitHub issues.")


def service() -> TriageService:
    return TriageService(RepositoryStore(settings.data_dir), get_embedder())


@app.command("import-repo")
def import_repo(repository: str, limit: int = 500) -> None:
    """Download and embed issues from OWNER/REPOSITORY."""

    async def run() -> int:
        github = GitHubClient(settings.github_token)
        try:
            return await service().import_repository(repository, limit, github)
        finally:
            await github.close()

    count = asyncio.run(run())
    typer.echo(f"Imported {count} issues from {repository}")


@app.command()
def search(repository: str, title: str, body: str = "", top_k: int = 5) -> None:
    """Find issues similar to a proposed issue."""
    for result in service().search(repository, title, body, top_k):
        typer.echo(
            f"{result.similarity:.4f}  #{result.issue.number}  "
            f"{result.issue.title}\n          {result.issue.html_url}"
        )


if __name__ == "__main__":
    app()

