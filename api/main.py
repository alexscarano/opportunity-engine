import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse

from api import db

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    # always 200: liveness must not depend on the DB; db state is reported, not enforced
    return {"status": "ok", "db": "ok" if db.ping() else "down"}


def mount_spa(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"
    if not index.is_file():
        return
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404)
        file = (root / path).resolve()
        if path and file.is_file() and file.is_relative_to(root):
            return FileResponse(file)
        return FileResponse(index)


def create_app(web_dist: Path | None = None) -> FastAPI:
    app = FastAPI(title="Marketing GEM")
    app.include_router(router)
    mount_spa(app, web_dist or Path(os.environ.get("WEB_DIST", "web/dist")))
    return app


app = create_app()
