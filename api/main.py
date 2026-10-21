from fastapi import APIRouter, FastAPI

from api import db

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    # always 200: liveness must not depend on the DB; db state is reported, not enforced
    return {"status": "ok", "db": "ok" if db.ping() else "down"}


def create_app() -> FastAPI:
    app = FastAPI(title="Marketing GEM")
    app.include_router(router)
    return app


app = create_app()
