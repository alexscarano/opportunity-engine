import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

import api.db
from api.main import create_app


def test_health_reports_db_down(monkeypatch):
    dead = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/x", connect_args={"connect_timeout": 1})
    monkeypatch.setattr(api.db, "engine", dead)

    r = TestClient(create_app()).get("/api/health")

    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": "down"}


def test_health_reports_db_ok():
    if not api.db.ping():
        pytest.skip("Postgres local fora do ar (docker compose up -d)")

    r = TestClient(create_app()).get("/api/health")

    assert r.json() == {"status": "ok", "db": "ok"}
