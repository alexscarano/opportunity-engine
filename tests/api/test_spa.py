import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(tmp_path):
    (tmp_path / "index.html").write_text("<html>spa</html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "x.js").write_text("console.log(1)")
    return TestClient(create_app(web_dist=tmp_path))


def test_unknown_route_falls_back_to_index(client):
    r = client.get("/w/acme/runs/42")
    assert r.status_code == 200
    assert r.text == "<html>spa</html>"


def test_root_serves_index(client):
    assert client.get("/").text == "<html>spa</html>"


def test_existing_asset_is_served(client):
    assert client.get("/assets/x.js").text == "console.log(1)"


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/nao-existe")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}


def test_path_traversal_does_not_escape_dist(client, tmp_path):
    (tmp_path.parent / "secret.txt").write_text("nope")
    r = client.get("/..%2Fsecret.txt")
    assert "nope" not in r.text


def test_no_dist_means_no_spa(tmp_path):
    r = TestClient(create_app(web_dist=tmp_path / "missing")).get("/")
    assert r.status_code == 404
