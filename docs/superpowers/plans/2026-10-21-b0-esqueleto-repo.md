# B0 — Esqueleto do repo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subir o esqueleto da plataforma M2: `api/` (FastAPI + SQLAlchemy/Alembic sem tabelas + worker stub), `web/` (Vite + React + TanStack), Docker com uma imagem só, Postgres no compose e CI no GitHub Actions.

**Architecture:** `api/` é um pacote na raiz (projeto virtual do uv, sem build-system), rodado sempre da raiz. O FastAPI serve `/api/*` e, se `web/dist` existir, o SPA com fallback para `index.html`. Mesma imagem roda API (padrão), worker (`python -m api.worker`) e Streamlit (até a remoção no B11).

**Tech Stack:** Python 3.13, uv, FastAPI, SQLAlchemy 2, Alembic, psycopg 3, pytest; bun, Vite, React, TypeScript, TanStack Router (file-based) + Query; Docker; GitHub Actions; mise.

**Spec:** [`docs/superpowers/specs/2026-10-21-b0-esqueleto-repo-design.md`](../specs/2026-10-21-b0-esqueleto-repo-design.md)

**Branch:** `b0-esqueleto` (já criado). Todos os comandos rodam na raiz do repo, salvo indicação.

**Baseline:** `uv run --with pytest pytest -q` → `209 passed` (21/out/2026). Nenhuma task pode reduzir isso.

## Ordem e paralelismo

| Lote | Tasks | Por quê |
|---|---|---|
| 1 | Task 1 (deps Python) · Task 5 (web) | arquivos disjuntos |
| 2 | Task 2 (health) · Task 4 (alembic/worker/compose/mise) | ambos dependem só da Task 1; Task 2 só toca `api/main.py`, `api/db.py`, testes; Task 4 só toca `api/worker.py`, `api/alembic.ini`, `api/migrations/`, `docker-compose.yml`, `mise.toml`. **Task 4 não cria `api/db.py` nem `api/__init__.py`** (são da Task 2) |
| 3 | Task 3 (SPA) · Task 7 (CI) | Task 3 mexe em `api/main.py` (depois da 2); Task 7 depende de 1 e 5 |
| 4 | Task 6 (Docker) | precisa de tudo |
| 5 | Task 8 (aceite manual) | — |

## Mapa de arquivos

| Arquivo | Responsabilidade | Task |
|---|---|---|
| `pyproject.toml`, `uv.lock` | deps + config do pytest | 1 |
| `requirements.txt` | **apagado** | 1 |
| `api/__init__.py` | marca o pacote (vazio) | 2 |
| `api/db.py` | `DATABASE_URL`, `engine`, `SessionLocal`, `ping()` | 2 |
| `api/main.py` | `create_app()`, `/api/health`, SPA | 2, 3 |
| `tests/api/test_health.py` | health com db down/ok | 2 |
| `tests/api/test_spa.py` | fallback do SPA, assets, 404 da API | 3 |
| `api/worker.py` | stub do Cloud Run Job | 4 |
| `api/alembic.ini`, `api/migrations/` | Alembic sem revisões | 4 |
| `docker-compose.yml` | Postgres 16 de dev | 4 |
| `mise.toml` | tools + tasks | 1, 4 |
| `web/**` | SPA Vite + React + TanStack | 5 |
| `Dockerfile`, `.dockerignore`, `.gitignore` | imagem única | 5, 6 |
| `.github/workflows/ci.yml` | CI | 7 |

---

### Task 1: Dependências Python e config do pytest

**Model:** haiku

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `mise.toml`
- Delete: `requirements.txt`

- [ ] **Step 1: Adicionar deps**

```bash
uv add fastapi "uvicorn[standard]" "sqlalchemy>=2" alembic "psycopg[binary]" pillow
uv add --dev pytest httpx
```

Expected: `pyproject.toml` ganha as 6 deps em `dependencies` e um bloco `[dependency-groups]` com `dev = ["httpx>=…", "pytest>=…"]`; `uv.lock` atualizado.

- [ ] **Step 2: Config do pytest**

Acrescentar ao fim de `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Apagar `requirements.txt` e trocar a task `test`**

```bash
git rm requirements.txt
```

Em `mise.toml`, trocar:

```toml
[tasks.test]
description = "Run tests"
run = "uv run --with pytest pytest"
```

por:

```toml
[tasks.test]
description = "Run tests"
run = "uv run pytest"
```

- [ ] **Step 4: Rodar a suíte**

Run: `uv run pytest -q`
Expected: `209 passed` (warnings ok).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock mise.toml requirements.txt
git commit -m "build: add api deps to pyproject, drop requirements.txt"
```

---

### Task 2: `api/db.py` + `GET /api/health`

**Model:** haiku

**Files:**
- Create: `api/__init__.py` (vazio), `api/db.py`, `api/main.py`
- Test: `tests/api/test_health.py`

- [ ] **Step 1: Escrever o teste que falha**

`tests/api/test_health.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/api/test_health.py -v`
Expected: erro de coleta `ModuleNotFoundError: No module named 'api'`.

- [ ] **Step 3: Implementar**

`api/__init__.py`: arquivo vazio.

`api/db.py`:

```python
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://app:app@localhost:5432/app")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 2})
SessionLocal = sessionmaker(engine)


def ping() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
```

`api/main.py`:

```python
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
```

Nota: `health` usa `db.ping()` (atributo do módulo), por isso o `monkeypatch.setattr(api.db, "engine", ...)` funciona.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/api/test_health.py -v`
Expected: `test_health_reports_db_down PASSED`; `test_health_reports_db_ok` `SKIPPED` (sem Postgres) ou `PASSED` (com).

- [ ] **Step 5: Suíte completa**

Run: `uv run pytest -q`
Expected: `210 passed, 1 skipped` (ou `211 passed`).

- [ ] **Step 6: Commit**

```bash
git add api/__init__.py api/db.py api/main.py tests/api/test_health.py
git commit -m "feat(api): add fastapi app with /api/health reporting db status"
```

---

### Task 3: Servir o SPA com fallback

**Model:** sonnet (path traversal e ordem de rotas)

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_spa.py`

- [ ] **Step 1: Escrever o teste que falha**

`tests/api/test_spa.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/api/test_spa.py -v`
Expected: FAIL com `TypeError: create_app() got an unexpected keyword argument 'web_dist'`.

- [ ] **Step 3: Implementar**

`api/main.py` completo:

```python
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
```

O catch-all é registrado **depois** do router, então `/api/health` casa primeiro; qualquer `/api/*` desconhecido cai no catch-all e vira 404 JSON.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/api -v`
Expected: todos `PASSED` (o `db_ok` pode estar `SKIPPED`).

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/api/test_spa.py
git commit -m "feat(api): serve spa build with index.html fallback"
```

---

### Task 4: Alembic, worker stub, Postgres no compose, tasks do mise

**Model:** haiku

**Files:**
- Create: `api/worker.py`, `api/alembic.ini`, `api/migrations/` (via `alembic init`), `docker-compose.yml`
- Modify: `mise.toml`

Pré-requisito: Task 1 feita (alembic instalado). **Não criar nem editar `api/db.py`, `api/main.py` ou `api/__init__.py`** — são da Task 2, que roda em paralelo. Se a Task 2 ainda não tiver criado `api/__init__.py`, o Step 6 espera por ela.

- [ ] **Step 1: Worker stub**

`api/worker.py`:

```python
import logging

log = logging.getLogger("api.worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # ponytail: stub so the Cloud Run Job command is fixed now; B4 implements the run loop
    log.info("worker: nothing to run yet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Alembic**

```bash
uv run alembic init api/migrations
mv alembic.ini api/alembic.ini
```

Em `api/alembic.ini`, trocar a linha `script_location = ...` por:

```ini
script_location = %(here)s/migrations
```

e apagar a linha `sqlalchemy.url = driver://user:pass@localhost/dbname` (a URL vem do código).

Em `api/migrations/env.py`, logo após `config = context.config`, acrescentar:

```python
from api.db import DATABASE_URL

# configparser treats % as interpolation
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))
```

Manter `target_metadata = None` (o B1 troca para o `Base.metadata`).

- [ ] **Step 3: `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: app
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 4: `mise.toml`**

Arquivo completo:

```toml
[tools]
python = "3.13"
uv = "latest"
node = "22"

[env]
_.python.venv = { path = ".venv", create = true, uv_create_args = ["--seed"] }

[tasks.test]
description = "Run tests"
run = "uv run pytest"

[tasks.dev]
description = "Development Streamlit"
run = "streamlit run scripts/streamlit_app.py"

[tasks.api]
description = "FastAPI with hot reload on :8000"
run = "uv run uvicorn api.main:app --reload --port 8000"

[tasks.web]
description = "Vite dev server (proxies /api to :8000)"
dir = "web"
run = "bun run dev"

[tasks.worker]
description = "Run the job worker once"
run = "uv run python -m api.worker"

[tasks.db-migrate]
description = "Apply Alembic migrations"
run = "uv run alembic -c api/alembic.ini upgrade head"
```

- [ ] **Step 5: Verificar worker**

Run: `uv run python -m api.worker`
Expected: `INFO:api.worker:worker: nothing to run yet`, exit 0.

- [ ] **Step 6: Verificar migração**

```bash
docker compose up -d
uv run alembic -c api/alembic.ini upgrade head
```

Expected: saída com `Context impl PostgresqlImpl.` e exit 0 (sem revisões, nada a aplicar). Se o Docker não estiver disponível no ambiente do agente, registrar isso no relatório e seguir; a Task 8 repete a verificação.

- [ ] **Step 7: Suíte**

Run: `uv run pytest -q`
Expected: mesma contagem da Task 2 (com Postgres no ar, `test_health_reports_db_ok` passa a `PASSED`).

- [ ] **Step 8: Commit**

```bash
git add api/worker.py api/alembic.ini api/migrations docker-compose.yml mise.toml
git commit -m "feat(api): add alembic setup, worker stub, postgres compose and mise tasks"
```

---

### Task 5: `web/` — Vite + React + TanStack

**Model:** sonnet

**Files:**
- Create: `web/**`
- Modify: `.gitignore`

- [ ] **Step 1: Scaffold**

```bash
bun create vite web --template react-ts
cd web
bun install
bun add @tanstack/react-router @tanstack/react-query
bun add -d @tanstack/router-plugin
```

Apagar o que o template traz e não será usado:

```bash
rm -f src/App.tsx src/App.css src/index.css src/assets/react.svg public/vite.svg
```

- [ ] **Step 2: `web/vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { tanstackRouter } from '@tanstack/router-plugin/vite'

export default defineConfig({
  plugins: [tanstackRouter({ target: 'react', autoCodeSplitting: true }), react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
```

O plugin do router precisa vir **antes** do `react()`.

- [ ] **Step 3: Rotas**

`web/src/routes/__root.tsx`:

```tsx
import { Outlet, createRootRoute } from '@tanstack/react-router'

export const Route = createRootRoute({ component: Outlet })
```

`web/src/routes/index.tsx`:

```tsx
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'

type Health = { status: string; db: string }

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: async (): Promise<Health> => {
      const r = await fetch('/api/health')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json()
    },
  })

  if (health.isPending) return <p>Carregando…</p>
  if (health.isError) return <p>API fora do ar: {health.error.message}</p>
  return (
    <p>
      API: {health.data.status} · banco: {health.data.db}
    </p>
  )
}
```

- [ ] **Step 4: `web/src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { routeTree } from './routeTree.gen'

const router = createRouter({ routeTree })
const queryClient = new QueryClient()

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
```

Em `web/index.html`, remover o `<link rel="icon" ... href="/vite.svg" />` e trocar o `<title>` para `Marketing GEM`.

- [ ] **Step 5: Gerar a árvore de rotas e checar tipos**

```bash
cd web
bun run build
```

Expected: o plugin gera `src/routeTree.gen.ts`, `tsc -b` passa sem erros e `vite build` escreve `web/dist/index.html`. O `routeTree.gen.ts` **é versionado** (recomendação do TanStack; o CI roda `tsc` antes de qualquer geração).

- [ ] **Step 6: `.gitignore`**

Na raiz, acrescentar:

```gitignore
# Front
web/node_modules/
web/dist/
```

Conferir que o template não criou `web/.gitignore` conflitante (pode ficar, é inofensivo).

- [ ] **Step 7: Commit**

```bash
cd ..
git add .gitignore web
git status --short web | grep -E "node_modules|dist" && echo "ERRO: build/deps no stage" || true
git commit -m "feat(web): scaffold vite react spa with tanstack router and query"
```

---

### Task 6: Dockerfile único

**Model:** sonnet

**Files:**
- Modify: `Dockerfile` (reescrita), `.dockerignore`

- [ ] **Step 1: `.dockerignore`**

Acrescentar ao fim:

```
web/node_modules/
web/dist/
```

- [ ] **Step 2: `Dockerfile`**

Arquivo completo:

```dockerfile
# syntax=docker/dockerfile:1

# ---- web: build the SPA ----
FROM oven/bun:1.3.14 AS web
WORKDIR /web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build

# ---- builder: resolve python deps from uv.lock into /app/.venv ----
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock .python-version ./
# strip vendored test suites from deps (~55MB dead weight, e.g. scipy/pandas/statsmodels tests)
RUN uv sync --frozen --no-dev --no-install-project \
    && find /app/.venv -depth -type d \( -name "tests" -o -name "test" \) -exec rm -rf {} +

# ---- runtime: api (default), worker job and streamlit share this image ----
FROM python:3.13-slim
WORKDIR /app

RUN groupadd -r app && useradd -r -m -d /home/app -g app app
ENV HOME=/home/app \
    PATH=/app/.venv/bin:$PATH \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /app/.venv /app/.venv
COPY logger.py .
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY Logos/ ./Logos/
COPY --from=web /web/dist ./web/dist

# runtime-only data for streamlit, mount as volumes (see .gitignore)
RUN mkdir -p data/log inputs outputs && chown -R app:app /app

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/api/health', timeout=3)" || exit 1

# shell form so Cloud Run's $PORT expands. Worker: `python -m api.worker`.
# Streamlit: `streamlit run scripts/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true`
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

Diferenças conscientes em relação ao Dockerfile antigo: `pip -r requirements.txt` → `uv sync --frozen`; `ENTRYPOINT` → `CMD` (para trocar o comando com `docker run <img> <cmd>` e no Cloud Run Job); o `VOLUME` saiu (Cloud Run não usa; o Streamlit local monta com `-v` se precisar). O healthcheck bate em `/api/health`, então ele reporta "unhealthy" quando a imagem roda como Streamlit; aceitável até a remoção no B11.

- [ ] **Step 3: Build**

Run: `docker build -t oe .`
Expected: build conclui. Se o Docker não estiver disponível no ambiente do agente, registrar isso no relatório; a Task 8 cobre.

- [ ] **Step 4: Smoke da API + SPA**

```bash
docker run -d --name oe-smoke -p 8080:8080 oe
curl -s localhost:8080/api/health
curl -s localhost:8080/w/x/runs/1 | head -c 200
docker rm -f oe-smoke
```

Expected: `{"status":"ok","db":"down"}` (sem banco na rede do container) e HTML contendo `<div id="root">`.

- [ ] **Step 5: Smoke do worker**

Run: `docker run --rm oe python -m api.worker`
Expected: `INFO:api.worker:worker: nothing to run yet`.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build(docker): single image serving api + spa, uv lock install"
```

---

### Task 7: CI no GitHub Actions

**Model:** haiku

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Workflow**

`.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  py:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --frozen
      - run: uv run pytest -q

  web:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v2
      - run: bun install --frozen-lockfile
      - run: bunx tsc -b
      - run: bun run build
```

`tsc -b` é o typecheck do template do Vite (os `tsconfig.*.json` já têm `noEmit`); `tsc --noEmit` no `tsconfig.json` raiz não checaria nada, porque ele só tem `references`.

- [ ] **Step 2: Validar localmente o que dá**

```bash
uv sync --frozen && uv run pytest -q
cd web && bun install --frozen-lockfile && bunx tsc -b && bun run build
```

Expected: tudo sai 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add pytest and web typecheck/build workflow"
```

O CI só roda de verdade no push. Testes que dependem de dados ignorados pelo git (`exemplo_csv/`, `inputs/`, `outputs/`) podem passar local e falhar no CI: se isso acontecer, **reportar ao usuário** quais testes falharam, sem pular nem apagar teste.

---

### Task 8: Aceite manual (com o usuário)

**Model:** — (executado pelo controlador, com o usuário)

- [ ] **Step 1: Hot reload local**

```bash
docker compose up -d
mise run db-migrate
mise run api      # terminal 1
mise run web      # terminal 2
```

Abrir `http://localhost:5173`: mostra `API: ok · banco: ok`. Editar o texto em `web/src/routes/index.tsx` → a página atualiza sozinha. Editar `api/main.py` → o uvicorn recarrega (log `Reloading...`).

- [ ] **Step 2: Imagem**

```bash
docker build -t oe .
docker run --rm -p 8080:8080 oe
```

`curl localhost:8080/api/health` → 200; `curl localhost:8080/runs/x` → HTML do SPA.

- [ ] **Step 3: Streamlit na mesma imagem**

```bash
docker run --rm -p 8501:8501 oe streamlit run scripts/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
```

Abrir `http://localhost:8501`: tela de login do Streamlit.

- [ ] **Step 4: CI verde**

Push do branch e PR para `main` (via `superpowers:finishing-a-development-branch`); conferir os jobs `py` e `web` verdes.

- [ ] **Step 5: Aviso ao Dev A**

Registrar no PR: quando o A0 criar `engine/`, adicionar `COPY engine/ ./engine/` no estágio runtime do `Dockerfile`.
