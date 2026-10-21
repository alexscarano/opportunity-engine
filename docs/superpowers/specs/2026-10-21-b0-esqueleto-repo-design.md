# B0 — Esqueleto do repo (api/, web/, Docker, CI)

Item B0 de [`docs/m2/trilha-b-plataforma.md`](../../m2/trilha-b-plataforma.md). Depende de nada; bloqueia toda a trilha B.

## Contexto

Hoje o repo só sabe rodar o Streamlit: `Dockerfile` na porta 8501 instalando `requirements.txt` (export pinado que diverge do `pyproject.toml`, sem fastapi nem pillow), `mise.toml` com `test` e `dev`, sem CI. O M2 troca a UI por React servido por FastAPI, com worker em Cloud Run Job na mesma imagem e Postgres como banco do app (README do M2 › Decisões).

## Decisões

| Tema | Decisão |
|---|---|
| Empacotamento | `api/` e `engine/` como diretórios na raiz; `pyproject.toml` segue projeto virtual do uv (sem build-system). Rodar sempre a partir da raiz. |
| Streamlit | Continua na mesma imagem até a remoção (B11); sobe trocando o comando. |
| Banco no B0 | SQLAlchemy + Alembic configurados, **sem tabelas**. Primeira migração nasce no B1. |
| Config | `os.environ` direto. pydantic-settings só quando houver mais de 3 variáveis. |
| UI lib/tema | Fora do B0 (B10, tokens do Metrics). |
| Docker no CI | Fora do B0; verificado manualmente, entra no CI no B11. |

## Python e dependências

- `pyproject.toml`: adicionar `fastapi`, `uvicorn[standard]`, `sqlalchemy>=2`, `alembic`, `psycopg[binary]`, `pillow`. `pytest` e `httpx` (TestClient) no grupo `dev`.
- `[tool.pytest.ini_options]`: `pythonpath = ["."]`, `testpaths = ["tests"]`.
- Apagar `requirements.txt`. Regenerar `uv.lock`.

## `api/`

```
api/
  __init__.py
  main.py        # app FastAPI
  db.py          # engine + SessionLocal
  worker.py      # stub do Cloud Run Job
  alembic.ini    # script_location = api/migrations
  migrations/    # alembic init, sem revisões
```

- `main.py`
  - `APIRouter(prefix="/api")` com `GET /api/health`.
  - `/api/health` responde **sempre 200**: `{"status": "ok", "db": "ok" | "down"}`. `db` vem de `SELECT 1` com timeout curto; exceção vira `"down"`. Assim a imagem passa no aceite sem banco e o estado do banco fica visível.
  - SPA: se `web/dist` existir, serve os arquivos estáticos em `/` e devolve `index.html` para qualquer GET fora de `/api` que não case com arquivo (rotas do TanStack Router). Caminho de `web/dist` via env `WEB_DIST` (default `web/dist`). Sem `web/dist`, nada é montado (dev usa o Vite).
  - `/api/*` inexistente devolve 404 JSON, nunca o `index.html`.
- `db.py`: `DATABASE_URL` (default `postgresql+psycopg://app:app@localhost:5432/app`, o do compose), `engine = create_engine(..., pool_pre_ping=True)`, `SessionLocal = sessionmaker(engine)`.
- `migrations/env.py`: URL lida de `api.db.DATABASE_URL`; `target_metadata = None` até o B1 criar o `Base`.
- `worker.py`: `python -m api.worker` loga uma linha e sai 0. O B4 implementa; o B0 só fixa o comando.

## `web/`

- Vite + React + TypeScript, TanStack Router file-based (`@tanstack/router-plugin/vite`) e TanStack Query. Gerenciador: `bun`.
- Rota `/` (`src/routes/index.tsx`) faz `useQuery` em `/api/health` e mostra `status`/`db`. Prova o caminho front → API.
- `vite.config.ts`: proxy `/api` → `http://localhost:8000`.
- `tsconfig` em modo `strict`.

## Infra

- `docker-compose.yml`: só `postgres:16`, usuário/senha/db `app`, porta 5432, volume nomeado `pgdata`.
- `mise.toml`: `[tools]` ganha `node = "22"` (bun já instalado fora do mise; fallback npm). Tasks:

| Task | Comando |
|---|---|
| `api` | `uv run uvicorn api.main:app --reload --port 8000` |
| `web` | `bun run dev` (dir `web/`) |
| `worker` | `uv run python -m api.worker` |
| `db-migrate` | `uv run alembic -c api/alembic.ini upgrade head` |
| `test` | `uv run pytest` |
| `dev` | Streamlit, como hoje |

- `Dockerfile` multi-stage:
  1. `oven/bun` → `bun install --frozen-lockfile && bun run build` em `web/`.
  2. builder Python com uv → `uv sync --frozen --no-dev --no-install-project` para `/app/.venv`; mantém a remoção das pastas `tests/` das deps.
  3. runtime `python:3.13-slim`, usuário não-root: copia `.venv`, `api/`, `scripts/`, `logger.py`, `Logos/` e `web/dist`. `PATH` aponta para `.venv/bin`.
  - Comando padrão: `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}` (forma shell para expandir `$PORT` do Cloud Run).
  - Job: `python -m api.worker`. Streamlit: `streamlit run scripts/streamlit_app.py --server.port=8501 ...`.
  - `HEALTHCHECK` em `/api/health`.
  - Quando o A0 criar `engine/`, a linha `COPY engine/` entra junto (aviso ao Dev A).
- `.dockerignore`: adicionar `web/node_modules`, `web/dist`.
- `.gitignore`: `web/node_modules`, `web/dist`.

## CI — `.github/workflows/ci.yml`

Em `push` e `pull_request`:
- `py`: `astral-sh/setup-uv`, `uv sync --frozen`, `uv run pytest`.
- `web`: `oven-sh/setup-bun`, `bun install --frozen-lockfile`, `bunx tsc --noEmit`, `bun run build`.

## Testes

- `tests/api/test_health.py`
  - com `DATABASE_URL` inválida (porta fechada): 200 e `db == "down"`.
  - com banco no ar: 200 e `db == "ok"`; `pytest.skip` se o Postgres não responder (CI não sobe Postgres no B0).
- `tests/api/test_spa.py`: `WEB_DIST` aponta para um `tmp_path` com `index.html` e `assets/x.js`; `GET /qualquer/rota` → `index.html`; `GET /assets/x.js` → o arquivo; `GET /api/nao-existe` → 404 JSON.
- Os testes atuais continuam verdes.

## Aceite

| Critério (B0) | Verificação |
|---|---|
| `mise run api` + `mise run web` sobem com hot reload | manual: editar `main.py` e `index.tsx`, ver recarregar; página `/` mostra `status: ok` |
| `docker build` gera imagem que responde `/api/health` e serve o SPA | manual: `docker build -t oe .`, `docker run -p 8080:8080 oe`, `curl :8080/api/health` → 200; `curl :8080/runs/x` → HTML do SPA |
| Streamlit na mesma imagem | manual: `docker run -p 8501:8501 oe streamlit run scripts/streamlit_app.py ...` abre a tela |
| `mise run db-migrate` | manual: `docker compose up -d` + `mise run db-migrate` sai 0 |
| CI verde | workflow passando no PR do branch `b0-esqueleto` |

## Fora do B0

Auth/RBAC e tabelas (B1), storage (B2), worker real (B4), tema Metrics (B10), deploy e Docker no CI (B11), `engine/` (A0).
