# Trilha B — Plataforma: API, worker, front e infra (Dev B)

Dono: Dev B (full-stack). Leia antes o [`README.md`](README.md), principalmente **Decisões tomadas** e **O contrato A ↔ B**. A trilha B só consome o motor pelas 4 superfícies do contrato. Ela não lê CSV ou PNG de saída e não reimplementa nenhuma conta.

Princípio da trilha: **multi-marca primeiro**. Toda query, todo arquivo e todo endpoint é escopado por workspace desde o primeiro commit. Isolamento entre clientes não se acrescenta depois.

Para cada item: **BRD** · **Hoje** · **Fazer** · **Aceite** · **Em aberto**.

---

## B0 · Esqueleto do repo — semana 1

- **BRD:** Capabilities › Tech stack; Figure 2 (arquitetura).
- **Depende de:** nada. **Bloqueia:** todo o resto da trilha B.
- **Hoje:**
  - Dockerfile (Streamlit, porta 8501, instala `requirements.txt`).
  - `requirements.txt`, um export pinado que já traz uvicorn e pydantic mas não fastapi, divergindo do `pyproject.toml` (12 deps soltas, sem pillow).
  - `mise.toml` com as tasks `test` e `dev`. Não há CI.
- **Fazer:**
  - Diretórios `api/` (FastAPI) e `web/` (Vite + React + TS + TanStack Router + Query).
  - Um `pyproject.toml` com `uv.lock` na raiz; apagar `requirements.txt` e fazer o Docker instalar pelo lock.
  - `docker-compose.yml` com Postgres 16 para dev.
  - Tasks no `mise.toml`: `api`, `web`, `worker`, `test`, `db-migrate`. Para o front, usar `bun` (fallback npm).
  - Uma imagem: build do `web/` copiado para dentro e servido pelo FastAPI em `/`, com `/api/*` para a API. Comando padrão `uvicorn`; o job usa `python -m api.worker`.
  - `GET /api/health`.
  - CI no GitHub Actions: pytest + `tsc --noEmit` + build do front.
- **Aceite:**
  - `mise run api` + `mise run web` sobem localmente com hot reload.
  - `docker build` gera uma imagem que responde `/api/health` e serve o SPA.
  - CI verde.

## B1 · Autenticação, papéis e workspaces — semanas 1–2

- **BRD:** Capabilities › Authentication & user management; Journey 3 (Admin onboarding); Improvement Requests › Access; deliverable "Authentication & management".
- **Depende de:** B0.
- **Hoje:**
  - SQLite (`scripts/db.py`), PBKDF2 100k (`:59`), auto-cadastro aberto, sem papéis.
  - **Falha grave:** o cookie guarda `user_id`/`username` sem assinatura e o app confia nele (`streamlit_app.py:681-685, 705-710`). Trocar o `user_id` loga como outro usuário.
  - O token de sessão vai na URL (`?sid=`).
  - A chave Gemini fica por usuário, em texto puro.
- **Fazer:**
  - Tabelas:
    - `users` (email, nome, hash, `is_platform_admin`, `active`);
    - `workspaces` (nome, slug, `settings jsonb`, com os `gate_thresholds` que o Admin ajusta, BRD §2);
    - `memberships` (user, workspace, `role` admin|analyst|viewer);
    - `sessions` (hash do token, user, expira).
  - Senha com `hashlib.scrypt` (stdlib).
  - Sessão em cookie `HttpOnly; Secure; SameSite=Lax`, guardando token aleatório (hash no banco), nunca id.
  - **Sem auto-cadastro**: o Admin cria usuários com senha temporária e troca obrigatória no 1º login. Bloqueio após 5 falhas seguidas por 15 min.
  - Matriz de permissões:
    - **viewer** vê runs publicados do seu workspace, mexe em cenários e exporta;
    - **analyst** também cria simulação, faz upload, roda, publica e despublica;
    - **admin** também gerencia membros e thresholds do workspace;
    - **platform admin** (Data Tech) cria workspaces e vê o uso de todos.
  - Toda rota recebe o workspace por dependência do FastAPI que checa a membership. Recurso de outro workspace devolve **404**, não 403.
  - Telas: login, troca de senha, seletor de workspace, admin de membros, admin de workspaces.
  - Migrar os usuários do SQLite atual? Ver em aberto.
- **Aceite:**
  - Teste parametrizado papel × endpoint cobrindo toda rota.
  - Acesso entre workspaces → 404.
  - Cookie adulterado → 401.
  - Senha nunca aparece em log.
- **Em aberto:** e-mail para reset de senha ou só o Admin reseta? Sugestão: só o Admin no M2. Migrar contas do protótipo? Sugestão: não, recriar.

## B2 · Upload, armazenamento e validação — semana 2

- **BRD:** Capabilities › Data upload & validation; Journey 1 passos 2–3; Figure 3 passos 3–4; Datasets; In scope ("aggregated, no personal data").
- **Depende de:** B1, A2 (até lá, mock de `validate` com o fixture).
- **Hoje:**
  - Upload direto para `inputs/user_{uid}/{proj}/` (`streamlit_app.py:1374`), limite de 200 MB.
  - O mapeamento de colunas é adivinhado pelos `guess_*_col` (`:1394-1403`) sem confirmação do usuário.
- **Fazer:**
  - `storage.py` com `put/get/delete/list`, e duas implementações escolhidas por env: GCS (`gs://bucket`) ou diretório local.
  - Caminho `ws/{workspace_id}/sim/{simulation_id}/inputs/{sha256}.csv`; o sha256 garante a rastreabilidade do histórico.
  - Fluxo:
    1. criar a simulação (nome, KPI, alvo);
    2. subir investimento, performance e trends opcional;
    3. a API chama `engine.validate` e devolve o `ValidationReport`;
    4. a tela mostra `error` (bloqueia), `warning` (checkbox de confirmação) e `info`;
    5. o analista confirma o mapeamento de colunas;
    6. se houver colinearidade, a tela de agrupamento (A3) oferece agrupar ou bloquear por canal.
  - Para DV360/CM360 sobrepostos, o analista escolhe qual manter.
  - Templates CSV padrão disponíveis para download.
  - Upload do resultado de experimento (A7 › calibração).
  - Só CSV, ≤ 200 MB, nome sanitizado; nunca executar nada do arquivo.
- **Aceite:**
  - Upload com duplicata (data, canal) mostra o erro em pt-BR e o botão Rodar fica desabilitado.
  - O arquivo vai para o bucket do workspace certo.
  - Viewer não consegue subir arquivo.

## B3 · Parâmetros com sugestão do Gemini — semana 2

- **BRD:** Capabilities › Parameter configuration with Gemini assist ("explaining each suggestion; the analyst accepts or edits"); deliverables "Parameter input" e "Gemini AI".
- **Depende de:** A0 (`EngineConfig`), B2.
- **Hoje:**
  - Formulário em `streamlit_app.py:991-1235`, com campos e defaults tabelados.
  - Vários valores hardcoded (`:1405-1467`): `target_icpa=999999`, `max_events_to_analyze=3`, `post_event_days=14`.
  - Sugestão determinística em `suggest_causal_config` (`data_preprocessor.py:236`) e via Gemini em `suggest_form_fields` (`google_api.py:81`), que devolve só valores, **sem explicação**.
  - Chave e modelo do Gemini escolhidos pelo usuário no form.
- **Fazer:**
  - O formulário é gerado do JSON Schema do `EngineConfig`, com agrupamento e rótulos pt-BR definidos no front. A validação final é da API, com o mesmo modelo pydantic.
  - Todos os campos do BRD: alvo, KPI, ticket, CPA/iCPA/ROAS/iROAS, limit factor, p-value e os causais.
  - `POST /api/simulations/{id}/suggest` devolve `{campo: {value, reason, source: "gemini"|"dados"}}`. Ajustar o prompt de `suggest_form_fields` para incluir `reason`.
  - Na UI, o campo sugerido ganha um selo e a explicação; o analista aceita ou edita.
  - Chave Gemini da plataforma (Secret Manager); remover chave e modelo por usuário.
  - Gemini fora do ar → as sugestões determinísticas continuam funcionando.
- **Aceite:**
  - Sem Gemini, o form abre com as sugestões dos dados.
  - Valor inválido → 422 com mensagem do pydantic mapeada para o campo.

## B4 · Jobs assíncronos e log em tempo real — semana 2

- **BRD:** Capabilities › Processing engine & real-time log; Improvement Requests › Execution e Logs; deliverable "Real-time log".
- **Depende de:** A0, B2, B3.
- **Hoje:**
  - `subprocess.Popen` no mesmo host (`streamlit_app.py:1496-1535`), watchdog de 900 s.
  - Thread lê o stdout para uma fila; o render é throttled (≥ 25 linhas ou 0,4 s, `:107`) e mostra as últimas 300 linhas.
  - A tradução pt-BR fica em `logger.py` (`_RULES`, ~87 regras; `TranslationFilter`).
- **Fazer:**
  - Tabelas:
    - `simulations` (workspace, nome, criador);
    - `runs` com `id`, `simulation_id`, `status` (`queued → running → succeeded|inconclusive|failed`), `config jsonb` (snapshot), `inputs jsonb` (caminhos + sha256), `engine_version`, `results_uri`, `summary jsonb` (métricas-chave para listar e comparar), `published_at/by`, `created_by`, timestamps, `error`;
    - `run_logs` (`bigserial id`, `run_id`, `ts`, `level`, `message`).
  - `launcher` com duas implementações por env:
    - local, por subprocess `python -m api.worker <run_id>`;
    - cloud, pela API do Cloud Run Jobs com override de env `RUN_ID`.
  - Worker:
    1. baixa os inputs;
    2. roda `engine.run` com `on_log` que grava linhas traduzidas em `run_logs`, em lote a cada 0,5 s;
    3. escreve stdout em JSON para o Cloud Logging;
    4. sobe `results.json`;
    5. atualiza o run.
  - Timeout 15 min (config do Job). Worker morto → run `failed` (o API marca `running` sem heartbeat > 20 min como `failed`).
  - `GET /api/runs/{id}/logs/stream` em SSE, fazendo tail de `run_logs` por `id > last`.
  - Painel de log no front com as últimas 300 linhas e o "etapa em andamento há N s".
- **Aceite:**
  - Dois runs simultâneos em workspaces diferentes não misturam log nem arquivo.
  - Fechar a aba e voltar retoma o log de onde estava.
  - Erro do motor aparece em pt-BR no painel, com traceback só no Cloud Logging.

## B5 · Dashboards Causal Impact e Elasticidade — semana 3

- **BRD:** deliverables "Causal Impact dashboard" e "Elasticity dashboard"; Journeys 1 e 2; Wireframe 2; Capabilities › gates ("kept out of dashboards").
- **Depende de:** A1 (fixtures servem desde a semana 1), A8 (`scenario`).
- **Hoje:**
  - **Aba Causal** (`streamlit_app.py:1635-1898`): overview de eventos, seletor, veredito, 3 métricas, gráficos (linha real × contrafactual, barras de investimento e KPI, acurácia) e apêndices.
  - **Aba Elasticidade** (`:1902-2973`):
    - filtros na sidebar (orçamento, CPA, iCPA, ROAS, iROAS, KPI mín, % incrementalidade);
    - tabela Atual / Recomendado / Saturação;
    - banner de baixa confiança;
    - curvas de saturação, iCPA, receita/ROI;
    - mix por canal, comparação de saturação e curva individual por canal.
  - Builders Plotly puros em `scripts/dashboard_charts.py` (tabela de funções no mapeamento).
  - Há um hack de regex `align_insight_text` (`:2921`) que troca "15,8M" no texto; não portar.
- **Fazer:**
  - Rotas TanStack `/w/{ws}/runs/{id}/causal` e `/elasticity`.
  - Ler `results.json` via API.
  - Portar os builders para TS + plotly.js com a mesma aparência.
  - Filtros e slider chamam `GET /api/runs/{id}/scenario` com debounce de 200 ms; o front não faz conta nenhuma.
  - Mostrar os rótulos de gate:
    - evento `rejected` esmaecido, com o motivo;
    - canal `identifiable=false` sem curva própria, com o aviso;
    - run `inconclusive` mostra o card do experiment design no lugar dos resultados;
    - bandas de confiança e marcação de extrapolação nas curvas;
    - degraus de `ramp` na tabela de cenários.
  - O relatório Gemini aparece com o selo de grounding.
  - Viewer só vê runs publicados.
- **Aceite:**
  - Paridade visual e numérica com o Streamlit no mesmo run (checklist por gráfico).
  - Nenhum número de canal não identificável na tela.
  - Slider responde em < 500 ms.

## B6 · Histórico de simulações — semana 3

- **BRD:** Capabilities › Simulation history & traceability; deliverable "Simulation history"; Journey 1 passo 4; Improvement Requests › Outputs ("re-runs overwrite results" hoje).
- **Depende de:** B4.
- **Hoje:** "Projetos Anteriores" na sidebar com renomear e apagar (`streamlit_app.py:854-971`). Re-run sobrescreve.
- **Fazer:**
  - Lista de runs do workspace com busca por nome, filtro por status, autor e período, e colunas do `summary`.
  - Detalhe com config, inputs (sha256, baixar), versão do motor e resultados.
  - "Rodar de novo" clona a config.
  - Publicar/despublicar: revisão humana antes de o Viewer ver (BRD › In scope).
  - **Comparar dois runs** lado a lado (métricas do `summary`). Cortável: o 4º da ordem de corte.
  - **Modo demo** (Journey 4): workspace "Demo" com dataset anonimizado e um run publicado, criado por um script de seed.
- **Aceite:** um re-run cria um novo registro sem tocar no anterior. Todo run mostra a versão do motor e o hash dos inputs.

## B7 · Export CSV/PDF — semana 3

- **BRD:** Capabilities › Dashboards ("export to PDF/CSV"); Journey 2 passo 3; Figure 3 passo 14.
- **Depende de:** B5.
- **Hoje:** nenhum botão de download no Streamlit. Os HTMLs existem em disco, mas não são oferecidos.
- **Fazer:**
  - CSV de eventos, curva de resposta e cenários, gerado pela API a partir do `results.json`.
  - PDF pelo CSS `@media print` nas telas de dashboard e relatório, com `window.print()`.
  - O cabeçalho impresso traz workspace, run, data e status de grounding.
- **Aceite:** o PDF de um run publicado sai legível em A4, sem cortar gráfico.

## B8 · Conectores Google Ads e GA4 — semanas 2–3

- **BRD:** Technical Details §1 (inclusive o fallback); Capabilities › Google connectors; deliverable "Google connectors"; Datasets.
- **Depende de:** projeto GCP e acesso das contas (README › dependências), B2.
- **Hoje:** só CSV.
- **Fazer (plano padrão = fallback):**
  - Por workspace, o Admin registra:
    - o dataset BigQuery do **Data Transfer Service do Google Ads** (customer id via MCC Almap);
    - o dataset do **export nativo do GA4** (property id e evento/métrica do KPI).
  - A carga diária é agendada pelo próprio DTS.
  - Na criação da simulação, a fonte "Google" roda duas queries:
    - spend diário por tipo de campanha (Search, PMax, YouTube, Demand Gen; mapeamento `advertising_channel_type` → canal fixo no código);
    - KPI diário do GA4.
  - O resultado é materializado nos mesmos CSV canônicos e passa pelo **mesmo** `engine.validate`.
  - Acesso de leitura via service account.
  - **OAuth nativo** (Ads API + GA4 Data API, refresh token no Secret Manager): só se o developer token sair a tempo; é o último da ordem de corte.
- **Aceite:** num workspace com DTS configurado, criar simulação sem upload gera dados que passam no `validate` e batem com o total do Google Ads UI no período (±1%).
- **Em aberto:** o DTS por cliente fica no projeto Omnicom ou no do cliente? Quem configura o export GA4 de cada Core?

## B9 · Uso, logs e retenção — semana 3

- **BRD:** Capabilities › Mathematical audit & usage tracking ("usage table"); Journey 3 passo 2; Improvement Requests › Usage tracking e Logs; Figure 2 › Governance.
- **Depende de:** B4.
- **Fazer:**
  - View SQL `usage` com runs por workspace × usuário × mês × status, derivada de `runs`, sem tabela nova. Tela para o platform admin, com export CSV.
  - Logs estruturados JSON no stdout (Cloud Logging) com `run_id`, `workspace_id`, `user_id`.
  - Ação de admin "apagar simulação": apaga as linhas e os arquivos no bucket. Política de retenção no Em aberto.
  - Cópia analítica no BigQuery (BRD › Storage: metadata do run, curva, KPIs por cenário): ao terminar, o worker faz append em 3 tabelas. Cortável: o 6º da ordem de corte.
- **Em aberto:** prazo de retenção de dados de cliente (LGPD) — perguntar ao Data Tech.

## B10 · Identidade visual e integração com o Almap Metrics — semana 1 e contínuo

- **BRD:** deliverables "Metrics integration" e "Visual synergy"; Improvement Requests › Look & feel; Journey 2 ("Opens the Elasticity dashboard inside Metrics").
- **Hoje:** CSS próprio no Streamlit (`streamlit_app.py:343-579`) e logos DASH/AlmapBBDO em `Logos/`.
- **Fazer:**
  - Semana 1: aprovar o layout com o time Metrics, que é o scope freeze.
  - Se o Metrics tiver biblioteca de componentes React ou tokens, consumir; senão, tokens CSS próprios espelhando o Metrics. Ao implementar, seguir a regra **UI sources** do CLAUDE.md global.
  - Definir o modo de integração e implementar o mínimo dele.
- **Em aberto (bloqueia o design):** como o GEM entra no Metrics — link no menu, iframe embutido ou módulo do mesmo app? E o login, é o mesmo do Metrics ou separado? O BRD pede login/senha próprio.

## B11 · Deploy GCP, QA e performance — semana 4

- **BRD:** Phase 2 Week 4; deliverable "Tests & QA"; Process ownership (Cloud Run, BigQuery, GCS); Dependencies (GCP).
- **Depende de:** projeto GCP.
- **Fazer:**
  - `deploy/` com scripts `gcloud` idempotentes, sem Terraform:
    - Cloud Run service `gem-api`;
    - Cloud Run Job `gem-worker`;
    - Cloud SQL Postgres;
    - bucket GCS;
    - secrets (DB, Gemini, chave de sessão);
    - service accounts de privilégio mínimo.
  - Ambientes staging e prod. Migrações Alembic rodam antes do deploy da API.
  - Teste de carga: 10 runs simultâneos e 30 usuários navegando, e anotar os tempos.
  - Homologação com o Metrics: checklist das 4 jornadas do BRD.
  - **Remover o Streamlit**: `scripts/streamlit_app.py`, `scripts/db.py`, o CSS e as deps `streamlit*`.
  - Guia do usuário v1, com a parte de interpretação vinda da trilha A.
- **Aceite:**
  - As 4 jornadas do BRD rodam ponta a ponta em staging.
  - Rollback = redeploy da revisão anterior do Cloud Run, testado uma vez.
