# M2 — Plano de trabalho da plataforma Marketing GEM

Plano do Milestone 2 do BRD *"[Omnicom - AlmapBBDO] BRD - Marketing GEM - 2026 v2.docx"* (raiz do repo, não versionado): transformar o protótipo Streamlit numa plataforma web multi-marca, segura e auditada, até **18/nov/2026**.

Time: **Dev A** (motor/estatística), **Dev B** (plataforma/full-stack) e Claude (specs + implementação via subagentes, revisada pelos devs).

- Trilha A, motor e gates: [`trilha-a-motor.md`](trilha-a-motor.md)
- Trilha B, plataforma: [`trilha-b-plataforma.md`](trilha-b-plataforma.md)

Este arquivo guarda o que as duas trilhas compartilham: decisões, arquitetura, o **contrato** entre A e B, calendário e cortes. Referências `arquivo:linha` valem para o commit `bbad5af` (antes do item A0 mover `scripts/` para `engine/`); confira a linha antes de citá-la num spec.

---

## Como escrever um spec a partir deste plano

1. Leia este README inteiro e **só o item** da trilha correspondente (ex.: `A3`).
2. Leia no BRD as seções citadas em **BRD:** do item. Para extrair o texto do docx:
   `python -I -c "import zipfile,re,sys;x=zipfile.ZipFile(sys.argv[1]).read('word/document.xml').decode();print('\n'.join(t for t in(''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>',p))for p in re.findall(r'<w:p[ >].*?</w:p>',x,re.S))if t.strip()))" "<caminho do docx>"`
3. Leia o código citado em **Hoje:** e confirme que ainda é verdade.
4. Leve as perguntas de **Em aberto:** ao usuário antes de fechar o design.
5. Escreva o spec em `docs/superpowers/specs/AAAA-MM-DD-<id>-<slug>-design.md` (fluxo `superpowers:brainstorming`). Cada linha de **Aceite:** do item vira pelo menos um teste ou verificação no spec.

O spec está pronto quando: todo critério de **Aceite** tem verificação concreta, toda pergunta **Em aberto** tem resposta registrada, e todo campo novo do contrato (abaixo) está escrito no spec com nome e tipo.

---

## Decisões tomadas (20/out/2026)

| Tema | Decisão | Por quê |
|---|---|---|
| Motor | **Reaproveitar** o Python atual como pacote `engine/` | ~5k linhas já auditadas (julho); reescrever é risco sem ganho |
| UI | **Substituir Streamlit** por React | BRD exige visual do Almap Metrics (React), RBAC e jobs assíncronos — Streamlit não entrega |
| Front | Vite + React + TypeScript + TanStack Router + TanStack Query, gráficos com plotly.js | Metrics é React; plotly.js permite portar `dashboard_charts.py` quase 1:1 |
| API | FastAPI, servindo também o build estático do front (mesma origem, sem CORS) | um serviço Cloud Run só |
| Execução | Cloud Run **Jobs** (worker) — mesma imagem Docker da API, outro comando | BRD: "cloud execution as asynchronous jobs" |
| Banco do app | **Postgres** (Cloud SQL), SQLAlchemy 2 + Alembic | SQLite em disco não sobrevive ao Cloud Run e trava com escrita concorrente |
| Arquivos | Cloud Storage (local: diretório em disco, mesma interface) | BRD: uploads, snapshots, relatórios |
| BigQuery | Dados de mídia (Ads/GA4) e **cópia analítica** do histórico. Postgres é a fonte de verdade de runs | BigQuery não serve como banco transacional |
| Gemini | Chave única da plataforma no Secret Manager (fim da chave por usuário). SDK `google-genai` (aceita API key e Vertex) | BRD fala em Vertex AI; `google-generativeai` está descontinuado e não tem `response_schema` confiável |
| Matemática | **Fica só em Python.** O front nunca recalcula curva, CPA ou ponto ótimo; pede à API | a auditoria cobre um lugar só |
| Streamlit | Continua funcionando até o React atingir paridade (fim da semana 3), depois é removido | é o plano B de demo e a referência de paridade |
| Conectores Google | Começar pelo **fallback** do BRD (BigQuery Data Transfer + export nativo do GA4); OAuth próprio só se o developer token sair a tempo | 2 devs; o BRD autoriza o fallback |
| Export PDF | CSS de impressão + `window.print()` | nativo, zero dependência |

---

## Arquitetura alvo

```
web/ (React SPA) ──HTTP/SSE──▶ api/ (FastAPI, Cloud Run service)
                                 │  auth, RBAC, upload, validação síncrona,
                                 │  sugestão Gemini, runs, histórico, cenários
                                 ├──▶ Postgres (Cloud SQL): users, workspaces, runs, run_logs
                                 ├──▶ Cloud Storage: inputs/, results.json, artefatos
                                 └──dispara──▶ worker (Cloud Run Job, mesma imagem)
                                                  └─ engine.run(...) → results.json
                                                     logs → run_logs (UI) + stdout JSON (Cloud Logging)
BigQuery: tabelas de Ads/GA4 (DTS) ──consulta──▶ api/ingestão  ·  worker ──append──▶ histórico analítico
```

Layout do repo:

```
engine/        # Trilha A. Ex-scripts/ (git mv, mesmos nomes de módulo). Sem I/O de rede além do Gemini.
api/           # Trilha B. FastAPI, worker.py, storage.py, migrations/ (Alembic)
web/           # Trilha B. Vite + React
deploy/        # Trilha B. scripts gcloud (sem Terraform)
tests/         # pytest, ambos
docs/m2/       # este plano
```

Um único `pyproject.toml`/`uv.lock` na raiz, e a imagem Docker instala a partir do lock. Hoje a imagem usa `requirements.txt`, que diverge do `pyproject.toml`; eliminar essa duplicação faz parte do B0.

---

## O contrato A ↔ B

A trilha B nunca lê CSV, PNG ou HTML do motor. Ela só conhece as quatro superfícies abaixo, e qualquer mudança nelas exige aviso à outra trilha e atualização desta seção.

### 1. `EngineConfig` (dono: A, item A0)
Modelo pydantic em `engine/config.py`, **única fonte** dos parâmetros e defaults. A API usa o mesmo modelo para validar o formulário e o front recebe o JSON Schema dele (`EngineConfig.model_json_schema()`), assim nada de defaults duplicados no TypeScript. Campos mínimos, pelo BRD: `optimization_target` (REVENUE|CONVERSIONS), coluna de KPI, `kpi_is_monetary`, `average_ticket`, `conversion_rate_from_kpi_to_bo`, `financial_targets` (target_cpa, target_icpa, target_roas, target_iroas), `investment_limit_factor` (default **1.5**), `p_value_threshold`, parâmetros causais (`min_pre_period_days`, `r_squared_threshold`, `increase/decrease_threshold_percent`, `require_*`), `column_mapping`, `channel_groups`, `controls`, `gate_thresholds`, `max_budget_step` (default 0.20).

### 2. `engine.validate(inputs, config) -> ValidationReport` (dono: A, item A2)
Síncrona e rápida (< 10 s para 2 anos diários × 15 canais). A API chama no upload, antes de existir um run.
```json
{"ok": false,
 "cadence": "daily", "period_days": 1, "date_range": ["2024-10-01","2026-09-30"],
 "channels": ["SEARCH","PMAX","YOUTUBE"],
 "checks": [{"id": "duplicate_date_channel", "level": "error",
             "message": "12 linhas repetidas de (data, canal) no investimento",
             "details": {"rows": [...]}}],
 "collinearity": {"pairs": [{"a":"PMAX","b":"SEARCH","r":0.91}], "vif": {"PMAX": 14.2},
                  "suggested_groups": [["PMAX","SEARCH"]]}}
```
`level`: `error` bloqueia o run; `warning` exige confirmação do analista; `info` só informa. `message` sai em pt-BR, pronto para a tela. `id` é estável, e o front pode usá-lo para lógica específica (ex.: escolher entre DV360 e CM360).

### 3. `engine.run(inputs, config, out_dir, on_log) -> dict` = `results.json` (dono: A, itens A0/A1)
Cada seção abaixo traz `status` e motivos. **Publicar é decisão do motor**: o que falhou num gate vem marcado, e a API/front só escondem ou rotulam, sem recalcular nada.
```json
{"schema_version": 1, "engine_version": "<git sha>", "status": "succeeded|inconclusive|failed",
 "validation": { ...ValidationReport... },
 "sufficiency": {"passed": true, "spend_variation": {...}, "validated_events": 3, "pre_wmape": 0.08},
 "causal": {
   "events_detected": [{"date":"2025-08-18","channels":["YOUTUBE"],"investment_change_pct":89.7}],
   "events": [{"id":"YOUTUBE_2025-08-18","channels":["YOUTUBE"],"start":"…","end":"…",
               "status":"published|rejected","reasons":["placebo_failed"],
               "lift":{"abs":4640000,"ci":[…,…],"rel_pct":12.1},"p_value":0.01,
               "incremental_investment":3400000,"cpa_incremental":…,"fit":{"r2":…,"wmape":…},
               "placebo":{"p_value":0.62,"passed":true},
               "series":{"line":[…],"accuracy":[…],"investment":[…],"kpi":[…]}}]},
 "elasticity": {
   "metrics": {"cv_r2":…, "r2":…, "media_contribution_pct":…, "low_confidence":false, "reallocation_suppressed":false},
   "channels": [{"name":"YOUTUBE","identifiable":true,"coef_ci":[…,…],"hill":{"K":…,"n":…},"adstock":…,
                 "observed_spend_range":[…,…],"curve":[{"x":…,"y":…,"lo":…,"hi":…,"extrapolated":false}]}],
   "groups": [{"name":"PMAX + SEARCH","members":["PMAX","SEARCH"]}],
   "response_curve": [{"investment":…,"kpi":…,"kpi_lo":…,"kpi_hi":…,"revenue":…,"icpa":…,"iroas":…,
                       "extrapolated":false,"mix":{"YOUTUBE":…}}],
   "scenarios": [{"name":"current|recommended|saturation","investment":…,"kpi":…,"kpi_lo":…,"kpi_hi":…,
                  "cpa":…,"roas":…,"extrapolated":false,"ramp":[{"step":1,"investment":…}]}]},
 "experiment_design": null,
 "reports": {"events": {"<event id>": {…narrativa…}}, "global": {…},
             "grounding": {"status":"passed|regenerated|blocked","issues":[…]}}}
```
O formato exato de cada seção é fechado no spec do item dono, e os nomes acima são o ponto de partida. Até o fim da semana 1, A entrega **fixtures** reais em `tests/fixtures/results/` (gerados a partir de `exemplo_csv/`) para B construir telas sem esperar o motor.

### 4. `engine.scenario(results, filters) -> dict` (dono: A, item A8)
Recalcula ponto ótimo, ponto de saturação e tabela de cenários quando o usuário mexe em slider ou filtro (target CPA/ROAS/iCPA/iROAS, faixa de orçamento). Substitui a lógica que hoje mora em `streamlit_app.py:2208-2366` e `dashboard_charts.py:15-78`. A API expõe como `GET /runs/{id}/scenario?...`.

---

## Calendário e pontos de sincronização

| Semana | Trilha A | Trilha B | Sync na sexta |
|---|---|---|---|
| **1** · 19–23/out | A0 pacote `engine` · A1 `results.json` + fixtures | B0 esqueleto · B1 auth/RBAC (início) · B10 layout com Metrics | `EngineConfig` + fixtures entregues; layout aprovado (scope freeze) |
| **2** · 26–30/out | A2 validação/dedup · A3 colinearidade · A4 controles causais | B1 fim · B2 upload · B3 parâmetros/Gemini · B4 jobs + log | upload → validate → run → log fim-a-fim **local** |
| **3** · 2–6/nov | A5 identificabilidade · A6 suficiência + placebo · A7 experiment designer · A8 guardrails/cenários · A9 grounding | B5 dashboards · B6 histórico · B7 export · B8 conectores · B9 uso | todos os gates no `results.json`; React em paridade com Streamlit; 1º deploy staging |
| **4** · 9–13/nov | A10 auditoria matemática | B11 deploy prod, QA/perf com Metrics · remove Streamlit | auditoria aprovada; prod no ar |
| buffer · 16–18/nov | correções | homologação, guia do usuário v1 | **M2: demo ao Google** |

Em paralelo: o onboarding das 4 marcas Core vai de 26/out a 13/nov. Rode `engine.validate` (A2) nos dados reais delas assim que o A2 existir; é o melhor teste dos gates.

---

## Dependências externas (bloqueiam a semana 2/3 se não andarem)

| O quê | Quem destrava | Prazo | Sem isso |
|---|---|---|---|
| Projeto GCP Omnicom aprovado (Cloud Run, SQL, GCS, BigQuery, Secret Manager, Vertex) | Marcio (Data Tech) | antes de 26/out | sem deploy nem staging; tudo roda só local |
| Google Ads developer token (Basic) + OAuth consent (adwords, analytics.readonly) | Data Tech | pedir na semana 1 | usar DTS + export GA4 (plano padrão) |
| Contas Google Ads das Core no MCC Almap + acesso GA4 | Account teams | 26/out | CSV |
| Tokens visuais / design system do Almap Metrics | time Metrics | semana 1 | tema próprio, ajuste depois |
| Critérios numéricos de aprovação da auditoria | Data Science | antes de 9/nov | "auditoria aprovada" fica subjetivo |

---

## Ordem de corte (se apertar, corte de cima para baixo)

1. Séries do Banco Central (A4) — Trends via upload já cobre controle exógeno.
2. Status de learning/budget-limited do Google Ads no planner (A8) — depende do OAuth.
3. Uso do resultado de experimento como calibração (A7) — guardar o upload basta.
4. Comparação lado a lado no histórico (B6) — busca e filtro bastam.
5. Overlap DV360/CM360 (A2) — vira warning manual.
6. Cópia do histórico no BigQuery (B9) — Postgres basta para o M2.
7. OAuth nativo Ads/GA4 (B8) — o fallback DTS já é o padrão.

**Nunca cortar:** gates de publicação, RBAC por workspace, jobs na nuvem, grounding do Gemini e auditoria. É isso que o Google homologa no M2.

---

## Fora do M2 (BRD, "Feature Requests — Post Launch")
APIs de Meta/DV360/CM360, runs agendados com alerta, visão de portfólio, integração com Meridian, export para Slides/PowerPoint, multi-KPI e multi-mercado, self-service onboarding.
