# Plano — Ingestão Confiável v2 (datas, cadência, projeção)

**Data:** 2026-07-27
**Branch:** `feat/robust-csv-ingestion`
**Gatilho:** `exemplo_csv/investimento_pmax_semanal.csv` + `exemplo_csv/performance_pmax_semanal.csv` travam a UI e produzem datas sem sentido.

---

## 1. Diagnóstico (verificado, não suposto)

Rodei os dois arquivos pela engine offline real
(`scripts/local_main-without-gemini.py`). O CLI termina em **24s** com o
resultado errado; a UI é que trava. Cadeia completa:

### 1.1 Parsing de data embaralha DD/MM silenciosamente — causa raiz

Os arquivos são `;`-separados, com BOM, datas `DD/MM/YYYY`, cadência semanal
(domingos). `robust_date_parsing` (`scripts/data_preprocessor.py:39`) não tem
detecção de `dayfirst`. Nenhum formato explícito bate, então o pandas cai no
parser por linha (dateutil, month-first), que inverte dia e mês sempre que o
dia é ≤ 12:

```
01/01/2025 → 2025-01-01  (ok por coincidência)
05/01/2025 → 2025-05-01  ← deveria ser 05/jan
12/01/2025 → 2025-12-01  ← deveria ser 12/jan
19/01/2025 → 2025-01-19  (ok, dia > 12)
```

O resultado tem **0 NaT**, então nenhum guard-rail existente dispara.

| | real | observado |
|---|---|---|
| range de datas | 2025-01-01 → 2026-06-28 | 2025-01-01 → **2026-12-04** |
| eventos detectados | 12 | **28** |
| cadência | 7 dias, domingos | caótica |

**Agravante:** `scripts/streamlit_app.py:1301` chumba
`date_formats: {"investment_file": "%Y-%m-%d", ...}` para **todo** upload,
ignorando o arquivo. O formato configurado sempre falha e joga a coluna
exatamente no fallback que embaralha.

### 1.2 `KeyError: 'Projected_Revenue'` zera 100% da análise causal

Bug independente do CSV, mas mascara qualquer teste de ingestão.

`scripts/analysis.py:927-933`: o filtro de ROAS entra quando `min_roas > 0`,
mas `Projected_Revenue` só é criada dentro de `if revenue_mode:` (linha 828).
Em modo `CONVERSIONS` com qualquer `target_roas` preenchido:

```
run_opportunity_projection() → KeyError → except → retorna ({}, {}, ...)
  → projection_model_params = {}
    → run_causal_impact_analysis() → KeyError: 'alpha'   × todos os eventos
```

No meu run: 28 eventos, 28 falhas idênticas, zero saída causal. A UI
(`streamlit_app.py:1285`) preenche `target_roas` mesmo em modo CONVERSIONS,
então isso atinge qualquer cliente de conversão que digitar uma meta de ROAS.

### 1.3 Dados semanais tratados como diários

- `post_event_days = 14` (chumbado em `streamlit_app.py:1298`) = **2 linhas**
  pós-evento numa série semanal. Poder estatístico ~zero.
- `min_pre_period_days` compara contra `len(pre_data)`, ou seja conta linhas,
  não dias — semanticamente errado fora do regime diário.
- `analysis.py:39-56` / `elasticity_analysis.py:51-66` geram `dayofweek`,
  `is_weekend`, `is_payday_period`, `is_holiday`, `day_0..day_6`. Com datas
  corretas todas são constantes (só domingos) e o filtro de variância as
  descarta. Com datas **embaralhadas** viram ruído aleatório e o LassoCV
  chega a selecioná-las.
- `elasticity_analysis.py:612` faz `df[col].mean()` e chama de
  `avg_daily_spend`. Com linhas semanais isso é gasto **semanal**.
  `gemini_report.py:804,810,813,815` e `streamlit_app.py:1726` multiplicam
  esse valor por `30` para virar "mensal" → superestimam o orçamento em ~4,3×.

### 1.4 Semanas parciais na virada do ano

`01/01/2025` (quarta, cobre 4 dias) e `01/01/2026` (quinta, cobre 3 dias)
entram como linhas semanais cheias. Investimento cai ~50% e KPI cai junto →
`find_events` marca queda significativa. Eventos falsos.

Note que `01/01/2026` fica **no meio** da série (entre `28/12/2025` e
`04/01/2026`), não na borda.

### 1.5 `guess_kpi_col` escolhe coluna de texto

`data_preprocessor.py:330`: `"leads"` não está em `COLUMN_NAME_HINTS["kpi"]`.
Se o KPI digitado pelo usuário não bater exatamente, o fallback é
`df.columns[1]` — que neste arquivo é `Canal` (string `"Pmax"`).

Verificado: `guess_kpi_col(perf, "Conversions")` → `"Canal"`. KPI vira
all-NaN → `dropna` esvazia o df → erro "Nenhuma linha válida restou".

### 1.6 Carregamento infinito da UI

`scripts/streamlit_app.py:1391-1397`:

```python
for line in iter(process.stdout.readline, ""):
    full_log += line
    log_container.code(f"...\n{full_log}", language="shell", height=400)
```

Re-renderiza o log **inteiro acumulado** a cada linha de stdout. Custo
O(n²) em bytes pelo websocket. Com 28 eventos × traceback completo o
navegador fica preso renderizando. Não há timeout nem kill do subprocess.

O mesmo run no CLI: 24 segundos.

### 1.7 Coluna de canal no arquivo de performance é ignorada

O arquivo de performance tem `Data;Canal;Leads`, mas o pipeline trata
performance como série única total. Aqui só existe um canal, então passa;
com dois ou mais, linhas de canais diferentes na mesma data entram
duplicadas sem agregação nem aviso.

---

## 2. Escopo fechado

Decisões confirmadas com o usuário:

| Tema | Decisão |
|---|---|
| Escopo | Os 4 blocos: ingestão, `Projected_Revenue`, cadência semanal, log da UI |
| Granularidade | **Detectar e adaptar** — inferir passo, converter janelas, desligar features de calendário |
| Data ambígua | **Falhar alto** com mensagem clara — nunca chutar em silêncio |
| Período parcial | **Detectar e descartar** — duração < cadência sai da série |

**Desvio deliberado (§1.4):** a opção escolhida dizia "descartar das bordas",
mas o caso real `01/01/2026` é interior. A regra implementada descarta
períodos curtos **em qualquer posição**, que é o que a descrição da opção
pedia ("períodos com duração menor que a cadência saem da série; evita
eventos falsos de -50% na virada do ano"). Se preferir restringir só às
bordas, é um `if` a menos.

**Fora de escopo:** renomear `Daily_Investment` (usado em ~20 lugares,
exports CSV e dashboard — churn alto, ganho analítico zero). A correção é
nas extrapolações, não no nome. Agregação multi-canal do arquivo de
performance (§1.7) entra apenas como **aviso**, não como feature.

---

## 3. Plano de execução

Ordem importa: a Fase 0 destrava o teste de tudo o mais.

### Fase 0 — Destravar a análise causal

**Arquivo:** `scripts/analysis.py`

1. Linha 927: só aplicar filtros de ROAS quando `revenue_mode` for verdadeiro.
   Em modo CONVERSIONS, `target_roas`/`target_iroas` não têm significado —
   ignorar com um `print` de aviso uma única vez.
2. Linha 993: no `except`, logar qual configuração causou a falha antes de
   devolver tuplas vazias.
3. `run_causal_impact_analysis`: se `model_params` vier vazio, falhar com
   mensagem explícita (`"modelo de projeção não treinou — análise causal
   abortada"`) em vez de estourar `KeyError: 'alpha'` 28 vezes.

**Arquivo:** `scripts/streamlit_app.py:1282-1287`
Não gravar `target_roas` quando `optimization_target == "CONVERSIONS"` e
`kpi_is_monetary` for falso.

**Verificação:**
```
mise exec -- python scripts/local_main-without-gemini.py --config <cfg CONVERSIONS+target_roas>
```
Nenhum `KeyError: 'alpha'` no log; `response_curve_data.csv` gerado.

**Teste:** `tests/test_analysis.py::test_projection_conversions_mode_with_roas_target`
— config CONVERSIONS + `target_roas=4` retorna `model_params` com chave `alpha`.

---

### Fase 1 — Parsing de data determinístico

**Arquivo:** `scripts/data_preprocessor.py`

Substituir o fallback por-linha de `robust_date_parsing` por tentativa de
formatos candidatos sobre a **coluna inteira**:

```
%Y-%m-%d, %d/%m/%Y, %m/%d/%Y, %d-%m-%Y, %m-%d-%Y, %Y/%m/%d, %d.%m.%Y
```

Regras:
- Formato do config tem prioridade; só é aceito se cobrir 100% das linhas não-nulas.
- Testar cada candidato com `errors="coerce"`; manter os que dão 0 NaT.
- **1 vencedor** → usar, logar qual foi.
- **`%d/%m` e `%m/%d` empatam** (todos os dias ≤ 12) → `ValueError` com a
  mensagem: coluna, 3 exemplos de valores, e instrução de definir
  `date_formats` no config / escolher o formato na UI.
- **0 vencedores** → `ValueError` listando exemplos das linhas que falharam.

O parser por linha (`format="mixed"` / dateutil) **sai**. Foi ele que criou
o problema.

**Arquivo:** `scripts/streamlit_app.py:1301-1305`
Remover os `"%Y-%m-%d"` chumbados — gravar `null` e deixar a detecção decidir.
Se a detecção levantar ambiguidade, mostrar `st.error` com a mensagem do
`ValueError` (que já é acionável).

**Verificação:**
```
mise exec -- python -c "... robust_date_parsing(read_csv_robust('exemplo_csv/investimento_pmax_semanal.csv')['Data'])"
```
Esperado: `min=2025-01-01`, `max=2026-06-28`, 0 NaT, log `formato %d/%m/%Y`.

**Testes** (`tests/test_data_preprocessor.py`):
- `%d/%m/%Y` com dias > 12 → resolve dayfirst
- `%m/%d/%Y` com meses > 12 → resolve monthfirst
- só dias ≤ 12 → `ValueError` mencionando ambiguidade
- ISO `%Y-%m-%d` continua funcionando (não regredir)
- formato do config que não cobre tudo → cai na detecção, não silencia

---

### Fase 2 — Detecção de cadência + poda de períodos parciais

**Arquivo:** `scripts/data_preprocessor.py`

Nova função `detect_cadence(dates) -> int` (dias): mediana do `diff()` das
datas únicas ordenadas. Retorna 1 / 7 / 28-31, ou o valor bruto.

Nova função `drop_partial_periods(df, cadence)`: descarta linhas cuja
distância até a próxima data seja `< 0.6 * cadence`, em qualquer posição.
Logar quantas e quais saíram.

Em `load_and_prepare_data`:
- detectar cadência do arquivo de investimento;
- se investimento e performance divergirem, erro claro (não tentar conciliar);
- aplicar `drop_partial_periods` nos dois;
- gravar `config["period_days"] = cadence` para o resto do pipeline;
- logar `"Cadência detectada: semanal (7 dias), 80 períodos, 2 parciais descartados"`.

**Verificação:** nos arquivos de exemplo — cadência 7, exatamente 2 linhas
descartadas (`2025-01-01`, `2026-01-01`), 78 restantes.

**Testes:** série diária → 1; série semanal → 7; semanal com stub no meio →
stub removido; série com cadências divergentes entre os dois arquivos → erro.

---

### Fase 3 — Adaptar o pipeline à cadência

Consumidores de `config["period_days"]`:

**`scripts/analysis.py`**
- `create_features` (l.39-56): só gerar `dayofweek`/`day_i`/`is_weekend`/
  `is_payday_period` quando `period_days == 1`. `is_holiday` idem. `month`
  continua sempre.
- `find_events` (l.93-98): pular o re-agrupamento `to_period("W-MON")`
  quando os dados já são semanais ou mais grossos — hoje ele desloca os
  buckets de domingo para segunda.
- Janelas: `post_event_periods = max(4, ceil(post_event_days / period_days))`;
  `min_pre_periods = max(8, ceil(min_pre_period_days / period_days))`.
  Se a janela em dias do usuário render menos que o mínimo, avisar no log
  que o mínimo foi aplicado.

**`scripts/elasticity_analysis.py`**
- `create_features` (l.51-66): mesmo gate.
- `avg_daily_spend` (l.612, l.1027): manter o cálculo, documentar que é
  **por período**.

**Extrapolações mensais** — trocar `* 30` por `* (30 / period_days)`:
- `scripts/gemini_report.py:804, 810, 813, 815`
- `scripts/streamlit_app.py:1726`

**Rótulos de UI:** onde hoje se lê "diário", usar o rótulo da cadência
detectada ("semanal"/"mensal"). Sem renomear colunas de dados.

**Verificação:** run completo nos arquivos de exemplo. Conferir que o
orçamento mensal no relatório bate com `média_semanal × 30/7`, não
`média_semanal × 30`.

**Testes:** `create_features` com cadência 7 não emite colunas `day_*`;
conversão de janela com `period_days=7, post_event_days=14` → 4 (mínimo
aplicado, com aviso).

---

### Fase 4 — Resolução de colunas mais segura

**Arquivo:** `scripts/data_preprocessor.py`

1. `COLUMN_NAME_HINTS["kpi"]`: adicionar `leads`, `lead`, `vendas`, `sales`,
   `sessoes`, `sessões`, `pedidos`, `orders`, `transacoes`, `transações`.
2. `guess_kpi_col` / `guess_investment_col`: antes do fallback posicional,
   ler ~50 linhas e descartar candidatas que não convertem para número via
   `robust_numeric_parsing`. Se nenhuma coluna numérica sobrar, devolver o
   palpite do usuário e deixar a validação falhar com mensagem boa.
3. Validação pós-carga: se a coluna de KPI ou de investimento virar
   >50% NaN, erro nomeando a coluna, o arquivo e 3 valores de exemplo — em
   vez do genérico "Nenhuma linha válida restou".
4. `load_and_prepare_data:586`: usar `raise ... from e` para preservar o
   traceback original.
5. §1.7: se o arquivo de performance tiver uma coluna que resolva como
   canal com mais de um valor distinto, emitir `AVISO` de que as linhas
   serão somadas por data (e somar explicitamente, em vez de deixar o
   merge duplicar).

**Verificação:** `guess_kpi_col('exemplo_csv/performance_pmax_semanal.csv',
'Conversions')` → `"Leads"`, não `"Canal"`.

**Testes:** hint de `leads`; fallback posicional pula coluna de texto;
coluna de KPI majoritariamente não-numérica → erro nomeando a coluna.

---

### Fase 5 — Streaming de log da UI

**Arquivo:** `scripts/streamlit_app.py:1390-1400`

- Acumular em `list` e re-renderizar no máximo a cada **0,4 s** ou a cada
  **25 linhas**, o que vier primeiro.
- Exibir apenas as **últimas 300 linhas** (`"\n".join(lines[-300:])`); o log
  completo já vai para arquivo via `LogContext`.
- Render final garantido após o loop.
- `process.wait(timeout=900)`; no estouro, `process.kill()` + `st.error`
  com as últimas linhas.

**Verificação:** rodar a UI (`mise run dev`) com os arquivos de exemplo e
confirmar que o painel de log atualiza e o run conclui.

Depois das fases 0-4 o volume de log já cai muito (12 eventos em vez de 28,
sem traceback por evento), mas o O(n²) precisa sair de qualquer jeito.

---

## 4. Aceitação

Rodar os dois arquivos de `exemplo_csv/` ponta a ponta pela UI e conferir:

1. Range de datas exibido: **2025-01-05 → 2026-06-28** (após poda das parciais).
2. Log informa `cadência semanal (7 dias)` e `2 período(s) parcial(is) descartado(s)`.
3. Eventos detectados: **12** (não 28).
4. Nenhum `KeyError` no log.
5. Análise causal produz resultado para pelo menos um evento (ou reprova por
   critério estatístico explícito — não por exceção).
6. Orçamento mensal do relatório ≈ `média semanal × 4,3`, não `× 30`.
7. A UI conclui sem travar; painel de log responsivo.

Regressão: `mise run test` verde, incluindo os testes existentes de
`test_data_preprocessor.py` (formato ISO, BR/US numérico, delimitadores).

---

## 5. Arquivos tocados

| Arquivo | Fases |
|---|---|
| `scripts/data_preprocessor.py` | 1, 2, 4 |
| `scripts/analysis.py` | 0, 3 |
| `scripts/elasticity_analysis.py` | 3 |
| `scripts/streamlit_app.py` | 0, 1, 3, 5 |
| `scripts/gemini_report.py` | 3 |
| `tests/test_data_preprocessor.py` | 1, 2, 4 |
| `tests/test_analysis.py` (novo) | 0, 3 |

Fase 0 é independente e pode ir sozinha se você quiser um fix rápido em
produção antes do resto.
