# Mais gráficos e insights (dados já existentes)

## Contexto

O Streamlit app (`scripts/streamlit_app.py`) já calcula, mas não exibe, uma quantidade relevante de dados nos CSVs de saída:

- `outputs/<advertiser>/global_saturation_analysis/response_curve_data.csv`: curva de investimento x KPI para toda a faixa de orçamento, incluindo `iCPA`, `Projected_Revenue`, `Incremental_ROI` e o spend por canal em 3 mixes (`Historical`/`Optimized`/`Strategic`) — hoje só o ponto ótimo final é lido (para as métricas e os 2 donuts).
- `outputs/<advertiser>/global_saturation_analysis/individual_response_curves_data.csv`: curva de saturação isolada por canal — hoje só visualizada uma imagem estática (matplotlib) por vez, via um `selectbox`.
- `outputs/<advertiser>/detected_events.csv`: todos os eventos (picos/quedas de investimento) detectados pelo motor — hoje só navegável um de cada vez via `selectbox` na aba Impacto Causal, sem visão agregada.

Esta sessão foca exclusivamente em **extrair mais gráficos e insights desses dados já calculados**. Não inclui novos parâmetros opcionais de modelo nem novos filtros — ambos ficam para uma sessão futura.

## Escopo

### Fora de escopo (explícito)
- Novos parâmetros de configuração/modelo (`.agents/AGENTS.md` financial guardrails, adstock, etc.)
- Novos filtros de UI
- Qualquer mudança em `analysis.py`, `saturation_curve.py`, `elasticity_analysis.py`, `local_main.py` ou `local_main-without-gemini.py` — os dados usados já existem nos CSVs de saída atuais.

### Arquitetura

Novo módulo `scripts/dashboard_charts.py`, com uma função por gráfico. Cada função recebe um DataFrame já preparado (filtrado/derivado pelo código existente em `streamlit_app.py`) e devolve uma `plotly.graph_objects.Figure`. `streamlit_app.py` continua fazendo a leitura/filtragem de dados como já faz hoje (reaproveitando variáveis já em escopo nas abas 2 e 3) e só chama `st.plotly_chart(dashboard_charts.build_x(...))`.

Motivo: `streamlit_app.py` já tem 1605 linhas; isolar as 5 funções novas evita inflar ainda mais esse arquivo e mantém cada gráfico testável isoladamente. O código existente (curva principal, donuts) não é tocado.

Cada gráfico novo vem acompanhado de um pequeno texto em `st.markdown(...)` (título + 1-2 frases) explicando o que o gráfico mostra e para que serve — mesmo padrão já usado nas seções existentes (ex.: a introdução da "Curva de Saturação de Investimentos").

## Aba Elasticidade (Tab 3) — 4 novos gráficos

Todos lidos de `response_curve_data.csv` / `individual_response_curves_data.csv`, sem mudança de pipeline.

### 1. Curva de iCPA Marginal
- **Posição:** logo após a "Curva de Saturação de Investimentos" existente.
- **Dados:** `df["Monthly_Investment"]` (X) vs `df["iCPA"]` (Y) — coluna já calculada em `streamlit_app.py` (linha ~1111), nunca plotada.
- **Linhas de referência:** horizontais para `target_cpa`/`target_icpa` quando os checkboxes da sidebar ("Aplicar Limite de Target CPA" / "iCPA Marginal") estiverem ativos (reaproveita as variáveis já calculadas no tab3).
- **Marcadores:** `optimal_point` e `saturation_point` (os mesmos já destacados na curva principal), para consistência visual.
- **Texto de apoio:** explica que a curva mostra o custo marginal de cada KPI adicional a cada nível de investimento, e que o ponto onde ela cruza a linha de referência é onde o investimento deixa de compensar.

### 2. Evolução do mix de canais por orçamento
- **Posição:** nova seção após a curva de iCPA, antes dos donuts existentes.
- **Dados:** área 100% empilhada. X = `Monthly_Investment`; Y = `Spend_{channel}_Strategic` de cada canal como % do total daquela linha, para todas as linhas de `response_curve_data.csv` (hoje só o ponto ótimo é usado, nos 2 donuts).
- **Linhas verticais:** no investimento base (histórico) e no investimento ótimo, no mesmo estilo (cor/tracejado) da curva principal.
- **Texto de apoio:** explica que o gráfico mostra como a alocação recomendada entre canais muda à medida que o orçamento total escala — útil para saber quais canais "absorvem" mais verba incremental.

### 3. Comparativo interativo de saturação entre canais
- **Posição:** nova subseção *acima* do `selectbox` + imagem estática por canal já existentes (que permanecem exatamente como estão).
- **Dados:** uma linha por canal, a partir de `individual_response_curves_data.csv`. X = `Channel_Spend` (R$ bruto, por canal). Y = KPI do canal normalizado min-max para 0–100% da própria faixa (necessário porque os canais têm escalas absolutas muito diferentes — sem normalizar, canais pequenos ficam invisíveis ao lado dos grandes).
- **Legenda:** padrão Plotly (clique para isolar/ocultar canais).
- **Texto de apoio:** explica que a normalização serve para comparar a *velocidade* de saturação entre canais (não o volume absoluto), e que o eixo X continua em R$ reais por canal.

### 4. Curva de Receita/ROI
- **Posição:** logo após a curva de iCPA, antes da evolução do mix.
- **Condição de exibição:** somente quando `config["optimization_target"] == "REVENUE"`.
- **Dados:** eixo duplo. Y esquerdo = `Projected_Revenue` (linha); Y direito = `Incremental_ROI` (linha); X = `Monthly_Investment`. Ambas colunas já existem em `response_curve_data.csv`, não usadas na UI hoje.
- **Texto de apoio:** explica a leitura do eixo duplo e por que o gráfico só aparece quando o objetivo é Receita.

## Aba Impacto Causal (Tab 2) — Visão geral de eventos

- **Posição:** nova seção acima do `selectbox` "Selecione o Evento:" existente (que permanece como está).
- **Dados:** `detected_events.csv` (todos os candidatos detectados por `find_events()`), cruzado com a descoberta de `event_dirs` já existente no código atual do tab2, para marcar quais candidatos realmente têm um relatório navegável.
- **Gráfico:** um marcador/barra por evento detectado. X = data; Y = `percentage_change` (com sinal — quedas ficam negativas). Cor por direção (alta/queda). Opacidade cheia = evento validado (tem relatório); opacidade reduzida/cinza = descartado (não passou nos critérios estatísticos ou ficou fora do `max_events_to_analyze`). Hover mostra os canais envolvidos.
- **Sem clique-para-selecionar:** o gráfico é só uma visão geral (hover); não sincroniza com o `selectbox` abaixo (isso exigiria `st.plotly_chart(..., on_select=...)`, dependente de versão do Streamlit não confirmada no ambiente atual). Se depois for necessário, é uma extensão futura pontual.
- **Texto de apoio:** explica o que o gráfico mostra e o que significa a diferença de opacidade (validado vs. descartado).

## Fora de escopo / decisões explícitas
- Os 4 gráficos de PNG estático da aba Impacto Causal (accuracy, line chart, investment, sessions — gerados pelo pipeline via `presentation.py`) não são alterados.
- Não há mudança nos critérios de detecção/validação de eventos, nem novos dados persistidos pelo pipeline — tudo lido dos CSVs/JSON que já existem em `outputs/`.
