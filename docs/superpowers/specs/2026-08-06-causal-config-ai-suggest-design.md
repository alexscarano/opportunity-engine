# Sugestão automática dos limiares de Análise Causal

## Contexto

O Streamlit (`scripts/streamlit_app.py`) já tem um botão "Sugerir com IA" que usa o
Gemini para preencher, a partir de uma amostra do CSV de performance, três campos:
Coluna do KPI, se o KPI é monetário, e o Objetivo da Otimização. O expander
"Configurações da Análise Causal" (min dias pré-evento, R², p-value, limiares de
pico/queda de investimento, limite de simulação da curva de elasticidade) continua
100% manual — o texto de ajuda do botão inclusive documenta essa exclusão
explicitamente.

Um erro real (`ERRO CRÍTICO: ... Invalid value '[24617. ...]' for dtype 'int64'`)
levou a investigar o pipeline de dados (`data_preprocessor.py`) e, na sequência, a
essa conversa sobre estender a sugestão automática também para parte dos campos da
Análise Causal.

## Motivação

Ao ler o código de consumo desses campos (`analysis.py`, `elasticity_analysis.py`)
descobrimos que:

- `min_pre_period_days` já tem um **piso automático de 8 períodos** aplicado em
  `analysis.py:389` via `periods_to_days(...)`, independente do valor manual
  configurado. Para cadência semanal/mensal esse piso já ultrapassa o
  `max_value=90` do campo na UI — ou seja, hoje o campo não consegue nem
  *exibir* o comportamento real do motor para essas cadências.
- `increase_threshold_percent`/`decrease_threshold_percent` (40%/30% fixos hoje)
  são comparados contra uma distribuição de variação real calculada em
  `find_events` (`analysis.py:152-187`), que já loga um aviso quando o limiar
  sinaliza mais de 25% dos períodos como "evento" (ruído demais).
- `investment_limit_factor` escala o **pico histórico** de investimento
  (`max_hist_inv = daily_investment_df["investment"].max()`, `analysis.py:858`),
  não a média — o texto de ajuda atual do slider ("...vezes o gasto médio
  atual...") está desalinhado com o código, mas isso é um problema pré-existente
  fora de escopo aqui.

Esses três campos têm um valor "certo" derivável dos próprios dados. Os dois
campos de rigor estatístico (R², p-value) ficam de fora: são decisões de
tolerância a risco, não características objetivas dos dados, e calibrá-los mal
enfraquece silenciosamente uma salvaguarda.

## Escopo

Ganham sugestão automática:
1. `min_pre_period_days`
2. `increase_threshold_percent`
3. `decrease_threshold_percent`
4. `investment_limit_factor`

Ficam manuais (sem mudança): `r_squared_threshold`, `p_value_threshold`,
`require_statistical_significance`, `require_logical_direction`,
`require_model_fit`.

## Gatilho

Mesmo botão "Sugerir com IA" (`ai_suggest_btn`, `streamlit_app.py:1243`), mas o
gate atual (`elif not (gemini_key or ...) or not gemini_model: st.warning(...)`)
passa a valer **só** para a parte que chama o Gemini (Coluna do KPI / KPI
monetário / Objetivo). A parte determinística roda sempre que os dois arquivos
(Investimento e Performance) estiverem presentes, com ou sem chave configurada.

Hoje o handler só salva e lê `perf_file` (5 linhas, só pra achar nome de coluna).
Ele passa a também salvar `inv_file` em disco (mesmo padrão de
`dynamic_dir`/`perf_path` já existente) e ler o arquivo **completo**, porque as
fórmulas abaixo precisam da série real, não de uma amostra.

## Fórmulas

Todas calculadas em cima do arquivo de Investimento completo, sem precisar do de
Performance (as três leituras de `find_events`/`analysis.py` que embasam essas
fórmulas usam só `daily_investment_df`).

Pipeline de parsing reaproveitado tal qual existe hoje em `load_and_prepare_data`:
`guess_date_col`/`guess_channel_col`/`guess_investment_col` →
`read_csv_robust` → `drop_bi_export_footer_rows` → `robust_date_parsing` +
`robust_numeric_parsing` → dataframe longo `Date`/`Product Group`/`investment`.

Nova função pura `suggest_causal_config(daily_investment_df)` em
`data_preprocessor.py` (sem chamar IA, só pandas/numpy):

- **`min_pre_period_days`** = `8 * detect_cadence(daily_investment_df["Date"])`.
  Replica o piso que `analysis.py` já aplica de qualquer forma.
- **`increase_threshold_percent` / `decrease_threshold_percent`** = percentil 90
  da distribuição de `percentage_change` (mesmo cálculo de `find_events`: por
  `Product Group`, `investment` vs. média móvel de 12 períodos anteriores,
  `shift(1)`). O `percentage_change` de todos os canais é calculado separado
  (cada canal com sua própria média móvel) e depois **agrupado numa única
  distribuição pooled**; dessa distribuição pooled, separa-se valores positivos
  (percentil 90 → aumento) de negativos em módulo (percentil 90 → queda).
  Arredondado pra inteiro, clampado em `[1, 100]` (bounds atuais do
  `st.number_input`).
- **`investment_limit_factor`** = `clamp(daily_investment_df["investment"].max() /
  daily_investment_df["investment"].mean(), 1.5, 5.0)`, arredondado ao step de
  0.5 do slider. Mesma base (`.max()`/`.mean()` sobre a coluna `investment` em
  formato longo, por linha canal×período) que `analysis.py:858` já consome.

## Mudanças na UI (`streamlit_app.py`)

- `min_pre_period_days`: `max_value` do `st.number_input` sobe de `90` para
  `365` (senão o próprio valor sugerido para cadência mensal, 240, não caberia
  no campo).
- Os 4 campos passam a usar `value=st.session_state.get("ai_suggested_<campo>",
  <default atual>)`, mesmo padrão já usado para `ai_suggested_kpi_column` etc.
- Tooltip do botão "Sugerir com IA" (linhas 1247-1254) atualizado: passa a
  mencionar os 4 novos campos sugeridos e deixa explícito que só R²/p-value
  continuam manuais.
- Se `st.rerun()` acontecer sem chave do Gemini configurada (fluxo 100%
  determinístico), a Coluna do KPI/Objetivo mantêm o valor anterior — só os 4
  campos causais são atualizados.

## Erros e casos-limite

- Arquivo de investimento com menos de 2 datas únicas: `detect_cadence` já
  retorna `1` (dia) como default seguro — comportamento existente, sem mudança.
- Menos de 3 períodos por canal (equivalente ao guard de `find_events`,
  `len(weekly_investment_df) < 3: continue`): canal é ignorado no cálculo de
  percentual, mesma regra já usada em `find_events` pra evitar rolling window
  sem dado suficiente. Se **nenhum** canal tiver dado suficiente, os campos de
  aumento/queda mantêm o default atual (40/30) e um aviso é logado — não é
  fatal, é só "não há dado suficiente pra sugerir, mantendo manual".
- Qualquer exceção na leitura/parsing do arquivo de investimento (mesmo
  `try/except` já usado para a chamada do Gemini) cai no `st.warning(...)`
  existente e mantém os valores atuais, sem quebrar o formulário.

## Testes

Teste unitário para `suggest_causal_config` em
`tests/test_data_preprocessor.py`: dado um `daily_investment_df` sintético com
cadência e volatilidade conhecidas, valida que os 4 valores retornados batem com
a fórmula (dentro dos bounds, arredondamento correto). Sem framework adicional —
segue o padrão de teste já usado nesse arquivo.
