# Spec — Sugestão automática de campos do formulário via Gemini

**Data:** 2026-07-27
**Branch:** `feat/robust-csv-ingestion`

## 1. Objetivo

No formulário de configuração (`scripts/streamlit_app.py`, `st.form("setup_form")`),
adicionar um botão que usa o Gemini para sugerir o preenchimento de 3 campos,
a partir dos CSVs já enviados: `Coluna do KPI`, `O KPI já está em R$`,
`Objetivo da Otimização`. O usuário revisa/edita os valores sugeridos antes de
enviar o formulário normalmente.

## 2. Não-objetivo

- **Ticket Médio** e **Taxa de Conversão** não são sugeridos — são fatos do
  negócio que não estão nos CSVs de investimento/performance. Continuam
  manuais.
- Thresholds de detecção de evento/estatísticos (`min_pre_period_days`,
  `r_squared_threshold`, `p_value_threshold`, `increase_threshold_percent`,
  `decrease_threshold_percent`) não são sugeridos — são conhecimento de
  produto/rigor metodológico, não do dataset específico.
- Não usa o CSV de investimento nem o de tendências no prompt — só o de
  performance.
- Não pede descrição textual do negócio ao usuário. A sugestão se baseia
  só em nomes de coluna + amostra de linhas do CSV de performance.
- Não substitui a heurística de fallback já existente (`guess_kpi_col` em
  `scripts/data_preprocessor.py:549`), que continua rodando no backend
  como rede de segurança independente do que estiver no campo de texto.

## 3. UI

Dentro do `st.form("setup_form")` já existente (`scripts/streamlit_app.py:977`),
a linha de botões usa hoje `col_btn1, col_btn2 = st.columns(2)`
(`scripts/streamlit_app.py:1212`). Trocar para
`col_btn1, col_btn2, col_btn3 = st.columns(3)` e adicionar o terceiro
botão na nova coluna, junto dos dois `st.form_submit_button` atuais
(`Construir Motor de Oportunidades`, `Salvar Configurações`):

```python
ai_suggest_btn = st.form_submit_button(
    "Sugerir com IA",
    type="secondary",
    use_container_width=True,
    help=(
        "Analisa os arquivos enviados e sugere: Coluna do KPI, se o KPI já "
        "está em R$, e o Objetivo da Otimização. Não sugere Ticket Médio, "
        "Taxa de Conversão nem os limites da Análise Causal -- esses "
        "continuam manuais. Decide só pelos nomes de coluna e uma amostra "
        "de linhas do CSV de Performance; requer Chave de API do Gemini "
        "preenchida."
    ),
)
```

Os 3 widgets afetados passam a usar `st.session_state` como default,
mantendo o valor atual como fallback:

```python
kpi_column = st.text_input(
    "Coluna do KPI no CSV de Performance",
    value=st.session_state.get("ai_suggested_kpi_column", "Sessions"),
    ...
)
kpi_is_monetary = st.checkbox(
    "O KPI já está em R$ (ex: Faturamento, Receita)",
    value=st.session_state.get("ai_suggested_kpi_is_monetary", False),
    ...
)
```

Para `optimization_target_label` (um `st.selectbox`), calcular o `index=`
a partir da sugestão:

```python
_OPT_LABELS = [
    "Maximizar Volume de Conversões (Leads, Vendas, etc.)",
    "Maximizar Receita / Faturamento (Revenue)",
]
_suggested_target = st.session_state.get("ai_suggested_optimization_target")
_default_index = 1 if _suggested_target == "REVENUE" else 0
optimization_target_label = st.selectbox(
    "Objetivo da Otimização",
    options=_OPT_LABELS,
    index=_default_index,
    ...
)
```

## 4. Fluxo (branch `if ai_suggest_btn:`)

1. Validar `inv_file` e `perf_file` presentes — reaproveitar o mesmo
   `st.error` já usado no branch `submit_btn` (`scripts/streamlit_app.py:1232-1235`).
2. Validar que há uma chave Gemini ativa (`gemini_key` do campo ou
   `GEMINI_API_KEY` do ambiente) e um `gemini_model` selecionado. Se não
   houver, `st.warning("Preencha a Chave de API do Gemini para usar a "
   "sugestão automática.")` e parar (sem chamar a API, sem tocar
   `session_state`).
3. Salvar `perf_file` em disco reaproveitando o mesmo padrão de path já
   usado no branch de submit (`dynamic_dir`/`performance.csv`) — só o
   arquivo de performance precisa ser persistido para este fluxo.
4. Ler amostra com `read_csv_robust(perf_path, nrows=5)`
   (`scripts/data_preprocessor.py:416`).
5. Montar prompt (ver seção 5) e chamar:
   ```python
   genai.configure(api_key=active_key)
   model = genai.GenerativeModel(gemini_model)
   response = model.generate_content(prompt)
   ```
   (mesmo padrão de `scripts/gemini_report.py:616-621`.)
6. Limpar a resposta (`strip()`, remover fences ```` ```json ```` /
   ```` ``` ````) e `json.loads`.
7. Validar o dict resultante (ver seção 6). Se válido:
   `st.session_state["ai_suggested_kpi_column"] = ...`,
   `["ai_suggested_kpi_is_monetary"] = ...`,
   `["ai_suggested_optimization_target"] = ...`, depois `st.rerun()`.
8. Se inválido em qualquer etapa (exceção na chamada, JSON malformado,
   `kpi_column` sugerida fora de `df.columns`, `optimization_target` fora
   de `{"CONVERSIONS", "REVENUE"}`): `st.warning(f"Não foi possível gerar "
   f"sugestão automática: {motivo}. Mantendo os valores atuais.")`. Rejeita
   a sugestão inteira (não faz aceite parcial campo a campo) e não mexe em
   `session_state`.

## 5. Prompt e schema

Nova função `suggest_form_fields(model, performance_df_sample) -> dict`,
em `scripts/google_api.py` (mesmo módulo de `authenticate_gemini`).

Prompt (texto simples, sem few-shot): lista as colunas de
`performance_df_sample.columns.tolist()` e as primeiras 5 linhas
(`performance_df_sample.head(5).to_csv(index=False)`), pede para retornar
**apenas** um JSON no formato:

```json
{
  "kpi_column": "<nome exato de uma das colunas listadas>",
  "kpi_is_monetary": true/false,
  "optimization_target": "CONVERSIONS" ou "REVENUE"
}
```

Instrução explícita no prompt: `kpi_is_monetary` é `true` só se os valores
da coluna parecerem moeda (decimais, valores altos e variáveis típicos de
receita); `optimization_target` é `"REVENUE"` só se `kpi_is_monetary` for
`true` ou a coluna claramente representar receita/faturamento — caso
contrário `"CONVERSIONS"`.

## 6. Validação da resposta

Em `scripts/google_api.py`, após o `json.loads`:

- `kpi_column` deve ser `str` e estar em `performance_df_sample.columns`.
- `kpi_is_monetary` deve ser `bool` (ou `"true"`/`"false"` string — normalizar).
- `optimization_target` deve ser exatamente `"CONVERSIONS"` ou `"REVENUE"`.

Qualquer desvio → levanta `ValueError` com mensagem descritiva, que o
branch em `streamlit_app.py` captura e converte no `st.warning` da seção 4.

## 7. Testes

`tests/test_google_api.py` (novo, ou seção nova se já existir arquivo
correlato):

- Resposta JSON válida → retorna dict com os 3 campos.
- Resposta com fences ```` ```json ```` → limpa e faz parse corretamente.
- `kpi_column` sugerida fora das colunas reais → `ValueError`.
- `optimization_target` fora do enum → `ValueError`.
- JSON malformado → `ValueError` (não deixa `json.JSONDecodeError` vazar
  sem contexto).

Verificação manual: rodar a UI (`mise run dev`), subir os CSVs de
`exemplo_csv/`, clicar em "Sugerir com IA", confirmar que os 3 campos são
pré-preenchidos e que o restante do formulário (Ticket Médio, thresholds)
não muda.

## 8. Arquivos tocados

| Arquivo | Mudança |
|---|---|
| `scripts/google_api.py` | nova função `suggest_form_fields` |
| `scripts/streamlit_app.py` | novo botão + branch `ai_suggest_btn` + `value=`/`index=` dos 3 widgets lendo `session_state` |
| `tests/test_google_api.py` | testes da seção 7 |
