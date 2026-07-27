# Sugestão automática de campos do formulário via Gemini — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar um botão "Sugerir com IA" ao formulário de configuração que usa o Gemini para pré-preencher `Coluna do KPI`, `O KPI já está em R$` e `Objetivo da Otimização`, a partir dos CSVs já enviados.

**Architecture:** Nova função `suggest_form_fields(model, performance_df_sample)` em `scripts/google_api.py` monta o prompt, chama `model.generate_content`, valida e devolve um dict (ou levanta `ValueError`). `scripts/streamlit_app.py` ganha um terceiro `st.form_submit_button` dentro do form existente; seu branch salva o CSV de performance, lê uma amostra, chama a função nova, e grava o resultado em `st.session_state` antes de um `st.rerun()` — os 3 widgets afetados leem seus defaults desse `session_state`.

**Tech Stack:** Python 3.13, Streamlit, `google-generativeai`, pandas, pytest (`unittest.TestCase` + `unittest.mock.MagicMock`, seguindo o padrão de `tests/test_gemini_report.py`).

**Spec:** `docs/superpowers/specs/2026-07-27-sugestao-ia-preenchimento-form-design.md`

---

### Task 1: `suggest_form_fields` em `scripts/google_api.py`

**Files:**
- Modify: `scripts/google_api.py`
- Test: `tests/test_google_api.py` (novo)

- [ ] **Step 1: Escrever os testes falhando**

Criar `tests/test_google_api.py`:

```python
import sys
import os
import json
import unittest
from unittest.mock import MagicMock

import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from google_api import suggest_form_fields


def _sample_df():
    return pd.DataFrame(
        {
            "Data": ["2026-01-05", "2026-01-12", "2026-01-19"],
            "Canal": ["Pmax", "Pmax", "Pmax"],
            "Leads": [120, 135, 98],
        }
    )


def _mock_model(response_text):
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate_content.return_value = response
    return model


class TestSuggestFormFields(unittest.TestCase):
    def test_valid_response_returns_dict(self):
        model = _mock_model(
            json.dumps(
                {
                    "kpi_column": "Leads",
                    "kpi_is_monetary": False,
                    "optimization_target": "CONVERSIONS",
                }
            )
        )
        result = suggest_form_fields(model, _sample_df())
        self.assertEqual(
            result,
            {
                "kpi_column": "Leads",
                "kpi_is_monetary": False,
                "optimization_target": "CONVERSIONS",
            },
        )

    def test_response_with_json_fences_is_parsed(self):
        model = _mock_model(
            "```json\n"
            + json.dumps(
                {
                    "kpi_column": "Leads",
                    "kpi_is_monetary": False,
                    "optimization_target": "CONVERSIONS",
                }
            )
            + "\n```"
        )
        result = suggest_form_fields(model, _sample_df())
        self.assertEqual(result["kpi_column"], "Leads")

    def test_kpi_column_not_in_dataframe_raises(self):
        model = _mock_model(
            json.dumps(
                {
                    "kpi_column": "Coluna_Inexistente",
                    "kpi_is_monetary": False,
                    "optimization_target": "CONVERSIONS",
                }
            )
        )
        with self.assertRaises(ValueError):
            suggest_form_fields(model, _sample_df())

    def test_invalid_optimization_target_raises(self):
        model = _mock_model(
            json.dumps(
                {
                    "kpi_column": "Leads",
                    "kpi_is_monetary": False,
                    "optimization_target": "LUCRO",
                }
            )
        )
        with self.assertRaises(ValueError):
            suggest_form_fields(model, _sample_df())

    def test_malformed_json_raises_value_error(self):
        model = _mock_model("isto não é json")
        with self.assertRaises(ValueError):
            suggest_form_fields(model, _sample_df())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `uv run --with pytest pytest tests/test_google_api.py -v`
Expected: `ImportError: cannot import name 'suggest_form_fields' from 'google_api'` (ou `ModuleNotFoundError`, dependendo de como o import falha) em todos os testes.

- [ ] **Step 3: Implementar `suggest_form_fields`**

Adicionar ao final de `scripts/google_api.py` (após `authenticate_gemini`, mantendo `import json` no topo do arquivo junto dos imports existentes):

```python
import json
```

(adicionar essa linha junto de `import os` no topo do arquivo, linha 2)

E ao final do arquivo:

```python
def suggest_form_fields(model, performance_df_sample):
    """Pede ao Gemini para sugerir Coluna do KPI, se é monetário, e o
    Objetivo da Otimização a partir de uma amostra do CSV de performance.

    Levanta ValueError se a resposta vier ausente, malformada, ou falhar
    na validação contra as colunas reais da amostra.
    """
    columns = performance_df_sample.columns.tolist()
    sample_csv = performance_df_sample.head(5).to_csv(index=False)

    prompt = f"""Você está configurando uma análise de marketing. Aqui estão as
colunas disponíveis no arquivo de performance e uma amostra de linhas:

Colunas: {columns}

Amostra (CSV):
{sample_csv}

Responda APENAS com um JSON no formato exato abaixo, sem texto adicional e
sem markdown fences:

{{"kpi_column": "<nome exato de uma das colunas listadas acima>", "kpi_is_monetary": true ou false, "optimization_target": "CONVERSIONS" ou "REVENUE"}}

Regras:
- "kpi_column" deve ser o nome de uma métrica de resultado (ex: conversões,
  leads, vendas, receita), nunca uma coluna de data ou de texto categórico.
- "kpi_is_monetary" é true só se os valores da coluna parecerem moeda
  (decimais, valores altos e variáveis típicos de receita/faturamento).
- "optimization_target" é "REVENUE" só se "kpi_is_monetary" for true ou a
  coluna claramente representar receita/faturamento; caso contrário
  "CONVERSIONS".
"""

    response = model.generate_content(prompt)
    cleaned_response_text = (
        response.text.strip().replace("```json\n", "").replace("\n```", "").replace("```", "")
    )

    try:
        suggestion = json.loads(cleaned_response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"resposta da IA não é um JSON válido: {e}") from e

    kpi_column = suggestion.get("kpi_column")
    if not isinstance(kpi_column, str) or kpi_column not in columns:
        raise ValueError(
            f"coluna de KPI sugerida ('{kpi_column}') não existe nas colunas do arquivo"
        )

    kpi_is_monetary = suggestion.get("kpi_is_monetary")
    if isinstance(kpi_is_monetary, str):
        kpi_is_monetary = kpi_is_monetary.strip().lower() == "true"
    if not isinstance(kpi_is_monetary, bool):
        raise ValueError("'kpi_is_monetary' sugerido não é um booleano")

    optimization_target = suggestion.get("optimization_target")
    if optimization_target not in ("CONVERSIONS", "REVENUE"):
        raise ValueError(
            f"'optimization_target' sugerido ('{optimization_target}') não é "
            "'CONVERSIONS' nem 'REVENUE'"
        )

    return {
        "kpi_column": kpi_column,
        "kpi_is_monetary": kpi_is_monetary,
        "optimization_target": optimization_target,
    }
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `uv run --with pytest pytest tests/test_google_api.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/google_api.py tests/test_google_api.py
git commit -m "feat(google_api): add suggest_form_fields for AI-assisted form prefill"
```

---

### Task 2: Botão "Sugerir com IA" no formulário

**Files:**
- Modify: `scripts/streamlit_app.py:610-617` (imports)
- Modify: `scripts/streamlit_app.py:1212-1222` (layout de botões)

- [ ] **Step 1: Importar `read_csv_robust` e `suggest_form_fields`**

Em `scripts/streamlit_app.py`, o bloco de import de `data_preprocessor` (linhas 610-617) hoje é:

```python
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    guess_date_col,
    guess_channel_col,
    guess_investment_col,
    guess_trends_col,
    guess_kpi_col,
)
```

Adicionar `read_csv_robust` à lista, e logo abaixo um import de `google_api`:

```python
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    guess_date_col,
    guess_channel_col,
    guess_investment_col,
    guess_trends_col,
    guess_kpi_col,
    read_csv_robust,
)
from google_api import suggest_form_fields
```

- [ ] **Step 2: Trocar a linha de botões de 2 para 3 colunas**

Em `scripts/streamlit_app.py:1212-1222`, hoje:

```python
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_btn = st.form_submit_button(
                "Construir Motor de Oportunidades",
                type="primary",
                use_container_width=True,
            )
        with col_btn2:
            save_settings_btn = st.form_submit_button(
                "Salvar Configurações", type="secondary", use_container_width=True
            )
```

Trocar por:

```python
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            submit_btn = st.form_submit_button(
                "Construir Motor de Oportunidades",
                type="primary",
                use_container_width=True,
            )
        with col_btn2:
            save_settings_btn = st.form_submit_button(
                "Salvar Configurações", type="secondary", use_container_width=True
            )
        with col_btn3:
            ai_suggest_btn = st.form_submit_button(
                "Sugerir com IA",
                type="secondary",
                use_container_width=True,
                help=(
                    "Analisa os arquivos enviados e sugere: Coluna do KPI, se o "
                    "KPI já está em R$, e o Objetivo da Otimização. Não sugere "
                    "Ticket Médio, Taxa de Conversão nem os limites da Análise "
                    "Causal -- esses continuam manuais. Decide só pelos nomes de "
                    "coluna e uma amostra de linhas do CSV de Performance; requer "
                    "Chave de API do Gemini preenchida."
                ),
            )
```

- [ ] **Step 3: Verificar que a UI ainda sobe sem erro de sintaxe**

Run: `uv run python -c "import ast; ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read())"`
Expected: sem output (parse ok). Isso só confirma sintaxe válida; o branch de `ai_suggest_btn` ainda não existe (Task 3), então clicar no botão real na UI não faz nada até lá.

- [ ] **Step 4: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(streamlit): add AI-suggest button to setup form"
```

---

### Task 3: Branch `ai_suggest_btn` — chamar a IA e gravar sugestão

**Files:**
- Modify: `scripts/streamlit_app.py:1224-1226` (logo após o branch de `save_settings_btn`)

- [ ] **Step 1: Adicionar o branch**

Em `scripts/streamlit_app.py`, logo depois de:

```python
    if save_settings_btn:
        update_user_api_key(st.session_state["user_id"], gemini_key)
        st.success("Configurações salvas com sucesso!")
```

adicionar:

```python
    if ai_suggest_btn:
        if not inv_file or not perf_file:
            st.error(
                "Por favor, faça upload dos arquivos de Investimento e Performance para continuar."
            )
        elif not (gemini_key or os.environ.get("GEMINI_API_KEY")) or not gemini_model:
            st.warning(
                "Preencha a Chave de API do Gemini para usar a sugestão automática."
            )
        else:
            with st.spinner("Analisando arquivos com IA..."):
                try:
                    active_key = gemini_key or os.environ.get("GEMINI_API_KEY")

                    safe_adv_name = (
                        advertiser_name.replace(" ", "_").replace("/", "").replace("\\", "")
                    )
                    dynamic_dir = os.path.join(
                        "inputs",
                        f"user_{st.session_state['user_id']}",
                        f"{safe_adv_name}_dynamic",
                    )
                    os.makedirs(dynamic_dir, exist_ok=True)
                    perf_path = os.path.join(dynamic_dir, "performance.csv")
                    with open(perf_path, "wb") as f:
                        f.write(perf_file.getbuffer())

                    performance_sample = read_csv_robust(perf_path, nrows=5)

                    import google.generativeai as genai

                    genai.configure(api_key=active_key)
                    model = genai.GenerativeModel(gemini_model)
                    suggestion = suggest_form_fields(model, performance_sample)

                    st.session_state["ai_suggested_kpi_column"] = suggestion["kpi_column"]
                    st.session_state["ai_suggested_kpi_is_monetary"] = suggestion["kpi_is_monetary"]
                    st.session_state["ai_suggested_optimization_target"] = suggestion["optimization_target"]
                    st.rerun()
                except Exception as e:
                    st.warning(
                        f"Não foi possível gerar sugestão automática: {e}. "
                        "Mantendo os valores atuais."
                    )
```

- [ ] **Step 2: Verificar sintaxe**

Run: `uv run python -c "import ast; ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read())"`
Expected: sem output.

- [ ] **Step 3: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(streamlit): call Gemini suggestion on AI-suggest button click"
```

---

### Task 4: Widgets lendo o valor sugerido do `session_state`

**Files:**
- Modify: `scripts/streamlit_app.py:1025-1052`

- [ ] **Step 1: Atualizar os 3 widgets**

Hoje (`scripts/streamlit_app.py:1025-1052`):

```python
            kpi_column = st.text_input(
                "Coluna do KPI no CSV de Performance",
                value="Sessions",
                help="Nome exato da coluna (ex: Sessions, Conversions, Leads).",
            )
            kpi_is_monetary = st.checkbox(
                "O KPI já está em R$ (ex: Faturamento, Receita)",
                value=False,
                help="Marque se a coluna de KPI já é o valor monetário final (não uma contagem "
                "de conversões/leads). Isso ignora Taxa de Conversão e Ticket Médio abaixo, e "
                "troca CPA/iCPA por ROAS/iROAS no dashboard.",
            )
            optimization_target_label = st.selectbox(
                "Objetivo da Otimização",
                options=[
                    "Maximizar Volume de Conversões (Leads, Vendas, etc.)",
                    "Maximizar Receita / Faturamento (Revenue)",
                ],
                help="Define o que a aba Elasticidade tenta maximizar ao recomendar como dividir "
                "a verba entre canais. 'Volume' busca o maior número de conversões; 'Receita' usa "
                "o Ticket Médio para buscar o maior faturamento. Também decide se o dashboard "
                "mostra CPA/iCPA (custo por conversão) ou ROAS/iROAS (retorno por real investido).",
            )
            optimization_target = (
                "CONVERSIONS"
                if "Conversões" in optimization_target_label
                else "REVENUE"
            )
```

Trocar por:

```python
            kpi_column = st.text_input(
                "Coluna do KPI no CSV de Performance",
                value=st.session_state.get("ai_suggested_kpi_column", "Sessions"),
                help="Nome exato da coluna (ex: Sessions, Conversions, Leads).",
            )
            kpi_is_monetary = st.checkbox(
                "O KPI já está em R$ (ex: Faturamento, Receita)",
                value=st.session_state.get("ai_suggested_kpi_is_monetary", False),
                help="Marque se a coluna de KPI já é o valor monetário final (não uma contagem "
                "de conversões/leads). Isso ignora Taxa de Conversão e Ticket Médio abaixo, e "
                "troca CPA/iCPA por ROAS/iROAS no dashboard.",
            )
            _opt_labels = [
                "Maximizar Volume de Conversões (Leads, Vendas, etc.)",
                "Maximizar Receita / Faturamento (Revenue)",
            ]
            _suggested_target = st.session_state.get("ai_suggested_optimization_target")
            optimization_target_label = st.selectbox(
                "Objetivo da Otimização",
                options=_opt_labels,
                index=1 if _suggested_target == "REVENUE" else 0,
                help="Define o que a aba Elasticidade tenta maximizar ao recomendar como dividir "
                "a verba entre canais. 'Volume' busca o maior número de conversões; 'Receita' usa "
                "o Ticket Médio para buscar o maior faturamento. Também decide se o dashboard "
                "mostra CPA/iCPA (custo por conversão) ou ROAS/iROAS (retorno por real investido).",
            )
            optimization_target = (
                "CONVERSIONS"
                if "Conversões" in optimization_target_label
                else "REVENUE"
            )
```

- [ ] **Step 2: Verificar sintaxe**

Run: `uv run python -c "import ast; ast.parse(open('scripts/streamlit_app.py', encoding='utf-8').read())"`
Expected: sem output.

- [ ] **Step 3: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(streamlit): prefill KPI/monetary/objective fields from AI suggestion"
```

---

### Task 5: Verificação manual e suíte completa

**Files:** nenhum (só verificação)

- [ ] **Step 1: Rodar a suíte inteira**

Run: `uv run --with pytest pytest`
Expected: todos os testes passam, incluindo os 5 novos de `tests/test_google_api.py` e nenhuma regressão nos existentes.

- [ ] **Step 2: Rodar a UI manualmente**

Run: `mise run dev` (ou `streamlit run scripts/streamlit_app.py`)

1. Preencher a Chave de API do Gemini.
2. Subir `exemplo_csv/investimento_pmax_semanal.csv` e `exemplo_csv/performance_pmax_semanal.csv`.
3. Clicar em "Sugerir com IA".
4. Confirmar que `Coluna do KPI` vira `Leads`, `O KPI já está em R$` fica desmarcado, e `Objetivo da Otimização` fica em "Maximizar Volume de Conversões".
5. Confirmar que Ticket Médio, Taxa de Conversão e os campos dentro de "Configurações da Análise Causal" **não mudam**.
6. Editar manualmente um dos campos sugeridos e confirmar que o valor editado é o que vai pro `config_dynamic.json` ao clicar em "Construir Motor de Oportunidades" (a sugestão só define o default inicial, não trava o campo).

- [ ] **Step 3: Testar o caminho de erro**

Repetir o passo acima sem preencher a Chave de API do Gemini, clicar em "Sugerir com IA", confirmar que aparece o aviso "Preencha a Chave de API do Gemini..." e nenhum campo muda.

Nenhum commit neste task — é só verificação do que já foi commitado nos Tasks 1-4.
