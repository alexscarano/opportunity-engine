# Sugestão automática dos limiares de Análise Causal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing "Sugerir com IA" button in the Streamlit app so it also prefills `min_pre_period_days`, `increase_threshold_percent`, `decrease_threshold_percent`, and `investment_limit_factor` from the uploaded investment CSV, using deterministic formulas (no LLM call needed for this part).

**Architecture:** A new pure function `suggest_causal_config(daily_investment_df)` in `scripts/data_preprocessor.py` computes the four values from a parsed investment dataframe, reusing `detect_cadence` and mirroring the rolling-12-period `percentage_change` math `analysis.py`'s `find_events` already applies at runtime. `scripts/streamlit_app.py`'s `ai_suggest_btn` handler is restructured into two independent blocks: a deterministic one (parses the investment file, calls `suggest_causal_config`, always runs when both files are uploaded) and the existing Gemini-based one (unchanged logic, still gated on having an API key). The four form widgets read their defaults from `st.session_state["ai_suggested_*"]`, same pattern already used for `kpi_column`/`kpi_is_monetary`/`optimization_target`.

**Tech Stack:** Python, pandas, numpy, pytest, Streamlit (no new dependencies).

**Spec:** `docs/superpowers/specs/2026-08-06-causal-config-ai-suggest-design.md`

---

### Task 1: `suggest_causal_config` in `data_preprocessor.py`

**Files:**
- Modify: `scripts/data_preprocessor.py` (insert new function after line 233, before `drop_partial_periods`)
- Test: `tests/test_data_preprocessor.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data_preprocessor.py`. First, add `suggest_causal_config` to the existing import block at the top of the file (currently lines 12-27):

```python
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    resolve_column,
    robust_date_parsing,
    robust_numeric_parsing,
    read_csv_robust,
    load_and_prepare_data,
    guess_date_col,
    guess_channel_col,
    guess_investment_col,
    guess_trends_col,
    guess_kpi_col,
    detect_cadence,
    drop_partial_periods,
    drop_bi_export_footer_rows,
    suggest_causal_config,
)
```

Then append these tests at the end of the file:

```python
# --- suggest_causal_config ---


def _flat_then_spike_channel(product_group, spike_value, n_baseline=5):
    """Builds one channel's rows: `n_baseline` weekly periods flat at 100,
    then one final period at `spike_value`. With a perfectly flat baseline,
    every baseline period's percentage_change vs. its own rolling history is
    exactly 0%, so only the final spike shows up as a non-zero pct change --
    that isolates a single, hand-computable data point per channel."""
    dates = pd.date_range("2025-01-01", periods=n_baseline + 1, freq="7D")
    investment = [100.0] * n_baseline + [float(spike_value)]
    return pd.DataFrame(
        {"Date": dates, "Product Group": product_group, "investment": investment}
    )


def test_suggest_causal_config_min_pre_period_days_matches_8x_cadence():
    df = _flat_then_spike_channel("A", 110)
    result = suggest_causal_config(df)
    assert result["min_pre_period_days"] == 56  # 8 * 7-day (weekly) cadence


def test_suggest_causal_config_min_pre_period_days_capped_at_365():
    # 6 dates 60 days apart -> median 60 -> non-canonical cadence, raw 60.
    dates = pd.date_range("2025-01-01", periods=6, freq="60D")
    df = pd.DataFrame(
        {"Date": dates, "Product Group": "A", "investment": [100.0] * 6}
    )
    result = suggest_causal_config(df)
    assert result["min_pre_period_days"] == 365  # 8 * 60 = 480, capped


def test_suggest_causal_config_percentile_90_increase_and_decrease():
    # Four isolated channels, each contributing exactly one non-zero
    # percentage_change: +10%, +50%, -5%, -25%. Pooled increases = [10, 50],
    # pooled decreases (abs) = [5, 25]. numpy's default linear percentile:
    # percentile(90, [10, 50]) = 10 + 0.9*(50-10) = 46
    # percentile(90, [5, 25])  = 5 + 0.9*(25-5)   = 23
    df = pd.concat(
        [
            _flat_then_spike_channel("A", 110),  # +10%
            _flat_then_spike_channel("B", 150),  # +50%
            _flat_then_spike_channel("C", 95),  # -5%
            _flat_then_spike_channel("D", 75),  # -25%
        ],
        ignore_index=True,
    )
    result = suggest_causal_config(df)
    assert result["increase_threshold_percent"] == 46
    assert result["decrease_threshold_percent"] == 23


def test_suggest_causal_config_thresholds_clamped_to_100():
    # A single +900% spike -> percentile of a 1-element array is that
    # element itself (900), which must be clamped to the field's max (100).
    df = _flat_then_spike_channel("A", 1000)
    result = suggest_causal_config(df)
    assert result["increase_threshold_percent"] == 100


def test_suggest_causal_config_falls_back_to_defaults_when_insufficient_data():
    # Each channel has only 2 periods -- below find_events' own "< 3 periods"
    # guard -- so neither channel contributes any percentage_change and the
    # thresholds fall back to the pre-existing manual defaults (40/30).
    dates = pd.date_range("2025-01-01", periods=2, freq="7D")
    df = pd.concat(
        [
            pd.DataFrame(
                {"Date": dates, "Product Group": "A", "investment": [100.0, 150.0]}
            ),
            pd.DataFrame(
                {"Date": dates, "Product Group": "B", "investment": [100.0, 60.0]}
            ),
        ],
        ignore_index=True,
    )
    result = suggest_causal_config(df)
    assert result["increase_threshold_percent"] == 40
    assert result["decrease_threshold_percent"] == 30


def test_suggest_causal_config_investment_limit_factor_matches_max_over_mean():
    # 9 periods at 100 + 1 at 400 -> mean=130, max=400, ratio=400/130=3.0769,
    # rounded to the slider's 0.5 step -> 3.0.
    dates = pd.date_range("2025-01-01", periods=10, freq="1D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Product Group": "A",
            "investment": [100.0] * 9 + [400.0],
        }
    )
    result = suggest_causal_config(df)
    assert result["investment_limit_factor"] == 3.0


def test_suggest_causal_config_investment_limit_factor_clamped_to_bounds():
    flat_dates = pd.date_range("2025-01-01", periods=5, freq="1D")
    flat_df = pd.DataFrame(
        {"Date": flat_dates, "Product Group": "A", "investment": [100.0] * 5}
    )
    assert suggest_causal_config(flat_df)["investment_limit_factor"] == 1.5

    # max/mean is mathematically bounded above by n (the point count) as the
    # spike grows -- with only 5 points the ratio can never actually reach 5,
    # so this uses 10 points (9 flat + 1 huge spike) to genuinely exceed the
    # 5.0 ceiling and exercise the clamp, not just favorable rounding.
    extreme_dates = pd.date_range("2025-01-01", periods=10, freq="1D")
    extreme_df = pd.DataFrame(
        {
            "Date": extreme_dates,
            "Product Group": "A",
            "investment": [10.0] * 9 + [100000.0],
        }
    )
    assert suggest_causal_config(extreme_df)["investment_limit_factor"] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine" && uv run pytest tests/test_data_preprocessor.py -k suggest_causal_config -v`
Expected: FAIL (or ERROR/ImportError, since `suggest_causal_config` doesn't exist yet in `data_preprocessor.py`)

- [ ] **Step 3: Implement `suggest_causal_config`**

In `scripts/data_preprocessor.py`, insert this new function right after `detect_cadence` (after line 233, before the blank lines preceding `def drop_partial_periods` on line 236):

```python
def suggest_causal_config(daily_investment_df):
    """
    Deriva sugestões para min_pre_period_days, increase/decrease_threshold_percent
    e investment_limit_factor a partir do histórico real de investimento --
    sem chamar IA, só estatística sobre os dados. Espelha exatamente a
    matemática que o motor já aplica em tempo de execução, pra a sugestão
    bater com o comportamento real:

    - min_pre_period_days: analysis.py já aplica um piso de 8 períodos via
      periods_to_days(), independente do valor manual configurado. Sugerimos
      esse piso direto (8 * cadência), capado em 365 dias.
    - increase/decrease_threshold_percent: find_events() (analysis.py) calcula
      percentage_change de cada período vs. média móvel dos 12 períodos
      anteriores, por canal, e já loga aviso quando >25% dos períodos viram
      "evento" (limiar capturando ruído). Replicamos esse cálculo aqui e
      usamos o percentil 90 da distribuição real (positivos = aumento,
      negativos em módulo = queda) -- deixa uma folga clara abaixo dos 25%.
    - investment_limit_factor: run_opportunity_projection (analysis.py) usa
      max(investment) * investment_limit_factor pra definir até onde simular
      a curva de resposta. Sugerimos max/mean do histórico como um proxy de
      "quanto essa conta já provou suportar de variação", clampado nos
      mesmos limites do slider (1.5-5.0) e arredondado ao seu step (0.5).
    """
    cadence = detect_cadence(daily_investment_df["Date"])
    min_pre_period_days = int(min(365, 8 * cadence))

    all_pct_changes = []
    for _, product_df in daily_investment_df.groupby("Product Group"):
        product_df = product_df.sort_values("Date")
        if cadence >= 7:
            bucket_key = product_df["Date"]
        else:
            bucket_key = product_df["Date"].dt.to_period("W-MON").dt.start_time
        period_investment = (
            product_df.assign(_bucket=bucket_key)
            .groupby("_bucket")["investment"]
            .sum()
            .reset_index()
        )
        if len(period_investment) < 3:
            continue

        period_investment["historical_avg"] = (
            period_investment["investment"]
            .rolling(window=12, min_periods=1)
            .mean()
            .shift(1)
        )
        period_investment = period_investment.dropna(subset=["historical_avg"])
        period_investment = period_investment[period_investment["historical_avg"] > 0]
        if period_investment.empty:
            continue

        pct_change = (
            period_investment["investment"] / period_investment["historical_avg"] - 1
        ) * 100
        all_pct_changes.extend(pct_change.tolist())

    if all_pct_changes:
        pct_series = pd.Series(all_pct_changes)
        increases = pct_series[pct_series > 0]
        decreases = pct_series[pct_series < 0].abs()
        increase_threshold_percent = (
            int(round(np.percentile(increases, 90))) if len(increases) else 40
        )
        decrease_threshold_percent = (
            int(round(np.percentile(decreases, 90))) if len(decreases) else 30
        )
        increase_threshold_percent = max(1, min(100, increase_threshold_percent))
        decrease_threshold_percent = max(1, min(100, decrease_threshold_percent))
    else:
        increase_threshold_percent = 40
        decrease_threshold_percent = 30

    investment = daily_investment_df["investment"]
    mean_investment = investment.mean()
    raw_factor = investment.max() / mean_investment if mean_investment > 0 else 1.5
    investment_limit_factor = round(max(1.5, min(5.0, raw_factor)) * 2) / 2

    return {
        "min_pre_period_days": min_pre_period_days,
        "increase_threshold_percent": increase_threshold_percent,
        "decrease_threshold_percent": decrease_threshold_percent,
        "investment_limit_factor": investment_limit_factor,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine" && uv run pytest tests/test_data_preprocessor.py -k suggest_causal_config -v`
Expected: 7 passed

- [ ] **Step 5: Run the full data_preprocessor suite to check for regressions**

Run: `cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine" && uv run pytest tests/test_data_preprocessor.py -q`
Expected: all tests pass (77 passed, up from 76)

- [ ] **Step 6: Commit**

```bash
cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine"
git add scripts/data_preprocessor.py tests/test_data_preprocessor.py
git commit -m "feat(data_preprocessor): add suggest_causal_config to derive causal-analysis thresholds from data"
```

---

### Task 2: Wire the suggestion into the Streamlit "Sugerir com IA" button

**Files:**
- Modify: `scripts/streamlit_app.py`

- [ ] **Step 1: Import the new pieces**

In `scripts/streamlit_app.py`, the `from data_preprocessor import (...)` block currently reads (lines 620-628):

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
```

Replace it with:

```python
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    guess_date_col,
    guess_channel_col,
    guess_investment_col,
    guess_trends_col,
    guess_kpi_col,
    read_csv_robust,
    robust_date_parsing,
    robust_numeric_parsing,
    drop_bi_export_footer_rows,
    suggest_causal_config,
)
```

- [ ] **Step 2: Widen `min_pre_period_days` bounds and read session_state defaults on the 4 widgets**

The `Configurações da Análise Causal` expander currently reads (lines 1096-1187):

```python
            with st.expander("Configurações da Análise Causal"):
                min_pre_period_days = st.number_input(
                    "Dias Mínimos Pré-Evento",
                    min_value=7,
                    max_value=90,
                    value=14,
                    help="Dias de histórico exigidos antes do evento pra treinar o modelo causal "
                    "(padrão: 14, reduzido de 30). Menos dias deixa mais eventos serem analisados, "
                    "mas com um modelo mais instável e um R² menos confiável; mais dias analisa "
                    "menos eventos, só os que têm histórico suficiente, mas com estimativas mais "
                    "robustas.",
                )
```

Replace the `min_pre_period_days` widget with:

```python
            with st.expander("Configurações da Análise Causal"):
                min_pre_period_days = st.number_input(
                    "Dias Mínimos Pré-Evento",
                    min_value=7,
                    max_value=365,
                    value=st.session_state.get("ai_suggested_min_pre_period_days", 14),
                    help="Dias de histórico exigidos antes do evento pra treinar o modelo causal "
                    "(padrão: 14, reduzido de 30). Menos dias deixa mais eventos serem analisados, "
                    "mas com um modelo mais instável e um R² menos confiável; mais dias analisa "
                    "menos eventos, só os que têm histórico suficiente, mas com estimativas mais "
                    "robustas.",
                )
```

Then find the `increase_threshold_percent` widget (lines 1132-1141):

```python
                increase_threshold_percent = st.number_input(
                    "Var. Mínima de Aumento de Investimento (%)",
                    min_value=1,
                    max_value=100,
                    value=40,
                    help="Quanto o investimento semanal precisa subir (vs. média das 12 semanas "
                    "anteriores) pra ser detectado como um 'pico' a analisar. Valor baixo encontra "
                    "mais eventos candidatos, inclusive picos pequenos/ruído; valor alto só "
                    "detecta aumentos bem grandes de verba.",
                )
```

Replace its `value=40` with `value=st.session_state.get("ai_suggested_increase_threshold_percent", 40)`.

Then find the `decrease_threshold_percent` widget (lines 1142-1150):

```python
                decrease_threshold_percent = st.number_input(
                    "Var. Mínima de Queda de Investimento (%)",
                    min_value=1,
                    max_value=100,
                    value=30,
                    help="Mesma lógica do campo acima, mas pra quedas de investimento: quanto a "
                    "verba semanal precisa cair (vs. média das 12 semanas anteriores) pra virar um "
                    "evento de 'corte de verba' a analisar.",
                )
```

Replace its `value=30` with `value=st.session_state.get("ai_suggested_decrease_threshold_percent", 30)`.

Then find the `investment_limit_factor` widget (lines 1178-1187):

```python
                investment_limit_factor = st.slider(
                    "Limite de Investimento Simulado (Elasticidade)",
                    min_value=1.5,
                    max_value=5.0,
                    value=1.5,
                    step=0.5,
                    help="Até quantas vezes o gasto médio atual a curva de resposta (aba Elasticidade) "
                    "é simulada. Se o 'Cenário de Saturação' sempre empatar com o 'Ponto Recomendado', "
                    "aumente este limite -- o teto real pode estar além do que 1.5x consegue mostrar.",
                )
```

Replace its `value=1.5` with `value=st.session_state.get("ai_suggested_investment_limit_factor", 1.5)`.

- [ ] **Step 3: Update the "Sugerir com IA" button's tooltip**

Find (lines 1243-1255):

```python
        with col_btn3:
            ai_suggest_btn = st.form_submit_button(
                "Sugerir com IA",
                type="secondary",
                width="stretch",
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

Replace with:

```python
        with col_btn3:
            ai_suggest_btn = st.form_submit_button(
                "Sugerir com IA",
                type="secondary",
                width="stretch",
                help=(
                    "Analisa os arquivos enviados e sugere: Coluna do KPI, se o KPI já "
                    "está em R$, e o Objetivo da Otimização (via IA -- requer Chave de "
                    "API do Gemini preenchida). Também sugere, a partir do histórico do "
                    "arquivo de Investimento (cálculo estatístico, não usa IA nem "
                    "precisa de chave): Dias Mínimos Pré-Evento, Var. Mínima de "
                    "Aumento/Queda de Investimento e Limite de Investimento Simulado. "
                    "Não sugere Ticket Médio, Taxa de Conversão, Ajuste Mínimo do "
                    "Modelo (R²) nem Significância Máxima (p-value) -- esses continuam "
                    "manuais."
                ),
            )
```

- [ ] **Step 4: Restructure the `ai_suggest_btn` handler**

Find the current handler (lines 1261-1304):

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

Replace it with:

```python
    if ai_suggest_btn:
        if not inv_file or not perf_file:
            st.error(
                "Por favor, faça upload dos arquivos de Investimento e Performance para continuar."
            )
        else:
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
            inv_path = os.path.join(dynamic_dir, "investment.csv")
            with open(inv_path, "wb") as f:
                f.write(inv_file.getbuffer())

            suggested_anything = False

            # Deterministic part: no IA, funciona mesmo sem chave do Gemini --
            # só precisa dos dois arquivos, já garantidos pelo check acima.
            with st.spinner("Analisando histórico de investimento..."):
                try:
                    inv_date_col = guess_date_col(inv_path)
                    inv_channel_col = guess_channel_col(inv_path)
                    inv_investment_col = guess_investment_col(inv_path)

                    inv_df = read_csv_robust(inv_path)
                    inv_df = inv_df.rename(
                        columns={
                            inv_date_col: "Date",
                            inv_channel_col: "Product Group",
                            inv_investment_col: "investment",
                        }
                    )
                    inv_df = drop_bi_export_footer_rows(inv_df, "Date")
                    inv_df["Product Group"] = (
                        inv_df["Product Group"].str.strip().str.upper()
                    )
                    inv_df["investment"] = robust_numeric_parsing(
                        inv_df["investment"], column_name="investment"
                    )
                    inv_df["Date"] = robust_date_parsing(inv_df["Date"])
                    inv_df = inv_df.dropna(
                        subset=["Date", "investment", "Product Group"]
                    )

                    causal_suggestion = suggest_causal_config(inv_df)
                    st.session_state["ai_suggested_min_pre_period_days"] = (
                        causal_suggestion["min_pre_period_days"]
                    )
                    st.session_state["ai_suggested_increase_threshold_percent"] = (
                        causal_suggestion["increase_threshold_percent"]
                    )
                    st.session_state["ai_suggested_decrease_threshold_percent"] = (
                        causal_suggestion["decrease_threshold_percent"]
                    )
                    st.session_state["ai_suggested_investment_limit_factor"] = (
                        causal_suggestion["investment_limit_factor"]
                    )
                    suggested_anything = True
                except Exception as e:
                    st.warning(
                        "Não foi possível sugerir os limiares de Análise Causal "
                        f"automaticamente a partir do arquivo de Investimento: {e}. "
                        "Mantendo os valores atuais."
                    )

            # IA part: só roda com chave do Gemini configurada.
            if not (gemini_key or os.environ.get("GEMINI_API_KEY")) or not gemini_model:
                st.warning(
                    "Preencha a Chave de API do Gemini para sugerir Coluna do KPI, "
                    "KPI monetário e Objetivo da Otimização."
                )
            else:
                with st.spinner("Analisando arquivos com IA..."):
                    try:
                        active_key = gemini_key or os.environ.get("GEMINI_API_KEY")
                        performance_sample = read_csv_robust(perf_path, nrows=5)

                        import google.generativeai as genai

                        genai.configure(api_key=active_key)
                        model = genai.GenerativeModel(gemini_model)
                        suggestion = suggest_form_fields(model, performance_sample)

                        st.session_state["ai_suggested_kpi_column"] = suggestion["kpi_column"]
                        st.session_state["ai_suggested_kpi_is_monetary"] = suggestion["kpi_is_monetary"]
                        st.session_state["ai_suggested_optimization_target"] = suggestion["optimization_target"]
                        suggested_anything = True
                    except Exception as e:
                        st.warning(
                            f"Não foi possível gerar sugestão automática: {e}. "
                            "Mantendo os valores atuais."
                        )

            if suggested_anything:
                st.rerun()
```

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine" && uv run pytest -q`
Expected: all tests pass, same count as before Task 1 plus the 7 new ones (no test file targets `streamlit_app.py`'s form handler directly — it's Streamlit glue code, verified manually in Step 6, matching how the rest of this file is already tested only via the `ast`-extracted-pure-function pattern in `tests/test_streamlit_throttle.py`)

- [ ] **Step 6: Manual verification in the browser**

Run: `cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine" && uv run streamlit run scripts/streamlit_app.py`

In the browser:
1. Log in / land on the form.
2. Upload `exemplo_csv/Tcross/Investimento_Tcross_2026_v2.csv` as Investimento and `exemplo_csv/Tcross/Leads_tcross_2026_v2.csv` as Performance.
3. **Without** a Gemini API key filled in, click "Sugerir com IA". Expect: a warning about the missing Gemini key (for KPI/objetivo), but "Dias Mínimos Pré-Evento", "Var. Mínima de Aumento/Queda de Investimento (%)" and "Limite de Investimento Simulado" update to new values (not the old 14/40/30/1.5 defaults) after the page reruns. Expand "Configurações da Análise Causal" to confirm.
4. Fill in a valid Gemini API key and click "Sugerir com IA" again. Expect: both the causal fields (still reflecting step 3's values) and Coluna do KPI/Objetivo update, no errors shown.
5. Confirm "Construir Motor de Oportunidades" still works end-to-end with these prefilled values (reuses the bugfix already applied to `data_preprocessor.py` in this session).

- [ ] **Step 7: Commit**

```bash
cd "C:\Users\floppydisk\Documents\Trabalho\opportunity-engine"
git add scripts/streamlit_app.py
git commit -m "feat(streamlit): auto-suggest causal-analysis thresholds from investment history"
```
