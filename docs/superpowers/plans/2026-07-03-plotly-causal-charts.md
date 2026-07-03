# Plotly Conversion for Impacto Causal Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 4 static matplotlib PNGs in the "Impacto Causal" Streamlit tab (accuracy, causal line chart, investment bar, sessions bar) with interactive Plotly charts, without touching the Gemini HTML report or breaking old output folders.

**Architecture:** Persist the 4 dataframes already computed by `analysis.run_causal_impact_analysis` as CSVs (new `save_causal_chart_data` helper in `presentation.py`), add 4 pure `build_*(df) -> go.Figure` functions to `dashboard_charts.py` following the file's existing pattern, and render them in `streamlit_app.py`'s tab2 (both the online/HTML-report and offline/MD-report branches) gated on the CSVs' existence — old runs without CSVs keep their current PNG/iframe behavior untouched.

**Tech Stack:** Python 3.13, pandas, plotly (graph_objects), Streamlit, pytest (unittest-style), `mise exec -- uv run --with pytest pytest`.

---

## Reference

- Spec: `docs/superpowers/specs/2026-07-03-plotly-causal-charts-design.md`
- Existing Plotly builder patterns to mirror: `scripts/dashboard_charts.py` (e.g. `build_icpa_curve`, `build_revenue_roi_curve`)
- Existing dataframe shapes (source of truth): `scripts/analysis.py:479-500` (`run_causal_impact_analysis` return values)
- Existing matplotlib chart logic being ported: `scripts/presentation.py:77-225` (`save_accuracy_plot`, `save_line_chart_plot`, `save_investment_bar_plot`, `save_sessions_bar_plot`)

**Known pre-existing issue (not in scope):** `tests/test_presentation.py::TestPresentationChartsTranslation::test_save_line_chart_plot_translation` currently fails on `main` (asserts a stale title string, unrelated to this work). Don't try to fix it as part of this plan — if it's still the only failure at the end, that's expected.

---

### Task 1: Persist causal chart dataframes as CSV

**Files:**
- Modify: `scripts/presentation.py`
- Test: `tests/test_presentation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_presentation.py` (new imports go in the existing `from presentation import (...)` block, new test class at the end before `if __name__ == "__main__":`):

```python
import tempfile
```
(add this near the top, with the other stdlib imports)

```python
from presentation import (
    save_accuracy_plot,
    save_line_chart_plot,
    save_investment_bar_plot,
    save_sessions_bar_plot,
    save_causal_chart_data,
)
```

```python
class TestSaveCausalChartData(unittest.TestCase):
    def test_persists_all_four_dataframes_with_mae_column(self):
        line_df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2026-01-01", periods=3),
                "Actual_KPI": [10, 20, 15],
                "Forecasted_KPI": [12, 18, 14],
                "Investment": [100, 200, 150],
            }
        )
        inv_bar_df = pd.DataFrame({"Investment": [1000, 2000]}, index=["Pre-Event", "Event"])
        inv_bar_df.index.name = "Period"
        sessions_bar_df = pd.DataFrame({"kpi": [50, 60]}, index=["Forecasted", "Actual"])
        sessions_bar_df.index.name = "Category"
        accuracy_df = pd.DataFrame(
            {
                "Date": pd.date_range(start="2026-01-01", periods=3),
                "kpi": [10, 20, 15],
                "Predicted": [12, 18, 14],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_causal_chart_data(
                tmp_dir, line_df, inv_bar_df, sessions_bar_df, accuracy_df, mae=12.34
            )

            line_out = pd.read_csv(os.path.join(tmp_dir, "line_chart_data.csv"))
            self.assertEqual(list(line_out["Actual_KPI"]), [10, 20, 15])
            self.assertEqual(list(line_out["Investment"]), [100, 200, 150])

            inv_out = pd.read_csv(os.path.join(tmp_dir, "investment_data.csv"), index_col=0)
            self.assertEqual(list(inv_out.index), ["Pre-Event", "Event"])
            self.assertEqual(list(inv_out["Investment"]), [1000, 2000])

            sessions_out = pd.read_csv(os.path.join(tmp_dir, "sessions_data.csv"), index_col=0)
            self.assertEqual(list(sessions_out.index), ["Forecasted", "Actual"])
            self.assertEqual(list(sessions_out["kpi"]), [50, 60])

            accuracy_out = pd.read_csv(os.path.join(tmp_dir, "accuracy_data.csv"))
            self.assertEqual(list(accuracy_out["Predicted"]), [12, 18, 14])
            self.assertEqual(list(accuracy_out["mae"]), [12.34, 12.34, 12.34])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_presentation.py::TestSaveCausalChartData -v`
Expected: FAIL with `ImportError: cannot import name 'save_causal_chart_data'`

- [ ] **Step 3: Implement `save_causal_chart_data`**

Add to `scripts/presentation.py`, right after `save_sessions_bar_plot` (after line 225, before `save_opportunity_curve_plot`):

```python
def save_causal_chart_data(event_output_dir, line_df, inv_bar_df, sessions_bar_df, accuracy_df, mae):
    """Persists the raw dataframes behind the causal charts so Streamlit can render them as Plotly."""
    line_df.to_csv(os.path.join(event_output_dir, "line_chart_data.csv"), index=False)
    inv_bar_df.to_csv(os.path.join(event_output_dir, "investment_data.csv"))
    sessions_bar_df.to_csv(os.path.join(event_output_dir, "sessions_data.csv"))

    accuracy_out = accuracy_df.copy()
    accuracy_out["mae"] = mae
    accuracy_out.to_csv(os.path.join(event_output_dir, "accuracy_data.csv"), index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_presentation.py::TestSaveCausalChartData -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/presentation.py tests/test_presentation.py
git commit -m "feat(dashboard): add save_causal_chart_data to persist causal chart dataframes"
```

---

### Task 2: Add `build_accuracy_chart`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Test: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Write the failing test**

Add `build_accuracy_chart` to the import tuple at the top of `tests/test_dashboard_charts.py`:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
)
```

Add this test class before `if __name__ == "__main__":`:

```python
class TestBuildAccuracyChart(unittest.TestCase):

    def setUp(self):
        self.accuracy_df = pd.DataFrame({
            "Date": pd.date_range(start="2026-01-01", periods=3),
            "kpi": [10, 20, 15],
            "Predicted": [12, 18, 14],
            "mae": [1.5, 1.5, 1.5],
        })

    def test_plots_real_and_predicted_lines(self):
        fig = build_accuracy_chart(self.accuracy_df, kpi_name="Vendas")
        names = [trace.name for trace in fig.data]
        self.assertIn("Vendas Real", names)
        self.assertIn("Vendas Previsto (In-Sample)", names)

        real_trace = fig.data[names.index("Vendas Real")]
        self.assertEqual(list(real_trace.y), [10, 20, 15])
        pred_trace = fig.data[names.index("Vendas Previsto (In-Sample)")]
        self.assertEqual(list(pred_trace.y), [12, 18, 14])

    def test_shows_mae_annotation(self):
        fig = build_accuracy_chart(self.accuracy_df)
        self.assertEqual(len(fig.layout.annotations), 1)
        self.assertIn("1.50", fig.layout.annotations[0].text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildAccuracyChart -v`
Expected: FAIL with `ImportError: cannot import name 'build_accuracy_chart'`

- [ ] **Step 3: Implement `build_accuracy_chart`**

Add to `scripts/dashboard_charts.py`, after `build_events_overview` (end of file):

```python
def build_accuracy_chart(accuracy_df, kpi_name="kpi"):
    """Real vs Predicted (in-sample) KPI line chart, with MAE shown as an annotation."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["Date"],
            y=accuracy_df["kpi"],
            mode="lines",
            name=f"{kpi_name} Real",
            line=dict(color="black", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["Date"],
            y=accuracy_df["Predicted"],
            mode="lines",
            name=f"{kpi_name} Previsto (In-Sample)",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    mae = accuracy_df["mae"].iloc[0] if "mae" in accuracy_df.columns and not accuracy_df.empty else 0
    fig.add_annotation(
        text=f"MAE (últimos 90 dias): {mae:.2f}",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        bgcolor="wheat",
        opacity=0.8,
        align="left",
    )

    fig.update_layout(
        title="Acurácia do Modelo: Real vs. Previsto (Período Pré-Evento)",
        yaxis_title=kpi_name,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildAccuracyChart -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_accuracy_chart"
```

---

### Task 3: Add `build_causal_line_chart`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Test: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Write the failing test**

Update the import tuple at the top of `tests/test_dashboard_charts.py` to:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
)
```

Add this test class:

```python
class TestBuildCausalLineChart(unittest.TestCase):

    def setUp(self):
        self.line_df = pd.DataFrame({
            "Date": pd.date_range(start="2026-01-01", periods=3),
            "Actual_KPI": [10, 20, 15],
            "Forecasted_KPI": [12, 18, 14],
            "Investment": [100, 200, 150],
        })

    def test_plots_kpi_lines_on_primary_axis(self):
        fig = build_causal_line_chart(self.line_df, kpi_name="Vendas")
        names = [trace.name for trace in fig.data]
        self.assertIn("Vendas Real", names)
        self.assertIn("Vendas Previsto", names)
        real_trace = fig.data[names.index("Vendas Real")]
        self.assertEqual(list(real_trace.y), [10, 20, 15])

    def test_plots_investment_bars_on_secondary_axis(self):
        fig = build_causal_line_chart(self.line_df)
        names = [trace.name for trace in fig.data]
        inv_trace = fig.data[names.index("Investimento")]
        self.assertEqual(inv_trace.type, "bar")
        self.assertEqual(inv_trace.yaxis, "y2")
        self.assertEqual(list(inv_trace.y), [100, 200, 150])
        self.assertEqual(fig.layout.yaxis2.overlaying, "y")
        self.assertEqual(fig.layout.yaxis2.side, "right")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildCausalLineChart -v`
Expected: FAIL with `ImportError: cannot import name 'build_causal_line_chart'`

- [ ] **Step 3: Implement `build_causal_line_chart`**

Add to `scripts/dashboard_charts.py`, after `build_accuracy_chart`:

```python
def build_causal_line_chart(line_df, kpi_name="kpi"):
    """Dual-axis chart: Real vs Forecasted KPI (left axis) with Investment overlay (right axis, bars)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_df["Date"],
            y=line_df["Actual_KPI"],
            mode="lines",
            name=f"{kpi_name} Real",
            line=dict(color="black", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_df["Date"],
            y=line_df["Forecasted_KPI"],
            mode="lines",
            name=f"{kpi_name} Previsto",
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=line_df["Date"],
            y=line_df["Investment"],
            name="Investimento",
            yaxis="y2",
            marker=dict(color="blue", opacity=0.3),
        )
    )

    fig.update_layout(
        title="Análise de Impacto Causal: Real vs. Previsto",
        xaxis_title="Data",
        yaxis=dict(title=kpi_name),
        yaxis2=dict(title="Investimento", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildCausalLineChart -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_causal_line_chart"
```

---

### Task 4: Add `build_investment_bar_chart`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Test: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Write the failing test**

Update the import tuple at the top of `tests/test_dashboard_charts.py` to:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
)
```

Add this test class:

```python
class TestBuildInvestmentBarChart(unittest.TestCase):

    def test_renames_periods_and_colors_bars(self):
        inv_bar_df = pd.DataFrame({"Investment": [1000, 2000]}, index=["Pre-Event", "Event"])
        inv_bar_df.index.name = "Period"

        fig = build_investment_bar_chart(inv_bar_df)
        self.assertEqual(list(fig.data[0].x), ["Pré-Evento", "Evento"])
        self.assertEqual(list(fig.data[0].y), [1000, 2000])
        self.assertEqual(list(fig.data[0].marker.color), ["gray", "green"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildInvestmentBarChart -v`
Expected: FAIL with `ImportError: cannot import name 'build_investment_bar_chart'`

- [ ] **Step 3: Implement `build_investment_bar_chart`**

Add to `scripts/dashboard_charts.py`, after `build_causal_line_chart`:

```python
def build_investment_bar_chart(inv_bar_df):
    """Pre-event vs event investment, 2-bar comparison."""
    labels = {"Pre-Event": "Pré-Evento", "Event": "Evento"}
    x_labels = [labels.get(idx, idx) for idx in inv_bar_df.index]
    colors = ["gray" if idx == "Pre-Event" else "green" for idx in inv_bar_df.index]

    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=inv_bar_df["Investment"],
            marker=dict(color=colors),
        )
    )
    fig.update_layout(
        title="Investimento: Pré-Evento vs. Evento",
        yaxis_title="Investimento Total",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildInvestmentBarChart -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_investment_bar_chart"
```

---

### Task 5: Add `build_sessions_bar_chart`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Test: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Write the failing test**

Update the import tuple at the top of `tests/test_dashboard_charts.py` to:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
)
```

Add this test class:

```python
class TestBuildSessionsBarChart(unittest.TestCase):

    def test_renames_categories_and_colors_bars(self):
        sessions_bar_df = pd.DataFrame({"kpi": [100, 150]}, index=["Forecasted", "Actual"])
        sessions_bar_df.index.name = "Category"

        fig = build_sessions_bar_chart(sessions_bar_df, kpi_name="Cliques")
        self.assertEqual(list(fig.data[0].x), ["Cliques Previsto", "Cliques Real"])
        self.assertEqual(list(fig.data[0].y), [100, 150])
        self.assertEqual(list(fig.data[0].marker.color), ["red", "black"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildSessionsBarChart -v`
Expected: FAIL with `ImportError: cannot import name 'build_sessions_bar_chart'`

- [ ] **Step 3: Implement `build_sessions_bar_chart`**

Add to `scripts/dashboard_charts.py`, after `build_investment_bar_chart`:

```python
def build_sessions_bar_chart(sessions_bar_df, kpi_name="kpi"):
    """Forecasted vs actual KPI totals, 2-bar comparison."""
    labels = {
        "Forecasted": f"{kpi_name} Previsto",
        "Actual": f"{kpi_name} Real",
    }
    x_labels = [labels.get(idx, idx) for idx in sessions_bar_df.index]
    colors = ["red" if idx == "Forecasted" else "black" for idx in sessions_bar_df.index]

    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=sessions_bar_df["kpi"],
            marker=dict(color=colors),
        )
    )
    fig.update_layout(
        title=f"{kpi_name} Real vs. Previsto",
        yaxis_title=f"Total de {kpi_name}",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py::TestBuildSessionsBarChart -v`
Expected: PASS

- [ ] **Step 5: Run the full dashboard_charts + presentation test files to check for regressions**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py tests/test_presentation.py -v`
Expected: All pass except the pre-existing, unrelated `test_save_line_chart_plot_translation` failure noted in the Reference section above.

- [ ] **Step 6: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_sessions_bar_chart"
```

---

### Task 6: Wire `save_causal_chart_data` into both local_main entry points

**Files:**
- Modify: `scripts/local_main.py:32-41` (import block), `scripts/local_main.py:358-362` (call site)
- Modify: `scripts/local_main-without-gemini.py:32-41` (import block), `scripts/local_main-without-gemini.py:356-360` (call site)

This is a straight wiring task (calling a function already tested in Task 1) — no new automated test, verified manually in Step 3.

- [ ] **Step 1: Update imports in `scripts/local_main.py`**

Find:
```python
from presentation import (
    save_accuracy_plot,
    save_line_chart_plot,
    save_investment_bar_plot,
    save_sessions_bar_plot,
    save_opportunity_curve_plot,
    create_comparative_saturation_md,
    save_investment_distribution_donuts,
    create_presentation_dataframe,
)
```

Replace with:
```python
from presentation import (
    save_accuracy_plot,
    save_line_chart_plot,
    save_investment_bar_plot,
    save_sessions_bar_plot,
    save_causal_chart_data,
    save_opportunity_curve_plot,
    create_comparative_saturation_md,
    save_investment_distribution_donuts,
    create_presentation_dataframe,
)
```

- [ ] **Step 2: Add the call site in `scripts/local_main.py`**

Find (around line 358-362, right after the `save_sessions_bar_plot(...)` call, before the "Generate and save the comprehensive presentation data CSV" comment):

```python
                                save_sessions_bar_plot(
                                    analyzed_event["sessions_bar_df"],
                                    image_paths["sessions"],
                                    kpi_name=kpi_col,
                                )

                                # Generate and save the comprehensive presentation data CSV for this event
```

Replace with:

```python
                                save_sessions_bar_plot(
                                    analyzed_event["sessions_bar_df"],
                                    image_paths["sessions"],
                                    kpi_name=kpi_col,
                                )

                                save_causal_chart_data(
                                    event_output_dir,
                                    analyzed_event["line_df"],
                                    analyzed_event["inv_bar_df"],
                                    analyzed_event["sessions_bar_df"],
                                    analyzed_event["accuracy_df"],
                                    mae=results_data.get("mae", 0),
                                )

                                # Generate and save the comprehensive presentation data CSV for this event
```

- [ ] **Step 3: Repeat Steps 1-2 identically in `scripts/local_main-without-gemini.py`**

Same two edits, same anchor text (the call site is at line 356-360 instead of 358-362 in this file, but the surrounding text is identical — search for the same `save_sessions_bar_plot(...)` block).

- [ ] **Step 4: Verify both files still compile cleanly**

Run: `mise exec -- python -m py_compile scripts/local_main.py scripts/local_main-without-gemini.py`
Expected: No output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/local_main.py scripts/local_main-without-gemini.py
git commit -m "feat(causal): persist chart dataframes as CSV during event analysis"
```

---

### Task 7: Render Plotly charts in Streamlit tab2

**Files:**
- Modify: `scripts/streamlit_app.py:267-273` (import block), `scripts/streamlit_app.py:1083-1085` (render block, offsets may shift slightly after Task 6 changes elsewhere — search for the anchor text below, it's unique)

No automated test (no existing test file covers `streamlit_app.py` page code — verified manually in Task 8).

- [ ] **Step 1: Update the dashboard_charts import**

Find:
```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
)
```

Replace with:
```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
)
```

- [ ] **Step 2: Add the Plotly render block after both tab2 branches**

Find (the unique anchor is the "Nenhum relatório encontrado" warning followed by the `if event_dirs:`-closing `else:`):

```python
                else:
                    st.warning("Nenhum relatório encontrado para este evento.")
        else:
            st.info(
```

Replace with:

```python
                else:
                    st.warning("Nenhum relatório encontrado para este evento.")

            csv_names = [
                "line_chart_data.csv",
                "accuracy_data.csv",
                "investment_data.csv",
                "sessions_data.csv",
            ]
            if all(os.path.exists(os.path.join(selected_dir, f)) for f in csv_names):
                st.markdown("### Gráficos Interativos")
                kpi_name = active_config.get("primary_business_metric_name", "kpi")

                line_chart_df = pd.read_csv(os.path.join(selected_dir, "line_chart_data.csv"))
                st.plotly_chart(
                    build_causal_line_chart(line_chart_df, kpi_name=kpi_name),
                    use_container_width=True,
                )

                accuracy_chart_df = pd.read_csv(os.path.join(selected_dir, "accuracy_data.csv"))
                st.plotly_chart(
                    build_accuracy_chart(accuracy_chart_df, kpi_name=kpi_name),
                    use_container_width=True,
                )

                col1, col2 = st.columns(2)
                investment_chart_df = pd.read_csv(
                    os.path.join(selected_dir, "investment_data.csv"), index_col=0
                )
                sessions_chart_df = pd.read_csv(
                    os.path.join(selected_dir, "sessions_data.csv"), index_col=0
                )
                with col1:
                    st.plotly_chart(
                        build_investment_bar_chart(investment_chart_df),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        build_sessions_bar_chart(sessions_chart_df, kpi_name=kpi_name),
                        use_container_width=True,
                    )

        else:
            st.info(
```

This block sits at the same indentation level as `if html_in_dir:` — a sibling that always runs after that branch, for both the online (iframe) and offline (MD+PNG) paths, gated on all 4 CSVs existing.

- [ ] **Step 3: Verify the file compiles**

Run: `mise exec -- python -m py_compile scripts/streamlit_app.py`
Expected: No output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(dashboard): render interactive causal charts in Impacto Causal tab"
```

---

### Task 8: End-to-end manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the offline engine against a config with at least one valid event**

`inputs/` and `outputs/` are gitignored — this repo checkout has no advertiser config or CSV data files locally (only leftover `outputs/*/global_saturation_analysis/` artifacts from past runs, no per-event "Impacto Causal" folders yet). Use your own advertiser config (see `config.example.json` at the repo root for the schema) placed in `inputs/`, pointing at investment/performance data with at least one event that will pass the significance/R² thresholds.

Run: `mise exec -- python scripts/local_main-without-gemini.py --config inputs/<your_config>.json`

Confirm in the output: the event's output directory (`outputs/<advertiser>/<channel>/<date>/`) now contains `line_chart_data.csv`, `accuracy_data.csv`, `investment_data.csv`, `sessions_data.csv` alongside the existing `*.png` files.

- [ ] **Step 2: Launch Streamlit and check the new charts render**

Run: `mise exec -- streamlit run scripts/streamlit_app.py`

In the browser: go to "Dashboard de Impacto Causal", select the event from Step 1. Confirm:
- The existing report (MD text + PNGs, or HTML iframe) still renders as before.
- A new "### Gráficos Interativos" section appears below it with 4 Plotly charts: causal line chart (with investment bar overlay), accuracy chart (with MAE annotation), and investment/sessions bar charts side by side.
- Hover tooltips work on all 4 charts.

- [ ] **Step 3: Check backward compatibility with an old (pre-change) output folder**

No pre-existing per-event "Impacto Causal" output folder exists in this checkout to test against directly, so simulate one: temporarily move the 4 new CSVs (`line_chart_data.csv`, `accuracy_data.csv`, `investment_data.csv`, `sessions_data.csv`) out of the event's output directory from Step 1, reload the Streamlit page, and confirm:
- No errors.
- No empty/broken "Gráficos Interativos" section — it simply doesn't appear, and the existing PNG/iframe display is unchanged.

Then move the CSVs back.

- [ ] **Step 4: Run the full test suite one last time**

Run: `mise run test`
Expected: All pass except the pre-existing `test_save_line_chart_plot_translation` failure (see Reference section) — confirm no *other* failures were introduced.
