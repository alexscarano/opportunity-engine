# Plotly Conversion for Impacto Causal Charts

## Problem

The "Impacto Causal" tab (tab2 in `streamlit_app.py`) shows 4 per-event charts (model accuracy, causal line chart, investment bar, sessions/KPI bar) as static matplotlib PNGs. The dataframes behind them (`line_df`, `inv_bar_df`, `sessions_bar_df`, `accuracy_df`, returned by `analysis.run_causal_impact_analysis`) are used once to render the PNGs in `local_main.py` / `local_main-without-gemini.py`, then discarded. This means the charts can't be interactive (zoom, hover, tooltips) the way the Elasticidade tab's charts already are (`dashboard_charts.py`).

## Goals

- Replace the 4 static PNGs in tab2 with interactive Plotly charts, following the existing `dashboard_charts.py` pattern (pure `build_*(df) -> go.Figure` functions).
- Keep the Gemini HTML report (`gemini_report_*.html`) untouched — it keeps embedding matplotlib PNGs as before.
- Don't break old output folders that only have PNGs (no persisted data).

## Non-goals

- No changes to `gemini_report.py`'s image embedding.
- No removal of matplotlib/PNG generation in `presentation.py` — it still feeds the HTML report.
- No new dependencies (no kaleido, no server-side PNG export from Plotly).

## Design

### 1. Data persistence (`presentation.py`)

New helper, called once per event right after the existing `save_*_plot(...)` calls:

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

- Filenames are fixed (no `file_base_name` suffix) — `event_output_dir` is already unique per channel+date, so the suffix used by the PNGs would be redundant here.
- `inv_bar_df` (index=`Period`: Pre-Event/Event) and `sessions_bar_df` (index=`Category`: Forecasted/Actual) keep their index on write — it's the bar label.
- `accuracy_df` carries `mae` as a column so the CSV is self-contained (the matplotlib version pulls it separately from `results_data["mae"]` for the annotation box).

Called identically from `local_main.py` and `local_main-without-gemini.py`, next to their existing `save_accuracy_plot`/`save_line_chart_plot`/`save_investment_bar_plot`/`save_sessions_bar_plot` calls (~line 329-362 in both files).

### 2. Plotly chart builders (`dashboard_charts.py`)

Four new pure functions, matching the file's existing style (`go.Figure`, `hovertemplate`, horizontal legend anchored above the plot, `margin=dict(l=20, r=20, t=50, b=20)`):

- `build_accuracy_chart(accuracy_df, kpi_name="kpi")` — Real (black solid) vs Previsto/In-Sample (red dashed) line over `Date`; MAE (read from the `mae` column) shown as an annotation in the top-left, mirroring the boxed text in `save_accuracy_plot`.
- `build_causal_line_chart(line_df, kpi_name="kpi")` — dual-axis: Real/Previsto KPI lines on the left y-axis, `Investment` as semi-transparent bars on the right y-axis (mirrors `save_line_chart_plot`).
- `build_investment_bar_chart(inv_bar_df)` — 2-bar chart: Pré-Evento (gray) vs Evento (green). Renames the `Pre-Event`/`Event` index values for display, same as `save_investment_bar_plot`.
- `build_sessions_bar_chart(sessions_bar_df, kpi_name="kpi")` — 2-bar chart: Previsto (red) vs Real (black). Renames `Forecasted`/`Actual`, same as `save_sessions_bar_plot`.

### 3. Streamlit wiring (`streamlit_app.py`, tab2)

One shared block, appended after **both** existing branches (the HTML-report/online iframe, and the MD-report/offline PNG fallback) — same `selected_dir` is in scope for both:

```python
csv_names = ["line_chart_data.csv", "accuracy_data.csv", "investment_data.csv", "sessions_data.csv"]
if all(os.path.exists(os.path.join(selected_dir, f)) for f in csv_names):
    st.markdown("### Gráficos Interativos")
    kpi_name = active_config.get("primary_business_metric_name", "kpi")

    line_df = pd.read_csv(os.path.join(selected_dir, "line_chart_data.csv"))
    st.plotly_chart(build_causal_line_chart(line_df, kpi_name=kpi_name), use_container_width=True)

    accuracy_df = pd.read_csv(os.path.join(selected_dir, "accuracy_data.csv"))
    st.plotly_chart(build_accuracy_chart(accuracy_df, kpi_name=kpi_name), use_container_width=True)

    col1, col2 = st.columns(2)
    inv_df = pd.read_csv(os.path.join(selected_dir, "investment_data.csv"), index_col=0)
    sessions_df = pd.read_csv(os.path.join(selected_dir, "sessions_data.csv"), index_col=0)
    with col1:
        st.plotly_chart(build_investment_bar_chart(inv_df), use_container_width=True)
    with col2:
        st.plotly_chart(build_sessions_bar_chart(sessions_df, kpi_name=kpi_name), use_container_width=True)
```

`active_config` is already loaded at the top of tab2 (used today for the event-detection thresholds text), so it's in scope for both branches.

Order: line chart (main summary) → accuracy (model validation) → investment/sessions side-by-side. Matches the layout order already used in the Gemini HTML report.

### Fallback behavior (old output folders without CSVs)

- **Offline/MD branch:** unchanged — if the 4 CSVs are missing, the existing `st.image(png_files)` block still renders the PNGs exactly as it does today. No double-rendering: the new Plotly block simply doesn't fire.
- **Online/HTML branch:** if CSVs are missing, nothing new renders. The iframe (with its baked-in PNGs) is all you get, same as today.

## Testing

Manual verification (per project convention — Streamlit UI, no dedicated test suite for this module):
1. Run `local_main-without-gemini.py` (or `local_main.py`) against a config with at least one valid event, confirm the 4 new CSVs appear in the event's output directory alongside the existing PNGs.
2. Open the Impacto Causal tab in Streamlit for that event, confirm the 4 Plotly charts render (hover tooltips work, dual-axis/investment overlay is legible) below the existing report content.
3. Point Streamlit at an old output directory (pre-change, PNGs only) and confirm the existing PNG/iframe behavior is unchanged (no errors, no empty "Gráficos Interativos" section).
