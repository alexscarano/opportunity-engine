# Mais Gráficos e Insights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new Plotly charts (4 in the Elasticidade tab, 1 in the Impacto Causal tab) that surface data the pipeline already computes but the Streamlit UI never displays — no changes to `analysis.py`, `saturation_curve.py`, `elasticity_analysis.py`, or the pipeline entry points.

**Architecture:** New module `scripts/dashboard_charts.py` holds one pure function per chart — each takes an already-shaped `pandas.DataFrame` (plus a few scalar params) and returns a `plotly.graph_objects.Figure`. `scripts/streamlit_app.py` keeps doing its existing data loading/filtering and just calls these functions + `st.plotly_chart(...)`. Existing chart code (main saturation curve, donuts, static PNGs) is untouched.

**Tech Stack:** Python 3.13, pandas, numpy, plotly (`graph_objects` + `express`), Streamlit, pytest (via `uv run --with pytest pytest`, per `mise.toml`).

**Spec:** `docs/superpowers/specs/2026-07-03-more-charts-and-insights-design.md`

---

## Reference: exact data shapes used below

- `response_curve_data.csv` (loaded as `df` in `streamlit_app.py` tab3, columns already include after existing code runs): `Monthly_Investment`, `Monthly_KPI`, `CPA`, `iCPA`, `Projected_Revenue`, `Incremental_ROI`, and per channel `Spend_{channel}_Historical` / `_Optimized` / `_Strategic`.
- `individual_response_curves_data.csv`: columns `Channel`, `Channel_Spend`, `Projected_Total_KPIs`.
- `detected_events.csv`: columns `date` (e.g. `2025-01-06`), `ad_product` (e.g. `"AWIN, BING, KAIAK"`), `percentage_change` (signed float).
- Event report directories on disk: `outputs/<advertiser>/<ad_product with ", " replaced by "_">/<date>/` (built in `local_main.py:343-350`).

---

### Task 1: `dashboard_charts.py` scaffold + `build_icpa_curve`

**Files:**
- Create: `scripts/dashboard_charts.py`
- Create: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_charts.py`:

```python
import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from dashboard_charts import build_icpa_curve


class TestBuildIcpaCurve(unittest.TestCase):

    def setUp(self):
        self.df_plot = pd.DataFrame({
            "Monthly_Investment": [0, 1000, 2000, 3000],
            "iCPA": [np.inf, 5.0, 6.0, 8.0],
        })
        self.optimal_point = pd.Series({"Monthly_Investment": 2000, "iCPA": 6.0})
        self.saturation_point = pd.Series({"Monthly_Investment": 3000, "iCPA": 8.0})

    def test_filters_non_finite_and_plots_line(self):
        fig = build_icpa_curve(
            self.df_plot, self.optimal_point, self.saturation_point, kpi_name="Vendas"
        )
        self.assertEqual(fig.data[0].name, "iCPA Marginal")
        self.assertEqual(list(fig.data[0].x), [1000, 2000, 3000])
        self.assertEqual(list(fig.data[0].y), [5.0, 6.0, 8.0])
        self.assertEqual(fig.layout.yaxis.title.text, "iCPA Marginal (Vendas)")

    def test_marks_optimal_and_saturation_points(self):
        fig = build_icpa_curve(
            self.df_plot, self.optimal_point, self.saturation_point
        )
        names = [trace.name for trace in fig.data]
        self.assertIn("Ponto Escolhido (Ótimo)", names)
        self.assertIn("Cenário de Saturação", names)
        optimal_trace = fig.data[names.index("Ponto Escolhido (Ótimo)")]
        self.assertEqual(list(optimal_trace.x), [2000])
        self.assertEqual(list(optimal_trace.y), [6.0])

    def test_adds_target_reference_lines_only_when_provided(self):
        fig_no_targets = build_icpa_curve(
            self.df_plot, self.optimal_point, self.saturation_point
        )
        self.assertEqual(len(fig_no_targets.layout.shapes), 0)

        fig_with_targets = build_icpa_curve(
            self.df_plot,
            self.optimal_point,
            self.saturation_point,
            target_cpa=7.0,
            target_icpa=4.0,
        )
        self.assertEqual(len(fig_with_targets.layout.shapes), 2)
        y_values = sorted(shape.y0 for shape in fig_with_targets.layout.shapes)
        self.assertEqual(y_values, [4.0, 7.0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard_charts'`

- [ ] **Step 3: Create `scripts/dashboard_charts.py` with `build_icpa_curve`**

```python
# -*- coding: utf-8 -*-
"""Plotly figure builders for the Streamlit dashboard (Elasticidade and Impacto Causal tabs).

Every function here is pure: it takes an already-shaped DataFrame (and a few
scalars) and returns a plotly.graph_objects.Figure. All data loading/filtering
stays in streamlit_app.py.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_icpa_curve(
    df_plot,
    optimal_point,
    saturation_point,
    kpi_name="kpi",
    target_cpa=None,
    target_icpa=None,
):
    """Marginal iCPA vs Monthly_Investment, with optional target reference lines."""
    finite_df = df_plot[np.isfinite(df_plot["iCPA"])]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=finite_df["Monthly_Investment"],
            y=finite_df["iCPA"],
            mode="lines",
            name="iCPA Marginal",
            line=dict(color="purple", width=3),
            hovertemplate="<b>Investimento:</b> R$ %{x:.2s}<br><b>iCPA:</b> R$ %{y:.2f}<extra></extra>",
        )
    )

    if optimal_point is not None and np.isfinite(optimal_point.get("iCPA", np.nan)):
        fig.add_trace(
            go.Scatter(
                x=[optimal_point["Monthly_Investment"]],
                y=[optimal_point["iCPA"]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                name="Ponto Escolhido (Ótimo)",
            )
        )

    if saturation_point is not None and np.isfinite(saturation_point.get("iCPA", np.nan)):
        fig.add_trace(
            go.Scatter(
                x=[saturation_point["Monthly_Investment"]],
                y=[saturation_point["iCPA"]],
                mode="markers",
                marker=dict(color="orange", size=12, symbol="diamond"),
                name="Cenário de Saturação",
            )
        )

    if target_cpa is not None:
        fig.add_hline(
            y=target_cpa,
            line_dash="dash",
            line_color="green",
            annotation_text="Target CPA",
            annotation_position="top left",
        )

    if target_icpa is not None:
        fig.add_hline(
            y=target_icpa,
            line_dash="dot",
            line_color="crimson",
            annotation_text="Target iCPA Marginal",
            annotation_position="bottom left",
        )

    fig.update_layout(
        xaxis_title="Investimento Mensal",
        yaxis_title=f"iCPA Marginal ({kpi_name})",
        xaxis=dict(tickformat=".2s"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_icpa_curve chart"
```

---

### Task 2: `build_revenue_roi_curve`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Modify: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Add the failing test**

Add to `tests/test_dashboard_charts.py` (append import name and new test class):

```python
from dashboard_charts import build_icpa_curve, build_revenue_roi_curve
```

(replace the existing single-name import line with the one above)

```python
class TestBuildRevenueRoiCurve(unittest.TestCase):

    def setUp(self):
        self.df_plot = pd.DataFrame({
            "Monthly_Investment": [1000, 2000, 3000],
            "Projected_Revenue": [100, 150, 200],
            "Incremental_ROI": [np.inf, 2.5, 3.0],
        })

    def test_revenue_trace_scales_daily_to_monthly(self):
        fig = build_revenue_roi_curve(self.df_plot, kpi_name="Vendas")
        self.assertEqual(fig.data[0].name, "Receita Projetada (Mensal)")
        self.assertEqual(list(fig.data[0].y), [3000, 4500, 6000])

    def test_roi_trace_filters_non_finite_and_uses_secondary_axis(self):
        fig = build_revenue_roi_curve(self.df_plot)
        self.assertEqual(fig.data[1].name, "ROI Incremental")
        self.assertEqual(fig.data[1].yaxis, "y2")
        self.assertEqual(list(fig.data[1].x), [2000, 3000])
        self.assertEqual(list(fig.data[1].y), [2.5, 3.0])
        self.assertEqual(fig.layout.yaxis2.overlaying, "y")
        self.assertEqual(fig.layout.yaxis2.side, "right")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_revenue_roi_curve'`

- [ ] **Step 3: Add `build_revenue_roi_curve` to `scripts/dashboard_charts.py`**

Append:

```python


def build_revenue_roi_curve(df_plot, kpi_name="kpi"):
    """Dual-axis chart: Projected Revenue (left) and Incremental ROI (right) vs Monthly_Investment."""
    finite_roi = df_plot[np.isfinite(df_plot["Incremental_ROI"])]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["Monthly_Investment"],
            y=df_plot["Projected_Revenue"] * 30,
            mode="lines",
            name="Receita Projetada (Mensal)",
            line=dict(color="royalblue", width=3),
            hovertemplate="<b>Investimento:</b> R$ %{x:.2s}<br><b>Receita:</b> R$ %{y:.2s}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=finite_roi["Monthly_Investment"],
            y=finite_roi["Incremental_ROI"],
            mode="lines",
            name="ROI Incremental",
            line=dict(color="darkorange", width=3, dash="dash"),
            yaxis="y2",
            hovertemplate="<b>Investimento:</b> R$ %{x:.2s}<br><b>ROI Incremental:</b> %{y:.2f}x<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Investimento Mensal",
        yaxis=dict(title="Receita Projetada (R$)", tickformat=".2s"),
        yaxis2=dict(title="ROI Incremental", overlaying="y", side="right"),
        xaxis=dict(tickformat=".2s"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_revenue_roi_curve chart"
```

---

### Task 3: `build_channel_mix_evolution`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Modify: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Add the failing test**

Update the import line:

```python
from dashboard_charts import build_icpa_curve, build_revenue_roi_curve, build_channel_mix_evolution
```

Append:

```python
class TestBuildChannelMixEvolution(unittest.TestCase):

    def test_stacks_shares_ordered_by_average_share_descending(self):
        df_plot = pd.DataFrame({
            "Monthly_Investment": [1000, 2000],
            "Spend_GOOGLE_Strategic": [700, 1600],
            "Spend_META_Strategic": [300, 400],
        })
        fig = build_channel_mix_evolution(
            df_plot, baseline_monthly_inv=1000, optimal_monthly_inv=2000
        )
        names = [trace.name for trace in fig.data]
        self.assertEqual(names, ["GOOGLE", "META"])
        self.assertEqual(list(fig.data[0].y), [70.0, 80.0])
        self.assertEqual(list(fig.data[1].y), [30.0, 20.0])
        self.assertEqual(fig.data[0].stackgroup, "one")

    def test_adds_baseline_and_optimal_vlines(self):
        df_plot = pd.DataFrame({
            "Monthly_Investment": [1000, 2000],
            "Spend_GOOGLE_Strategic": [1000, 2000],
        })
        fig = build_channel_mix_evolution(
            df_plot, baseline_monthly_inv=1000, optimal_monthly_inv=2000
        )
        self.assertEqual(len(fig.layout.shapes), 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_channel_mix_evolution'`

- [ ] **Step 3: Add `build_channel_mix_evolution` to `scripts/dashboard_charts.py`**

Append:

```python


def build_channel_mix_evolution(df_plot, baseline_monthly_inv=None, optimal_monthly_inv=None):
    """100%-stacked area of each channel's Strategic-mix share vs Monthly_Investment."""
    strategic_cols = [
        c for c in df_plot.columns if c.startswith("Spend_") and c.endswith("_Strategic")
    ]
    channel_names = [c.replace("Spend_", "").replace("_Strategic", "") for c in strategic_cols]

    totals = df_plot[strategic_cols].sum(axis=1)
    shares = df_plot[strategic_cols].div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
    shares.columns = channel_names

    order = shares.mean().sort_values(ascending=False).index
    palette = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, channel in enumerate(order):
        fig.add_trace(
            go.Scatter(
                x=df_plot["Monthly_Investment"],
                y=shares[channel],
                mode="lines",
                name=channel,
                stackgroup="one",
                line=dict(width=0.5, color=palette[i % len(palette)]),
                hovertemplate=f"<b>{channel}:</b> %{{y:.1f}}%<extra></extra>",
            )
        )

    if baseline_monthly_inv:
        fig.add_vline(
            x=baseline_monthly_inv,
            line_dash="dash",
            line_color="green",
            annotation_text="Base Histórica",
        )
    if optimal_monthly_inv:
        fig.add_vline(
            x=optimal_monthly_inv,
            line_dash="dot",
            line_color="red",
            annotation_text="Ponto Ótimo",
        )

    fig.update_layout(
        xaxis_title="Investimento Mensal",
        yaxis_title="Participação no Mix (%)",
        xaxis=dict(tickformat=".2s"),
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_channel_mix_evolution chart"
```

---

### Task 4: `build_channel_saturation_comparison`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Modify: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Add the failing test**

Update the import line:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
)
```

Append:

```python
class TestBuildChannelSaturationComparison(unittest.TestCase):

    def test_normalizes_each_channel_to_its_own_0_100_range(self):
        individual_df = pd.DataFrame({
            "Channel": ["GOOGLE", "GOOGLE", "META", "META"],
            "Channel_Spend": [0, 100, 0, 50],
            "Projected_Total_KPIs": [10, 20, 5, 5],
        })
        fig = build_channel_saturation_comparison(individual_df)

        names = [trace.name for trace in fig.data]
        self.assertEqual(names, ["GOOGLE", "META"])

        google_trace = fig.data[names.index("GOOGLE")]
        self.assertEqual(list(google_trace.x), [0, 100])
        self.assertEqual(list(google_trace.y), [0.0, 100.0])

        # META has a flat curve (min == max): normalization must not divide by zero
        meta_trace = fig.data[names.index("META")]
        self.assertEqual(list(meta_trace.y), [0.0, 0.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_channel_saturation_comparison'`

- [ ] **Step 3: Add `build_channel_saturation_comparison` to `scripts/dashboard_charts.py`**

Append:

```python


def build_channel_saturation_comparison(individual_df):
    """Overlay of every channel's saturation curve, KPI normalized 0-100% per channel for shape comparison."""
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    channels = sorted(individual_df["Channel"].unique())
    for i, channel in enumerate(channels):
        channel_df = individual_df[individual_df["Channel"] == channel].sort_values(
            "Channel_Spend"
        )
        y = channel_df["Projected_Total_KPIs"]
        y_range = y.max() - y.min()
        normalized = ((y - y.min()) / y_range * 100) if y_range > 0 else y * 0

        fig.add_trace(
            go.Scatter(
                x=channel_df["Channel_Spend"],
                y=normalized,
                mode="lines",
                name=channel,
                line=dict(color=palette[i % len(palette)], width=2),
                hovertemplate=(
                    f"<b>{channel}</b><br>Investimento: R$ %{{x:.2s}}"
                    "<br>Saturação: %{y:.0f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Investimento Diário no Canal (R$)",
        yaxis_title="% da Saturação Máxima do Canal",
        xaxis=dict(tickformat=".2s"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_channel_saturation_comparison chart"
```

---

### Task 5: `build_events_overview`

**Files:**
- Modify: `scripts/dashboard_charts.py`
- Modify: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Add the failing test**

Update the import line:

```python
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
)
```

Append:

```python
class TestBuildEventsOverview(unittest.TestCase):

    def test_colors_by_direction_and_fades_discarded_events(self):
        events_df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-13"],
            "ad_product": ["AWIN, BING", "GOOGLE"],
            "percentage_change": [150.0, -80.0],
        })
        validated_keys = {("AWIN_BING", "2025-01-06")}

        fig = build_events_overview(events_df, validated_keys)

        self.assertEqual(list(fig.data[0].x), list(pd.to_datetime(events_df["date"])))
        self.assertEqual(list(fig.data[0].y), [150.0, -80.0])
        self.assertEqual(tuple(fig.data[0].marker.color), ("#2ca02c", "#d62728"))
        self.assertEqual(tuple(fig.data[0].marker.opacity), (1.0, 0.35))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_events_overview'`

- [ ] **Step 3: Add `build_events_overview` to `scripts/dashboard_charts.py`**

Append:

```python


def build_events_overview(events_df, validated_keys):
    """Bar chart of every detected event (spike/drop), faded if it never became a validated report."""
    events = events_df.copy()
    events["date"] = pd.to_datetime(events["date"])
    events["channel_folder"] = events["ad_product"].str.replace(", ", "_", regex=False)
    events["validated"] = events.apply(
        lambda row: (row["channel_folder"], row["date"].strftime("%Y-%m-%d")) in validated_keys,
        axis=1,
    )

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in events["percentage_change"]]
    opacities = [1.0 if v else 0.35 for v in events["validated"]]
    hover_labels = [
        f"{ch}<br>{'Validado (tem relatório)' if val else 'Descartado (sem relatório)'}"
        for ch, val in zip(events["ad_product"], events["validated"])
    ]

    fig = go.Figure(
        go.Bar(
            x=events["date"],
            y=events["percentage_change"],
            marker=dict(color=colors, opacity=opacities),
            text=hover_labels,
            hovertemplate=(
                "<b>Data:</b> %{x|%d/%m/%Y}<br><b>Variação:</b> %{y:.1f}%"
                "<br>%{text}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Data do Evento",
        yaxis_title="Variação de Investimento (%)",
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- uv run --with pytest pytest tests/test_dashboard_charts.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_charts.py tests/test_dashboard_charts.py
git commit -m "feat(dashboard): add build_events_overview chart"
```

---

### Task 6: Wire the 4 new charts into the Elasticidade tab (Tab 3)

**Files:**
- Modify: `scripts/streamlit_app.py:252-253` (import)
- Modify: `scripts/streamlit_app.py:1410-1414` (insert new charts between the main curve and the individual-channel-curves section)

- [ ] **Step 1: Add the import**

In `scripts/streamlit_app.py`, find:

```python
from db import init_db, create_user, verify_user, add_user_project, get_user_projects, verify_project_ownership, delete_user_project, rename_user_project, get_user_api_key, update_user_api_key, create_session, get_session
from streamlit_cookies_controller import CookieController
```

Replace with:

```python
from db import init_db, create_user, verify_user, add_user_project, get_user_projects, verify_project_ownership, delete_user_project, rename_user_project, get_user_api_key, update_user_api_key, create_session, get_session
from streamlit_cookies_controller import CookieController
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
)
```

- [ ] **Step 2: Insert the 4 new charts after the main saturation curve**

Find (existing code, unchanged so far):

```python
                st.plotly_chart(fig_curve, use_container_width=True)

                # --- NEW: Individual Curves Visualization ---
                st.markdown("---")
                st.markdown("### Curvas de Resposta Individuais por Canal")
```

Replace with:

```python
                st.plotly_chart(fig_curve, use_container_width=True)

                st.markdown("---")
                st.markdown("### Curva de Custo Marginal (iCPA)")
                st.markdown(
                    "Mostra quanto custa cada KPI adicional em cada nível de investimento. "
                    "Onde a curva cruza a linha de referência (seu Target CPA/iCPA definido na "
                    "barra lateral) é o ponto em que o investimento incremental deixa de compensar."
                )
                st.plotly_chart(
                    build_icpa_curve(
                        df_plot,
                        optimal_point,
                        saturation_point,
                        kpi_name=kpi_name,
                        target_cpa=target_cpa,
                        target_icpa=target_icpa,
                    ),
                    use_container_width=True,
                )

                if config.get("optimization_target") == "REVENUE":
                    st.markdown("---")
                    st.markdown("### Curva de Receita e ROI Incremental")
                    st.markdown(
                        "Eixo esquerdo: receita projetada (mensal) em cada nível de investimento. "
                        "Eixo direito: ROI incremental — quanto retorno cada Real adicional investido "
                        "está gerando."
                    )
                    st.plotly_chart(
                        build_revenue_roi_curve(df_plot, kpi_name=kpi_name),
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown("### Evolução do Mix de Canais por Orçamento")
                st.markdown(
                    "Mostra como a alocação recomendada entre canais (Modelo de Elasticidade) muda "
                    "à medida que o orçamento total escala — útil para saber quais canais absorvem "
                    "mais verba incremental conforme você investe mais."
                )
                st.plotly_chart(
                    build_channel_mix_evolution(
                        df_plot,
                        baseline_monthly_inv=baseline_monthly_inv,
                        optimal_monthly_inv=optimal_point["Monthly_Investment"],
                    ),
                    use_container_width=True,
                )

                ind_csv_path_overview = os.path.join(
                    output_dir, "individual_response_curves_data.csv"
                )
                if os.path.exists(ind_csv_path_overview):
                    st.markdown("---")
                    st.markdown("### Comparativo de Saturação entre Canais")
                    st.markdown(
                        "Sobrepõe a curva de saturação de todos os canais, normalizada para 0-100% "
                        "da própria faixa de cada canal — permite comparar a *velocidade* de saturação "
                        "entre canais mesmo com escalas de investimento muito diferentes. Clique na "
                        "legenda para isolar um canal."
                    )
                    st.plotly_chart(
                        build_channel_saturation_comparison(
                            pd.read_csv(ind_csv_path_overview)
                        ),
                        use_container_width=True,
                    )

                # --- NEW: Individual Curves Visualization ---
                st.markdown("---")
                st.markdown("### Curvas de Resposta Individuais por Canal")
```

- [ ] **Step 3: Manually verify in the browser**

Run: `mise exec -- streamlit run scripts/streamlit_app.py`

1. Log in and open a project that already has data in `outputs/` (e.g. `Meu_Projeto_Dynamic_dynamic`).
2. Go to the **Elasticidade** tab.
3. Confirm, in order below the existing "Curva de Saturação de Investimentos": a new "Curva de Custo Marginal (iCPA)" chart renders with a line and (if you enable the sidebar CPA/iCPA checkboxes) horizontal reference lines; a "Curva de Receita e ROI Incremental" section appears **only if** the project's `optimization_target` is `REVENUE` (the sample config uses `CONVERSIONS`, so confirm it's absent there, then temporarily flip `optimization_target` to `"REVENUE"` in that project's `config_dynamic.json` and reload to confirm it appears); an "Evolução do Mix de Canais por Orçamento" stacked area chart; a "Comparativo de Saturação entre Canais" chart with one line per channel, toggled by legend clicks.
4. Confirm the existing selector + static per-channel image below still works unchanged.

- [ ] **Step 4: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(dashboard): wire new elasticidade charts into streamlit tab"
```

---

### Task 7: Wire the events overview into the Impacto Causal tab (Tab 2)

**Files:**
- Modify: `scripts/streamlit_app.py:931-934`

- [ ] **Step 1: Insert the events overview chart**

Find (existing code, unchanged so far):

```python
        for r in md_reports:
            event_dirs.add(os.path.dirname(r))

        if event_dirs:
            report_options = {}
```

Replace with:

```python
        for r in md_reports:
            event_dirs.add(os.path.dirname(r))

        if event_dirs:
            validated_keys = set()
            for d in event_dirs:
                parts = d.split(os.sep)
                if len(parts) >= 2:
                    validated_keys.add((parts[-2], parts[-1]))

            detected_events_path = os.path.join(adv_dir, "detected_events.csv")
            if os.path.exists(detected_events_path):
                st.markdown("### Visão Geral dos Eventos Detectados")
                st.markdown(
                    "Todos os picos e quedas de investimento detectados ao longo do histórico. "
                    "Barras cheias tiveram significância estatística confirmada e têm um relatório "
                    "navegável abaixo; barras esmaecidas foram descartadas (não passaram nos "
                    "critérios de p-value/R² ou ficaram fora do limite de eventos analisados)."
                )
                st.plotly_chart(
                    build_events_overview(
                        pd.read_csv(detected_events_path), validated_keys
                    ),
                    use_container_width=True,
                )
                st.markdown("---")

            report_options = {}
```

- [ ] **Step 2: Manually verify in the browser**

Run: `mise exec -- streamlit run scripts/streamlit_app.py`

1. Go to the **Impacto Causal** tab for a project with at least one detected event (e.g. `Meu_Projeto_Dynamic_dynamic`, which has `outputs/user_1/Meu_Projeto_Dynamic_dynamic/detected_events.csv`).
2. Confirm the "Visão Geral dos Eventos Detectados" bar chart renders above the existing "Selecione o Evento:" dropdown, with green bars for increases and red for decreases, and confirm hovering shows the channel list and validated/discarded label.
3. Confirm the existing dropdown + report/image drill-down below still works unchanged.

- [ ] **Step 3: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(dashboard): wire events overview chart into impacto causal tab"
```

---

## Plan Self-Review Notes

- **Spec coverage:** all 4 Elasticidade charts (iCPA curve, Revenue/ROI, mix evolution, channel comparison) → Tasks 1-4 + Task 6. Events overview → Task 5 + Task 7. Explicit out-of-scope items (model params, filters, static PNG charts, pipeline changes) are untouched by every task above.
- **No placeholders:** every step has complete, runnable code — no "add error handling" stubs.
- **Type/name consistency checked:** `build_icpa_curve`, `build_revenue_roi_curve`, `build_channel_mix_evolution`, `build_channel_saturation_comparison`, `build_events_overview` are named identically across their definition (Tasks 1-5), their tests, and their call sites (Tasks 6-7).
