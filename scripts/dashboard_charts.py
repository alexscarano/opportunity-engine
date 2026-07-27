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


def find_saturation_point(df, optimal_point, min_investment=None):
    """
    Finds the point on the aggregate response curve (sorted by ascending
    investment) where marginal KPI gain has sustainedly collapsed to under 10%
    of the peak marginal gain observed at or above `min_investment`.

    `min_investment` should be the current baseline daily spend when known.
    A "saturation ceiling" is meant to describe how far *scaling up* keeps
    paying off -- without this floor, any normal concave response curve (the
    expected shape: steepest at the very first dollar, decaying from there)
    reports its own first simulated step as the peak and the ceiling lands
    a few steps later, below what's already being spent today. That is a
    correct answer to "where does the curve's slope collapse from *zero*"
    but not to "how far above today can we push" -- restricting the search to
    the region actually being scaled into is what makes it the latter.

    Uses that region's own peak as the 100% reference (not literally its
    first step, which can still be an extrapolation artifact -- see
    predict_clipped_kpi in elasticity_analysis.py) and returns the point
    right after the *last* index still above threshold, so a transient dip
    that recovers afterward doesn't get mistaken for the real, sustained
    saturation ceiling.

    Falls back to `optimal_point` when the curve never rises (or is too short
    to compute a derivative from) in the searched region.
    """
    search_df = df
    if min_investment is not None:
        search_df = df[df["Daily_Investment"] >= min_investment]

    incremental_kpis = search_df["Projected_Total_KPIs"].diff().fillna(0).values
    investment_steps = search_df["Daily_Investment"].diff().fillna(1).values
    first_derivative = incremental_kpis / investment_steps

    if len(first_derivative) <= 1:
        return optimal_point

    peak_marginal_gain = first_derivative.max()
    if peak_marginal_gain <= 0:
        return optimal_point

    saturation_threshold = peak_marginal_gain * 0.1
    above_threshold = np.where(first_derivative > saturation_threshold)[0]
    if len(above_threshold) == 0:
        return optimal_point

    saturation_idx = above_threshold[-1] + 1
    if saturation_idx >= len(search_df):
        return optimal_point

    return search_df.iloc[saturation_idx]


def compute_incremental_cpa(investimento_incremental, kpi_incremental):
    """
    Ratio of incremental investment to incremental KPI ("iCPA"), or NaN when
    the scenario didn't produce a positive KPI gain.

    A non-positive incremental KPI makes the ratio not meaningful (it is not
    a real, free-or-cheap cost -- it's a scenario that performs worse than the
    baseline), so it must not be silently rendered as a literal 0.
    """
    return np.where(kpi_incremental > 0, investimento_incremental / kpi_incremental, np.nan)


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
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_revenue_roi_curve(df_plot, kpi_name="kpi", monthly_factor=30):
    """Dual-axis chart: Projected Revenue (left) and Incremental ROI (right) vs Monthly_Investment.

    `monthly_factor` extrapolates the per-period Projected_Revenue column to a
    monthly figure (30/period_days periods per month -- defaults to the old
    flat 30, i.e. daily cadence, when the caller doesn't pass it)."""
    finite_roi = df_plot[np.isfinite(df_plot["Incremental_ROI"])]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["Monthly_Investment"],
            y=df_plot["Projected_Revenue"] * monthly_factor,
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
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_channel_mix_evolution(df_plot, baseline_monthly_inv=None, optimal_monthly_inv=None):
    """100%-stacked area of each channel's Strategic-mix share vs Monthly_Investment (log scale).

    Keeps only the top channels (by average share) distinct and folds the rest into
    "Outros" — past ~7-8 series a stacked chart stops being legible.
    """
    top_n = 6
    strategic_cols = [
        c for c in df_plot.columns if c.startswith("Spend_") and c.endswith("_Strategic")
    ]
    channel_names = [c.replace("Spend_", "").replace("_Strategic", "") for c in strategic_cols]

    # Log x-axis can't plot Investment=0, and the saturation curve is flat there anyway.
    plot_df = df_plot[df_plot["Monthly_Investment"] > 0]

    totals = plot_df[strategic_cols].sum(axis=1)
    shares = plot_df[strategic_cols].div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
    shares.columns = channel_names

    order = shares.mean().sort_values(ascending=False).index
    top_channels = list(order[:top_n])
    other_channels = list(order[top_n:])

    plot_shares = shares[top_channels].copy()
    if other_channels:
        plot_shares["Outros"] = shares[other_channels].sum(axis=1)
    channel_order = top_channels + (["Outros"] if other_channels else [])

    palette = px.colors.qualitative.Plotly

    fig = go.Figure()
    for i, channel in enumerate(channel_order):
        color = "gray" if channel == "Outros" else palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=plot_df["Monthly_Investment"],
                y=plot_shares[channel],
                mode="lines",
                name=channel,
                stackgroup="one",
                line=dict(width=0.5, color=color),
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
        xaxis=dict(type="log", tickformat=".2s"),
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


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
            textposition="none",
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
        height=450,
    )
    return fig


def build_accuracy_chart(accuracy_df, kpi_name="kpi"):
    """Real vs Predicted (in-sample) KPI line chart, with MAE shown as an annotation."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["Date"],
            y=accuracy_df["kpi"],
            mode="lines",
            name=f"{kpi_name} Real",
            line=dict(color="#2ca02c", width=2),
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
        font=dict(color="black"),
        opacity=0.8,
        align="left",
    )

    fig.update_layout(
        title="Acurácia do Modelo: Real vs. Previsto (Período Pré-Evento)",
        yaxis_title=kpi_name,
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_causal_line_chart(line_df, kpi_name="kpi"):
    """Dual-axis chart: Real vs Forecasted KPI (left axis) with Investment overlay (right axis, bars)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_df["Date"],
            y=line_df["Actual_KPI"],
            mode="lines",
            name=f"{kpi_name} Real",
            line=dict(color="#2ca02c", width=2),
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
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


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


def build_sessions_bar_chart(sessions_bar_df, kpi_name="kpi"):
    """Forecasted vs actual KPI totals, 2-bar comparison."""
    labels = {
        "Forecasted": f"{kpi_name} Previsto",
        "Actual": f"{kpi_name} Real",
    }
    x_labels = [labels.get(idx, idx) for idx in sessions_bar_df.index]
    colors = ["red" if idx == "Forecasted" else "#2ca02c" for idx in sessions_bar_df.index]

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


def build_response_curve_individual(channel_df, channel_name):
    """Spend vs Projected KPI for a single channel, with historical avg and recommended spend markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=channel_df["Channel_Spend"],
            y=channel_df["Projected_Total_KPIs"],
            mode="lines",
            name=channel_name,
            line=dict(color="blue", width=2),
        )
    )

    hist_spend = channel_df["Historical_Avg"].iloc[0]
    fig.add_vline(
        x=hist_spend,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Média Histórica (R$ {hist_spend:,.2f})",
        annotation_position="top",
    )

    rec_spend = channel_df["Recommended"].iloc[0]
    if pd.notna(rec_spend):
        fig.add_vline(
            x=rec_spend,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Recomendado (R$ {rec_spend:,.2f})",
            annotation_position="bottom",
        )

    fig.update_layout(
        title=f"Curva de Resposta: {channel_name}",
        xaxis_title="Investimento Diário",
        yaxis_title="Total Projetado de KPIs Diários",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
