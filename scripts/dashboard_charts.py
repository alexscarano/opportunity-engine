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
