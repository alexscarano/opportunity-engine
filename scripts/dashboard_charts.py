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
