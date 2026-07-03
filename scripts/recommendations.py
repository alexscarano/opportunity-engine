# -*- coding: utf-8 -*-
"""
This module generates recommendation outputs, including budget scenarios
and strategic markdown files.
"""

import os
import traceback
from presentation import format_number
import pandas as pd
import presentation

# --- Budget Scenario Generation ---


def generate_elasticity_budget_scenarios(contribution_pct, total_budget):
    """
    Calculates budget split based on Elasticity contribution percentages.
    Ensures the total adds up perfectly.
    """
    if not contribution_pct or sum(contribution_pct.values()) == 0:
        return {}

    total_pct = sum(contribution_pct.values())
    normalized_pct = {k: (v / total_pct) for k, v in contribution_pct.items()}

    budget_split = {
        channel: total_budget * pct for channel, pct in normalized_pct.items()
    }
    return budget_split


def generate_historical_split_scenarios(investment_df, total_budget):
    """
    Calculates budget split based on the top-performing historical weeks.
    """
    # Resample to weekly frequency to calculate weekly investment
    weekly_investment = (
        investment_df.set_index("Date").resample("W-Mon").sum(numeric_only=True)
    )

    # Use total weekly investment as a proxy for efficiency for finding top weeks
    weekly_investment["efficiency"] = weekly_investment.sum(axis=1)

    # Use a 4-week rolling average to find sustained success
    weekly_investment["rolling_efficiency"] = (
        weekly_investment["efficiency"].rolling(window=4).mean()
    )

    # Identify the top 10 best-performing weeks
    top_weeks = weekly_investment.nlargest(10, "rolling_efficiency")

    # Calculate the average investment mix from those top weeks
    optimal_mix_proportions = top_weeks.drop(
        columns=["efficiency", "rolling_efficiency"]
    ).mean()

    # Normalize the proportions
    total_investment = optimal_mix_proportions.sum()
    if total_investment == 0:
        return {}
    normalized_proportions = optimal_mix_proportions / total_investment

    # Allocate the total budget based on this optimal historical mix
    budget_split = {
        channel: total_budget * prop for channel, prop in normalized_proportions.items()
    }
    return budget_split

def generate_basic_recommendations_file(results_data, config, output_dir):
    """
    Generates a basic recommendations markdown file when elasticity is not available.
    """
    try:
        avg_ticket = config.get("average_ticket", 0)
        conversion_rate = config.get("conversion_rate_from_kpi_to_bo", 1)
        optimization_target = config.get("optimization_target", "REVENUE").upper()

        causal_incremental_kpi = results_data.get("absolute_lift", 0)
        causal_incremental_orders = causal_incremental_kpi * conversion_rate
        causal_incremental_revenue = causal_incremental_orders * avg_ticket

        inv_pre = results_data.get("total_investment_pre_period", 0)
        inv_post = results_data.get("total_investment_post_period", 0)
        investment_change = inv_post - inv_pre

        if optimization_target == "REVENUE":
            gain_metric = "em receita incremental"
            formatted_gain = format_number(causal_incremental_revenue, currency=True)
            roas = (
                causal_incremental_revenue / investment_change
                if investment_change > 0
                else 0
            )
            efficiency_text = f"O ROAS (Retorno sobre Investimento em Publicidade) real foi de **{roas:.2f}x**."
        else:
            gain_metric = "em conversões incrementais"
            formatted_gain = format_number(causal_incremental_orders)
            cpa = (
                investment_change / causal_incremental_orders
                if causal_incremental_orders > 0
                else 0
            )
            efficiency_text = f"O custo por aquisição incremental (iCPA) foi de **{format_number(cpa, currency=True)}**."

        recommendation_text = (
            f"A análise de Impacto Causal indica que a mudança de investimento "
            f"gerou um impacto de **{formatted_gain}** {gain_metric}. "
            f"{efficiency_text}\n\n"
            f"**Estatísticas do Modelo:**\n"
            f"- Aumento de Investimento: **{results_data.get('investment_change_pct', 0):.2f}%**\n"
            f"- Confiança Estatística: **{(1 - results_data.get('p_value', 1)) * 100:.2f}%**\n"
            f"- Precisão do Modelo (R²): **{results_data.get('model_r_squared', 0):.4f}**\n"
        )

        content = f"""# Resumo de Impacto Causal

## Análise da Oportunidade
{recommendation_text}
"""
        output_path = os.path.join(output_dir, "RECOMMENDATIONS.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(
            f"   - Successfully generated basic recommendations file at: {output_path}"
        )

    except Exception as e:
        print(
            f"   - ERROR: Could not generate basic recommendations file. Details: {e}"
        )
