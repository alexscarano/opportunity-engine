# -*- coding: utf-8 -*-
"""
This module handles the generation of saturation curves at both the event-specific
and global brand levels.
"""

import os
import traceback
import pandas as pd
import analysis
import presentation
import data_preprocessor


def generate_event_saturation_curves(
    kpi_df,
    daily_investment_df,
    trends_df,
    config,
    product_group_for_report,
    event_output_dir,
):
    """
    Generates saturation curves for the specific product group(s) of a single event.
    This is called within the event loop and saves results to the event's directory.
    """
    print(
        f"   - Generating event-specific saturation curves for '{product_group_for_report}'..."
    )

    saturation_dir = os.path.join(event_output_dir, "saturation_curves")
    os.makedirs(saturation_dir, exist_ok=True)

    markdown_content = ""

    try:
        (
            full_response_curve_df,
            scenarios_df,
            baseline_point,
            max_efficiency_point,
            diminishing_return_point,
            saturation_point,
            strategic_limit_point,
            model_params,
            channel_proportions,
        ) = analysis.run_opportunity_projection(
            kpi_df, daily_investment_df, trends_df, product_group_for_report, config
        )

        if full_response_curve_df is None or full_response_curve_df.empty:
            raise ValueError("Failed to generate the main response curve.")

        is_combined = len(product_group_for_report.split(",")) > 1
        plot_name = (
            "combined_event_saturation_curve.png"
            if is_combined
            else f"{product_group_for_report.replace(' ', '_')}_saturation_curve.png"
        )

        plot_filename = os.path.join(saturation_dir, plot_name)
        presentation.save_opportunity_curve_plot(
            full_response_curve_df,
            baseline_point,
            max_efficiency_point,
            diminishing_return_point,
            saturation_point,
            plot_filename,
            kpi_name=config.get("performance_kpi_column", "Sessions"),
            strategic_limit_point=strategic_limit_point,
            config=config,
        )

        # --- Build the Detailed Investment Breakdown Table ---
        scenarios_to_process = [
            "Cenário Atual",
            "Máxima Eficiência",
            "Limite Estratégico",
        ]

        if channel_proportions:
            header = "| Canal | " + " | ".join(scenarios_to_process) + " |\n"
            separator = (
                "|:---| " + " | ".join([":---" for _ in scenarios_to_process]) + " |\n"
            )

            body = ""
            sorted_channels = sorted(channel_proportions.keys())

            for channel in sorted_channels:
                row = f"| **{channel}** |"
                proportion = channel_proportions.get(channel, 0)

                for scenario_name in scenarios_to_process:
                    scenario_row = scenarios_df[
                        scenarios_df["Scenario"] == scenario_name
                    ]
                    if not scenario_row.empty:
                        total_investment = scenario_row["Daily_Investment"].iloc[0] * 30
                        channel_investment = total_investment * proportion
                        row += f" {presentation.format_number(channel_investment, currency=True)} |"
                    else:
                        row += " N/A |"
                body += row + "\n"

            investment_table = f"### Detalhamento do Investimento por Canal\n\n{header}{separator}{body}"
        else:
            investment_table = "### Detalhamento do Investimento por Canal\n\nNão foi possível calcular a divisão de investimento por canal.\n"

        # --- Assemble Final Content ---
        final_markdown = f"# Análise da Curva de Saturação Mensal para o Evento\n\n## Cenários para: {product_group_for_report}\n\n"

        kpi_name = config.get("performance_kpi_column", "KPIs")
        final_markdown += f"| Cenário | Investimento Mensal | Projeção de {kpi_name} | Custo por {kpi_name} (CPA) | Investimento Incremental | {kpi_name} Incrementais | iCPA |\n"
        final_markdown += "|:---|:---|:---|:---|:---|:---|:---|\n"

        optimization_target = config.get("optimization_target", "REVENUE").upper()

        for _, row in scenarios_df.iterrows():
            title = row.get("Scenario", "Desconhecido")
            inv = row.get("Daily_Investment", 0) * 30
            kpi = row.get("Projected_Total_KPIs", 0) * 30
            inc_kpi = row.get("Incremental_KPI", 0) * 30

            # Use baseline data to calculate clean incrementals to avoid pandas noise
            inc_inv = inv - (baseline_point["Daily_Investment"] * 30)
            cpa = inv / kpi if kpi > 0 else 0
            icpa = inc_inv / inc_kpi if inc_kpi > 0 else 0

            # Force clean zeros for the baseline row
            if title == "Cenário Atual":
                inc_inv = 0.0
                inc_kpi = 0.0
                icpa = 0.0

            inv_str = presentation.format_number(inv, currency=True)
            kpi_str = presentation.format_number(kpi)
            cpa_str = presentation.format_number(cpa, currency=True)
            inc_inv_str = presentation.format_number(inc_inv, currency=True)
            inc_kpi_str = presentation.format_number(inc_kpi)
            icpa_str = presentation.format_number(icpa, currency=True)

            final_markdown += f"| **{title}** | {inv_str} | {kpi_str} | {cpa_str} | {inc_inv_str} | {inc_kpi_str} | {icpa_str} |\n"

        final_markdown += "\n\n"
        final_markdown += investment_table

        saturation_filepath = os.path.join(event_output_dir, "SATURATION_CURVE.md")
        with open(saturation_filepath, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        print(
            f"   - Successfully generated event saturation analysis file at: {saturation_filepath}"
        )

    except Exception as e:
        print(f"   - WARNING: Could not generate event saturation curve. Details: {e}")
        return pd.DataFrame(), pd.DataFrame(), None, None, None, None, None, None, None

    return (
        full_response_curve_df,
        scenarios_df,
        baseline_point,
        max_efficiency_point,
        diminishing_return_point,
        saturation_point,
        strategic_limit_point,
        model_params,
        channel_proportions,
    )


def run_global_saturation_analysis(config):
    """
    Generates a comprehensive saturation analysis for all individual channels and the combined total,
    independent of event detection. Saves results to a dedicated global directory.
    """
    print("\n" + "=" * 50 + "\nStarting Global Saturation Analysis...\n" + "=" * 50)

    try:
        kpi_df, daily_investment_df, trends_df, _ = (
            data_preprocessor.load_and_prepare_data(config)
        )

        advertiser_name = config.get("advertiser_name", "default_advertiser")
        output_dir = os.path.join(
            os.getcwd(),
            config["output_directory"],
            advertiser_name,
            "global_saturation_analysis",
        )
        os.makedirs(output_dir, exist_ok=True)

        all_scenarios = []

        investment_split_table = None
        optimized_split_table = None
        baseline_point_global = None
        channel_proportions_global = None

        all_channels = [
            ch for ch in daily_investment_df["Product Group"].unique() if ch != "OTHER"
        ]
        print(
            f"   - Found {len(all_channels)} unique channels to analyze: {', '.join(all_channels)}"
        )

        if len(all_channels) > 1:
            print("     - Analyzing combination of all channels...")
            try:
                (
                    full_response_curve_df,
                    scenarios_df,
                    baseline_point_global,
                    max_efficiency_point,
                    diminishing_return_point,
                    saturation_point,
                    strategic_limit_point,
                    _,
                    channel_proportions_global,
                ) = analysis.run_opportunity_projection(
                    kpi_df,
                    daily_investment_df,
                    trends_df,
                    ", ".join(all_channels),
                    config,
                )
                print(
                    f"   - DEBUG: full_response_curve_df empty: {full_response_curve_df.empty if full_response_curve_df is not None else True}"
                )
                print(
                    f"   - DEBUG: scenarios_df empty: {scenarios_df.empty if scenarios_df is not None else True}"
                )

                if (
                    full_response_curve_df is not None
                    and not full_response_curve_df.empty
                ):
                    plot_filename = os.path.join(
                        output_dir, "combined_all_channels_saturation_curve.png"
                    )
                    presentation.save_opportunity_curve_plot(
                        full_response_curve_df,
                        baseline_point_global,
                        max_efficiency_point,
                        diminishing_return_point,
                        saturation_point,
                        plot_filename,
                        kpi_name=config.get("performance_kpi_column", "Sessions"),
                        strategic_limit_point=strategic_limit_point,
                        config=config,
                    )

                    scenarios_df["Channel"] = "Total Combinado"
                    all_scenarios.append(scenarios_df)

                    # --- New Code: Prepare data for the investment split table ---
                    if channel_proportions_global:
                        split_data = []
                        for _, row in scenarios_df.iterrows():
                            scenario_investment = row["Daily_Investment"] * 30
                            split_row = {"Cenário": row["Scenario"]}
                            for (
                                channel,
                                proportion,
                            ) in channel_proportions_global.items():
                                split_row[f"Investimento {channel}"] = (
                                    scenario_investment * proportion
                                )
                            split_data.append(split_row)

                        split_df = pd.DataFrame(split_data)
                        investment_split_table = split_df.to_markdown(
                            index=False, floatfmt=",.2f"
                        )
                    # --- End New Code ---

                    # --- New Scenario: Optimal Historical Mix ---
                    optimal_historical_mix = analysis.find_optimal_historical_mix(
                        kpi_df, daily_investment_df
                    )
                    historical_optimal_split_table = None
                    if optimal_historical_mix:
                        split_data_optimal = []
                        scenarios_to_process = [
                            "Máxima Eficiência",
                            "Limite Estratégico",
                        ]
                        for _, row in scenarios_df[
                            scenarios_df["Scenario"].isin(scenarios_to_process)
                        ].iterrows():
                            scenario_investment = row["Daily_Investment"] * 30
                            split_row = {"Cenário": row["Scenario"]}
                            for channel, proportion in optimal_historical_mix.items():
                                # Ensure the channel exists in the proportions before allocating
                                if channel in channel_proportions_global:
                                    split_row[f"Investimento {channel}"] = (
                                        scenario_investment * proportion
                                    )
                            split_data_optimal.append(split_row)

                        if split_data_optimal:
                            optimal_split_df = pd.DataFrame(split_data_optimal)
                            # Fill any missing channel columns with 0 to avoid errors
                            for channel in channel_proportions_global.keys():
                                if (
                                    f"Investimento {channel}"
                                    not in optimal_split_df.columns
                                ):
                                    optimal_split_df[f"Investimento {channel}"] = 0
                            historical_optimal_split_table = (
                                optimal_split_df.to_markdown(
                                    index=False, floatfmt=",.2f"
                                )
                            )
                    # --- End New Scenario ---

            except Exception as e:
                print(
                    f"   - WARNING: Could not generate combined saturation curve for all channels. Details: {e}"
                )
                traceback.print_exc()

        if all_scenarios:
            full_scenarios_df = pd.concat(all_scenarios, ignore_index=True)
            markdown_content = (
                f"# Análise Global da Curva de Saturação para {advertiser_name}\n\n"
            )

            channel_order = sorted(
                [
                    c
                    for c in full_scenarios_df["Channel"].unique()
                    if c != "Total Combinado"
                ]
            )
            if "Total Combinado" in full_scenarios_df["Channel"].unique():
                channel_order.append("Total Combinado")

            for channel_name in channel_order:
                group = full_scenarios_df[full_scenarios_df["Channel"] == channel_name]
                markdown_content += f"## Cenários para: {channel_name}\n\n"
                display_df = group[
                    [
                        "Scenario",
                        "Daily_Investment",
                        "Projected_Total_KPIs",
                        "Incremental_Investment",
                        "Incremental_Revenue",
                        "Incremental_ROI",
                    ]
                ].copy()

                display_df["Daily_Investment"] *= 30
                display_df["Projected_Total_KPIs"] *= 30
                display_df["Incremental_Investment"] *= 30
                display_df["Incremental_Revenue"] *= 30

                avg_ticket = config.get("average_ticket", 0)
                if avg_ticket > 0:
                    display_df["Receita Total Projetada"] = (
                        display_df["Projected_Total_KPIs"] * avg_ticket
                    )

                display_df.rename(
                    columns={
                        "Scenario": "Cenário",
                        "Daily_Investment": "Investimento Mensal",
                        "Projected_Total_KPIs": f"KPI Projetado Mensal ({config.get('performance_kpi_column', 'Sessions')})",
                        "Incremental_Investment": "Investimento Incremental",
                        "Incremental_Revenue": "Receita Incremental",
                        "Incremental_ROI": "ROI Incremental",
                    },
                    inplace=True,
                )

                markdown_content += display_df.to_markdown(index=False, floatfmt=",.2f")
                markdown_content += "\n\n"

                if channel_name == "Total Combinado":
                    if investment_split_table:
                        markdown_content += "### Divisão de Investimento Mensal por Canal (Baseado no Histórico)\n\n"
                        markdown_content += investment_split_table
                        markdown_content += "\n\n"
                    if historical_optimal_split_table:
                        markdown_content += "### Divisão de Investimento Mensal por Canal (Baseado no Mix de Máxima Eficiência Histórica)\n\n"
                        markdown_content += historical_optimal_split_table
                        markdown_content += "\n\n"

                markdown_content += """

## Como a Distribuição de Investimento é Calculada

Para cada cenário, a distribuição do investimento entre os canais é feita da seguinte forma:

### 1. Cenário Atual
- **Orçamento:** Utiliza o nível de investimento total atual.
- **Distribuição:** O orçamento é dividido entre os canais com base na **média histórica geral** de investimento. Analisamos todo o histórico de seus dados de investimento e calculamos a porcentagem que foi para cada canal (ex: 40% para Search, 30% para PMAX, etc.).

### 2. Cenário Otimizado (Máxima Eficiência)
- **Orçamento:** Utiliza o nível de investimento de "Máxima Eficiência", que é o ponto na curva de resposta que oferece o melhor retorno possível sobre o investimento.
- **Distribuição:** A divisão é **orientada por dados**, baseada nos seus períodos históricos de maior sucesso. Para encontrar o "Mix Ótimo", seguimos estes passos:
    1. Calculamos uma **pontuação de eficiência semanal** (KPIs divididos pelo Investimento).
    2. Para considerar o atraso do marketing e encontrar períodos de sucesso *sustentado*, usamos uma **média móvel de 4 semanas** dessa pontuação de eficiência.
    3. Identificamos as **10 melhores semanas** que tiveram a maior média de eficiência.
    4. Finalmente, calculamos a média da combinação de investimento *apenas desses períodos de melhor desempenho*. Isso se torna o "Mix Ótimo".

### 3. Cenário Estratégico (Limite Estratégico)
- **Orçamento:** Utiliza o nível de investimento de "Limite Estratégico", um orçamento mais alto projetado para o crescimento máximo, mesmo que isso signifique um ROI marginalmente menor.
- **Distribuição:** Utiliza o **mesmo "Mix Ótimo"** que foi calculado para o Cenário Otimizado.

"""

            # --- Generate Donut Chart Visualization ---
            donut_scenarios = []
            # 1. Current Scenario (using historical average mix)
            if baseline_point_global and channel_proportions_global:
                current_total_investment = (
                    baseline_point_global.get("Daily_Investment", 0) * 30
                )
                donut_scenarios.append(
                    {
                        "title": "Cenário Atual",
                        "data": {
                            ch: current_total_investment * p
                            for ch, p in channel_proportions_global.items()
                        },
                    }
                )

            # 2. Max Efficiency Scenario (using optimal historical mix)
            max_eff_row = scenarios_df[scenarios_df["Scenario"] == "Máxima Eficiência"]
            if not max_eff_row.empty and optimal_historical_mix:
                max_eff_investment = max_eff_row.iloc[0]["Daily_Investment"] * 30
                donut_scenarios.append(
                    {
                        "title": "Cenário Otimizado",
                        "data": {
                            ch: max_eff_investment * p
                            for ch, p in optimal_historical_mix.items()
                            if ch in channel_proportions_global
                        },
                    }
                )

            # 3. Strategic Limit Scenario (using optimal historical mix)
            strategic_limit_row = scenarios_df[
                scenarios_df["Scenario"] == "Limite Estratégico"
            ]
            if not strategic_limit_row.empty and optimal_historical_mix:
                strategic_limit_investment = (
                    strategic_limit_row.iloc[0]["Daily_Investment"] * 30
                )
                donut_scenarios.append(
                    {
                        "title": "Cenário Estratégico",
                        "data": {
                            ch: strategic_limit_investment * p
                            for ch, p in optimal_historical_mix.items()
                            if ch in channel_proportions_global
                        },
                    }
                )

            if donut_scenarios:
                donut_filename = os.path.join(
                    output_dir, "investment_distribution_donuts.png"
                )
                presentation.save_investment_distribution_donuts(
                    donut_scenarios, donut_filename
                )
                markdown_content += (
                    "### Visualização da Distribuição de Investimento\n\n"
                )
                markdown_content += f"![Distribuição de Investimento](./investment_distribution_donuts.png)\n\n"

            saturation_filepath = os.path.join(output_dir, "SATURATION_CURVE.md")
            with open(saturation_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(
                f"   - Successfully generated global saturation analysis file at: {saturation_filepath}"
            )

        print("=" * 50 + "\nGlobal Saturation Analysis Complete.\n" + "=" * 50)

    except Exception as e:
        print(f"A critical error occurred during the global saturation analysis: {e}")
        traceback.print_exc()
