# -*- coding: utf-8 -*-
"""
Main entrypoint for the Automated Total Opportunity Case Study Generator.

This script orchestrates the entire workflow:
1.  Loads configuration from a JSON file.
2.  Authenticates with Google services.
3.  Loads and validates input data.
4.  Runs the analysis to find significant events.
5.  Generates a Google Slides presentation for each valid event.
"""

import argparse
import json
import re
import traceback
import pandas as pd
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv

import os

import analysis
import google_api
import recommendations
import saturation_curve
import elasticity_analysis as mmm_analysis

from gemini_report import generate_html_report, generate_global_gemini_report
from presentation import (
    save_accuracy_plot,
    save_line_chart_plot,
    save_investment_bar_plot,
    save_sessions_bar_plot,
    save_opportunity_curve_plot,
    create_comparative_saturation_md,
    save_investment_distribution_donuts,
)


def _create_presentation_dataframe(causal_results, config, post_period):
    """Creates a DataFrame with all data points for the presentation CSV."""
    presentation_data = {}
    conversion_rate = config.get("conversion_rate_from_kpi_to_bo", 0)
    avg_ticket = config.get("average_ticket", 0)

    # Assumptions
    presentation_data["p_value_limiar"] = config["p_value_threshold"]
    presentation_data["taxa_de_conversao_kpi_para_pedidos"] = conversion_rate
    presentation_data["ticket_medio"] = avg_ticket
    presentation_data["precisao_preditiva_mape_pct"] = (
        causal_results.get("mape", 0) * 100
    )
    presentation_data["confianca_estatistica_pct"] = (
        1 - causal_results.get("p_value", 1)
    ) * 100
    presentation_data["estatistica_p_value"] = causal_results.get("p_value", 1)
    presentation_data["estatistica_r_squared"] = causal_results.get(
        "model_r_squared", 0
    )
    presentation_data["estatistica_mape"] = causal_results.get("mape", 0)
    presentation_data["estatistica_mae"] = causal_results.get("mae", 0)

    # Causal Impact Results
    causal_incremental_kpi = causal_results.get("absolute_lift", 0)
    causal_incremental_orders = causal_incremental_kpi * conversion_rate
    causal_incremental_revenue = causal_incremental_orders * avg_ticket

    presentation_data["causal_periodo_inicio"] = post_period[0]
    presentation_data["causal_periodo_fim"] = post_period[1]
    presentation_data["causal_aumento_investimento_pct"] = causal_results.get(
        "investment_change_pct", 0
    )
    presentation_data["causal_kpis_incrementais"] = causal_incremental_kpi
    presentation_data["causal_pedidos_incrementais"] = causal_incremental_orders
    presentation_data["causal_receita_incremental"] = causal_incremental_revenue
    presentation_data["causal_investimento_pre_evento"] = causal_results.get(
        "total_investment_pre_period", 0
    )
    presentation_data["causal_investimento_durante_evento"] = causal_results.get(
        "total_investment_post_period", 0
    )

    return pd.DataFrame(list(presentation_data.items()), columns=["Metrica", "Valor"])


import data_preprocessor


def main(config, args):
    """Main execution block for the script."""

    load_dotenv()
    model_name = config.get("gemini_model")
    gemini_client = google_api.authenticate_gemini(model_name=model_name)

    try:
        kpi_df, daily_investment_df, trends_df, correlation_matrix = (
            data_preprocessor.load_and_prepare_data(config)
        )

        # --- New Code: Load performance data for additional covariates ---
        performance_df = pd.read_csv(
            config["performance_file_path"], on_bad_lines="skip"
        )
        perf_map = config.get("column_mapping", {}).get("performance_file", {})
        date_col = perf_map.get("date_col", "date")
        performance_df.rename(columns={date_col: "Date"}, inplace=True)
        # --- End New Code ---

        kpi_col = config.get("performance_kpi_column", "Sessions")

        market_analysis_df = pd.merge(
            kpi_df.rename(columns={"Sessions": config["advertiser_name"]}),
            trends_df,
            on="Date",
            how="left",
        ).fillna(0)

        increase_ratio = 1 + (config["increase_threshold_percent"] / 100)
        decrease_ratio = 1 - (config["decrease_threshold_percent"] / 100)

        base_output_dir = os.path.join(os.getcwd(), config["output_directory"])
        advertiser_name = config.get("advertiser_name", "default_advertiser")
        brand_output_dir = os.path.join(base_output_dir, advertiser_name)

        event_map_df, _, _ = analysis.find_events(
            daily_investment_df,
            config["advertiser_name"],
            increase_ratio,
            decrease_ratio,
            config["post_event_days"],
            output_dir=brand_output_dir,
        )

        if event_map_df is None or event_map_df.empty:
            print("\nAnalysis complete: No significant events were detected.")

        else:
            product_filter = config.get("product_group_filter")
            if product_filter:
                filtered_events_df = event_map_df[
                    event_map_df["ad_product"].isin(product_filter)
                ].copy()
            else:
                filtered_events_df = event_map_df.copy()

            if filtered_events_df.empty:
                print(
                    f"\nAnalysis complete: No events found for the specified products."
                )

            else:
                min_historical_date = kpi_df["Date"].min().strftime("%Y-%m-%d")
                candidate_events_df = pd.DataFrame(
                    [
                        {
                            "event_id": f"{config['advertiser_name']}_{row['ad_product'].replace(' ', '_')}_{pd.to_datetime(row['date']).date()}",
                            "start_date": min_historical_date,
                            "intervention_date": pd.to_datetime(row["date"]).strftime(
                                "%Y-%m-%d"
                            ),
                            "end_date": (
                                pd.to_datetime(row["date"])
                                + timedelta(days=config["post_event_days"])
                            ).strftime("%Y-%m-%d"),
                            "product_group": row["ad_product"],
                        }
                        for _, row in filtered_events_df.iterrows()
                    ]
                )

                if args.min_intervention_date:
                    candidate_events_df = candidate_events_df[
                        pd.to_datetime(candidate_events_df["intervention_date"])
                        >= pd.to_datetime(args.min_intervention_date)
                    ]

                if candidate_events_df.empty:
                    print(
                        f"\nAnalysis complete: No events found after the specified min_intervention_date."
                    )

                else:
                    print(
                        f"\n"
                        + "=" * 50
                        + f"\nAnalyzing {len(candidate_events_df)} candidates...\n"
                        + "=" * 50
                    )
                    analyzed_events = []

                    # Pre-train the response model on the entire dataset for all event product groups
                    all_event_product_groups = ", ".join(
                        filtered_events_df["ad_product"].unique()
                    )
                    _, _, _, _, _, _, _, projection_model_params, _ = (
                        analysis.run_opportunity_projection(
                            kpi_df,
                            daily_investment_df,
                            trends_df,
                            all_event_product_groups,
                            config,
                        )
                    )

                    for index, event in candidate_events_df.iterrows():
                        pre_period = [
                            event["start_date"],
                            (
                                pd.to_datetime(event["intervention_date"])
                                - timedelta(days=1)
                            ).strftime("%Y-%m-%d"),
                        ]
                        post_period = [event["intervention_date"], event["end_date"]]

                        print(
                            f"\n"
                            + "-" * 50
                            + f"\n▶ Analyzing Event: {event['product_group']} on {event['intervention_date']}"
                        )

                        (
                            results_data,
                            line_df,
                            inv_bar_df,
                            sessions_bar_df,
                            accuracy_df,
                        ) = analysis.run_causal_impact_analysis(
                            kpi_df,
                            daily_investment_df,
                            trends_df,
                            performance_df,
                            pre_period,
                            post_period,
                            event["event_id"],
                            event["product_group"],
                            projection_model_params,
                            config,
                        )

                        if not results_data:
                            print(
                                "   - FAILED: Causal impact analysis could not be completed."
                            )
                            continue

                        # --- Optimization: R-squared threshold from config ---
                        r_squared_threshold = config.get("r_squared_threshold", 0.6)

                        is_significant = (
                            results_data["p_value"] < config["p_value_threshold"]
                        )
                        is_logical = (
                            results_data["investment_change_pct"] > 0
                            and results_data["absolute_lift"] > 0
                        ) or (
                            results_data["investment_change_pct"] < 0
                            and results_data["absolute_lift"] < 0
                        )
                        has_good_fit = (
                            results_data.get("model_r_squared", 0)
                            >= r_squared_threshold
                        )

                        if is_significant and is_logical and has_good_fit:
                            print(
                                f"   - PASSED: Event is statistically significant, logical, and has a good model fit (R² >= {r_squared_threshold})."
                            )
                            analyzed_events.append(
                                {
                                    "event": event,
                                    "results_data": results_data,
                                    "line_df": line_df,
                                    "inv_bar_df": inv_bar_df,
                                    "sessions_bar_df": sessions_bar_df,
                                    "accuracy_df": accuracy_df,
                                }
                            )
                        else:
                            print(
                                "   - SKIPPED: Event did not meet validation criteria."
                            )
                            if not is_significant:
                                print(
                                    f"     - Reason: p-value ({results_data.get('p_value', 999):.4f}) is not below threshold ({config['p_value_threshold']})."
                                )
                            if not is_logical:
                                print(
                                    f"     - Reason: Investment change ({results_data.get('investment_change_pct', 0):.2f}%) and lift direction ({results_data.get('absolute_lift', 0):.2f}) are not logical."
                                )
                            if not has_good_fit:
                                print(
                                    f"     - Reason: Model R-squared ({results_data.get('model_r_squared', 0):.4f}) is below the {r_squared_threshold} threshold."
                                )

                    if not analyzed_events:
                        print(
                            "\nAnalysis complete: No valid, impactful events were found."
                        )

                    else:
                        analyzed_events.sort(
                            key=lambda x: abs(x["results_data"]["absolute_lift"]),
                            reverse=True,
                        )
                        top_events_to_report = analyzed_events[
                            : config.get("max_events_to_analyze", 5)
                        ]

                        print(
                            f"\n"
                            + "=" * 50
                            + f"\nTop {len(top_events_to_report)} Events Selected. Generating Reports...\n"
                            + "=" * 50
                        )

                        successful_events = []
                        base_output_dir = os.path.join(
                            os.getcwd(), config["output_directory"]
                        )
                        advertiser_name = config.get(
                            "advertiser_name", "default_advertiser"
                        )
                        csv_output_filename = os.path.join(
                            base_output_dir, f"{advertiser_name}_analysis_results.csv"
                        )

                        for analyzed_event in top_events_to_report:
                            event = analyzed_event["event"]
                            results_data = analyzed_event["results_data"]
                            product_group_for_report = event["product_group"]

                            print(
                                f"\n"
                                + "-" * 50
                                + f"\nGenerating Report for Event: {product_group_for_report} on {event['intervention_date']}"
                            )

                            event_output_dir = os.path.join(
                                base_output_dir,
                                config["advertiser_name"],
                                product_group_for_report.replace(", ", "_"),
                                pd.to_datetime(event["intervention_date"]).strftime(
                                    "%Y-%m-%d"
                                ),
                            )
                            os.makedirs(event_output_dir, exist_ok=True)

                            # Filter investment data to only the product group(s) being analyzed
                            event_channels = [
                                ch.strip() for ch in product_group_for_report.split(",")
                            ]
                            filtered_investment_df = daily_investment_df[
                                daily_investment_df["Product Group"].isin(
                                    event_channels
                                )
                            ].copy()

                            try:
                                safe_pg_name = re.sub(
                                    r"[^\w-]", "_", product_group_for_report
                                )
                                file_base_name = f"{config['advertiser_name']}_{safe_pg_name}_{event['intervention_date']}"

                                image_paths = {}
                                image_paths["accuracy"] = os.path.join(
                                    event_output_dir,
                                    f"accuracy_plot_{file_base_name}.png",
                                )
                                save_accuracy_plot(
                                    results_data,
                                    analyzed_event["accuracy_df"],
                                    image_paths["accuracy"],
                                    kpi_name=kpi_col,
                                )

                                image_paths["line"] = os.path.join(
                                    event_output_dir, f"line_chart_{file_base_name}.png"
                                )
                                save_line_chart_plot(
                                    analyzed_event["line_df"],
                                    image_paths["line"],
                                    kpi_name=kpi_col,
                                )

                                image_paths["investment"] = os.path.join(
                                    event_output_dir,
                                    f"investment_chart_{file_base_name}.png",
                                )
                                save_investment_bar_plot(
                                    analyzed_event["inv_bar_df"],
                                    image_paths["investment"],
                                )

                                image_paths["sessions"] = os.path.join(
                                    event_output_dir,
                                    f"sessions_chart_{file_base_name}.png",
                                )
                                save_sessions_bar_plot(
                                    analyzed_event["sessions_bar_df"],
                                    image_paths["sessions"],
                                    kpi_name=kpi_col,
                                )

                                # Generate and save the comprehensive presentation data CSV for this event
                                presentation_df = _create_presentation_dataframe(
                                    results_data, config, post_period
                                )
                                csv_filename = os.path.join(
                                    event_output_dir,
                                    f"{file_base_name}_presentation_data.csv",
                                )
                                presentation_df.to_csv(
                                    csv_filename, index=False, float_format="%.2f"
                                )
                                print(
                                    f"   SUCCESS! Comprehensive data saved to: {csv_filename}"
                                )

                                html_report_filename = os.path.join(
                                    event_output_dir,
                                    f"gemini_report_{file_base_name}.html",
                                )

                                print(
                                    f"   - Generating Strategic Narrative with Gemini..."
                                )
                                try:
                                    generate_html_report(
                                        gemini_client,
                                        results_data,
                                        config,
                                        image_paths,
                                        html_report_filename,
                                        market_analysis_df,
                                        analyzed_event["line_df"],
                                        csv_output_filename=csv_filename,
                                        correlation_matrix=correlation_matrix,
                                    )
                                    successful_events.append(event_output_dir)
                                    print(
                                        f"   SUCCESS! View the Gemini HTML report here: {html_report_filename}"
                                    )
                                except Exception as e:
                                    print(
                                        f"   - ERROR: Could not generate or parse the narrative from Gemini. Details: {e}"
                                    )

                            except Exception as e:
                                print(f"Report generation failed for this event: {e}")
                                traceback.print_exc()
                                continue

                        if successful_events:
                            print(
                                "\n\n" + "=" * 50 + "\nAll tasks complete.\n" + "=" * 50
                            )
                            for path in successful_events:
                                print(f"   - {path}")
                        else:
                            print(
                                "\n\nAnalysis complete: No events met all criteria for reporting."
                            )

        # --- New Global Analysis Workflow using Elasticity Model ---
        print("\n" + "=" * 50 + "\nStarting Global Elasticity Analysis...\n" + "=" * 50)

        mmm_results = mmm_analysis.run_mmm_engine(config)

        if mmm_results:
            investment_pivot_df = (
                daily_investment_df.pivot_table(
                    index="Date", columns="Product Group", values="investment"
                )
                .fillna(0)
                .reset_index()
            )
            global_output_dir = os.path.join(
                base_output_dir, config["advertiser_name"], "global_saturation_analysis"
            )
            os.makedirs(global_output_dir, exist_ok=True)

            # --- Prepare Data for Donut Charts and Tables ---
            total_investment_per_channel = investment_pivot_df.drop(
                columns="Date"
            ).sum()
            current_budget_split = (
                total_investment_per_channel / total_investment_per_channel.sum()
            ).to_dict()

            optimized_budget_split = analysis.find_optimal_historical_mix(
                kpi_df, daily_investment_df
            )
            if not optimized_budget_split:
                optimized_budget_split = {}

            # --- Generate and Save Aggregated Response Curve ---
            (
                response_curve_df,
                baseline_point,
                max_efficiency_point,
                strategic_limit_point,
                diminishing_return_point,
                saturation_point,
                strategic_reallocation_point,
            ) = mmm_analysis.generate_aggregated_response_curve(
                mmm_results,
                config,
                optimized_mix=optimized_budget_split,
                output_dir=global_output_dir,
            )

            # --- Generate and Save Individual Response Curves ---
            mmm_analysis.generate_individual_response_curves(
                mmm_results,
                config,
                output_dir=global_output_dir,
                strategic_limit_point=strategic_limit_point,
            )

            # --- DYNAMICALLY SET TOTAL INVESTMENT FROM MODEL BASELINE ---
            total_monthly_investment = 0
            if baseline_point and "Daily_Investment" in baseline_point:
                total_monthly_investment = baseline_point["Daily_Investment"] * 30
            # ---------------------------------------------------------

            plot_filename = os.path.join(
                global_output_dir, "combined_all_channels_saturation_curve.png"
            )
            save_opportunity_curve_plot(
                response_curve_df,
                baseline_point,
                max_efficiency_point,
                diminishing_return_point,
                saturation_point,
                plot_filename,
                kpi_name=config.get("performance_kpi_column", "Sessions"),
                strategic_limit_point=strategic_limit_point,
                strategic_reallocation_point=strategic_reallocation_point,
                config=config,
            )

            # The MMM model returns contributions as percentages (0-100). Normalize to ratios (0-1).
            strategic_budget_split_pct = mmm_results["contribution_pct"]
            strategic_budget_split_ratio = {
                k: v / 100.0 for k, v in strategic_budget_split_pct.items()
            }

            scenarios = [
                {
                    "title": "Cenário de Investimento Atual",
                    "description": "Mantém o volume de investimento total idêntico à média histórica atual. Serve como linha de base para avaliarmos o ganho puro de eficiência ao alterar o mix.",
                    "total_investment": total_monthly_investment,
                    "splits": {
                        "Média Histórica": {
                            k: v * total_monthly_investment
                            for k, v in current_budget_split.items()
                        },
                        "Pico de Eficiência": {
                            k: v * total_monthly_investment
                            for k, v in optimized_budget_split.items()
                        },
                        "Modelo de Elasticidade": {
                            k: v * total_monthly_investment
                            for k, v in strategic_budget_split_ratio.items()
                        },
                    },
                    "projected_kpis": {
                        mix: kpi * 30
                        for mix, kpi in baseline_point.get("projected_kpis", {}).items()
                    },
                },
                {
                    "title": "Cenário de Investimento Otimizado",
                    "description": "Escala o investimento total para o nível onde foi observado o maior pico de eficiência teórica antes da saturação acelerada.",
                    "total_investment": max_efficiency_point["Daily_Investment"] * 30,
                    "splits": {
                        "Média Histórica": {
                            k: v * max_efficiency_point["Daily_Investment"] * 30
                            for k, v in current_budget_split.items()
                        },
                        "Pico de Eficiência": {
                            k: v * max_efficiency_point["Daily_Investment"] * 30
                            for k, v in optimized_budget_split.items()
                        },
                        "Modelo de Elasticidade": {
                            k: v * max_efficiency_point["Daily_Investment"] * 30
                            for k, v in strategic_budget_split_ratio.items()
                        },
                    },
                    "projected_kpis": {
                        mix: kpi * 30
                        for mix, kpi in max_efficiency_point.get(
                            "projected_kpis", {}
                        ).items()
                    },
                },
                {
                    "title": "Cenário de Investimento Estratégico",
                    "description": "Expande o orçamento até o limite estratégico calculado pelo modelo de elasticidade, focado em maximizar o volume de resultados.",
                    "total_investment": strategic_limit_point["Daily_Investment"] * 30,
                    "splits": {
                        "Média Histórica": {
                            k: v * strategic_limit_point["Daily_Investment"] * 30
                            for k, v in current_budget_split.items()
                        },
                        "Pico de Eficiência": {
                            k: v * strategic_limit_point["Daily_Investment"] * 30
                            for k, v in optimized_budget_split.items()
                        },
                        "Modelo de Elasticidade": {
                            k: v * strategic_limit_point["Daily_Investment"] * 30
                            for k, v in strategic_budget_split_ratio.items()
                        },
                    },
                    "projected_kpis": {
                        mix: kpi * 30
                        for mix, kpi in strategic_limit_point.get(
                            "projected_kpis", {}
                        ).items()
                    },
                },
            ]

            kpi_projections = {
                "current": baseline_point,
                "optimized": max_efficiency_point,
                "strategic": strategic_limit_point,
                "reallocation": strategic_reallocation_point,
            }

            # Create the comparative markdown file
            saturation_md_output_path = os.path.join(
                global_output_dir, "SATURATION_CURVE.md"
            )
            kpi_name = config.get("primary_business_metric_name", "KPIs")
            create_comparative_saturation_md(
                scenarios,
                saturation_md_output_path,
                kpi_projections=kpi_projections,
                kpi_name=kpi_name,
            )

            # --- Generate Global Gemini Report ---
            print("   - Generating Global Strategic Narrative with Gemini...")
            try:
                generate_global_gemini_report(
                    gemini_client,
                    config,
                    scenarios,
                    total_investment=total_monthly_investment,
                    kpi_projections=kpi_projections,
                )
                print(f"   SUCCESS! Global strategic analysis complete.")
            except Exception as e:
                print(
                    f"   - ERROR: Could not generate or parse the global narrative from Gemini. Details: {e}"
                )

        else:
            print(
                "   - ERROR: Global MMM analysis failed. Skipping global report generation."
            )

    except FileNotFoundError as e:
        print(
            f"ERROR: Input file not found. Please check the path in your config file. Details: {e}"
        )
        exit(1)

    except ValueError as e:
        print(f"ERROR: A data validation or processing error occurred. Details: {e}")
        exit(1)

    except Exception as e:
        print(f"A critical, unexpected error occurred during the main process: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from logger import LogContext

    with LogContext("local_main"):
        parser = argparse.ArgumentParser(
            description="Automated Total Opportunity Case Study Generator"
        )
        parser.add_argument(
            "--config", required=True, help="Path to the JSON configuration file."
        )
        parser.add_argument(
            "--min_intervention_date",
            help="Optional: Filter events to only include those after this date (YYYY-MM-DD).",
        )
        args = parser.parse_args()

        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found at '{args.config}'")
            exit(1)
        except json.JSONDecodeError:
            print(
                f"ERROR: Could not decode JSON from the configuration file '{args.config}'. Please check for syntax errors."
            )
            exit(1)

        main(config, args)
