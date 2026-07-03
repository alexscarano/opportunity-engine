# -*- coding: utf-8 -*-
"""
This module handles the generation of all presentation-ready outputs,
including charts, markdown files, and data for HTML reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import numpy as np
import sys

sys.path.append(os.path.abspath('scripts'))

try:
    import analysis
except ImportError as e:
    print(f"ERROR: Failed to import the 'analysis' module. Make sure it's in the 'scripts' directory.")
    print(f"   Details: {e}")
    sys.exit(1)

def format_number(n, currency=False):
    """Formats a number into a compact, readable string with abbreviations (k, M, B) and optional currency symbol."""
    prefix = 'R$ ' if currency else ''
    if n is None or pd.isna(n):
        return f"{prefix}0.00"
    
    is_negative = n < 0
    n = abs(n)

    if n == 0:
        return f"{prefix}0.00"
    elif n < 10:
        # High precision for CPA and small values
        formatted_num = f"{n:,.2f}"
    elif n < 1_000:
        formatted_num = f"{n:,.0f}"
    elif n < 1_000_000:
        formatted_num = f"{n/1_000:,.1f}k"
    elif n < 1_000_000_000:
        formatted_num = f"{n/1_000_000:,.1f}M"
    else:
        formatted_num = f"{n/1_000_000_000:,.1f}B"
        
    return f"-{prefix}{formatted_num}" if is_negative else f"{prefix}{formatted_num}"

def save_accuracy_plot(results_data, accuracy_df, output_path, kpi_name='kpi'):
    """Saves the model accuracy plot to a file, matching the reference style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plot with specific styles
    ax.plot(accuracy_df['Date'], accuracy_df['kpi'], color='black', linestyle='-', label=f'Actual {kpi_name}')
    ax.plot(accuracy_df['Date'], accuracy_df['Predicted'], color='red', linestyle='--', label=f'Predicted {kpi_name} (In-Sample)')
    
    ax.set_title(f'Model Accuracy: Actual vs. Predicted (Pre-Event Period)', fontsize=18, pad=20)
    ax.set_ylabel(kpi_name, fontsize=14)
    
    # Add styled text box for MAE
    mae_text = f"MAE (last 90 days): {results_data.get('mae', 0):.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, mae_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
            
    # Rotate date labels for readability
    fig.autofmt_xdate()
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_line_chart_plot(line_df, output_path, kpi_name='kpi'):
    """Saves the causal impact line chart with investment overlay to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # Plotting the KPI data on the primary y-axis (ax1)
    ax1.plot(line_df['Date'], line_df['Actual_KPI'], color='black', linestyle='-', label=f'Actual {kpi_name}')
    ax1.plot(line_df['Date'], line_df['Forecasted_KPI'], color='red', linestyle='--', label=f'Forecasted {kpi_name}')
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel(kpi_name, fontsize=14)
    ax1.tick_params(axis='y')

    # Creating a secondary y-axis for the investment data
    ax2 = ax1.twinx()
    ax2.bar(line_df['Date'], line_df['Investment'], color='blue', alpha=0.3, label='Investment')
    ax2.set_ylabel('Investment', color='blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='blue')

    # Formatting and legends
    ax1.set_title('Causal Impact Analysis: Actual vs. Forecasted', fontsize=18, pad=20)
    fig.tight_layout()
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines + bars, labels + bar_labels, loc='upper right')

    plt.savefig(output_path)
    plt.close(fig)

def save_investment_bar_plot(inv_bar_df, output_path):
    """Saves the investment bar chart to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    inv_bar_df.plot(kind='bar', ax=ax, color=['gray', 'green'], legend=None)
    ax.set_title('Investment: Pre-Event vs. Event', fontsize=16)
    ax.set_ylabel('Total Investment', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    # Use a generic formatter for the y-axis
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_sessions_bar_plot(sessions_bar_df, output_path, kpi_name='kpi'):
    """Saves the sessions bar chart to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Robust index renaming
    if 'Forecasted' in sessions_bar_df.index and 'Actual' in sessions_bar_df.index:
         new_index = {
             'Forecasted': f'Forecasted {kpi_name}',
             'Actual': f'Actual {kpi_name}'
         }
         sessions_bar_df = sessions_bar_df.rename(index=new_index)
    else:
         # Fallback if upstream logic changes (though analysis.py currently sets it correctly)
         sessions_bar_df.index = [f'Forecasted {kpi_name}', f'Actual {kpi_name}']
    
    # Ensure colors map to the correct semantic meaning regardless of order
    colors = ['red' if 'Forecasted' in str(x) else 'black' for x in sessions_bar_df.index]
    
    sessions_bar_df.plot(kind='bar', ax=ax, color=colors, legend=None)
    ax.set_title(f'Actual vs. Forecasted {kpi_name}', fontsize=16)
    ax.set_ylabel(f'Total {kpi_name}', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_opportunity_curve_plot(response_curve_df, baseline_point, max_efficiency_point,
                                diminishing_return_point, saturation_point, filename,
                                kpi_name='kpi', strategic_limit_point=None, strategic_reallocation_point=None, config=None):
    """
    Saves the saturation curve plot with smart labeling using ax.annotate.
    """
    # 1. Safety Check
    if response_curve_df is None or response_curve_df.empty:
        print(f"   - WARNING: Response curve data is empty. Skipping plot generation for {filename}.")
        return

    # 2. Data Preparation
    plot_df = response_curve_df.copy()

    # Standardize to monthly for plotting
    if 'Daily_Investment' in plot_df.columns:
        plot_df['Monthly_Investment'] = plot_df['Daily_Investment'] * 30
        plot_df['Monthly_KPIs'] = plot_df['Projected_Total_KPIs'] * 30
    else:
        plot_df.rename(columns={'Projected_Total_KPIs': 'Monthly_KPIs',
                                'Investment': 'Monthly_Investment'}, inplace=True)

    # Define formatters
    def currency_fmt(x):
        if x >= 1e6: return f'R${x/1e6:.1f}M'
        if x >= 1e3: return f'R${x/1e3:.0f}k'
        return f'R${x:.0f}'

    x_label = 'Investimento Mensal (R$)'

    # 3. Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot the curves
    if 'Projected_Total_KPIs_Historical' in plot_df.columns:
        ax.plot(plot_df['Monthly_Investment'], plot_df['Projected_Total_KPIs_Historical'] * 30,
                label='Curva - Mix Histórico', color='gray', linewidth=2, linestyle=':')
    if 'Projected_Total_KPIs_Optimized' in plot_df.columns:
        ax.plot(plot_df['Monthly_Investment'], plot_df['Projected_Total_KPIs_Optimized'] * 30,
                label='Curva - Mix Otimizado', color='salmon', linewidth=2, linestyle='--')
    ax.plot(plot_df['Monthly_Investment'], plot_df['Monthly_KPIs'],
            label='Curva - Elasticidade Estratégica', color='royalblue', linewidth=3)

    # --- INTERNAL HELPER FOR PLOTTING SCENARIO LINES ---
    def plot_scenario(point, color, label, xytext=(0, 10)):
        if not point: return

        d_inv = point.get('Daily_Investment', 0)
        d_kpi = point.get('Projected_Total_KPIs', 0)
        monthly_inv = d_inv * 30
        monthly_kpi = d_kpi * 30

        if monthly_inv == 0 and monthly_kpi == 0 and label != 'Cenário Atual': return

        # Draw a vertical line spanning the y-axis for this investment level
        ax.axvline(x=monthly_inv, color=color, linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)

        # Plot a very small dot at the exact intersection
        ax.scatter(monthly_inv, monthly_kpi, color=color, s=40, zorder=5)

        # Format Label Text (showing Total KPI as a full integer)
        kpi_str = f'{monthly_kpi:,.0f}' # Using .0f for full integer with commas
        label_text = f'{label}\n{currency_fmt(monthly_inv)}\n{kpi_str} {kpi_name}'

        ax.annotate(label_text,
                    xy=(monthly_inv, monthly_kpi),
                    xytext=xytext,
                    textcoords='offset points',
                    ha='center', va='top', # va='top' for below-point placement
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=color))

    # 4. Plotting Strategic Scenarios
    # Use different offsets to prevent overlap if lines coincide
    plot_scenario(baseline_point, 'gray', 'Cenário Atual',
                  xytext=(0, -60))

    # If Max Efficiency is the same as Baseline, offset its label upwards
    max_eff_offset = (0, 60) if max_efficiency_point and baseline_point and \
                      abs(max_efficiency_point.get('Daily_Investment', 0) - baseline_point.get('Daily_Investment', 0)) < 1e-3 \
                      else (0, -60)
    
    plot_scenario(max_efficiency_point, 'salmon', 'Máxima Eficiência',
                  xytext=max_eff_offset)

    if strategic_limit_point:
        plot_scenario(strategic_limit_point, 'royalblue', 'Limite Estratégico',
                   xytext=(0, -60))

    if diminishing_return_point:
         plot_scenario(diminishing_return_point, 'orange', 'Retorno Decrescente',
                    xytext=(0, -50))

    if strategic_reallocation_point:
        plot_scenario(strategic_reallocation_point, 'royalblue', 'Realocação (Mesmo Orçamento)',
                   xytext=(0, 60))

    # 5. Final Cosmetics
    ax.set_title('Curva de Resposta: Cenários Estratégicos de Investimento (Mensal)', fontsize=20, pad=20)
    ax.set_xlabel(x_label, fontsize=16, labelpad=15)
    ax.set_ylabel(f'{kpi_name} Projetado (Mensal)', fontsize=16, labelpad=15)

    # Set formatters for axes (with full integer for y-axis)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: currency_fmt(x).replace('R$', 'R$ ')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:,.0f}')) # .0f for full integer

    # --- Extend the x-axis ---
    if strategic_limit_point and 'Daily_Investment' in strategic_limit_point:
        max_investment_on_plot = strategic_limit_point['Daily_Investment'] * 30
        ax.set_xlim(right=max_investment_on_plot * 1.5)
    # --- End New ---

    ax.legend(fontsize=14, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"   - Chart saved to {filename}")

def create_comparative_saturation_md(scenarios, output_filename, kpi_projections=None, kpi_name='KPIs'):
    """
    Generates a markdown file with a separate table for each budget scenario,
    each with the three splits.
    """
    
    markdown_content = "# Análise Comparativa de Alocação de Orçamento\n\n"

    # --- New: Summary Table ---
    if kpi_projections:
        markdown_content += "## Resumo dos Cenários Projetados\n\n"
        markdown_content += "A tabela abaixo apresenta os resultados projetados de quatro cenários de investimento chave, assumindo que seus respectivos mix recomendados sejam aplicados:\n\n"
        markdown_content += "- **Cenário Atual (Média Histórica):** Mantém o nível de investimento e o mix idênticos às médias observadas.\n"
        markdown_content += "- **Cenário Otimizado (Pico de Eficiência):** Escala o investimento total para o ponto de maior eficiência detectado e usa o mix dos melhores períodos.\n"
        markdown_content += "- **Cenário Estratégico (Modelo de Elasticidade):** Expande o orçamento até o limite ótimo de saturação calculado pelo modelo iterativo.\n"
        markdown_content += "- **Realocação Estratégica (Mesmo Orçamento):** Mantém o investimento atual, mas redistribui a verba segundo o Modelo de Elasticidade para ganho puro de eficiência.\n\n"
        markdown_content += f"| Cenário | Investimento Mensal | Projeção de {kpi_name} | Custo por {kpi_name} (CPA) | Investimento Incremental | {kpi_name} Incrementais | iCPA |\n"
        markdown_content += "|:---|:---|:---|:---|:---|:---|:---|\n"
        
        scenario_map = [
            ('Cenário Atual (Média Histórica)', 'current'),
            ('Cenário Otimizado (Pico de Eficiência)', 'optimized'),
            ('Cenário Estratégico (Modelo de Elasticidade)', 'strategic'),
            ('Realocação Estratégica (Mesmo Orçamento)', 'reallocation')
        ]
        
        baseline_point = kpi_projections.get('current', {})
        baseline_inv = baseline_point.get('Daily_Investment', 0) * 30
        
        for title, key in scenario_map:
            point = kpi_projections.get(key)
            if point:
                # Scale from daily to monthly just once here
                inv = point.get('Daily_Investment', 0) * 30
                kpi = point.get('Projected_Total_KPIs', 0) * 30
                inc_kpi = point.get('Incremental_KPI', 0) * 30
                
                # Calculate metrics
                inc_inv = inv - baseline_inv
                cpa = inv / kpi if kpi > 0 else 0
                icpa = inc_inv / inc_kpi if inc_kpi > 0 else 0
                
                # Force strictly 0 incrementals for the baseline itself to avoid floating point artifacts
                if key == 'current':
                    inc_inv = 0.0
                    inc_kpi = 0.0
                    icpa = 0.0
                
                # Format strings
                inv_str = format_number(inv, currency=True)
                kpi_str = format_number(kpi)
                cpa_str = format_number(cpa, currency=True)
                inc_inv_str = format_number(inc_inv, currency=True)
                inc_kpi_str = format_number(inc_kpi)
                icpa_str = format_number(icpa, currency=True)
                
                markdown_content += f"| **{title}** | {inv_str} | {kpi_str} | {cpa_str} | {inc_inv_str} | {inc_kpi_str} | {icpa_str} |\n"
        markdown_content += "\n---\n\n"
    # --- End New ---
    
    for scen in scenarios:
        markdown_content += f"### {scen['title']}\n\n"
        if 'description' in scen:
            markdown_content += f"{scen['description']}\n\n"
        
        header = "| Canal | Média Histórica | Pico de Eficiência | Modelo de Elasticidade |\n"
        separator = "|:---|:---|:---|:---|\n"
        
        body = ""
        all_channels = sorted(list(set(ch for split in scen['splits'].values() for ch in split.keys())))
        
        for channel in all_channels:
            row = f"| **{channel}** |"
            for split_name in ['Média Histórica', 'Pico de Eficiência', 'Modelo de Elasticidade']:
                split = scen['splits'][split_name]
                investment = split.get(channel, 0)
                total_investment = sum(split.values())
                percentage = (investment / total_investment * 100) if total_investment > 0 else 0
                row += f" {format_number(investment, currency=True)} ({percentage:.2f}%) |"
            body += row + "\n"
            
        total_row = "| **Total** |"
        for split_name in ['Média Histórica', 'Pico de Eficiência', 'Modelo de Elasticidade']:
            split = scen['splits'][split_name]
            total_row += f" **{format_number(sum(split.values()), currency=True)} (100.00%)** |"
        body += total_row + "\n"
        
        # --- NEW: Append KPIs and CPA rows ---
        projected_kpis = scen.get('projected_kpis', {})
        total_inv = scen.get('total_investment', 0)
        
        if projected_kpis:
            kpi_row = f"| **Projeção de {kpi_name}** |"
            cpa_row = f"| **CPA** |"
            
            for split_name in ['Média Histórica', 'Pico de Eficiência', 'Modelo de Elasticidade']:
                kpi_val = projected_kpis.get(split_name, 0)
                cpa_val = total_inv / kpi_val if kpi_val > 0 else 0
                
                kpi_row += f" **{format_number(kpi_val)}** |"
                cpa_row += f" **{format_number(cpa_val, currency=True)}** |"
                
            body += kpi_row + "\n" + cpa_row + "\n"
        
        markdown_content += header + separator + body + "\n"

    methodology = """
---
## Metodologia de Distribuição de Investimento (Mix de Canais)

Para cada nível de orçamento projetado nas tabelas anteriores, a análise apresenta três abordagens diferentes de distribuição da verba entre os canais (as colunas das tabelas). Cada uma serve a um propósito estratégico:

### 1. Mix: Média Histórica
- **Metodologia:** Esta abordagem distribui a verba reproduzindo a proporção exata observada no passado.
- **Ponto Forte:** Serve primariamente como a linha de base para avaliarmos o ganho da modelagem sem mudanças bruscas.

### 2. Mix: Pico de Eficiência
- **Metodologia:** Identifica isoladamente os 10% das semanas com o melhor índice de eficiência geral e aplica o mix exato que estava rodando lá.
- **Ponto Forte:** É uma visão baseada em sucessos observados de curto prazo. Ideal para alavancar crescimento em cima de setups que "já sabemos que performaram bem".

### 3. Mix: Modelo de Elasticidade
- **Metodologia:** Utiliza as equações causais do motor (Curvas de Saturação e Adstock) calculadas em cima de todo o histórico para decompor o impacto de cada canal na base. A verba é iterativamente alocada para igualar a eficiência marginal entre canais de forma matemática.
- **Ponto Forte:** A principal e mais robusta visão estratégica de "always-on". Otimiza retornos de longo prazo prevendo o ponto em que cada canal esgota sua força, gerando a maior receita ou volume de aquisições toleráveis no budget estipulado.
"""

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        f.write(methodology)

    print(f"   - Successfully generated comparative MD file at: {output_filename}")

def save_investment_distribution_donuts(donut_scenarios, output_path, total_investment=None):
    """
    Generates and saves a figure with multiple donut charts visualizing the investment mix for different scenarios.
    Includes absolute values if total_investment is provided.
    """
    try:
        if not donut_scenarios:
            print("   - WARNING: No budget scenarios to plot.")
            return

        num_scenarios = len(donut_scenarios)
        fig, axs = plt.subplots(1, num_scenarios, figsize=(8 * num_scenarios, 8))
        if num_scenarios == 1:
            axs = [axs]

        all_labels = sorted(list(set(label for scenario in donut_scenarios for label in scenario.get('data', {}).keys())))
        colors = plt.cm.get_cmap('tab20', len(all_labels))
        color_map = {label: colors(i) for i, label in enumerate(all_labels)}

        # Custom autopct function to display both percentage and absolute value
        def make_autopct(values, total):
            def my_autopct(pct):
                if total is None or total == 0:
                    return f'{pct:.1f}%'
                absolute = pct/100.*total
                return f'{pct:.1f}%\n(R$ {absolute/1e6:.1f}M)'
            return my_autopct

        for i, scenario in enumerate(donut_scenarios):
            ax = axs[i]
            title = scenario.get('title', '')
            data = scenario.get('data', {})

            if not data or sum(data.values()) == 0:
                ax.text(0.5, 0.5, 'Sem dados', horizontalalignment='center', verticalalignment='center')
                ax.set_title(title, fontsize=18, pad=20, weight="bold")
                ax.axis('off')
                continue
            
            # Filter out channels with zero or negligible investment for plotting
            plot_data = {k: v for k, v in data.items() if v > 0.001}
            
            labels = list(plot_data.keys())
            sizes = list(plot_data.values())
            pie_colors = [color_map[label] for label in labels]

            # Use the custom autopct function
            autopct_func = make_autopct(sizes, total_investment) if total_investment else '%1.1f%%'

            wedges, texts, autotexts = ax.pie(
                sizes, autopct=autopct_func, startangle=90,
                pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor='w'),
                colors=pie_colors,
                textprops={'color': "white"}
            )
            
            plt.setp(autotexts, size=12, weight="bold")
            ax.set_title(title, fontsize=18, pad=20, weight="bold")

        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in all_labels]
        fig.legend(handles, all_labels, title="Canais", loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(all_labels)//2)

        fig.suptitle("Distribuição de Investimento por Cenário", fontsize=24, weight='bold')
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"   - Donut charts saved to {output_path}")

    except Exception as e:
        print(f"   - ERROR: Could not generate donut charts. Details: {e}")
        import traceback
        traceback.print_exc()