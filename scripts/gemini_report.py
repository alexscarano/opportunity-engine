# -*- coding: utf-8 -*-
"""
This module contains the Gemini integration, responsible for generating a strategic
narrative and an HTML report based on the causal impact analysis results.
It uses a single, comprehensive prompt to ensure a cohesive and insightful narrative
aligned with the Total Opportunity framework.
"""

import base64
import json
import os
import html
import pandas as pd
import google.generativeai as genai
from presentation import format_number

def _get_image_as_base64(path):
    """Reads an image file and returns it as a base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"   - WARNING: Image file not found at {path}. It will be omitted from the HTML report.")
        return None
    except Exception as e:
        print(f"   - WARNING: Could not read image file at {path}. Error: {e}")
        return None

def _generate_full_report_narrative(gemini_client, results_data, config, market_analysis_df, csv_output_filename=None, correlation_matrix=None):
    """
    Generates the entire report narrative with a single, comprehensive prompt.
    """
    print("   - Generating full strategic narrative with Gemini...")

    # --- 1. Prepare all data for the prompt ---
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)
    
    # --- Efficiency & Causal Calculations ---
    inv_post = results_data.get('total_investment_post_period', 0)
    inv_change_pct = results_data.get('investment_change_pct', 0)
    baseline_inv_post = inv_post / (1 + (inv_change_pct / 100)) if inv_change_pct != -100 else 0
    incremental_investment = inv_post - baseline_inv_post
    
    # --- DYNAMIC LOGIC ---
    if avg_ticket > 0:
        business_impact_value = business_impact_sales * avg_ticket
        business_impact_label = "incremental_revenue"
        business_impact_formatted_value = format_number(business_impact_value, currency=True)
        recommendation_kpi = "receita incremental"
        
        roi = (business_impact_value - incremental_investment) / incremental_investment if incremental_investment > 0 else 0
        efficiency_metric = f"ROI Incremental: {roi:.2f}x" if incremental_investment > 0 else "N/A (Investimento Reduzido)"
    else:
        business_impact_value = business_impact_sales
        business_impact_label = "incremental_orders"
        business_impact_formatted_value = f"{business_impact_value:,.0f} pedidos"
        recommendation_kpi = "pedidos incrementais"
        
        cpa = incremental_investment / business_impact_value if business_impact_value > 0 else 0
        efficiency_metric = f"CPA Incremental: {format_number(cpa, currency=True)}" if business_impact_value > 0 else "N/A (S/ Lift Positivo)"
    # --- END DYNAMIC LOGIC ---

    # Consolidate data into a dictionary for easy serialization
    prompt_data = {
        "client_name": config['advertiser_name'],
        "client_industry": config.get('client_industry', 'their industry'),
        "business_goal": config['client_business_goal'],
        "kpi_name": config['primary_business_metric_name'],
        "causal_impact_results": {
            "product_group": results_data['product_group'],
            "is_significant": str(results_data['p_value'] < config['p_value_threshold']),
            "p_value": f"{results_data['p_value']:.4f}",
            "investment_change_pct": f"{inv_change_pct:.1f}%",
            "incremental_investment": format_number(incremental_investment, currency=True),
            "absolute_lift_kpi": f"{results_data['absolute_lift']:,.0f}",
            business_impact_label: business_impact_formatted_value,
            "efficiency_metric": efficiency_metric
        },
        "model_validation_metrics": {
            "r_squared": f"{results_data.get('model_r_squared', 0):.2f}",
            "mae": f"{results_data.get('mae', 0):.2f}",
            "mape": f"{results_data.get('mape', 0):.2f}%"
        }
    }

    if csv_output_filename and os.path.exists(csv_output_filename):
        presentation_df = pd.read_csv(csv_output_filename)
        prompt_data['presentation_data_csv'] = presentation_df.to_string()

    # --- 2. Define the JSON structure Gemini should return ---
    json_schema = f"""
    {{
      "report_title": "A concise, executive-level title for the report.",
      "executive_verdict": "A clear, decisive statement on whether the investment change was a strategic success or failure based on efficiency and ROI/CPA.",
      "detailed_analysis": "A deep dive into the investment vs. return. Analyze if the increased/decreased spend was justified by the proportional change in {recommendation_kpi}. Discuss the efficiency of the incremental spend.",
      "value_delivered": {{
        "narrative": "A paragraph summarizing the proven past impact. It must quantify the incremental lift in business terms ({recommendation_kpi}) using the provided data.",
        "methodology_narrative": "A brief, executive-friendly explanation of the statsmodels.tsa.UnobservedComponents methodology used for the causal impact analysis."
      }},
      "next_steps": [
        {{ "step": "Step 1 Title", "description": "A tactical, actionable recommendation on what to do next given this specific outcome (e.g., scale further, pull back, investigate creative)." }},
        {{ "step": "Step 2 Title", "description": "A second tactical, actionable recommendation." }},
        {{ "step": "Step 3 Title", "description": "A third tactical, actionable recommendation." }}
      ]
    }}
    """

    # --- 3. Construct the final prompt ---
    prompt = f"""
    As a senior Google marketing strategist, your task is to create a comprehensive business report for {prompt_data['client_name']}.
    The report should focus on the "Value Delivered", proving the past impact of the marketing intervention and analyzing its efficiency.

    **CRITICAL INSTRUCTION: Your entire output must be in Brazilian Portuguese (pt-BR).**

    Analyze all the provided data, paying special attention to the `efficiency_metric`, to generate a cohesive and insightful narrative. Your entire output must be a single, valid JSON object matching the schema provided below. Do not include any text before or after the JSON object.

    **DATA FOR ANALYSIS:**
    ```json
    {json.dumps(prompt_data, indent=2)}
    ```

    **REQUIRED JSON OUTPUT STRUCTURE:**
    ```json
    {json_schema}
    ```
    """

    if correlation_matrix is not None:
        prompt += f"""
<correlation_analysis>
### Análise de Correlação
A matriz de correlação entre o investimento diário total e os KPIs de negócio (leads e conversões) é a seguinte:
{correlation_matrix.to_string()}

**Instrução Adicional:** Use esta matriz de correlação como um insight fundamental em sua análise. Destaque as relações fracas ou negativas e discuta como isso impacta a previsibilidade do modelo e a estratégia de marketing.
</correlation_analysis>
"""
    try:
        response = gemini_client.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
        narrative = json.loads(cleaned_response_text)
        print("   - Full narrative generated and parsed successfully.")
        return narrative
    except Exception as e:
        print(f"   - ERROR: Could not generate or parse the full narrative from Gemini. Details: {e}")
        return json.loads(json_schema.replace('...', 'Error: Could not generate content.'))

def generate_markdown_report_from_narrative(narrative, results_data, config, output_filename):
    """
    Generates a clean, causal-impact focused RECOMMENDATIONS.md from the Gemini narrative.
    """
    print(f"   - Generating Markdown report to '{output_filename}'...")
    
    report_title = narrative.get('report_title', 'Recomendações de Investimento e Impacto Causal')
    executive_verdict = narrative.get('executive_verdict', '')
    detailed_analysis = narrative.get('detailed_analysis', '')
    value_delivered = narrative.get('value_delivered', {}).get('narrative', '')
    
    # Calculate efficiency metrics
    inv_post = results_data.get('total_investment_post_period', 0)
    inv_change_pct = results_data.get('investment_change_pct', 0)
    baseline_inv_post = inv_post / (1 + (inv_change_pct / 100)) if inv_change_pct != -100 else 0
    incremental_investment = inv_post - baseline_inv_post
    
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)
    
    metrics_md = ""
    if avg_ticket > 0:
        business_impact_value = business_impact_sales * avg_ticket
        roi = (business_impact_value - incremental_investment) / incremental_investment if incremental_investment > 0 else 0
        metrics_md = (
            f"- **Investimento Incremental:** {format_number(incremental_investment, currency=True)}\n"
            f"- **Receita Incremental:** {format_number(business_impact_value, currency=True)}\n"
            f"- **ROI Incremental:** {roi:.2f}x\n"
        )
    else:
        business_impact_value = business_impact_sales
        cpa = incremental_investment / business_impact_value if business_impact_value > 0 else 0
        metrics_md = (
            f"- **Investimento Incremental:** {format_number(incremental_investment, currency=True)}\n"
            f"- **Pedidos Incrementais:** {format_number(business_impact_value)}\n"
            f"- **CPA Incremental:** {format_number(cpa, currency=True)}\n"
        )

    next_steps_md = "## Próximos Passos Estratégicos\n\n"
    if narrative.get('next_steps') and isinstance(narrative['next_steps'], list):
        for item in narrative['next_steps']:
            if isinstance(item, dict):
                next_steps_md += f"### {item.get('step', '')}\n{item.get('description', '')}\n\n"

    md_content = f"""# {report_title}

## Veredito Executivo
**{executive_verdict}**

## Métricas de Eficiência Incremental
{metrics_md}

## Análise Aprofundada
{detailed_analysis}

## O Impacto Causal e Valor Entregue
{value_delivered}

{next_steps_md}
"""
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print("   - Markdown report generated successfully.")
    except Exception as e:
        print(f"   - ERROR: Could not write Markdown report to file. Details: {e}")

def generate_html_report(gemini_client, results_data, config, image_paths, output_filename, market_analysis_df, causal_impact_df, csv_output_filename=None, correlation_matrix=None):
    """
    Generates a self-contained HTML report using the AI-generated narrative for event impact.
    """
    print(f"   - Assembling Gemini HTML report to '{output_filename}'...")

    # --- 1. Image Embedding ---
    image_b64s = {key: _get_image_as_base64(path) for key, path in image_paths.items() if path}

    # --- 2. AI Narrative Generation ---
    narrative = _generate_full_report_narrative(gemini_client, results_data, config, market_analysis_df, csv_output_filename=csv_output_filename, correlation_matrix=correlation_matrix)

    # --- 3. Data Calculation for Tables ---
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)
    business_impact_revenue = business_impact_sales * avg_ticket

    inv_post = results_data.get('total_investment_post_period', 0)
    inv_change_pct = results_data.get('investment_change_pct', 0)
    baseline_inv_post = inv_post / (1 + (inv_change_pct / 100)) if inv_change_pct != -100 else 0
    incremental_investment = inv_post - baseline_inv_post

    incremental_investment_str = format_number(incremental_investment, currency=True)
    if avg_ticket > 0:
        business_impact_label_str = "Receita"
        business_impact_str = format_number(business_impact_revenue, currency=True)
        roi = (business_impact_revenue - incremental_investment) / incremental_investment if incremental_investment > 0 else 0
        efficiency_label = "ROI Incremental"
        efficiency_val = f"{roi:.2f}x" if incremental_investment > 0 else "N/A"
    else:
        business_impact_label_str = "Pedidos"
        business_impact_str = format_number(business_impact_sales)
        cpa = incremental_investment / business_impact_sales if business_impact_sales > 0 else 0
        efficiency_label = "CPA Incremental"
        efficiency_val = format_number(cpa, currency=True) if business_impact_sales > 0 else "N/A"

    # --- 4. Build HTML Components ---

    # Next Steps List
    next_steps_html = ""
    if narrative.get('next_steps') and isinstance(narrative['next_steps'], list):
        for item in narrative['next_steps']:
            if isinstance(item, dict):
                next_steps_html += f"<li><strong>{item.get('step', '')}:</strong> {item.get('description', '')}</li>"

    # Generate Markdown Report directly
    markdown_filename = os.path.join(os.path.dirname(output_filename), "RECOMMENDATIONS.md")
    generate_markdown_report_from_narrative(narrative, results_data, config, markdown_filename)

    # --- 5. Final HTML Assembly ---
    html_template = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{ font-family: 'Google Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #3c4043; }}
            .container {{ max-width: 900px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.24); }}
            .header {{ background-color: #4285F4; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .section {{ padding: 20px; border-bottom: 1px solid #e0e0e0; }}
            .section:last-child {{ border-bottom: none; }}
            .section h2 {{ font-size: 20px; color: #1a73e8; margin-top: 0; }}
            .section h3 {{ font-size: 18px; color: #3c4043; margin-top: 20px; }}
            .section p, .section li {{ font-size: 16px; line-height: 1.6; }}
            .chart-container {{ text-align: center; margin-top: 20px; }}
            .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; }}
            .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #5f6368; }}
            .highlight-section {{ background-color: #f1f3f4; border-left: 4px solid #1a73e8; padding: 20px; margin-top: 20px; border-radius: 4px; }}
            .metrics-list li {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>{report_title}</h1></div>

            <div class="highlight-section">
                <h2 style="color: #202124; margin-top: 0;">Veredito Executivo</h2>
                <p style="font-size: 18px; font-weight: 500; margin-bottom: 0;">{executive_verdict}</p>
            </div>

            <div class="section">
                <h2>Análise Aprofundada e Eficiência</h2>
                <p>{detailed_analysis}</p>
                <div style="background-color: #fff; border: 1px solid #dadce0; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <h3 style="margin-top: 0; margin-bottom: 10px;">Métricas de Eficiência Incremental</h3>
                    <ul class="metrics-list" style="margin-bottom: 0;">
                        <li><strong>Investimento Incremental:</strong> {incremental_investment_str}</li>
                        <li><strong>Lift Mensurável ({business_impact_label_str}):</strong> {business_impact_str}</li>
                        <li><strong>{efficiency_label}:</strong> {efficiency_val}</li>
                    </ul>
                </div>
            </div>

            <div class="section">
                <h2>O Impacto Causal e Metodologia</h2>
                <p>{value_delivered_narrative}</p>
                <div class="chart-container"><img src="data:image/png;base64,{line_img}" alt="Gráfico de Análise de Impacto Causal"></div>
                <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                    <div class="chart-container" style="width: 48%;"><img src="data:image/png;base64,{investment_img}" alt="Gráfico de Investimento"></div>
                    <div class="chart-container" style="width: 48%;"><img src="data:image/png;base64,{sessions_img}" alt="Gráfico de Sessões"></div>
                </div>
                <h3>A Metodologia Opcional</h3>
                <p>{methodology_narrative}</p>
            </div>

            <div class="section">
                <h2>Próximos Passos Estratégicos</h2>
                <ul>{next_steps_html}</ul>
            </div>

            <div class="section">
                <h2>Apêndice: Validação do Modelo Estatístico</h2>
                <p>A validade desta análise baseia-se na capacidade do modelo de prever com precisão o desempenho durante o período pré-evento. As métricas abaixo demonstram a robustez e a confiabilidade do modelo:</p>
                <ul class="metrics-list">
                    <li><strong>R-squared (R²):</strong> {r_squared:.2f}. Isso indica que o modelo explica {r_squared_pct:.0%} da variância no KPI principal, demonstrando um forte ajuste.</li>
                    <li><strong>P-value (Significância do Lift):</strong> {p_value:.4f}. Este valor representa a probabilidade de o lift observado ter ocorrido por acaso. Um valor baixo como este nos dá alta confiança de que o impacto da campanha é real.</li>
                    <li><strong>Mean Absolute Percentage Error (MAPE):</strong> {mape:.2f}%. Em média, as previsões do modelo no período pré-evento desviaram apenas este percentual dos valores reais, indicando alta precisão.</li>
                </ul>
                <div class="chart-container"><img src="data:image/png;base64,{accuracy_img}" alt="Gráfico de Precisão do Modelo"></div>
            </div>

            <div class="section">
                <h2>Apêndice: Premissas da Análise</h2>
                <p>Esta análise foi baseada nas seguintes premissas e configurações-chave:</p>
                <ul>
                    <li><strong>Valor Médio por Venda (Ticket Médio):</strong> {avg_ticket_formatted}</li>
                    <li><strong>Taxa de Conversão (de KPI para Venda):</strong> {conversion_rate:.4f}</li>
                    <li><strong>Limiar de Significância Estatística (p-value):</strong> {p_value_threshold}</li>
                </ul>
            </div>
            <div class="footer"><p>Gerado pelo Módulo de Inteligência Artificial da Max Impact Engine (Total Opportunity).</p></div>
        </div>
    </body>
    </html>
    """.format(
        report_title=html.escape(narrative.get('report_title', 'Análise de Impacto Causal')),
        executive_verdict=narrative.get('executive_verdict', ''),
        detailed_analysis=narrative.get('detailed_analysis', ''),
        value_delivered_narrative=narrative.get('value_delivered', {}).get('narrative', ''),
        methodology_narrative=narrative.get('value_delivered', {}).get('methodology_narrative', ''),
        next_steps_html=next_steps_html,
        incremental_investment_str=incremental_investment_str,
        business_impact_label_str=business_impact_label_str,
        business_impact_str=business_impact_str,
        efficiency_label=efficiency_label,
        efficiency_val=efficiency_val,
        line_img=image_b64s.get('line', ''),
        investment_img=image_b64s.get('investment', ''),
        sessions_img=image_b64s.get('sessions', ''),
        accuracy_img=image_b64s.get('accuracy', ''),
        r_squared=results_data.get('model_r_squared', 0),
        r_squared_pct=results_data.get('model_r_squared', 0),
        p_value=results_data.get('p_value', 0),
        mape=results_data.get('mape', 0),
        avg_ticket_formatted=format_number(avg_ticket, currency=True),
        conversion_rate=config.get('conversion_rate_from_kpi_to_bo', 0),
        p_value_threshold=config.get('p_value_threshold', 0.05)
    )

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"   - Gemini HTML report saved successfully.")
    except Exception as e:
        print(f"   - ERROR: Could not write HTML report to file. Details: {e}")


def generate_global_gemini_report(gemini_client, config, scenarios=None, total_investment=None, kpi_projections=None):
    """
    Generates a dedicated Gemini report for the global saturation analysis.
    """
    print("\n" + "="*50 + "\nGenerating Global Gemini Report...\n" + "="*50)
    
    advertiser_name = config.get('advertiser_name', 'default_advertiser')
    global_output_dir = os.path.join(os.getcwd(), config['output_directory'], advertiser_name, 'global_saturation_analysis')
    
    # --- 1. Define paths and read artifacts ---
    markdown_path = os.path.join(global_output_dir, 'SATURATION_CURVE.md')
    response_curve_path = os.path.join(global_output_dir, 'combined_all_channels_saturation_curve.png')
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"   - ERROR: Could not find SATURATION_CURVE.md at {markdown_path}. Halting global report generation.")
        return

    image_b64s = {
        "response_curve": _get_image_as_base64(response_curve_path)
    }

    # --- 2. Define the JSON structure for Gemini ---
    json_schema = """
    {{
      "report_title": "A concise, executive-level title for the global marketing strategy report.",
      "executive_summary": "A high-level summary of the key findings and the main strategic recommendation.",
      "analysis_of_scenarios": {{
        "introduction": "A paragraph introducing the three investment scenarios (Current, Optimized, Strategic).",
        "scenario_table": [
          {{
            "scenario_name": "Atual (Média Histórica)",
            "analysis": "An analysis of the 'Atual (Média Histórica)' scenario, explaining its performance and investment mix based on the historical average."
          }},
          {{
            "scenario_name": "Otimizado (Pico de Eficiência)",
            "analysis": "An analysis of the 'Otimizado (Pico de Eficiência)' scenario, highlighting the efficiency gains from adopting the investment mix from peak performance weeks."
          }},
          {{
            "scenario_name": "Estratégico (Modelo de Elasticidade)",
            "analysis": "An analysis of the 'Estratégico (Modelo de Elasticidade)' scenario, explaining the rationale for the budget allocation based on the model's long-term contribution findings."
          }},
          {{
            "scenario_name": "Realocação Estratégica (Mesmo Orçamento)",
            "analysis": "An analysis of the 'Realocação Estratégica' scenario, highlighting the efficiency gains and incremental value achieved purely by reallocating the current budget according to the elasticity model, without any actual increase in spend."
          }}
        ]
      }},
      "strategic_recommendations": [
        { "recommendation": "A primary, actionable recommendation based on the analysis." },
        { "recommendation": "A secondary, actionable recommendation." }
      ]
    }}
    """

    # --- 3. Construct the prompt ---
    prompt = f"""
    Como um estrategista de marketing sênior do Google, sua tarefa é criar um relatório executivo de estratégia de marketing global para {advertiser_name}.
    O relatório deve ser conciso, focado em insights acionáveis e totalmente em Português do Brasil (pt-BR).
    Sua saída deve ser um único objeto JSON válido, correspondendo ao esquema fornecido abaixo. Não inclua texto antes ou depois do JSON.

    **DADOS PARA ANÁLISE:**

    **1. Relatório Markdown com Tabelas de Cenários:**
    ```markdown
    {markdown_content}
    ```

    **2. Visualizações Chave:**
    - A imagem é a 'Curva de Resposta', mostrando o KPI projetado em diferentes níveis de investimento mensal.

    **SUA TAREFA:**
    Analise os dados e as imagens para gerar uma narrativa coesa e perspicaz. O foco é a clareza e a concisão para um público executivo.

    **ESTRUTURA DE SAÍDA JSON OBRIGATÓRIA:**
    ```json
    {json_schema}
    ```
    """

    # --- 4. Call Gemini API ---
    try:
        model = genai.GenerativeModel(gemini_client.model_name)
        response = gemini_client.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
        narrative = json.loads(cleaned_response_text)
        print("   - Global narrative generated and parsed successfully.")
        
        # NEW: Save the JSON payload so the Streamlit UI can render the insights without re-running Gemini
        json_output_path = os.path.join(global_output_dir, 'global_narrative.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(narrative, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"   - ERROR: Could not generate or parse the global narrative from Gemini. Details: {e}")
        return

    # --- 5. Assemble HTML Report ---
    output_filename = os.path.join(global_output_dir, 'global_report.html')
    
    # --- New: Dynamically build the scenarios analysis table ---
    scenarios_analysis_html = '<table class="scenarios-table"><tr><th>Cenário</th><th>Análise</th></tr>'
    scenario_table_data = narrative.get('analysis_of_scenarios', {}).get('scenario_table', [])
    for row in scenario_table_data:
        scenarios_analysis_html += f"<tr><td><strong>{row.get('scenario_name', '')}</strong></td><td>{row.get('analysis', '')}</td></tr>"
    scenarios_analysis_html += "</table>"
    # --- End New ---

    # --- New: Build detailed channel mix table ---
    channel_mix_html = ""
    if scenarios:
        for scen in scenarios:
            title = scen['title']
            
            channel_mix_html += f"<h3>{title}</h3>"
            if 'description' in scen:
                channel_mix_html += f"<p>{scen['description']}</p>"
                
            header = "<th>Canal</th><th>Média Histórica</th><th>Pico de Eficiência</th><th>Modelo de Elasticidade</th>"
            
            rows = ""
            all_channels = sorted(list(set(ch for split in scen['splits'].values() for ch in split.keys())))
            
            for channel in all_channels:
                row = f"<tr><td><strong>{channel}</strong></td>"
                for split_name in ['Média Histórica', 'Pico de Eficiência', 'Modelo de Elasticidade']:
                    split = scen['splits'][split_name]
                    investment = split.get(channel, 0)
                    total_investment = sum(split.values())
                    percentage = (investment / total_investment * 100) if total_investment > 0 else 0
                    row += f"<td>{format_number(investment, currency=True)} ({percentage:.2f}%)</td>"
                row += "</tr>"
                rows += row
                
            total_row = "<tr><td><strong>Total</strong></td>"
            for split_name in ['Média Histórica', 'Pico de Eficiência', 'Modelo de Elasticidade']:
                split = scen['splits'][split_name]
                total_row += f"<td><strong>{format_number(sum(split.values()), currency=True)} (100.00%)</strong></td>"
            total_row += "</tr>"
            
            # --- NEW: Append KPIs and CPA rows to HTML table ---
            projected_kpis = scen.get('projected_kpis', {})
            total_inv = scen.get('total_investment', 0)
            kpi_cpa_rows = ""
            
            if projected_kpis:
                avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
                conv_rate = config.get('conversion_rate_from_kpi_to_bo', 0)
                
                if avg_ticket > 0:
                    kpi_row = "<tr><td><strong>Projeção de Receita</strong></td>"
                    cpa_row = "<tr><td><strong>ROAS</strong></td>"
                else:
                    kpi_label = config.get('primary_business_metric_name', 'KPIs')
                    kpi_row = f"<tr><td><strong>Projeção de {kpi_label}</strong></td>"
                    cpa_row = "<tr><td><strong>CPA</strong></td>"
                
                for split_name in ['Média Histórica', 'Pico de Eficiência', 'Modelo de Elasticidade']:
                    raw_kpi = projected_kpis.get(split_name, 0)
                    actual_kpi = raw_kpi * conv_rate if conv_rate > 0 else raw_kpi
                    
                    if avg_ticket > 0:
                        revenue = actual_kpi * avg_ticket
                        roas = revenue / total_inv if total_inv > 0 else 0
                        kpi_row += f"<td><strong>{format_number(revenue, currency=True)}</strong></td>"
                        cpa_row += f"<td><strong>{roas:.2f}</strong></td>"
                    else:
                        cpa_val = total_inv / actual_kpi if actual_kpi > 0 else 0
                        kpi_row += f"<td><strong>{format_number(actual_kpi)}</strong></td>"
                        cpa_row += f"<td><strong>{format_number(cpa_val, currency=True)}</strong></td>"
                
                kpi_row += "</tr>"
                cpa_row += "</tr>"
                kpi_cpa_rows = kpi_row + cpa_row
            
            channel_mix_html += f"""
            <table class="scenarios-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{rows}{total_row}{kpi_cpa_rows}</tbody>
            </table>
            """
    # --- End New ---

    # Define CSS styles separately
    css_styles = r"""
    body { font-family: 'Google Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #3c4043; }
    .container { max-width: 1000px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.24); }
    .header { background-color: #1a73e8; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }
    .header h1 { margin: 0; font-size: 26px; }
    .section { padding: 25px; border-bottom: 1px solid #e0e0e0; }
    .section:last-child { border-bottom: none; }
    .section h2 { font-size: 22px; color: #1a73e8; margin-top: 0; }
    .section h3 { font-size: 18px; color: #3c4043; margin-top: 20px; }
    .section p, .section li { font-size: 16px; line-height: 1.7; }
    .chart-container { text-align: center; margin: 25px 0; }
    .chart-container img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #5f6368; }
    .recommendations ul { list-style-type: none; padding-left: 0; }
    .recommendations li { background-color: #e8f0fe; border-left: 4px solid #4285f4; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    .scenarios-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    .scenarios-table th, .scenarios-table td { border: 1px solid #e0e0e0; padding: 12px; text-align: left; vertical-align: top; }
    .scenarios-table th { background-color: #f2f2f2; font-weight: bold; }
    """

    # --- New: Build Summary Table ---
    summary_table_html = ""
    if kpi_projections:
        avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
        conv_rate = config.get('conversion_rate_from_kpi_to_bo', 0)
        
        summary_table_html += '<div class="section">'
        summary_table_html += '<h2>Resumo dos Cenários Projetados</h2>'
        summary_table_html += '<p>A tabela abaixo apresenta os resultados projetados de quatro cenários de investimento chave, assumindo que seus respectivos mix recomendados sejam aplicados:</p>'
        summary_table_html += '<ul>'
        summary_table_html += '<li><strong>Cenário Atual (Média Histórica):</strong> Mantém o nível de investimento e o mix idênticos às médias observadas.</li>'
        summary_table_html += '<li><strong>Cenário Otimizado (Pico de Eficiência):</strong> Escala o investimento total para o ponto de maior eficiência detectado e usa o mix dos melhores períodos.</li>'
        summary_table_html += '<li><strong>Cenário Estratégico (Modelo de Elasticidade):</strong> Expande o orçamento até o limite ótimo de saturação calculado pelo modelo iterativo.</li>'
        summary_table_html += '<li><strong>Realocação Estratégica (Mesmo Orçamento):</strong> Mantém o investimento atual, mas redistribui a verba segundo o Modelo de Elasticidade para ganho puro de eficiência.</li>'
        summary_table_html += '</ul>'
        summary_table_html += '<table class="scenarios-table">'
        
        if avg_ticket > 0:
            header = '<thead><tr><th>Cenário</th><th>Investimento Mensal</th><th>Receita Projetada</th><th>Investimento Incremental</th><th>Receita Incremental</th><th>ROI Incremental</th></tr></thead><tbody>'
        else:
            kpi_label = config.get('primary_business_metric_name', 'KPIs')
            header = f'<thead><tr><th>Cenário</th><th>Investimento Mensal</th><th>Projeção de {kpi_label}</th><th>Investimento Incremental</th><th>{kpi_label} Incrementais</th><th>Custo por {kpi_label} Incremental</th></tr></thead><tbody>'
        
        summary_table_html += header
        
        scenario_map = [
            ('Cenário Atual (Média Histórica)', 'current'),
            ('Cenário Otimizado (Pico de Eficiência)', 'optimized'),
            ('Cenário Estratégico (Modelo de Elasticidade)', 'strategic'),
            ('Realocação Estratégica (Mesmo Orçamento)', 'reallocation')
        ]
        
        current_point = kpi_projections.get('current')
        current_inv = current_point.get('Daily_Investment', 0) * 30 if current_point else 0
        
        for title, key in scenario_map:
            point = kpi_projections.get(key)
            if point:
                inv = point.get('Daily_Investment', 0) * 30
                inc_inv = inv - current_inv if key != 'current' else 0
                
                kpi = point.get('Projected_Total_KPIs', 0) * 30
                inc_kpi = point.get('Incremental_KPI', 0) * 30 if key != 'current' else 0
                
                orders = kpi * conv_rate if conv_rate > 0 else kpi
                inc_orders = inc_kpi * conv_rate if conv_rate > 0 else inc_kpi
                
                inv_str = format_number(inv, currency=True)
                inc_inv_str = format_number(inc_inv, currency=True) if key != 'current' else '-'
                
                row_style = ' style="background-color: #e8f0fe;"' if key == 'strategic' else ''
                
                if avg_ticket > 0:
                    revenue = orders * avg_ticket
                    inc_revenue = inc_orders * avg_ticket
                    roi = (inc_revenue - inc_inv) / inc_inv if inc_inv > 0 else 0
                    
                    rev_str = format_number(revenue, currency=True)
                    inc_rev_str = format_number(inc_revenue, currency=True) if key != 'current' else '-'
                    roi_str = f"{roi:.2f}" if key != 'current' and inc_inv > 0 else '-'
                    
                    summary_table_html += f"<tr{row_style}><td><strong>{title}</strong></td><td>{inv_str}</td><td>{rev_str}</td><td>{inc_inv_str}</td><td>{inc_rev_str}</td><td>{roi_str}</td></tr>"
                else:
                    kpi_str = format_number(orders)
                    inc_kpi_str = format_number(inc_orders) if key != 'current' else '-'
                    
                    cpa_qty = inc_orders
                    cpa = inc_inv / cpa_qty if cpa_qty > 0 else 0
                    cpa_str = format_number(cpa, currency=True) if key != 'current' and cpa_qty > 0 else '-'
                    
                    summary_table_html += f"<tr{row_style}><td><strong>{title}</strong></td><td>{inv_str}</td><td>{kpi_str}</td><td>{inc_inv_str}</td><td>{inc_kpi_str}</td><td>{cpa_str}</td></tr>"
        
        summary_table_html += '</tbody></table></div>'
    # --- End New ---

    html_template = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>{report_title}</title>
        <style>
            {css_styles}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>{report_title}</h1></div>

            {summary_table_html}

            <div class="section">
                <h2>Sumário Executivo</h2>
                <p>{executive_summary}</p>
            </div>

            <div class="section">
                <h2>Análise da Curva de Resposta Global</h2>
                <p>O gráfico abaixo ilustra a relação entre o investimento total de marketing e o retorno projetado em KPIs. Ele nos ajuda a identificar os pontos ótimos de investimento para maximizar a eficiência e o crescimento.</p>
                <div class="chart-container"><img src="data:image/png;base64,{response_curve_img}" alt="Gráfico da Curva de Resposta Global"></div>
            </div>

            <div class="section">
                <h2>Análise Comparativa dos Cenários de Investimento</h2>
                {channel_mix_html}
                <p>{scenarios_intro}</p>
                {scenarios_analysis_html}
            </div>

            <div class="section recommendations">
                <h2>Recomendações Estratégicas</h2>
                <ul>
                    <li>{recommendation_1}</li>
                    <li>{recommendation_2}</li>
                </ul>
            </div>
            
            <div class="footer"><p>Relatório global gerado pela Max Impact Engine (Total Opportunity) com tecnologia Gemini.</p></div>
        </div>
    </body>
    </html>
    """.format(
        report_title=html.escape(narrative.get('report_title', f'Análise Estratégica Global para {advertiser_name}')),
        summary_table_html=summary_table_html,
        executive_summary=narrative.get('executive_summary', ''),
        scenarios_intro=narrative.get('analysis_of_scenarios', {}).get('introduction', ''),
        scenarios_analysis_html=scenarios_analysis_html,
        channel_mix_html=channel_mix_html,
        recommendation_1=narrative.get('strategic_recommendations', [{}])[0].get('recommendation', ''),
        recommendation_2=narrative.get('strategic_recommendations', [{}, {}])[1].get('recommendation', ''),
        response_curve_img=image_b64s.get('response_curve', ''),
        css_styles=css_styles
    )

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"   - Global Gemini HTML report saved successfully to: {output_filename}")
    except Exception as e:
        import traceback
        print(f"   - ERROR: Could not write global HTML report to file. Details: {e}")
        traceback.print_exc()
