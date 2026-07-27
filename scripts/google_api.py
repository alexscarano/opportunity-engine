# -*- coding: utf-8 -*-
import json
import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
import google.generativeai as genai


def authenticate_gemini(api_key=None, model_name=None):
    """Authenticates the Gemini client using an API key and selects the best available model."""
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(
            "ERROR: No Gemini API key found. Please set the GEMINI_API_KEY environment variable."
        )
        return None

    try:
        genai.configure(api_key=api_key)

        available_models = []
        try:
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)
            available_models = [
                m
                for m in available_models
                if not any(
                    dep in m for dep in ["-2.0-", "-1.5-", "gemini-2.0", "gemini-1.5"]
                )
            ]
        except Exception as list_err:
            print(
                f"Warning: Could not list models: {list_err}. Defaulting to fallbacks."
            )

        selected_model = model_name if model_name else "gemini-3.5-flash"

        if available_models:
            if model_name:
                matched = False
                for a_model in available_models:
                    if a_model == model_name or a_model.endswith(f"/{model_name}"):
                        selected_model = a_model
                        matched = True
                        break
                if not matched:
                    print(
                        f"Warning: Requested model '{model_name}' not found in available models. Selecting a priority fallback instead."
                    )
                    model_name = None

            if not model_name:
                priority = [
                    "gemini-3.5-flash",
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3.1-flash-lite",
                ]
                selected_model = next(
                    (a for p in priority for a in available_models if a == p or a.endswith(f"/{p}")),
                    available_models[0],
                )

        gemini_client = genai.GenerativeModel(selected_model)
        return gemini_client
    except Exception as e:
        print(f"An unexpected error occurred during Gemini authentication: {e}")
        return None


def suggest_form_fields(model, performance_df_sample):
    """Pede ao Gemini para sugerir Coluna do KPI, se é monetário, e o
    Objetivo da Otimização a partir de uma amostra do CSV de performance.

    Levanta ValueError se a resposta vier ausente, malformada, ou falhar
    na validação contra as colunas reais da amostra.
    """
    columns = performance_df_sample.columns.tolist()
    sample_csv = performance_df_sample.head(5).to_csv(index=False)

    prompt = f"""Você está configurando uma análise de marketing. Aqui estão as
colunas disponíveis no arquivo de performance e uma amostra de linhas:

Colunas: {columns}

Amostra (CSV):
{sample_csv}

Responda APENAS com um JSON no formato exato abaixo, sem texto adicional e
sem markdown fences:

{{"kpi_column": "<nome exato de uma das colunas listadas acima>", "kpi_is_monetary": true ou false, "optimization_target": "CONVERSIONS" ou "REVENUE"}}

Regras:
- "kpi_column" deve ser o nome de uma métrica de resultado (ex: conversões,
  leads, vendas, receita), nunca uma coluna de data ou de texto categórico.
- "kpi_is_monetary" é true só se os valores da coluna parecerem moeda
  (decimais, valores altos e variáveis típicos de receita/faturamento).
- "optimization_target" é "REVENUE" só se "kpi_is_monetary" for true ou a
  coluna claramente representar receita/faturamento; caso contrário
  "CONVERSIONS".
"""

    response = model.generate_content(prompt)
    cleaned_response_text = (
        response.text.strip().replace("```json\n", "").replace("\n```", "").replace("```", "")
    )

    try:
        suggestion = json.loads(cleaned_response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"resposta da IA não é um JSON válido: {e}") from e

    kpi_column = suggestion.get("kpi_column")
    if not isinstance(kpi_column, str) or kpi_column not in columns:
        raise ValueError(
            f"coluna de KPI sugerida ('{kpi_column}') não existe nas colunas do arquivo"
        )

    kpi_is_monetary = suggestion.get("kpi_is_monetary")
    if isinstance(kpi_is_monetary, str):
        kpi_is_monetary = kpi_is_monetary.strip().lower() == "true"
    if not isinstance(kpi_is_monetary, bool):
        raise ValueError("'kpi_is_monetary' sugerido não é um booleano")

    optimization_target = suggestion.get("optimization_target")
    if optimization_target not in ("CONVERSIONS", "REVENUE"):
        raise ValueError(
            f"'optimization_target' sugerido ('{optimization_target}') não é "
            "'CONVERSIONS' nem 'REVENUE'"
        )

    return {
        "kpi_column": kpi_column,
        "kpi_is_monetary": kpi_is_monetary,
        "optimization_target": optimization_target,
    }
