import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import threading
import queue
import plotly.express as px
import plotly.graph_objects as go
import logging

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import setup_logging

# Same handlers as the engine (console + data/log/streamlit_app*.log). The
# console line stays plain-text so Google Cloud Logging can capture stdout.
setup_logging("streamlit_app")
log = logging.getLogger(__name__)
# Structured user-action tracking, kept on its own channel.
logger = logging.getLogger("opp_engine_tracker")

# Optional: keep logging for raw actions without email barriers, if desired later, but removing barrier logic here.

from PIL import Image

try:
    favicon = Image.open("Logos/DASH_CAIXA_POSITIVO.png")
except Exception:
    favicon = None

st.set_page_config(
    page_title="Opportunity Engine",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def get_available_gemini_models(gemini_key):
    """
    Lists available Gemini models that support generateContent and filters the top 5.
    """
    MODELS_INFO = {
        "gemini-3.5-flash": "Gemini 3.5 Flash (Mais rápido e inteligente, recomendado)",
        "gemini-2.5-flash": "Gemini 2.5 Flash (Rápido e econômico)",
        "gemini-2.5-pro": "Gemini 2.5 Pro (Raciocínio complexo, ideal para relatórios detalhados)",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite (Ultra rápido e leve)",
    }

    preferred_order = [
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.1-flash-lite",
    ]

    if not gemini_key:
        return preferred_order, MODELS_INFO

    try:
        import google.generativeai as genai

        genai.configure(api_key=gemini_key)
        api_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                api_models.append(name)

        # Filter out deprecated versions (1.5, 2.0)
        api_models = [
            m
            for m in api_models
            if not any(
                dep in m for dep in ["-2.0-", "-1.5-", "gemini-2.0", "gemini-1.5"]
            )
        ]

        if api_models:
            filtered = [m for m in preferred_order if m in api_models]
            others = [m for m in api_models if m not in filtered]
            combined = (filtered + others)[:5]
            if combined:
                return combined, MODELS_INFO
    except Exception:
        pass

    return preferred_order, MODELS_INFO


def escape_markdown_dollars(obj):
    """Recursively escapes all dollar signs in strings to prevent Streamlit LaTeX rendering bugs."""
    import re
    if isinstance(obj, str):
        return re.sub(r'(?<!\\)\$', r'\\$', obj)
    elif isinstance(obj, dict):
        return {k: escape_markdown_dollars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [escape_markdown_dollars(x) for x in obj]
    return obj


def should_render(lines_since_last_render, seconds_since_last_render, min_lines=25, min_seconds=0.4):
    """Throttle decision for the live subprocess-log panel: render if either
    enough new lines piled up or enough time passed since the last render."""
    return lines_since_last_render >= min_lines or seconds_since_last_render >= min_seconds


def _load_event_narrative(selected_dir):
    """Loads the narrative dict and metrics from JSON, parses HTML fallback, or parses Markdown fallback."""
    import glob
    import json
    import re

    # Initialize default structure
    res = {
        "report_title": "Análise de Impacto Causal",
        "executive_verdict": "",
        "detailed_analysis": "",
        "value_delivered": {
            "narrative": "",
            "methodology_narrative": ""
        },
        "next_steps": [],
        "metrics": {
            "incremental_investment": "N/D",
            "incremental_investment_str": "N/D",
            "business_impact_label": "KPI",
            "business_impact_value": "N/D",
            "efficiency_label": "Eficiência",
            "efficiency_value": "N/D",
            "r_squared": 0.0,
            "p_value": 0.0,
            "mape": 0.0,
            "mae": 0.0,
            "avg_ticket": 0.0,
            "conversion_rate": 0.0,
            "p_value_threshold": 0.05
        }
    }

    # 1. Check for JSON narrative
    json_files = glob.glob(os.path.join(selected_dir, "gemini_report_*.json"))
    json_files = [f for f in json_files if "global_report" not in f]
    if json_files:
        try:
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                res.update(data)
                # Ensure nested dicts and default keys are populated
                if "value_delivered" in data and isinstance(data["value_delivered"], dict):
                    res["value_delivered"].update(data["value_delivered"])
                if "metrics" in data and isinstance(data["metrics"], dict):
                    res["metrics"].update(data["metrics"])
                # Metrics render via st.metric(), which shows text as-is (no
                # markdown/LaTeX parsing) -- escaping them would print a
                # literal backslash in front of every "R$" on screen.
                escaped = escape_markdown_dollars(res)
                escaped["metrics"] = res["metrics"]
                return escaped
        except Exception as e:
            log.warning(f"Error loading JSON narrative: {e}", exc_info=True)

    # 2. Check for HTML narrative fallback
    html_files = glob.glob(os.path.join(selected_dir, "gemini_report_*.html"))
    html_files = [f for f in html_files if "global_report" not in f]
    if html_files:
        try:
            with open(html_files[0], "r", encoding="utf-8") as f:
                html_content = f.read()

            title_match = re.search(r"<title>(.*?)</title>", html_content)
            if title_match:
                res["report_title"] = title_match.group(1).strip()
            
            verdict_match = re.search(r'Veredito Executivo</h2>\s*<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
            if verdict_match:
                res["executive_verdict"] = verdict_match.group(1).strip()
            
            analysis_match = re.search(r'Análise Aprofundada e Eficiência</h2>\s*<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
            if analysis_match:
                res["detailed_analysis"] = analysis_match.group(1).strip()
            
            value_match = re.search(r'O Impacto Causal e Metodologia</h2>\s*<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
            if value_match:
                res["value_delivered"]["narrative"] = value_match.group(1).strip()
            
            methodology_match = re.search(r'A Metodologia Opcional</h3>\s*<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
            if methodology_match:
                res["value_delivered"]["methodology_narrative"] = methodology_match.group(1).strip()
            
            next_steps = []
            next_steps_section = re.search(r'Próximos Passos Estratégicos</h2>\s*<ul>(.*?)</ul>', html_content, re.DOTALL)
            if next_steps_section:
                items = re.findall(r'<li>(.*?)</li>', next_steps_section.group(1))
                for item in items:
                    step_match = re.search(r'<strong>(.*?):</strong>(.*)', item)
                    if step_match:
                        next_steps.append({
                            "step": step_match.group(1).strip(),
                            "description": step_match.group(2).strip()
                        })
                    else:
                        next_steps.append({
                            "step": "Recomendação",
                            "description": item.strip()
                        })
            res["next_steps"] = next_steps

            # Parse metrics from HTML lists
            m_inv = re.search(r'<li><strong>Investimento Incremental:</strong>\s*(.*?)</li>', html_content)
            if m_inv:
                res["metrics"]["incremental_investment_str"] = m_inv.group(1).strip()
                res["metrics"]["incremental_investment"] = m_inv.group(1).strip()
            
            m_lift = re.search(r'<li><strong>Lift Mensurável \((.*?)\):</strong>\s*(.*?)</li>', html_content)
            if m_lift:
                res["metrics"]["business_impact_label"] = m_lift.group(1).strip()
                res["metrics"]["business_impact_value"] = m_lift.group(2).strip()
            
            m_eff = re.search(r'<li><strong>(ROI Incremental|CPA Incremental):</strong>\s*(.*?)</li>', html_content)
            if m_eff:
                res["metrics"]["efficiency_label"] = m_eff.group(1).strip()
                res["metrics"]["efficiency_value"] = m_eff.group(2).strip()

            # Robust numeric pattern: handles negatives, decimals, scientific notation
            _NUM_RE = r'-?[\d]+(?:\.[\d]+)?(?:[eE][-+]?\d+)?'

            # Parse validation metrics — each isolated so one failure doesn't kill the rest
            r2_m = re.search(r'<li><strong>R-squared \(R²\):</strong>\s*(' + _NUM_RE + r')', html_content)
            if r2_m:
                try:
                    res["metrics"]["r_squared"] = float(r2_m.group(1))
                except ValueError:
                    pass
            p_val_m = re.search(r'<li><strong>P-value \(Significância do Lift\):</strong>\s*(' + _NUM_RE + r')', html_content)
            if p_val_m:
                try:
                    res["metrics"]["p_value"] = float(p_val_m.group(1))
                except ValueError:
                    pass
            mape_m = re.search(r'<li><strong>Mean Absolute Percentage Error \(MAPE\):</strong>\s*(' + _NUM_RE + r')%', html_content)
            if mape_m:
                try:
                    res["metrics"]["mape"] = float(mape_m.group(1))
                except ValueError:
                    pass

            # Parse assumptions
            ticket_m = re.search(r'<li><strong>Valor Médio por Venda \(Ticket Médio\):</strong>\s*(.*?)</li>', html_content)
            if ticket_m:
                try:
                    clean_t = re.sub(r'[^\d,\.]', '', ticket_m.group(1)).replace('.', '').replace(',', '.')
                    res["metrics"]["avg_ticket"] = float(clean_t)
                except (ValueError, AttributeError):
                    pass
            conv_m = re.search(r'<li><strong>Taxa de Conversão \(de KPI para Venda\):</strong>\s*(' + _NUM_RE + r')', html_content)
            if conv_m:
                try:
                    res["metrics"]["conversion_rate"] = float(conv_m.group(1))
                except ValueError:
                    pass
            p_thresh_m = re.search(r'<li><strong>Limiar de Significância Estatística \(p-value\):</strong>\s*(' + _NUM_RE + r')', html_content)
            if p_thresh_m:
                try:
                    res["metrics"]["p_value_threshold"] = float(p_thresh_m.group(1))
                except ValueError:
                    pass

            # Metrics render via st.metric() (no markdown/LaTeX parsing) --
            # leave them unescaped, see the JSON-path comment above.
            escaped = escape_markdown_dollars(res)
            escaped["metrics"] = res["metrics"]
            return escaped
        except Exception as e:
            log.warning(f"Error parsing HTML narrative: {e}", exc_info=True)

    # 3. Check for Markdown narrative fallback
    md_path = os.path.join(selected_dir, "RECOMMENDATIONS.md")
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            title_match = re.search(r"^#\s+(.*)", md_content)
            if title_match:
                res["report_title"] = title_match.group(1).strip()

            verdict_match = re.search(r"## Veredito Executivo\s*\n\s*\*\*(.*?)\*\*", md_content, re.DOTALL)
            if verdict_match:
                res["executive_verdict"] = verdict_match.group(1).strip()

            analysis_match = re.search(r"## Análise Aprofundada\s*\n\s*(.*?)\n\s*(?:##|$)", md_content, re.DOTALL)
            if analysis_match:
                res["detailed_analysis"] = analysis_match.group(1).strip()

            value_match = re.search(r"## O Impacto Causal e Valor Entregue\s*\n\s*(.*?)\n\s*(?:##|$)", md_content, re.DOTALL)
            if value_match:
                res["value_delivered"]["narrative"] = value_match.group(1).strip()

            next_steps = []
            next_steps_section = re.search(r"## Próximos Passos Estratégicos\s*\n\s*(.*)", md_content, re.DOTALL)
            if next_steps_section:
                items = re.findall(r"###\s+(.*?)\n\s*(.*?)(?=\n\s*(?:###|##|$))", next_steps_section.group(1), re.DOTALL)
                for step_title, step_desc in items:
                    next_steps.append({
                        "step": step_title.strip(),
                        "description": step_desc.strip()
                    })
            res["next_steps"] = next_steps

            # Parse metrics from markdown
            m_inv = re.search(r'-\s+\*\*Investimento Incremental:\*\*\s*(.*)', md_content)
            if m_inv:
                res["metrics"]["incremental_investment_str"] = m_inv.group(1).strip()
                res["metrics"]["incremental_investment"] = m_inv.group(1).strip()
            
            m_lift = re.search(r'-\s+\*\*(Receita Incremental|Pedidos Incrementais):\*\*\s*(.*)', md_content)
            if m_lift:
                res["metrics"]["business_impact_label"] = m_lift.group(1).strip()
                res["metrics"]["business_impact_value"] = m_lift.group(2).strip()
                
            m_eff = re.search(r'-\s+\*\*(ROI Incremental|CPA Incremental):\*\*\s*(.*)', md_content)
            if m_eff:
                res["metrics"]["efficiency_label"] = m_eff.group(1).strip()
                res["metrics"]["efficiency_value"] = m_eff.group(2).strip()

            # Metrics render via st.metric() (no markdown/LaTeX parsing) --
            # leave them unescaped, see the JSON-path comment above.
            escaped = escape_markdown_dollars(res)
            escaped["metrics"] = res["metrics"]
            return escaped
        except Exception as e:
            log.warning(f"Error parsing Markdown narrative: {e}", exc_info=True)

    return None


PREMIUM_CSS = """
<style>
    /* Default / Light mode styles */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Increase specificity with container prefix to avoid using !important */
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), 
                    border-color 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), 
                    box-shadow 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: var(--primary-color, #ff4b4b);
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.15);
    }
    
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] label,
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
    }
    
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] label,
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #5f6368;
        font-weight: 500;
    }
    
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color);
        font-weight: 700;
    }
    
    h1, h2, h3 {
        color: #202124;
    }
    
    .card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    .insight-box {
        background-color: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 15px;
        border-radius: 4px;
        margin-top: 15px;
        color: #1a73e8;
    }
    
    /* Hide the loading bar that appears during reruns/reconnects */
    [data-testid="stStatusWidget"] { display: none !important; }
    div[class*="stDecoration"] { display: none !important; }
    
    /* Smooth fade-in so the page appears gracefully instead of popping in */
    [data-testid="stAppViewContainer"] {
        animation: fadeIn 0.25s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    /* Theme-aware CSS overrides for Dark Mode */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: var(--background-color);
        }
        div[data-testid="stAppViewContainer"] [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        div[data-testid="stAppViewContainer"] [data-testid="stMetric"]:hover {
            border-color: var(--primary-color, #ff4b4b);
            box-shadow: 0 8px 16px rgba(255, 75, 75, 0.25);
        }
        div[data-testid="stAppViewContainer"] [data-testid="stMetric"] label,
        div[data-testid="stAppViewContainer"] [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--text-color);
            opacity: 0.8;
        }
        h1, h2, h3 {
            color: var(--text-color);
        }
        div[data-testid="stAppViewContainer"] .card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        div[data-testid="stAppViewContainer"] .insight-box {
            background-color: rgba(26, 115, 232, 0.1);
            color: var(--text-color);
        }
    }

    /* Target specific Streamlit theme classes/attributes if present */
    [data-theme="dark"] .main,
    html[data-theme="dark"] .main {
        background-color: var(--background-color);
    }
    
    [data-theme="dark"] [data-testid="stMetric"],
    html[data-theme="dark"] [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    [data-theme="dark"] [data-testid="stMetric"]:hover,
    html[data-theme="dark"] [data-testid="stMetric"]:hover {
        border-color: var(--primary-color, #ff4b4b);
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.25);
    }
    
    [data-theme="dark"] [data-testid="stMetric"] label,
    html[data-theme="dark"] [data-testid="stMetric"] label,
    [data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricLabel"],
    html[data-theme="dark"] [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-color);
        opacity: 0.8;
    }
    
    [data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3,
    html[data-theme="dark"] h1, html[data-theme="dark"] h2, html[data-theme="dark"] h3 {
        color: var(--text-color);
    }
    
    [data-theme="dark"] .card,
    html[data-theme="dark"] .card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    [data-theme="dark"] .insight-box,
    html[data-theme="dark"] .insight-box {
        background-color: rgba(26, 115, 232, 0.1);
        color: var(--text-color);
    }
    
    /* Logo styling */
    .logo-container {
        text-align: center;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .logo-container.side-by-side {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 20px !important;
        margin-bottom: 25px;
        padding: 10px 0;
    }
    .logo-item {
        flex: 0 1 auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .logo-divider {
        width: 1px;
        height: 35px;
        background-color: var(--st-border-color, rgba(128, 128, 128, 0.3)) !important;
    }

    /* Logo theme swap: initial guess before the JS below runs (also the fallback if JS is
       blocked). Overridden instantly client-side on theme toggle — see script below. */
    .logo-light { display: block; }
    .logo-dark { display: none; }
    @media (prefers-color-scheme: dark) {
        .logo-light { display: none; }
        .logo-dark { display: block; }
    }
</style>
<script>
(function() {
    function isAppDark() {
        var el = document.querySelector('[data-testid="stApp"]') || document.body;
        var bg = window.getComputedStyle(el).backgroundColor;
        var m = bg.match(/[\\d.]+/g);
        if (!m || m.length < 3) return null;
        var luminance = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
        return luminance < 128;
    }
    function applyLogoTheme() {
        var isDark = isAppDark();
        if (isDark === null) return;
        document.querySelectorAll('.logo-light').forEach(function(img) {
            img.style.display = isDark ? 'none' : 'block';
        });
        document.querySelectorAll('.logo-dark').forEach(function(img) {
            img.style.display = isDark ? 'block' : 'none';
        });
    }
    applyLogoTheme();
    // Streamlit re-injects its theme stylesheet on toggle, mutating the document head; watch
    // only that, not the whole body (which mutates constantly from normal app reruns/widgets).
    new MutationObserver(applyLogoTheme).observe(document.head, {
        childList: true, subtree: true, characterData: true
    });
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyLogoTheme);
    }
})();
</script>
"""


# Ensure we can import modules from the local scripts directory
import sys

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from db import (
    init_db,
    create_user,
    verify_user,
    add_user_project,
    get_user_projects,
    verify_project_ownership,
    delete_user_project,
    rename_user_project,
    get_user_api_key,
    update_user_api_key,
    create_session,
    get_session,
)
from streamlit_cookies_controller import CookieController
import importlib
import dashboard_charts
importlib.reload(dashboard_charts)
from dashboard_charts import (
    build_icpa_curve,
    build_revenue_roi_curve,
    build_channel_mix_evolution,
    build_channel_saturation_comparison,
    build_events_overview,
    build_accuracy_chart,
    build_causal_line_chart,
    build_investment_bar_chart,
    build_sessions_bar_chart,
    build_response_curve_individual,
    find_saturation_point,
    compute_incremental_cpa,
)
from data_preprocessor import (
    COLUMN_NAME_HINTS,
    guess_date_col,
    guess_channel_col,
    guess_investment_col,
    guess_trends_col,
    guess_kpi_col,
    read_csv_robust,
)
from google_api import suggest_form_fields

import base64

@st.cache_data(ttl=3600)
def load_logo_base64(filename):
    """
    Carrega uma imagem da pasta Logos e retorna como string base64.
    """
    logos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logos")
    filepath = os.path.join(logos_dir, filename)
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Erro ao carregar logo {filename}: {e}")
        return ""

# Carrega os logos necessários
logo_dash_light = load_logo_base64("DASH_POSITIVO (1).png")
logo_dash_dark = load_logo_base64("DASH_NEGATIVO.png")
logo_almap_light = load_logo_base64("AF_ALMAPBBDO_LOGO_FINAL FILIPE-01.png")
logo_almap_dark = load_logo_base64("AF_ALMAPBBDO_LOGO_FINAL FILIPE-02.png")

# st.context.theme.type é a única info de tema exposta pelo Streamlit (light/dark), mas pode
# vir None no primeiro load da sessão ou no rerun logo após o usuário trocar o tema
# (https://github.com/streamlit/streamlit/issues/11920). Quando None, o CSS acima decide via
# prefers-color-scheme; quando conhecido, o inline style abaixo força a variante correta.
_theme_type = getattr(st.context.theme, "type", None)
if _theme_type == "dark":
    _logo_light_style, _logo_dark_style = "display: none;", "display: block;"
elif _theme_type == "light":
    _logo_light_style, _logo_dark_style = "display: block;", "display: none;"
else:
    _logo_light_style, _logo_dark_style = "", ""


init_db()
# Restore session WITHOUT cookie component (synchronous, no rerun needed)
if "user_id" not in st.session_state:
    # 1. URL query param: ?sid=<token> — survives F5
    sid = st.query_params.get("sid")
    if sid:
        row = get_session(sid)
        if row:
            st.session_state["user_id"], st.session_state["username"] = row

if "user_id" not in st.session_state:
    # 2. HTTP cookie headers (Streamlit >= 1.38, works if cookies reach the server)
    try:
        _uid = st.context.cookies.get("user_id")
        _uname = st.context.cookies.get("username")
        if _uid and _uname:
            st.session_state["user_id"] = int(_uid)
            st.session_state["username"] = _uname
    except Exception:
        pass

# If already authenticated, stub 'cookies' in session state BEFORE creating CookieController.
# CookieController.__init__ only calls _cookie_controller() (renders the async JS component)
# when 'cookies' is NOT in session state. The component then fires setComponentValue() which
# triggers an extra rerun — causing the visible flicker. Stubbing prevents the component call.
if "user_id" in st.session_state and "cookies" not in st.session_state:
    st.session_state[
        "cookies"
    ] = {}  # ponytail: stub — CookieController takes else branch, no component rerun

# Initialize CookieController (needed for login cookie write and legacy cookie read fallback)
if "cookie_controller" not in st.session_state:
    st.session_state.cookie_controller = CookieController()
controller = st.session_state.cookie_controller

if "user_id" not in st.session_state:
    # 3. Cookie component fallback (async — component triggers its own rerun when it has data)
    cookie_user_id = controller.get("user_id")
    cookie_username = controller.get("username")
    if cookie_user_id and cookie_username:
        st.session_state["user_id"] = int(cookie_user_id)
        st.session_state["username"] = cookie_username
        st.rerun()

if "user_id" not in st.session_state:
    st.html(PREMIUM_CSS, unsafe_allow_javascript=True)
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 50px; margin-bottom: 20px;">
            <div class="logo-container side-by-side">
                <div class="logo-item">
                    <img src="data:image/png;base64,{logo_dash_light}" class="logo-light" style="max-height: 55px; width: auto; {_logo_light_style}" />
                    <img src="data:image/png;base64,{logo_dash_dark}" class="logo-dark" style="max-height: 55px; width: auto; {_logo_dark_style}" />
                </div>
                <div class="logo-divider"></div>
                <div class="logo-item">
                    <img src="data:image/png;base64,{logo_almap_light}" class="logo-light" style="max-height: 55px; width: auto; {_logo_light_style}" />
                    <img src="data:image/png;base64,{logo_almap_dark}" class="logo-dark" style="max-height: 55px; width: auto; {_logo_dark_style}" />
                </div>
            </div>
            <p style="color: #5f6368; font-size: 1.1rem; margin-top: 15px;">Por favor, faça o login para acessar a plataforma de Otimização de Oportunidades.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        auth_mode = st.radio("Selecione a ação", ["Login", "Cadastrar Novo Usuário"])
        username = st.text_input("Usuário", key="login_username")
        password = st.text_input("Senha", type="password", key="login_password")

        if auth_mode == "Login":
            if st.button("Entrar", width="stretch"):
                user_id = verify_user(username, password)
                if user_id:
                    st.session_state["user_id"] = user_id
                    st.session_state["username"] = username
                    token = create_session(user_id, username)
                    st.query_params["sid"] = token
                    controller.set("user_id", str(user_id), max_age=30 * 86400)
                    controller.set("username", username, max_age=30 * 86400)
                    st.success(f"Bem-vindo, {username}!")
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")
        else:
            if st.button("Cadastrar", width="stretch"):
                if not username or not password:
                    st.error("Preencha usuário e senha.")
                else:
                    try:
                        user_id = create_user(username, password)
                        st.session_state["user_id"] = user_id
                        st.session_state["username"] = username
                        token = create_session(user_id, username)
                        st.query_params["sid"] = token
                        controller.set("user_id", str(user_id), max_age=30 * 86400)
                        controller.set("username", username, max_age=30 * 86400)
                        st.success(f"Cadastro realizado! Bem-vindo, {username}!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    st.stop()

# Authenticated — inject CSS now
st.html(PREMIUM_CSS, unsafe_allow_javascript=True)


st.title("Opportunity Engine")
st.markdown(
    "Explore alocações de orçamento ótimas, preveja retornos de KPI e encontre interativamente seu cenário ideal."
)


try:
    import scripts.data_preprocessor as data_preprocessor
except ImportError:
    try:
        import data_preprocessor
    except ImportError:
        data_preprocessor = None

# Get projects associated with the logged-in user
db_projects = get_user_projects(st.session_state["user_id"])
project_options = {name: path for name, path in db_projects}

# Initialize session state for config path
if "active_config_path" not in st.session_state:
    if project_options:
        st.session_state["active_config_path"] = list(project_options.values())[0]
    else:
        st.session_state["active_config_path"] = ""

# Enforce IDOR check on load
if st.session_state.get("active_config_path"):
    if not verify_project_ownership(
        st.session_state["user_id"], st.session_state["active_config_path"]
    ):
        st.session_state["active_config_path"] = ""

# Sidebar logos
st.sidebar.markdown(
    f"""
    <div class="logo-container side-by-side">
        <div class="logo-item">
            <img src="data:image/png;base64,{logo_dash_light}" class="logo-light" style="max-height: 40px; width: auto; {_logo_light_style}" />
            <img src="data:image/png;base64,{logo_dash_dark}" class="logo-dark" style="max-height: 40px; width: auto; {_logo_dark_style}" />
        </div>
        <div class="logo-divider"></div>
        <div class="logo-item">
            <img src="data:image/png;base64,{logo_almap_light}" class="logo-light" style="max-height: 40px; width: auto; {_logo_light_style}" />
            <img src="data:image/png;base64,{logo_almap_dark}" class="logo-dark" style="max-height: 40px; width: auto; {_logo_dark_style}" />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar user card & Logout
st.sidebar.markdown(f"**Conectado como:** {st.session_state['username']}")
if st.sidebar.button("Sair/Logout", width="stretch"):
    # Remove server-side session
    sid = st.query_params.get("sid")
    if sid:
        from db import delete_session

        delete_session(sid)
    # Clear query param and cookies (best-effort — cookie may not be in component cache)
    st.query_params.clear()
    try:
        controller.remove("user_id")
        controller.remove("username")
    except Exception:
        pass
    for key in [
        "user_id",
        "username",
        "active_config_path",
        "_cookie_init_done",
        "cookies",
    ]:
        st.session_state.pop(key, None)
    st.rerun()
st.sidebar.markdown("---")

st.sidebar.header("Projetos Anteriores")
if project_options:
    option_keys = list(project_options.keys())
    current_index = 0
    for i, key in enumerate(option_keys):
        if (
            project_options[key].replace("\\", "/").strip()
            == st.session_state.get("active_config_path", "").replace("\\", "/").strip()
        ):
            current_index = i
            break

    selected_project = st.sidebar.selectbox(
        "Selecione um Projeto:", options=option_keys, index=current_index
    )
    if selected_project:
        st.session_state["active_config_path"] = project_options[selected_project]

    # Project Rename UI
    if st.sidebar.button(
        "Renomear Projeto Atual", key="rename_proj_btn", width="stretch"
    ):
        st.session_state["show_rename_confirm"] = True

    if st.session_state.get("show_rename_confirm"):
        new_name = st.sidebar.text_input(
            "Novo nome:", value=selected_project, key="rename_proj_input"
        )
        col_ren1, col_ren2 = st.sidebar.columns(2)
        with col_ren1:
            if st.button("Salvar", key="confirm_rename_btn", width="stretch"):
                if rename_user_project(
                    st.session_state["user_id"], selected_project, new_name
                ):
                    st.toast(f"Projeto renomeado para '{new_name.strip()}'.")
                    st.session_state["show_rename_confirm"] = False
                    st.rerun()
                else:
                    st.error("Nome inválido ou já em uso.")
        with col_ren2:
            if st.button("Cancelar", key="cancel_rename_btn", width="stretch"):
                st.session_state["show_rename_confirm"] = False
                st.rerun()

    # Project Deletion UI
    if st.sidebar.button(
        "Excluir Projeto Atual", key="delete_proj_btn", width="stretch"
    ):
        st.session_state["show_delete_confirm"] = True

    if st.session_state.get("show_delete_confirm"):
        st.sidebar.warning(
            f"Deseja mesmo excluir o projeto '{selected_project}' e todos os seus arquivos?"
        )
        col_del1, col_del2 = st.sidebar.columns(2)
        with col_del1:
            if st.button(
                "Sim, Excluir", key="confirm_delete_btn", width="stretch"
            ):
                path_to_delete = st.session_state["active_config_path"]
                # Verify ownership to prevent IDOR on delete
                if verify_project_ownership(
                    st.session_state["user_id"], path_to_delete
                ):
                    import shutil
                    import json

                    # 1. Determine output directory from config file before deleting it
                    actual_output_dir = None
                    if os.path.exists(path_to_delete):
                        try:
                            with open(path_to_delete, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                            out_dir_base = cfg.get("output_directory", "outputs")
                            adv_name = cfg.get("advertiser_name", selected_project)
                            actual_output_dir = os.path.join(out_dir_base, adv_name)
                        except Exception:
                            pass

                    if not actual_output_dir:
                        actual_output_dir = os.path.join(
                            "outputs",
                            f"user_{st.session_state['user_id']}",
                            selected_project,
                        )

                    # 2. Delete input directory
                    project_dir = os.path.dirname(path_to_delete)
                    if os.path.exists(project_dir):
                        try:
                            shutil.rmtree(project_dir)
                        except Exception as e:
                            st.sidebar.error(
                                f"Erro ao excluir arquivos de entrada: {e}"
                            )

                    # 3. Delete output directory
                    if os.path.exists(actual_output_dir):
                        try:
                            shutil.rmtree(actual_output_dir)
                        except Exception as e:
                            st.sidebar.error(f"Erro ao excluir arquivos de saída: {e}")

                    # 4. Delete from SQLite
                    delete_user_project(st.session_state["user_id"], selected_project)

                    st.toast(f"Projeto '{selected_project}' excluído com sucesso.")
                    st.session_state["active_config_path"] = ""
                    st.session_state["show_delete_confirm"] = False
                    st.rerun()
                else:
                    st.error("Erro: Acesso não autorizado.")
        with col_del2:
            if st.button("Cancelar", key="cancel_delete_btn", width="stretch"):
                st.session_state["show_delete_confirm"] = False
                st.rerun()
else:
    st.sidebar.info("Nenhum projeto encontrado. Faça o setup de um novo.")


tab1, tab2, tab3 = st.tabs(
    ["Setup & Execução", "Dashboard de Impacto Causal", "Dashboard de Elasticidade"]
)

with tab1:
    st.header("Configuração de Nova Análise")
    st.markdown(
        "Faça o upload dos seus dados e configure os parâmetros financeiros para rodar um novo Motor de Oportunidades."
    )

    if st.session_state.pop("show_run_success_balloons", False):
        st.success("Análise Causal e Otimização concluídas com sucesso!")
        st.balloons()
        st.info(
            "Os dados foram gerados! Explore as abas de Impacto Causal e Elasticidade."
        )

    with st.form("setup_form"):
        st.markdown(
            """
            <style>
            div[data-testid="stColumn"]:has(.col-divider-anchor) {
                border-right: 1px solid rgba(128, 128, 128, 0.3);
                padding-right: 2rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                '<span class="col-divider-anchor"></span>', unsafe_allow_html=True
            )
            st.subheader("Configurações Gerais")
            advertiser_name = st.text_input(
                "Nome do Projeto",
                value="Meu_Projeto_Dynamic",
                help="Usado para nomear as pastas de resultados.",
            )
            # Fetch saved API Key from SQLite
            saved_key = get_user_api_key(st.session_state["user_id"]) or ""
            gemini_key = st.text_input(
                "Chave de API do Gemini",
                value=saved_key,
                type="password",
            )

            env_key = os.environ.get("GEMINI_API_KEY")
            active_key = gemini_key if gemini_key else env_key
            model_options, models_info = get_available_gemini_models(active_key)

            gemini_model = st.selectbox(
                "Modelo do Gemini",
                options=model_options,
                format_func=lambda x: models_info.get(x, x),
                help="Gera só o texto narrativo dos relatórios (o 'Veredito Executivo' e as "
                "recomendações em português) -- não muda nenhum número, gráfico ou cálculo. "
                "Modelos 'Pro' escrevem análises mais elaboradas e são mais lentos/caros; "
                "'Flash' é mais rápido e mais barato.",
            )

            st.divider()
            st.subheader("Parâmetros do Negócio")
            kpi_column = st.text_input(
                "Coluna do KPI no CSV de Performance",
                value=st.session_state.get("ai_suggested_kpi_column", "Sessions"),
                help="Nome exato da coluna (ex: Sessions, Conversions, Leads).",
            )
            kpi_is_monetary = st.checkbox(
                "O KPI já está em R$ (ex: Faturamento, Receita)",
                value=st.session_state.get("ai_suggested_kpi_is_monetary", False),
                help="Marque se a coluna de KPI já é o valor monetário final (não uma contagem "
                "de conversões/leads). Isso ignora Taxa de Conversão e Ticket Médio abaixo, e "
                "troca CPA/iCPA por ROAS/iROAS no dashboard.",
            )
            _opt_labels = [
                "Maximizar Volume de Conversões (Leads, Vendas, etc.)",
                "Maximizar Receita / Faturamento (Revenue)",
            ]
            _suggested_target = st.session_state.get("ai_suggested_optimization_target")
            optimization_target_label = st.selectbox(
                "Objetivo da Otimização",
                options=_opt_labels,
                index=1 if _suggested_target == "REVENUE" else 0,
                help="Define o que a aba Elasticidade tenta maximizar ao recomendar como dividir "
                "a verba entre canais. 'Volume' busca o maior número de conversões; 'Receita' usa "
                "o Ticket Médio para buscar o maior faturamento. Também decide se o dashboard "
                "mostra CPA/iCPA (custo por conversão) ou ROAS/iROAS (retorno por real investido).",
            )
            optimization_target = (
                "CONVERSIONS"
                if "Conversões" in optimization_target_label
                else "REVENUE"
            )

            conversion_rate = (
                st.number_input(
                    "Taxa de Conversão do KPI (%)",
                    value=1.0,
                    step=0.1,
                    help="Percentual do KPI que de fato vira venda. Ex: KPI = 'Leads' e 20% deles "
                    "fecham negócio → use 20%. Multiplica direto a Receita e o ROI/ROAS de todos "
                    "os relatórios -- errar aqui infla ou reduz artificialmente o resultado "
                    "financeiro inteiro. Deixe 100% se o KPI já for a venda final. Ignorado se "
                    "'O KPI já está em R\\$' acima estiver marcado.",
                )
                / 100.0
            )
            avg_ticket = st.number_input(
                "Ticket Médio (R$)",
                value=100.0,
                step=10.0,
                help="Valor médio (R\\$) de cada venda/conversão. Receita = KPI × Taxa de Conversão "
                "× Ticket Médio. Deixar este valor errado (ex: 1.00 sem ajustar) faz os "
                "relatórios mostrarem 'R\\$' que na verdade são só a contagem bruta do KPI, não "
                "dinheiro real. Ignorado se 'O KPI já está em R\\$' acima estiver marcado.",
            )
            if kpi_is_monetary:
                # ponytail: KPI is already money, conversion_rate/avg_ticket must be a no-op
                conversion_rate = 1.0
                avg_ticket = 1.0

            with st.expander("Configurações da Análise Causal"):
                min_pre_period_days = st.number_input(
                    "Dias Mínimos Pré-Evento",
                    min_value=7,
                    max_value=90,
                    value=14,
                    help="Dias de histórico exigidos antes do evento pra treinar o modelo causal "
                    "(padrão: 14, reduzido de 30). Menos dias deixa mais eventos serem analisados, "
                    "mas com um modelo mais instável e um R² menos confiável; mais dias analisa "
                    "menos eventos, só os que têm histórico suficiente, mas com estimativas mais "
                    "robustas.",
                )
                r_squared_threshold = st.slider(
                    "Ajuste Mínimo do Modelo (R²)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.30,
                    step=0.05,
                    help="Nota de qualidade do modelo causal: 1.0 = previsão perfeita, 0 = tão bom "
                    "quanto chutar a média histórica, negativo = pior que chutar a média (sinal de "
                    "que a estimativa de impacto não é confiável). Só é aplicado se 'Exigir Ajuste "
                    "Mínimo do Modelo' abaixo estiver marcado. Subir o valor rejeita mais eventos "
                    "como 'não confiáveis'; descer aceita eventos com previsões mais fracas.",
                )
                p_value_threshold = st.slider(
                    "Significância Máxima (p-value)",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.10,
                    step=0.01,
                    help="Um evento só passa se a chance de o resultado ser coincidência (ruído) "
                    "for menor que este valor. Só é aplicado se 'Exigir Significância Estatística' "
                    "abaixo estiver marcado. Valores maiores (ex: 0.10) deixam passar mais eventos, "
                    "inclusive alguns menos confiáveis; valores menores (ex: 0.01) são mais "
                    "rigorosos e aprovam menos eventos.",
                )
                increase_threshold_percent = st.number_input(
                    "Var. Mínima de Aumento de Investimento (%)",
                    min_value=1,
                    max_value=100,
                    value=40,
                    help="Quanto o investimento semanal precisa subir (vs. média das 12 semanas "
                    "anteriores) pra ser detectado como um 'pico' a analisar. Valor baixo encontra "
                    "mais eventos candidatos, inclusive picos pequenos/ruído; valor alto só "
                    "detecta aumentos bem grandes de verba.",
                )
                decrease_threshold_percent = st.number_input(
                    "Var. Mínima de Queda de Investimento (%)",
                    min_value=1,
                    max_value=100,
                    value=30,
                    help="Mesma lógica do campo acima, mas pra quedas de investimento: quanto a "
                    "verba semanal precisa cair (vs. média das 12 semanas anteriores) pra virar um "
                    "evento de 'corte de verba' a analisar.",
                )
                require_statistical_significance = st.checkbox(
                    "Exigir Significância Estatística (p-value)",
                    value=True,
                    help="Se marcado, descarta eventos cujo resultado pode ser só coincidência "
                    "(p-value acima do limite definido acima). Desmarcar deixa passar eventos sem "
                    "nenhuma confiança estatística -- útil só pra explorar dados com pouco "
                    "histórico, não recomendado pra decisões reais de verba.",
                )
                require_logical_direction = st.checkbox(
                    "Exigir Direção Lógica do Impacto",
                    value=True,
                    help="Se marcado, descarta eventos em que investimento e resultado andaram em "
                    "direções opostas (ex: a verba caiu, mas o relatório atribui um 'ganho' a "
                    "ela) -- esses casos são estatisticamente incoerentes. Desmarcar pode gerar "
                    "relatórios com Investimento Incremental negativo e receita incremental "
                    "positiva ao mesmo tempo, o que é confuso e não deve virar recomendação de "
                    "negócio.",
                )
                require_model_fit = st.checkbox(
                    "Exigir Ajuste Mínimo do Modelo (R²)",
                    value=True,
                    help="Se marcado, descarta eventos cujo modelo causal tem ajuste (R²) abaixo "
                    "do limite definido acima -- ou seja, a previsão de 'o que teria acontecido "
                    "sem o evento' não é confiável o bastante pra calcular a diferença. Desmarcar "
                    "permite que eventos com modelo ruim (R² negativo, pior que chutar a média) "
                    "ainda apareçam no relatório como se tivessem sido validados.",
                )
                investment_limit_factor = st.slider(
                    "Limite de Investimento Simulado (Elasticidade)",
                    min_value=1.5,
                    max_value=5.0,
                    value=1.5,
                    step=0.5,
                    help="Até quantas vezes o gasto médio atual a curva de resposta (aba Elasticidade) "
                    "é simulada. Se o 'Cenário de Saturação' sempre empatar com o 'Ponto Recomendado', "
                    "aumente este limite -- o teto real pode estar além do que 1.5x consegue mostrar.",
                )

        with col2:
            st.subheader("Dados Brutos (CSV)")
            inv_file = st.file_uploader(
                "Investimento (obrigatório)",
                type=["csv"],
                help="Investimento por canal de mídia (diário, semanal ou mensal).",
            )
            perf_file = st.file_uploader(
                "Performance (obrigatório)",
                type=["csv"],
                help="Histórico de resultados/KPIs (diário, semanal ou mensal).",
            )
            trends_file = st.file_uploader(
                "Tendências (opcional)",
                type=["csv"],
                help="Variável de controle (ex: Google Trends).",
            )

            st.divider()
            st.subheader("Restrições de Eficiência (Opcional)")
            target_cpa = st.number_input(
                "CPA Máximo (R$)",
                value=0.0,
                help="Custo máximo aceitável por conversão. Níveis de investimento na aba "
                "Elasticidade que ultrapassem este CPA ficam de fora da faixa considerada válida "
                "pra recomendar verba -- o 'Cenário Estratégico' nunca vai sugerir gastar tanto a "
                "ponto de passar deste teto. 0.00 = sem restrição, considera qualquer nível.",
            )
            target_roas = st.number_input(
                "ROAS Mínimo",
                value=0.0,
                help="Retorno mínimo aceitável por real investido (Receita ÷ Investimento). Ex: "
                "2.5 = R\\$2,50 de volta pra cada R\\$1,00 investido. Níveis de investimento com ROAS "
                "abaixo deste valor ficam de fora da faixa considerada válida pra recomendar "
                "verba. 0.00 = sem restrição. Só é usado quando o KPI é monetário ou o Objetivo "
                "da Otimização é 'Receita'.",
            )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            submit_btn = st.form_submit_button(
                "Construir Motor de Oportunidades",
                type="primary",
                width="stretch",
            )
        with col_btn2:
            save_settings_btn = st.form_submit_button(
                "Salvar Configurações", type="secondary", width="stretch"
            )
        with col_btn3:
            ai_suggest_btn = st.form_submit_button(
                "Sugerir com IA",
                type="secondary",
                width="stretch",
                help=(
                    "Analisa os arquivos enviados e sugere: Coluna do KPI, se o "
                    "KPI já está em R$, e o Objetivo da Otimização. Não sugere "
                    "Ticket Médio, Taxa de Conversão nem os limites da Análise "
                    "Causal -- esses continuam manuais. Decide só pelos nomes de "
                    "coluna e uma amostra de linhas do CSV de Performance; requer "
                    "Chave de API do Gemini preenchida."
                ),
            )

    if save_settings_btn:
        update_user_api_key(st.session_state["user_id"], gemini_key)
        st.success("Configurações salvas com sucesso!")

    if ai_suggest_btn:
        if not inv_file or not perf_file:
            st.error(
                "Por favor, faça upload dos arquivos de Investimento e Performance para continuar."
            )
        elif not (gemini_key or os.environ.get("GEMINI_API_KEY")) or not gemini_model:
            st.warning(
                "Preencha a Chave de API do Gemini para usar a sugestão automática."
            )
        else:
            with st.spinner("Analisando arquivos com IA..."):
                try:
                    active_key = gemini_key or os.environ.get("GEMINI_API_KEY")

                    safe_adv_name = (
                        advertiser_name.replace(" ", "_").replace("/", "").replace("\\", "")
                    )
                    dynamic_dir = os.path.join(
                        "inputs",
                        f"user_{st.session_state['user_id']}",
                        f"{safe_adv_name}_dynamic",
                    )
                    os.makedirs(dynamic_dir, exist_ok=True)
                    perf_path = os.path.join(dynamic_dir, "performance.csv")
                    with open(perf_path, "wb") as f:
                        f.write(perf_file.getbuffer())

                    performance_sample = read_csv_robust(perf_path, nrows=5)

                    import google.generativeai as genai

                    genai.configure(api_key=active_key)
                    model = genai.GenerativeModel(gemini_model)
                    suggestion = suggest_form_fields(model, performance_sample)

                    st.session_state["ai_suggested_kpi_column"] = suggestion["kpi_column"]
                    st.session_state["ai_suggested_kpi_is_monetary"] = suggestion["kpi_is_monetary"]
                    st.session_state["ai_suggested_optimization_target"] = suggestion["optimization_target"]
                    st.rerun()
                except Exception as e:
                    st.warning(
                        f"Não foi possível gerar sugestão automática: {e}. "
                        "Mantendo os valores atuais."
                    )

    if submit_btn:
        # Persist user's Gemini API Key in SQLite
        update_user_api_key(st.session_state["user_id"], gemini_key)

        if not inv_file or not perf_file:
            st.error(
                "Por favor, faça upload dos arquivos de Investimento e Performance para continuar."
            )
        else:
            with st.spinner("Preparando arquivos e gerando configuração..."):
                import os
                import subprocess

                safe_adv_name = (
                    advertiser_name.replace(" ", "_").replace("/", "").replace("\\", "")
                )
                dynamic_dir = os.path.join(
                    "inputs",
                    f"user_{st.session_state['user_id']}",
                    f"{safe_adv_name}_dynamic",
                )
                os.makedirs(dynamic_dir, exist_ok=True)

                inv_path = os.path.join(dynamic_dir, "investment.csv")
                perf_path = os.path.join(dynamic_dir, "performance.csv")
                with open(inv_path, "wb") as f:
                    f.write(inv_file.getbuffer())
                with open(perf_path, "wb") as f:
                    f.write(perf_file.getbuffer())

                trends_path = ""
                if trends_file:
                    trends_path = os.path.join(dynamic_dir, "trends.csv")
                    with open(trends_path, "wb") as f:
                        f.write(trends_file.getbuffer())

                inv_date = guess_date_col(inv_path)
                perf_date = guess_date_col(perf_path)
                trends_date = guess_date_col(trends_path) if trends_path else "Day"

                inv_channel = guess_channel_col(inv_path)
                inv_investment = guess_investment_col(inv_path)
                perf_kpi = guess_kpi_col(perf_path, kpi_column)
                trends_col = (
                    guess_trends_col(trends_path) if trends_path else "Ad Opportunities"
                )

                dynamic_config = {
                    "gemini_model": gemini_model,
                    "advertiser_name": f"{safe_adv_name}_dynamic",
                    "client_industry": "Dynamic Execution",
                    "client_business_goal": "Optimize through Streamlit",
                    "primary_business_metric_name": perf_kpi,
                    "investment_file_path": inv_path,
                    "performance_file_path": perf_path,
                    "generic_trends_file_path": trends_path if trends_path else None,
                    "output_directory": f"outputs/user_{st.session_state['user_id']}/",
                    "performance_kpi_column": perf_kpi,
                    "average_ticket": avg_ticket,
                    "conversion_rate_from_kpi_to_bo": conversion_rate,
                    "kpi_is_monetary": kpi_is_monetary,
                    "financial_targets": {
                        "target_cpa": target_cpa if target_cpa > 0 else 999999,
                        "target_icpa": 999999,
                        # ROAS only makes sense when the KPI is money (kpi_is_monetary)
                        # or the goal is REVENUE -- otherwise there's no
                        # Projected_Revenue to compute ROAS against, so keep it
                        # unset (0) even if the user typed a value in the field.
                        "target_roas": (
                            target_roas
                            if target_roas > 0
                            and (kpi_is_monetary or optimization_target == "REVENUE")
                            else 0
                        ),
                        "target_iroas": 0,
                    },
                    "optimization_target": optimization_target,
                    "investment_limit_factor": investment_limit_factor,
                    "p_value_threshold": p_value_threshold,
                    "r_squared_threshold": r_squared_threshold,
                    "increase_threshold_percent": increase_threshold_percent,
                    "decrease_threshold_percent": decrease_threshold_percent,
                    "min_pre_period_days": min_pre_period_days,
                    "require_statistical_significance": require_statistical_significance,
                    "require_logical_direction": require_logical_direction,
                    "require_model_fit": require_model_fit,
                    "post_event_days": 14,
                    "max_events_to_analyze": 3,
                    "treat_outliers": False,
                    "date_formats": {
                        "investment_file": None,
                        "performance_file": None,
                        "generic_trends_file": None,
                    },
                    "column_mapping": {
                        "investment_file": {
                            "date_col": inv_date,
                            "channel_col": inv_channel,
                            "investment_col": inv_investment,
                        },
                        "performance_file": {
                            "date_col": perf_date,
                            "kpi_col": perf_kpi,
                        },
                        "generic_trends_file": {
                            "date_col": trends_date,
                            "trends_col": trends_col,
                        },
                    },
                }

                config_path_gen = os.path.join(dynamic_dir, "config_dynamic.json")
                with open(config_path_gen, "w", encoding="utf-8") as f:
                    json.dump(dynamic_config, f, indent=4)

                st.session_state["active_config_path"] = config_path_gen
                add_user_project(
                    st.session_state["user_id"],
                    f"{safe_adv_name}_dynamic",
                    config_path_gen,
                )

            st.success(
                "Configuração salva! Iniciando a engine Causais + Elasticidade..."
            )

            logger.info(
                json.dumps(
                    {
                        "event": "Execution Run",
                        "project": advertiser_name,
                        "kpi": kpi_column,
                    }
                )
            )

            log_container = st.empty()
            status_container = st.empty()
            env = os.environ.copy()
            if gemini_key:
                env["GEMINI_API_KEY"] = gemini_key
            env["PYTHONPATH"] = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"

            import sys

            python_bin = sys.executable

            if gemini_key:
                target_main_script = os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    "scripts",
                    "local_main.py",
                )
            else:
                target_main_script = os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    "scripts",
                    "local_main-without-gemini.py",
                )
            target_config_path = os.path.abspath(
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    config_path_gen,
                )
            )

            process = subprocess.Popen(
                [python_bin, target_main_script, "--config", target_config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                env=env,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            )

            log_lines = []
            lines_since_render = 0
            last_render_time = time.monotonic()
            last_line_time = time.monotonic()

            def render_log():
                shown = log_lines[-300:]
                header = f"Engine de Oportunidades Rodando... ({len(log_lines)} linhas"
                if len(log_lines) > len(shown):
                    header += f", exibindo as últimas {len(shown)}"
                header += ")"
                idle = time.monotonic() - last_line_time
                if idle >= 3:
                    header += f" -- etapa em andamento há {int(idle)}s"
                log_container.code(
                    header + "\n" + "\n".join(shown),
                    language="shell",
                    height=400,
                )

            # ponytail: o process.wait(timeout=900) anterior era código morto -- quando
            # o engine trava (ex.: retry silencioso de 503 do Gemini) ele trava DENTRO
            # do readline abaixo e nunca chega no wait(). O watchdog mata o processo,
            # o readline recebe EOF e o loop termina.
            killed = threading.Event()

            def _kill_on_timeout():
                killed.set()
                process.kill()

            watchdog = threading.Timer(900, _kill_on_timeout)
            watchdog.daemon = True
            watchdog.start()

            # ponytail: readline bloqueia, e o painel só era redesenhado quando uma
            # linha NOVA chegava. Numa pausa longa (chamada ao Gemini) a tela
            # congelava sem ter mostrado as últimas linhas lidas -- é isso que
            # parecia travamento. A thread lê o pipe e o loop principal redesenha
            # também quando a fila fica quieta, mostrando há quanto tempo espera.
            line_queue = queue.Queue()

            def _pump_stdout():
                for line in iter(process.stdout.readline, ""):
                    line_queue.put(line)
                line_queue.put(None)

            threading.Thread(target=_pump_stdout, daemon=True).start()

            try:
                while True:
                    try:
                        line = line_queue.get(timeout=1.0)
                    except queue.Empty:
                        render_log()
                        last_render_time = time.monotonic()
                        lines_since_render = 0
                        continue
                    if line is None:
                        break
                    log_lines.append(line.rstrip("\n"))
                    last_line_time = time.monotonic()
                    lines_since_render += 1
                    now = time.monotonic()
                    if should_render(lines_since_render, now - last_render_time):
                        render_log()
                        last_render_time = now
                        lines_since_render = 0
            finally:
                watchdog.cancel()

            process.stdout.close()
            render_log()  # final render: guarantee the last lines are shown

            return_code = process.wait()
            timed_out = killed.is_set()

            if timed_out:
                status_container.error(
                    "A execução do motor excedeu o limite de 15 minutos (900 "
                    "segundos) e foi interrompida.\n\nÚltimas linhas capturadas:\n"
                    + "\n".join(log_lines[-300:])
                )
            elif return_code == 0:
                st.session_state["active_config_path"] = config_path_gen
                st.session_state["show_run_success_balloons"] = True
                st.rerun()
            else:
                error_line = next(
                    (line for line in reversed(log_lines) if "ERRO" in line), None
                )
                msg = "Houve um erro na execução do motor."
                if error_line:
                    msg += f"\n\n{error_line.strip()}"
                msg += "\n\nVerifique os logs acima para mais detalhes."
                status_container.error(msg)

with tab2:
    st.header("Análise de Impacto Causal (Por Evento)")
    st.markdown(
        "Selecione um evento analisado abaixo para visualizar o relatório detalhado do Gemini avaliando o impacto causal deste pico de investimento."
    )

    if os.path.exists(st.session_state["active_config_path"]):
        import json

        with open(st.session_state["active_config_path"], "r", encoding="utf-8") as f:
            active_config = json.load(f)
        adv_name = active_config.get("advertiser_name", "default_advertiser")
        output_base = active_config.get("output_directory", "outputs").rstrip("/").rstrip("\\")
        adv_dir = os.path.join(output_base, adv_name)

        import glob

        html_reports = glob.glob(
            os.path.join(adv_dir, "**", "gemini_report_*.html"), recursive=True
        )
        md_reports = glob.glob(
            os.path.join(adv_dir, "**", "RECOMMENDATIONS.md"), recursive=True
        )

        event_dirs = set()
        for r in html_reports:
            if "global_report.html" not in r:
                event_dirs.add(os.path.dirname(r))
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
                    width="stretch",
                )
                st.markdown("---")

            report_options = {}
            for d in event_dirs:
                parts = d.split(os.sep)
                if len(parts) >= 2:
                    channel = parts[-2]
                    date_event = parts[-1]
                    readable_name = (
                        f"Pico em {date_event} ({channel.replace('_', ', ')})"
                    )
                else:
                    readable_name = os.path.basename(d)
                report_options[readable_name] = d

            selected_report_name = st.selectbox(
                "Selecione o Evento:", list(report_options.keys())
            )
            selected_dir = report_options[selected_report_name]

            # Load narrative dict
            narrative = _load_event_narrative(selected_dir)

            csv_names = [
                "line_chart_data.csv",
                "accuracy_data.csv",
                "investment_data.csv",
                "sessions_data.csv",
            ]
            has_csvs = all(os.path.exists(os.path.join(selected_dir, f)) for f in csv_names)

            if narrative:
                # 1. Report Title
                st.markdown(f"## {narrative.get('report_title', 'Análise de Impacto Causal')}")

                # 2. Executive Verdict
                verdict = narrative.get("executive_verdict", "")
                if verdict:
                    import html
                    safe_verdict = html.escape(verdict).replace(r'\$', '$')
                    st.markdown(
                        f"""
                        <div class="insight-box" style="margin-bottom: 25px;">
                            <h3 style="margin-top: 0; color: #1a73e8;">Veredito Executivo</h3>
                            <p style="font-size: 1.1rem; font-weight: 500; line-height: 1.6; margin: 0;">{safe_verdict}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 3. Metrics
                metrics = narrative.get("metrics", {})
                if metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Investimento Incremental", metrics.get("incremental_investment_str", "N/D"))
                    with col2:
                        st.metric(f"Lift ({metrics.get('business_impact_label', 'KPI')})", metrics.get("business_impact_value", "N/D"))
                    with col3:
                        st.metric(metrics.get("efficiency_label", "Eficiência"), metrics.get("efficiency_value", "N/D"))

                # 4. Detailed Analysis
                detailed_analysis = narrative.get("detailed_analysis", "")
                if detailed_analysis:
                    st.markdown("---")
                    st.subheader("Análise Aprofundada e Eficiência")
                    st.write(detailed_analysis)

                # 5. Causal Impact & Methodology
                value_delivered = narrative.get("value_delivered", {})
                value_narrative = value_delivered.get("narrative", "") if isinstance(value_delivered, dict) else ""
                methodology_narrative = value_delivered.get("methodology_narrative", "") if isinstance(value_delivered, dict) else ""

                if value_narrative:
                    st.markdown("---")
                    st.subheader("O Impacto Causal e Metodologia")
                    st.write(value_narrative)

                kpi_name = active_config.get("primary_business_metric_name", "kpi")

                # Causal Line Chart
                if has_csvs:
                    line_chart_df = pd.read_csv(os.path.join(selected_dir, "line_chart_data.csv"))
                    st.plotly_chart(
                        build_causal_line_chart(line_chart_df, kpi_name=kpi_name),
                        width="stretch",
                    )
                else:
                    # Fallback to PNG line chart
                    png_line_charts = glob.glob(os.path.join(selected_dir, "*line_chart*.png"))
                    if png_line_charts:
                        st.image(png_line_charts[0], caption="Gráfico Resumo (Impacto Causal)", width="stretch")

                # Bar charts
                if has_csvs:
                    col1, col2 = st.columns(2)
                    investment_chart_df = pd.read_csv(
                        os.path.join(selected_dir, "investment_data.csv"), index_col=0
                    )
                    sessions_chart_df = pd.read_csv(
                        os.path.join(selected_dir, "sessions_data.csv"), index_col=0
                    )
                    with col1:
                        st.plotly_chart(
                            build_investment_bar_chart(investment_chart_df),
                            width="stretch",
                        )
                    with col2:
                        st.plotly_chart(
                            build_sessions_bar_chart(sessions_chart_df, kpi_name=kpi_name),
                            width="stretch",
                        )
                else:
                    # Fallback to PNG bar charts
                    png_inv = glob.glob(os.path.join(selected_dir, "*investment*.png")) + glob.glob(os.path.join(selected_dir, "*cost*.png"))
                    png_sess = glob.glob(os.path.join(selected_dir, "*sessions*.png")) + glob.glob(os.path.join(selected_dir, "*kpi*.png"))
                    
                    if png_inv or png_sess:
                        col1, col2 = st.columns(2)
                        with col1:
                            if png_inv:
                                st.image(png_inv[0], caption="Pico de Investimento (Intervenção)", width="stretch")
                        with col2:
                            if png_sess:
                                st.image(png_sess[0], caption="Efeito Causal no KPI", width="stretch")

                if methodology_narrative:
                    st.write(methodology_narrative)

                # 6. Next steps
                next_steps = narrative.get("next_steps", [])
                if next_steps:
                    st.markdown("---")
                    st.subheader("Próximos Passos Estratégicos")
                    for step in next_steps:
                        if isinstance(step, dict):
                            st.markdown(f"- **{step.get('step', '')}:** {step.get('description', '')}")
                        else:
                            st.markdown(f"- {step}")

                # 7. Model Validation Appendix
                r_squared = metrics.get("r_squared", 0.0)
                p_value = metrics.get("p_value", 0.0)
                mape = metrics.get("mape", 0.0)
                
                st.markdown("---")
                st.subheader("Apêndice: Validação do Modelo Estatístico")
                st.write("A validade desta análise baseia-se na capacidade do modelo de prever com precisão o desempenho durante o período pré-evento. As métricas abaixo demonstram a robustez e a confiabilidade do modelo:")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R-squared (R²)", f"{r_squared:.2f}", help="Indica a variância explicada pelo modelo")
                with col2:
                    st.metric("P-value", f"{p_value:.4f}", help="Significância estatística do lift")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape), help="Erro percentual médio absoluto no período pré-evento")

                # Accuracy chart
                if has_csvs:
                    accuracy_chart_df = pd.read_csv(os.path.join(selected_dir, "accuracy_data.csv"))
                    st.plotly_chart(
                        build_accuracy_chart(accuracy_chart_df, kpi_name=kpi_name),
                        width="stretch",
                    )
                else:
                    # Fallback to PNG accuracy plot
                    png_acc = glob.glob(os.path.join(selected_dir, "*accuracy*.png"))
                    if png_acc:
                        st.image(png_acc[0], caption="Acurácia do Modelo Pré-Intervenção", width="stretch")

                # 8. Assumptions Appendix
                avg_ticket = metrics.get("avg_ticket", 0.0)
                conversion_rate = metrics.get("conversion_rate", 0.0)
                p_value_threshold = metrics.get("p_value_threshold", 0.05)
                
                st.markdown("---")
                st.subheader("Apêndice: Premissas da Análise")
                st.markdown(
                    f"""
                    - **Valor Médio por Venda (Ticket Médio):** R$ {avg_ticket:,.2f}
                    - **Taxa de Conversão (de KPI para Venda):** {conversion_rate:.4%}
                    - **Limiar de Significância Estatística (p-value):** {p_value_threshold}
                    """
                )
            else:
                st.warning("Nenhum relatório encontrado para este evento.")

        else:
            st.info(
                "Nenhum relatório de Impacto Causal encontrado. Rode o motor na aba Setup ou verifique as restrições abaixo."
            )
            cond_lines = [
                f'- Pico ou queda de investimento semanal de ao menos **+{active_config.get("increase_threshold_percent", 50)}%** ou **-{active_config.get("decrease_threshold_percent", 30)}%** vs. a média das últimas 12 semanas',
                f'- Pelo menos **{active_config.get("min_pre_period_days", 14)} dias** de dados antes do evento',
            ]
            if active_config.get("require_statistical_significance", True):
                cond_lines.append(f'- Significância estatística: **p-value < {active_config.get("p_value_threshold", 0.1)}**')
            if active_config.get("require_model_fit", True):
                cond_lines.append(f'- Ajuste do modelo: **R² ≥ {active_config.get("r_squared_threshold", 0.3)}**')
            if active_config.get("require_logical_direction", True):
                cond_lines.append('- Direção lógica (investimento sobe e KPI sobe, ou ambos caem)')

            cond_str = "\n".join(cond_lines)
            st.markdown(
                f"""
**Um evento só gera relatório se passar por todas as condições ativas:**
{cond_str}

Consulte o log de execução da aba Setup para ver o motivo exato de cada evento descartado.
"""
            )
    else:
        st.info("Configuração não encontrada. Faça o Setup para começar.")

with tab3:
    st.sidebar.markdown("---")

    def load_data(config_path):
        # Cache buster comment to force Streamlit to reload data
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            advertiser_name = config.get("advertiser_name", "default_advertiser")
            output_dir = os.path.join(
                config.get("output_directory", "outputs").rstrip("/"),
                advertiser_name,
                "global_saturation_analysis",
            )

            csv_path = os.path.join(output_dir, "response_curve_data.csv")
            if not os.path.exists(csv_path):
                return config, None, None, output_dir, 0.0, 0.0

            df = pd.read_csv(csv_path)

            narrative_path = os.path.join(output_dir, "global_narrative.json")
            narrative = {}
            if os.path.exists(narrative_path):
                with open(narrative_path, "r", encoding="utf-8") as f:
                    narrative = json.load(f)

            true_baseline_monthly_inv = 0.0
            true_baseline_monthly_kpi = 0.0

            if data_preprocessor is not None:
                try:
                    kpi_df, daily_investment_df, _, _ = (
                        data_preprocessor.load_and_prepare_data(config)
                    )
                    investment_pivot_df = daily_investment_df.pivot_table(
                        index="Date", columns="Product Group", values="investment"
                    ).fillna(0)

                    active_spend_cols = [
                        col
                        for col in investment_pivot_df.columns
                        if investment_pivot_df[col].mean() > 0 and col != "OTHER"
                    ]
                    total_avg_daily_spend = sum(
                        investment_pivot_df[col].mean() for col in active_spend_cols
                    )
                    # Monthly extrapolation factor: 30/period_days periods per
                    # month. Reduces to the old "* 30" for daily data
                    # (period_days=1); scales correctly for weekly/monthly
                    # cadences instead of overstating projections.
                    monthly_factor = 30 / (config.get("period_days", 1) or 1)
                    true_baseline_monthly_inv = total_avg_daily_spend * monthly_factor

                    if not df.empty:
                        closest_idx = (
                            (df["Daily_Investment"] - total_avg_daily_spend)
                            .abs()
                            .idxmin()
                        )
                        true_baseline_monthly_kpi = (
                            df.loc[closest_idx, "Projected_Total_KPIs_Historical"]
                            * monthly_factor
                        )
                    else:
                        true_baseline_monthly_kpi = (
                            kpi_df["kpi"].mean() * monthly_factor
                        )
                except Exception as e:
                    log.error(
                        f"Error during data_preprocessor in Streamlit: {e}",
                        exc_info=True,
                    )
                    st.warning(
                        f"Não foi possível calcular o baseline real a partir dos dados de origem: {e}"
                    )

            return (
                config,
                df,
                narrative,
                output_dir,
                true_baseline_monthly_inv,
                true_baseline_monthly_kpi,
            )
        except Exception as e:
            log.error(
                f"Error loading global saturation data in load_data: {e}", exc_info=True
            )
            return None, None, None, None, 0.0, 0.0

    if st.session_state["active_config_path"]:
        (
            config,
            df,
            narrative,
            output_dir,
            true_baseline_monthly_inv,
            true_baseline_monthly_kpi,
        ) = load_data(st.session_state["active_config_path"])

        if df is not None:
            kpi_name = config.get("primary_business_metric_name", "Transactions")
            kpi_is_monetary = config.get("kpi_is_monetary", False)
            # Kept the DAYS_IN_MONTH name (used in many places below) but the
            # value now reflects the detected reporting cadence: 30/period_days
            # periods per month, reducing to 30 for daily data (period_days=1).
            DAYS_IN_MONTH = 30 / (config.get("period_days", 1) or 1)
            df["Monthly_Investment"] = df["Daily_Investment"] * DAYS_IN_MONTH
            df["Monthly_KPI"] = df["Projected_Total_KPIs"] * DAYS_IN_MONTH
            baseline_monthly_inv = true_baseline_monthly_inv

            df["CPA"] = df["Daily_Investment"] / df["Projected_Total_KPIs"]
            df["CPA"] = df["CPA"].replace([np.inf, -np.inf], float("nan"))

            # compute_incremental_cpa (not a raw ratio + fillna(0)): points below
            # baseline have Incremental_Investment clipped to 0 upstream, so a raw
            # ratio reads as a literal "R$0.00 marginal cost" instead of N/A --
            # exactly the bug already fixed for the scenario table below.
            df["iCPA"] = compute_incremental_cpa(
                df["Incremental_Investment"], df["Incremental_KPI"]
            )

            if "Projected_Revenue" in df.columns:
                df["ROAS"] = (df["Projected_Revenue"] / df["Monthly_Investment"]).replace([np.inf, -np.inf], float("nan"))
                df["iROAS"] = (df["Incremental_Revenue"] / df["Incremental_Investment"]).replace([np.inf, -np.inf], float("nan")).fillna(0)

            df["Pct_Incrementality"] = np.where(
                df["Projected_Total_KPIs"] > 0,
                df["Incremental_KPI"] / df["Projected_Total_KPIs"] * 100,
                0.0,
            )

            st.sidebar.header("Filtros de Limitação")
            st.sidebar.markdown(
                "Cada filtro marcado abaixo remove pontos da curva de investimento simulada que "
                "não atendem ao critério. O **Ponto Recomendado** é sempre o ponto de **maior "
                "investimento** entre os que sobram -- filtros mais soltos (ou desmarcados) "
                "tendem a recomendar mais verba; filtros mais apertados recomendam menos."
            )

            max_inv_val = float(df["Monthly_Investment"].max())
            min_inv_val = float(df["Monthly_Investment"].min())

            # --- Orçamento ---
            st.sidebar.caption(
                "Faixa de investimento mensal considerada. Sempre ativo -- estreitar esta faixa "
                "é a forma mais direta de forçar um Ponto Recomendado maior ou menor."
            )
            budget_col1, budget_col2 = st.sidebar.columns(2)
            min_budget_millions = budget_col1.number_input(
                "Orçamento Mín. (M)",
                min_value=0.0,
                max_value=max_inv_val / 1e6,
                value=min_inv_val / 1e6,
                step=0.05,
                format="%.2f",
            )
            max_budget_millions = budget_col2.number_input(
                "Orçamento Máx. (M)",
                min_value=0.0,
                max_value=max_inv_val / 1e6,
                value=max_inv_val / 1e6,
                step=0.05,
                format="%.2f",
            )
            min_budget = min_budget_millions * 1e6
            max_budget = max_budget_millions * 1e6

            # --- CPA ---
            use_cpa_target = st.sidebar.checkbox(
                "Aplicar Limite de Target CPA",
                value=False,
                help="Descarta pontos da curva cujo custo por conversão (CPA = Investimento ÷ "
                "KPI) ultrapasse o valor abaixo. Usado quando o KPI não é monetário.",
            )
            target_cpa = None
            if use_cpa_target:
                max_cpa_val = float(df["CPA"].max()) if "CPA" in df.columns else 100.0
                if np.isnan(max_cpa_val):
                    max_cpa_val = 100.0
                target_cpa = st.sidebar.number_input(
                    "Target CPA Máximo",
                    min_value=0.0,
                    max_value=max_cpa_val * 2,
                    value=max_cpa_val * 0.5,
                    step=1.0,
                    format="%.2f",
                    help="Custo máximo aceitável por conversão. Diminuir este valor elimina os "
                    "níveis de investimento mais altos da curva primeiro (são os que custam mais "
                    "por conversão), puxando o Ponto Recomendado pra baixo.",
                )

            # --- iCPA ---
            use_icpa_target = st.sidebar.checkbox(
                "Aplicar Limite de iCPA Marginal",
                value=False,
                help="Descarta pontos onde o custo do PRÓXIMO real investido (iCPA, não o custo "
                "médio) ultrapassa o valor abaixo -- mais rigoroso que o CPA médio, pois pega "
                "onde a curva já está cara de escalar mesmo com CPA médio ainda razoável.",
            )
            target_icpa = None
            if use_icpa_target:
                max_icpa_val = float(df["iCPA"].max())
                if np.isnan(max_icpa_val) or max_icpa_val <= 0:
                    max_icpa_val = 1000.0
                target_icpa = st.sidebar.number_input(
                    "Marginal iCPA Máximo",
                    min_value=0.0,
                    max_value=max_icpa_val * 2,
                    value=max_icpa_val * 0.5,
                    step=1.0,
                    format="%.2f",
                    help="Custo marginal máximo aceitável pra continuar escalando o investimento.",
                )

            # --- ROAS ---
            use_roas = st.sidebar.checkbox(
                "Aplicar ROAS Mínimo",
                value=False,
                help="Descarta pontos cujo retorno médio (Receita ÷ Investimento) fique abaixo "
                "do valor abaixo. Usado quando o KPI é monetário ou o Objetivo é 'Receita'.",
            )
            min_roas = None
            if use_roas and "ROAS" in df.columns:
                roas_max = float(df["ROAS"].replace([np.inf, -np.inf], np.nan).dropna().max())
                if np.isnan(roas_max) or roas_max <= 0:
                    roas_max = 10.0
                roas_step = max(0.001, roas_max / 100)
                min_roas = st.sidebar.number_input(
                    "ROAS Mínimo",
                    min_value=0.0,
                    max_value=roas_max,
                    value=roas_max * 0.5,
                    step=roas_step,
                    format="%.3f",
                    help="Ex: 2.5 = R\\$2,50 de retorno pra cada R\\$1,00 investido, em média. Subir "
                    "este valor elimina os níveis de investimento mais altos primeiro (é onde o "
                    "retorno médio cai mais), puxando o Ponto Recomendado pra baixo.",
                )

            # --- iROAS ---
            use_iroas = st.sidebar.checkbox(
                "Aplicar iROAS Mínimo",
                value=False,
                help="Descarta pontos onde o retorno do PRÓXIMO real investido (iROAS, não a "
                "média) fica abaixo do valor abaixo -- mais rigoroso que o ROAS médio.",
            )
            min_iroas = None
            if use_iroas and "iROAS" in df.columns:
                iroas_max = float(df["iROAS"].replace([np.inf, -np.inf], np.nan).dropna().max())
                if np.isnan(iroas_max) or iroas_max <= 0:
                    iroas_max = 10.0
                iroas_step = max(0.001, iroas_max / 100)
                min_iroas = st.sidebar.number_input(
                    "iROAS Mínimo",
                    min_value=0.0,
                    max_value=iroas_max,
                    value=iroas_max * 0.5,
                    step=iroas_step,
                    format="%.3f",
                    help="Retorno marginal mínimo aceitável pra continuar escalando o "
                    "investimento.",
                )

            # --- KPI Mínimo ---
            use_min_kpi = st.sidebar.checkbox(
                "Aplicar KPI Mínimo",
                value=False,
                help=f"Descarta pontos que projetam menos de um certo volume mensal de "
                f"{kpi_name}, independente do custo -- útil quando existe uma meta de volume "
                "que precisa ser batida, não só de eficiência.",
            )
            min_kpi_val = None
            if use_min_kpi:
                kpi_max = float(df["Monthly_KPI"].max())
                min_kpi_val = st.sidebar.number_input(
                    f"{kpi_name} Mínimo (mensal)",
                    min_value=0.0,
                    max_value=kpi_max,
                    value=kpi_max * 0.5,
                    step=max(1.0, kpi_max / 100),
                    format="%.0f",
                )

            # --- % Incrementalidade ---
            use_min_incrementality = st.sidebar.checkbox(
                "Aplicar % de Incrementalidade Mínima",
                value=False,
                help="Descarta pontos onde uma fatia pequena demais do KPI projetado é "
                "realmente incremental (isto é, atribuível ao investimento em si, e não à sua "
                "baseline orgânica). Protege contra recomendar investimento alto num nível onde "
                "a maior parte do resultado já aconteceria de qualquer forma.",
            )
            min_incrementality_pct = None
            if use_min_incrementality:
                min_incrementality_pct = st.sidebar.slider(
                    "% Incrementalidade Mínima",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    format="%.0f%%",
                )

            filtered_df = df[
                (df["Monthly_Investment"] >= min_budget)
                & (df["Monthly_Investment"] <= max_budget)
            ]

            if use_cpa_target and "CPA" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["CPA"] <= target_cpa]

            if use_icpa_target and "iCPA" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["iCPA"] <= target_icpa]

            if use_roas and min_roas is not None and "ROAS" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["ROAS"] >= min_roas]

            if use_iroas and min_iroas is not None and "iROAS" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["iROAS"] >= min_iroas]

            if use_min_kpi and min_kpi_val is not None:
                filtered_df = filtered_df[filtered_df["Monthly_KPI"] >= min_kpi_val]

            if use_min_incrementality and min_incrementality_pct is not None:
                filtered_df = filtered_df[filtered_df["Pct_Incrementality"] >= min_incrementality_pct]

            if filtered_df.empty:
                st.warning(
                    "Nenhum cenário corresponde aos critérios selecionados. Flexibilize seus limites."
                )
            else:
                optimal_point = filtered_df.iloc[-1]

                saturation_point = find_saturation_point(
                    df, optimal_point, min_investment=baseline_monthly_inv / DAYS_IN_MONTH
                )

                st.markdown(f"### Resumo dos Cenários Projetados - {kpi_name}")
                st.markdown(
                    "A tabela abaixo apresenta a comparação entre a sua média histórica real e o novo cenário de investimento simulado."
                )

                base_inv = true_baseline_monthly_inv
                base_kpi = true_baseline_monthly_kpi
                sim_inv = optimal_point["Monthly_Investment"]
                sim_kpi = optimal_point["Monthly_KPI"]

                scenario_data = {
                    "Cenário": [
                        "Cenário Atual",
                        "Ponto Recomendado",
                        "Cenário de Saturação",
                    ],
                    "Investimento Mensal": [
                        base_inv,
                        sim_inv,
                        saturation_point["Monthly_Investment"],
                    ],
                    f"Projeção de {kpi_name}": [
                        base_kpi,
                        sim_kpi,
                        saturation_point["Monthly_KPI"],
                    ],
                }
                scenario_df = pd.DataFrame(scenario_data)

                # ponytail: when the KPI is already R$, "cost per KPI" (R$/R$) is meaningless
                # -- show ROAS/iROAS (KPI/investment) instead of CPA/iCPA (investment/KPI).
                cost_col = f"ROAS ({kpi_name})" if kpi_is_monetary else f"Custo por {kpi_name}"
                icpa_col = "iROAS" if kpi_is_monetary else "iCPA"

                if kpi_is_monetary:
                    scenario_df[cost_col] = (
                        scenario_df[f"Projeção de {kpi_name}"]
                        / scenario_df["Investimento Mensal"]
                    )
                else:
                    scenario_df[cost_col] = (
                        scenario_df["Investimento Mensal"]
                        / scenario_df[f"Projeção de {kpi_name}"]
                    )
                scenario_df["Investimento Incremental"] = (
                    scenario_df["Investimento Mensal"] - base_inv
                )
                scenario_df[f"{kpi_name} Incrementais"] = (
                    scenario_df[f"Projeção de {kpi_name}"] - base_kpi
                )
                if kpi_is_monetary:
                    scenario_df[icpa_col] = (
                        scenario_df[f"{kpi_name} Incrementais"]
                        / scenario_df["Investimento Incremental"]
                    ).replace([np.inf, -np.inf], float("nan"))
                else:
                    scenario_df[icpa_col] = compute_incremental_cpa(
                        scenario_df["Investimento Incremental"],
                        scenario_df[f"{kpi_name} Incrementais"],
                    )

                scenario_df.loc[
                    0, ["Investimento Incremental", f"{kpi_name} Incrementais", icpa_col]
                ] = 0.0

                def format_currency(val):
                    if pd.isna(val):
                        return "N/A"
                    if val == 0:
                        return "R$ 0.00"
                    # Negative scenarios (e.g. a saturation point below baseline
                    # spend) must still abbreviate -- format on the magnitude,
                    # reapply the sign after.
                    sign = "-" if val < 0 else ""
                    abs_val = abs(val)
                    if abs_val >= 1_000_000:
                        return f"R$ {sign}{abs_val / 1_000_000:,.1f}M"
                    if abs_val >= 1_000:
                        return f"R$ {sign}{abs_val / 1_000:,.1f}k"
                    return f"R$ {sign}{abs_val:,.2f}"

                def format_number_kpi(val):
                    if pd.isna(val):
                        return "N/A"
                    if val == 0:
                        return "0.00"
                    sign = "-" if val < 0 else ""
                    abs_val = abs(val)
                    if abs_val >= 1_000_000:
                        return f"{sign}{abs_val / 1_000_000:,.1f}M"
                    if abs_val >= 1000:
                        return f"{sign}{abs_val / 1000:,.1f}k"
                    return f"{sign}{abs_val:,.0f}"

                def format_ratio(val):
                    if pd.isna(val):
                        return "N/A"
                    return f"{val:,.2f}x"

                kpi_formatter = format_currency if kpi_is_monetary else format_number_kpi
                cost_formatter = format_ratio if kpi_is_monetary else format_currency

                scenario_df_display = scenario_df.copy()
                scenario_df_display["Investimento Mensal"] = scenario_df_display[
                    "Investimento Mensal"
                ].apply(format_currency)
                scenario_df_display[f"Projeção de {kpi_name}"] = scenario_df_display[
                    f"Projeção de {kpi_name}"
                ].apply(kpi_formatter)
                scenario_df_display[cost_col] = scenario_df_display[cost_col].apply(
                    cost_formatter
                )
                scenario_df_display["Investimento Incremental"] = scenario_df_display[
                    "Investimento Incremental"
                ].apply(format_currency)
                scenario_df_display[f"{kpi_name} Incrementais"] = scenario_df_display[
                    f"{kpi_name} Incrementais"
                ].apply(kpi_formatter)
                scenario_df_display[icpa_col] = scenario_df_display[icpa_col].apply(
                    cost_formatter
                )

                st.dataframe(
                    scenario_df_display, width="stretch", hide_index=True
                )

                st.markdown("---")
                st.markdown("### Métricas da Estratégia Ótima")

                # Low-confidence guardrail. When the marketing model has no
                # meaningful out-of-sample fit, the engine already collapsed the
                # recommended mix back to the historical one (no reallocation) and
                # flagged it in global_saturation_metrics.json. Surface it here so
                # the numbers below read as directional, not as a solid channel
                # reshuffle recommendation.
                sat_metrics = {}
                try:
                    _mpath = os.path.join(output_dir, "global_saturation_metrics.json")
                    if os.path.exists(_mpath):
                        with open(_mpath, encoding="utf-8") as _f:
                            sat_metrics = json.load(_f)
                except Exception:
                    sat_metrics = {}

                # Calculate baseline ROAS early for the conditional warning expander
                baseline_roas_for_warning = (
                    true_baseline_monthly_kpi / true_baseline_monthly_inv
                    if true_baseline_monthly_inv > 0
                    else None
                )
                implausible_roas_threshold = 20.0

                # Render warning expanders sequentially below the title
                if sat_metrics.get("low_confidence"):
                    with st.expander("Alerta: Projeção de baixa confiança", expanded=True):
                        st.markdown(
                            "**Projeção de baixa confiança.** O modelo de marketing "
                            f"explica apenas {sat_metrics.get('cv_r_squared', 0) * 100:.1f}% "
                            "da variação fora da amostra (validação walk-forward), e a "
                            "contribuição de mídia medida foi de "
                            f"{sat_metrics.get('marketing_contribution_of_kpi_pct', 0):.1f}% "
                            f"do {kpi_name}. Por isso **a recomendação de realocar "
                            "orçamento entre canais foi desativada** — os números abaixo "
                            "refletem só o efeito de gastar mais no **mesmo mix atual**, e "
                            "devem ser lidos como direcionais, não como decisão de mix. "
                            "Sinal fraco assim costuma vir de **escopo** (o KPI conta "
                            "vendas de todos os canais, não só os pagos) ou de pouca "
                            "variação de investimento no período analisado."
                        )

                if kpi_is_monetary and baseline_roas_for_warning and baseline_roas_for_warning > implausible_roas_threshold:
                    with st.expander("Alerta: ROAS do Cenário Atual elevado", expanded=False):
                        st.markdown(
                            f"ROAS do Cenário Atual está em {baseline_roas_for_warning:.1f}x -- "
                            "muito acima do que a maioria dos negócios reais alcança "
                            "(tipicamente até 10-15x). Confira se 'O KPI já está em "
                            f"R$' e o Ticket Médio (acima, em Parâmetros do Negócio) "
                            f"realmente correspondem à coluna '{kpi_name}' do seu CSV "
                            "antes de usar esses números para decisão.\n\n"
                            "Se essas configurações já estiverem corretas, o problema "
                            "mais provável é de **escopo**: o arquivo de investimento "
                            "pode cobrir só uma fatia da mídia (ex: só estes canais "
                            f"pagos), enquanto '{kpi_name}' pode estar contando vendas "
                            "de todos os canais (orgânico, direto, etc.). Nesse caso "
                            "nenhuma combinação de checkbox resolve -- é preciso um "
                            "arquivo de investimento mais completo, ou isolar a métrica "
                            "só do que esses canais realmente influenciam."
                        )

                cfg_avg_ticket = config.get("average_ticket", 0)
                cfg_conv_rate = config.get("conversion_rate_from_kpi_to_bo", 0)
                if kpi_is_monetary:
                    st.caption(
                        f"Premissas usadas: KPI já é R$ (Ticket Médio e Taxa de "
                        f"Conversão ignorados) · Receita = {kpi_name} × 1,0"
                    )
                else:
                    st.caption(
                        f"Premissas usadas: KPI não é R$ · Ticket Médio = R$ "
                        f"{cfg_avg_ticket:,.2f} · Taxa de Conversão = {cfg_conv_rate:.1%} "
                        f"· Receita = {kpi_name} × Taxa de Conversão × Ticket Médio"
                    )

                col1, col2, col3, col4 = st.columns(4)

                inv_val = optimal_point["Monthly_Investment"]
                kpi_val = optimal_point["Monthly_KPI"]
                inc_kpi_val = optimal_point["Incremental_KPI"] * DAYS_IN_MONTH

                inv_str = (
                    f"R$ {inv_val / 1e6:,.2f}M"
                    if inv_val >= 1e6
                    else f"R$ {inv_val:,.0f}"
                )
                delta_inv = inv_val - baseline_monthly_inv
                delta_inv_str = (
                    f"R$ {delta_inv / 1e6:,.2f}M vs Baseline"
                    if abs(delta_inv) >= 1e6
                    else f"R$ {delta_inv:,.0f} vs Baseline"
                )

                col1.metric(
                    "Orçamento Mensal Otimizado", value=inv_str, delta=delta_inv_str
                )

                kpi_prefix = "R$ " if kpi_is_monetary else ""
                kpi_str = (
                    f"{kpi_prefix}{kpi_val / 1e6:,.2f}M"
                    if kpi_val >= 1e6
                    else f"{kpi_prefix}{kpi_val:,.0f}"
                )
                delta_kpi_str = (
                    f"{kpi_prefix}{inc_kpi_val / 1e6:,.2f}M Incremental"
                    if abs(inc_kpi_val) >= 1e6
                    else f"{kpi_prefix}{inc_kpi_val:,.0f} Incremental"
                )

                col2.metric(
                    f"Projeção Mensal de {kpi_name}", value=kpi_str, delta=delta_kpi_str
                )

                if kpi_is_monetary:
                    # ponytail: KPI is already R$, so investment/KPI ("cost per KPI") is
                    # meaningless -- show ROAS (KPI/investment) instead of CPA.
                    roas_val = kpi_val / inv_val if inv_val > 0 else 0.0
                    baseline_roas = (
                        true_baseline_monthly_kpi / true_baseline_monthly_inv
                        if true_baseline_monthly_inv > 0
                        else None
                    )
                    roas_delta = (
                        f"{roas_val - baseline_roas:+,.2f}x vs Baseline"
                        if baseline_roas
                        else None
                    )
                    col3.metric(
                        "Global ROAS",
                        value=f"{roas_val:.2f}x",
                        delta=roas_delta,
                        delta_color="normal",
                    )

                    iroas_val = optimal_point["iROAS"] if "iROAS" in optimal_point else 0.0
                    if pd.isna(iroas_val):
                        iroas_val = 0.0
                    target_iroas_cfg = active_config.get("financial_targets", {}).get("target_iroas")
                    if target_iroas_cfg and target_iroas_cfg > 0 and iroas_val > 0:
                        iroas_delta = f"limite {target_iroas_cfg:.2f}x"
                        iroas_delta_color = "normal" if iroas_val >= target_iroas_cfg else "inverse"
                    elif iroas_val > 0:
                        iroas_delta = "sem limite configurado"
                        iroas_delta_color = "off"
                    else:
                        iroas_delta, iroas_delta_color = None, "off"
                    col4.metric(
                        "Marginal iROAS",
                        value=f"{iroas_val:.2f}x" if iroas_val > 0 else "N/A",
                        delta=iroas_delta,
                        delta_color=iroas_delta_color,
                    )

                    # Implausible ROAS check has been moved to the top expanders.
                else:
                    cpa_val = (
                        optimal_point["CPA"]
                        if "CPA" in optimal_point
                        else (
                            optimal_point["Daily_Investment"]
                            / optimal_point["Projected_Total_KPIs"]
                        )
                    )
                    baseline_cpa = (
                        true_baseline_monthly_inv / true_baseline_monthly_kpi
                        if true_baseline_monthly_kpi > 0
                        else None
                    )
                    cpa_delta = f"R$ {cpa_val - baseline_cpa:+,.2f} vs Baseline" if baseline_cpa else None
                    col3.metric("Global CPA", value=f"R$ {cpa_val:,.2f}", delta=cpa_delta, delta_color="inverse")

                    icpa_val = optimal_point["iCPA"] if "iCPA" in optimal_point else 0.0
                    if pd.isna(icpa_val):
                        icpa_val = 0.0
                    target_icpa_cfg = active_config.get("financial_targets", {}).get("target_icpa")
                    # ponytail: 999999 is the sentinel for "no limit set" (see line ~857)
                    if target_icpa_cfg and target_icpa_cfg < 999999 and icpa_val > 0:
                        icpa_delta = f"limite R$ {target_icpa_cfg:,.2f}"
                        icpa_delta_color = "normal" if icpa_val <= target_icpa_cfg else "inverse"
                    elif icpa_val > 0:
                        icpa_delta = "sem limite configurado"
                        icpa_delta_color = "off"
                    else:
                        icpa_delta, icpa_delta_color = None, "off"
                    col4.metric(
                        "Marginal iCPA",
                        value=f"R$ {icpa_val:,.2f}" if icpa_val > 0 else "N/A",
                        delta=icpa_delta,
                        delta_color=icpa_delta_color,
                    )

                _, c2, c3, c4, _ = st.columns([0.5, 1, 1, 1, 0.5])

                kpi_gain_pct = (
                    (kpi_val - true_baseline_monthly_kpi) / true_baseline_monthly_kpi * 100
                    if true_baseline_monthly_kpi > 0
                    else 0.0
                )
                kpi_gain_abs = kpi_val - true_baseline_monthly_kpi
                gain_delta_label = (
                    f"{kpi_prefix}{kpi_gain_abs:,.0f}"
                    if kpi_is_monetary
                    else f"{kpi_gain_abs:,.0f} unidades"
                )
                c2.metric(
                    f"Ganho de {kpi_name} (%)",
                    value=f"{kpi_gain_pct:+.1f}%",
                    delta=gain_delta_label,
                )

                inc_inv_val = optimal_point["Incremental_Investment"] * DAYS_IN_MONTH
                inc_inv_str = (
                    f"R$ {inc_inv_val / 1e6:,.2f}M" if inc_inv_val >= 1e6 else f"R$ {inc_inv_val:,.0f}"
                )
                c3.metric(
                    "Investimento Incremental",
                    value=inc_inv_str,
                    delta="vs. Cenário Atual",
                    delta_color="off",
                )

                inc_rev = optimal_point.get("Incremental_Revenue", 0.0) if hasattr(optimal_point, "get") else optimal_point["Incremental_Revenue"] if "Incremental_Revenue" in optimal_point.index else 0.0
                inc_inv_daily = optimal_point["Incremental_Investment"] if "Incremental_Investment" in optimal_point.index else 0.0
                if not pd.isna(inc_rev) and inc_rev > 0 and inc_inv_daily > 0:
                    # True ROI (profit / investment), matching gemini_report.py's formula.
                    # inc_rev / inc_inv_daily alone is ROAS, not ROI — off by exactly 1.0x.
                    iroi = (inc_rev - inc_inv_daily) / inc_inv_daily
                    c4.metric("ROI Incremental", value=f"{iroi:.2f}x", delta="(receita - investimento) / investimento", delta_color="off")
                elif inc_kpi_val > 0 and inc_inv_val > 0:
                    efficiency = inc_kpi_val / (inc_inv_val / 1000)
                    c4.metric(f"{kpi_name} / R$1k", value=f"{efficiency:.1f}", delta="eficiência incremental", delta_color="off")
                else:
                    c4.metric("Eficiência Incremental", value="N/A")

                st.markdown("---")
                st.markdown("### Curva de Saturação de Investimentos")
                st.markdown(
                    "A curva abaixo mostra a relação entre investimento mensal e KPI projetado. "
                    "O **ponto ótimo** (⭐) indica onde o retorno marginal começa a diminuir significativamente — "
                    "investir além desse ponto gera ganhos decrescentes. "
                    "A **linha verde** marca a base histórica de investimento para referência."
                )

                plot_limit = max_budget * 1.30
                df_plot = df[df["Monthly_Investment"] <= plot_limit]

                fig_curve = go.Figure()
                fig_curve.add_trace(
                    go.Scatter(
                        x=df_plot["Monthly_Investment"],
                        y=df_plot["Monthly_KPI"],
                        mode="lines",
                        name="Modelo de Elasticidade",
                        line=dict(color="blue", width=3),
                        hovertemplate="<b>Investimento:</b> R$ %{x:.2s}<br><b>KPI Projetado:</b> %{y:.2s}<extra></extra>",
                    )
                )

                fig_curve.add_vline(
                    x=baseline_monthly_inv,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Base Histórica",
                    annotation_position="top left",
                )

                fig_curve.add_trace(
                    go.Scatter(
                        x=[optimal_point["Monthly_Investment"]],
                        y=[optimal_point["Monthly_KPI"]],
                        mode="markers",
                        marker=dict(color="red", size=12, symbol="star"),
                        name="Ponto Escolhido (Ótimo)",
                    )
                )

                fig_curve.add_hline(
                    y=optimal_point["Monthly_KPI"],
                    line_dash="dot",
                    line_color="red",
                    opacity=0.5,
                )
                fig_curve.add_vline(
                    x=optimal_point["Monthly_Investment"],
                    line_dash="dot",
                    line_color="red",
                    opacity=0.5,
                )

                fig_curve.update_layout(
                    xaxis_title="Investimento Mensal",
                    yaxis_title=f"KPI Projetado - {kpi_name}",
                    xaxis=dict(tickformat=".2s"),
                    yaxis=dict(tickformat=".2s"),
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5
                    ),
                    margin=dict(l=20, r=20, t=50, b=20),
                )

                st.plotly_chart(fig_curve, width="stretch")

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
                    width="stretch",
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
                        build_revenue_roi_curve(
                            df_plot, kpi_name=kpi_name, monthly_factor=DAYS_IN_MONTH
                        ),
                        width="stretch",
                    )

                st.markdown("---")
                st.markdown("### Evolução do Mix de Canais por Orçamento")
                st.markdown(
                    "Mostra como a alocação recomendada entre canais (Modelo de Elasticidade) muda "
                    "conforme o orçamento total escala: a cada nível de investimento, o mix é "
                    "reotimizado para o de maior retorno projetado (respeitando o teto de "
                    f"{int(config.get('max_channel_mix_share', 0.4) * 100)}% de participação por "
                    "canal), então as porcentagens podem variar de um nível para o outro."
                )
                st.plotly_chart(
                    build_channel_mix_evolution(
                        df_plot,
                        baseline_monthly_inv=baseline_monthly_inv,
                        optimal_monthly_inv=optimal_point["Monthly_Investment"],
                    ),
                    width="stretch",
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
                        width="stretch",
                    )

                # --- NEW: Individual Curves Visualization ---
                st.markdown("---")
                st.markdown("### Curvas de Resposta Individuais por Canal")
                st.markdown(
                    "Simula a curva de resposta de um canal específico, mantendo os "
                    "demais canais na média histórica de investimento. As linhas "
                    "verticais marcam o investimento médio histórico e o investimento "
                    "recomendado pela otimização."
                )

                ind_csv_path = os.path.join(
                    output_dir, "individual_response_curves_data.csv"
                )
                if os.path.exists(ind_csv_path):
                    ind_df = pd.read_csv(ind_csv_path)
                    channels = ind_df["Channel"].unique()

                    selected_channel = st.selectbox(
                        "Selecione um Canal para Visualizar a Curva", channels
                    )

                    channel_df = ind_df[ind_df["Channel"] == selected_channel]

                    if "Historical_Avg" in ind_df.columns:
                        st.plotly_chart(
                            build_response_curve_individual(
                                channel_df, selected_channel
                            ),
                            width="stretch",
                        )
                    else:
                        # Sanitize channel name for filename
                        safe_channel_name = "".join(
                            [
                                c if c.isalnum() or c in ["-", "_"] else "_"
                                for c in selected_channel
                            ]
                        )
                        img_path = os.path.join(
                            output_dir,
                            f"individual_response_curve_{safe_channel_name}.png",
                        )

                        if os.path.exists(img_path):
                            st.image(
                                img_path,
                                caption=f"Curva de Resposta Individual: {selected_channel}",
                            )
                        else:
                            st.warning(
                                f"Imagem da curva não encontrada para o canal: {selected_channel}"
                            )
                else:
                    st.info(
                        "Os dados das curvas individuais não foram encontrados. Certifique-se de rodar a análise primeiro."
                    )

                st.markdown("### Mix de Orçamento Recomendado")
                row_donut1, row_donut2 = st.columns(2)

                hist_cols = [
                    c
                    for c in optimal_point.index
                    if c.startswith("Spend_") and c.endswith("_Historical")
                ]
                hist_data = [
                    {
                        "Channel": c.replace("Spend_", "").replace("_Historical", ""),
                        "Budget": optimal_point[c] * DAYS_IN_MONTH,
                    }
                    for c in hist_cols
                    if optimal_point[c] > 0
                ]
                hist_df = pd.DataFrame(hist_data)

                with row_donut1:
                    if not hist_df.empty:
                        total_hist = hist_df["Budget"].sum()
                        hist_df["Pct"] = hist_df["Budget"] / total_hist * 100
                        hist_df["Label"] = hist_df.apply(
                            lambda r: (
                                f"R$ {r['Budget']/1e6:.1f}M" if r['Budget'] >= 1e6
                                else (f"R$ {r['Budget']/1e3:.1f}k" if r['Budget'] >= 1e3 else f"R$ {r['Budget']:,.0f}")
                            ) + f" ({r['Pct']:.1f}%)",
                            axis=1,
                        )
                        hist_df = hist_df.sort_values("Budget")
                        fig_hist = px.bar(
                            hist_df,
                            x="Budget",
                            y="Channel",
                            color="Channel",
                            orientation="h",
                            title="Alocação Histórica",
                            text="Label",
                        )
                        fig_hist.update_traces(textposition="outside", cliponaxis=False)
                        fig_hist.update_layout(
                            xaxis_title=None,
                            yaxis_title=None,
                            xaxis_showticklabels=False,
                            showlegend=False,
                            margin=dict(r=160),
                        )
                        st.plotly_chart(fig_hist, width="stretch")

                strat_cols = [
                    c
                    for c in optimal_point.index
                    if c.startswith("Spend_") and c.endswith("_Strategic")
                ]
                strat_data = [
                    {
                        "Channel": c.replace("Spend_", "").replace("_Strategic", ""),
                        "Budget": optimal_point[c] * DAYS_IN_MONTH,
                    }
                    for c in strat_cols
                    if optimal_point[c] > 0
                ]
                strat_df = pd.DataFrame(strat_data)

                with row_donut2:
                    if not strat_df.empty:
                        total_strat = strat_df["Budget"].sum()
                        strat_df["Pct"] = strat_df["Budget"] / total_strat * 100
                        strat_df["Label"] = strat_df.apply(
                            lambda r: (
                                f"R$ {r['Budget']/1e6:.1f}M" if r['Budget'] >= 1e6
                                else (f"R$ {r['Budget']/1e3:.1f}k" if r['Budget'] >= 1e3 else f"R$ {r['Budget']:,.0f}")
                            ) + f" ({r['Pct']:.1f}%)",
                            axis=1,
                        )
                        strat_df = strat_df.sort_values("Budget")
                        fig_strat = px.bar(
                            strat_df,
                            x="Budget",
                            y="Channel",
                            color="Channel",
                            orientation="h",
                            title="Alocação Recomendada",
                            text="Label",
                        )
                        fig_strat.update_traces(textposition="outside", cliponaxis=False)
                        fig_strat.update_layout(
                            xaxis_title=None,
                            yaxis_title=None,
                            xaxis_showticklabels=False,
                            showlegend=False,
                            margin=dict(r=160),
                        )
                        st.plotly_chart(fig_strat, width="stretch")

                st.markdown("---")
                st.markdown("## Recomendações Estratégicas")
                if narrative and "executive_summary" in narrative:
                    import re

                    optimal_inv_val = optimal_point["Monthly_Investment"]
                    optimal_inv_str = (
                        f"R$ {optimal_inv_val / 1e6:,.1f}M".replace(".", ",")
                        if optimal_inv_val >= 1e6
                        else f"R$ {optimal_inv_val / 1e3:,.0f}k".replace(".", ",")
                    )

                    def align_insight_text(text):
                        res = re.sub(
                            r"(?:R\$?\s*)?15[,.]8M",
                            optimal_inv_str,
                            text,
                            flags=re.IGNORECASE,
                        )
                        return res.replace("R$", "R\\$")

                    dynamic_summary = align_insight_text(narrative["executive_summary"])
                    st.markdown(f"**Resumo Executivo:** {dynamic_summary}")

                    if "strategic_recommendations" in narrative:
                        st.markdown("### Oportunidades Listadas")
                        recs_list = []
                        for rec in narrative["strategic_recommendations"]:
                            rec_text = (
                                rec.get(
                                    "recommendation", rec.get("description", str(rec))
                                )
                                if isinstance(rec, dict)
                                else str(rec)
                            )
                            dynamic_rec = align_insight_text(rec_text)
                            recs_list.append(f"- {dynamic_rec}")
                        st.markdown("\n".join(recs_list))
                else:
                    st.info(
                        "O modelo Gemini não rodou / O arquivo `global_narrative.json` de insights não foi localizado no backend."
                    )

                st.markdown("---")
                st.markdown("## Metodologia")
                st.markdown("""
Esta ferramenta opera como um **Agente Autônomo de Otimização Causais**, com foco exclusivo na construção de um **Modelo Global de Elasticidade**.

**Como o Modelo Funciona:**
Ele compila e analisa rigorosamente todo o histórico de alocações da sua empresa ao longo do tempo. O motor constrói e simula milhões de cenários matemáticos contra **Curvas de AdStock e Efeitos de Retardo (Diminishing Returns)**. 

O objetivo principal desta abordagem algorítmica é mapear com exatidão o ponto exato em que o investimento marginal em cada canal de aquisição (como Search, PMAX, App, etc.) começa a saturar — ou seja, o momento em que cada Real adicional investido passa a trazer menos retorno do que o Real anterior.

Ao compreender matematicamente o formato dessas curvas de saturação individuais de cada canal, o sistema consegue redistribuir dinamicamente a verba total. Ele busca equilibrar o peso entre os canais até encontrar a alocação perfeita, extraindo o **Retorno Marginal Máximo** de toda a carteira de investimentos e compondo uma estratégia "Always-On" blindada contra desperdícios de escala.
                """)

                st.markdown("---")
                st.markdown("## Entendendo os Cenários")
                st.markdown("""
A sua **Curva de Saturação** dita o limite máximo quantitativo que a sua carteira conseguirá atingir. Para uma compreensão plena, categorizamos o resultado nos seguintes blocos:

- **Cenário Atual:** Esta é a sua linha de base. Exibe como sua marca vem performando no histórico consolidado com as eficiências e alocações passadas.
- **Ponto Recomendado:** Essa é a "estrela vermelha" sinalizada no gráfico! Você pode deslizá-la ativamente alterando os limites na barra lateral. O modelo algorítmico **força todo o mix de verba recalibrando-se para bater as restrições que você impôs**, extraindo o máximo de vendas focando em um custo de aquisição agressivamente eficiente. Trata-se do ganho puro de eficiência com balanço marginal ideal para o cenário que você dita.
- **Cenário de Saturação:** Reflete matematicamente onde o limite teto da sua operação existe — e a partir dali, você operará sem tração. Enviar recursos além disso será o custo absoluto de um CPA severamente mais caro. Trata-se do topo visível da curva que cede à linha reta limitante.
                """)
        else:
            st.info(
                "Nenhum dado encontrado para as configurações no caminho definido. Utilize a aba de Setup para gerar os datasets ou verifique o log backend."
            )


