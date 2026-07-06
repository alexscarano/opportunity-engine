import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import logging

# Configure basic logging for standard output (capturable by Google Cloud Logging)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("opp_engine_tracker")

# Optional: keep logging for raw actions without email barriers, if desired later, but removing barrier logic here.

st.set_page_config(
    page_title="Opportunity Engine",
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
    
    /* Logo styling and responsive switching */
    .logo-container {
        text-align: center;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .logo-light {
        display: block !important;
    }
    .logo-dark {
        display: none !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .logo-light {
            display: none !important;
        }
        .logo-dark {
            display: block !important;
        }
    }
    
    [data-theme="dark"] .logo-light,
    html[data-theme="dark"] .logo-light {
        display: none !important;
    }
    [data-theme="dark"] .logo-dark,
    html[data-theme="dark"] .logo-dark {
        display: block !important;
    }
</style>
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
logo_almap_dark = load_logo_base64("AF_ALMAPBBDO_LOGO_FINAL FILIPE-04.png")

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
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 50px; margin-bottom: 20px;">
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_dash_light}" class="logo-light" style="max-height: 80px; margin: auto; display: block;" />
                <img src="data:image/png;base64,{logo_dash_dark}" class="logo-dark" style="max-height: 80px; margin: auto; display: block;" />
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
            if st.button("Entrar", use_container_width=True):
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
            if st.button("Cadastrar", use_container_width=True):
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
        
        # Footer logos
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 40px; margin-bottom: 20px;">
                <p style="color: #888; font-size: 0.8rem; margin-bottom: 8px;">Powered by</p>
                <div class="logo-container">
                    <img src="data:image/png;base64,{logo_almap_light}" class="logo-light" style="max-height: 30px; margin: auto; display: block;" />
                    <img src="data:image/png;base64,{logo_almap_dark}" class="logo-dark" style="max-height: 30px; margin: auto; display: block;" />
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# Authenticated — inject CSS now
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


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
    <div class="logo-container" style="padding: 10px 0; margin-bottom: 10px;">
        <img src="data:image/png;base64,{logo_dash_light}" class="logo-light" style="max-height: 50px; margin: auto; display: block;" />
        <img src="data:image/png;base64,{logo_dash_dark}" class="logo-dark" style="max-height: 50px; margin: auto; display: block;" />
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar user card & Logout
st.sidebar.markdown(f"**Conectado como:** {st.session_state['username']}")
if st.sidebar.button("Sair/Logout", use_container_width=True):
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
        "Renomear Projeto Atual", key="rename_proj_btn", use_container_width=True
    ):
        st.session_state["show_rename_confirm"] = True

    if st.session_state.get("show_rename_confirm"):
        new_name = st.sidebar.text_input(
            "Novo nome:", value=selected_project, key="rename_proj_input"
        )
        col_ren1, col_ren2 = st.sidebar.columns(2)
        with col_ren1:
            if st.button("Salvar", key="confirm_rename_btn", use_container_width=True):
                if rename_user_project(
                    st.session_state["user_id"], selected_project, new_name
                ):
                    st.toast(f"Projeto renomeado para '{new_name.strip()}'.")
                    st.session_state["show_rename_confirm"] = False
                    st.rerun()
                else:
                    st.error("Nome inválido ou já em uso.")
        with col_ren2:
            if st.button("Cancelar", key="cancel_rename_btn", use_container_width=True):
                st.session_state["show_rename_confirm"] = False
                st.rerun()

    # Project Deletion UI
    if st.sidebar.button(
        "Excluir Projeto Atual", key="delete_proj_btn", use_container_width=True
    ):
        st.session_state["show_delete_confirm"] = True

    if st.session_state.get("show_delete_confirm"):
        st.sidebar.warning(
            f"Deseja mesmo excluir o projeto '{selected_project}' e todos os seus arquivos?"
        )
        col_del1, col_del2 = st.sidebar.columns(2)
        with col_del1:
            if st.button(
                "Sim, Excluir", key="confirm_delete_btn", use_container_width=True
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
            if st.button("Cancelar", key="cancel_delete_btn", use_container_width=True):
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
            )

            st.divider()
            st.subheader("Parâmetros do Negócio")
            kpi_column = st.text_input(
                "Coluna do KPI no CSV de Performance",
                value="Sessions",
                help="Nome exato da coluna (ex: Sessions, Conversions, Leads).",
            )
            optimization_target_label = st.selectbox(
                "Objetivo da Otimização",
                options=[
                    "Maximizar Volume de Conversões (Leads, Vendas, etc.)",
                    "Maximizar Receita / Faturamento (Revenue)",
                ],
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
                    help="Deixe 100% se o KPI já for a venda final.",
                )
                / 100.0
            )
            avg_ticket = st.number_input(
                "Ticket Médio (R$)",
                value=100.0,
                step=10.0,
            )

        with col2:
            st.subheader("Dados Brutos (CSV)")
            inv_file = st.file_uploader(
                "Investimento (obrigatório)",
                type=["csv"],
                help="Investimento diário por canal de mídia.",
            )
            perf_file = st.file_uploader(
                "Performance (obrigatório)",
                type=["csv"],
                help="Histórico diário de resultados/KPIs.",
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
                help="0.00 = sem restrição.",
            )
            target_roas = st.number_input(
                "ROAS Mínimo",
                value=0.0,
                help="Ex: 2.5 = R$2,50 por R$1,00 investido. 0.00 = sem restrição.",
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_btn = st.form_submit_button(
                "Construir Motor de Oportunidades",
                type="primary",
                use_container_width=True,
            )
        with col_btn2:
            save_settings_btn = st.form_submit_button(
                "Salvar Configurações", type="secondary", use_container_width=True
            )

    if save_settings_btn:
        update_user_api_key(st.session_state["user_id"], gemini_key)
        st.success("Configurações salvas com sucesso!")

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

                import pandas as pd

                def get_date_col(file_path):
                    if not file_path:
                        return "date"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in ["date", "dates", "data", "day", "dia"]:
                                return col
                        return df.columns[0]
                    except:
                        return "date"

                def get_channel_col(file_path):
                    if not file_path:
                        return "product_group"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in [
                                "channel",
                                "product_group",
                                "product",
                                "media",
                                "source",
                                "campaign",
                                "canal",
                                "grupo",
                            ]:
                                return col
                        return df.columns[0]
                    except:
                        return "product_group"

                def get_investment_col(file_path):
                    if not file_path:
                        return "total_revenue"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in [
                                "investment",
                                "spend",
                                "cost",
                                "investimento",
                                "revenue",
                                "total_revenue",
                                "valor",
                            ]:
                                return col
                        return df.columns[-1]
                    except:
                        return "total_revenue"

                def get_trends_col(file_path):
                    if not file_path:
                        return "Ad Opportunities"
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        for col in df.columns:
                            if col.lower() in [
                                "searches",
                                "trends",
                                "opportunities",
                                "ad opportunities",
                                "volume",
                                "generic searches",
                            ]:
                                return col
                        return df.columns[-1]
                    except:
                        return "Ad Opportunities"

                def get_kpi_col(file_path, user_kpi):
                    if not file_path:
                        return user_kpi
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        if user_kpi in df.columns:
                            return user_kpi
                        for col in df.columns:
                            if col.lower() in [
                                "kpi",
                                "sessions",
                                "conversions",
                                "revenue",
                                "conversoes",
                                "cliques",
                                "clicks",
                            ]:
                                return col
                        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    except:
                        return user_kpi

                inv_date = get_date_col(inv_path)
                perf_date = get_date_col(perf_path)
                trends_date = get_date_col(trends_path) if trends_path else "Day"

                inv_channel = get_channel_col(inv_path)
                inv_investment = get_investment_col(inv_path)
                perf_kpi = get_kpi_col(perf_path, kpi_column)
                trends_col = (
                    get_trends_col(trends_path) if trends_path else "Ad Opportunities"
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
                    "financial_targets": {
                        "target_cpa": target_cpa if target_cpa > 0 else 999999,
                        "target_icpa": 999999,
                        "target_roas": target_roas if target_roas > 0 else 0,
                        "target_iroas": 0,
                    },
                    "optimization_target": optimization_target,
                    "investment_limit_factor": 1.5,
                    "p_value_threshold": 0.1,
                    "r_squared_threshold": 0.5,
                    "increase_threshold_percent": 20,
                    "decrease_threshold_percent": 10,
                    "post_event_days": 14,
                    "max_events_to_analyze": 3,
                    "treat_outliers": False,
                    "date_formats": {
                        "investment_file": "%Y-%m-%d",
                        "performance_file": "%Y-%m-%d",
                        "generic_trends_file": "%Y-%m-%d",
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

            full_log = ""
            for line in iter(process.stdout.readline, ""):
                full_log += line
                log_container.code(
                    f"Engine de Oportunidades Rodando...\n{full_log}",
                    language="shell",
                    height=400,
                )

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                st.session_state["active_config_path"] = config_path_gen
                st.session_state["show_run_success_balloons"] = True
                st.rerun()
            else:
                status_container.error(
                    "Houve um erro na execução do motor. Verifique os logs acima."
                )

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
        adv_dir = os.path.join("outputs", adv_name)

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
                    use_container_width=True,
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

            html_in_dir = glob.glob(os.path.join(selected_dir, "gemini_report_*.html"))
            html_in_dir = [r for r in html_in_dir if "global_report.html" not in r]

            if html_in_dir:
                with open(html_in_dir[0], "r", encoding="utf-8") as f:
                    html_content = f.read()
                import streamlit.components.v1 as components

                components.html(html_content, height=800, scrolling=True)
            else:
                md_path = os.path.join(selected_dir, "RECOMMENDATIONS.md")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    st.markdown(md_content)

                    st.markdown("---")
                    st.markdown("### Gráficos da Análise")
                    png_files = glob.glob(os.path.join(selected_dir, "*.png"))
                    if png_files:
                        for png_file in sorted(png_files):
                            filename = os.path.basename(png_file).lower()
                            if "accuracy" in filename:
                                caption = "Acurácia do Modelo Pré-Intervenção (Predict vs Actual)"
                            elif "sessions" in filename or "kpi" in filename:
                                caption = "Efeito Causal no KPI"
                            elif "investment" in filename or "cost" in filename:
                                caption = "Pico de Investimento (Intervenção)"
                            elif "line_chart" in filename:
                                caption = "Gráfico Resumo (Impacto Causal)"
                            else:
                                caption = "Gráfico da Análise"

                            st.image(
                                png_file, caption=caption, use_container_width=True
                            )
                else:
                    st.warning("Nenhum relatório encontrado para este evento.")

            csv_names = [
                "line_chart_data.csv",
                "accuracy_data.csv",
                "investment_data.csv",
                "sessions_data.csv",
            ]
            if all(os.path.exists(os.path.join(selected_dir, f)) for f in csv_names):
                st.markdown("### Gráficos Interativos")
                kpi_name = active_config.get("primary_business_metric_name", "kpi")

                line_chart_df = pd.read_csv(os.path.join(selected_dir, "line_chart_data.csv"))
                st.plotly_chart(
                    build_causal_line_chart(line_chart_df, kpi_name=kpi_name),
                    use_container_width=True,
                )

                accuracy_chart_df = pd.read_csv(os.path.join(selected_dir, "accuracy_data.csv"))
                st.plotly_chart(
                    build_accuracy_chart(accuracy_chart_df, kpi_name=kpi_name),
                    use_container_width=True,
                )

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
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        build_sessions_bar_chart(sessions_chart_df, kpi_name=kpi_name),
                        use_container_width=True,
                    )

        else:
            st.info(
                "Nenhum relatório de Impacto Causal encontrado. Rode o motor na aba Setup ou verifique as restrições abaixo."
            )
            st.markdown(
                f"""
**Um evento só gera relatório se passar por todas as condições:**
- Pico ou queda de investimento semanal de ao menos **+{active_config.get("increase_threshold_percent", 50)}%** ou **-{active_config.get("decrease_threshold_percent", 30)}%** vs. a média das últimas 12 semanas
- Pelo menos **30 dias** de dados antes do evento
- Significância estatística: **p-value < {active_config.get("p_value_threshold", 0.1)}**
- Ajuste do modelo: **R² ≥ {active_config.get("r_squared_threshold", 0.6)}**
- Direção lógica (investimento sobe e KPI sobe, ou ambos caem)

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
                        if investment_pivot_df[col].mean() > 0 and col != "Other"
                    ]
                    total_avg_daily_spend = sum(
                        investment_pivot_df[col].mean() for col in active_spend_cols
                    )
                    true_baseline_monthly_inv = total_avg_daily_spend * 30

                    if not df.empty:
                        closest_idx = (
                            (df["Daily_Investment"] - total_avg_daily_spend)
                            .abs()
                            .idxmin()
                        )
                        true_baseline_monthly_kpi = (
                            df.loc[closest_idx, "Projected_Total_KPIs_Historical"] * 30
                        )
                    else:
                        true_baseline_monthly_kpi = kpi_df["kpi"].mean() * 30
                except Exception as e:
                    import traceback

                    print(f"Error during data_preprocessor in Streamlit: {e}")
                    traceback.print_exc()

            return (
                config,
                df,
                narrative,
                output_dir,
                true_baseline_monthly_inv,
                true_baseline_monthly_kpi,
            )
        except Exception as e:
            import traceback

            print(f"Error loading global saturation data in load_data: {e}")
            traceback.print_exc()
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
            DAYS_IN_MONTH = 30
            df["Monthly_Investment"] = df["Daily_Investment"] * DAYS_IN_MONTH
            df["Monthly_KPI"] = df["Projected_Total_KPIs"] * DAYS_IN_MONTH
            baseline_monthly_inv = true_baseline_monthly_inv

            df["CPA"] = df["Daily_Investment"] / df["Projected_Total_KPIs"]
            df["CPA"] = df["CPA"].replace([np.inf, -np.inf], float("nan"))

            df["iCPA"] = df["Incremental_Investment"] / df["Incremental_KPI"]
            df["iCPA"] = df["iCPA"].replace([np.inf, -np.inf], float("nan")).fillna(0)

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
                "Use estes limites para encontrar o ponto ótimo na curva."
            )

            max_inv_val = float(df["Monthly_Investment"].max())
            min_inv_val = float(df["Monthly_Investment"].min())

            # --- Orçamento ---
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
                "Aplicar Limite de Target CPA", value=False
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
                )

            # --- iCPA ---
            use_icpa_target = st.sidebar.checkbox(
                "Aplicar Limite de iCPA Marginal", value=False
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
                )

            # --- ROAS ---
            use_roas = st.sidebar.checkbox("Aplicar ROAS Mínimo", value=False)
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
                )

            # --- iROAS ---
            use_iroas = st.sidebar.checkbox("Aplicar iROAS Mínimo", value=False)
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
                )

            # --- KPI Mínimo ---
            use_min_kpi = st.sidebar.checkbox("Aplicar KPI Mínimo", value=False)
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
                "Aplicar % de Incrementalidade Mínima", value=False
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

                saturation_point = find_saturation_point(df, optimal_point)

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

                scenario_df[f"Custo por {kpi_name}"] = (
                    scenario_df["Investimento Mensal"]
                    / scenario_df[f"Projeção de {kpi_name}"]
                )
                scenario_df["Investimento Incremental"] = (
                    scenario_df["Investimento Mensal"] - base_inv
                )
                scenario_df[f"{kpi_name} Incrementais"] = (
                    scenario_df[f"Projeção de {kpi_name}"] - base_kpi
                )
                scenario_df["iCPA"] = compute_incremental_cpa(
                    scenario_df["Investimento Incremental"],
                    scenario_df[f"{kpi_name} Incrementais"],
                )

                scenario_df.loc[
                    0, ["Investimento Incremental", f"{kpi_name} Incrementais", "iCPA"]
                ] = 0.0

                def format_currency(val):
                    if pd.isna(val):
                        return "N/A"
                    if val == 0:
                        return "R$ 0.00"
                    if val >= 1_000_000:
                        return f"R$ {val / 1_000_000:,.1f}M"
                    if val >= 1_000:
                        return f"R$ {val / 1_000:,.1f}k"
                    return f"R$ {val:,.2f}"

                def format_number_kpi(val):
                    if pd.isna(val):
                        return "N/A"
                    if val == 0:
                        return "0.00"
                    if val >= 1_000_000:
                        return f"{val / 1_000_000:,.1f}M"
                    if val >= 1000:
                        return f"{val / 1000:,.1f}k"
                    return f"{val:,.0f}"

                scenario_df_display = scenario_df.copy()
                scenario_df_display["Investimento Mensal"] = scenario_df_display[
                    "Investimento Mensal"
                ].apply(format_currency)
                scenario_df_display[f"Projeção de {kpi_name}"] = scenario_df_display[
                    f"Projeção de {kpi_name}"
                ].apply(format_number_kpi)
                scenario_df_display[f"Custo por {kpi_name}"] = scenario_df_display[
                    f"Custo por {kpi_name}"
                ].apply(format_currency)
                scenario_df_display["Investimento Incremental"] = scenario_df_display[
                    "Investimento Incremental"
                ].apply(format_currency)
                scenario_df_display[f"{kpi_name} Incrementais"] = scenario_df_display[
                    f"{kpi_name} Incrementais"
                ].apply(format_number_kpi)
                scenario_df_display["iCPA"] = scenario_df_display["iCPA"].apply(
                    format_currency
                )

                st.dataframe(
                    scenario_df_display, use_container_width=True, hide_index=True
                )

                st.markdown("---")
                st.markdown("### Métricas da Estratégia Ótima")
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

                kpi_str = (
                    f"{kpi_val / 1e6:,.2f}M" if kpi_val >= 1e6 else f"{kpi_val:,.0f}"
                )
                delta_kpi_str = (
                    f"{inc_kpi_val / 1e6:,.2f}M Incremental"
                    if abs(inc_kpi_val) >= 1e6
                    else f"{inc_kpi_val:,.0f} Incremental"
                )

                col2.metric(
                    f"Projeção Mensal de {kpi_name}", value=kpi_str, delta=delta_kpi_str
                )

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
                c2.metric(
                    f"Ganho de {kpi_name} (%)",
                    value=f"{kpi_gain_pct:+.1f}%",
                    delta=f"{kpi_val - true_baseline_monthly_kpi:,.0f} unidades",
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
                    # ponytail: both Incremental_Revenue and Incremental_Investment are daily — ratio is valid as-is
                    iroi = inc_rev / inc_inv_daily
                    c4.metric("ROI Incremental", value=f"{iroi:.2f}x", delta="receita / investimento", delta_color="off")
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

                st.plotly_chart(fig_curve, use_container_width=True)

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
                    use_container_width=True,
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
                        build_revenue_roi_curve(df_plot, kpi_name=kpi_name),
                        use_container_width=True,
                    )

                st.markdown("---")
                st.markdown("### Evolução do Mix de Canais por Orçamento")
                st.markdown(
                    "Mostra a alocação recomendada entre canais (Modelo de Elasticidade) — uma "
                    "proporção fixa — em reais por canal, conforme o orçamento total escala. As "
                    "porcentagens permanecem constantes; o que muda é o valor em R$ que cada canal "
                    "recebe a cada nível de investimento total."
                )
                st.plotly_chart(
                    build_channel_mix_evolution(
                        df_plot,
                        baseline_monthly_inv=baseline_monthly_inv,
                        optimal_monthly_inv=optimal_point["Monthly_Investment"],
                    ),
                    use_container_width=True,
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
                        use_container_width=True,
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
                            use_container_width=True,
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
                        st.plotly_chart(fig_hist, use_container_width=True)

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
                        st.plotly_chart(fig_strat, use_container_width=True)

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

# Sidebar footer logo (AlmapBBDO)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 10px;">
        <p style="color: #888; font-size: 0.75rem; margin-bottom: 5px;">Powered by</p>
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_almap_light}" class="logo-light" style="max-height: 25px; margin: auto; display: block;" />
            <img src="data:image/png;base64,{logo_almap_dark}" class="logo-dark" style="max-height: 25px; margin: auto; display: block;" />
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

