import streamlit as st
import time

# Sync frame delay to prevent F5 flicker
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    time.sleep(0.05)  # Allow WebSocket connection handshake to populate cookies in st.context
    st.rerun()

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


# Custom CSS for Premium Look
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .stMetric label {
        color: #5f6368;
        font-weight: 500;
    }
    .stMetric .css-1wivap2 {
        color: #1a73e8;
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
    }
    .insight-box {
        background-color: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 15px;
        border-radius: 4px;
        margin-top: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Ensure we can import modules from the local scripts directory
import sys

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from db import init_db, create_user, verify_user, add_user_project, get_user_projects, verify_project_ownership, delete_user_project, get_user_api_key, update_user_api_key
from streamlit_cookies_controller import CookieController

init_db()

# Initialize CookieController as a singleton in session state
if 'cookie_controller' not in st.session_state:
    st.session_state.cookie_controller = CookieController()
controller = st.session_state.cookie_controller

# Check cookies first to auto-login if session state is empty
if 'user_id' not in st.session_state:
    try:
        # 1. Read synchronously from request headers (works instantly on F5)
        cookie_user_id = st.context.cookies.get('user_id')
        cookie_username = st.context.cookies.get('username')
        
        # 2. Fallback to component get if headers cookies aren't loaded yet
        if not cookie_user_id or not cookie_username:
            cookie_user_id = controller.get('user_id')
            cookie_username = controller.get('username')
            
        if cookie_user_id and cookie_username:
            st.session_state['user_id'] = int(cookie_user_id)
            st.session_state['username'] = cookie_username
            st.rerun()
    except Exception:
        pass

# Authentication check
if 'user_id' not in st.session_state:
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; margin-bottom: 20px;">
            <h1 style="color: #1a73e8; font-size: 2.5rem; font-weight: 700;">Opportunity Engine</h1>
            <p style="color: #5f6368; font-size: 1.1rem;">Por favor, faça o login para acessar a plataforma de Otimização de Oportunidades.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        auth_mode = st.radio("Selecione a ação", ["Login", "Cadastrar Novo Usuário"])
        username = st.text_input("Usuário", key="login_username")
        password = st.text_input("Senha", type="password", key="login_password")
        
        if auth_mode == "Login":
            if st.button("Entrar", use_container_width=True):
                user_id = verify_user(username, password)
                if user_id:
                    st.session_state['user_id'] = user_id
                    st.session_state['username'] = username
                    # Persist cookies for 30 days
                    controller.set('user_id', str(user_id), max_age=30*86400)
                    controller.set('username', username, max_age=30*86400)
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
                        st.session_state['user_id'] = user_id
                        st.session_state['username'] = username
                        # Persist cookies for 30 days
                        controller.set('user_id', str(user_id), max_age=30*86400)
                        controller.set('username', username, max_age=30*86400)
                        st.success(f"Cadastro realizado! Bem-vindo, {username}!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    st.stop()


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
db_projects = get_user_projects(st.session_state['user_id'])
project_options = {name: path for name, path in db_projects}

# Initialize session state for config path
if "active_config_path" not in st.session_state:
    if project_options:
        st.session_state["active_config_path"] = list(project_options.values())[0]
    else:
        st.session_state["active_config_path"] = ""

# Enforce IDOR check on load
if st.session_state.get("active_config_path"):
    if not verify_project_ownership(st.session_state['user_id'], st.session_state['active_config_path']):
        st.session_state["active_config_path"] = ""

# Sidebar user card & Logout
st.sidebar.markdown(f"**Conectado como:** {st.session_state['username']}")
if st.sidebar.button("Sair/Logout", use_container_width=True):
    controller.remove('user_id')
    controller.remove('username')
    del st.session_state['user_id']
    del st.session_state['username']
    if 'active_config_path' in st.session_state:
        del st.session_state['active_config_path']
    st.rerun()
st.sidebar.markdown("---")

st.sidebar.header("Projetos Anteriores")
if project_options:
    option_keys = list(project_options.keys())
    current_index = 0
    for i, key in enumerate(option_keys):
        if project_options[key] == st.session_state.get("active_config_path"):
            current_index = i
            break

    selected_project = st.sidebar.selectbox(
        "Selecione um Projeto:", options=option_keys, index=current_index
    )
    if selected_project:
        st.session_state["active_config_path"] = project_options[selected_project]
        
    # Project Deletion UI
    if st.sidebar.button("Excluir Projeto Atual", key="delete_proj_btn", use_container_width=True):
        st.session_state['show_delete_confirm'] = True

    if st.session_state.get('show_delete_confirm'):
        st.sidebar.warning(f"Deseja mesmo excluir o projeto '{selected_project}' e todos os seus arquivos?")
        col_del1, col_del2 = st.sidebar.columns(2)
        with col_del1:
            if st.button("Sim, Excluir", key="confirm_delete_btn", use_container_width=True):
                path_to_delete = st.session_state['active_config_path']
                # Verify ownership to prevent IDOR on delete
                if verify_project_ownership(st.session_state['user_id'], path_to_delete):
                    import shutil
                    # Delete files on disk
                    project_dir = os.path.dirname(path_to_delete)
                    if os.path.exists(project_dir):
                        shutil.rmtree(project_dir)
                    
                    # Delete output directory if it exists
                    user_output_dir = os.path.join("outputs", f"user_{st.session_state['user_id']}", selected_project)
                    if os.path.exists(user_output_dir):
                        shutil.rmtree(user_output_dir)
                    
                    # Delete from SQLite
                    delete_user_project(st.session_state['user_id'], selected_project)
                    
                    st.toast(f"Projeto '{selected_project}' excluído com sucesso.")
                    st.session_state['active_config_path'] = ""
                    st.session_state['show_delete_confirm'] = False
                    st.rerun()
                else:
                    st.error("Erro: Acesso não autorizado.")
        with col_del2:
            if st.button("Cancelar", key="cancel_delete_btn", use_container_width=True):
                st.session_state['show_delete_confirm'] = False
                st.rerun()
else:
    st.sidebar.info("Nenhum projeto encontrado. Faça o setup de um novo.")


tab1, tab2, tab3 = st.tabs(
    ["Setup & Execução", "Dashboard de Causal Impact", "Dashboard de Elasticidade"]
)

with tab1:
    st.header("Configuração de Nova Análise")
    st.markdown(
        "Faça o upload dos seus dados e configure os parâmetros financeiros para rodar um novo Motor de Oportunidades."
    )

    with st.form("setup_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Configurações Gerais")
            advertiser_name = st.text_input(
                "Nome do Projeto (Anunciante)", value="Meu_Projeto_Dynamic"
            )
            # Fetch saved API Key from SQLite
            saved_key = get_user_api_key(st.session_state['user_id']) or ""
            gemini_key = st.text_input(
                "Gemini API Key",
                value=saved_key,
                type="password",
                help="Necessária para geração de insights automáticos.",
            )

            env_key = os.environ.get("GEMINI_API_KEY")
            active_key = gemini_key if gemini_key else env_key
            model_options, models_info = get_available_gemini_models(active_key)

            gemini_model = st.selectbox(
                "Gemini Model",
                options=model_options,
                format_func=lambda x: models_info.get(x, x),
                help="Selecione o modelo do Gemini ideal para a sua tarefa. Se uma chave de API válida for informada, os modelos serão filtrados dinamicamente.",
            )

            st.subheader("Dados Brutos (CSV)")
            inv_file = st.file_uploader(
                "Dados de Investimento (obrigatório)", type=["csv"]
            )
            perf_file = st.file_uploader(
                "Dados de Performance (obrigatório)", type=["csv"]
            )
            trends_file = st.file_uploader(
                "Dados de Tendências (opcional)", type=["csv"]
            )

        with col2:
            st.subheader("Parâmetros do Negócio")
            kpi_column = st.text_input(
                "Nome da Coluna do KPI (Ex: Sessions, Conversions)", value="Sessions"
            )
            optimization_target = st.selectbox(
                "Alvo de Otimização", ["CONVERSIONS", "REVENUE"]
            )

            conversion_rate = (
                st.number_input(
                    "Taxa de Conversão KPI -> Venda (%)", value=1.0, step=0.1
                )
                / 100.0
            )
            avg_ticket = st.number_input("Ticket Médio (R$)", value=100.0, step=10.0)

            st.subheader("Restrições de Eficiência (Opcional)")
            target_cpa = st.number_input(
                "Target CPA Máximo Permissível (R$)",
                value=0.0,
                help="0 = Sem restrição",
            )
            target_roas = st.number_input(
                "Target ROAS Mínimo Permissível", value=0.0, help="0 = Sem restrição"
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_btn = st.form_submit_button(
                "Construir Motor de Oportunidades", type="primary", use_container_width=True
            )
        with col_btn2:
            save_settings_btn = st.form_submit_button(
                "Salvar Configurações", type="secondary", use_container_width=True
            )

    if save_settings_btn:
        update_user_api_key(st.session_state['user_id'], gemini_key)
        st.success("Configurações salvas com sucesso!")

    if submit_btn:
        # Persist user's Gemini API Key in SQLite
        update_user_api_key(st.session_state['user_id'], gemini_key)
        
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
                dynamic_dir = os.path.join("inputs", f"user_{st.session_state['user_id']}", f"{safe_adv_name}_dynamic")
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
                add_user_project(st.session_state['user_id'], f"{safe_adv_name}_dynamic", config_path_gen)

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
                log_lines = full_log.split("\n")
                display_log = (
                    "\n".join(log_lines[-20:]) if len(log_lines) > 20 else full_log
                )
                log_container.code(
                    f"Engine de Oportunidades Rodando...\n{display_log}",
                    language="shell",
                )

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                status_container.success(
                    "Análise Causal e Otimização concluídas com sucesso!"
                )
                st.balloons()
                st.info(
                    "Os dados foram gerados! Explore as abas de Causal Impact e Elasticidade."
                )
            else:
                status_container.error(
                    "Houve um erro na execução do motor. Verifique os logs acima."
                )

with tab2:
    st.header("Análise de Causal Impact (Por Evento)")
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
                                caption = "Gráfico Resumo (Causal Impact)"
                            else:
                                caption = "Gráfico da Análise"

                            st.image(
                                png_file, caption=caption, use_container_width=True
                            )
                else:
                    st.warning("Nenhum relatório encontrado para este evento.")
        else:
            st.info(
                "Nenhum relatório de Causal Impact encontrado. Rode o motor na aba Setup ou verifique as restrições."
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
                "outputs", advertiser_name, "global_saturation_analysis"
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

            st.sidebar.header("Filtros de Limitação")
            st.sidebar.markdown(
                "Use estes limites para encontrar o ponto ótimo na curva."
            )

            max_inv_val = float(df["Monthly_Investment"].max())
            min_inv_val = float(df["Monthly_Investment"].min())

            max_budget_millions = st.sidebar.slider(
                "Orçamento Mensal Máximo",
                min_value=min_inv_val / 1e6,
                max_value=max_inv_val / 1e6,
                value=max_inv_val / 1e6,
                step=0.05,
                format="%.2fM",
            )
            max_budget = max_budget_millions * 1e6

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

            filtered_df = df[df["Monthly_Investment"] <= max_budget]

            if use_cpa_target and "CPA" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["CPA"] <= target_cpa]

            if use_icpa_target and "iCPA" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["iCPA"] <= target_icpa]

            if filtered_df.empty:
                st.warning(
                    "Nenhum cenário corresponde aos critérios selecionados. Flexibilize seus limites."
                )
            else:
                optimal_point = filtered_df.iloc[-1]

                incremental_kpis = df["Projected_Total_KPIs"].diff().fillna(0).values
                investment_steps = df["Daily_Investment"].diff().fillna(1).values
                first_derivative = incremental_kpis / investment_steps
                saturation_point = optimal_point
                if len(first_derivative) > 1:
                    initial_marginal_gain = first_derivative[1]
                    if initial_marginal_gain > 0:
                        saturation_threshold = initial_marginal_gain * 0.1
                        sat_indices = np.where(
                            first_derivative[1:] < saturation_threshold
                        )[0]
                        if len(sat_indices) > 0:
                            saturation_idx = sat_indices[0] + 1
                            if saturation_idx < len(df):
                                saturation_point = df.iloc[saturation_idx]

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
                scenario_df["iCPA"] = np.where(
                    scenario_df[f"{kpi_name} Incrementais"] > 0,
                    scenario_df["Investimento Incremental"]
                    / scenario_df[f"{kpi_name} Incrementais"],
                    0,
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
                col3.metric("Global CPA", value=f"R$ {cpa_val:,.2f}")

                icpa_val = optimal_point["iCPA"] if "iCPA" in optimal_point else 0.0
                if pd.isna(icpa_val):
                    icpa_val = 0.0
                col4.metric(
                    "Marginal iCPA",
                    value=f"R$ {icpa_val:,.2f}" if icpa_val > 0 else "N/A",
                    delta_color="inverse",
                )

                st.markdown("---")
                st.markdown("### Curva de Saturação de Investimentos")

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

                # --- NEW: Individual Curves Visualization ---
                st.markdown("---")
                st.markdown("### Curvas de Resposta Individuais por Canal")

                ind_csv_path = os.path.join(
                    output_dir, "individual_response_curves_data.csv"
                )
                if os.path.exists(ind_csv_path):
                    ind_df = pd.read_csv(ind_csv_path)
                    channels = ind_df["Channel"].unique()

                    selected_channel = st.selectbox(
                        "Selecione um Canal para Visualizar a Curva", channels
                    )

                    # Sanitize channel name for filename
                    safe_channel_name = "".join(
                        [
                            c if c.isalnum() or c in ["-", "_"] else "_"
                            for c in selected_channel
                        ]
                    )
                    img_path = os.path.join(
                        output_dir, f"individual_response_curve_{safe_channel_name}.png"
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
                        hist_df["Formatted_Budget"] = hist_df["Budget"].apply(
                            lambda x: (
                                f"R$ {x / 1e6:.1f}M"
                                if x >= 1e6
                                else (
                                    f"R$ {x / 1e3:.1f}k" if x >= 1e3 else f"R$ {x:,.0f}"
                                )
                            )
                        )
                        fig_hist = px.pie(
                            hist_df,
                            values="Budget",
                            names="Channel",
                            title="Alocação Histórica",
                            hole=0.4,
                            custom_data=["Formatted_Budget"],
                        )
                        fig_hist.update_traces(
                            textposition="inside",
                            texttemplate="%{label}<br>%{percent}<br>%{customdata[0]}",
                        )
                        fig_hist.update_layout(showlegend=False)
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
                        strat_df["Formatted_Budget"] = strat_df["Budget"].apply(
                            lambda x: (
                                f"R$ {x / 1e6:.1f}M"
                                if x >= 1e6
                                else (
                                    f"R$ {x / 1e3:.1f}k" if x >= 1e3 else f"R$ {x:,.0f}"
                                )
                            )
                        )
                        fig_strat = px.pie(
                            strat_df,
                            values="Budget",
                            names="Channel",
                            title="Alocação Recomendada",
                            hole=0.4,
                            custom_data=["Formatted_Budget"],
                        )
                        fig_strat.update_traces(
                            textposition="inside",
                            texttemplate="%{label}<br>%{percent}<br>%{customdata[0]}",
                        )
                        fig_strat.update_layout(showlegend=False)
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
