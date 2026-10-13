# Max Impact Engine
** THIS IS NOT A GOOGLE OFFICIAL TOOL.**
This Python application automates a comprehensive marketing analytics workflow. It begins by analyzing historical data to find and validate the impact of specific marketing campaigns, then performs a holistic **Global Saturation Analysis** to provide strategic, forward-looking budget recommendations based on diminishing returns.

## Features

*   **Interactive Dashboard UI:** Run analyses seamlessly via a complete, interactive Streamlit frontend with local user accounts (SQLite), per-user projects, and visual file uploaders.
*   **Interactive Dashboards:** Plotly-based *Impacto Causal* and *Elasticidade* tabs (iCPA curve, revenue/ROI curve, channel mix evolution, saturation comparison, events overview).
*   **AI-Assisted Setup:** A "Sugerir com IA" button uses Gemini to prefill KPI, monetary and objective fields, and causal-analysis thresholds are auto-suggested from the investment history.
*   **Robust CSV Ingestion:** Detects data cadence (daily, weekly, monthly, irregular), prunes partial periods, parses dates per whole column, handles datetime-with-time values and BI export footer rows, and reports clear errors on column or cadence mismatches.
*   **Configuration Driven:** All parameters and file paths can also be managed in central `config.json` files for CLI execution.
*   **Automated Event Detection:** Scans investment data to automatically find and validate periods of significant budget changes.
*   **Causal Impact Analysis:** Uses `statsmodels` to build a time-series model that isolates the incremental impact of past marketing campaigns.
*   **Global Elasticity Analysis:** After analyzing individual events, the script runs a holistic analysis on the entire dataset to model long-term channel contributions and diminishing returns.
*   **Dynamic Financial Guardrails:** Strictly bounds investment recommendations based on real-world business constraints like Target CPA and Target ROAS.
*   **Automated Reporting:** Generates detailed HTML reports with strategic narratives powered by the Gemini API, alongside clean offline CSV and Markdown fallbacks.
*   **Structured Logging:** A `logging`-based setup (`logger.py`) shows pt-BR messages in the UI and writes full technical logs to `data/log/`.

---
## How the Analysis Works

The script is a powerful engine that runs a complete analysis in two distinct stages:

### Stage 1: Event-Level Causal Analysis

1.  **Event Detection:** The script first analyzes the `investment-data.csv` file to find significant changes in spending, flagging any period where investment changed beyond the thresholds defined in your config file.

2.  **Causal Impact Modeling:** For each significant event, a causal impact analysis is performed. This model forecasts what your business results *would have been* without the investment change. The difference between the actual results and this forecast is the **incremental lift**, proving the true impact of your campaign. Significance is measured from the state-space model's own uncertainty, and each event must pass the validation gates set in the config (`min_pre_period_days`, `require_statistical_significance`, `require_logical_direction`, `require_model_fit` with `r_squared_threshold`).

3.  **Event-Level Reporting:** For each event that passes statistical validation, the script generates a detailed report, including a saturation curve for that specific channel mix.

### Stage 2: Global Strategic Analysis

After analyzing individual events, the script moves to a higher-level, strategic analysis of your entire business.

4.  **Global Elasticity Modeling:** The script runs a holistic analysis on your complete historical dataset. This model determines the long-term contribution of each individual marketing channel while accounting for ad-stock and saturation (diminishing returns).

5.  **Global Report Generation:** All the findings from the global analysis, including comparative charts and response curves, are compiled into a final, comprehensive `global_report.html`. This report focuses purely on causal validation and optimal saturation points, offering a clear, data-driven narrative supported by Gemini.

---

## Customization

This project is designed to be adaptable. For detailed instructions on how to map the script to your specific CSV column names, change KPIs, or fine-tune the financial limits, please refer to our detailed **[Advanced Customization Guide](CUSTOMIZATION.md)**.

---

## Getting Started

### 1. Prerequisites
- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)

### 2. Installation

**a. Clone the repository:**
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**b. Install the dependencies:**
```bash
uv sync
```

### 3. Running the Streamlit UI Dashboard (Recommended)

The easiest and most interactive way to run the Max Impact Engine (Total Opportunity) is via its built-in dashboard.

```bash
uv run streamlit run scripts/streamlit_app.py
```

This will open a browser window at `http://localhost:8501`. 
1. Log in, or create an account with **"Cadastrar Novo Usuário"** (users, sessions and projects are stored in `data/database.db`).
2. Navigate to the **Setup & Execução** tab and enter your Gemini API key (saved per user) or set `GEMINI_API_KEY` in `.env`.
3. Upload your CSV files (`investment`, `performance`, and optionally `trends`). Investment and performance files must share the same cadence.
4. Optionally click **"Sugerir com IA"** to prefill the form, then click **"Construir Motor de Oportunidades"**.

The application will dynamically generate your configuration, run the engines, and print logs directly to your UI. If a run fails, the real error message is shown in the panel.

### 4. Running via the Command Line (CLI)

If you prefer terminal execution or automation pipelines, you can define your `config.json` manually and call the main engine directly.

**a. Set up your Gemini API Key:**
Create a file named `.env` in the root of the project directory and add your API key:
```
GEMINI_API_KEY="your_api_key_here"
```

**b. Run the Main Script:**
```bash
uv run python scripts/local_main.py --config inputs/your_project/my_config.json
```
*Note: If you do not have a Gemini API key or want to run entirely offline, use `uv run python scripts/local_main-without-gemini.py ...` instead. It will generate `RECOMMENDATIONS.md` instead of HTML.*

### 5. Running with Docker

```bash
docker build -t opportunity-engine .
docker run -p 8501:8501 --env-file .env   -v "$PWD/data:/app/data" -v "$PWD/inputs:/app/inputs" -v "$PWD/outputs:/app/outputs"   opportunity-engine
```

The image runs Streamlit as a non-root user with a health check on `/_stcore/health`. Mount `data/`, `inputs/` and `outputs/` to persist the database, logs and reports.

### 6. Running the Tests

```bash
uv run --with pytest pytest
```

---

## Outputs

The script generates two main types of outputs inside the `outputs/` directory, organized by advertiser name.

### 1. Global Strategic Report (Primary Output)

This is the main output of the analysis, providing your engine's holistic validation.

- **Location:** `outputs/<advertiser_name>/global_saturation_analysis/`
- **Key Files:**
    - `global_report.html`: The final, comprehensive HTML report with the Gemini-powered narrative. *(Or `RECOMMENDATIONS.md` if running offline).*
    - `SATURATION_CURVE.md`: A markdown file with detailed metrics on your global mix elasticity.
    - `response_curve_data.csv`: A raw data extract of simulated budgets vs predicted KPI / Revenue for visualization pipelines.
    - `combined_all_channels_saturation_curve.png`: The aggregated saturation curve for your business.

### 2. Event-Specific Reports

For each individual marketing event that passes isolation and causal validation, a report is generated.

- **Location:** `outputs/<advertiser_name>/<product_group>/<event_date>/`
- **Key Files:**
    - `gemini_report_... .html`: A detailed HTML report for that specific event.
    - `RECOMMENDATIONS.md` and `SATURATION_CURVE.md`: Markdown files with the event-specific analysis.
    - Various `causal_impact...png` chart files.

### 3. Logs

- **Location:** `data/log/`
- `<namespace>.log` holds the full technical record (DEBUG+, with tracebacks); `<namespace>.errors.log` keeps only warnings and errors for quick triage.
