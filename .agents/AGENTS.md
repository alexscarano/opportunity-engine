# AI Agent Rules & Guidelines - Max Impact Engine

This document defines project-specific rules, architectural patterns, and development guidelines for AI agents working on the **Max Impact Engine** (Total Opportunity) project.

---

## 1. Project Overview & Scope
The Max Impact Engine is a Python-based marketing analytics platform. It automates two distinct analytical stages:
1. **Stage 1 (Event-Level Causal Analysis):** Detects significant investment changes (`investment-data.csv`) and uses time-series modeling (`statsmodels`) to calculate incremental lift.
2. **Stage 2 (Global Saturation/Elasticity Analysis):** Models long-term channel contributions and diminishing returns (saturation response curves) subject to business financial guardrails.

---

## 2. Technology Stack & Environment
- **Runtime:** Python 3.13+ managed via `mise` (`mise.toml`) and `uv` as the package resolver/installer.
- **Execution:** Always execute Python commands using `mise exec -- X` (e.g., `mise exec -- python scripts/local_main.py`). If a specific command is executed repeatedly, create/define a task in `mise.toml` under `[tasks]` to simplify run commands.
- **Frontend:** Streamlit (`scripts/streamlit_app.py`) with Google Sign-in integration.
- **Data & Modeling:** `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `plotly`, `matplotlib`.
- **AI Integrations:** Google Gemini API (`google-generativeai`) for strategic text reports.
- **Environment Config:** `.env` for keys (e.g., `GEMINI_API_KEY`).

---

## 3. Directory Structure
- `scripts/`: Analytical core and UI.
  - `streamlit_app.py`: Streamlit frontend.
  - `local_main.py`: CLI execution entry point (online / with Gemini).
  - `local_main-without-gemini.py`: CLI entry point (offline fallback).
  - `analysis.py`, `elasticity_analysis.py`, `saturation_curve.py`: Math/statistical cores.
  - `gemini_report.py`, `recommendations.py`, `presentation.py`: Reporting modules.
- `src/`: Package utilities.
  - `src/utils.py`: Time formatting and other utilities.
- `logger.py`: Centralized logger context (`LogContext`) to capture stdout/stderr to files.
- `inputs/`: Location for advertiser configuration JSONs (e.g. `config.example.json`) and data files.
- `outputs/`: Standard output directories grouped by advertiser name.

---

## 4. Key Coding Rules & Guidelines

### Configuration-Driven Design
- **Never hardcode column names.** Always resolve columns using `column_mapping` parameters in the configuration:
  - `column_mapping.investment_file`: `date_col`, `channel_col`, `investment_col`
  - `column_mapping.performance_file`: `date_col`, `kpi_col`
  - `column_mapping.generic_trends_file`: `date_col`, `trends_col`
- Date columns must be dynamically standardized to a single column named `Date` during pre-processing.

### Optimization & Financial Guardrails
- Respect the `optimization_target` configuration: `"REVENUE"` (requires `average_ticket`) or `"CONVERSIONS"`.
- Support the following dynamic guardrails under `financial_targets` to prune saturation curves:
  - `target_cpa` (Average Cost Per Acquisition)
  - `target_icpa` (Incremental CPA)
  - `target_roas` (Return on Ad Spend)
  - `target_iroas` (Incremental ROAS)
- Support configurable outlier treatment with `treat_outliers` (Boolean or list of target columns).

### Execution Fallbacks
- Maintain strict compatibility between online (`local_main.py`) and offline (`local_main-without-gemini.py`) modes.
- If Gemini API is unavailable or offline mode is chosen, fall back cleanly to Markdown-based reporting (`RECOMMENDATIONS.md`).

### Logging & Error Handling
- **Never use `print()`.** Every module does `log = logging.getLogger(__name__)` and logs through it.
- `logger.py` owns the setup (`setup_logging`), wiring three handlers:
  - console (INFO+): the pt-BR line the end user reads in the Streamlit panel. `_RULES` translate the technical message and suppress noise.
  - `data/log/<namespace>.log` (DEBUG+, rotating): full record with timestamp, level, `module:line`, traceback.
  - `data/log/<namespace>.errors.log` (WARNING+, rotating): triage without scanning the full log.
- Pick the level by impact, not verbosity:
  - `DEBUG` internal parameters, per-iteration detail, resolved paths
  - `INFO` pipeline milestones the user should see
  - `WARNING` degraded but continuing: fallback used, event skipped, bad input tolerated
  - `ERROR` one unit of work failed (an event, a report) -- the pipeline goes on
  - `CRITICAL` the run cannot continue
- Log exceptions with `exc_info=True` instead of `traceback.print_exc()` -- the traceback lands in the file log, never in the user's panel.
- New user-facing message: add a `(regex, template)` row to `_RULES` in `logger.py`. Without a rule the raw English message reaches the panel.
- CLI entrypoints wrap `main()` in `LogContext(namespace)`, which calls `setup_logging` and routes any residual `print()` through logging.
- Handle individual event errors gracefully so they do not crash the entire pipeline run when processing multiple channels or events.
