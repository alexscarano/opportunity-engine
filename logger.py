"""Logging setup for the engine.

Three handlers, three audiences:

- console (INFO+)   -> the pt-BR line the end user reads in the Streamlit panel.
                       `_RULES` translate the technical message and drop noise.
- <ns>.log (DEBUG+) -> full technical record: timestamp, level, module:line, traceback.
- <ns>.errors.log   -> WARNING+ only, for triage without scanning the full log.

Module code should do `log = logging.getLogger(__name__)` and pick the level by
impact, not by verbosity:

    DEBUG    internal parameters, per-iteration detail, resolved paths
    INFO     pipeline milestones the user should see
    WARNING  degraded but continuing: fallback used, event skipped, bad input tolerated
    ERROR    one unit of work failed (an event, a report) -- pipeline goes on
    CRITICAL the run cannot continue
"""

import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("./data/log")
_FILE_FORMAT = (
    "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(funcName)s() | %(message)s"
)
# ponytail: root runs at DEBUG, and these libraries are unusable at that level
# (matplotlib's font manager alone emits thousands of lines per run).
_NOISY = (
    "matplotlib",
    "PIL",
    "urllib3",
    "httpx",
    "httpcore",
    "google",
    "numexpr",
    "asyncio",
)


def _banner(text) -> str:
    return f"{'=' * 50}\n{text}\n{'=' * 50}"


def _divider(text) -> str:
    return f"{'-' * 50}\n{text}"


# ponytail: table-driven translation. Add a (regex, template) row for new log lines;
# template fields are filled from the regex capture groups, stripped.
_RULES = [
    (
        r"Loading, Cleaning, and Preparing Data\.\.\.",
        _banner("Carregando e preparando os dados enviados..."),
    ),
    (
        r"WARNING: Configured date format '([^']+)'",
        "   - AVISO: O formato de data configurado ({0}) falhou em algumas linhas. Ajustando automaticamente...",
    ),
    (
        r"ERROR: Date parsing failed completely",
        "   - ERRO: Falha total na leitura das datas.",
    ),
    (
        r"WARNING: Generic trends file not found",
        "   - AVISO: Arquivo de tendências genéricas não encontrado. Continuando sem dados de tendência.",
    ),
    (
        r"INFO: No generic trends file path provided",
        "   - INFO: Nenhum arquivo de tendências fornecido. Continuando sem dados de tendência.",
    ),
    (
        r"Applying outlier treatment\.\.\.",
        "   - Tratando valores atípicos nos dados (outliers)...",
    ),
    (
        r"Treated outliers in KPI column: '([^']+)'",
        "     - Valores atípicos tratados na coluna: '{0}'",
    ),
    (
        r"KPI Data Date Range:\s*(.*?)\s*to\s*(.*)",
        "   - Período dos dados de performance (KPI): {0} até {1}",
    ),
    (
        r"Investment Data Date Range:\s*(.*?)\s*to\s*(.*)",
        "   - Período dos dados de investimento: {0} até {1}",
    ),
    (
        r"Data loaded and columns renamed successfully\.",
        "   - Dados carregados e colunas organizadas com sucesso!",
    ),
    (
        r"Checking for negative correlations and applying adstock",
        "   - Analisando padrões de comportamento e influência de mídia...",
    ),
    (
        r"Applying adstock to '([^']+)'",
        "     - Aplicando efeito de memória (adstock) no canal '{0}'...",
    ),
    (
        r"Data preparation complete\.",
        "   - Preparação dos dados concluída com sucesso.",
    ),
    (r"Final Correlation Matrix", _banner("Tabela de Correlação entre Canais")),
    (
        r"Starting Comprehensive Event Detection & Grouping\.\.\.",
        _banner("Iniciando busca por picos significativos de investimento..."),
    ),
    (
        r"Analyzing\s*(\d+).*unique ad products",
        "   - Analisando {0} canais de anúncio em busca de grandes alterações...",
    ),
    (
        r"WARNING: threshold flagged (\d+) of (\d+) periods for '([^']+)' \(([^)]+)\)",
        "   - AVISO: o limiar marcou {0} de {1} períodos do canal '{2}' como evento "
        "({3}). Isso é oscilação normal de verba, não pico: aumente a 'Var. Mínima "
        "de Aumento/Queda de Investimento' para obter uma lista de eventos com significado.",
    ),
    (
        r"Detected events saved to:",
        "   - Picos de investimento detectados salvos com sucesso.",
    ),
    (
        r"Analyzing\s*(\d+).*candidates\.\.\.",
        _banner("Avaliando {0} possíveis picos de investimento..."),
    ),
    (
        r"▶ Analyzing Event \[(\d+/\d+)\]:\s*(.*?)\s*on\s*(.*)",
        _divider("▶ [{0}] Analisando impacto do pico em '{1}' no dia {2}"),
    ),
    (
        r"▶ Analyzing Event:\s*(.*?)\s*on\s*(.*)",
        _divider("▶ Analisando impacto do pico em '{0}' no dia {1}"),
    ),
    (
        r"FAILED: Causal impact analysis could not be completed\.",
        "   - FALHOU: Não foi possível isolar o impacto causal deste evento.",
    ),
    (
        r"PASSED: Event is statistically significant.*R²\s*>=\s*([\d.]+)",
        "   - APROVADO: Pico de investimento gerou impacto real comprovado (R² >= {0}).",
    ),
    (
        r"SKIPPED: Event did not meet validation criteria\.",
        "   - DESCONSIDERADO: Evento não passou nos critérios de validação estatística.",
    ),
    (
        r"Reason: p-value",
        "     - Motivo: Sem confiança estatística suficiente para comprovar o impacto.",
    ),
    (
        r"Reason: Investment change",
        "     - Motivo: A variação de investimento e o resultado não são consistentes entre si.",
    ),
    (
        r"Reason: Model R-squared.*below the\s*([\d.]+)",
        "     - Motivo: A precisão do modelo estatístico ficou abaixo do limite mínimo ({0}).",
    ),
    (
        r"Analysis complete: No valid, impactful events were found\.",
        "\nAnálise concluída: Nenhum pico de investimento relevante passou na validação estatística.",
    ),
    (
        r"Top\s*(\d+).*Events Selected\. Generating Reports\.\.\.",
        _banner("Gerando relatórios para os {0} principais picos detectados..."),
    ),
    (
        r"Generating Report for Event:\s*(.*?)\s*on\s*(.*)",
        _divider("Criando relatório para '{0}' do dia {1}"),
    ),
    (
        r"SUCCESS! Comprehensive data saved to:",
        "   SUCESSO! Dados detalhados salvos com sucesso.",
    ),
    (
        r"Generating Strategic Narrative with Gemini\.\.\.",
        "   - Criando narrativa de negócios recomendada pela Inteligência Artificial Gemini...",
    ),
    (
        r"SUCCESS! View the Gemini HTML report here:",
        "   SUCESSO! Relatório de recomendações em HTML gerado com sucesso.",
    ),
    (
        r"ERROR: Could not generate or parse the (?:full |global )?narrative from Gemini.*Details:\s*(.*)",
        "   - ERRO: Falha ao obter narrativa da IA Gemini: {0}",
    ),
    (
        r"Report generation failed for this event:\s*(.*)",
        "   - ERRO: Falha na geração do relatório: {0}",
    ),
    (r"All tasks complete\.", _banner("Todas as tarefas concluídas com sucesso!")),
    (
        r"Analysis complete: No events met all criteria for reporting\.",
        "\nAnálise concluída: Nenhum evento atendeu a todos os critérios para geração de relatórios.",
    ),
    (
        r"Starting Global Elasticity Analysis\.\.\.",
        _banner("Iniciando modelagem global de elasticidade e retorno (MMM)..."),
    ),
    (
        r"Starting Two-Stage Elasticity Analysis Engine\.\.\.",
        _divider("Modelando baseline organico e resposta de midia em duas etapas..."),
    ),
    (
        r"Training Response Model for Projection on INCREMENTAL Data\.\.\.",
        _divider("Treinando modelo de projecao sobre o resultado incremental..."),
    ),
    (
        r"Finding the Investment Sweet Spot & Projecting Opportunity\.\.\.",
        _divider(
            "Procurando o ponto ideal de investimento e projetando a oportunidade..."
        ),
    ),
    (
        r"Starting Global Saturation Analysis\.\.\.",
        _banner("Iniciando analise global de saturacao dos canais..."),
    ),
    (
        r"Global Saturation Analysis Complete\.",
        "   - Analise global de saturacao concluida.",
    ),
    (
        r"Generating Global Strategic Narrative with Gemini\.\.\.",
        "   - Gerando recomendações globais de verba com a IA Gemini...",
    ),
    (
        r"SUCCESS! Global strategic analysis complete\.",
        "   SUCESSO! Análise estratégica global concluída.",
    ),
    (
        r"ERROR: Global MMM analysis failed\.",
        "   - ERRO: A modelagem de elasticidade (MMM) global falhou.",
    ),
    (
        r"ERROR: Input file not found\..*Details:\s*(.*)",
        "ERRO: Arquivo de dados não encontrado. Verifique as configurações: {0}",
    ),
    (
        r"ERROR: A data validation or processing error occurred\..*Details:\s*(.*)",
        "ERRO: Ocorreu um erro no processamento dos dados: {0}",
    ),
    (
        r"A critical, unexpected error occurred during the main process:\s*(.*)",
        "ERRO CRÍTICO: Erro inesperado no processamento: {0}",
    ),
    (
        r"ERROR: Configuration file not found at",
        "ERRO: Arquivo de configuração não encontrado.",
    ),
    (
        r"ERROR: Could not decode JSON from the configuration file",
        "ERRO: Arquivo de configuração inválido ou corrompido.",
    ),
    (
        r"Selected features for causal model:\s*(.*)",
        "   - Variáveis adicionais selecionadas: {0}",
    ),
    (
        r"Causal model R-squared \(in-sample\):\s*(.*)",
        "   - Precisão do modelo causal (R²): {0}",
    ),
    (
        r"Using pre-trained ad-stock alpha.*model:\s*(.*)",
        "   - Utilizando taxa de retenção de mídia pré-treinada: {0}",
    ),
    (
        r"Using pre-trained saturation params.*model:\s*(.*)",
        "   - Utilizando saturação pré-treinada: {0}",
    ),
    (
        r"Modeling baseline KPI using non-investment features\.\.\.",
        "   - Calculando vendas de base (sem investimentos)...",
    ),
    (
        r"Baseline subtracted\. Modeling response on incremental",
        "   - Isolando vendas incrementais geradas pelos investimentos...",
    ),
    (
        r"Best ad-stock alpha \(incremental data\):\s*(.*)",
        "   - Efeito de retenção acumulado ideal (adstock): {0}",
    ),
    (
        r"Best saturation params \(incremental data\):\s*(.*)",
        "   - Parâmetros ideais de saturação de canais: {0}",
    ),
    (
        r"Calculated channel investment proportions:\s*(.*)",
        "   - Proporção de investimento calculada por canal: {0}",
    ),
    (
        r"Incremental projection model training complete\.",
        "   - Treinamento do modelo de projeção incremental concluído.",
    ),
    (
        r"Analyzing historical data to find the optimal investment mix\.\.\.",
        "   - Analisando histórico para encontrar a melhor combinação de verbas...",
    ),
    (
        r"No historical investment data to analyze for optimal mix\.",
        "   - Sem histórico de investimentos suficiente para calcular combinação ideal.",
    ),
    (
        r"Could not identify top efficiency weeks\.",
        "   - Não foi possível identificar as semanas de maior eficiência.",
    ),
    (
        r"Found optimal historical mix from top\s*(.*)",
        "   - Combinação ótima histórica encontrada a partir das melhores {0}",
    ),
    (
        r"Individual simulation data exported for UI:",
        "   - Dados de simulação exportados com sucesso.",
    ),
    (
        r"Individual curve plot saved for\s+([^:]+):",
        "   - Gráfico de saturação gerado para: {0}",
    ),
    (r"Chart saved to", "   - Gráfico consolidado gerado com sucesso."),
    (
        r"Successfully generated comparative MD file at:",
        "   - Arquivo de comparação global em Markdown gerado com sucesso.",
    ),
    (r"^={10,}$", None),
    (r"Generating Global Gemini Report\.\.\.", None),
    (
        r"Global narrative generated and parsed successfully\.",
        "   - Recomendações geradas com sucesso pela IA.",
    ),
    (
        r"Global Gemini HTML report saved successfully to:",
        "   - Relatório estratégico global em HTML gerado com sucesso.",
    ),
    (r"Assembling Gemini HTML report to", None),
    (r"Generating Markdown report to", None),
    (r"Markdown report generated successfully\.", None),
    (
        r"Gemini HTML report saved successfully\.",
        "   - Relatório de recomendações em HTML gerado com sucesso.",
    ),
    (r"^-{10,}$", None),
    (
        r"Running automated feature selection for causal model\.\.\.",
        "   - Analisando variáveis de contexto para o modelo causal...",
    ),
    (r"No additional performance covariates found\. Proceeding without them\.", None),
    (r"Warning\b", None),
    (r"site-packages", None),
    (r"^\s*warn\(", None),
    (r"^\s*self\._init_dates", None),
    (r"^\s*trend\s*=", None),
    (r"^\s*var\s*[*]?=", None),
    (r"^\s*return\s+get_prediction_index", None),
]


def translate_line(line: str) -> str | None:
    if not line.strip():
        return line
    for pattern, template in _RULES:
        m = re.search(pattern, line)
        if m:
            if template is None:
                return None
            return template.format(*(g.strip() for g in m.groups()))
    return line


class TranslationFilter(logging.Filter):
    """Console only: drop records the pt-BR ruleset marks as noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "py.warnings":
            return (
                False  # library warnings belong in the file log, not the user's panel
            )
        return translate_line(record.getMessage()) is not None


class TranslationFormatter(logging.Formatter):
    """Console only: render the pt-BR line, without traceback or metadata."""

    def format(self, record: logging.LogRecord) -> str:
        return translate_line(record.getMessage()) or ""


class _StreamToLogger:
    """Residual print()/library stdout writes, routed through logging.

    Keeps stray output from bypassing the handlers (and the translation filter).
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message: str) -> None:
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)

    def flush(self) -> None:
        if self.buffer.strip():
            self.logger.log(self.level, self.buffer)
        self.buffer = ""

    def isatty(self) -> bool:
        return False


def setup_logging(
    namespace: str,
    stream=None,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure the root logger. Idempotent: re-running replaces the handlers."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stream = stream if stream is not None else sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # already-detached or non-reconfigurable stream (tests, pipes)

    console = logging.StreamHandler(stream)
    console.setLevel(console_level)
    console.addFilter(TranslationFilter())
    console.setFormatter(TranslationFormatter())

    def _file(name: str, min_level: int, max_bytes: int, backups: int):
        handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / name,
            maxBytes=max_bytes,
            backupCount=backups,
            encoding="utf-8",
            delay=True,
        )
        handler.setLevel(min_level)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        return handler

    full = _file(f"{namespace}.log", logging.DEBUG, 5_000_000, 5)
    errors = _file(f"{namespace}.errors.log", logging.WARNING, 1_000_000, 3)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    root.handlers = [console, full, errors]

    logging.captureWarnings(True)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Rotation mixes runs in one file, so mark where each one starts.
    full.handle(
        logging.LogRecord(
            namespace,
            logging.INFO,
            __file__,
            0,
            "=== run started %s pid=%s ===",
            (datetime.now().isoformat(timespec="seconds"), os.getpid()),
            None,
            func="setup_logging",
        )
    )
    return root


class LogContext:
    """Set up logging for a CLI run and capture anything that still prints."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        setup_logging(self.namespace, stream=self.stdout)
        stray = logging.getLogger("stdout")
        sys.stdout = _StreamToLogger(stray, logging.INFO)
        sys.stderr = _StreamToLogger(stray, logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if exc_type and not issubclass(exc_type, SystemExit):
            logging.getLogger(self.namespace).critical(
                "A critical, unexpected error occurred during the main process: %s",
                exc_val,
                exc_info=(exc_type, exc_val, exc_tb),
            )
        for handler in logging.getLogger().handlers:
            handler.flush()
