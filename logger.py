import datetime
import re
import sys
from pathlib import Path


def time_s() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def log(namespace: str, data: str) -> None:
    path = Path(f"./data/log/{namespace}_{time_s()}.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


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


class LogContext:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log_data = []
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            translated = translate_line(line)
            if translated is None:
                continue
            try:
                self.stdout.write(translated + "\n")
            except UnicodeEncodeError:
                enc = getattr(self.stdout, "encoding", "utf-8") or "utf-8"
                self.stdout.write(
                    (translated + "\n").encode(enc, errors="replace").decode(enc)
                )
            self.log_data.append(translated + "\n")

    def flush(self):
        if self.buffer:
            translated = translate_line(self.buffer)
            if translated is not None:
                try:
                    self.stdout.write(translated)
                except UnicodeEncodeError:
                    enc = getattr(self.stdout, "encoding", "utf-8") or "utf-8"
                    self.stdout.write(
                        translated.encode(enc, errors="replace").decode(enc)
                    )
                self.log_data.append(translated)
            self.buffer = ""
        self.stdout.flush()
        self.stderr.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if exc_type:
            import traceback

            tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.write(tb_str)
        log(self.namespace, "".join(self.log_data))
