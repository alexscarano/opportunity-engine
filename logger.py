import sys
import re
from src.utils import time_s
from pathlib import Path


def log(namespace: str, data: str) -> None:
    path = Path(f"./data/log/{namespace}_{time_s()}.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def translate_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return line

    # Checking for exact or prefix matches to translate to friendly Portuguese
    if "Loading, Cleaning, and Preparing Data..." in line:
        return "==================================================\nCarregando e preparando os dados enviados...\n=================================================="
    if "WARNING: Configured date format" in line:
        fmt_match = re.search(r"format '([^']+)'", line)
        fmt = fmt_match.group(1) if fmt_match else ""
        return f"   - AVISO: O formato de data configurado ({fmt}) falhou em algumas linhas. Ajustando automaticamente..."
    if "ERROR: Date parsing failed completely" in line:
        return "   - ERRO: Falha total na leitura das datas."
    if "WARNING: Generic trends file not found" in line:
        return "   - AVISO: Arquivo de tendências genéricas não encontrado. Continuando sem dados de tendência."
    if "INFO: No generic trends file path provided" in line:
        return "   - INFO: Nenhum arquivo de tendências fornecido. Continuando sem dados de tendência."
    if "Applying outlier treatment..." in line:
        return "   - Tratando valores atípicos nos dados (outliers)..."
    if "Treated outliers in KPI column" in line:
        col = re.search(r"column: '([^']+)'", line)
        col_name = col.group(1) if col else "KPI"
        return f"     - Valores atípicos tratados na coluna: '{col_name}'"
    if "KPI Data Date Range:" in line:
        match = re.search(r"KPI Data Date Range:\s*(.*?)\s*to\s*(.*)", line)
        if match:
            return f"   - Período dos dados de performance (KPI): {match.group(1)} até {match.group(2)}"
    if "Investment Data Date Range:" in line:
        match = re.search(r"Investment Data Date Range:\s*(.*?)\s*to\s*(.*)", line)
        if match:
            return f"   - Período dos dados de investimento: {match.group(1)} até {match.group(2)}"
    if "Data loaded and columns renamed successfully." in line:
        return "   - Dados carregados e colunas organizadas com sucesso!"
    if "Checking for negative correlations and applying adstock" in line:
        return "   - Analisando padrões de comportamento e influência de mídia..."
    if "Applying adstock to" in line:
        col = re.search(r"to '([^']+)'", line)
        col_name = col.group(1) if col else "canal"
        return f"     - Aplicando efeito de memória (adstock) no canal '{col_name}'..."
    if "Data preparation complete." in line:
        return "   - Preparação dos dados concluída com sucesso."
    if "Final Correlation Matrix" in line:
        return "==================================================\nTabela de Correlação entre Canais\n=================================================="
    if "Starting Comprehensive Event Detection & Grouping..." in line:
        return "==================================================\nIniciando busca por picos significativos de investimento...\n=================================================="
    if "Analyzing" in line and "unique ad products" in line:
        match = re.search(r"Analyzing\s*(\d+)", line)
        n = match.group(1) if match else "vários"
        return (
            f"   - Analisando {n} canais de anúncio em busca de grandes alterações..."
        )
    if "Detected events saved to:" in line:
        return f"   - Picos de investimento detectados salvos em: {line.split('saved to:')[-1].strip()}"
    if "Analyzing" in line and "candidates..." in line:
        match = re.search(r"Analyzing\s*(\d+)", line)
        n = match.group(1) if match else ""
        return f"==================================================\nAvaliando {n} possíveis picos de investimento...\n=================================================="
    if "▶ Analyzing Event:" in line:
        match = re.search(r"Analyzing Event:\s*(.*?)\s*on\s*(.*)", line)
        if match:
            return f"--------------------------------------------------\n▶ Analisando impacto do pico em '{match.group(1)}' no dia {match.group(2)}"
    if "FAILED: Causal impact analysis could not be completed." in line:
        return "   - FALHOU: Não foi possível isolar o impacto causal deste evento."
    if "PASSED: Event is statistically significant" in line:
        r_sq = re.search(r"R²\s*>=\s*([\d\.]+)", line)
        r_val = r_sq.group(1) if r_sq else "0.6"
        return f"   - APROVADO: Pico de investimento gerou impacto real comprovado (R² >= {r_val})."
    if "SKIPPED: Event did not meet validation criteria." in line:
        return "   - DESCONSIDERADO: Evento não passou nos critérios de validação estatística."
    if "Reason: p-value" in line:
        return "     - Motivo: Sem confiança estatística suficiente para comprovar o impacto."
    if "Reason: Investment change" in line:
        return "     - Motivo: A variação de investimento e o resultado não são consistentes entre si."
    if "Reason: Model R-squared" in line:
        r_sq = re.search(r"below the\s*([\d\.]+)", line)
        r_val = r_sq.group(1) if r_sq else "limite"
        return f"     - Motivo: A precisão do modelo estatístico ficou abaixo do limite mínimo ({r_val})."
    if "Analysis complete: No valid, impactful events were found." in line:
        return "\nAnálise concluída: Nenhum pico de investimento relevante passou na validação estatística."
    if "Events Selected. Generating Reports..." in line:
        match = re.search(r"Top\s*(\d+)", line)
        n = match.group(1) if match else ""
        return f"==================================================\nGerando relatórios para os {n} principais picos detectados...\n=================================================="
    if "Generating Report for Event:" in line:
        match = re.search(r"Event:\s*(.*?)\s*on\s*(.*)", line)
        if match:
            return f"--------------------------------------------------\nCriando relatório para '{match.group(1)}' do dia {match.group(2)}"
    if "SUCCESS! Comprehensive data saved to:" in line:
        path_val = line.split("saved to:")[-1].strip()
        return f"   SUCESSO! Dados detalhados salvos em: {path_val}"
    if "Generating Strategic Narrative with Gemini..." in line:
        return "   - Criando narrativa de negócios recomendada pela Inteligência Artificial Gemini..."
    if "SUCCESS! View the Gemini HTML report here:" in line:
        path_val = line.split("here:")[-1].strip()
        return f"   SUCESSO! Relatório estratégico gerado em: {path_val}"
    if "ERROR: Could not generate or parse the narrative from Gemini" in line:
        return f"   - ERRO: Falha ao obter narrativa da IA Gemini: {line.split('Details:')[-1].strip()}"
    if "Report generation failed for this event:" in line:
        return f"   - ERRO: Falha na geração do relatório: {line.split('event:')[-1].strip()}"
    if "All tasks complete." in line:
        return "==================================================\nTodas as tarefas concluídas com sucesso!\n=================================================="
    if "Analysis complete: No events met all criteria for reporting." in line:
        return "\nAnálise concluída: Nenhum evento atendeu a todos os critérios para geração de relatórios."
    if "Starting Global Elasticity Analysis..." in line:
        return "==================================================\nIniciando modelagem global de elasticidade e retorno (MMM)...\n=================================================="
    if "Generating Global Strategic Narrative with Gemini..." in line:
        return "   - Gerando recomendações globais de verba com a IA Gemini..."
    if "SUCCESS! Global strategic analysis complete." in line:
        return "   SUCESSO! Análise estratégica global concluída."
    if "ERROR: Global MMM analysis failed." in line:
        return "   - ERRO: A modelagem de elasticidade (MMM) global falhou."
    if "ERROR: Input file not found." in line:
        return f"ERRO: Arquivo de dados não encontrado. Verifique as configurações: {line.split('Details:')[-1].strip()}"
    if "ERROR: A data validation or processing error occurred." in line:
        return f"ERRO: Ocorreu um erro no processamento dos dados: {line.split('Details:')[-1].strip()}"
    if "A critical, unexpected error occurred during the main process:" in line:
        return f"ERRO CRÍTICO: Erro inesperado no processamento: {line.split('process:')[-1].strip()}"
    if "ERROR: Configuration file not found at" in line:
        return f"ERRO: Arquivo de configuração não encontrado."
    if "ERROR: Could not decode JSON from the configuration file" in line:
        return "ERRO: Arquivo de configuração inválido ou corrompido."
    if "Selected features for causal model:" in line:
        feats = re.search(r"model:\s*(.*)", line)
        feats_val = feats.group(1) if feats else ""
        return f"   - Variáveis adicionais selecionadas: {feats_val}"
    if "Causal model R-squared (in-sample):" in line:
        r2 = re.search(r"in-sample\):\s*(.*)", line)
        r2_val = r2.group(1) if r2 else ""
        return f"   - Precisão do modelo causal (R²): {r2_val}"
    if "Using pre-trained ad-stock alpha" in line:
        val = re.search(r"model:\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Utilizando taxa de retenção de mídia pré-treinada: {val_str}"
    if "Using pre-trained saturation params" in line:
        val = re.search(r"model:\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Utilizando saturação pré-treinada: {val_str}"
    if "Modeling baseline KPI using non-investment features..." in line:
        return "   - Calculando vendas de base (sem investimentos)..."
    if "Baseline subtracted. Modeling response on incremental" in line:
        return "   - Isolando vendas incrementais geradas pelos investimentos..."
    if "Best ad-stock alpha (incremental data):" in line:
        val = re.search(r"data\):\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Efeito de retenção acumulado ideal (adstock): {val_str}"
    if "Best saturation params (incremental data):" in line:
        val = re.search(r"data\):\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Parâmetros ideais de saturação de canais: {val_str}"
    if "Calculated channel investment proportions:" in line:
        val = re.search(r"proportions:\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Proporção de investimento calculada por canal: {val_str}"
    if "Incremental projection model training complete." in line:
        return "   - Treinamento do modelo de projeção incremental concluído."
    if "Analyzing historical data to find the optimal investment mix..." in line:
        return (
            "   - Analisando histórico para encontrar a melhor combinação de verbas..."
        )
    if "No historical investment data to analyze for optimal mix." in line:
        return "   - Sem histórico de investimentos suficiente para calcular combinação ideal."
    if "Could not identify top efficiency weeks." in line:
        return "   - Não foi possível identificar as semanas de maior eficiência."
    if "Found optimal historical mix from top" in line:
        val = re.search(r"top\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Combinação ótima histórica encontrada a partir das melhores {val_str}"
    if "Individual simulation data exported for UI:" in line:
        return f"   - Dados de simulação exportados com sucesso."
    if "Individual curve plot saved for" in line:
        val = re.search(r"saved for\s*(.*)", line)
        val_str = val.group(1) if val else ""
        return f"   - Gráfico de saturação gerado para: {val_str.strip()}"
    if "Chart saved to" in line:
        return "   - Gráfico consolidado gerado com sucesso."
    if "Successfully generated comparative MD file at:" in line:
        return "   - Arquivo de comparação global em Markdown gerado com sucesso."

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
            try:
                self.stdout.write(translated)
            except UnicodeEncodeError:
                enc = getattr(self.stdout, "encoding", "utf-8") or "utf-8"
                self.stdout.write(translated.encode(enc, errors="replace").decode(enc))
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
