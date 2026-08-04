# UX do erro de cadência divergente entre CSVs

## Contexto

Investigação a pedido do usuário sobre um erro intermitente ao rodar o projeto `Pmax_VW_dynamic`:

```
ERRO CRÍTICO: Erro inesperado no processamento: An unexpected error occurred during data
preparation: Cadências divergentes entre o arquivo de investimento (7 dia(s)) e o arquivo de
performance (1 dia(s)). Não é possível conciliar automaticamente séries com cadências
diferentes -- verifique se ambos os arquivos reportam na mesma frequência (diária/semanal/mensal).
ERRO CRÍTICO: Erro inesperado no processamento: 1
```

**Causa raiz confirmada:** `detect_cadence` (`scripts/data_preprocessor.py:198`) infere a
cadência de cada arquivo a partir da mediana da distância entre datas únicas consecutivas — não
lê config nem metadado. No caso relatado, o `investment.csv` real (subido por outra pessoa, com
os mesmos nomes de arquivo do exemplo local mas conteúdo diferente) tinha ~52 datas únicas/ano
(mediana ~7 dias) contra ~365 no `performance.csv` (mediana ~1 dia). O guard-rail em
`data_preprocessor.py:863` (`abs(inv_cadence - kpi_cadence) > 2`) existe de propósito, adicionado
no plano `2026-07-27-ingestao-confiavel-v2.md` — antes dele o pipeline misturava cadências
diferentes silenciosamente e gerava eventos falsos e orçamento ~4x superestimado. Hipótese mais
provável para a causa dos dados: o seletor de segmentação de data do Google Ads ("Dia" vs
"Semana") ficou em "Semana" na exportação do arquivo de investimento — fica salvo por
sessão/navegador, então é fácil acontecer sem querer numa exportação e não na próxima. **Fora do
escopo desta spec** validar essa hipótese linha a linha (precisaria do CSV real, que não persiste
localmente) — o foco aqui é dar ao usuário como identificar e evitar o problema sozinho.

Dois problemas secundários, de apresentação, tornaram o erro mais confuso do que precisava ser:

1. **Linha duplicada e sem sentido no fim do log** (`"...: 1"`): `local_main.py` já loga o erro
   real nos blocos `except ValueError`/`except Exception` (linhas 686-697) e em seguida chama
   `exit(1)` de propósito. Isso levanta `SystemExit`, que escapa do bloco `with
   LogContext("local_main")` e é pego por `LogContext.__exit__` (`logger.py:501-506`), que trata
   qualquer exceção que escape como um crash não tratado e loga uma segunda linha crítica —
   `str(SystemExit(1))` é `"1"`, daí o `"ERRO CRÍTICO: ...: 1"` final sem contexto nenhum.
2. **Erro real escondido dentro de um log de até 300 linhas**: quando a run falha,
   `streamlit_app.py:1567-1569` mostra só `"Houve um erro na execução do motor. Verifique os logs
   acima."` — a mensagem que realmente explica o problema (a frase sobre cadências divergentes)
   fica em algum lugar dentro do painel de log rolável, sem destaque.
3. **Nenhum aviso prévio**: os `file_uploader` de Investimento/Performance
   (`streamlit_app.py:1191-1199`) já mencionam que aceitam dado diário/semanal/mensal, mas não
   deixam explícito que os dois arquivos de um mesmo projeto precisam usar a **mesma** frequência
   entre si.

## Escopo

Três mudanças pequenas e independentes, só em logging/apresentação — nenhuma mudança na lógica de
detecção de cadência ou de parsing de dados, que já está correta.

**Fora de escopo:** dicas de correção específicas por tipo de erro (ex: sugerir "confira a
segmentação Dia/Semana no Google Ads" quando for especificamente erro de cadência) — decidido com
o usuário durante o brainstorm para manter a primeira versão simples; a mensagem de erro já
existente já é acionável o suficiente uma vez visível. Também fora de escopo: validação
client-side (mostrar a cadência detectada de cada arquivo antes de rodar a engine inteira) — mais
esforço, não pedido nesta rodada.

## Mudanças

### 1. `logger.py` — não duplicar o erro quando a saída é deliberada

`LogContext.__exit__` loga qualquer exceção que escape do `with` como se fosse um crash não
tratado. `SystemExit` levantado por um `exit(1)` já precedido de `log.critical(...)` é uma saída
deliberada — o erro real já foi logado por quem chamou `exit()`. Ignorar `SystemExit` aqui:

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    ...
    if exc_type and not issubclass(exc_type, SystemExit):
        logging.getLogger(self.namespace).critical(
            "A critical, unexpected error occurred during the main process: %s",
            exc_val,
            exc_info=(exc_type, exc_val, exc_tb),
        )
    ...
```

O exit code do processo continua 1 (o `exit(1)` original não muda) — o Streamlit ainda detecta a
falha via `return_code != 0`. Só some a segunda linha sem contexto.

### 2. `streamlit_app.py` — destacar a mensagem de erro real

Em vez de só `"Houve um erro na execução do motor. Verifique os logs acima."`, extrair a última
linha do log que contenha `"ERRO"` (cobre tanto `"ERRO:"` quanto `"ERRO CRÍTICO:"`, os únicos
prefixos que `logger.py` usa para falhas fatais) e mostrar em destaque:

```python
else:
    error_line = next((l for l in reversed(log_lines) if "ERRO" in l), None)
    msg = "Houve um erro na execução do motor."
    if error_line:
        msg += f"\n\n{error_line.strip()}"
    status_container.error(msg + "\n\nVerifique os logs acima para mais detalhes.")
```

Mantém o log completo abaixo como já é hoje (suporte/debug), só adiciona destaque pra frase que
importa. Se por algum motivo nenhuma linha bater (`error_line is None`), cai no texto genérico de
hoje — sem regressão.

### 3. `streamlit_app.py` — avisar sobre cadência já no upload

Adicionar ao `help=` dos dois `file_uploader` (Investimento e Performance,
`streamlit_app.py:1191-1199`) que os dois arquivos de um projeto precisam reportar na mesma
frequência:

```python
inv_file = st.file_uploader(
    "Investimento (obrigatório)",
    type=["csv"],
    help="Investimento por canal de mídia (diário, semanal ou mensal). Precisa ter a mesma "
    "frequência do arquivo de Performance -- não é possível misturar um diário com outro "
    "semanal.",
)
perf_file = st.file_uploader(
    "Performance (obrigatório)",
    type=["csv"],
    help="Histórico de resultados/KPIs (diário, semanal ou mensal). Precisa ter a mesma "
    "frequência do arquivo de Investimento -- não é possível misturar um diário com outro "
    "semanal.",
)
```

## Testes

- `logger.py`: teste unitário de `LogContext` cobrindo que uma `SystemExit` dentro do bloco `with`
  não gera uma segunda entrada `critical` no logger (só a exceção original, se houver, ou nenhuma
  se o código já tratou e chamou `exit()` explicitamente).
- `streamlit_app.py`: não tem suíte de testes automatizados hoje para essa tela (é Streamlit,
  renderização manual) — verificação é manual, rodando a UI localmente e forçando os dois cenários
  (arquivo com cadência divergente; falha de config) e conferindo a mensagem exibida.

## Verificação

1. `mise run test` (ou equivalente) verde, incluindo o novo teste de `LogContext`.
2. Rodar a UI localmente (`mise run dev` ou equivalente), subir um par investimento
   semanal/performance diário forjado, e confirmar:
   - o painel de log mostra só **uma** linha `ERRO CRÍTICO` (sem o `"...: 1"` duplicado);
   - a caixa de erro na tela mostra a frase real ("Cadências divergentes...") em destaque, não só
     "verifique os logs acima";
   - o texto de ajuda dos dois `file_uploader` menciona a exigência de mesma frequência.
