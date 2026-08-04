# UX do erro de cadência divergente entre CSVs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tornar o erro de "cadências divergentes" (e qualquer outra falha fatal do motor) legível e acionável na UI, e avisar o usuário sobre a exigência de mesma frequência entre os arquivos antes mesmo de rodar.

**Architecture:** Três mudanças pequenas e independentes, só em logging/apresentação — nenhuma mudança na lógica de detecção de cadência ou parsing de dados (`data_preprocessor.py`), que já está correta. (1) `logger.py`: `LogContext.__exit__` para de tratar um `SystemExit` deliberado (de um `exit(1)` que já logou o erro real) como um segundo crash. (2) `scripts/streamlit_app.py`: ao falhar, extrai a última linha `"ERRO"`/`"ERRO CRÍTICO"` do log e mostra em destaque em vez do genérico "verifique os logs acima". (3) `scripts/streamlit_app.py`: help text dos uploaders de Investimento/Performance deixa explícito que os dois arquivos precisam ter a mesma frequência.

**Tech Stack:** Python, `unittest` (padrão do repo, ver `tests/test_logger.py`), Streamlit.

**Spec:** `docs/superpowers/specs/2026-08-04-erro-cadencia-csv-ux-design.md`

---

### Task 1: `LogContext` não duplica o erro quando a saída é um `exit()` deliberado

**Contexto para quem for implementar:** `local_main.py` (o script que a UI chama por trás) já loga
o erro real dentro dos blocos `except ValueError`/`except Exception` de `main()` e, em seguida,
chama `exit(1)` de propósito para sinalizar falha ao processo pai. `exit(1)` levanta `SystemExit`,
que escapa do bloco `with LogContext("local_main"):` em `local_main.py`. Hoje,
`LogContext.__exit__` (`logger.py:496-506`) trata **qualquer** exceção que escape do `with` como
se fosse um crash não tratado, e loga uma segunda linha crítica — como `str(SystemExit(1))` é só
`"1"`, isso produz uma linha confusa tipo `"ERRO CRÍTICO: Erro inesperado no processamento: 1"`
sem nenhum contexto, logo depois da linha real do erro. O fix é fazer `LogContext.__exit__` ignorar
especificamente `SystemExit` (é uma saída deliberada — quem chamou `exit()` já logou o que
precisava) e continuar logando normalmente qualquer outra exceção não tratada.

**Files:**
- Modify: `logger.py:496-506`
- Test: `tests/test_logger.py`

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_logger.py`, dentro da classe `TestLoggingSetup` (mesma classe de
`test_log_context_records_unhandled_exception`, que já existe logo acima), adicionar:

```python
    def test_log_context_ignores_deliberate_system_exit(self):
        with self.assertRaises(SystemExit):
            with LogContext("ns"):
                raise SystemExit(1)

        errors = (Path(self._tmp.name) / "ns.errors.log").read_text(encoding="utf-8")
        self.assertNotIn("critical, unexpected error", errors)
```

Isso cobre as duas partes do contrato: o `SystemExit` ainda precisa propagar normalmente (pro
processo sair com código 1), mas `LogContext` não pode logar uma segunda linha crítica genérica
pra ele.

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run python -m pytest tests/test_logger.py -k test_log_context_ignores_deliberate_system_exit -v`

Expected: FAIL — `errors.log` contém `"critical, unexpected error"` (o comportamento atual loga a
segunda linha).

- [ ] **Step 3: Implementar o fix**

Em `logger.py`, trocar o `__exit__` de `LogContext` (linhas 496-506):

```python
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
```

(Única mudança: `if exc_type:` → `if exc_type and not issubclass(exc_type, SystemExit):`.)

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `uv run python -m pytest tests/test_logger.py -v`

Expected: PASS em todos os testes de `tests/test_logger.py`, incluindo o novo e
`test_log_context_records_unhandled_exception` (que precisa continuar passando — exceções que
não são `SystemExit` ainda devem ser logadas).

- [ ] **Step 5: Commit**

```bash
git add logger.py tests/test_logger.py
git commit -m "fix(logger): don't double-report a deliberate exit(1) as a second crash"
```

---

### Task 2: Mostrar a mensagem de erro real na UI quando a execução falha

**Contexto para quem for implementar:** Em `scripts/streamlit_app.py`, depois que o subprocesso do
motor termina, o código verifica `return_code`. Hoje, no caso de falha (`else`, linha 1567), a UI
só mostra `"Houve um erro na execução do motor. Verifique os logs acima."` — a frase que realmente
explica o problema (ex: a mensagem de cadências divergentes) fica em algum lugar dentro do
`log_container` de até 300 linhas, sem destaque. `log_lines` (lista populada a partir do stdout do
subprocesso, já com as linhas traduzidas para pt-BR pelo `logger.py`, incluindo o prefixo `"ERRO"`
tanto em `"ERRO:"` quanto em `"ERRO CRÍTICO:"`) já está disponível nesse ponto do código — é só
varrer de trás pra frente e pegar a última linha que contenha `"ERRO"`.

**Files:**
- Modify: `scripts/streamlit_app.py:1567-1570`

Não há suíte de testes automatizados para essa tela (é renderização Streamlit; confirmado que só
existe `tests/test_streamlit_throttle.py`, que testa uma função auxiliar de cadência de
renderização do log, não esse bloco). Verificação é manual — passos no final desta tarefa.

- [ ] **Step 1: Implementar a mudança**

Em `scripts/streamlit_app.py`, trocar o bloco `else` (linhas 1567-1570):

De:
```python
            else:
                status_container.error(
                    "Houve um erro na execução do motor. Verifique os logs acima."
                )
```

Para:
```python
            else:
                error_line = next(
                    (line for line in reversed(log_lines) if "ERRO" in line), None
                )
                msg = "Houve um erro na execução do motor."
                if error_line:
                    msg += f"\n\n{error_line.strip()}"
                msg += "\n\nVerifique os logs acima para mais detalhes."
                status_container.error(msg)
```

- [ ] **Step 2: Verificação manual**

Rodar a UI localmente (ex: `mise run dev`, ou o comando de start já usado neste projeto) e forçar
um erro de cadência divergente:

1. Criar dois CSVs de teste com cadências diferentes — por exemplo, reusar
   `exemplo_csv/investimento_pmax_semanal.csv` (semanal) como investimento e
   `exemplo_csv/VW/v2/performance_vendas via lead.csv` (diário) como performance.
2. Subir um novo projeto na UI com esses dois arquivos.
3. Rodar a engine e confirmar que a caixa de erro na tela mostra a frase real ("Cadências
   divergentes entre o arquivo de investimento (...) e o arquivo de performance (...)...") em
   destaque, não só "verifique os logs acima".
4. Confirmar que o painel de log mostra só **uma** linha `ERRO CRÍTICO` — sem o `"...: 1"`
   duplicado (efeito da Task 1).

- [ ] **Step 3: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "feat(streamlit): surface the real error message when a run fails"
```

---

### Task 3: Avisar sobre a exigência de mesma cadência já no upload

**Contexto para quem for implementar:** Os dois `st.file_uploader` de Investimento e Performance
(`scripts/streamlit_app.py:1191-1199`) já mencionam no `help=` que aceitam dado diário, semanal ou
mensal, mas não deixam explícito que os dois arquivos de um mesmo projeto precisam usar a **mesma**
frequência entre si — essa é justamente a causa do erro investigado nesta spec.

**Files:**
- Modify: `scripts/streamlit_app.py:1191-1199`

- [ ] **Step 1: Implementar a mudança**

De:
```python
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
```

Para:
```python
            inv_file = st.file_uploader(
                "Investimento (obrigatório)",
                type=["csv"],
                help="Investimento por canal de mídia (diário, semanal ou mensal). Precisa ter "
                "a mesma frequência do arquivo de Performance -- não é possível misturar um "
                "diário com outro semanal.",
            )
            perf_file = st.file_uploader(
                "Performance (obrigatório)",
                type=["csv"],
                help="Histórico de resultados/KPIs (diário, semanal ou mensal). Precisa ter a "
                "mesma frequência do arquivo de Investimento -- não é possível misturar um "
                "diário com outro semanal.",
            )
```

(O `file_uploader` de `trends_file`, logo abaixo, não muda — tendência não entra no check de
cadência.)

- [ ] **Step 2: Verificação manual**

Rodar a UI localmente, abrir a tela de novo projeto, passar o mouse sobre o ícone de ajuda (`?`)
dos campos "Investimento" e "Performance" e confirmar que o texto novo aparece.

- [ ] **Step 3: Commit**

```bash
git add scripts/streamlit_app.py
git commit -m "docs(streamlit): warn that investment and performance CSVs must share cadence"
```

---

## Verificação final (depois das 3 tasks)

1. `uv run python -m pytest tests/test_logger.py -v` — verde.
2. `uv run python -m pytest` (suíte completa) — verde, sem regressão.
3. Fluxo manual descrito na Task 2 (cadências divergentes forçadas) de ponta a ponta: aviso no
   upload visível, uma única linha `ERRO CRÍTICO` no log, mensagem real destacada na tela.
