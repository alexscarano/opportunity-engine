# Ingestão robusta de CSVs diversos (data_preprocessor.py)

## Contexto

Investigação a pedido do usuário sobre um crash ao processar os arquivos de exemplo em
`exemplo_csv/VW`: `TypeError: dtype 'str' does not support operation 'mean'` no
`daily_investment_df.pivot_table(...)` de `data_preprocessor.py:220`.

Causa raiz confirmada reproduzindo o bug: `load_and_prepare_data` usa `pd.read_csv(..., thousands=",")`
fixo (assume formato numérico US: `1,234.56`) e só a coluna `kpi` recebe limpeza manual de string
(linhas 135-147) — a coluna `investment` (e a coluna de trends `Generic Searches`) não recebem
tratamento numérico nenhum. Gerando um CSV com formato BR a partir de `exemplo_csv/VW/investimento.xlsx`
(`63.115,13` em vez de `63115.13`) reproduz o erro exato: a coluna fica `dtype=object` (string) e o
`pivot_table` quebra no agregador `mean`.

Esse é o sintoma de um problema mais amplo: o pipeline assume implicitamente um único "dialeto" de CSV
(delimitador vírgula, encoding UTF-8, números em formato US, nomes de coluna batendo exatamente com o
`column_mapping` do config). Como a ferramenta recebe uploads de clientes/planilhas diferentes, esse
conjunto de suposições quebra com frequência e sempre da mesma forma: um traceback profundo e pouco
acionável em vez de um erro claro na hora da leitura.

## Escopo

### Fora de escopo (explícito)
- Detecção de formato numérico *dentro da mesma coluna* quando há mistura real de dois dialetos na mesma
  planilha (célula a célula) — assume-se um dialeto consistente por arquivo, que é o caso real de
  exports de uma única ferramenta/planilha.
- Novas dependências externas (`chardet`, `babel`, etc.) — tudo resolvido com stdlib (`csv.Sniffer`) e
  pandas, que já estão no projeto.
- Configuração explícita de locale/delimitador no `config.json` — a meta é detecção automática; se a
  heurística falhar de forma sistemática para um cliente específico, isso é um problema separado a
  avaliar depois com dados reais.
- Qualquer mudança em `local_main.py`, `local_main-without-gemini.py`, `streamlit_app.py` além do que for
  estritamente necessário para consumir as novas funções.

### Fixes

Todos os itens abaixo vivem em `scripts/data_preprocessor.py`, seguindo o mesmo padrão que
`robust_date_parsing` já estabelece no arquivo (função pura, fallback automático, aviso via `print`
quando o caminho não-trivial é tomado).

#### 1. Leitura de CSV tolerante a delimitador/encoding (`read_csv_robust`)
Nova função que substitui as 3 chamadas `pd.read_csv(...)` (performance, investment, trends):
- Encoding: tenta `utf-8-sig` (lê UTF-8 normal e também remove BOM de exports do Excel-Windows sem
  precisar de tratamento especial), cai para `latin-1` em `UnicodeDecodeError` (`latin-1` nunca falha,
  é o fallback final).
- Delimitador: `csv.Sniffer().sniff()` numa amostra do arquivo (primeiros ~8KB), restrito a candidatos
  `,`, `;`, `\t`; se o Sniffer não conseguir decidir (`csv.Error`), usa `,` como default.
- Pós-leitura: `df.columns = df.columns.str.strip()` remove espaços acidentais nos headers.
- Loga um `AVISO` (mesmo estilo dos avisos de data já existentes) só quando o delimitador ou encoding
  detectado não é o default (`,` / utf-8 sem BOM), para não poluir o log no caso comum.
- Não recebe mais `thousands=`; parsing numérico fica inteiramente por conta do item 2, aplicado depois,
  coluna por coluna — decisão deliberada para não repetir o bug original (uma flag global de
  `read_csv` assumindo um único dialeto numérico para o arquivo inteiro).

#### 2. Resolução de coluna tolerante a espaço/case (`resolve_column`)
Hoje `daily_investment_df.rename(columns={inv_map.get("date_col", "dates"): "Date", ...})` falha em
silêncio se o nome configurado não bater exatamente com o header do CSV (`"Dates"` vs `"dates"`, ou com
espaço à volta) — o `rename` simplesmente não renomeia nada, e o erro real só aparece páginas depois como
`KeyError: 'Date'`, sem dizer qual coluna configurada é a culpada.

Nova função `resolve_column(df, configured_name, purpose)`:
1. Tenta match exato.
2. Tenta match tolerante (`.strip().lower()` dos dois lados).
3. Se não encontrar, `raise ValueError` citando o nome configurado, o `purpose` (ex.: `"coluna de data do
   arquivo de investimento"`) e a lista de colunas disponíveis no arquivo.

Usada para resolver todas as colunas hoje lidas via `inv_map`/`perf_map`/`trends_map` antes de renomear,
substituindo os `.get(...)` diretos passados pro `rename`.

#### 3. Parsing numérico robusto (`robust_numeric_parsing`)
Nova função aplicada uniformemente em `kpi_df["kpi"]`, `daily_investment_df["investment"]` e
`trends_df["Generic Searches"]` — substitui o bloco manual atual (linhas 135-147), que hoje só existe
para `kpi` e mesmo assim não trata formato BR de verdade (`1.234,56` quebra nele também; não apareceu no
log porque o crash do `investment` acontece antes).

Regras, nessa ordem:
1. Se a série já é numérica, retorna sem alteração (no-op).
2. Normaliza para string, `strip()`, remove símbolos de moeda (`R$`, `$`, `€`) e `%`.
3. Detecta negativo contábil (`(1.234,56)` → negativo) antes de remover os parênteses.
4. Trata tokens nulos comuns (`-`, `N/A`, `null`, `none`, string vazia, case-insensitive) como ausente
   antes de tentar converter — não conta como falha de parsing.
5. **Detecção de locale por coluna** (BR `.` milhar / `,` decimal vs US `,` milhar / `.` decimal), com
   votação por padrão de dígitos em vez de "tenta os dois, fica com o que falha menos" (esse critério
   sozinho não resolve `"1.234"`, que é um parse *válido* nos dois formatos com valores diferentes — um
   parse errado-mas-bem-sucedido é pior que um crash, porque não é percebido):
   - Valor com os dois separadores presentes: o que aparece mais à direita é o decimal (inequívoco).
   - Valor com um separador repetido (`"1.234.567"`): é milhar, nunca decimal (inequívoco).
   - Valor com um único separador e exatamente 2 dígitos depois: decimal (padrão de centavos).
   - Valor com um único separador e exatamente 3 dígitos depois, ocorrendo uma vez só: ambíguo por
     linha — cada linha ambígua vota no default; ver abaixo.
   - A coluna inteira decide por maioria dos votos não-ambíguos; empate ou coluna 100% ambígua usa US
     (milhar) como default — mantém o comportamento atual do código pra esse caso, e loga um `AVISO`
     avisando que a decisão foi por padrão (não detectada com confiança), com exemplos de valores.
6. Converte com `pd.to_numeric(errors="coerce")`; valores que não vieram de token nulo e ainda assim
   viraram `NaN` geram um `AVISO` com contagem e até 5 exemplos do valor original.

Os `dropna(subset=[...])` que já existem em `kpi_df`/`daily_investment_df` continuam sendo a rede de
segurança final para linhas que não deu pra recuperar.

#### 4. Normalização de case do canal (`daily_investment_df["Product Group"]`)
Hoje só `.str.strip()`. Adicionar `.str.upper()` logo em seguida. Sem isso, `"Google Ads"` e
`"GOOGLE ADS"` em linhas diferentes do mesmo export viram duas colunas diferentes no `pivot_table`,
fragmentando o investimento do mesmo canal em duas séries — silencioso, sem erro, só um resultado de
análise errado. Os dados de exemplo já vêm em uppercase (`"FACEBOOK & INSTAGRAM"`, `"GOOGLE ADS"`), então
não muda a saída pro caso atual — só protege o caso de mistura de case.

#### 5. Guarda de dataframe vazio pós-limpeza
Depois de cada `dropna(subset=[...])` (kpi, investment), checar `if df.empty: raise ValueError(...)`
citando qual arquivo (`config["performance_file_path"]` / `config["investment_file_path"]`) e sugerindo a
causa provável (formato de data ou número não reconhecido em nenhuma linha). Hoje, se isso acontecer, o
pipeline segue adiante com um dataframe vazio e só quebra páginas depois, num traceback sem relação óbvia
com a causa real.

## Testes

Novo `tests/test_data_preprocessor.py`, seguindo o padrão dos testes existentes (funções `pytest` diretas,
sem fixtures/framework extra):

- `robust_numeric_parsing`: coluna BR (`"1.234,56"`), coluna US (`"1,234.56"`), símbolo de moeda
  (`"R$ 63.115,13"`), negativo contábil (`"(1.234,56)"` → `-1234.56`), tokens nulos (`"-"`, `"N/A"`) viram
  `NaN` sem contar como falha, coluna já numérica é no-op, coluna ambígua (`"1.234"` sozinho, sem outro
  indício) cai no default US e loga aviso.
- `read_csv_robust`: arquivo com `;` como delimitador, arquivo com BOM UTF-8, arquivo comum com `,` —
  todos devem produzir as mesmas colunas/valores esperados; nomes de coluna com espaço (`" Data "`) saem
  stripados.
- `resolve_column`: match exato, match tolerante a espaço/case, coluna ausente levanta `ValueError` com o
  nome configurado e a lista de colunas disponíveis na mensagem.
- Normalização de canal: `daily_investment_df` sintético com `"Google Ads"` e `"GOOGLE ADS"` em linhas
  diferentes deve colapsar para uma única categoria depois do processamento.
- Guarda de dataframe vazio: CSV sintético onde todas as datas são inválidas deve levantar `ValueError`
  claro em vez de seguir com dataframe vazio.
