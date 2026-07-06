# Correção de inconsistências no dashboard de Elasticidade

## Contexto

Investigação a pedido do usuário sobre valores zerados/inconsistentes nos gráficos e tabelas da aba
Elasticidade (`scripts/streamlit_app.py`, `scripts/dashboard_charts.py`, `scripts/elasticity_analysis.py`).
Todos os sintomas reportados (linhas achatadas em 0%, mix "evoluindo" que nunca muda, "Cenário de
Saturação" com investimento abaixo da média histórica e iCPA R$0,00) foram confirmados com dados reais
de `outputs/user_1/Meu_Projeto_Dynamic_2_dynamic/global_saturation_analysis/`.

Achados, em ordem de causa → efeito:

1. **Colapso do Ridge**: `run_mmm_engine` (Stage 2) otimiza 39 parâmetros (alpha/k/s × 13 canais) num
   único `scipy.minimize()` a partir de 1 ponto inicial heurístico. Nesse dataset, o resultado atribui
   100% do "lift" de marketing ao Skyscanner e 0% aos outros 11 canais — confirmado tanto pelo
   `individual_response_curves_data.csv` (toda linha de canal exceto Skyscanner tem `Projected_Total_KPIs`
   idêntico bit a bit em qualquer nível de spend) quanto pelo texto do `global_narrative.json` gerado
   pelo Gemini ("zera os demais canais e aloca 100% no Skyscanner"). Um diagnóstico de VIF sobre o
   investimento bruto por canal descarta colinearidade simples como causa principal (BING, GOOGLE, KAIAK,
   RTB HOUSE têm VIF entre 1.4 e 2.1 — sem problema) — a causa mais provável é a otimização conjunta,
   não-convexa, ficar presa num mínimo local ruim a partir de um único ponto de partida.
2. **Extrapolação do `MinMaxScaler`**: as funções de simulação (`simulate_kpi` e o loop de
   `generate_individual_response_curves`) chamam `mkt_scaler.transform()` com spend simulado fora do
   range observado historicamente (investimento varre de 0 até 1.5-3x a média). `MinMaxScaler` não
   recorta o resultado — em investimento baixo, a feature transformada fica negativa, e multiplicada
   pelo coeficiente positivo do canal dominante gera KPI previsto **negativo**. Confirmado em
   `response_curve_data.csv`: no investimento diário R$0, `Projected_Total_KPIs = -141394.67`.
3. **Heurística de "Cenário de Saturação"** (`streamlit_app.py`): usa a derivada do **primeiro passo**
   da curva como referência de 100% e procura o primeiro ponto abaixo de 10% disso. Duas falhas
   compostas: (a) o artefato do item 2 cria um salto artificial enorme nesse primeiro passo; (b) mesmo
   sem esse artefato, o design é frágil — qualquer curva côncava desde a origem (o normal em curvas de
   resposta) já desacelera a partir do primeiro passo, então o critério dispara cedo demais por
   definição. Resultado: "Cenário de Saturação" trava em ~R$93,3k/mês (2-3% do caminho da curva), bem
   abaixo da média histórica de R$3,1M — o oposto do "teto da operação" que a métrica deveria
   representar.
4. **iCPA forçado a R$0,00**: como o item 3 produz um cenário com investimento/KPI incremental
   negativos, a regra `iCPA = 0 quando KPI_incremental <= 0` (em vez de deixar o valor real ou N/A)
   mostra "R$ 0,00" — parecendo ganho gratuito quando na verdade o dado de entrada já está quebrado.
5. **Legenda do "Evolução do Mix de Canais"**: o texto de apoio diz que a alocação "muda à medida que o
   orçamento total escala". Mas `Spend_{canal}_Strategic = Investimento_Total × mix_fixo` — uma fração
   constante multiplicada por um total variável sempre dá a mesma porcentagem. O gráfico não pode,
   matematicamente, mostrar o que o texto promete.

Investiguei também um suspeito de nome de canal duplicado (`"GOOGLE"` vs `"GOOGLE "` com espaço) — não é
bug real, `data_preprocessor.py:158-161` já normaliza com `.str.strip()`. Descartado.

Não encontrei os mesmos padrões em `saturation_curve.py` (módulo do Stage 1 / análise causal de eventos
por `analysis.py`) — é um código isolado, sem sinais de estar afetado.

## Escopo

### Fora de escopo (explícito)
- Reescrever a arquitetura do motor de MMM (adstock + Hill + Ridge two-stage).
- Mix de canais realmente dinâmico por nível de orçamento (reotimizar a alocação em cada ponto da
  curva) — seria uma feature nova, não uma correção de bug. Aqui só corrigimos o texto para não prometer
  isso.
- Qualquer mudança em `analysis.py`, `saturation_curve.py`, `local_main.py`,
  `local_main-without-gemini.py` — sem evidência de problema.
- Normalização de nome de canal — já existe.

### Fixes

#### 1. Multi-start na otimização Stage 2 (`elasticity_analysis.py::run_mmm_engine`)
Trocar a chamada única de `minimize(elasticity_objective_function, initial_params, ...)` por até 6
tentativas: a primeira usa exatamente o `initial_params` heurístico atual (comportamento hoje
preservado como um dos candidatos); as outras 5 usam `numpy.random.default_rng(seed=42).uniform(low, high)`
por parâmetro, respeitando os `bounds` já existentes. Mantém o resultado (`alphas`, `ks`, `ss`) com
menor `result.fun`. Bounds e a própria `elasticity_objective_function` não mudam — é só uma busca mais
ampla pelo mesmo problema, reduzindo a chance de travar num mínimo local degenerado.

Sem novo parâmetro de config: o número de tentativas fica hardcoded (constante local na função), não é
algo que precise ser ajustável por advertiser agora.

#### 2. Clip da extrapolação do scaler (`elasticity_analysis.py`)
Em `simulate_kpi` (usado por `generate_aggregated_response_curve`) e no loop de
`generate_individual_response_curves`, após `mkt_scaler.transform(...)`, aplicar
`np.clip(scaled, 0.0, 1.0)` antes de passar pro `mkt_model.predict(...)`. Isso limita a simulação ao
range fisicamente sensato (0% a 100% da saturação histórica de cada canal) mesmo quando o investimento
simulado extrapola além do que foi observado — eliminando o KPI previsto negativo.

#### 3. Heurística de saturação (`streamlit_app.py`, bloco atual ~linhas 1423-1437)
Trocar a referência de "derivada do primeiro passo" por "derivada de **pico**" da curva completa
(`first_derivative.max()`), e escanear o cruzamento do limiar de 10% **a partir do índice do pico em
diante** (não do início). Isso corrige as duas falhas do item 3 do diagnóstico: o pico deixa de ser um
artefato de extrapolação (já corrigido pelo fix 2) e a busca não pode mais disparar antes da curva
sequer atingir seu ponto mais "produtivo".

#### 4. iCPA da tabela de cenários (`streamlit_app.py`, bloco atual ~linhas 1478-1487)
Trocar o `0` do `np.where(kpi_incrementais > 0, ratio, 0)` por `np.nan`. `format_currency` já trata
`NaN` como `"N/A"` (linha ~1490) — não precisa de mudança na função de formatação, só no valor produzido.
A linha do "Cenário Atual" (índice 0, que já é zerada manualmente logo depois) continua sendo zerada
explicitamente do mesmo jeito (esse zero é correto: é a própria baseline, incremental de si mesma).

#### 5. Texto do "Evolução do Mix de Canais" (onde o `st.markdown` de apoio desse gráfico está em
`streamlit_app.py`)
Reescrever a frase de apoio para descrever o que o gráfico realmente mostra: a alocação recomendada
(mix "Modelo de Elasticidade") é uma proporção fixa; o gráfico mostra quanto cada canal recebe em R$
conforme o orçamento total escala, não uma mudança de proporção entre canais.

## Testes

- `tests/test_elasticity_analysis.py`: um teste novo para o multi-start (fix 1) — construir um cenário
  sintético pequeno onde o ponto de partida heurístico único é conhecidamente ruim (mínimo local raso) e
  verificar que o multi-start encontra um `result.fun` menor ou igual; um teste para o clip (fix 2) —
  chamar `simulate_kpi`/o loop individual com spend fora do range de fit e checar que o KPI previsto
  nunca fica abaixo do organic baseline (não pode ser "pior que não gastar nada").
- `tests/test_dashboard_charts.py` ou novo teste em `streamlit_app`-adjacent: teste unitário da lógica de
  detecção de saturação (fix 3) isolada em função pura, com uma curva sintética côncava-desde-a-origem
  (caso que quebrava antes) confirmando que o ponto escolhido fica na região de alto investimento, não
  nos primeiros 5% da curva.
- Teste da tabela de cenários (fix 4): KPI incremental negativo → iCPA deve ser `NaN`/"N/A", não `0`.

Sem framework novo, sem fixtures elaboradas — segue o padrão dos testes existentes nesses mesmos
arquivos.
