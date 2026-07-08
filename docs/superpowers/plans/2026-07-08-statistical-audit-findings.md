# Auditoria estatística — incoerências no treinamento dos modelos

**Data:** 2026-07-08
**Escopo:** `scripts/elasticity_analysis.py` (MMM/elasticidade), `scripts/analysis.py`
(causal + projeção), `scripts/data_preprocessor.py`, config de setup.
**Método:** leitura linha-a-linha + treino real dos modelos sobre o projeto
`Meu_Projeto_Dynamic_dynamic` (config real), com experimento de controle.

> **Conclusão de uma linha:** os números do dashboard não são confiáveis hoje —
> a maior parte do "lift de marketing" (~15%) é um **artefato de recorte
> (`clip(lower=0)`) sobre ruído**, não um efeito medido; o modelo de resposta
> praticamente não ajusta (R²≈0,10) e seus parâmetros ficam presos nos limites.

---

## Evidência empírica (treino real)

Rodei os modelos de verdade na config atual:

| Métrica | Valor | Leitura |
|---|---|---|
| R² do modelo de marketing (Stage 2, in-sample, no *lift*) | **0,098** | quase não explica nada |
| Média do baseline orgânico vs média do KPI | **9.999.778 == 9.999.778** | baseline "come" 100% da média |
| Resíduo (KPI − baseline) médio | **−0,00** | regressão passa pelo meio |
| Dias com KPI < baseline (recortados p/ 0) | **54%** | metade dos dias vira "0 de lift" |
| Adstock α ajustado | BING=0,000 · PMAX=0,000 · GOOGLE=0,900 | **presos nos limites** |
| Hill `s` (FACEBOOK) | **1.585.070** | inflexão muito além do gasto real → curva ~linear |
| **Controle: KPI trocado por ruído branco puro** | **"lift" de 16,3%** | efeito ZERO real ainda "acha" 16% de marketing |
| Projeção/causal `analysis.py` — Hill `k` | **5,46** (limite=10) | retornos **acelerando** (convexo) |

O experimento de controle é o achado central: com KPI = ruído branco (sem nenhum
efeito de marketing por construção), o pipeline ainda atribui **16,3%** ao
marketing. Logo os ~15% no dado real são majoritariamente artefato de método.

---

## Registro de incoerências

### 🔴 CRÍTICAS — invalidam os números apresentados

**F1 — `clip(lower=0)` fabrica lift a partir de ruído** (confirmado por controle)
*Onde:* `elasticity_analysis.py:315` (`y_lift = (y_total - baseline).clip(lower=0)`)
e `analysis.py:570` (`incremental_kpi = (y_base - baseline).clip(lower=0)`).
*Problema:* o baseline (com intercepto) passa pela média, então o resíduo tem
média ≈ 0. Recortar os negativos em 0 mantém só a metade positiva do ruído e a
chama de "lift de marketing". Isso garante atribuição positiva mesmo sem efeito
nenhum. É a raiz dos ROAS/iROAS absurdos que você vinha vendo.
*Correção:* não recortar o lift. Modelar a resposta de marketing **conjuntamente**
com o baseline (um único modelo com features de calendário **e** de mídia), ou no
mínimo permitir resíduo assinado e validar out-of-sample. Curto prazo: remover o
`.clip` e reportar contribuição líquida (pode ser ~0 — e é essa a verdade aqui).

**F2 — parâmetros de transformação super-ajustados a MSE in-sample, presos nos limites**
*Onde:* `elasticity_objective_function` (`:151`) reajusta Ridge a cada iteração e
minimiza MSE **in-sample**; `run_mmm_engine` otimiza α/k/s (3×n params) sem holdout.
*Problema:* α caiu em 0,000/0,900 (cantos), Hill `s` estourou p/ 1,6M (curva vira
linear), R²=0,098. Os parâmetros não estão identificados; a mistura ótima e o ROAS
saem de um modelo que não ajusta.
*Correção:* validar α/k/s por CV temporal (walk-forward), não por MSE in-sample;
regularizar/limitar `s` à faixa de gasto observada; rejeitar canais não
identificáveis em vez de fixá-los no limite.

**F3 — `analysis.py` ainda ajusta Hill `k` até 10 (retornos convexos)**
*Onde:* `_train_response_model` (`:606`, `bounds=([0.1,0],[10, ...])`, `p0=[2,...]`).
*Problema:* o teto `k≤1,0` que corrigimos em `elasticity_analysis.py` **não** foi
aplicado aqui. No treino real deu `k=5,46` → curva de resposta com ROI marginal
**crescente**. Isso alimenta `run_opportunity_projection` e o modelo causal
(`run_causal_impact_analysis` reusa `k`), reintroduzindo o artefato de 490x.
*Correção:* limitar `k` a `(0.1, 1.0)` e `p0` k=1,0, igual ao MMM. Idealmente
extrair `hill`/adstock para um módulo único compartilhado.

**F4 — significância causal via teste-t em resíduos autocorrelacionados**
*Onde:* `analysis.py:389` (`stats.ttest_1samp(impact_df["impact"], 0)`).
*Problema:* os "impactos" diários são resíduos de um forecast (série
autocorrelacionada, erro que cresce no horizonte). O teste-t assume amostras
i.i.d. → N efetivo inflado → p-valor subestimado → **falsa significância**. Esse
p-valor é justamente o portão `require_statistical_significance`.
*Correção:* usar o intervalo preditivo posterior do próprio `UnobservedComponents`
(`get_forecast().conf_int()`) para julgar o efeito, em vez de um teste-t sobre o
ponto. Alternativa: `CausalImpact` propriamente, ou bloco-bootstrap dos resíduos.

### 🟠 ALTAS

**F5 — fallback de ruído fabrica ≥5% de atribuição** (latente; não disparou aqui)
*Onde:* `elasticity_analysis.py:388-437`. Se a contribuição medida < 5% do KPI, o
código **inventa** 5%, distribui por share de gasto e abaixa o baseline. Não
disparou neste dataset (deu 14,93%), mas em dado mais quieto vai disparar e não há
aviso ao usuário de que o número é sintético.
*Correção:* remover a fabricação; se o efeito for < piso, reportar "não
detectável" em vez de inventar. No mínimo, marcar claramente como estimado.

**F6 — `positive=True` no baseline orgânico silencia features que derrubam o KPI**
*Onde:* `elasticity_analysis.py:310`. No treino, **5 de 15** coeficientes ficaram
presos em 0. Fim de semana/feriado que reduzem venda não podem ser modelados, e
esse efeito vaza para o "lift".
*Correção:* remover `positive=True` do Stage 1 (o baseline tem intercepto e pode
ter efeitos negativos legítimos). Manter positividade só no Stage 2 (mídia).

**F7 — Ridge `alpha=1.0` fixo, sem CV, escala inconsistente**
*Onde:* todos os `Ridge(alpha=1.0)`; `KFold` é importado e **nunca usado**.
*Problema:* features em [0,1] (MinMax) mas `y` na casa dos milhões → o termo de
regularização é desprezível/arbitrário e não é invariante à escala. Magnitude dos
coeficientes (→ contribuição → ROAS) fica ao acaso.
*Correção:* padronizar `y` (ou usar `RidgeCV`/`LassoCV` com CV temporal) e escolher
`alpha` por validação, não por chute. Remover o `KFold` morto ou usá-lo.

**F8 — modelo causal aplica um α/k/s global a todos os canais**
*Onde:* `analysis.py:285-292` aplica `best_alpha/k/s` (ajustados na **soma** dos
canais do evento) a **cada** canal individualmente.
*Correção:* ajustar transformação por canal, ou deixar explícito que o causal é
agregado por evento (e não por canal).

### 🟡 MÉDIAS

- **F9 — incerteza do forecast descartada / R² in-sample fraco:** `analysis.py`
  usa só `predicted_mean` e ignora `conf_int`; o R² é in-sample de um local-level
  (quase sempre alto) → o portão `require_model_fit` (threshold 0,3) quase não
  filtra. Usar erro out-of-sample / horizonte.
- **F10 — três convenções de adstock diferentes:** `data_preprocessor` usa
  `ewm().mean()` (normalizado), `analysis` usa `lfilter` (soma 1/(1−α)),
  `elasticity_analysis` usa convolução `αⁱ`. O α escolhido por correlação numa
  convenção é aplicado noutra. Unificar num módulo só.
- **F11 — contradição na config:** `optimization_target=CONVERSIONS` com
  `kpi_is_monetary=true`. O código força `revenue_mode` (mídia monetária) e ignora
  o alvo CONVERSIONS. `optimization_target` está morto quando o KPI é monetário —
  ou respeitar, ou avisar no setup.

### 🟢 BAIXAS

- **F12** — `strategic_mix` calculado e nunca usado (`elasticity_analysis.py:581-587`).
- **F13** — armadilha de dummies no OLS baseline de `analysis.py` (`day_0..6` +
  constante + `is_weekend` colinear).
- **F14** — `financial_targets` desativados (999999 / 0) → "Limite Estratégico" é
  só 1,5× o gasto, não um limite financeiro real.

---

## Plano de correção (faseado, sem implementar ainda)

**Fase 0 — parar de enganar (rápido, alto impacto)**
1. F3: limitar `k≤1.0` e `p0=1.0` em `analysis.py._train_response_model` (paridade
   com o MMM). *1 linha de bounds + 1 de p0.* Testável com o probe.
2. F6: remover `positive=True` do baseline Stage 1 (mantém no Stage 2).
3. F5: remover a fabricação de 5% (ou trocar por "efeito não detectável").

**Fase 1 — corrigir o viés estrutural do lift (o achado central F1)**
4. Substituir o pipeline de duas etapas com `clip(lower=0)` por um **modelo único**
   (calendário + mídia juntos), ou remover o clip e reportar contribuição líquida.
   Requer re-teste das curvas e do dashboard. É a mudança que muda os números.

**Fase 2 — identificação honesta dos parâmetros (F2, F7)**
5. CV temporal (walk-forward) para α/k/s e para o `alpha` do Ridge; padronizar `y`;
   limitar `s` à faixa observada; sinalizar canais não identificáveis.

**Fase 3 — inferência causal honesta (F4, F8, F9)**
6. Trocar o teste-t pelo intervalo preditivo do state-space; usar erro
   out-of-sample no portão de fit; transformação por canal (ou rótulo agregado).

**Fase 4 — limpeza (F10-F14)**
7. Unificar adstock/hill num módulo; resolver a contradição CONVERSIONS×monetário
   no setup; remover código morto; corrigir dummies; deixar claro o limite 1,5×.

**Ordem recomendada:** Fase 0 primeiro (baixo risco, corrige o pior artefato
convexo e o silenciamento do baseline), depois decidir com você se F1 (Fase 1) é
refactor agora ou reporta-se honestamente como "efeito não detectável" enquanto
não há um modelo conjunto.
