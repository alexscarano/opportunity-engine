# Guia de teste — o que validar depois das correções estatísticas

**Data:** 2026-07-08 · **Branch:** `feat/robust-csv-ingestion`

Este guia diz, para cada correção: **o que testar**, **como saber se está certo**
(o resultado esperado) e **o sinal de alerta** (o que indicaria que algo voltou a
quebrar). Feito para você validar rodando o app / relatórios, não o código.

---

## 0. Teste rápido de fumaça (rode primeiro)

```bash
# suite de testes automatizada
./.venv/Scripts/python.exe -m pytest tests/ -q      # esperado: 115 passed

# treino real de ponta a ponta (usa a config do seu projeto)
./.venv/Scripts/python.exe local_main-without-gemini.py   # ou o run normal do app
```

✅ **Certo:** `115 passed`; o treino termina sem traceback.
🚩 **Alerta:** qualquer `FAILED`, ou `UnicodeEncodeError` no console (era o bug do
emoji, já corrigido — se voltar, algum print novo tem emoji).

---

## 1. Números do dashboard de Elasticidade agora são honestos (F1)

**O que testar:** abra o dashboard de Elasticidade no seu projeto e olhe a
contribuição de marketing e os ROAS/iROAS.

✅ **Certo:** os valores caíram muito em relação a antes (no seu projeto real a
contribuição de marketing caiu de ~15% para **~2-3%**). ROAS/iROAS absurdos
(centenas de x) sumiram. Se o log/tela mostra **"AVISO (sinal fraco)"**, isso é
**esperado e correto** para este projeto: a mídia desses canais explica quase nada
de "Vendas via lead" (o KPI conta vendas de todos os canais, não só os pagos).

🚩 **Alerta:** contribuição de marketing voltar a ~15% ou mais sem justificativa;
"lift" alto com R² perto de zero.

**Como provar o conceito (opcional):** o teste de controle mostra que, com KPI =
ruído puro, o pipeline *não* deve mais inventar 15% de lift. Antes inventava.

---

## 2. Confiança do modelo: R² validado (F2/F7)

**O que testar:** no log do treino da Elasticidade, procure as duas linhas:
```
- Marketing Model R² (in-sample on lift): 0.00xx
- Marketing Model R² (walk-forward CV):   0.00xx
```

✅ **Certo:** as duas aparecem. O **CV R² é a que importa** (fora da amostra). No
seu projeto ele dá ~0 → o aviso de baixa confiança aparece, o que é honesto. Em um
projeto com sinal real de mídia, o CV R² deve ser claramente positivo (ex. > 0,2)
e o mix recomendado deve ser estável entre execuções.

🚩 **Alerta:** CV R² negativo **sem** o aviso de baixa confiança; ou o mix
recomendado mudar radicalmente de uma execução para outra (sinal de que os canais
não estão identificados — comum quando o CV R² ≈ 0, e é por isso que o aviso existe).

**Custo esperado:** o treino da Elasticidade ficou mais lento (~30s no seu
projeto, antes ~5s) porque a seleção de α/k/s agora valida fora da amostra. É
esperado. Se passar de ~2 min, me avise.

---

## 3. Curvas de resposta sem "retorno acelerando" (F3)

**O que testar:** nas "Curvas de Resposta Individuais por Canal" e na curva
agregada, olhe o formato.

✅ **Certo:** toda curva é **côncava** — cresce e vai achatando (retornos
decrescentes). Nunca acelera (fica mais íngreme) à medida que o investimento sobe.
Vale tanto para a aba de Elasticidade quanto para a projeção/oportunidade (era lá
que o `k` ainda subia até 10 e dava curva convexa).

🚩 **Alerta:** qualquer curva com trecho que fica mais íngreme para a direita, ou
iROAS que **aumenta** com o orçamento.

---

## 4. Significância causal honesta (F4/F9)

**O que testar:** rode uma análise causal de um evento e olhe o p-valor e (novo) o
intervalo de confiança do impacto.

✅ **Certo:**
- O p-valor agora vem da incerteza do próprio modelo (não de um teste-t). Ele é
  **mais conservador** que antes (menos eventos "significativos" por acaso).
- O relatório traz `absolute_lift_ci_lower` / `absolute_lift_ci_upper` (IC 95%).
- **Coerência:** se o IC 95% **contém 0**, o evento **não** deve ser significativo
  (p ≥ 0,05 no threshold padrão). Se o IC não contém 0, deve ser significativo.
  Os dois têm que concordar.

🚩 **Alerta:** IC contém 0 mas o evento aparece como significativo (ou vice-versa);
p-valor minúsculo (ex. 0,001) num impacto visivelmente dentro do ruído.

> Nota: o `require_statistical_significance` no seu config usa `p_value_threshold`
> = 0,1 (90%). Se quiser mais rigor, suba para 0,05.

---

## 5. Aviso de contradição de config (F11)

**O que testar:** seu config tem `optimization_target=CONVERSIONS` **e**
`kpi_is_monetary=true`.

✅ **Certo:** no log aparece **"AVISO (config): optimization_target='CONVERSIONS'
foi ignorado porque kpi_is_monetary=true..."**. Isso confirma que o alvo
CONVERSIONS não tem efeito quando o KPI é monetário (o app usa modo receita/ROAS).

**Ação sugerida:** decida qual é a verdade do seu KPI "Vendas via lead":
- Se é **valor em R$** → deixe `kpi_is_monetary=true` e mude o alvo para RECEITA
  (só para tirar a contradição; o resultado não muda).
- Se é **contagem** (nº de vendas/leads) → **desmarque** "KPI já está em R$" e
  preencha Ticket Médio corretamente. Isso muda os números.

🚩 **Alerta:** o aviso não aparecer apesar da combinação estar no config.

---

## 6. Baseline e limpezas (F6/F12/F13)

**F6 (baseline sem `positive=True`):** o baseline orgânico agora pode ter efeitos
negativos de calendário. ✅ Efeito: menos "lift" vazado (contribuição de marketing
um pouco menor). Não há tela específica; entra no item 1.

**F12/F13:** limpeza interna (código morto e colinearidade). ✅ Não devem mudar
nenhum número visível — se algum resultado mudar por causa deles, é bug.

---

## Resumo: os 3 checks que mais importam

1. **Contribuição de marketing / ROAS despencaram e o aviso de sinal fraco aparece**
   → o artefato de ruído acabou (F1).
2. **Log mostra CV R² e ele bate com a confiança do resultado** → identificação
   honesta (F2).
3. **Nenhuma curva acelera; causal com IC coerente com o p-valor** → saturação e
   inferência corretas (F3/F4).

Se esses três baterem, as correções estatísticas estão funcionando. Diferido de
propósito (não é regressão): F8, F10 e o "adstock duplo" — ver o doc de achados.
