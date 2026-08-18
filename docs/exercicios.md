# Exercícios do Lab 2

Trabalhe em duplas. Cada exercício tem uma pergunta a responder, não só um
código a rodar. Anote as respostas: elas são o entregável.

Antes de começar:

```bash
uv sync --all-groups
uv run lab2 all
```

---

## 1. Trocar de modelo sem tocar no código (aquecimento, 10 min)

O `conf/config.yaml` decide qual algoritmo é usado. Rode os quatro:

```bash
uv run lab2 evaluate --algoritmo ridge --sem-graficos
uv run lab2 evaluate --algoritmo arvore --sem-graficos
uv run lab2 evaluate --algoritmo floresta --sem-graficos
uv run lab2 evaluate --algoritmo boosting --sem-graficos
```

Monte uma tabela com MAE e R² de teste de cada um.

**Perguntas**

1. A diferença entre o pior e o melhor modelo é maior ou menor do que a
   diferença entre o baseline e o pior modelo? O que isso sugere sobre onde
   está o gargalo: no algoritmo ou nos dados?
2. O `ridge` é o modelo mais simples e o mais rápido. Quanto ele perde? Vale a
   pena a complexidade dos outros?
3. Abra o `conf/config.yaml` e mude o grid do `boosting`. Rodar mais
   combinações melhora o teste ou só o treino?

---

## 2. Quebrar a validação de propósito (o principal, 20 min)

Hoje o projeto divide treino e teste **por linha de ônibus**. Vamos ver o que
acontece com uma divisão aleatória comum.

Em `src/lab2_sptrans/model.py`, troque:

```python
divisor = GroupShuffleSplit(n_splits=1, test_size=..., random_state=...)
idx_treino, idx_teste = next(divisor.split(X, y, groups=grupos))
```

por uma divisão que ignore o grupo:

```python
from sklearn.model_selection import ShuffleSplit
divisor = ShuffleSplit(n_splits=1, test_size=..., random_state=...)
idx_treino, idx_teste = next(divisor.split(X, y))
```

e, em `ajustar`, troque `GroupKFold` por `KFold(n_splits=..., shuffle=True,
random_state=cfg.seed)`.

Rode `uv run lab2 evaluate` e compare com o resultado original.

**Perguntas**

1. Quanto o R² de teste subiu? E o MAE, quanto caiu?
2. Esse ganho é real? Escreva em uma frase o que exatamente o modelo passou a
   fazer que antes não fazia.
3. Imagine que a SPTrans vai usar o modelo para estimar a frequência de uma
   **linha nova**, que nunca operou. Qual das duas avaliações descreve melhor o
   que ele vai entregar?
4. Volte o código como estava. No projeto da integradora do seu grupo, existe
   algum agrupamento parecido? Cliente, município, empresa, processo, período?
   Escreva qual é.

::: nota
Este exercício é a razão de ser do lab. Um modelo com validação errada não é um
modelo pior: é um modelo cujo número não significa nada.
:::

---

## 3. Adicionar uma variável (25 min)

O GTFS tem informação que o pipeline ainda não usa. Escolha **uma**:

- **Densidade de paradas nas pontas**: quantas paradas existem num raio de
  500 m da parada de origem. Proxy de região adensada.
- **Hora como variável cíclica**: `sin(2*pi*hora/24)` e `cos(2*pi*hora/24)`. A
  hora 23 e a hora 0 são vizinhas, e o modelo não sabe disso.
- **Sobreposição com outras linhas**: quantas linhas distintas compartilham pelo
  menos 5 paradas com esta. Proxy de concorrência interna da rede.
- **Faixa de extensão**: `extensao_km` cortada em categorias (curta, média,
  longa), como categórica, em vez de numérica.

Passos:

1. Crie a coluna em `src/lab2_sptrans/transform.py`, numa função nova, com
   docstring explicando o que ela mede.
2. Registre o nome em `conf/config.yaml`, na lista certa (`numericas` ou
   `categoricas`).
3. Escreva **um teste** em `tests/test_transform.py` para a função nova.
4. Rode `uv run pytest`, depois `uv run lab2 transform` e `uv run lab2 evaluate`.

**Perguntas**

1. O MAE de teste melhorou? Quanto?
2. A variável nova aparece no topo da importância por permutação
   (`reports/importancia.csv`)?
3. Você precisou mexer em `features.py` ou em `model.py`? Por que não?

---

## 4. Aberto: do resíduo para a decisão (20 min)

O relatório (`notebooks/relatorio.qmd`) lista as linhas cuja oferta é mais
espaçada do que o perfil sugere. Transforme isso em algo que alguém usaria.

Escolha um caminho:

- **Visualização**: um mapa ou gráfico dos maiores resíduos por região da
  cidade, com `lat_origem` e `lon_origem`. Qual zona concentra o problema?
- **Recorte**: os resíduos mudam por período do dia? Existe um horário em que a
  rede é sistematicamente pior do que o modelo espera?
- **Investigação**: pegue as três linhas com maior resíduo, procure o
  itinerário no site da SPTrans e escreva se o modelo achou um problema real ou
  uma peculiaridade de cadastro.

**Pergunta única**: escreva o parágrafo que você mandaria para um gestor da
SPTrans. Uma decisão, um número, uma ressalva.

---

## 5. Desafio (para quem terminar antes)

**Trocar a tarefa.** Em vez de prever o intervalo em segundos, preveja se a
linha é de **alta frequência** (intervalo até 15 minutos) ou não.

O que muda:

- o alvo vira binário (`headway_seg <= 900`);
- `HistGradientBoostingRegressor` vira `HistGradientBoostingClassifier`;
- `TransformedTargetRegressor` sai (não faz sentido em classificação);
- `scoring` vira `"roc_auc"` ou `"balanced_accuracy"`;
- as métricas viram matriz de confusão, precisão e revocação.

**O que não muda**: o `ColumnTransformer`, o `Pipeline`, o `GroupKFold` e a
estrutura do projeto inteiro. Esse é o ponto.

**Pergunta**: o `Pipeline` que você já tinha aceitou a troca com quantas linhas
de mudança? Se foram muitas, alguma coisa estava no lugar errado.

---

## Entrega (opcional, com feedback)

Um fork do repositório com:

- pelo menos o exercício 2 e um dos outros resolvidos;
- as respostas das perguntas em `docs/respostas.md`;
- os testes passando (`uv run pytest`) e o CI verde.

Prazo: uma semana. Manda o link no Teams.
