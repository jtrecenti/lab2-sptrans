# Lab 2: pipeline de dados e modelo com a base da SPTrans

Laboratório 2 da disciplina **Prática Avançada em Data Science e Visualização**
(PADS, Insper). O Lab 1 mostrou como sair do notebook solto e chegar a um
pacote Python versionado. Este lab pega essa estrutura e coloca dentro dela um
**pipeline completo**: extrair dado real, transformar, treinar um modelo e
avaliar, com um comando só e o mesmo resultado em qualquer máquina.

O modelo aqui é **meio, não fim**. Ele existe para produzir uma leitura sobre a
oferta de ônibus em São Paulo; o que se aprende é a **forma de trabalhar** que
faz esse modelo sobreviver fora do notebook.

```bash
git clone https://github.com/jtrecenti/lab2-sptrans
cd lab2-sptrans
uv sync --all-groups
cp .env.example .env      # opcional: token da API Olho Vivo
uv run lab2 all
```

Isso baixa os dados, monta a tabela analítica, ajusta o modelo e escreve as
métricas em `reports/`.

---

## A pergunta

> **Quanta oferta de ônibus a SPTrans coloca em cada linha, em cada hora do
> dia, e o que explica essa decisão?**

A resposta é o **headway**: o intervalo programado entre dois ônibus da mesma
linha. Headway de 300 segundos significa um ônibus a cada 5 minutos. É a
variável que traduz "nível de serviço" em número.

O modelo aprende esse padrão a partir de características estruturais da linha
(extensão, número de paradas, região atendida, uso de corredor, horário). Serve
para duas coisas concretas:

1. **Achar o fora do padrão.** Linhas com oferta muito abaixo do que o modelo
   previa, dado o perfil, são candidatas a revisão de serviço. Esse é o
   resíduo, não a predição.
2. **Estimar oferta para linha nova.** Ao desenhar um itinerário novo, o modelo
   dá uma referência de frequência compatível com o resto da rede.

Quem usaria: planejamento de transporte público, consultoria de mobilidade,
mercado imobiliário (acessibilidade por endereço), operadora de frota.

## Os dados

| Fonte | O que traz | Acesso |
| --- | --- | --- |
| **GTFS SPTrans** | rede planejada: linhas, paradas, traçados, horários e frequências | ZIP público (espelho oficial MobilityData) |
| **API Olho Vivo** | camada em tempo real: corredores, paradas de corredor, posição da frota | token gratuito no [cadastro de desenvolvedor](https://www.sptrans.com.br/desenvolvedores/) |

As duas fontes se cruzam **por proximidade geográfica**, não por chave: o GTFS e
a API usam códigos de parada diferentes, então uma parada do GTFS a menos de
120 m de uma parada de corredor da API é tratada como a mesma parada. É o tipo
de junção suja que aparece em todo projeto real e que nenhum dataset de curso
tem.

Sem o token o projeto **continua rodando**: as colunas de corredor ficam
faltantes e o pré-processamento lida com isso. Isso é proposital.

**Unidade de análise**: linha x sentido x faixa horária. Cerca de 37 mil
registros, 1.142 linhas de ônibus.

## Como o repositório está organizado

```
lab2-sptrans/
├── conf/config.yaml          # toda escolha de projeto mora aqui, não no código
├── src/lab2_sptrans/
│   ├── config.py             # carrega o YAML e o .env; resolve caminhos
│   ├── extract.py            # E: baixa GTFS e fala com a API Olho Vivo
│   ├── transform.py          # T: tabelas cruas -> tabela analítica
│   ├── features.py           # a "receita": ColumnTransformer
│   ├── model.py              # o Pipeline: receita + estimador
│   ├── evaluate.py           # métricas, importância, gráficos
│   └── cli.py                # cada etapa vira um comando
├── tests/                    # testes que rodam sem rede e sem token
├── notebooks/relatorio.qmd   # o relatório reprodutível (Quarto)
├── .github/workflows/        # ci.yml (testes) e coleta.yml (coleta agendada)
├── data/  models/  reports/  # gerados, fora do Git
└── Makefile
```

Quatro regras que valem para o projeto de vocês também:

**1. Dado gerado não entra no Git.** `data/`, `models/` e `reports/` estão no
`.gitignore`. O repositório guarda o código que **recria** os dados. Quem clona
roda `uv run lab2 all` e chega no mesmo lugar.

**2. Segredo não entra no Git.** O token mora no `.env`, que está ignorado. O
`.env.example` documenta qual variável existe, sem o valor. No CI, o mesmo
token entra como *secret* do repositório.

**3. Parâmetro não fica escondido no código.** Semente, fração de teste,
algoritmo, grid de hiperparâmetros e lista de variáveis estão no
`conf/config.yaml`. Trocar de modelo é editar uma linha de YAML, não caçar um
número no meio de uma função.

**4. Cada etapa é um comando que lê e escreve arquivo.**

```bash
uv run lab2 extract     # -> data/raw/
uv run lab2 transform   # -> data/processed/viagens.csv.gz
uv run lab2 train       # -> models/modelo.joblib
uv run lab2 evaluate    # -> reports/
uv run lab2 predict --entrada novas_linhas.csv
```

É isso que permite repetir só o que mudou, chamar uma etapa de dentro de um
agendador e transformar o mesmo código numa API depois.

## O ponto central: `Pipeline`

O erro mais comum em projeto de dados é padronizar, imputar e criar dummies
**antes** de separar treino e teste. Quando isso acontece, a média usada para
imputar já viu o teste, o modelo parece melhor do que é, e a diferença só
aparece em produção.

O `Pipeline` do scikit-learn resolve isso ao tornar o pré-processamento parte
do modelo:

```python
Pipeline([
    ("preprocessador", ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("padronizar", StandardScaler())]), numericas),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="constant",
                                                    fill_value="desconhecido")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist"))]),
         categoricas),
    ])),
    ("modelo", HistGradientBoostingRegressor()),
])
```

Três consequências práticas:

- em cada dobra da validação cruzada, o pré-processamento é reajustado só no
  treino daquela dobra;
- o `.joblib` salvo contém a receita **e** o modelo, então quem for servir a
  API recebe o `DataFrame` cru e chama `.predict`;
- o `GridSearchCV` pode ajustar hiperparâmetro do modelo e do pré-processamento
  ao mesmo tempo.

Quem vem do R: isso é `recipe()` + `workflow()` do tidymodels. O material da
aula traz o de-para termo a termo e um par de notebooks que ajusta o mesmo
lasso e a mesma floresta nas duas linguagens.

## Validação por grupo

O detalhe que muda o resultado: a divisão treino/teste é feita **por linha de
ônibus**, com `GroupShuffleSplit` e `GroupKFold`.

Cada linha aparece na tabela com cerca de 30 faixas horárias. Numa divisão
aleatória comum, a mesma linha cai nos dois lados e o modelo decora a linha em
vez de aprender o padrão de oferta. O R² sobe, o modelo não serve para linha
nova, e o problema só aparece quando alguém usa o modelo de verdade.

Vale rodar dos dois jeitos e comparar. A diferença é grande, e é o tipo de coisa
que uma métrica bonita esconde.

## Resultados

Números da última execução (`reports/resumo.json`):

| Estimador | Partição | MAE | R² |
| --- | --- | --- | --- |
| baseline (mediana) | teste | 667 s | -0,22 |
| boosting | treino | 382 s | 0,62 |
| boosting | teste | **538 s** | **0,30** |

Leitura honesta: o modelo erra, em média, cerca de 9 minutos de intervalo, uns
2 minutos a menos que o chute constante. O R² de 0,30 diz que a estrutura
física da linha explica **parte** da oferta, e nada mais que isso. O resto é
demanda, contrato de operação e decisão de planejamento, que não estão no GTFS.

Isso é um resultado, não um fracasso: ele diz **quais dados faltam** para o
produto ficar de pé. É exatamente a conversa que a atividade integradora pede.

Duas observações que costumam gerar dúvida:

- **O R² do baseline é negativo.** Não é bug. O R² compara com prever a
  *média*, e o baseline prevê a *mediana*, que é o chute constante certo sob
  MAE. Métrica e baseline precisam falar a mesma língua.
- **A diferença entre treino (0,62) e teste (0,30)** é o custo de honestidade
  da validação por grupo. Numa divisão aleatória o número de teste ficaria bem
  mais bonito e bem menos verdadeiro.

Variáveis que mais pesam, por importância de permutação: hora do dia, tipo de
dia, área de operação, número de paradas e distância ao centro.

## Rodando

Requisitos: [uv](https://docs.astral.sh/uv/) e Git. Quarto só para o relatório.

```bash
uv sync --all-groups                      # ambiente de trabalho
uv run lab2 all                           # pipeline completo
uv run pytest                             # testes
uv run ruff check .                       # estilo

uv sync --all-groups --extra notebooks    # so' para renderizar Quarto
uv run quarto render notebooks/relatorio.qmd
```

Trocar de modelo sem tocar no código:

```bash
uv run lab2 evaluate --algoritmo lasso
uv run lab2 evaluate --algoritmo ridge
uv run lab2 evaluate --algoritmo arvore
uv run lab2 evaluate --algoritmo floresta
uv run lab2 evaluate --algoritmo boosting
```

## Problemas comuns

**`quarto render` quebra com `UnicodeDecodeError` no Windows.** É a codificação
padrão do Python no Windows (`cp1252`) encontrando um arquivo em UTF-8. Rode com
o modo UTF-8 ligado:

```bash
PYTHONUTF8=1 uv run quarto render notebooks/relatorio.qmd   # bash
$env:PYTHONUTF8=1; uv run quarto render notebooks/relatorio.qmd   # PowerShell
```

**`.venv\Scriptsctivate` recusa a rodar no PowerShell.** Política de execução
do Windows:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Instalei uma dependência e o terminal não a encontra.** Feche e abra o
terminal (às vezes o editor inteiro): a sessão antiga guarda o `PATH` antigo.

**Não tenho o token da API.** Rode assim mesmo. `pct_paradas_corredor` fica
zerada e `corredor_principal` fica faltante; o `SimpleImputer` do pipeline
resolve, e o modelo perde pouco.

## Material do laboratório

Este repositório é o **exemplo de código**. Os slides, o plano de aula, os
exercícios e o par de notebooks que compara tidymodels e scikit-learn ficam
separados, no material da aula, e são distribuídos pelo Blackboard.

O que está aqui:

- [`notebooks/relatorio.qmd`](notebooks/relatorio.qmd): o relatório reprodutível
  que consome o pacote.

## Licença

MIT. Os dados são públicos e pertencem à SPTrans; este repositório só os
consome.
