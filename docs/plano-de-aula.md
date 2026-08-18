# Plano do Lab 2: pipelines com scikit-learn

**Disciplina**: Prática Avançada em Data Science e Visualização (PADS Insper, turma PADSONL08)
**Data**: sexta, 21 de agosto de 2026, 19h00 às 22h30 (remota, 3h30)
**Responsável**: Julio Trecenti (monitor)
**Repositório**: <https://github.com/jtrecenti/lab2-sptrans>

---

## Onde este lab se encaixa

No Lab 1 (05/08) a turma saiu do notebook solto e chegou a um pacote Python
versionado no GitHub, com `uv`, ambiente virtual, Quarto e GitHub Pages. O
fechamento daquele lab já anunciou este: *"no Lab 2 eu vou falar de pipelines
do scikit-learn, e de quebra um de-para do que a gente tinha de tidymodels no R
para o que a gente vai ter no scikit-learn"*.

Este lab cumpre exatamente isso. Ele pega a estrutura de pacote do Lab 1 e
coloca dentro dela um **fluxo de produção**: ETL de dado real, pré-processamento
que não vaza, ajuste de modelo, avaliação honesta e artefato serializado pronto
para virar API na disciplina de Deploy.

## Objetivos de aprendizagem

Ao final da aula, o aluno deve conseguir:

1. **Traduzir** um fluxo de tidymodels para scikit-learn e vice-versa (OV6).
2. **Construir** um `Pipeline` com `ColumnTransformer` que trata variáveis
   numéricas e categóricas, e explicar por que o pré-processamento precisa
   morar dentro dele (OV6, prepara Deploy).
3. **Escolher** uma estratégia de validação coerente com o uso do modelo,
   identificando vazamento por grupo (OV6).
4. **Ler** um resultado de modelo criticamente: comparar com baseline,
   distinguir treino de teste e dizer o que o número não responde (OV1, OV5).
5. **Rodar e modificar** um pipeline de ponta a ponta a partir de um comando só,
   com parâmetros em arquivo de configuração (OV4, prepara Deploy).

Nível de Bloom: Aplicar, Analisar e Avaliar.
Dinâmica: laboratório guiado, programação em pares e discussão de resultados.

## Antes da aula

Enviar no Teams, com dois dias de antecedência:

- link do repositório e o comando de preparo:
  ```bash
  git clone https://github.com/jtrecenti/lab2-sptrans
  cd lab2-sptrans
  uv sync --all-groups
  uv run lab2 extract
  ```
- pedido de cadastro (gratuito) em <https://www.sptrans.com.br/desenvolvedores/>
  para quem quiser o token da API Olho Vivo. **É opcional**: o pipeline roda
  sem ele.
- aviso de que o `uv sync` baixa cerca de 400 MB e é melhor fazer em casa, não
  na hora da aula. A lição do Lab 1 é que instalação em sala consome uma hora.

::: nota
**Apresentações de gráficos.** A planilha de apresentações
([link](https://docs.google.com/spreadsheets/d/1rUQfvhQMjehs0hrNUdlAZCbUz63O59famTXF9j0WIv0/edit))
não tem linha para 21/08. Criar a linha e abrir para até 4 voluntários. Se
ninguém se inscrever, antecipar os quatro de 26/08 (Willian, Beatriz, Leonardo
Koga e Celso), o que alivia aquela aula. Combinar com a Gabrielle, representante
da turma.
:::

## Roteiro

| Horário | Bloco | Formato | Duração |
| --- | --- | --- | --- |
| 19h00 | Abertura e recados | exposição | 10 min |
| 19h10 | **Apresentações de gráficos** (até 4 alunos) | apresentação + crítica | 25 min |
| 19h35 | Do Lab 1 ao Lab 2: a pergunta e o repositório | exposição dialogada | 15 min |
| 19h50 | **Slides 1**: dividir, receita, modelo | exposição com código lado a lado | 20 min |
| 20h10 | **Mão na massa 1**: rodar o pipeline inteiro | laboratório guiado | 25 min |
| 20h35 | Intervalo | | 15 min |
| 20h50 | **Slides 2**: `Pipeline`, tuning, validação por grupo | exposição dialogada | 15 min |
| 21h05 | **Mão na massa 2**: exercícios 1 a 3, em duplas | programação em pares | 45 min |
| 21h50 | Discussão: o que o resultado diz e o que não diz | discussão em plenário | 15 min |
| 22h05 | **Mão na massa 3**: exercício aberto + ponte com a integradora | laboratório com mentoria | 20 min |
| 22h25 | Fechamento: o que entregar e o que vem no Lab 3 | exposição | 5 min |

**Tempo de mão na massa: 90 minutos**, 43% da aula. É o bloco que não pode ser
comprimido; se algo atrasar, cortar dos slides, não daqui.

---

## Detalhamento por bloco

### 19h00 (10 min) Abertura e recados

- Retomada de uma frase do Lab 1: *o Claude Code faria essa estrutura sozinho, e
  é justamente por isso que você precisa saber qual estrutura pedir*. Hoje é a
  mesma lógica um nível acima: a IA escreve o `Pipeline`; você precisa saber se
  ele está certo, e o que acontece se estiver errado.
- Aula extra (material de compensação do sábado 01/08): confirmar data com a
  Gabrielle.
- Combinar a dinâmica: câmeras abertas nos blocos de mão na massa, salas
  separadas por dupla no Teams, monitor circulando.

### 19h10 (25 min) Apresentações de gráficos

Até quatro alunos, 3 a 5 minutos cada, sobre uma visualização boa ou ruim vista
na mídia. Depois de cada uma, 1 minuto de crítica coletiva com uma pergunta
fixa: **qual decisão esse gráfico apoia?**

Isso amarra o bloco no tema do dia: o modelo também precisa apoiar decisão, não
só existir.

### 19h35 (15 min) Do Lab 1 ao Lab 2

Abrir o repositório no navegador e percorrer a estrutura, sem código ainda:

- a pergunta: **quanta oferta de ônibus a SPTrans coloca em cada linha, em cada
  hora, e o que explica isso?**;
- as duas fontes, GTFS e API Olho Vivo, e o fato de que elas se cruzam por
  proximidade geográfica, não por chave. Essa junção suja é a diferença entre
  dado de curso e dado de verdade;
- as quatro regras do README: dado gerado não entra no Git, segredo não entra no
  Git, parâmetro fica no YAML, cada etapa é um comando;
- por que este exemplo usa o tema 3 da integradora (mobilidade em SP): no Lab 1
  o exemplo foi o tema 2 (dados jurídicos). Deixar claro que **não** é
  recomendação de tema.

### 19h50 (20 min) Slides, parte 1

Slides 1 a 8 de `slides/tidymodels-para-sklearn.qmd`: o mapa dos pacotes, a
diferença de filosofia (papéis contra colunas), divisão de dados, receita e
modelo.

Ritmo: mostrar o R primeiro, deixar a turma reconhecer, e só então o Python.
A tabela de de-para dos `step_*` é o slide para deixar aberto mais tempo.

### 20h10 (25 min) Mão na massa 1

Todos rodando, cada um na sua máquina:

```bash
uv run lab2 all
```

Enquanto roda, pedir que **leiam o log**: quantos registros de treino e de
teste, quantas linhas de ônibus de cada lado, quantas combinações de
hiperparâmetros. Depois, abrir `reports/resumo.json` e responder em voz alta:

1. Quanto o modelo ganha do baseline, em minutos?
2. Por que o R² do baseline é negativo?
3. Por que treino e teste são tão diferentes?

Quem travar na instalação forma dupla com quem já rodou (mesma dinâmica de
squads do Lab 1: ninguém larga mão de ninguém).

### 20h50 (15 min) Slides, parte 2

Slides 9 a 16: `workflow` = `Pipeline`, o operador `__`, ajuste conjunto de
receita e modelo, finalização, métricas e serialização.

Slide para insistir: **onde o scikit-learn não segura sua mão** (sem fórmula,
nomes de coluna somem, tipo não vem de graça, `groups=` precisa ser passado
sempre).

### 21h05 (45 min) Mão na massa 2

Exercícios 1 a 3 de [`exercicios.md`](exercicios.md), em duplas:

1. trocar o algoritmo pelo `conf/config.yaml` e comparar;
2. quebrar a validação de propósito, trocando `GroupKFold` por `KFold`, e medir
   o tamanho da mentira;
3. adicionar uma variável nova ao pré-processamento.

O exercício 2 é o coração do lab. Reservar tempo para ele.

Monitor circula pelas salas. A cada 15 minutos, um "pulso": alguém compartilha a
tela e mostra onde está.

### 21h50 (15 min) Discussão

Plenário, com três perguntas na tela:

1. **O R² deu 0,30. O modelo presta?** Puxar a resposta para o uso: para achar
   linha fora do padrão, o resíduo já serve; para prever oferta de linha nova,
   não basta.
2. **Quem ganhou mais R² trocando o `GroupKFold` por `KFold`?** Coletar os
   números e mostrar que o ganho é ilusório.
3. **Que dado falta?** Demanda (Censo por setor), realizado (coleta da API),
   contrato de concessão. Fechar com: *o modelo diz quais dados faltam, e isso
   já é um entregável*.

### 22h05 (20 min) Mão na massa 3

Exercício 4 (aberto) e conversa individual sobre a integradora: cada grupo
identifica, no próprio projeto, qual é o alvo, qual é a unidade de análise e
qual seria o agrupamento correto da validação. Anotar no repositório do grupo.

### 22h25 (5 min) Fechamento

- **O que entregar** (opcional, com feedback): um fork do repositório com pelo
  menos um exercício resolvido e o `README` explicando o que mudou. Prazo: uma
  semana.
- Próximo lab (31/08): dashboard interativo conectado à saída do modelo, e um
  pouco de UX e WebAssembly.
- Recado sobre o Deploy: o `models/modelo.joblib` gerado hoje é literalmente o
  arquivo que vai virar endpoint de API lá.

---

## Planos B

| Se acontecer | Fazer |
| --- | --- |
| Muita gente sem ambiente | Rodar em duplas, uma máquina por dupla. Não parar a aula para instalar. |
| Internet do Insper cair (aconteceu no Lab 1) | Passar o compartilhamento de tela para um aluno voluntário e conduzir por voz. |
| Download do GTFS lento | O ZIP tem 12 MB e vem de um espelho público; se falhar, distribuir por link direto no Teams. |
| Atraso acumulado de 20 min | Cortar os slides 9 a 16 e deixá-los como material de leitura. Não cortar o exercício 2. |
| Sobrar tempo | Exercício 5 (desafio): trocar o alvo para classificação de alta e baixa frequência. |

## Material

- Slides: `slides/tidymodels-para-sklearn.qmd` (renderizado em HTML)
- Relatório de exemplo: `notebooks/relatorio.qmd`
- Exercícios: `docs/exercicios.md`
- Repositório completo, público, com CI verde
