"""O `Pipeline` de modelagem: pré-processamento + estimador, num objeto só.

Por que um `Pipeline` e não dois passos soltos:

1. **Não vaza.** Em cada dobra da validação cruzada, o pré-processamento é
   reajustado só no treino daquela dobra.
2. **Vai inteiro para produção.** O `.joblib` salvo contém a receita e o
   modelo. Quem for servir a API recebe o `DataFrame` cru e chama `.predict`.
3. **Ajusta junto.** O `GridSearchCV` pode buscar hiperparâmetro do modelo e
   do pré-processamento ao mesmo tempo.

Equivalente no tidymodels: `workflow()` + `tune_grid()` + `finalize_workflow()`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import Config
from .features import construir_preprocessador, separar_X_y

log = logging.getLogger(__name__)


def criar_estimador(nome: str, seed: int):
    """Devolve o estimador pedido no `conf/config.yaml`."""
    catalogo = {
        "ridge": Ridge(random_state=None),
        "arvore": DecisionTreeRegressor(random_state=seed),
        # n_jobs=1 de proposito: quem paraleliza e o GridSearchCV. Duas camadas
        # de paralelismo brigam pelos mesmos nucleos e deixam tudo mais lento.
        "floresta": RandomForestRegressor(random_state=seed, n_jobs=1),
        "boosting": HistGradientBoostingRegressor(random_state=seed),
    }
    if nome not in catalogo:
        raise ValueError(f"algoritmo '{nome}' desconhecido. Use: {sorted(catalogo)}")
    return catalogo[nome]


def construir_pipeline(cfg: Config, algoritmo: str | None = None) -> Pipeline:
    """Monta `preprocessador -> modelo`.

    O alvo é o intervalo entre ônibus em segundos, com distribuição bem
    assimétrica (de 2 minutos a 1 hora). `TransformedTargetRegressor` treina em
    log e devolve a predição de volta em segundos, sem que quem usa o modelo
    precise saber disso.
    """
    algoritmo = algoritmo or cfg["modelo"]["algoritmo"]
    estimador = TransformedTargetRegressor(
        regressor=criar_estimador(algoritmo, cfg.seed),
        func=np.log,
        inverse_func=np.exp,
    )
    return Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador(cfg)),
            ("modelo", estimador),
        ]
    )


def construir_baseline(cfg: Config) -> Pipeline:
    """Baseline honesto: prever sempre a mediana do treino.

    Todo relatório de modelo precisa dizer contra o quê o modelo está ganhando.

    Detalhe que costuma assustar: o R2 deste baseline dá **negativo**. Não é
    erro. O R2 mede a redução do erro quadrático em relação a prever a *média*,
    e a mediana não minimiza erro quadrático. Sob MAE, que é a métrica que
    escolhemos aqui, a mediana é o chute constante certo. Métrica e baseline
    precisam falar a mesma língua.
    """
    return Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador(cfg)),
            ("modelo", DummyRegressor(strategy="median")),
        ]
    )


def separar_treino_teste(cfg: Config, X, y, grupos):
    """Divisão treino/teste **por grupo**: uma linha inteira vai para um lado só."""
    divisor = GroupShuffleSplit(
        n_splits=1, test_size=cfg["modelo"]["teste_frac"], random_state=cfg.seed
    )
    idx_treino, idx_teste = next(divisor.split(X, y, groups=grupos))
    log.info(
        "treino: %d registros / %d linhas | teste: %d registros / %d linhas",
        len(idx_treino),
        grupos.iloc[idx_treino].nunique(),
        len(idx_teste),
        grupos.iloc[idx_teste].nunique(),
    )
    return (
        X.iloc[idx_treino],
        X.iloc[idx_teste],
        y.iloc[idx_treino],
        y.iloc[idx_teste],
        grupos.iloc[idx_treino],
    )


def _prefixar_grid(grid: dict) -> dict:
    """Ajusta os nomes do grid para o `TransformedTargetRegressor`.

    No `conf/config.yaml` o grid é escrito como `modelo__n_estimators`, que é
    como se leria sem o embrulho do log. Como o estimador está dentro de um
    `TransformedTargetRegressor`, o caminho real é
    `modelo__regressor__n_estimators`.
    """
    return {
        chave.replace("modelo__", "modelo__regressor__", 1): valores
        for chave, valores in grid.items()
    }


def ajustar(cfg: Config, tabela: pd.DataFrame, algoritmo: str | None = None) -> dict:
    """Treina o pipeline com busca de hiperparâmetros e devolve tudo o que interessa."""
    algoritmo = algoritmo or cfg["modelo"]["algoritmo"]
    X, y, grupos = separar_X_y(cfg, tabela)
    X_treino, X_teste, y_treino, y_teste, grupos_treino = separar_treino_teste(cfg, X, y, grupos)

    pipeline = construir_pipeline(cfg, algoritmo)
    grid = _prefixar_grid(cfg["modelo"]["grid"].get(algoritmo, {}))

    validacao = GroupKFold(n_splits=cfg["modelo"]["n_folds"])
    busca = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=validacao,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    n_combinacoes = len(list(ParameterGrid(grid))) if grid else 1
    log.info(
        "ajustando %s: %d combinações x %d dobras",
        algoritmo,
        n_combinacoes,
        cfg["modelo"]["n_folds"],
    )
    busca.fit(X_treino, y_treino, groups=grupos_treino)
    log.info("melhores hiperparâmetros: %s", busca.best_params_)

    baseline = construir_baseline(cfg).fit(X_treino, y_treino)

    return {
        "algoritmo": algoritmo,
        "busca": busca,
        "pipeline": busca.best_estimator_,
        "baseline": baseline,
        "X_treino": X_treino,
        "X_teste": X_teste,
        "y_treino": y_treino,
        "y_teste": y_teste,
    }


def salvar(cfg: Config, pipeline: Pipeline, nome: str = "modelo.joblib") -> Path:
    """Serializa o pipeline inteiro (receita + modelo) para uso em produção."""
    destino = cfg.caminho("models") / nome
    joblib.dump(pipeline, destino)
    log.info("modelo salvo em %s", destino)
    return destino


def carregar(cfg: Config, nome: str = "modelo.joblib") -> Pipeline:
    caminho = cfg.caminho("models") / nome
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Rode `lab2 train` antes.")
    return joblib.load(caminho)
