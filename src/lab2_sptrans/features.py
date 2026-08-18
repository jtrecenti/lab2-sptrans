"""Pré-processamento como parte do modelo.

Este é o ponto central do laboratório. Toda transformação que **aprende algo
dos dados** (média para imputar, escala para padronizar, categorias para o
one-hot) precisa aprender apenas no treino e ser aplicada no teste. A forma de
garantir isso é colocá-la dentro do `Pipeline`, e não em um script anterior.

Equivalente no tidymodels: este módulo é a `recipe()`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config


def colunas(cfg: Config) -> tuple[list[str], list[str]]:
    """Nomes das colunas numéricas e categóricas, lidos do `conf/config.yaml`."""
    return list(cfg["features"]["numericas"]), list(cfg["features"]["categoricas"])


def construir_preprocessador(cfg: Config) -> ColumnTransformer:
    """Monta o `ColumnTransformer` com um caminho por tipo de variável.

    Numéricas
        mediana para o que falta, depois padronização. A mediana resiste a
        valores extremos, comuns em extensão de linha e velocidade.

    Categóricas
        categoria explícita `desconhecido` para o que falta, depois one-hot.
        `handle_unknown="infrequent_if_exist"` evita que uma categoria vista só
        no teste quebre a predição, o que aconteceria em produção.
    """
    numericas, categoricas = colunas(cfg)

    caminho_numerico = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("padronizar", StandardScaler()),
        ]
    )

    caminho_categorico = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", caminho_numerico, numericas),
            ("cat", caminho_categorico, categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def separar_X_y(cfg: Config, tabela: pd.DataFrame):
    """Devolve `X`, `y` e o vetor de grupos usado na validação.

    O grupo é a linha de ônibus (`route_id`). Sem ele, a mesma linha aparece em
    treino e em teste e a avaliação fica otimista: o modelo decora a linha em
    vez de aprender o padrão de oferta.
    """
    numericas, categoricas = colunas(cfg)
    alvo = cfg["modelo"]["alvo"]
    grupo = cfg["modelo"]["grupo"]

    faltando = [c for c in [*numericas, *categoricas, alvo, grupo] if c not in tabela.columns]
    if faltando:
        raise KeyError(f"colunas ausentes na tabela analítica: {faltando}")

    X = tabela[[*numericas, *categoricas]].copy()
    for coluna in categoricas:
        X[coluna] = como_texto(X[coluna])
    y = tabela[alvo].astype(float)
    grupos = tabela[grupo]
    return X, y, grupos


def como_texto(coluna: pd.Series) -> pd.Series:
    """Garante que a coluna categórica seja texto, preservando o que falta.

    CSV não guarda tipo. Uma coluna como `tipo_linha`, que tem "10" e "1A",
    volta da leitura com `int` e `str` misturados, e o `OneHotEncoder` recusa.
    Formatos com esquema (parquet, por exemplo) evitariam isso; aqui a
    conversão é explícita para o CSV continuar legível a olho nu.
    """
    return coluna.where(coluna.isna(), coluna.astype(str))
