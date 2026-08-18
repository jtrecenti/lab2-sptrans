"""Dados sintéticos para os testes.

Os testes não baixam nada. Eles usam uma tabela pequena, com a mesma forma da
tabela analítica de verdade, o que deixa a suíte rodar em segundos e no CI, sem
token e sem rede.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab2_sptrans.config import carregar_config


@pytest.fixture(scope="session")
def cfg():
    return carregar_config()


@pytest.fixture
def tabela_falsa(cfg) -> pd.DataFrame:
    """Tabela com as colunas de `conf/config.yaml` e ruído controlado."""
    aleatorio = np.random.default_rng(0)
    n = 400
    numericas = cfg["features"]["numericas"]

    dados = {coluna: aleatorio.normal(10, 3, n) for coluna in numericas}
    dados["hora_inicio"] = aleatorio.integers(0, 24, n)
    dados["periodo_dia"] = aleatorio.choice(["pico_manha", "entrepico", "noite"], n)
    dados["tipo_dia"] = aleatorio.choice(["dia_util", "todos_os_dias"], n)
    dados["sentido"] = aleatorio.choice(["ida", "volta"], n)
    dados["area_operacao"] = aleatorio.choice(list("12345"), n)
    dados["tipo_linha"] = aleatorio.choice(["10", "21", "41"], n)
    dados["corredor_principal"] = aleatorio.choice(
        ["Pirituba", "Santo Amaro", None], n, p=[0.15, 0.15, 0.7]
    )
    dados["route_id"] = aleatorio.integers(0, 40, n).astype(str)

    tabela = pd.DataFrame(dados)
    # alvo com sinal de verdade, para o teste conseguir cobrar que o modelo
    # aprenda alguma coisa em vez de só rodar sem erro
    tabela["headway_seg"] = (600 + 40 * tabela["hora_inicio"] + aleatorio.normal(0, 60, n)).clip(
        120, 3600
    )
    return tabela
