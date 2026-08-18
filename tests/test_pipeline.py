"""Testes do pipeline de modelagem.

A pergunta que estes testes respondem é a que interessa em produção: o objeto
salvo em disco recebe dado cru e devolve predição, sem passo manual no meio?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from lab2_sptrans import model
from lab2_sptrans.features import construir_preprocessador, separar_X_y


def test_preprocessador_trata_faltante_e_categoria(cfg, tabela_falsa):
    """Depois do pré-processamento não pode sobrar `NaN` nem texto."""
    X, _, _ = separar_X_y(cfg, tabela_falsa)
    assert X["corredor_principal"].isna().any(), "o teste precisa de faltantes de verdade"

    matriz = construir_preprocessador(cfg).fit_transform(X)
    assert not np.isnan(np.asarray(matriz, dtype=float)).any()
    assert matriz.shape[0] == len(X)


def test_split_por_grupo_nao_reparte_a_mesma_linha(cfg, tabela_falsa):
    """Nenhuma linha de ônibus pode estar em treino e em teste ao mesmo tempo."""
    X, y, grupos = separar_X_y(cfg, tabela_falsa)
    idx = pd.Series(range(len(X)))
    X = X.assign(_idx=idx.to_numpy())

    X_treino, X_teste, *_ = model.separar_treino_teste(cfg, X, y, grupos)
    grupos_treino = set(grupos.iloc[X_treino["_idx"].to_numpy()])
    grupos_teste = set(grupos.iloc[X_teste["_idx"].to_numpy()])
    assert grupos_treino.isdisjoint(grupos_teste)


def test_pipeline_aprende_mais_que_o_baseline(cfg, tabela_falsa):
    """Com sinal plantado nos dados, o modelo precisa ganhar do chute constante."""
    X, y, _ = separar_X_y(cfg, tabela_falsa)
    pipeline = model.construir_pipeline(cfg, "arvore").fit(X, y)
    baseline = model.construir_baseline(cfg).fit(X, y)

    erro_modelo = mean_absolute_error(y, pipeline.predict(X))
    erro_baseline = mean_absolute_error(y, baseline.predict(X))
    assert erro_modelo < erro_baseline


def test_modelo_salvo_recebe_dado_cru(cfg, tabela_falsa, tmp_path, monkeypatch):
    """O `.joblib` precisa carregar a receita junto: `predict` num `DataFrame` cru."""
    X, y, _ = separar_X_y(cfg, tabela_falsa)
    pipeline = model.construir_pipeline(cfg, "arvore").fit(X, y)

    monkeypatch.setattr(cfg.__class__, "caminho", lambda self, nome: tmp_path)
    caminho = model.salvar(cfg, pipeline, nome="teste.joblib")
    assert caminho.exists()

    recarregado = model.carregar(cfg, nome="teste.joblib")
    predicao = recarregado.predict(X.head(5))
    assert len(predicao) == 5
    assert np.all(predicao > 0), "headway em segundos nunca é negativo"


def test_categoria_nova_no_teste_nao_derruba_a_predicao(cfg, tabela_falsa):
    """Em produção aparece linha nova. O modelo tem que responder, não estourar."""
    X, y, _ = separar_X_y(cfg, tabela_falsa)
    pipeline = model.construir_pipeline(cfg, "arvore").fit(X, y)

    novo = X.head(3).copy()
    novo["corredor_principal"] = "Corredor Que Ainda Nao Existe"
    novo["area_operacao"] = "9"
    assert len(pipeline.predict(novo)) == 3


def test_algoritmo_desconhecido_falha_cedo(cfg):
    with pytest.raises(ValueError, match="desconhecido"):
        model.criar_estimador("rede_neural_gigante", seed=1)
