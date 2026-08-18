"""Testes da camada de transformação.

O que se testa aqui é regra de negócio do dado, não `pandas`. Cada teste é uma
armadilha real do GTFS que já mordeu alguém.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab2_sptrans.transform import (
    _distancia_km,
    _para_segundos,
    _periodo_do_dia,
    linhas_por_parada,
    marcar_paradas_de_corredor,
)


def test_hora_depois_da_meia_noite_nao_quebra():
    """O GTFS escreve 24:20:00 para 00h20 do dia seguinte."""
    horas = pd.Series(["00:00:00", "07:30:00", "24:20:00", "25:05:30"])
    segundos = _para_segundos(horas)
    assert segundos.tolist() == [0, 27000, 87600, 90330]
    assert segundos.is_monotonic_increasing


def test_distancia_conhecida():
    """Praça da Sé até o Pico do Jaraguá: cerca de 17 km em linha reta."""
    distancia = _distancia_km(-23.550520, -46.633308, -23.456389, -46.766111)
    assert 16.5 < distancia < 17.5


def test_periodo_do_dia_cobre_as_24_horas():
    periodos = _periodo_do_dia(pd.Series(range(24)))
    assert not periodos.isna().any()
    assert periodos[3] == "madrugada"
    assert periodos[7] == "pico_manha"
    assert periodos[13] == "entrepico"
    assert periodos[18] == "pico_tarde"
    assert periodos[22] == "noite"


def test_parada_fora_do_raio_nao_vira_corredor():
    """A junção espacial precisa respeitar o raio: perto não é igual."""
    paradas = pd.DataFrame(
        {
            "stop_id": ["a", "b"],
            "stop_lat": [-23.5500, -23.6000],  # 'a' colada, 'b' a ~5 km
            "stop_lon": [-46.6330, -46.6330],
        }
    )
    corredor = pd.DataFrame({"lat": [-23.5501], "lon": [-46.6330], "nome_corredor": ["Teste"]})
    resultado = marcar_paradas_de_corredor(paradas, corredor, raio_m=120)
    assert resultado.loc[0, "em_corredor"]
    assert resultado.loc[0, "nome_corredor"] == "Teste"
    assert not resultado.loc[1, "em_corredor"]
    assert pd.isna(resultado.loc[1, "nome_corredor"])


def test_sem_paradas_de_corredor_o_pipeline_segue():
    """Sem token da API, o ETL não pode quebrar: só fica sem a coluna."""
    paradas = pd.DataFrame({"stop_id": ["a"], "stop_lat": [-23.55], "stop_lon": [-46.63]})
    resultado = marcar_paradas_de_corredor(paradas, None, raio_m=120)
    assert not resultado["em_corredor"].any()


def test_linhas_por_parada_conta_linha_distinta_e_nao_viagem():
    """Ida e volta da mesma linha na mesma parada contam como uma linha só."""
    gtfs = {
        "stop_times": pd.DataFrame({"trip_id": ["t1", "t2", "t3"], "stop_id": ["p1", "p1", "p1"]}),
        "trips": pd.DataFrame({"trip_id": ["t1", "t2", "t3"], "route_id": ["L1", "L1", "L2"]}),
    }
    contagem = linhas_por_parada(gtfs)
    assert contagem.loc[contagem.stop_id == "p1", "n_linhas_parada"].item() == 2


def test_distancia_e_vetorizada():
    """A junção espacial depende de broadcasting; se quebrar, o ETL trava."""
    lat = np.array([[-23.55], [-23.56]])
    lon = np.array([[-46.63], [-46.64]])
    resultado = _distancia_km(lat, lon, np.array([[-23.55, -23.60]]), np.array([[-46.63, -46.60]]))
    assert resultado.shape == (2, 2)
