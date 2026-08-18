"""Transformação (o T do ETL): das tabelas cruas para a tabela analítica.

A unidade de análise é **linha x sentido x faixa horária**: uma linha de ônibus,
num sentido, numa faixa de horário do dia. É a granularidade em que a SPTrans
publica a oferta programada, no `frequencies.txt` do GTFS.

Saída: `data/processed/viagens.csv.gz`, uma linha por unidade de análise, com o
alvo (`headway_seg`) e as variáveis explicativas.

Nada de `scikit-learn` aqui. Esta camada produz uma tabela; o pré-processamento
que depende de treino/teste (imputação, padronização, one-hot) vive no
`features.py`, dentro do `Pipeline`. Fazer imputação aqui seria vazamento.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .extract import ler_gtfs

log = logging.getLogger(__name__)

RAIO_TERRA_KM = 6371.0

# service_id do GTFS da SPTrans: cada posição indica se o serviço opera em
# dia útil (U), sábado (S) e domingo/feriado (D). "U__" = só dia útil.
TIPO_DIA = {
    "USD": "todos_os_dias",
    "US_": "util_e_sabado",
    "U__": "dia_util",
    "_SD": "fim_de_semana",
    "_S_": "sabado",
    "__D": "domingo",
}


def _para_segundos(hora: pd.Series) -> pd.Series:
    """Converte "HH:MM:SS" do GTFS em segundos desde a meia-noite do serviço.

    O GTFS permite hora maior que 24 (uma viagem que sai 23h50 e chega 00h20 do
    dia seguinte é registrada como 24:20:00). `pd.to_datetime` quebraria nesses
    casos, então convertemos na mão.
    """
    partes = hora.str.split(":", expand=True).astype(float)
    return partes[0] * 3600 + partes[1] * 60 + partes[2]


def _distancia_km(lat1, lon1, lat2, lon2):
    """Distância de Haversine, em quilômetros, entre pontos (vetorizada)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def _periodo_do_dia(hora: pd.Series) -> pd.Series:
    """Agrupa a hora em faixas com significado operacional."""
    faixas = pd.Series("entrepico", index=hora.index, dtype=object)
    faixas[hora < 5] = "madrugada"
    faixas[(hora >= 5) & (hora < 9)] = "pico_manha"
    faixas[(hora >= 17) & (hora < 20)] = "pico_tarde"
    faixas[hora >= 20] = "noite"
    return faixas


# --------------------------------------------------------------------------- #
# Blocos da tabela analítica
# --------------------------------------------------------------------------- #
def atributos_por_viagem(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uma linha por `trip_id`: tamanho, duração e pontas do percurso."""
    tempos = gtfs["stop_times"].copy()
    tempos["stop_sequence"] = tempos["stop_sequence"].astype(int)
    tempos["segundos"] = _para_segundos(tempos["arrival_time"])
    tempos = tempos.sort_values(["trip_id", "stop_sequence"])

    agregado = tempos.groupby("trip_id").agg(
        n_paradas=("stop_id", "size"),
        segundos_inicio=("segundos", "min"),
        segundos_fim=("segundos", "max"),
        stop_origem=("stop_id", "first"),
        stop_destino=("stop_id", "last"),
    )
    agregado["duracao_min"] = (agregado["segundos_fim"] - agregado["segundos_inicio"]) / 60
    return agregado.reset_index()


def extensao_por_shape(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extensão do traçado, em km, por `shape_id`.

    O `shape_dist_traveled` do feed da SPTrans está em metros e é acumulado,
    então o último ponto do traçado dá a extensão total.
    """
    shapes = gtfs["shapes"][["shape_id", "shape_dist_traveled"]].copy()
    shapes["shape_dist_traveled"] = pd.to_numeric(shapes["shape_dist_traveled"], errors="coerce")
    extensao = shapes.groupby("shape_id")["shape_dist_traveled"].max() / 1000
    return extensao.rename("extensao_km").reset_index()


def coordenadas_paradas(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    paradas = gtfs["stops"][["stop_id", "stop_lat", "stop_lon"]].copy()
    paradas["stop_lat"] = pd.to_numeric(paradas["stop_lat"], errors="coerce")
    paradas["stop_lon"] = pd.to_numeric(paradas["stop_lon"], errors="coerce")
    return paradas.dropna(subset=["stop_lat", "stop_lon"])


def marcar_paradas_de_corredor(
    paradas: pd.DataFrame, paradas_corredor: pd.DataFrame | None, raio_m: int
) -> pd.DataFrame:
    """Cruzamento espacial GTFS x Olho Vivo.

    Os dois sistemas usam códigos de parada diferentes, então a junção é por
    proximidade: uma parada do GTFS a menos de `raio_m` de uma parada de
    corredor da API é considerada parada de corredor. Devolve `paradas` com as
    colunas `em_corredor` e `nome_corredor`.
    """
    resultado = paradas.copy()
    resultado["em_corredor"] = False
    resultado["nome_corredor"] = pd.NA

    if paradas_corredor is None or paradas_corredor.empty:
        log.warning("sem paradas de corredor: colunas de corredor ficarão vazias")
        return resultado

    lat_gtfs = resultado["stop_lat"].to_numpy()[:, None]
    lon_gtfs = resultado["stop_lon"].to_numpy()[:, None]
    lat_ov = paradas_corredor["lat"].to_numpy()[None, :]
    lon_ov = paradas_corredor["lon"].to_numpy()[None, :]

    distancias = _distancia_km(lat_gtfs, lon_gtfs, lat_ov, lon_ov) * 1000
    mais_proxima = distancias.argmin(axis=1)
    distancia_minima = distancias.min(axis=1)

    dentro = distancia_minima <= raio_m
    resultado["em_corredor"] = dentro
    nomes = paradas_corredor["nome_corredor"].to_numpy()[mais_proxima]
    resultado.loc[dentro, "nome_corredor"] = nomes[dentro]
    log.info(
        "%d de %d paradas do GTFS caem em corredor (raio de %d m)",
        int(dentro.sum()),
        len(resultado),
        raio_m,
    )
    return resultado


def cobertura_de_corredor(gtfs: dict[str, pd.DataFrame], paradas: pd.DataFrame) -> pd.DataFrame:
    """Por viagem: fração de paradas em corredor e corredor predominante."""
    tempos = gtfs["stop_times"][["trip_id", "stop_id"]].merge(
        paradas[["stop_id", "em_corredor", "nome_corredor"]], on="stop_id", how="left"
    )
    tempos["em_corredor"] = tempos["em_corredor"].fillna(False)

    proporcao = tempos.groupby("trip_id")["em_corredor"].mean().rename("pct_paradas_corredor")

    # corredor predominante: o mais frequente entre as paradas de corredor da
    # viagem. Fica faltante para a maioria das linhas, que não usa corredor.
    com_corredor = tempos.dropna(subset=["nome_corredor"])
    if com_corredor.empty:
        predominante = pd.Series(dtype=object, name="corredor_principal")
    else:
        predominante = (
            com_corredor.groupby("trip_id")["nome_corredor"]
            .agg(lambda serie: serie.value_counts().idxmax())
            .rename("corredor_principal")
        )
    return pd.concat([proporcao, predominante], axis=1).reset_index()


def linhas_por_parada(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quantas linhas distintas atendem cada parada.

    Feature de engenharia: uma parada servida por muitas linhas costuma ser um
    terminal ou um ponto de troca, e isso muda o papel da linha na rede. A
    informacao esta no GTFS, mas nao em nenhuma coluna: precisa ser construida.
    """
    ligacao = gtfs["stop_times"][["trip_id", "stop_id"]].merge(
        gtfs["trips"][["trip_id", "route_id"]], on="trip_id", how="left"
    )
    contagem = ligacao.groupby("stop_id")["route_id"].nunique()
    return contagem.rename("n_linhas_parada").reset_index()


def faixas_de_frequencia(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uma linha por viagem x faixa horária, com o alvo `headway_seg`."""
    freq = gtfs["frequencies"].copy()
    freq["headway_seg"] = pd.to_numeric(freq["headway_secs"], errors="coerce")
    segundos = _para_segundos(freq["start_time"])
    # hora de serviço passa de 24 em viagens que viram o dia; 24h -> 0h
    freq["hora_inicio"] = (segundos // 3600 % 24).astype(int)
    freq["periodo_dia"] = _periodo_do_dia(freq["hora_inicio"])
    return freq[["trip_id", "hora_inicio", "periodo_dia", "headway_seg"]].dropna(
        subset=["headway_seg"]
    )


# --------------------------------------------------------------------------- #
# Montagem
# --------------------------------------------------------------------------- #
def montar_tabela(cfg: Config) -> pd.DataFrame:
    """Junta tudo e devolve a tabela analítica."""
    gtfs = ler_gtfs(cfg)

    caminho_corredor = cfg.caminho("raw") / "paradas_corredor.csv"
    paradas_corredor = pd.read_csv(caminho_corredor) if caminho_corredor.exists() else None

    paradas = coordenadas_paradas(gtfs)
    paradas = marcar_paradas_de_corredor(paradas, paradas_corredor, cfg["etl"]["raio_corredor_m"])

    viagens = (
        gtfs["trips"][["trip_id", "route_id", "service_id", "direction_id", "shape_id"]]
        .merge(atributos_por_viagem(gtfs), on="trip_id", how="inner")
        .merge(extensao_por_shape(gtfs), on="shape_id", how="left")
        .merge(cobertura_de_corredor(gtfs, paradas), on="trip_id", how="left")
        .merge(
            gtfs["routes"][["route_id", "route_short_name", "route_long_name"]],
            on="route_id",
            how="left",
        )
    )

    # coordenadas das pontas do percurso, junto com o tamanho do no na rede
    pontas = paradas[["stop_id", "stop_lat", "stop_lon"]].merge(
        linhas_por_parada(gtfs), on="stop_id", how="left"
    )
    viagens = viagens.merge(
        pontas.rename(
            columns={
                "stop_id": "stop_origem",
                "stop_lat": "lat_origem",
                "stop_lon": "lon_origem",
                "n_linhas_parada": "n_linhas_origem",
            }
        ),
        on="stop_origem",
        how="left",
    ).merge(
        pontas.rename(
            columns={
                "stop_id": "stop_destino",
                "stop_lat": "lat_destino",
                "stop_lon": "lon_destino",
                "n_linhas_parada": "n_linhas_destino",
            }
        ),
        on="stop_destino",
        how="left",
    )

    centro_lat, centro_lon = cfg["etl"]["centro_lat"], cfg["etl"]["centro_lon"]
    viagens["dist_centro_origem_km"] = _distancia_km(
        viagens["lat_origem"], viagens["lon_origem"], centro_lat, centro_lon
    )
    viagens["dist_centro_destino_km"] = _distancia_km(
        viagens["lat_destino"], viagens["lon_destino"], centro_lat, centro_lon
    )

    # derivadas de tamanho da linha
    viagens["velocidade_kmh"] = viagens["extensao_km"] / (viagens["duracao_min"] / 60)
    viagens["paradas_por_km"] = viagens["n_paradas"] / viagens["extensao_km"]

    # o código da linha da SPTrans carrega informação: "8000-10" tem o dígito
    # inicial ligado à área de operação e o sufixo ao tipo de atendimento.
    # `.str.strip()` não é preciosismo: o feed traz espaço sobrando em alguns
    # códigos e, sem limpar, "10" e "10 " viram duas categorias diferentes no
    # one-hot, com a mesma informação repartida em duas colunas.
    codigo = viagens["route_short_name"].fillna(viagens["route_id"]).astype(str).str.strip()
    viagens["area_operacao"] = codigo.str[0]
    viagens["tipo_linha"] = codigo.str.split("-").str[-1].str.strip()

    viagens["tipo_dia"] = viagens["service_id"].map(TIPO_DIA).fillna("outro")
    viagens["sentido"] = np.where(viagens["direction_id"] == "0", "ida", "volta")

    tabela = viagens.merge(faixas_de_frequencia(gtfs), on="trip_id", how="inner")
    tabela = _limpar(tabela)

    log.info("tabela analítica: %d linhas x %d colunas", *tabela.shape)
    return tabela


def _limpar(tabela: pd.DataFrame) -> pd.DataFrame:
    """Descarta registros impossíveis e deixa os tipos coerentes.

    Descartar aqui é aceitável porque a regra não depende de treino ou teste:
    uma viagem com duração zero é erro de cadastro, não informação.
    """
    antes = len(tabela)
    tabela = tabela[
        tabela["duracao_min"].gt(0)
        & tabela["extensao_km"].gt(0)
        & tabela["headway_seg"].gt(0)
        & tabela["velocidade_kmh"].between(1, 80)
    ].copy()
    log.info("limpeza: %d registros descartados de %d", antes - len(tabela), antes)

    # ausência de parada de corredor é informação, não dado faltante: a fração
    # é zero mesmo. Já `corredor_principal` fica faltante de propósito, para
    # ser imputada dentro do Pipeline (ver `features.py`).
    tabela["pct_paradas_corredor"] = tabela["pct_paradas_corredor"].fillna(0.0)
    return tabela.reset_index(drop=True)


def executar(cfg: Config) -> pd.DataFrame:
    """Etapa `transform` do pipeline: monta e grava a tabela analítica."""
    tabela = montar_tabela(cfg)
    destino = cfg.caminho("processed") / "viagens.csv.gz"
    tabela.to_csv(destino, index=False, compression="gzip")
    log.info("tabela analítica salva em %s", destino)
    return tabela


def carregar_tabela(cfg: Config) -> pd.DataFrame:
    """Lê a tabela analítica já processada."""
    caminho = cfg.caminho("processed") / "viagens.csv.gz"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Rode `lab2 transform` antes.")
    return pd.read_csv(caminho, low_memory=False)
