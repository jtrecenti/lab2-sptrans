"""Extração (o E do ETL).

Duas fontes:

1. **GTFS da SPTrans**: o horário planejado da rede de ônibus (linhas, paradas,
   traçados, frequências). Arquivo ZIP, baixado uma vez e guardado em `data/raw/`.
2. **API Olho Vivo**: a camada em tempo real da SPTrans. Aqui usamos os dois
   endpoints estáveis (corredores e paradas de corredor) e, opcionalmente, um
   retrato da frota em operação.

Esta camada só busca e guarda dado bruto. Ela não limpa, não junta e não
inventa coluna: isso é trabalho do `transform.py`.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .config import Config, token_olhovivo

log = logging.getLogger(__name__)

ARQUIVOS_GTFS = (
    "agency.txt",
    "calendar.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
    "frequencies.txt",
    "shapes.txt",
)


# --------------------------------------------------------------------------- #
# GTFS
# --------------------------------------------------------------------------- #
def baixar_gtfs(cfg: Config, forcar: bool = False) -> Path:
    """Baixa o ZIP do GTFS da SPTrans para `data/raw/gtfs_sptrans.zip`."""
    destino = cfg.caminho("raw") / "gtfs_sptrans.zip"
    if destino.exists() and not forcar:
        log.info("GTFS já existe em %s (use --forcar para rebaixar)", destino)
        return destino

    url = cfg["fontes"]["gtfs_url"]
    log.info("baixando GTFS de %s", url)
    resposta = requests.get(url, timeout=600, stream=True)
    resposta.raise_for_status()
    with open(destino, "wb") as arquivo:
        for pedaco in resposta.iter_content(chunk_size=1 << 20):
            arquivo.write(pedaco)
    log.info("GTFS salvo em %s (%.1f MB)", destino, destino.stat().st_size / 1e6)
    return destino


def ler_gtfs(cfg: Config) -> dict[str, pd.DataFrame]:
    """Lê as tabelas do GTFS direto do ZIP, sem descompactar em disco.

    Tudo entra como texto: converter tipo é decisão do `transform.py`, não da
    leitura. Ler número como texto e converter depois evita perder zero à
    esquerda em código de linha e de parada.
    """
    caminho = cfg.caminho("raw") / "gtfs_sptrans.zip"
    if not caminho.exists():
        raise FileNotFoundError(f"GTFS não encontrado em {caminho}. Rode `lab2 extract` antes.")

    tabelas: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(caminho) as zip_gtfs:
        disponiveis = set(zip_gtfs.namelist())
        for nome in ARQUIVOS_GTFS:
            if nome not in disponiveis:
                log.warning("arquivo %s ausente no feed GTFS", nome)
                continue
            with zip_gtfs.open(nome) as fluxo:
                # O feed da SPTrans é UTF-8. Declarar isso importa: `latin-1`
                # decodifica qualquer byte sem levantar erro, então uma escolha
                # errada de codificação não quebra, só corrompe o acento em
                # silêncio ("METRÔ" vira "METRÃ"). Erro que não estoura é
                # o pior tipo de erro.
                tabelas[nome.removesuffix(".txt")] = pd.read_csv(
                    fluxo, dtype=str, encoding="utf-8"
                )
    return tabelas


# --------------------------------------------------------------------------- #
# API Olho Vivo
# --------------------------------------------------------------------------- #
class OlhoVivo:
    """Cliente mínimo da API Olho Vivo.

    A API autentica por cookie de sessão: um POST em `/Login/Autenticar` com o
    token e, daí em diante, as chamadas usam o cookie. Por isso guardamos uma
    `requests.Session` viva.
    """

    def __init__(self, base_url: str, token: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.sessao = requests.Session()
        self._autenticado = False

    def autenticar(self) -> None:
        resposta = self.sessao.post(
            f"{self.base_url}/Login/Autenticar",
            params={"token": self.token},
            headers={"Content-Length": "0"},
            timeout=self.timeout,
        )
        resposta.raise_for_status()
        if resposta.text.strip().lower() != "true":
            raise RuntimeError("Olho Vivo recusou o token (confira API_OLHO_VIVO no .env)")
        self._autenticado = True

    def get(self, rota: str, **params) -> object:
        if not self._autenticado:
            self.autenticar()
        resposta = self.sessao.get(
            f"{self.base_url}/{rota.lstrip('/')}", params=params, timeout=self.timeout
        )
        resposta.raise_for_status()
        return resposta.json()

    def corredores(self) -> pd.DataFrame:
        """Corredores de ônibus (faixas exclusivas) cadastrados."""
        return pd.DataFrame(self.get("Corredor")).rename(
            columns={"cc": "codigo_corredor", "nc": "nome_corredor"}
        )

    def paradas_do_corredor(self, codigo: int) -> pd.DataFrame:
        dados = pd.DataFrame(self.get("Parada/BuscarParadasPorCorredor", codigoCorredor=codigo))
        if dados.empty:
            return dados
        return dados.rename(
            columns={"cp": "codigo_parada_ov", "np": "nome_parada_ov", "py": "lat", "px": "lon"}
        )

    def posicao_frota(self) -> dict:
        """Retrato da frota em operação neste instante."""
        return self.get("Posicao")


def conectar_olhovivo(cfg: Config) -> OlhoVivo | None:
    """Devolve um cliente autenticado, ou `None` se não houver token no `.env`."""
    token = token_olhovivo()
    if not token:
        log.warning("API_OLHO_VIVO ausente no .env: seguindo apenas com o GTFS")
        return None
    cliente = OlhoVivo(cfg["fontes"]["olhovivo_base_url"], token)
    cliente.autenticar()
    return cliente


def baixar_paradas_corredor(cfg: Config, forcar: bool = False) -> Path | None:
    """Baixa as paradas de todos os corredores e grava um CSV em `data/raw/`."""
    destino = cfg.caminho("raw") / "paradas_corredor.csv"
    if destino.exists() and not forcar:
        log.info("paradas de corredor já existem em %s", destino)
        return destino

    cliente = conectar_olhovivo(cfg)
    if cliente is None:
        return None

    corredores = cliente.corredores()
    pedacos = []
    for linha in corredores.itertuples():
        paradas = cliente.paradas_do_corredor(linha.codigo_corredor)
        if paradas.empty:
            continue
        paradas["codigo_corredor"] = linha.codigo_corredor
        paradas["nome_corredor"] = linha.nome_corredor
        pedacos.append(paradas)

    if not pedacos:
        log.warning("nenhuma parada de corredor devolvida pela API")
        return None

    tabela = pd.concat(pedacos, ignore_index=True)
    tabela.to_csv(destino, index=False, encoding="utf-8")
    log.info("%d paradas de corredor salvas em %s", len(tabela), destino)
    return destino


def baixar_snapshot_frota(cfg: Config) -> Path | None:
    """Guarda um retrato da frota em operação, com carimbo de hora no nome.

    Rodado de tempos em tempos (ver `.github/workflows/coleta.yml`), este é o
    começo de uma base histórica de oferta realizada, que o GTFS não tem.
    """
    cliente = conectar_olhovivo(cfg)
    if cliente is None:
        return None

    dados = cliente.posicao_frota()
    pasta = cfg.caminho("raw") / "olhovivo"
    pasta.mkdir(parents=True, exist_ok=True)
    carimbo = pd.Timestamp.now(tz="America/Sao_Paulo").strftime("%Y%m%dT%H%M")
    destino = pasta / f"posicao_{carimbo}.json"
    destino.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    total = sum(item.get("qv", 0) for item in dados.get("l", []))
    log.info("snapshot com %d veículos salvo em %s", total, destino)
    return destino


def executar(cfg: Config, forcar: bool = False, com_frota: bool = False) -> None:
    """Etapa `extract` do pipeline."""
    baixar_gtfs(cfg, forcar=forcar)
    baixar_paradas_corredor(cfg, forcar=forcar)
    if com_frota:
        baixar_snapshot_frota(cfg)
