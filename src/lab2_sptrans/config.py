"""Configuração do projeto.

Regra do laboratório: nenhum caminho absoluto e nenhum segredo dentro do código.
Caminhos saem daqui (relativos à raiz do repositório) e segredos saem do `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# raiz do repositório: .../src/lab2_sptrans/config.py -> sobe 3 níveis
RAIZ = Path(__file__).resolve().parents[2]
CONFIG_PADRAO = RAIZ / "conf" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Configuração carregada do YAML, com caminhos já resolvidos."""

    bruto: dict[str, Any]

    def __getitem__(self, chave: str) -> Any:
        return self.bruto[chave]

    @property
    def seed(self) -> int:
        return int(self.bruto["seed"])

    def caminho(self, nome: str) -> Path:
        """Devolve um caminho da seção `caminhos`, já criado, relativo à raiz."""
        destino = RAIZ / self.bruto["caminhos"][nome]
        destino.mkdir(parents=True, exist_ok=True)
        return destino


def carregar_config(caminho: str | Path | None = None) -> Config:
    """Lê `conf/config.yaml` (ou outro arquivo) e o `.env` da raiz."""
    load_dotenv(RAIZ / ".env")
    caminho = Path(caminho) if caminho else CONFIG_PADRAO
    with open(caminho, encoding="utf-8") as arquivo:
        return Config(yaml.safe_load(arquivo))


def token_olhovivo() -> str | None:
    """Token da API Olho Vivo, lido do ambiente.

    Não é obrigatório: sem token o pipeline roda só com o GTFS e as colunas
    de corredor ficam faltantes, o que é tratado no pré-processamento.
    """
    return os.getenv("API_OLHO_VIVO") or os.getenv("OLHOVIVO_TOKEN")
