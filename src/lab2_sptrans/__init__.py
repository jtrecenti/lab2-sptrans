"""Lab 2 de Prática Avançada em Data Viz: pipeline de dados e modelo da SPTrans.

Fluxo: `extract` -> `transform` -> `features` -> `model` -> `evaluate`.
Cada módulo faz uma coisa só e devolve dado, não efeito colateral escondido.
"""

from .config import carregar_config

__version__ = "0.1.0"
__all__ = ["carregar_config", "__version__"]
