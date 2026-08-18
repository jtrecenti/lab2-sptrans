"""Interface de linha de comando.

Cada etapa do pipeline é um comando, e cada comando lê e escreve arquivo. Isso
é o que separa um notebook de um projeto: dá para rodar uma etapa sozinha,
repetir só o que mudou e chamar tudo de dentro de um agendador ou de um CI.

    uv run lab2 extract      # baixa GTFS e paradas de corredor
    uv run lab2 transform    # monta a tabela analítica
    uv run lab2 train        # ajusta o pipeline e salva o modelo
    uv run lab2 evaluate     # métricas, importância e gráficos
    uv run lab2 predict --entrada exemplo.csv
    uv run lab2 all          # tudo em sequência
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import evaluate, extract, model, transform
from .config import carregar_config
from .features import como_texto


def _configurar_log(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verboso else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab2", description=__doc__)
    parser.add_argument("--config", default=None, help="caminho de um YAML alternativo")
    parser.add_argument("-v", "--verboso", action="store_true")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_extract = sub.add_parser("extract", help="baixa os dados brutos")
    p_extract.add_argument("--forcar", action="store_true", help="rebaixa mesmo se já existir")
    p_extract.add_argument(
        "--com-frota", action="store_true", help="guarda também um retrato da frota agora"
    )

    sub.add_parser("transform", help="monta a tabela analítica")

    p_train = sub.add_parser("train", help="ajusta o pipeline e salva o modelo")
    p_train.add_argument(
        "--algoritmo", default=None, choices=["ridge", "arvore", "floresta", "boosting"]
    )

    p_eval = sub.add_parser("evaluate", help="avalia o modelo e gera relatórios")
    p_eval.add_argument(
        "--algoritmo", default=None, choices=["ridge", "arvore", "floresta", "boosting"]
    )
    p_eval.add_argument("--sem-graficos", action="store_true")

    p_pred = sub.add_parser("predict", help="aplica o modelo salvo a um CSV")
    p_pred.add_argument("--entrada", required=True)
    p_pred.add_argument("--saida", default=None)

    p_all = sub.add_parser("all", help="extract + transform + train + evaluate")
    p_all.add_argument(
        "--algoritmo", default=None, choices=["ridge", "arvore", "floresta", "boosting"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _configurar_log(args.verboso)
    cfg = carregar_config(args.config)
    log = logging.getLogger("lab2")

    if args.comando == "extract":
        extract.executar(cfg, forcar=args.forcar, com_frota=args.com_frota)

    elif args.comando == "transform":
        transform.executar(cfg)

    elif args.comando == "train":
        tabela = transform.carregar_tabela(cfg)
        ajuste = model.ajustar(cfg, tabela, algoritmo=args.algoritmo)
        model.salvar(cfg, ajuste["pipeline"])

    elif args.comando == "evaluate":
        tabela = transform.carregar_tabela(cfg)
        ajuste = model.ajustar(cfg, tabela, algoritmo=args.algoritmo)
        model.salvar(cfg, ajuste["pipeline"])
        resumo = evaluate.executar(cfg, ajuste, com_graficos=not args.sem_graficos)
        _imprimir_resumo(resumo)

    elif args.comando == "predict":
        pipeline = model.carregar(cfg)
        entrada = pd.read_csv(args.entrada, low_memory=False)
        # o CSV de entrada nao carrega tipo: uma coluna categorica so' com
        # numeros volta como int e o OneHotEncoder, treinado em texto, nao a
        # reconhece. Converter aqui e' o mesmo cuidado que uma API precisaria ter.
        for coluna in cfg["features"]["categoricas"]:
            if coluna in entrada.columns:
                entrada[coluna] = como_texto(entrada[coluna])
        entrada["headway_predito_seg"] = pipeline.predict(entrada)
        saida = args.saida or "predicoes.csv"
        entrada.to_csv(saida, index=False)
        log.info("predições salvas em %s", saida)

    elif args.comando == "all":
        extract.executar(cfg)
        tabela = transform.executar(cfg)
        ajuste = model.ajustar(cfg, tabela, algoritmo=args.algoritmo)
        model.salvar(cfg, ajuste["pipeline"])
        _imprimir_resumo(evaluate.executar(cfg, ajuste))

    return 0


def _imprimir_resumo(resumo: dict) -> None:
    print("\n=== resumo ===")
    print(f"algoritmo: {resumo['algoritmo']}")
    print(f"hiperparâmetros: {resumo['melhores_hiperparametros']}")
    for linha in resumo["metricas"]:
        print(
            f"  {linha['estimador']:<9} {linha['particao']:<7} "
            f"MAE {linha['mae_seg']:7.1f}s  R2 {linha['r2']:6.3f}"
        )
    print("variáveis mais importantes:")
    for linha in resumo["top_variaveis"][:5]:
        print(f"  {linha['variavel']:<24} +{linha['piora_mae_seg']:.1f}s de erro se embaralhada")


if __name__ == "__main__":
    sys.exit(main())
