"""Avaliação: métricas, importância de variáveis e gráficos de diagnóstico.

Uma métrica sozinha não diz nada. O relatório compara sempre com o baseline e
olha o erro por faixa da variável, porque um erro médio bom costuma esconder um
erro grande justamente onde a decisão acontece.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import Config

log = logging.getLogger(__name__)


def metricas(y_verdadeiro, y_predito) -> dict[str, float]:
    """MAE, RMSE, MAPE e R2 num dicionário."""
    y_verdadeiro = np.asarray(y_verdadeiro, dtype=float)
    y_predito = np.asarray(y_predito, dtype=float)
    return {
        "mae_seg": float(mean_absolute_error(y_verdadeiro, y_predito)),
        "rmse_seg": float(np.sqrt(mean_squared_error(y_verdadeiro, y_predito))),
        "mape": float(np.mean(np.abs((y_verdadeiro - y_predito) / y_verdadeiro))),
        "r2": float(r2_score(y_verdadeiro, y_predito)),
    }


def comparar(ajuste: dict) -> pd.DataFrame:
    """Tabela de métricas: baseline x modelo, em treino e em teste."""
    linhas = []
    for nome, estimador in (("baseline", ajuste["baseline"]), ("modelo", ajuste["pipeline"])):
        for particao in ("treino", "teste"):
            X, y = ajuste[f"X_{particao}"], ajuste[f"y_{particao}"]
            linhas.append(
                {"estimador": nome, "particao": particao, **metricas(y, estimador.predict(X))}
            )
    return pd.DataFrame(linhas)


def importancia(ajuste: dict, seed: int, n_repeticoes: int = 5) -> pd.DataFrame:
    """Importância por permutação, medida no teste.

    Diferente da importância interna da floresta, esta responde à pergunta que
    interessa: se eu embaralhar esta coluna, quanto o erro piora nos dados que
    o modelo nunca viu?
    """
    resultado = permutation_importance(
        ajuste["pipeline"],
        ajuste["X_teste"],
        ajuste["y_teste"],
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeticoes,
        random_state=seed,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "variavel": ajuste["X_teste"].columns,
                "piora_mae_seg": resultado.importances_mean,
                "desvio": resultado.importances_std,
            }
        )
        .sort_values("piora_mae_seg", ascending=False)
        .reset_index(drop=True)
    )


def residuos(ajuste: dict) -> pd.DataFrame:
    """Base de resíduos do teste, para os gráficos e para a leitura de negócio."""
    X_teste = ajuste["X_teste"].copy()
    y_teste = ajuste["y_teste"]
    predito = ajuste["pipeline"].predict(X_teste)
    return X_teste.assign(
        headway_observado=y_teste.to_numpy(),
        headway_predito=predito,
        residuo=y_teste.to_numpy() - predito,
    )


def gerar_graficos(cfg: Config, base_residuos: pd.DataFrame) -> list[Path]:
    """Dois diagnósticos: observado x predito e erro por período do dia."""
    from plotnine import (
        aes,
        geom_abline,
        geom_boxplot,
        geom_hline,
        geom_point,
        ggplot,
        labs,
        theme_minimal,
    )

    pasta = cfg.caminho("reports")
    salvos: list[Path] = []

    grafico_ajuste = (
        ggplot(base_residuos, aes("headway_observado", "headway_predito"))
        + geom_point(alpha=0.15, size=0.8)
        + geom_abline(intercept=0, slope=1, linetype="dashed")
        + labs(
            title="Intervalo entre ônibus: observado x predito",
            subtitle="Cada ponto é uma linha, num sentido, numa faixa horária (dados de teste)",
            x="Observado (segundos)",
            y="Predito (segundos)",
        )
        + theme_minimal()
    )
    caminho = pasta / "observado_vs_predito.png"
    grafico_ajuste.save(caminho, width=7, height=5, dpi=150, verbose=False)
    salvos.append(caminho)

    grafico_residuo = (
        ggplot(base_residuos, aes("periodo_dia", "residuo"))
        + geom_boxplot()
        + geom_hline(yintercept=0, linetype="dashed")
        + labs(
            title="Onde o modelo erra",
            subtitle="Resíduo positivo: a oferta real é mais espaçada do que o modelo previa",
            x="Período do dia",
            y="Resíduo (segundos)",
        )
        + theme_minimal()
    )
    caminho = pasta / "residuo_por_periodo.png"
    grafico_residuo.save(caminho, width=7, height=5, dpi=150, verbose=False)
    salvos.append(caminho)

    log.info("gráficos salvos em %s", pasta)
    return salvos


def executar(cfg: Config, ajuste: dict, com_graficos: bool = True) -> dict:
    """Etapa `evaluate`: grava métricas, importância e gráficos em `reports/`."""
    pasta = cfg.caminho("reports")

    tabela_metricas = comparar(ajuste)
    tabela_metricas.to_csv(pasta / "metricas.csv", index=False)

    tabela_importancia = importancia(ajuste, cfg.seed)
    tabela_importancia.to_csv(pasta / "importancia.csv", index=False)

    base_residuos = residuos(ajuste)
    base_residuos.to_csv(pasta / "residuos_teste.csv.gz", index=False, compression="gzip")

    if com_graficos:
        gerar_graficos(cfg, base_residuos)

    resumo = {
        "algoritmo": ajuste["algoritmo"],
        "melhores_hiperparametros": {
            chave: (valor if isinstance(valor, (int, float, str, type(None))) else str(valor))
            for chave, valor in ajuste["busca"].best_params_.items()
        },
        "metricas": tabela_metricas.to_dict(orient="records"),
        "top_variaveis": tabela_importancia.head(10).to_dict(orient="records"),
    }
    (pasta / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("avaliação salva em %s", pasta)
    return resumo
