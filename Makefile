# Atalhos do pipeline. `make all` reconstroi tudo do zero.
# No Windows sem make, rode direto o comando da direita.

.PHONY: setup extract transform train evaluate all test lint report slides limpar

setup:            ## instala as dependencias no ambiente do projeto
	uv sync --all-groups

extract:          ## baixa GTFS e paradas de corredor
	uv run lab2 extract

transform:        ## monta a tabela analitica
	uv run lab2 transform

train:            ## ajusta o pipeline e salva models/modelo.joblib
	uv run lab2 train

evaluate:         ## metricas, importancia e graficos em reports/
	uv run lab2 evaluate

all:              ## extract + transform + train + evaluate
	uv run lab2 all

test:             ## roda a suite de testes
	uv run pytest

lint:             ## checa estilo
	uv run ruff check .

report:           ## renderiza o relatorio Quarto (precisa do extra notebooks)
	uv sync --all-groups --extra notebooks
	PYTHONUTF8=1 uv run quarto render notebooks/relatorio.qmd

slides:           ## renderiza os slides (precisa do extra notebooks)
	uv sync --all-groups --extra notebooks
	PYTHONUTF8=1 uv run quarto render slides/tidymodels-para-sklearn.qmd

limpar:           ## apaga artefatos derivados (dado bruto fica)
	rm -rf data/processed data/interim models reports
