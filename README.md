# Flight Price Prediction

Este projeto implementa uma aplicação em Python para prever preços de passagens aéreas usando modelos de regressão supervisionada.

## O que a aplicação faz

O script em [price_prediction.py](price_prediction.py) realiza o seguinte fluxo:

- Carrega o dataset localizado em [dataset/airlines_flights_data.csv](dataset/airlines_flights_data.csv)
- Realiza uma análise exploratória inicial dos dados
- Pré-processa as colunas e reduz a amostra para acelerar o treinamento
- Gera visualizações em [plots/](plots/)
- Treina e avalia dois modelos de regressão:
  - Random Forest Regressor
  - Decision Tree Regressor
- Salva métricas e predições em [results/](results/)

## Requisitos

Antes de executar, instale as dependências:

```bash
pip install -r requirements.txt
```

## Como executar

Na raiz do projeto, execute:

```bash
python price_prediction.py
```

O script irá:

1. Carregar os dados
2. Exibir informações e estatísticas do dataset
3. Gerar gráficos
4. Treinar os modelos
5. Salvar os resultados em [results/](results/)

## Estrutura do projeto

- [price_prediction.py](price_prediction.py): pipeline completo de carregamento, pré-processamento, treinamento e avaliação
- [dataset/](dataset/): arquivos de entrada com os dados utilizados pelo modelo
- [plots/](plots/): gráficos gerados pela análise exploratória
- [results/](results/): métricas e arquivos de predições gerados pelos modelos
- [requirements.txt](requirements.txt): dependências do projeto

## Arquivos de saída

Após a execução, os seguintes arquivos são criados:

- [plots/airline_counts.png](plots/airline_counts.png)
- [plots/duration_distribution.png](plots/duration_distribution.png)
- [plots/correlation_heatmap.png](plots/correlation_heatmap.png)
- [results/model_metrics.csv](results/model_metrics.csv)
- [results/predictions_random_forest.csv](results/predictions_random_forest.csv)
- [results/predictions_decision_tree.csv](results/predictions_decision_tree.csv)

## Observações

- O script utiliza uma amostra de 15.000 linhas para reduzir o tempo de processamento.
- As métricas principais avaliadas são R² e RMSE.
- A validação cruzada também é executada para comparar a estabilidade dos modelos.
- [Artigo Completo: Previsão de Preços de Passagens Aéreas](https://www.reulison.com.br/blog/data-analytics/flight-price-prediction)
