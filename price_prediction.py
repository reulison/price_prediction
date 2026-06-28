#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Price Prediction Model for Airlines Flights
===========================================
Este script treina modelos de regressão para prever preços de passagens aéreas
utilizando Random Forest e Decision Tree.

Dataset esperado: dataset/airlines_flights_data.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Suprimir warnings para uma saída mais limpa
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")


def load_data():
    """
    Carrega o dataset do diretório 'dataset'.
    
    Returns:
        pd.DataFrame: DataFrame com os dados dos voos
    """
    # Caminho relativo ao script
    script_dir = Path(__file__).parent
    data_path = script_dir / "dataset" / "airlines_flights_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {data_path}. "
            f"Certifique-se de que o arquivo 'airlines_flights_data.csv' "
            f"esteja no diretório 'dataset/'"
        )
    
    print(f"📂 Carregando dados de: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas")
    return df


def explore_data(df):
    """
    Realiza análise exploratória inicial dos dados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos voos
    """
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*60)
    
    # Amostra dos dados
    print("\n📊 Primeiras linhas do dataset:")
    print(df.head())
    
    # Informações gerais
    print("\n📋 Informações do dataset:")
    print(df.info())
    
    # Valores nulos
    print("\n🔍 Valores nulos por coluna:")
    print(df.isnull().sum())
    
    # Duplicatas
    print(f"\n🔁 Registros duplicados: {df.duplicated().sum()}")
    
    # Estatísticas descritivas
    print("\n📈 Estatísticas descritivas:")
    print(df.describe())


def preprocess_data(df):
    """
    Pré-processa os dados: amostragem, conversão de datas e remoção de colunas.
    
    Args:
        df (pd.DataFrame): DataFrame original
    
    Returns:
        pd.DataFrame: DataFrame pré-processado
    """
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("="*60)
    
    # Amostragem para reduzir tempo de processamento
    df = df.sample(n=15000, random_state=42)
    print(f"🎯 Amostra reduzida para: {df.shape[0]} linhas")
    
    # Remover coluna 'index' se existir
    if 'index' in df.columns:
        df = df.drop(['index'], axis=1)
        print("🗑️ Coluna 'index' removida")
    
    # Converter colunas de data/hora
    df['departure_time'] = pd.to_datetime(df['departure_time'], errors='coerce')
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
    print("🕒 Colunas de data/hora convertidas")
    
    return df


def visualize_data(df):
    """
    Gera visualizações dos dados e salva como imagens.
    
    Args:
        df (pd.DataFrame): DataFrame pré-processado
    """
    print("\n" + "="*60)
    print("GERANDO VISUALIZAÇÕES")
    print("="*60)
    
    # Criar diretório para salvar gráficos
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Gráfico 1: Contagem de voos por companhia
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='airline', order=df['airline'].value_counts().index)
    plt.title('Flight Count by Airline', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'airline_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico salvo: plots/airline_counts.png")
    
    # Gráfico 2: Distribuição da duração dos voos
    plt.figure(figsize=(8, 5))
    sns.histplot(df['duration'], kde=True, bins=20)
    plt.title('Distribution of Flight Duration', fontsize=14)
    plt.xlabel('Duration (minutes)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico salvo: plots/duration_distribution.png")
    
    # Gráfico 3: Matriz de correlação
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap of Numeric Variables', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico salvo: plots/correlation_heatmap.png")


def prepare_features(df):
    """
    Prepara features e target para modelagem.
    
    Args:
        df (pd.DataFrame): DataFrame pré-processado
    
    Returns:
        tuple: (X, y, categorical_cols)
    """
    print("\n" + "="*60)
    print("PREPARANDO FEATURES")
    print("="*60)
    
    target = 'price'
    
    # Definir features (X) e target (y)
    X = df.drop(columns=['flight', 'stops', 'departure_time', 'arrival_time', target])
    y = df[target]
    
    # Identificar colunas categóricas
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    
    print(f"🎯 Target: {target}")
    print(f"📊 Features: {X.shape[1]} colunas")
    print(f"🏷️ Colunas categóricas: {categorical_cols}")
    
    return X, y, categorical_cols


def create_pipelines(preprocessor):
    """
    Cria pipelines de modelos.
    
    Args:
        preprocessor: ColumnTransformer para pré-processamento
    
    Returns:
        dict: Dicionário com os pipelines
    """
    pipelines = {
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        'decision_tree': Pipeline([
            ('preprocessor', preprocessor),
            ('model', DecisionTreeRegressor(random_state=42))
        ])
    }
    return pipelines


def train_and_evaluate(X, y, pipelines):
    """
    Treina e avalia os modelos.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        pipelines (dict): Dicionário com pipelines dos modelos
    
    Returns:
        tuple: (results, X_test, y_test)
    """
    print("\n" + "="*60)
    print("TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print("="*60)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Dados de treino: {X_train.shape[0]} amostras")
    print(f"📊 Dados de teste: {X_test.shape[0]} amostras")
    
    results = {}
    
    for name, pipeline in pipelines.items():
        print(f"\n🚀 Treinando {name.replace('_', ' ').title()}...")
        
        # Treinar
        pipeline.fit(X_train, y_train)
        
        # Predizer
        y_pred = pipeline.predict(X_test)
        
        # Métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'pipeline': pipeline,
            'r2': r2,
            'rmse': rmse,
            'y_pred': y_pred,
            'y_test': y_test  # Armazenar os valores reais do teste
        }
        
        print(f"   ✅ R²: {r2:.4f}")
        print(f"   ✅ RMSE: {rmse:.2f}")
    
    return results, X_test, y_test


def cross_validate(X, y, pipelines):
    """
    Realiza validação cruzada nos modelos.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        pipelines (dict): Dicionário com pipelines dos modelos
    """
    print("\n" + "="*60)
    print("VALIDAÇÃO CRUZADA")
    print("="*60)
    
    for name, pipeline in pipelines.items():
        print(f"\n🔄 Validando {name.replace('_', ' ').title()}...")
        
        scores = cross_val_score(
            pipeline, X, y,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        print(f"   📈 Scores R²: {scores}")
        print(f"   📊 Média R²: {np.mean(scores):.4f}")
        print(f"   📊 Desvio padrão: {np.std(scores):.4f}")


def save_results(results, X_test, y_test):
    """
    Salva os resultados e predições em arquivos CSV.
    
    Args:
        results (dict): Resultados dos modelos
        X_test (pd.DataFrame): Features de teste
        y_test (pd.Series): Target de teste
    """
    print("\n" + "="*60)
    print("SALVANDO RESULTADOS")
    print("="*60)
    
    # Criar diretório para resultados
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Salvar métricas
    metrics_df = pd.DataFrame([
        {
            'model': name,
            'r2_score': data['r2'],
            'rmse': data['rmse']
        }
        for name, data in results.items()
    ])
    metrics_df.to_csv(results_dir / 'model_metrics.csv', index=False)
    print("✅ Métricas salvas: results/model_metrics.csv")
    
    # Salvar predições para cada modelo com os valores reais
    for name, data in results.items():
        # Verificar se os arrays têm o mesmo tamanho
        if len(data['y_test']) != len(data['y_pred']):
            print(f"⚠️ Tamanhos diferentes para {name}: y_test={len(data['y_test'])}, y_pred={len(data['y_pred'])}")
            continue
            
        pred_df = pd.DataFrame({
            'actual_price': data['y_test'].values if hasattr(data['y_test'], 'values') else data['y_test'],
            'predicted_price': data['y_pred']
        })
        pred_df.to_csv(results_dir / f'predictions_{name}.csv', index=False)
        print(f"✅ Predições salvas: results/predictions_{name}.csv")


def main():
    """
    Função principal que executa todo o pipeline.
    """
    print("\n" + "="*60)
    print("✈️  AIRLINE PRICE PREDICTION MODEL")
    print("="*60)
    
    try:
        # 1. Carregar dados
        df = load_data()
        
        # 2. Análise exploratória
        explore_data(df)
        
        # 3. Pré-processamento
        df = preprocess_data(df)
        
        # 4. Visualizações
        visualize_data(df)
        
        # 5. Preparar features
        X, y, categorical_cols = prepare_features(df)
        
        # 6. Criar pré-processador
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # 7. Criar pipelines
        pipelines = create_pipelines(preprocessor)
        
        # 8. Treinar e avaliar
        results, X_test, y_test = train_and_evaluate(X, y, pipelines)
        
        # 9. Validar cruzadamente
        cross_validate(X, y, pipelines)
        
        # 10. Salvar resultados
        save_results(results, X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETO COM SUCESSO!")
        print("="*60)
        print("\n📁 Estrutura de saída:")
        print("   📂 plots/ - Gráficos gerados")
        print("   📂 results/ - Métricas e predições")
        print("   📂 dataset/ - Dados de entrada (deve existir)")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()