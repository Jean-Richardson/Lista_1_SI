

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # ---------------------------------------------------------
    # 1. Carregamento dos Dados
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(base_dir, 'Folds5x2_pp.csv'),
        os.path.join(base_dir, 'Folds5x2_pp.xlsx'),
        os.path.join(base_dir, 'Folds5x2_pp.ods'),
    ]

    data_path = next((c for c in candidatos if os.path.exists(c)), candidatos[0])
    print(f"Carregando dataset de: {data_path}...")

    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.ods'):
            df = pd.read_excel(data_path, engine='odf')
        else:
            raise ValueError(f"Formato não suportado: {data_path}")
    except FileNotFoundError:
        print("\n[Aviso] Arquivo não encontrado no caminho indicado.")
        print("Certifique-se de colocar o dataset CCPP na pasta do script com nome Folds5x2_pp.csv, .xlsx ou .ods.")
        print("Buscando criar uma simulação para fins de demonstração local se necessário...")
        return
    except Exception as e:
        print(f"\n[Aviso] Não foi possível ler o arquivo do dataset: {e}")
        print("Verifique se o arquivo está em um formato compatível e se as colunas estão nomeadas corretamente.")
        return

    print("\n--- Primeiras linhas do dataset ---")
    print(df.head())
    
    print("\n--- Informações do Dataset ---")
    print(df.info())

    # ---------------------------------------------------------
    # 2. Tratamento de Valores Ausentes
    # ---------------------------------------------------------
    missing_vals = df.isnull().sum()
    print("\n--- Valores Ausentes por Coluna ---")
    print(missing_vals)
    if missing_vals.sum() > 0:
        df = df.dropna()
        print("Valores ausentes removidos com sucesso.")

    # ---------------------------------------------------------
    # 3. Análise Exploratória e Correlações (Tarefas 1 e 2)
    # ---------------------------------------------------------
    features = ['AT', 'V', 'AP', 'RH']
    target = 'PE'

    # Tarefa 1: Correlação com a Alvo (PE)
    corr_target = df[features].corrwith(df[target]).sort_values(ascending=False)
    print("\n--- Tarefa 1: Correlação das Variáveis com a Alvo (PE) ---")
    print(corr_target)

    # Tarefa 2: Multicolinearidade (Matriz de Correlação entre Variáveis Independentes)
    corr_matrix = df.corr()
    print("\n--- Tarefa 2: Matriz de Correlação Completa ---")
    print(corr_matrix)

    # Plotagem do Heatmap de Correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação - Dataset CCPP")
    plt.tight_layout()
    plt.savefig('matriz_correlacao.png')
    plt.close()
    print("Gráfico 'matriz_correlacao.png' salvo.")

    # ---------------------------------------------------------
    # 4. Divisão Treino / Teste (80% / 20%) e Padronização (Tarefa 3)
    # ---------------------------------------------------------
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\n--- Tarefa 3: Divisão dos Dados ---")
    print(f"Treino: {X_train.shape[0]} amostras (80%) | Teste: {X_test.shape[0]} amostras (20%)")

    # Padronização (StandardScaler) ajustado APENAS no treino para evitar Data Leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 5. Treinamento da Regressão Linear Múltipla
    # ---------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("\n--- Parâmetros do Modelo Treinado ---")
    print(f"Intercepto (theta_0): {model.intercept_:.4f}")
    for coef, feat in zip(model.coef_, features):
        print(f"Coeficiente ({feat}): {coef:.4f}")

    # ---------------------------------------------------------
    # 6. Avaliação do Modelo no Conjunto de Teste (Tarefa 4)
    # ---------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Tarefa 4: Métricas de Avaliação no Teste ---")
    print(f"R² (Coeficiente de Determinação): {r2:.4f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f} MW")
    print(f"MAE (Erro Absoluto Médio): {mae:.4f} MW")

    # Gráficos de Avaliação (Reais vs Previstos e Resíduos)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Real vs Previsto
    axes[0].scatter(y_test, y_pred, alpha=0.3, color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel("Valores Reais (PE)")
    axes[0].set_ylabel("Valores Previstos (PE)")
    axes[0].set_title(f"Valores Reais vs Previstos (R² = {r2:.3f})")

    # Análise de Resíduos
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, color='purple')
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel("Valores Previstos (PE)")
    axes[1].set_ylabel("Resíduos (Real - Previsto)")
    axes[1].set_title("Análise de Resíduos")

    plt.tight_layout()
    plt.savefig('avaliacao_modelo.png')
    plt.close()
    print("Gráfico 'avaliacao_modelo.png' salvo com sucesso.")

if __name__ == '__main__':
    main()
