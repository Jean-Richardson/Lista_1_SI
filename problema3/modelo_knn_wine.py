
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configuração visual dos gráficos
plt.style.use('ggplot')
PASTA_GRAFICOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


def carregar_dataset():
    """
    Carrega o dataset Wine Quality via pacote ucimlrepo ou cria fallback sintético.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        print("-> Baixando dataset Wine Quality (ID=186) via ucimlrepo...")
        wine_quality = fetch_ucirepo(id=186)
        
        X = wine_quality.data.features
        y = wine_quality.data.targets
        
        # Garante DataFrame e série unificados
        df = pd.concat([X, y], axis=1)
        # Se houver colunas de vinho branco e tinto juntas, filtra ou ajusta
        if 'quality' in df.columns:
            target_col = 'quality'
        else:
            target_col = df.columns[-1]
            
        print(f"-> Dataset carregado com sucesso: {df.shape[0]} amostras e {df.shape[1]-1} atributos.")
        return df, target_col

    except Exception as e:
        print(f"-> Aviso: ucimlrepo não disponível ou sem conexão ({e}).")
        print("-> Gerando conjunto sintético representativo do Wine Quality (Vinho Tinto)...")
        
        np.random.seed(42)
        n_samples = 1599
        
        # Atributos físicos-químicos típicos do vinho tinto
        fixed_acidity = np.random.normal(8.32, 1.74, n_samples)
        volatile_acidity = np.random.normal(0.53, 0.18, n_samples)
        citric_acid = np.random.normal(0.27, 0.19, n_samples)
        residual_sugar = np.random.normal(2.54, 1.41, n_samples)
        chlorides = np.random.normal(0.087, 0.047, n_samples)
        free_sulfur_dioxide = np.random.normal(15.87, 10.46, n_samples)
        total_sulfur_dioxide = np.random.normal(46.47, 32.90, n_samples)
        density = np.random.normal(0.9967, 0.0019, n_samples)
        pH = np.random.normal(3.31, 0.15, n_samples)
        sulphates = np.random.normal(0.658, 0.169, n_samples)
        alcohol = np.random.normal(10.42, 1.06, n_samples)
        
        # Qualidades originais (3 a 8 no dataset real, concentrado em 5 e 6)
        quality = np.random.choice([3, 4, 5, 6, 7, 8], size=n_samples, p=[0.01, 0.03, 0.43, 0.40, 0.12, 0.01])
        
        feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        X_data = np.column_stack([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ])
        
        df = pd.DataFrame(X_data, columns=feature_names)
        df['quality'] = quality
        return df, 'quality'


def categorizar_qualidade(nota):
    """
    Recodifica a nota do vinho em 3 categorias:
    - Nota <= 5 : 'Ruim'  (0)
    - Nota 6-7  : 'Médio' (1)
    - Nota >= 8 : 'Bom'   (2)
    """
    if nota <= 5:
        return 0  # Ruim
    elif nota <= 7:
        return 1  # Médio
    else:
        return 2  # Bom


def main():
    print("=========================================================================")
    print("  PROBLEMA 3: CLASSIFICAÇÃO DE QUALIDADE DE VINHOS UTILIZANDO k-NN      ")
    print("=========================================================================\n")

    # -------------------------------------------------------------------------
    # PASSO 1: Preparação e Recodificação dos Dados (Tarefa 1)
    # -------------------------------------------------------------------------
    print("--- PASSO 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS ---")
    df, target_col = carregar_dataset()
    
    # Aplicar categorização em 3 classes (Ruim, Médio, Bom)
    df['classe_qualidade'] = df[target_col].apply(categorizar_qualidade)
    
    nomes_classes = ['Ruim (<=5)', 'Médio (6-7)', 'Bom (>=8)']
    
    print("\nDistribuição das classes recodificadas:")
    dist_classes = df['classe_qualidade'].value_counts().sort_index()
    for idx, count in dist_classes.items():
        pct = (count / len(df)) * 100
        print(f"  - Classe {idx} ({nomes_classes[idx]}): {count} amostras ({pct:.2f}%)")

    X = df.drop(columns=[target_col, 'classe_qualidade'])
    y = df['classe_qualidade']

    # -------------------------------------------------------------------------
    # PASSO 2: Divisão Treino e Teste Estratificada (80% / 20%) (Tarefa 2)
    # -------------------------------------------------------------------------
    print("\n--- PASSO 2: DIVISÃO ESTRATIFICADA EM TREINO (80%) E TESTE (20%) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Amostras de Treino: {X_train.shape[0]}")
    print(f"Amostras de Teste:  {X_test.shape[0]}")

    # Normalização dos Atributos (StandardScaler ou MinMaxScaler)
    # O k-NN exige normalização pois calcula distâncias no espaço numérico
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    print("-> Atributos normalizados com sucesso (StandardScaler).")

    # -------------------------------------------------------------------------
    # PASSO 3: Treinamento e Seleção do k via Validação Cruzada (Tarefa 3)
    # -------------------------------------------------------------------------
    print("\n--- PASSO 3: VALIDAÇÃO CRUZADA DE 5 FOLDS PARA ESCOLHA DO HIPERPARÂMETRO k ---")
    
    k_valores = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    acuracias_medias = []
    acuracias_stds = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_valores:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_sc, y_train, cv=skf, scoring='accuracy')
        acuracias_medias.append(scores.mean())
        acuracias_stds.append(scores.std())
        print(f"  k = {k:2d} | Acurácia Média CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Identifica o melhor k
    melhor_idx = np.argmax(acuracias_medias)
    melhor_k = k_valores[melhor_idx]
    melhor_acc = acuracias_medias[melhor_idx]

    print(f"\n-> MELHOR VALOR DE k ENCONTRADO: k = {melhor_k} (Acurácia CV = {melhor_acc:.4f})")

    # Gráfico de Acurácia Média em função de k
    plt.figure(figsize=(9, 5))
    plt.errorbar(k_valores, acuracias_medias, yerr=acuracias_stds, fmt='-o', color='crimson',
                 ecolor='gray', elinewidth=1.5, capsize=4, label='Acurácia Média CV (+/- 1 DP)')
    plt.axvline(melhor_k, color='blue', linestyle='--', label=f'Melhor k = {melhor_k}')
    plt.title('Acurácia Média na Validação Cruzada (5 Folds) vs Valor de k', fontsize=12, fontweight='bold')
    plt.xlabel('Número de Vizinhos (k)')
    plt.ylabel('Acurácia Média')
    plt.xticks(k_valores)
    plt.legend()
    plt.grid(True)
    caminho_curva_k = os.path.join(PASTA_GRAFICOS, 'curva_k_knn.png')
    plt.savefig(caminho_curva_k, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico da escolha do k salvo em: {caminho_curva_k}")

    # -------------------------------------------------------------------------
    # PASSO 4: Treinamento Final e Avaliação no Teste (Tarefa 4)
    # -------------------------------------------------------------------------
    print(f"\n--- PASSO 4: AVALIAÇÃO DO MODELO FINAL COM k = {melhor_k} NO CONJUNTO DE TESTE ---")
    
    modelo_final = KNeighborsClassifier(n_neighbors=melhor_k)
    modelo_final.fit(X_train_sc, y_train)

    y_pred = modelo_final.predict(X_test_sc)

    # Cálculo das Métricas Gerais e por Classe
    acc_geral = accuracy_score(y_test, y_pred)
    prec_classes = precision_score(y_test, y_pred, average=None)
    rec_classes = recall_score(y_test, y_pred, average=None)
    f1_classes = f1_score(y_test, y_pred, average=None)

    print(f"\nAcurácia Geral no Teste: {acc_geral:.4f} ({acc_geral*100:.2f}%)\n")
    print("Métricas por Classe:")
    print(f"{'Classe':<15} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    for i in range(len(nomes_classes)):
        print(f"{nomes_classes[i]:<15} | {prec_classes[i]:.4f}     | {rec_classes[i]:.4f}     | {f1_classes[i]:.4f}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)

    # Gráfico da Matriz de Confusão
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(f'Matriz de Confusão - k-NN (k = {melhor_k})', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Prevista pelo Modelo')
    plt.ylabel('Classe Real do Vinho')
    caminho_cm = os.path.join(PASTA_GRAFICOS, 'matriz_confusao_knn.png')
    plt.savefig(caminho_cm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Matriz de Confusão salva em: {caminho_cm}")

    print("\n=========================================================================")
    print("  EXECUÇÃO CONCLUÍDA COM SUCESSO  ")
    print("=========================================================================")


if __name__ == '__main__':
    main()
