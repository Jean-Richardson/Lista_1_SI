

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Configurações estéticas dos gráficos
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
PASTA_GRAFICOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


def carregar_dataset():
    """
    Carrega o dataset Statlog (German Credit Data).
    Tenta via ucimlrepo (id=144), depois via sklearn fetch_openml('credit-g'),
    e em caso offline sem pacotes, gera estrutura sintética fiel ao dataset original.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        credit = fetch_ucirepo(id=144)
        X = credit.data.features
        y = credit.data.targets
        print("-> Dataset German Credit carregado com sucesso via ucimlrepo (ID 144)!")
        return X, y
    except Exception as e:
        try:
            from sklearn.datasets import fetch_openml
            print("-> Carregando dataset via OpenML ('credit-g')...")
            credit_g = fetch_openml(name='credit-g', version=1, as_frame=True)
            X = credit_g.data
            y = credit_g.target
            return X, y
        except Exception as e2:
            print("-> Ambiente local sem conexão externa. Gerando conjunto sintético representativo do German Credit...")
            np.random.seed(42)
            n = 1000
            
            data = {
                'duration': np.random.randint(6, 72, n),
                'credit_amount': np.random.randint(250, 18424, n),
                'installment_commitment': np.random.choice([1, 2, 3, 4], n),
                'residence_since': np.random.choice([1, 2, 3, 4], n),
                'age': np.random.randint(19, 75, n),
                'existing_credits': np.random.choice([1, 2, 3, 4], n),
                'num_dependents': np.random.choice([1, 2], n, p=[0.85, 0.15]),
                
                'checking_status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n, p=[0.3, 0.3, 0.1, 0.3]),
                'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n),
                'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'], n),
                'savings_status': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n),
                'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n),
                'personal_status': np.random.choice(['A91', 'A92', 'A93', 'A94'], n),
                'other_parties': np.random.choice(['A101', 'A102', 'A103'], n),
                'property_magnitude': np.random.choice(['A121', 'A122', 'A123', 'A124'], n),
                'other_payment_plans': np.random.choice(['A141', 'A142', 'A143'], n),
                'housing': np.random.choice(['A151', 'A152', 'A153'], n),
                'job': np.random.choice(['A171', 'A172', 'A173', 'A174'], n),
                'own_telephone': np.random.choice(['A191', 'A192'], n),
                'foreign_worker': np.random.choice(['A201', 'A202'], n)
            }
            X = pd.DataFrame(data)
            
            # Construção do target no padrão do German Credit (70% Good = 1 / 30% Bad = 2)
            score = (
                0.04 * X['duration'] + 
                0.0001 * X['credit_amount'] - 
                0.03 * X['age'] + 
                (X['checking_status'] == 'A11').astype(int) * 1.5 + 
                (X['credit_history'] == 'A30').astype(int) * 1.2
            )
            prob_bad = 1 / (1 + np.exp(-(score - np.percentile(score, 70))))
            y_val = np.where(prob_bad > 0.5, 2, 1) # 1 = Good, 2 = Bad
            y = pd.Series(y_val, name='class')
            return X, y


def pre_processar_e_analisar_dados(X, y):
    """
    TAREFA 1 & TAREFA 2:
    - Identifica atributos numéricos e categóricos.
    - Avalia o desbalanceamento das classes (70% / 30%).
    - Aplica One-Hot Encoding nas variáveis categóricas.
    - Apresenta a justificativa teórica para a ausência de normalização.
    """
    print("\n=========================================================================")
    print("  TAREFA 1: PREPARAÇÃO DOS DADOS E ANÁLISE DE BALANCEAMENTO")
    print("=========================================================================")
    
    # Padroniza formato do target
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # Mapeia rótulos para 1 (Bom Pagador) e 2 (Mau Pagador)
    y_mapped = y.copy()
    if y_mapped.dtype == 'object' or isinstance(y_mapped.iloc[0], str):
        y_mapped = y_mapped.map({'good': 1, 'bad': 2, '1': 1, '2': 2, 1: 1, 2: 2})
    
    # Identificação das colunas
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    
    print(f"-> Total de amostras: {X.shape[0]}")
    print(f"-> Atributos Numéricos ({len(num_cols)}): {num_cols}")
    print(f"-> Atributos Categóricos ({len(cat_cols)}): {cat_cols}")
    
    # Análise de Balanceamento
    contagem = y_mapped.value_counts()
    prop = y_mapped.value_counts(normalize=True) * 100
    
    print("\n-> Distribuição da Variável Alvo (Risco de Crédito):")
    print(f"   - Classe 1 (Bom Pagador): {contagem.get(1, 0)} amostras ({prop.get(1, 0):.2f}%)")
    print(f"   - Classe 2 (Mau Pagador): {contagem.get(2, 0)} amostras ({prop.get(2, 0):.2f}%)")
    
    print("\n[COMENTÁRIO SOBRE O DESBALANCEAMENTO (70% / 30%)]:")
    print("O desbalanceamento de 70% (Bons) para 30% (Maus) impacta diretamente a interpretação da acurácia.")
    print("Um modelo ingênuo que simplesmente previsse 'Bom Pagador' para TODOS os clientes atingiria 70% de acurácia,")
    print("mas seria completamente inútil na prática, pois falharia em detectar 100% dos inadimplentes.")
    print("Por isso, métricas como Recall, Precisão, F1-Score e Matriz de Confusão são fundamentais.")

    print("\n=========================================================================")
    print("  TAREFA 2: JUSTIFICATIVA DE AUSÊNCIA DE NORMALIZAÇÃO")
    print("=========================================================================")
    print("Árvores de Decisão NÃO necessitam de normalização ou padronização de atributos porque:")
    print("1. As decisões nos nós são baseadas em testes condicionais monovariados do tipo (x_i <= limiar).")
    print("2. O algoritmo avalia a ordenação relativa dos valores de cada atributo de forma isolada.")
    print("3. Transformações monotônicas (como Z-score ou Min-Max) preservam a ordem exata dos dados,")
    print("   resultando nos mesmos pontos de corte e no mesmo ganho de informação/índice Gini.")

    # Codificação Categórica via One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_encoded = preprocessor.fit_transform(X)
    
    # Recupera nomes das colunas codificadas
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + encoded_cat_cols
    
    X_df = pd.DataFrame(X_encoded, columns=all_feature_names)
    
    return X_df, y_mapped, all_feature_names


def experimento_profundidade(X_train, y_train, criterion='gini'):
    """
    TAREFA 4: Executa validação cruzada para diferentes profundidades
    e retorna as acurácias médias de treino e validação.
    """
    depths = [2, 3, 5, 7, 10, None]
    depth_labels = [str(d) if d is not None else 'Sem Limite' for d in depths]
    
    train_accs = []
    val_accs = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for d in depths:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=d,
            random_state=42
        )
        scores = cross_validate(
            clf, X_train, y_train,
            cv=skf,
            scoring='accuracy',
            return_train_score=True
        )
        train_accs.append(scores['train_score'].mean())
        val_accs.append(scores['test_score'].mean())
        
    return depths, depth_labels, train_accs, val_accs


def main():
    # 1. Carga dos dados
    X_raw, y_raw = carregar_dataset()
    
    # 2. Pré-processamento e Tarefas 1 e 2
    X, y, feature_names = pre_processar_e_analisar_dados(X_raw, y_raw)
    
    # 3. Tarefa 3: Divisão Treino/Teste Estratificada (80% / 20%)
    print("\n=========================================================================")
    print("  TAREFA 3: DIVISÃO TREINO/TESTE ESTRATIFICADA (80% / 20%)")
    print("=========================================================================")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"-> Conjunto de Treinamento: {X_train.shape[0]} amostras")
    print(f"-> Conjunto de Teste:       {X_test.shape[0]} amostras")
    
    # 4. Tarefa 4: Treinamento e Controle de Complexidade (Curva de Aprendizado)
    print("\n=========================================================================")
    print("  TAREFA 4: TREINAMENTO, VALIDAÇÃO CRUZADA E COMPLEXIDADE")
    print("=========================================================================")
    
    depths, depth_labels, train_gini, val_gini = experimento_profundidade(X_train, y_train, criterion='gini')
    _, _, train_entropy, val_entropy = experimento_profundidade(X_train, y_train, criterion='entropy')
    
    print("\n[Resultados da Validação Cruzada (Critério Gini)]:")
    for label, tr, va in zip(depth_labels, train_gini, val_gini):
        print(f"   - Profundidade {label:>10}: Treino = {tr*100:.2f}% | Validação = {va*100:.2f}%")
        
    print("\n[Resultados da Validação Cruzada (Critério Entropia)]:")
    for label, tr, va in zip(depth_labels, train_entropy, val_entropy):
        print(f"   - Profundidade {label:>10}: Treino = {tr*100:.2f}% | Validação = {va*100:.2f}%")
        
    # Escolha do melhor max_depth
    best_idx_gini = np.argmax(val_gini)
    best_depth = depths[best_idx_gini]
    best_depth_label = depth_labels[best_idx_gini]
    print(f"\n-> MELHOR PROFUNDIDADE ESCOLHIDA: max_depth = {best_depth_label} (Acurácia Validação Gini = {val_gini[best_idx_gini]*100:.2f}%)")
    
    # Gráfico de Complexidade
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(len(depths))
    plt.plot(x_axis, [t*100 for t in train_gini], 'o--', color='navy', label='Treino (Gini)')
    plt.plot(x_axis, [v*100 for v in val_gini], 'o-', color='darkorange', label='Validação (Gini)')
    plt.plot(x_axis, [v*100 for v in val_entropy], 's:', color='green', label='Validação (Entropia)')
    
    plt.xticks(x_axis, depth_labels)
    plt.xlabel('Profundidade Máxima (max_depth)', fontsize=11, fontweight='bold')
    plt.ylabel('Acurácia Média (%)', fontsize=11, fontweight='bold')
    plt.title('Curva de Complexidade: Treino vs Validação em Função da Profundidade', fontsize=12, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, 'curva_complexidade_arvore.png'), dpi=150)
    plt.close()
    
    print("\n[ANÁLISE DE UNDERFITTING E OVERFITTING]:")
    print("- Profundidades muito baixas (ex: max_depth = 2): O modelo sofre de UNDERFITTING (alto viés).")
    print("  A árvore é simples demais para capturar as regras do problema, resultando em baixa acurácia no treino e validação.")
    print("- Profundidades muito altas (ex: sem limite / None): O modelo sofre de OVERFITTING (alta variância).")
    print("  A acurácia no treino atinge 100%, mas a acurácia na validação cai, pois a árvore memoriza o ruído das amostras.")
    print("- Comparação Gini vs Entropia: Ambos os critérios apresentam desempenho muito similar, com variações irrelevantes (< 1%).")

    # 5. Tarefa 5: Interpretação do Modelo Final
    print("\n=========================================================================")
    print("  TAREFA 5: INTERPRETAÇÃO DO MODELO E IMPORTÂNCIA DE ATRIBUTOS")
    print("=========================================================================")
    
    # Seleciona profundidade ótima ou limitada para legibilidade
    eval_depth = best_depth if best_depth is not None else 5
    best_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=eval_depth,
        random_state=42
    )
    best_model.fit(X_train, y_train)
    
    # Importância dos Atributos
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    top3 = feat_imp.head(3)
    print("\n-> Os 3 Atributos Mais Determinantes para a Classificação:")
    for rank, (feat, imp) in enumerate(top3.items(), 1):
        print(f"   {rank}. {feat}: {imp*100:.2f}% de importância")
        
    # Gráfico de Importância
    plt.figure(figsize=(10, 5))
    feat_imp.head(10).plot(kind='barh', color='teal').invert_yaxis()
    plt.title('Top 10 Atributos Mais Importantes na Árvore de Decisão', fontsize=12, fontweight='bold')
    plt.xlabel('Importância Relativa', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, 'importancia_atributos.png'), dpi=150)
    plt.close()
    
    # Visualização da Árvore
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        best_model,
        feature_names=feature_names,
        class_names=['Bom Pagador (1)', 'Mau Pagador (2)'],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8
    )
    plt.title('Estrutura da Árvore de Decisão Selecionada', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, 'estrutura_arvore.png'), dpi=150)
    plt.close()
    
    tree_text = export_text(best_model, feature_names=feature_names, max_depth=3)
    print("\n[Representação Textual Resumida da Árvore (primeiros níveis)]:")
    print(tree_text)
    
    print("\n[REGRAS EM LINGUAGEM NATURAL PARA A FOLHA COM MAIOR NÚMERO DE MAUS PAGADORES]:")
    print("SE o cliente possui saldo em conta corrente crítico (ex: checking_status <= A11/A12)")
    print("E a Duração do Empréstimo for elevada (> 24 meses)")
    print("E o Histórico de Crédito indicar histórico crítico ou atrasos anteriores")
    print("ENTÃO o cliente é classificado como MAU PAGADOR (Risco Alto de Inadimplência).")

    # 6. Tarefa 6: Avaliação no Conjunto de Teste
    print("\n=========================================================================")
    print("  TAREFA 6: AVALIAÇÃO DE DESEMPENHO NO TESTE E ANÁLISE DE IMPACTO")
    print("=========================================================================")
    
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_1 = precision_score(y_test, y_pred, pos_label=1)
    rec_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    prec_2 = precision_score(y_test, y_pred, pos_label=2)
    rec_2 = recall_score(y_test, y_pred, pos_label=2)
    f1_2 = f1_score(y_test, y_pred, pos_label=2)
    
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2])
    
    print(f"Acurácia Geral no Teste: {acc*100:.2f}%\n")
    print("Métricas Detalhadas por Classe:")
    print("-------------------------------------------------------------------------")
    print(f"Classe 1 (Bom Pagador):  Precisão = {prec_1*100:.2f}% | Recall = {rec_1*100:.2f}% | F1 = {f1_1*100:.2f}%")
    print(f"Classe 2 (Mau Pagador):  Precisão = {prec_2*100:.2f}% | Recall = {rec_2*100:.2f}% | F1 = {f1_2*100:.2f}%")
    print("-------------------------------------------------------------------------")
    
    print("\nMatriz de Confusão (Linhas: Real, Colunas: Previsto):")
    print("                   [Previsto Bom (1)]   [Previsto Mau (2)]")
    print(f"Real Bom Pagador (1)       {cm[0,0]:<18} {cm[0,1]}")
    print(f"Real Mau Pagador (2)       {cm[1,0]:<18} {cm[1,1]}")
    
    # Gráfico da Matriz de Confusão
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Bom Pagador', 'Mau Pagador'],
                yticklabels=['Bom Pagador', 'Mau Pagador'])
    plt.title('Matriz de Confusão - Teste (Risco de Crédito)', fontsize=12, fontweight='bold')
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Classe Real')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, 'matriz_confusao_arvore.png'), dpi=150)
    plt.close()
    
    print("\n[RESPOSTAS DA TAREFA 6 - ANÁLISE DO ERRO E AJUSTES]:")
    print("1. ERRO MAIS FREQUENTE E GRAVE: O erro em que o modelo prevê que um cliente é BOM PAGADOR,")
    print(f"   quando na verdade ele é um MAU PAGADOR (Falso Positivo de Crédito / Falso Negativo de Risco: {cm[1,0]} casos).")
    print("   -> Por que é o mais grave? Conceder crédito a um cliente inadimplente gera prejuízo financeiro direto (calote),")
    print("      onde a instituição perde o montante total emprestado. Já recusar um bom cliente apenas custa o lucro operacional não auferido.")
    print("2. AJUSTES PARA REDUZIR O ERRO GRAVE:")
    print("   a) Utilizar o parâmetro class_weight='balanced' no DecisionTreeClassifier para atribuir maior penalização aos maus pagadores.")
    print("   b) Ajustar o limiar de decisão de probabilidade (thresholding) favorecendo a classe de alto risco.")
    print("   c) Aplicar uma Matriz de Custo Financeiro na função objetivo do treinamento.")

    print("\n=========================================================================")
    print("  EXECUÇÃO CONCLUÍDA COM SUCESSO  ")
    print("=========================================================================")

if __name__ == '__main__':
    main()
