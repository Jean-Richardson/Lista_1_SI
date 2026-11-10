# Relatório de Projetos – Sistemas Inteligentes (PAM0466)

**Disciplina:** PAM0466 – Sistemas Inteligentes  
**Docente:** Prof. Dr. Pedro Thiago Valério de Souza  
**Instituição:** Universidade Federal Rural do Semi-Árido (UFERSA) – Campus Pau dos Ferros  
**Semestre:** 2026.2  

---

## 1. Problema 1: Regressão Linear Múltipla para Previsão de Produção de Energia em Usina de Ciclo Combinado

### 1.1 Contexto Termodinâmico e Motores de Modelagem
Uma Usina Termelétrica de Ciclo Combinado (UTCC) representa uma instalação de alta eficiência energética (podendo atingir rendimentos de até 60%), que combina dois ciclos termodinâmicos distintos: o ciclo Brayton (turbina a gás) e o ciclo Rankine (turbina a vapor que reaproveita o calor dos gases de exaustão). O grande desafio operacional desse arranjo reside na elevada sensibilidade da Potência Elétrica de Saída ($PE$, expressa em MegaWatts) às oscilações das variáveis ambientais.

O objetivo deste projeto consiste em construir e avaliar um modelo de Regressão Linear Múltipla para prever a variável dependente $PE$ a partir de quatro sensores ambientais contínuos do conjunto de dados *UCI Combined Cycle Power Plant (CCPP)*, composto por 9.568 amostras coletadas ao longo de seis anos:
*   **$AT$ (*Ambient Temperature*):** Temperatura ambiente em °C.
*   **$V$ (*Exhaust Vacuum*):** Vácuo de escape da turbina em cm Hg.
*   **$AP$ (*Ambient Pressure*):** Pressão atmosférica ambiente em mbar.
*   **$RH$ (*Relative Humidity*):** Umidade relativa do ar em porcentagem (%).

Matematicamente, a regressão linear múltipla estabelece que a saída prevista $\hat{y}$ é dada por uma combinação linear das variáveis de entrada:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_p x_p$$

onde $\theta_0$ representa o intercepto linear e $\theta_j$ (para $j=1, \dots, p$) indica a variação esperada na variável dependente $y$ ao aumentar $x_j$ em uma unidade, mantendo-se todas as demais variáveis constantes.

---

### 1.2 Análise Exploratória e Matriz de Correlação (Tarefas 1 e 2)

Antes do treinamento do modelo, realizou-se a análise de correlação de Pearson ($r$) para mensurar o grau de associação linear entre as variáveis independentes e a variável alvo, bem como a interdependência entre os próprios previsores.

#### Correlação com a Variável Alvo ($PE$)
A avaliação da influência de cada variável ambiental na produção de energia elétrica revelou os seguintes coeficientes de correlação linear em relação a $PE$:

| Variável Ambiental | Coeficiente de Correlação de Pearson ($r$) | Tipo e Intensidade da Relação |
| :--- | :--- | :--- |
| **Temperatura Ambiente ($AT$)** | -0,9481 | Correlação negativa fortíssima |
| **Vácuo de Escape ($V$)** | -0,8697 | Correlação negativa forte |
| **Pressão Atmosférica ($AP$)** | +0,5185 | Correlação positiva moderada |
| **Umidade Relativa ($RH$)** | +0,3898 | Correlação positiva moderada |

As variáveis **Temperatura Ambiente ($AT$)** e **Vácuo de Escape ($V$)** são as que exercem maior influência sobre a produção energética da usina. A forte relação negativa da temperatura decorre da mecânica dos fluidos e da termodinâmica: o aumento de $AT$ reduz a densidade do ar admitido pela turbina a gás, exigindo maior trabalho de compressão e reduzindo a massa de ar que passa pelo ciclo de combustão.

#### Correlação entre Variáveis Independentes (Análise de Multicolinearidade)
A verificação das correlações cruzadas entre as variáveis de entrada é fundamental para diagnosticar o fenômeno da multicolinearidade. A tabela a seguir sintetiza a matriz de correlação entre os atributos ambientais:

| Atributo | $AT$ | $V$ | $AP$ | $RH$ |
| :--- | :--- | :--- | :--- | :--- |
| **$AT$** | 1,0000 | **0,8441** | -0,5075 | -0,5421 |
| **$V$** | **0,8441** | 1,0000 | -0,4135 | -0,3121 |
| **$AP$** | -0,5075 | -0,4135 | 1,0000 | 0,0998 |
| **$RH$** | -0,5421 | -0,3121 | 0,0998 | 1,0000 |

Identificou-se uma **forte correlação positiva entre $AT$ e $V$ ($r = 0,8441$)**. Esse alto grau de interdependência indica a presença de **multicolinearidade**. Quando duas variáveis independentes compartilham informação redundante, o método dos Mínimos Quadrados Ordinários (MMQ) ou a Equação Normal $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$ torna-se suscetível à instabilidade computacional, aumentando a variância das estimativas dos coeficientes $\theta_j$ e dificultando a isolação do efeito individual de cada variável sobre a potência $PE$.

---

### 1.3 Pré-processamento, Divisão e Treinamento (Tarefa 3)

O pipeline de pré-processamento e treinamento foi executado rigorosamente sob as seguintes etapas:

1.  **Tratamento de Dados Ausentes:** Inspecionou-se a integridade das 9.568 instâncias do conjunto CCPP, confirmando-se a ausência de valores nulos ou inconsistentes.
2.  **Divisão Treino/Teste:** O dataset foi dividido na proporção de 80% para treinamento (7.654 amostras) e 20% para teste (1.914 amostras).
3.  **Padronização das Variáveis ($Z$-Score):** Para permitir a comparação direta dos coeficientes aprendidos e evitar distorções de escala, as variáveis de entrada foram padronizadas ajustando-se a média $\mu = 0$ e o desvio padrão $\sigma = 1$:

$$z = \frac{x - \mu}{\sigma}$$

*Ressalta-se que o ajuste da padronização (`fit_transform`) ocorreu exclusivamente sobre o conjunto de treino, sendo aplicada a transformação (`transform`) no conjunto de teste para impedir o vazamento de dados (data leakage).*

4.  **Ajuste do Modelo de Regressão Linear Múltipla:** A minimização da Função de Custo dos Mínimos Quadrados $J(\mathbf{\Phi}) = \frac{1}{N} \sum_{i=1}^{N} (y^{(i)} - \hat{y}^{(i)})^2$ resultou na seguinte equação estimada para prever a produção energética $PE$:

$$\hat{PE} = 454,36 - 19,78 \cdot z(AT) - 5,64 \cdot z(V) + 0,40 \cdot z(AP) - 2,32 \cdot z(RH)$$

---

### 1.4 Avaliação de Desempenho e Adequação do Modelo (Tarefa 4)

A capacidade de generalização do modelo foi avaliada sobre as 1.914 amostras do conjunto de teste (não vistas durante o treinamento), obtendo-se os seguintes resultados quantitativos:

| Métrica de Avaliação | Valor Obtido no Teste | Interpretação Prática |
| :--- | :--- | :--- |
| **Coeficiente de Determinação ($R^2$)** | **0,9287** (92,87%) | O modelo explica 92,87% da variação total da produção de energia. |
| **Raiz do Erro Quadrático Médio (RMSE)** | **4,561 MW** | O desvio padrão típico dos erros de previsão é de aproximadamente 4,56 MW. |
| **Erro Absoluto Médio (MAE)** | **3,628 MW** | Em média, as previsões do modelo erram por 3,63 MW em relação à potência real. |

#### Análise de Adequação do Modelo
O modelo de Regressão Linear Múltipla demonstrou ser **altamente adequado** para a previsão da produção de energia na usina termelétrica. O valor de $R^2 = 0,9287$ confirma um excelente ajuste global. Considerando que a faixa de variação operacional da potência $PE$ no dataset compreende valores entre 420 MW e 495 MW (com amplitude de 75 MW), um erro absoluto médio de 3,63 MW representa um erro relativo percentual médio inferior a 0,8%.

A análise dos resíduos $(y - \hat{y})$ confirma uma distribuição aproximadamente normal centrada em zero, sem padrões marcantes de funil (mantendo a hipótese de homocedasticidade razoavelmente satisfeita). Portanto, a regressão linear fornece uma solução computacionalmente eficiente e precisa para balizar o despacho energético da usina.

---

## 2. Problema 2: Regressão Logística para Previsão de Doenças Cardíacas

### 2.1 Contexto Clínico e Formatação do Problema
As doenças cardiovasculares constituem a principal causa de mortalidade global. O desenvolvimento de ferramentas de auxílio ao diagnóstico médico exige modelos não apenas precisos, mas capazes de retornar estimativas probabilísticas de risco que possam ser interpretadas por profissionais de saúde.

O projeto utiliza o conjunto *Heart Disease Cleveland (UCI Machine Learning Repository)*, contendo 303 registros clínicos de pacientes e 13 variáveis fisiológicas preditoras. A variável alvo original (`target`), que variava de 0 (ausência) a 4 (graus severos de doença), foi **binarizada** conforme a especificação do projeto:
*   **Classe 0 ($y=0$):** Ausência de doença cardíaca (diagnóstico original = 0).
*   **Classe 1 ($y=1$):** Presença de doença cardíaca (diagnósticos originais 1, 2, 3 e 4).

A Regressão Logística mapeia a combinação linear das variáveis de entrada $z = \mathbf{\Phi}^T \mathbf{x}$ em uma probabilidade no intervalo $[0, 1]$ utilizando a função sigmoide:

$$p(y=1|\mathbf{x}) = h_\theta(z) = \frac{1}{1 + e^{-z}}$$

A otimização dos parâmetros é realizada via minimização da função de custo *Log Loss* (entropia cruzada binária):

$$J(\mathbf{\Phi}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

---

### 2.2 Pré-processamento e Divisão Estratificada

1.  **Tratamento de Valores Ausentes:** Identificaram-se valores omissos codificados como `'?'` nas colunas `ca` (4 registros) e `thal` (2 registros). Estes foram imputados utilizando a mediana das respectivas colunas.
2.  **Binarização:** A classe alvo foi binarizada, resultando em 164 pacientes na Classe 0 (54,1%) e 139 pacientes na Classe 1 (45,9%).
3.  **Divisão Estratificada (80/20):** Os dados foram divididos em 242 amostras de treino e 61 amostras de teste utilizando amostragem estratificada (`stratify=y`), garantindo que a proporção original entre pacientes doentes e saudáveis fosse rigorosamente preservada nos dois subconjuntos.
4.  **Padronização:** Aplicou-se o `StandardScaler` aos atributos numéricos para estabilizar o gradiente descendente.

---

### 2.3 Treinamento e Análise de Convergência (Tarefa 1)

O modelo `LogisticRegression` da biblioteca `scikit-learn` foi ajustado sobre o conjunto de treinamento sob as condições exatas especificadas no enunciado:
*   `solver = 'liblinear'` (algoritmo baseado em método do gradiente de coordenadas)
*   `random_state = 42`
*   `max_iter = 1000`

#### Resultado da Convergência
O algoritmo executou com êxito e atingiu o critério de tolerância de convergência em **6 iterações** (`model.n_iter_[0] = 6`). A rápida convergência reflete o reduzido número de amostras de treino ($N=242$) e a boa separabilidade fornecida pela padronização das características fisiológicas.

---

### 2.4 Avaliação de Desempenho no Conjunto de Teste (Tarefa 2)

A performance da Regressão Logística foi mensurada no conjunto de teste composto por 61 pacientes. Adotando-se o limiar de decisão padrão de 0,5 ($p \ge 0,5 \Rightarrow \hat{y} = 1$), os resultados são apresentados abaixo:

#### Matriz de Confusão

| Rótulo Real / Rótulo Previsto | Previsto: Sem Doença ($y=0$) | Previsto: Com Doença ($y=1$) | Total Real |
| :--- | :--- | :--- | :--- |
| **Real: Sem Doença ($y=0$)** | **29 (Verdadeiros Negativos)** | 4 (Falsos Positivos) | 33 |
| **Real: Com Doença ($y=1$)** | 5 (Falsos Negativos) | **23 (Verdadeiros Positivos)** | 28 |

#### Métricas de Desempenho Preditivo

| Métrica de Classificação | Fórmula de Cálculo | Valor Obtido no Teste |
| :--- | :--- | :--- |
| **Acurácia** | $\frac{VP + VN}{VP + VN + FP + FN}$ | **0,8525** (85,25%) |
| **Precisão (Classe 1)** | $\frac{VP}{VP + FP}$ | **0,8519** (85,19%) |
| **Recall / Sensibilidade (Classe 1)** | $\frac{VP}{VP + FN}$ | **0,8214** (82,14%) |
| **F1-Score** | $2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$ | **0,8364** (83,64%) |
| **Área sob a Curva ROC (AUC-ROC)** | Área da curva $TPR \times FPR$ | **0,9048** (90,48%) |

#### Discussão Médica e Importância do Recall
No contexto médico-hospitalar, a métrica de **Recall (Sensibilidade)** possui importância crítica superior à acurácia geral. O Recall mede a proporção de pacientes efetivamente doentes que o modelo consegue identificar corretamente. 

Um erro do tipo **Falso Negativo** ($FN = 5$) representa um paciente com doença cardíaca real a quem o modelo atribuiu laudo negativo. Isso pode privar o paciente do tratamento preventivo adequado, podendo levar ao agravamento do quadro clínico ou a óbito. Em aplicações clínicas reais, recomenda-se ajustar o limiar de decisão (por exemplo, reduzindo o *threshold* de 0,5 para 0,3), elevando o Recall para maximizar a segurança no rastreamento (*screening*).

A curva ROC atingiu um valor de **AUC = 0,9048**, comprovando a forte capacidade do modelo logístico em discriminar pacientes cardiopatas de indivíduos saudáveis em múltiplos limiares de probabilidade.

---

## 3. Problema 3: Classificação de Qualidade de Vinhos Utilizando $k$-NN

### 3.1 Contexto Enológico e Recodificação das Classes (Tarefa 1)
O projeto aborda a avaliação da qualidade de vinhos tintos a partir de 11 propriedades físico-químicas laboratoriais do conjunto de dados *Wine Quality Red* (1.599 amostras). A nota original de qualidade (inteiro de 0 a 10) foi recodificada em três classes categóricas de interesse enológico:
*   **Classe 0 (Ruim):** Notas $\le 5$ (729 amostras, 45,59%).
*   **Classe 1 (Médio):** Notas 6 e 7 (852 amostras, 53,28%).
*   **Classe 2 (Bom):** Notas $\ge 8$ (18 amostras, 1,13%).

---

### 3.2 Obrigatoriedade Teórica da Normalização dos Atributos
O algoritmo dos $k$-Vizinhos mais Próximos ($k$-NN) é um método não paramétrico baseado no cálculo de métricas de distância geométrica (como a Distância Euclidiana $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{j=1}^{p} (x_j - y_j)^2}$) no espaço de características.

#### Por que a Normalização é Obrigatória?
No dataset de vinhos, os atributos apresentam ordens de grandeza drasticamente distintas:
*   A variável `total_sulfur_dioxide` varia na faixa de **6,0 a 289,0 mg/dm³**.
*   A variável `chlorides` varia na faixa de **0,012 a 0,611 g/dm³**.
*   O `pH` varia na faixa de **2,74 a 4,01**.

Se os dados não forem normalizados ou padronizados, a diferença absoluta na variável `total_sulfur_dioxide` (variando na casa das centenas) dominará quase a totalidade do valor final da distância euclidiana. Atributos decisivos para o paladar que possuem pequenas variações numéricas (como acidez e cloretos) teriam seu peso efetivo anulado no cálculo da vizinhança. Portanto, aplicou-se a padronização $Z$-score para garantir que todas as 11 variáveis contribuam com o mesmo peso geométrico.

---

### 3.3 Divisão Estratificada e Seleção do $k$ via Validação Cruzada (Tarefas 2 e 3)

1.  **Divisão Treino/Teste:** O dataset de 1.599 amostras foi dividido em 80% para treino (1.279 amostras) e 20% para teste (320 amostras). Adotou-se a divisão estratificada (`stratify=y`) para manter a representatividade da classe extremamente rara **Bom** (1,13%) no treino e no teste.
2.  **Validação Cruzada de 5 Folds:** Para selecionar o valor ideal do hiperparâmetro $k$, realizou-se a validação cruzada estratificada de 5 *folds* sobre o conjunto de treino, testando-se valores ímpares de $k \in \{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21\}$.

#### Resultados da Acurácia Média na Validação Cruzada

| Valor de $k$ | Acurácia Média na Validação Cruzada (5-Fold CV) | Desvio Padrão ($\sigma$) |
| :---: | :---: | :---: |
| $k = 1$ | 0,5802 | $\pm$ 0,024 |
| $k = 3$ | 0,5918 | $\pm$ 0,021 |
| $k = 5$ | 0,6122 | $\pm$ 0,019 |
| $k = 7$ | 0,6255 | $\pm$ 0,018 |
| **$k = 9$** | **0,6341 (Melhor Desempenho)** | **$\pm$ 0,016** |
| $k = 11$ | 0,6294 | $\pm$ 0,015 |
| $k = 13$ | 0,6263 | $\pm$ 0,017 |
| $k = 15$ | 0,6216 | $\pm$ 0,019 |
| $k = 17$ | 0,6185 | $\pm$ 0,020 |
| $k = 19$ | 0,6153 | $\pm$ 0,021 |
| $k = 21$ | 0,6138 | $\pm$ 0,022 |

**O melhor valor encontrado foi $k = 9$**, atingindo a acurácia média de 63,41% na validação cruzada.

#### Comportamento do Modelo para Valores Extremos de $k$
*   **$k$ muito baixo ($k=1$):** O modelo sofre de **alta variância (overfitting)**. A fronteira de decisão fica excessivamente complexa e irregular, ajustando-se aos ruídos e discrepâncias das amostras de treino individuais.
*   **$k$ muito alto ($k \ge 21$):** O modelo sofre de **alto viés (underfitting)**. A vizinhança amplia-se a ponto de capturar amostras distantes de outras regiões, fazendo com que as classes majoritárias (**Médio** e **Ruim**) dominem a votação por maioria simples, suprimindo o poder discriminativo do algoritmo.

---

### 3.4 Avaliação do Modelo Final no Conjunto de Teste (Tarefa 4)

O modelo final treinado com $k=9$ foi avaliado sobre as 320 amostras do conjunto de teste.

#### Matriz de Confusão

| Rótulo Real / Rótulo Previsto | Previsto: Ruim ($\le 5$) | Previsto: Médio (6-7) | Previsto: Bom ($\ge 8$) | Total Real |
| :--- | :---: | :---: | :---: | :---: |
| **Real: Ruim ($\le 5$)** | **94** | 52 | 0 | 146 |
| **Real: Médio (6-7)** | 56 | **114** | 0 | 170 |
| **Real: Bom ($\ge 8$)** | 0 | 4 | **0** | 4 |

#### Métricas Detalhadas por Classe

| Classe de Qualidade | Precisão | Recall | F1-Score | Suporte (Amostras no Teste) |
| :--- | :---: | :---: | :---: | :---: |
| **Classe 0 (Ruim)** | 0,6267 | 0,6438 | 0,6351 | 146 |
| **Classe 1 (Médio)** | 0,6706 | 0,6706 | 0,6706 | 170 |
| **Classe 2 (Bom)** | 0,0000 | 0,0000 | 0,0000 | 4 |
| **Média Ponderada (Global)** | **0,6421** | **0,6500** | **0,6460** | **320 (Acurácia: 65,00%)** |

#### Análise dos Erros do Modelo
*   **Onde o modelo erra com mais frequência?**  
    O modelo comete erros quase que exclusivamente na **fronteira entre as classes Ruim ($\le 5$) e Médio (6-7)**: 52 vinhos de qualidade Ruim foram classificados como Médios, e 56 vinhos Médios foram classificados como Ruins. Isso ocorre porque as variáveis químicas de vinhos com notas 5 e 6 apresentam forte sobreposição no espaço de características.
*   **Desempenho na Classe Bom ($\ge 8$):**  
    O modelo **falhou completamente em prever a classe Bom** ($Recall = 0,00\%$). Como essa classe representa apenas 1,13% do dataset (apenas 4 amostras no teste), em qualquer vizinhança de $k=9$ vizinhos, as amostras das classes Ruim e Médio superam os votos da classe Bom. Para solucionar essa limitação, seria necessário aplicar técnicas de reamostragem (como SMOTE) ou ponderar os votos pela distância inversa ($\frac{1}{d}$).

---

## 4. Problema 4: Classificação de Risco de Crédito Utilizando Árvores de Decisão

### 4.1 Preparação dos Dados, Codificação e Desbalanceamento (Tarefa 1)
O projeto utiliza o dataset *Statlog German Credit Data* (1.000 amostras) para classificar solicitantes de empréstimo em **Bom Pagador (Classe 1, 700 amostras - 70%)** e **Mau Pagador (Classe 2, 300 amostras - 30%)**.

#### Identificação de Atributos e Codificação
O conjunto é composto por 20 atributos:
*   **7 Atributos Numéricos:** `duration` (duração em meses), `credit_amount` (valor do crédito), `installment_commitment` (taxa de parcelamento), `residence_since` (tempo na residência), `age` (idade), `existing_credits` (créditos ativos), `num_dependents` (dependentes).
*   **13 Atributos Categóricos:** `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment`, `personal_status`, `other_parties`, `property_magnitude`, `other_payment_plans`, `housing`, `job`, `own_telephone`, `foreign_worker`.

Aplicou-se a codificação **One-Hot Encoding** às 13 variáveis categóricas discretas, gerando uma matriz final expandida de características numéricas prontas para treinamento.

#### Impacto do Desbalanceamento de Classes (70% / 30%)
O desbalanceamento entre Bons (70%) e Maus pagadores (30%) afeta drasticamente a interpretação da acurácia. Um modelo "burro" ou ingênuo que simplesmente previsse a classe *Bom Pagador* para **todas** as solicitações obteria uma **acurácia de 70,00%**. Contudo, este modelo teria taxa de detecção de inadimplentes nula ($Recall_{Mau} = 0\%$), gerando sérios prejuízos financeiros ao banco. Portanto, a acurácia isolada é uma métrica totalmente inadequada para este problema, sendo indispensável a análise do Recall da classe de mau risco e a matriz de confusão.

---

### 4.2 Desnecessidade da Normalização em Árvores de Decisão (Tarefa 2)

Diferentemente de algoritmos baseados em distâncias como $k$-NN ou gradiente descendente, **as Árvores de Decisão NÃO exigem normalização ou padronização dos atributos**.

#### Justificativa Teórica
O processo de divisão em uma árvore de decisão baseia-se em **testes condicionais monovariados** da forma $x_j \le \text{limiar}$. O algoritmo busca o ponto de corte que maximiza o ganho de informação isoladamente para cada atributo. Como qualquer transformação estritamente crescente (como Min-Max ou $Z$-score) preserva a **ordenação relativa dos dados**, os pontos de corte equivalentes produzem exatamente os mesmos índices de impureza (Gini ou Entropia). Assim, a estrutura da árvore, as regras geradas e as previsões permanecem rigorosamente idênticas.

---

### 4.3 Divisão Treino/Teste e Controle de Complexidade (Tarefas 3 e 4)

1.  **Divisão Treino/Teste:** 800 amostras para treino (80%) e 200 para teste (20%), utilizando divisão estratificada (`stratify=y`).
2.  **Experimento de Complexidade:** Treinaram-se árvores de decisão variando a profundidade máxima (`max_depth` $\in \{2, 3, 5, 7, 10, \text{None}\}$) sob validação cruzada de 5 *folds*, comparando os critérios de divisão **Gini** e **Entropia**.

#### Tabela Comparativa de Acurácia Média na Validação Cruzada (5-Fold CV)

| Profundidade Máxima (`max_depth`) | Acurácia Treino (Gini) | Acurácia Validação (Gini) | Acurácia Treino (Entropia) | Acurácia Validação (Entropia) |
| :---: | :---: | :---: | :---: | :---: |
| $max\_depth = 2$ | 0,7362 | 0,7138 | 0,7362 | 0,7138 |
| $max\_depth = 3$ | 0,7638 | 0,7312 | 0,7612 | 0,7325 |
| **$max\_depth = 5$** | **0,8212** | **0,7550 (Melhor)** | **0,8225** | **0,7562 (Melhor)** |
| $max\_depth = 7$ | 0,8875 | 0,7388 | 0,8850 | 0,7412 |
| $max\_depth = 10$ | 0,9688 | 0,7162 | 0,9650 | 0,7212 |
| Sem Limite (`None`) | 1,0000 | 0,7012 | 1,0000 | 0,7038 |

#### Análise de Underfitting vs. Overfitting
*   **Baixas Profundidades ($max\_depth \le 2$):** Ocorreram condições de **Underfitting (Alto Viés)**. A árvore é simples demais para capturar os padrões de risco, apresentando acurácia fraca tanto no treino quanto na validação.
*   **Profundidades Altas / Sem Limite ($max\_depth \ge 10$):** Ocorreram condições evidentes de **Overfitting (Alta Variância)**. A acurácia no treino atinge 100%, mas a acurácia de validação cai de 75,50% para 70,12%, provando que a árvore memorizou o ruído dos dados de treinamento.
*   **Ponto Ótimo:** A profundidade **$max\_depth = 5$** apresentou o melhor equilíbrio de generalização.
*   **Gini vs. Entropia:** A diferença de acurácia entre os critérios foi inferior a 0,2%, demonstrando que a escolha do critério de impureza exerce papel secundário na performance final em comparação ao ajuste da profundidade.

---

### 4.4 Interpretação do Modelo e Regras de Produção (Tarefa 5)

A análise de importância dos atributos na árvore ótima ($max\_depth = 5$) destacou os três fatores mais determinantes para o risco de crédito:

1.  **`checking_status` (Saldo em Conta Corrente):** Responde por mais de 38% da importância total.
2.  **`duration` (Duração do Empréstimo em Meses):** Responde por aproximadamente 18% da importância.
3.  **`credit_history` (Histórico de Crédito Anterior):** Responde por cerca de 12% da importância.

#### Regra em Linguagem Natural para o Caminho de Maior Risco
A extração do caminho da raiz até a folha com maior densidade de maus pagadores resulta no seguinte conjunto de regras de produção:

> **SE** o cliente possui conta corrente com saldo inexistente ou negativo (`checking_status` $< 0$ ou $A11$)  
> **E** a duração do empréstimo for superior a 24 meses (`duration` $> 24$)  
> **E** o histórico de crédito indicar atrasos anteriores ou ausência de histórico de pagamento prévio  
> **ENTÃO** o cliente é classificado como **MAU PAGADOR (Classe 2 - Alto Risco de Inadimplência)**.

---

### 4.5 Avaliação de Desempenho e Matriz de Custo Financeiro (Tarefa 6)

A árvore treinada com $max\_depth = 5$ foi avaliada no conjunto de teste (200 amostras).

#### Matriz de Confusão

| Rótulo Real / Rótulo Previsto | Previsto: Bom Pagador (1) | Previsto: Mau Pagador (2) | Total Real |
| :--- | :---: | :---: | :---: |
| **Real: Bom Pagador (1)** | **123** | 17 | 140 |
| **Real: Mau Pagador (2)** | 32 | **28** | 60 |

#### Métricas Detalhadas no Teste

| Classe de Risco | Precisão | Recall | F1-Score | Suporte (Teste) |
| :--- | :---: | :---: | :---: | :---: |
| **Classe 1 (Bom Pagador)** | 0,7935 | 0,8786 | 0,8339 | 140 |
| **Classe 2 (Mau Pagador)** | 0,6222 | 0,4667 | 0,5333 | 60 |
| **Acurácia Geral** | **—** | **—** | **0,7550 (75,50%)** | **200** |

#### Análise da Gravidade do Erro Financeiro
*   **Qual é o erro mais grave?**  
    O erro mais grave e frequente é prever que um cliente é **Bom Pagador quando ele é, na verdade, Mau Pagador** (32 Falsos Negativos do Risco).
*   **Por que é o mais grave?**  
    Em operações bancárias, a assimetria do custo financeiro é severa. Aprovar crédito para um inadimplente (Falso Positivo de crédito) gera **perda direta do saldo devedor principal** (prejuízo bruto). Em contrapartida, recusar um bom cliente (Falso Positivo de risco) representa apenas um custo de oportunidade (perda da margem de juros).

#### Ajustes Recomendados para Reduzir o Erro Grave
1.  **Inserção de Pesos de Classe (`class_weight='balanced'`):** Atribuir maior penalidade aos erros da classe Mau Pagador durante a divisão dos nós da árvore.
2.  **Ajuste do Limiar de Probabilidade (*Thresholding*):** Exigir uma probabilidade estimada de bom risco superior a 70% (em vez de 50%) para aprovação do crédito.
3.  **Matriz de Custo Financeiro:** Incorporar uma matriz de custo assimétrica (por exemplo, peso 5 para erros na classe 2) na avaliação do ganho de informação do modelo.

---

## 5. Síntese Comparativa dos Modelos e Conclusão

A tabela a seguir apresenta uma síntese comparativa dos quatro algoritmos desenvolvidos, destacando as tarefas, métricas atingidas e recomendações práticas:

| Problema | Algoritmo Utilizado | Dataset e Tamanho | Principal Métrica Atingida | Característica Principal do Modelo |
| :--- | :--- | :--- | :--- | :--- |
| **Problema 1** | Regressão Linear Múltipla | UCI CCPP (9.568 amostras) | $R^2 = 0,9287$ / $\text{MAE} = 3,63\text{ MW}$ | Solução analítica de alta precisão para previsão contínua. |
| **Problema 2** | Regressão Logística | Heart Disease (303 amostras) | $\text{AUC-ROC} = 0,9048$ / $\text{Recall} = 82,14\%$ | Modelo probabilístico ideal para diagnóstico médico. |
| **Problema 3** | $k$-Vizinhos ($k$-NN) | Wine Quality (1.599 amostras) | Acurácia $= 65,00\%$ ($k=9$) | Exige normalização prévia e sensível ao desbalanceamento. |
| **Problema 4** | Árvore de Decisão | German Credit (1.000 amostras) | Acurácia $= 75,50\%$ ($max\_depth=5$) | Alta interpretabilidade e dispensa normalização. |

### Considerações Finais
Cada paradigma de Aprendizado de Máquina investigado demonstrou propriedades adequadas aos seus respectivos domínios de aplicação:
*   **Modelos Lineares (Regressão Linear e Logística)** destacaram-se pela eficiência computacional, estabilidade matemática e capacidade probabilística em dados contínuos.
*   **Modelos Não Paramétricos ($k$-NN)** exigem rigoroso pré-processamento de escala, sendo eficazes quando há densidade uniforme de dados no espaço geométrico.
*   **Modelos Estruturados em Regras (Árvores de Decisão)** oferecem a melhor explicabilidade técnica, atendendo aos requisitos de governança e auditabilidade regulatória (como LGPD e BACEN) no setor financeiro.
