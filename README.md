# 🤖 Telecom X — Parte II: Prevendo Churn

Modelagem preditiva para **antecipar a evasão (churn)** na Telecom X. Esta etapa usa o dataset tratado na Parte I, prepara os dados para ML, treina modelos, avalia desempenho e gera insights acionáveis — tudo em **Jupyter Notebook**.

## Sumário
- [Objetivo](#objetivo)
- [Dados](#dados)
- [Tecnologias](#tecnologias)
- [Como rodar](#como-rodar)
- [Metodologia (Cards)](#metodologia-cards)
- [Resultados (preencha com seus números)](#resultados-preencha-com-seus-números)
- [Limitações e próximos passos](#limitações-e-próximos-passos)

---

## Objetivo
Construir um pipeline de **Machine Learning** para prever clientes com maior probabilidade de **churn**, comparando modelos e destacando variáveis relevantes para **retenção**.

---

## Dados
- Base de entrada: `data/telecomx_clean.csv` (gerado na Parte I).  
  Se o arquivo não existir, o notebook reconstrói a base a partir da **API JSON** e aplica a transformação essencial.
- Target: `Churn` (`Yes`/`No`).

---

## Tecnologias
- Python (Pandas, NumPy)
- Visualização: Matplotlib, Seaborn
- ML: scikit-learn (pipelines, OneHotEncoder, imputação, modelos e métricas)
- Opcional: imbalanced-learn (SMOTE/undersampling)

---

## Como rodar

### Google Colab (recomendado)
1. Abra `TelecomX_BR_Part_II.ipynb`.
2. Execute as células na ordem (o notebook se ajusta caso o CSV não exista).
3. Ao final, confira métricas, gráficos e o resumo conclusivo.

### Local (opcional)
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install pandas numpy requests matplotlib seaborn scikit-learn
    # (opcional) balanceamento:
    pip install imbalanced-learn
    jupyter notebook

---

## Metodologia (Cards)
- **Cards 1–6 — Preparação**: carrega a base tratada; remove colunas irrelevantes; **One-Hot** nas categóricas; checa desbalanceamento; balanceamento **opcional** (SMOTE/under); **normalização** aplicada só a modelos que precisam.
- **Cards 7–8 — Exploração**: **matriz de correlação** e análises direcionadas (ex.: **Tenure × Churn**, **TotalCharges × Churn**).
- **Cards 9–11 — Modelagem e Avaliação**: split **estratificado** (80/20);  
  - Modelo A (com normalização): **Logistic Regression** em pipeline com imputação e `StandardScaler`;  
  - Modelo B (árvore): **RandomForest** com imputação e `class_weight`;  
  - Métricas: **Acurácia, Precisão, Recall, F1, Matriz de Confusão**.
- **Card 12 — Importância/Coeficientes**: coeficientes (Logística) e importâncias (RandomForest) com gráficos.
- **Card 13 — Conclusão**: principais **drivers** de churn e recomendações de **retenção**.

---

## Resultados (preencha com seus números)
- Split: _ex.: 80/20 (train/test)_.  
- Distribuição do target (teste): _ex.: Yes = 27%, No = 73%_.  
- Logistic Regression (scaled): **Acurácia … | Precisão … | Recall … | F1 …**  
- RandomForest (tree): **Acurácia … | Precisão … | Recall … | F1 …**  
- Drivers (exemplos):  
  - ↑ churn: _ex.: Month-to-month, PaperlessBilling, certos métodos de pagamento_  
  - ↓ churn: _ex.: contrato longo, tenure alto, bundles de segurança/backup_

---

## Limitações e próximos passos
- Validar com **cross-validation (5-fold)** e ajustar hiperparâmetros (RF: `max_depth`, `min_samples_*`; LR: `C`).  
- **Threshold tuning** conforme objetivo (ex.: maximizar **Recall** para reduzir falsos negativos).  
- Interpretabilidade adicional (**Permutation Importance/SHAP**) e **engenharia de novas variáveis** (uso, tickets, inadimplência).
