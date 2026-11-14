# 🛡️ Detecção de Fraudes em Cartão de Crédito com Machine Learning e Arquitetura de Triagem Híbrida
Desafio do Bootcamp Bradesco da DIO

Projeto desenvolvido como desafio prático de Ciência de Dados e Machine Learning, focado na detecção de transações fraudulentas em um cenário de **desbalanceamento extremo (0,17% de fraudes)**, priorizando métricas orientadas ao risco financeiro (**Recall**, **Precision** e **PR-AUC**) e explicabilidade de modelo com **SHAP**.

---

## 📌 1. O Problema e o Impacto do Desbalanceamento

A base de dados contém **284.807 transações reais** de cartões de crédito realizadas em um período de 48 horas, das quais apenas **492 são fraudes (`0,1727%`)** e **284.315 são operações legítimas (`99,8273%`)**.

* **Por que a Acurácia engana?** Um modelo ingênuo que classifique 100% das transações como legítimas alcança **99,83% de acurácia**, mas falha em capturar qualquer fraude.
* **Métricas adotadas:** O foco da avaliação foi direcionado para a classe minoritária (`Class = 1`):
  * **Recall (Revocação):** Percentual de fraudes reais interceptadas (reduz prejuízo financeiro direto).
  * **Precision (Precisão):** Percentual de alertas disparados que eram realmente fraudes (evita bloqueio indevido de clientes honestos e atrito no atendimento).
  * **F1-Score e PR-AUC:** Equilíbrio entre captura e precisão em cenários desbalanceados.

---

## ⚙️ 2. Preparação dos Dados e Engenharia de Variáveis

1. **Engenharia de Variáveis (*Feature Engineering*):**
   * `Hour`: Conversão do contador bruto de segundos (`Time`) para a hora cíclica do dia (`0h` a `23h`), permitindo capturar padrões de comportamento na madrugada.
   * `Log_Amount`: Transformação logarítmica (`np.log1p`) sobre o valor da transação (`Amount`) para reduzir a assimetria de valores extremos.
2. **Escalonamento e Prevenção de *Data Leakage*:**
   * Uso de `StandardScaler` encapsulado dentro de `Pipeline` do `scikit-learn`, ajustado exclusivamente nos dados de treino.
3. **Divisão Estratificada:**
   * Separação em **80% Treino** (227.845 transações / 394 fraudes) e **20% Teste** (56.962 transações / 98 fraudes) com `stratify=y`, preservando a proporção original de `0,17%`.

---

## 📊 3. Comparação entre Modelos (Classe 1 — Fraude)

| Modelo / Configuração | Precision | Recall | F1-Score | Fraudes Capturadas (de 98) | Falsos Positivos |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | 82,05% | 65,31% | 72,73% | 64 | 14 |
| **Random Forest (`balanced`, limiar `0.50`)** | **96,00%** | 73,47% | 83,24% | 72 | **3** |
| **Random Forest (`balanced`, limiar `0.30`)** | 93,10% | **82,65%** | **87,57%** | **81** | 6 |
| **XGBoost (`scale_pos_weight`, limiar `0.80`)** | 59,57% | 85,71% | 70,29% | 84 | 57 |

---

## 🔍 4. Limiar de Decisão, Auditoria de Erros e Explicabilidade (SHAP)

* **Ajuste de Limiar (`0.50` $\rightarrow$ `0.30`):** A auditoria dos Falsos Negativos revelou que diversas fraudes camufladas recebiam probabilidades entre `0.30` e `0.49`. Ao reduzir o limiar de corte no Random Forest para `0.30`, o **Recall saltou de 73,47% para 82,65%** (+9 fraudes interceptadas), mantendo a precisão acima de **93%**.
* **Explicabilidade com SHAP:** A análise dos valores SHAP demonstrou que as componentes principais **`V14`**, **`V12`**, **`V4`**, **`V10`**, **`V3`**, **`V11`** e **`V17`** concentram o maior peso preditivo. Valores fortemente negativos em `V14` e `V12`, combinados com valores elevados em `V4` e `V11`, são os principais impulsionadores para a classificação de fraude.

---

## 🚀 5. Diferenciais Implementados além do Roteiro Base

1. **Auditoria de Falsos Negativos (*Error Analysis*):** Diagnóstico comparativo detalhado entre as fraudes capturadas e as que escaparam, identificando padrões de transações de teste (`$0.00` e `$1.00`) e operações de alto valor na madrugada.
2. **Simulador de Arquitetura de Decisão em 2 Camadas (Funil Bancário):**
   * **Camada 1 — Bloqueio Automático (`Prob >= 0.50`):** Barra **72/98 fraudes** imediatamente com apenas **3 falsos positivos**.
   * **Camada 2 — Fila de Desambiguação / *Step-up Auth* (`0.15 <= Prob < 0.50`):** Intercepta **+12 fraudes** adicionais para validação ativa (Push/SMS no app), acionando apenas **13 clientes legítimos** em mais de 56 mil compras.
   * **Resultado Combinado:** Cobertura total de **84/98 fraudes (`85,71%` de Recall)** com atrito mínimo ao cliente.
3. **Nota sobre Escopo de Dados Anonimizados:** Documentação das limitações impostas pela transformação PCA (`V1`–`V28`) e ausência de `Customer_ID` para rastreamento de velocidade/geolocalização por titular.

### 5.1. Racional de Negócio: Por que aceitar mais Falsos Positivos na Camada 1?
Em sistemas bancários reais, tratar a saída de um modelo de Machine Learning como um interruptor binário ("aprova" ou "bloqueia definitivamente") gera um dilema caro:
* **Limiar rígido (`0.50`):** Protege o cliente legítimo contra bloqueios indevidos (apenas 3 falsos positivos), mas deixa passar **26,5% das fraudes**, gerando prejuízo financeiro direto (*chargeback*) ao banco.
* **Limiar sensível (`0.15` a `0.30`) com Desambiguação:** Ao baixar a régua de corte para capturar fraudes camufladas, o modelo inevitavelmente gera mais alertas preventivos em clientes honestos. **Em produção, isso não é um erro — é uma estratégia de triagem.**

Como o custo de enviar uma notificação *Push* ou *SMS* de confirmação (*"Você reconhece esta compra?"*) é próximo de zero, enquanto o custo de uma fraude não detectada é alto, a arquitetura foi desenhada em **duas camadas de decisão**:

1. **Camada 1 — Bloqueio Automático (`Prob >= 0.50`):**
   * Transações com assinatura clara de fraude são barradas instantaneamente, sem esperar interação do usuário.
   * **Resultado:** **72/98 fraudes** bloqueadas na hora, com apenas **3 falsos positivos**.
2. **Camada 2 — Fila de Desambiguação / *Step-up Authentication* (`0.15 <= Prob < 0.50`):**
   * Transações na "zona cinzenta" não são bloqueadas de imediato (evitando atrito no caixa e reclamações no SAC). Em vez disso, são encaminhadas para validação contextual em tempo real (cruzamento com geolocalização, dispositivo habitual, histórico recente do titular ou desafio via Push/Token no app).
   * **Resultado:** Intercepta **+12 fraudes** que passariam despercebidas, acionando validação extra para apenas **13 clientes legítimos** em um universo de **56.864 compras normais** (taxa de acionamento de `0,02%`).
3. **Impacto Combinado:** A cobertura total salta para **84/98 fraudes (`85,71%` de Recall)**, maximizando a proteção financeira sem sacrificar a experiência do correntista.

### 5.2. Auditoria de Falsos Negativos (*Error Analysis*) e Escopo PCA
* **Diagnóstico de Erros:** Isolamento das fraudes que escaparam ao corte padrão (`0.50`), identificando padrões de transações de teste (`$0.00` e `$1.00`) e compras atípicas na madrugada.
* **Limitação de Dados Anonimizados:** Documentação explícita de que, devido à transformação PCA (`V1`–`V28`) e à ausência de `Customer_ID` por sigilo bancário, regras de velocidade por titular ou geolocalização impossível foram modeladas conceitualmente na camada de desambiguação.
---

## 🛠️ Tecnologias Utilizadas
* **Python 3**, **Pandas**, **NumPy**
* **Scikit-Learn** (`Pipeline`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`)
* **XGBoost** & **SHAP**
* **Matplotlib**
