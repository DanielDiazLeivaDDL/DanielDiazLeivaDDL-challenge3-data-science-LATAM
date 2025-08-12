# DanielDiazLeivaDDL-challenge3-data-science-LATAM
Desarrollo del challenge 'Telecom X 2' Estadisticas y Machine Learning G8 - ONE
# 📊 Análisis y Modelado de Churn de Clientes

Este proyecto contiene el código necesario para realizar un análisis exploratorio de datos, preprocesamiento y modelado predictivo sobre la cancelación de clientes (churn) en un conjunto de datos de telecomunicaciones.

---

## 📂 Contenido

El código implementa las siguientes etapas:

1. **Extracción de datos**  
   - Lectura del dataset desde una URL pública en formato CSV.
   - Inspección inicial de la estructura y valores únicos.

2. **Análisis exploratorio (EDA)**  
   - Visualizaciones de variables categóricas y numéricas.
   - Histogramas y diagramas de caja (`boxplot`) con **Plotly**, **Matplotlib** y **Seaborn**.
   - Análisis dirigido de relaciones clave (e.g., `tenure` vs `Churn`, `Charges.Total` vs `Churn`).

3. **Transformación de datos**  
   - Limpieza de valores nulos y espacios en blanco.
   - Codificación de variables categóricas con **OneHotEncoder** y `get_dummies`.
   - Mapeo binario de variables (`Yes/No`, `Male/Female`).
   - Cálculo de correlaciones con la variable objetivo.

4. **Construcción de modelos**  
   - **Baseline** con `DummyClassifier`.
   - **Árbol de Decisión** con diferentes profundidades.
   - **K-Nearest Neighbors (KNN)** con normalización.
   - Validación cruzada estratificada con `StratifiedKFold`.

5. **Balanceo de datos**  
   - **Oversampling** con `SMOTE`.
   - **Undersampling** con `NearMiss` y `RandomUnderSampler`.
   - Integración de balanceo en pipelines con `imbpipeline`.

6. **Evaluación de modelos**  
   - Métricas: Accuracy, Precision, Recall, F1, ROC AUC.
   - Matrices de confusión.
   - Comparación de modelos balanceados y sin balancear.

7. **Selección de modelo campeón**  
   - Comparación de Árbol con SMOTE vs KNN con Undersampling.
   - Guardado de modelo y transformador (`pickle`).

8. **Predicciones con nuevos clientes**  
   - Ejemplo de predicción con el modelo campeón usando datos ficticios.
   - Probabilidades de churn.

---

## 🛠️ Requerimientos técnicos

### 📦 Dependencias principales
- **Python 3.9+**
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [plotly](https://plotly.com/python/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [imbalanced-learn](https://imbalanced-learn.org/stable/)

Instalación rápida:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
