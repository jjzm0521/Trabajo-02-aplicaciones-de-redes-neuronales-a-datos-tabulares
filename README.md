# Modelo de Riesgo de Crédito — Probabilidad de Incumplimiento

Proyecto del curso de **Redes Neuronales Artificiales** que desarrolla un modelo para predecir la probabilidad de incumplimiento crediticio, utilizando el [Credit Risk Dataset](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data) de LendingClub.

## Objetivo

Crear y validar un modelo de clasificación binaria basado en redes neuronales artificiales que estime la probabilidad de que un prestatario no cumpla con el pago de su crédito. Se complementa con una scorecard interpretable y una aplicación web interactiva.

## Estructura del repositorio

```
├── notebooks/
│   ├── 01_carga_y_eda.ipynb          # Carga del dataset, recodificación y EDA
│   ├── 02_preprocesamiento.ipynb     # Limpieza, selección de variables (IV/WOE), splits
│   ├── 03_modelamiento.ipynb         # Regresión logística + redes neuronales
│   └── 04_scorecard.ipynb            # Scorecard y análisis de variables de riesgo
│
├── app/
│   └── app.py                        # Aplicación web (Streamlit)
│
├── data/                             # Dataset (no incluido — ver data/README.md)
├── figures/                          # Gráficas generadas por los notebooks
├── outputs/                          # Splits, resultados y probabilidades
├── docs/                             # Enunciado del trabajo
│
├── requirements.txt                  # Dependencias de Python
└── README.md
```

## Pipeline de análisis

### Notebook 01 — Carga y EDA

- Carga del dataset original (`loan.csv`, ~880k registros, 74 variables).
- Recodificación de `loan_status` en variable binaria `default` (0 = buen pagador, 1 = mal pagador), excluyendo categorías con resultado incierto.
- Análisis exploratorio: distribución de clases, histogramas, boxplots, tasa de incumplimiento por variable categórica, mapa de correlaciones y análisis de valores faltantes.
- Guardado en formato Parquet.

### Notebook 02 — Preprocesamiento

- Eliminación de variables por: identificadores, alto porcentaje de nulos (>40%), varianza cero, alta cardinalidad, y **data leakage** (variables que contienen información posterior al evento de incumplimiento).
- Conversión de fechas a escala numérica lineal.
- Imputación de valores faltantes (mediana / moda).
- Selección de variables mediante **Information Value** (IV ≥ 0.02).
- Transformación **WOE** (Weight of Evidence).
- Manejo del desbalance de clases (SMOTE / UnderSampling).
- División estratificada: 70% train / 15% validación / 15% test.

### Notebook 03 — Modelamiento

- **Modelo de referencia:** Regresión Logística (línea base).
- **Redes Neuronales Artificiales** con PyTorch:
  - Modelo A: 2 capas ocultas (64 → 32), dropout 0.3
  - Modelo B: 3 capas ocultas (64 → 32 → 16), dropout 0.3
  - Modelo C: 3 capas ocultas (128 → 64 → 32), dropout 0.4
- Entrenamiento con Adam, BCE Loss y early stopping.
- Comparación de los 4 modelos sobre validación (AUC-ROC, F1-Score).
- Evaluación final del mejor modelo sobre el conjunto de prueba.

### Notebook 04 — Scorecard

- Conversión de probabilidades a puntaje crediticio (300–850 pts) con PDO = 20.
- Análisis de variables de riesgo mediante Information Value y WOE.
- Segmentación en 5 bandas de riesgo (A–E).
- Caso de uso: cómo un individuo puede conocer su score y compararse con la población.

## Aplicación web

La aplicación web permite a los usuarios ingresar sus características y obtener:
- Su **score crediticio** en la escala 300–850.
- Su **probabilidad estimada de incumplimiento**.
- Su **banda de riesgo** (A–E).
- Una **comparación visual** con la distribución de la población.

### Ejecutar la aplicación

```bash
cd app
streamlit run app.py
```

## Instalación y ejecución

### Requisitos previos
- Python 3.10+
- pip

### Configuración

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/Trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares.git
cd Trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares

# Instalar dependencias
pip install -r requirements.txt

# Descargar el dataset (requiere kaggle configurado)
python download_dataset.py
```

### Ejecución de notebooks

Los notebooks deben ejecutarse en orden:

```
01_carga_y_eda.ipynb → 02_preprocesamiento.ipynb → 03_modelamiento.ipynb → 04_scorecard.ipynb
```

Se pueden ejecutar desde Jupyter Notebook, JupyterLab o Google Colab.

## Dataset

**Credit Risk Dataset** — LendingClub  
Disponible en: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data

> El dataset no se incluye en el repositorio por su tamaño. Consultar `data/README.md` para instrucciones de descarga.

## Referencias

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). *Credit Scoring and its Applications*. SIAM.
- Baesens, B., Roesch, D., & Scheule, H. (2016). *Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS*. Wiley.
