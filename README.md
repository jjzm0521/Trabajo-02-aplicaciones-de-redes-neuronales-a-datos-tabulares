<div align="center">
  <h1>💳 Modelo de Riesgo de Crédito — CreditScore</h1>
  <p><em>Predicción de probabilidad de incumplimiento mediante Redes Neuronales Artificiales</em></p>

  <!-- Badges -->
  <a href="https://trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares.streamlit.app/">
    <img src="https://img.shields.io/badge/Aplicación_Web-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="App Web"/>
  </a>
  <a href="https://jjzm0521.github.io/Trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares/reporte.html">
    <img src="https://img.shields.io/badge/Reporte_Técnico-Publicado-0F172A?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Reporte Técnico"/>
  </a>
  <a href="https://youtube.com/shorts/lDyEkjEELtQ">
    <img src="https://img.shields.io/badge/Video_Promocional-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Video"/>
  </a>
</div>

<br>

Proyecto desarrollado para el curso de **Redes Neuronales Artificiales**, que aborda un problema crítico en la industria financiera: predecir si un prestatario incumplirá sus pagos (probabilidad de impago o *default*). El modelo se basa en el [Credit Risk Dataset](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data) histórico de LendingClub.

## 🎯 Objetivo y Misión

Crear y validar un modelo de clasificación binaria altamente interpretable utilizando una Red Neuronal Artificial en **PyTorch**, estandarizado sobre factores de escala *Weight of Evidence (WOE)*. Este modelo alimenta una Scorecard clásica y se encuentra desplegado en una aplicación web interactiva orientada al usuario final, con un análisis de métricas sobre qué atributos hacen más riesgoso o confiable a un perfil crediticio.

---

## 🚀 Entregables Públicos

1. **[🖥️ Aplicación Web Interactiva](https://trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares.streamlit.app/)**: (⚠️ *Reemplazar con el link generado en Streamlit Cloud*). Permite a individuos simular su riesgo ingresando 4 datos básicos, visualizando su banda de riesgo y el impacto individual de cada atributo. 
2. **[📄 Reporte Técnico / Blog Post](https://jjzm0521.github.io/Trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares/docs/reporte.html)**: Documentación de la metodología, hipótesis, arquitecturas y lecciones aprendidas (Disponible una vez activo GitHub Pages).
3. **[🎬 Video Promocional](https://youtube.com/shorts/lDyEkjEELtQ)**: Explicación entusiasta y presentación del caso de éxito comercial y las contribuciones individuales.

---

## 🛠️ Instalación y Replicación (Desarrollo Local)

Para correr la aplicación o las libretas de experimentación en tu entorno local:

### 1. Requisitos Previos
- Python 3.10 o superior.

### 2. Configuración y Dependencias

```bash
# Clonar el proyecto
git clone https://github.com/jjzm0521/Trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares.git
cd Trabajo-02-aplicaciones-de-redes-neuronales-a-datos-tabulares

# Instalar los paquetes requeridos (PyTorch, Streamlit, SHAP, etc.)
pip install -r requirements.txt

# Descargar la base de datos (Requiere credenciales de Kaggle)
python download_dataset.py
```

### 3. Lanzar la Aplicación Web

La aplicación utilizará los pesos pre-entrenados del modelo ya existentes (`models/modelo_final.pt`).

```bash
streamlit run app/app.py
```

---

## 📂 Arquitectura del Repositorio

El pipeline end-to-end se encuentra estructurado en libretas independientes:

```text
├── notebooks/
│   ├── 01_carga_y_eda.ipynb          # Cargue, imputaciones base y EDA.
│   ├── 02_preprocesamiento.ipynb     # Limpieza WOE/IV y manejo de data leakage.
│   ├── 03_modelamiento.ipynb         # Redes Neuronales (Grid y Dropout) vs Regresión base.
│   └── 04_scorecard.ipynb            # Escalamiento PDO, calibración y bandas de riesgo.
│
├── app/
│   └── app.py                        # App en Streamlit orientada a negocio.
├── docs/                             # Reporte y dependencias de enlaces web.
├── outputs/                          # Resultados experimentales simulados (distribuciones reales).
└── models/                           # Pesos de Deep Learning (.pt) listos para inferencia.
```

## 📚 Bibliografía y Referencias Clave

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). *Credit Scoring and its Applications*. SIAM.
- Baesens, B., Roesch, D., & Scheule, H. (2016). *Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS*. Wiley.
