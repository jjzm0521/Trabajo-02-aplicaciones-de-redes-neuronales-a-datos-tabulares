# Enunciado del Trabajo

## Reto

Crear un modelo para predecir la probabilidad de que un individuo incumpla con el pago de su crédito.

La variable `loan_status` (incumplimiento de las obligaciones financieras) está dada en el archivo **Credit Risk Dataset**, disponible en: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data

### Objetivos

1. Crear y validar un modelo de probabilidad de incumplimiento basado en redes neuronales artificiales. Optimizar la arquitectura del modelo.
2. Representar este modelo con una scorecard.
3. Analizar qué variables hacen más riesgosa a una persona.
4. Crear una app web que permita a las personas, de acuerdo con sus características, conocer su scorecard y cómo se compara con la población.

## Entregables

- **Reporte técnico** publicado como entrada de blog.
- **Sitio web** donde el usuario puede ver cómo se comporta su score en función de sus características.
- **Video promocional** de la aplicación web.

## Criterios de evaluación

### Reporte técnico

- El problema está bien delimitado y se plantea una metodología para resolverlo.
- Se incluye un análisis descriptivo y se generan hipótesis a partir del mismo.
- Se plantean modelos y se evalúa su desempeño con diferentes conjuntos de datos. Se incluye un modelo de baja complejidad como referencia.
- Se incluye un listado de aprendizajes sobre el problema generados por el proceso de modelamiento.
- Se plantea un caso de uso del modelo.
- El reporte sigue las normas APA o las de algún formato de revista científica.
- Las gráficas y tablas están rotuladas y citadas dentro del texto.
- El reporte se apoya en bibliografía relevante correctamente citada.

### Aplicación Web

- El aplicativo es intuitivo y fácil de usar para nuevos usuarios.
- La aplicación resuelve un problema a través de un modelo.
- La aplicación contiene los enlaces al reporte técnico y el material publicitario.

### Video publicitario

- El video presenta la aplicación web como la solución a un problema.
- El video genera entusiasmo por utilizar la aplicación.

## Definición de la variable objetivo

Se crea una variable binaria donde:
- **0** = buen pagador (pagó su crédito completamente)
- **1** = mal pagador (no pagó su crédito)

### Clasificación de categorías

| Categoría original | Codificación | Justificación |
|---|:---:|---|
| `Fully Paid` | 0 | Buen pagador confirmado |
| `Does not meet the credit policy. Status:Fully Paid` | 0 | Buen pagador confirmado |
| `Charged Off` | 1 | Mal pagador confirmado |
| `Late (31-120 days)` | 1 | Se considera mal pagador |
| `Default` | 1 | Impago confirmado |
| `Does not meet the credit policy. Status:Charged Off` | 1 | Mal pagador confirmado |
| `Current` | NA | Comportamiento final desconocido |
| `Issued` | NA | Sin historial de pago |
| `In Grace Period` | NA | Aún no obligado a pagar |
| `Late (16-30 days)` | NA | Resultado incierto |

> Los casos marcados como NA no deben incluirse en el entrenamiento del modelo.

## Referencias útiles

- https://www.listendata.com/2019/08/credit-risk-modelling.html
- https://towardsdatascience.com/how-to-prepare-data-for-credit-risk-modeling-5523641882f2/
- https://towardsdatascience.com/credit-risk-modeling-with-machine-learning-8c8a2657b4c4/
