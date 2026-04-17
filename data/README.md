# Datos

## Obtención del dataset

El dataset utilizado es **Credit Risk Dataset** de Kaggle.

**Enlace:** https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data

### Opción 1 — Descarga manual

1. Ir al enlace anterior e iniciar sesión en Kaggle.
2. Descargar el archivo y descomprimirlo.
3. Colocar el archivo `loan.csv` dentro de la carpeta `data/loan/`.

### Opción 2 — Descarga por API

```bash
pip install kaggle
kaggle datasets download -d ranadeep/credit-risk-dataset -p data/ --unzip
```

> **Nota:** Para usar la API de Kaggle se necesita un archivo `kaggle.json` con las credenciales. Se obtiene desde [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token.

## Estructura esperada

```
data/
├── loan/
│   └── loan.csv          # Dataset original (~150 MB)
├── loan_recodificado.parquet   # Generado por Notebook 01
└── loan_procesado.parquet      # Generado por Notebook 02
```

Los archivos `.parquet` se generan al ejecutar los notebooks en orden.
