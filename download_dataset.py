import os
import subprocess
import sys

def download_data():
    dataset_name = "ranadeep/credit-risk-dataset"
    target_dir = os.path.join("data", "loan")
    
    print(f"Preparando la descarga del dataset '{dataset_name}'...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Importar kaggle requiere que el token ~/.kaggle/kaggle.json este configurado
        import kaggle
        print("Autenticacion con Kaggle exitosa. Descargando archivos...")
        kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        print(f"¡Descarga y extraccion completada con exito en '{target_dir}/loan.csv'!")
        
    except ImportError:
        print("Error: No tienes instalada la libreria 'kaggle'. Instalala usando: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"\nOcurrio un error al intentar descargar:")
        print(f"{e}")
        print("\n=== RECUERDA ===")
        print("Debes tener tu archivo 'kaggle.json' en la ruta:")
        print(" - Windows: C:\\Users\\<TuUsuario>\\.kaggle\\kaggle.json")
        print(" - Mac/Linux: ~/.kaggle/kaggle.json\n")
        print("Tambien puedes descargarlo y descomprimirlo manualmente desde:")
        print("https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset")
        sys.exit(1)

if __name__ == "__main__":
    download_data()
