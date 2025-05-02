# Actividad Cloud Computing - Aprendizaje Federado

Este repositorio contiene el desarrollo de una práctica de **Federated Learning** utilizando el dataset **MNIST**, con la implementación y comparación de tres estrategias de agregación de modelos: **FedAvg**, **FedProx** y **FedOpt**.

## Objetivo

Simular un entorno federado en el que múltiples clientes entrenan localmente un modelo CNN con datos distintos, exportan sus modelos `.keras`, y un servidor central agrega los pesos para construir un modelo global.

Se busca comparar el rendimiento de las estrategias de agregación midiendo el accuracy con el conjunto de prueba.

## Estructura del Equipo

El proyecto fue dividido en los siguientes módulos:

### Entrenamiento Local

- Archivo: `train_and_export.py`
- Funciones:
  - Recibe como argumento un archivo `.npz` con datos locales.
  - Entrena un modelo CNN (`ClassificationModelMNIST`) y lo guarda como `.keras` en la carpeta `local_models/`.

### Agregación Central (Federated Server)

- Archivo: `global_model.py`
- Funciones:
  - Carga los modelos locales guardados.
  - Aplica FedAvg, FedProx y FedOpt para combinar los pesos.
  - Evalúa cada modelo global y compara su accuracy.

### Arquitectura del Modelo

- Archivo: `model.py`
- Clase: `ClassificationModelMNIST`
- Descripción:
  - CNN funcional con capas `Conv2D`, `MaxPool`, `Dense`, y `Dropout`.
  - Métodos personalizados para compilación, entrenamiento, validación y visualización.

## Tecnologías Utilizadas

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- matplotlib

## Estructura del Proyecto

```Directory structure:
└──federated_learning/
    ├── README.md
    ├── model.py
    ├── train_and_export.py
    ├── global_model.py
    ├── pyproject.toml 
    ├── uv.lock
    ├── .python-version       
    └── local_models/ 
        ├── local_model_Alvarado.keras
        ├── local_model_Axl.keras
        ├── local_model_Bernardo.keras
        ├── local_model_Bimbo.keras
        ├── local_model_Majeee.keras
        └──local_model_Rodolfo.keras
```

## Pasos para ejecutar el proyecto

1. **Clonar el repositorio.**  
   Haz un fork y clónalo en tu máquina local:

   ```bash
   git clone https://github.com/tu-usuario/axl2301-federated_learning.git
   cd federated_learning
    ```
2. **Configurar entorno virtual e instalar dependencias**
Asegúrate de tener Python instalado, luego corre:

```bash
uv venv
uv source/bin/activate
uv sync
```

3. **Entrena los modelos locales**

Corriendo en terminal el siguiente comando por cada cliente (dataset local), generará el archivo .keras con el modelo dentro de la carpeta local_models:

```bash
python train_and_export.py --data local_dataset_cliente_1.npz
```
4. **Comparar los métodos de agregación de pesos**

```bash
python global_model.py
```
Ejemplo de salida

Modelo: FedAvg, Accuracy: 0.0974

Modelo: FedProx, Accuracy: 0.0974

Modelo: FedOpt, Accuracy: 0.1011
