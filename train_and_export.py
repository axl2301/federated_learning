import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import ClassificationModelMNIST

def main(dataset_path):
    # Cargar datos
    data = np.load(dataset_path)
    x_data = data['x']
    y_data = data['y']
    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7)
    # Llamar al modelo
    model = ClassificationModelMNIST()
    model.build_model()
    model.compile()
    # Entrenamiento
    model.train_model(x_train, y_train, x_test, y_test)
    # Validación
    model.validate_model(x_test, y_test)
    # History plot
    model.plot_history()
    # Guardar pesos
    output_path = os.path.join("local_models", dataset_path.replace(".npz", ".keras"))
    model.model.save(output_path)


"""
Para correr en terminal y automáticamente se guarde el modelo .keras 
dentro de la carpeta local_models
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Ruta al archivo .npz")
    args = parser.parse_args()
    main(args.data)

