import os
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from model import ClassificationModelMNIST  

"""
Funciones con los distintos métodos para integrar los pesos locales
"""

def fedavg(weights_list):
    """
    Promedia los pesos de los modelos locales.
    """
    return [np.mean(np.array(layer), axis=0) for layer in zip(*weights_list)]

def fedprox(weights_list, global_weights, mu=0.01):
    """
    Promedia los pesos de los modelos locales y agrega un término de regularización.
    """
    prox_weights = []
    for i in range(len(global_weights)):
        layer_avg = np.mean([
            w[i] - mu * (w[i] - global_weights[i])
            for w in weights_list
        ], axis=0)
        prox_weights.append(layer_avg)
    return prox_weights

def fedopt(global_weights, weights_list, lr=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):
    """
    Promedia los pesos de los modelos locales y aplica una optimización adicional.
    En este caso usamos una versión 'manual' de Adam ya que tf.keras.optimizers.Adam
    requiere más cosas que sólo un array de pesos sueltos.
    """
    m = [np.zeros_like(w) for w in global_weights]
    v = [np.zeros_like(w) for w in global_weights]
    grads = [np.mean([w[i] - global_weights[i] for w in weights_list], axis=0) for i in range(len(global_weights))]

    new_weights = []
    for i in range(len(global_weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
        m_hat = m[i] / (1 - beta1)
        v_hat = v[i] / (1 - beta2)
        update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        new_weights.append(global_weights[i] + update)
    return new_weights

# Cargar MNIST (sólo datos de test)
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test / 255.0, -1)

# Cargar modelos locales
local_models_folder = 'local_models'
loaded_local_models = [tf.keras.models.load_model(os.path.join(local_models_folder, file))for file in os.listdir(local_models_folder)if file.endswith('.keras')]

# Extraer pesos de los modelos locales
local_weights = [x.get_weights() for x in loaded_local_models]
averaged_weights = [np.mean(np.array(weights), axis=0) for weights in zip(*local_weights)]

# Para guardar los resultados
results = []

# Evaluar el modelo usando FedAvg
global_model = ClassificationModelMNIST()
global_model.build_model()
global_model.compile()
fedavg_weights = fedavg(local_weights)
global_model.model.set_weights(fedavg_weights)
global_model.validate_model(x_test, y_test)
y_pred = np.argmax(global_model.model.predict(x_test), axis=1)
acc = accuracy_score(y_test, y_pred)
global_model.validate_model(x_test, y_test)
results.append(['FedAvg', acc])

# Evaluar el modelo usando FedProx
global_model = ClassificationModelMNIST()
global_model.build_model()
global_model.compile()
init_weights = global_model.model.get_weights()
fedprox_weights = fedprox(local_weights, init_weights, mu=0.01)
global_model.model.set_weights(fedprox_weights)
y_pred = np.argmax(global_model.model.predict(x_test), axis=1)
acc = accuracy_score(y_test, y_pred)
global_model.validate_model(x_test, y_test)
results.append(['FedProx', acc])

# Evaluar el modelo usando FedOpt
global_model = ClassificationModelMNIST()
global_model.build_model()
global_model.compile()
init_weights = global_model.model.get_weights()
fedopt_weights = fedopt(init_weights, local_weights, lr=0.01)
global_model.model.set_weights(fedopt_weights)
y_pred = np.argmax(global_model.model.predict(x_test), axis=1)
acc = accuracy_score(y_test, y_pred)
global_model.validate_model(x_test, y_test)
results.append(['FedOpt', acc])

# Mostrar comparación final
for result in results:
    print(f"Modelo: {result[0]}, Accuracy: {result[1]:.4f}")

