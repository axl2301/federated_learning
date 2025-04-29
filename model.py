"""Clase para el modelo que será utilizado para clasificar
las imágenes del MNIST."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class ClassificationModelMNIST:
    """Clase para el modelo que será utilizado para clasificar
    las imágenes del MNIST."""
    def __init__(self):
        self.input_size = (28, 28, 1)
        self.output_size = 10

    def build_model(self):
        """Método principal para construir el modelo."""
        inputs = tf.keras.Input(shape=self.input_size)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(self.output_size, activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)

    def compile(self):
        """Compila el modelo."""
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=128):
        """Entrena el modelo con el train set."""
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    
    def validate_model(self, x_test, y_test):
        """Valida el modelo con el test set e imprime métricas usando
        el classification report."""
        pred_labels = self.model.predict(x_test)
        pred_labels = np.argmax(pred_labels, axis=1)
        print(classification_report(y_test, pred_labels))
        
    def plot_history(self):
        """Plotea el historial del accuracy y cross entropy loss de
        entrenamiento y validation para ver si está overfitteado el modelo."""
        hist = self.history.history       # shortcut
        epochs = range(1, len(hist["loss"]) + 1)

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(epochs, hist["loss"], label="train")
        plt.plot(epochs, hist["val_loss"], label="val")
        plt.title("Cross-entropy loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, hist["accuracy"], label="train")
        plt.plot(epochs, hist["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.show()