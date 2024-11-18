import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

class TestSmallNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        random_seed = 42
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        random.seed(random_seed)

        # Define model architecture with specific neurons
        self.model = keras.Sequential([
            layers.Dense(2, activation='relu', input_shape=(1,)),
            layers.Dense(1)
        ])

        # Compile the model with a specific optimizer and loss function
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Prepare specific training inputs and outputs
        self.X_train = np.array([[0], [1], [2], [3], [4]], dtype=float)
        self.y_train = np.array([[1], [3], [5], [7], [9]], dtype=float)

        # Define specific number of epochs for training
        self.epochs = 500

    def test_training_and_inference(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, verbose=0)

        # Perform inference with the trained model
        X_test = np.array([[5]], dtype=float)
        y_pred = self.model.predict(X_test)

        # Expected output for the test input
        expected_output = np.array([[11]], dtype=float)

        # Assert the inference result is within a small tolerance
        np.testing.assert_allclose(y_pred, expected_output, rtol=0.025)

if __name__ == '__main__':
    unittest.main()
