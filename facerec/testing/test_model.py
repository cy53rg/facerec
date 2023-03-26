import os
import sys
import unittest

# Add the parent directory to sys.path to allow importing of modules from other directories.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the models to be tested
from models.cnn_model import CNNModel
from models.knn_model import KNNModel
from models.svm_model import SVMModel

# Define a class for testing the models
class TestModels(unittest.TestCase):

    # Test the CNNModel
    def test_cnn_model(self):
        # Instantiate the model
        model = CNNModel()
        # Assert that the model was instantiated successfully
        self.assertIsNotNone(model)

    # Test the KNNModel
    def test_knn_model(self):
        # Instantiate the model
        model = KNNModel()
        # Assert that the model was instantiated successfully
        self.assertIsNotNone(model)

    # Test the SVMModel
    def test_svm_model(self):
        # Instantiate the model
        model = SVMModel()
        # Assert that the model was instantiated successfully
        self.assertIsNotNone(model)

# Run the tests if this file is executed directly
if __name__ == '__main__':
    unittest.main()
