import unittest
import numpy as np
from scripts.complete_mlflow_pipeline import MLflowPotateoPipeline

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.pipeline = MLflowPotateoPipeline()

    def test_preprocess_data(self):
        X, y, paths = self.pipeline.preprocess_data()
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(X), 0)
        self.assertEqual(X.shape[0], y.shape[0])

if __name__ == "__main__":
    unittest.main()
