import unittest
import numpy as np
from scripts.complete_mlflow_pipeline import MLflowPotateoPipeline

class TestMLflowDataset(unittest.TestCase):
    def setUp(self):
        self.pipeline = MLflowPotateoPipeline()
        self.X, self.y, self.paths = self.pipeline.preprocess_data()

    def test_create_mlflow_dataset(self):
        dataset = self.pipeline.create_mlflow_dataset(self.X, self.y, self.paths)
        self.assertIsNotNone(dataset)
        self.assertTrue(hasattr(dataset, 'name'))

if __name__ == "__main__":
    unittest.main()
