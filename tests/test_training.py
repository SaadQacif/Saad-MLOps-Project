import unittest
from scripts.complete_mlflow_pipeline import MLflowPotateoPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.pipeline = MLflowPotateoPipeline()
        X, y, _ = self.pipeline.preprocess_data()
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def test_train_single_model(self):
        for model_name in self.pipeline.model_configs.keys():
            results = self.pipeline.train_single_model(model_name, self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test)
            self.assertIn('accuracy', results)
            self.assertIn('model', results)
            self.assertGreaterEqual(results['accuracy'], 0)
            self.assertLessEqual(results['accuracy'], 1)

if __name__ == "__main__":
    unittest.main()
