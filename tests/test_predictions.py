import unittest
from scripts.complete_mlflow_pipeline import MLflowPotateoPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TestPredictions(unittest.TestCase):
    def setUp(self):
        self.pipeline = MLflowPotateoPipeline()
        X, y, _ = self.pipeline.preprocess_data()
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_test = y_test
        self.y_train = y_train

    def test_model_predictions(self):
        for model_name in self.pipeline.model_configs.keys():
            results = self.pipeline.train_single_model(model_name, self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test)
            y_pred = results['predictions']
            self.assertEqual(len(y_pred), len(self.y_test))
            # Check that predictions are within the valid label range
            self.assertTrue(((y_pred >= 0) & (y_pred < len(set(self.y_test)))).all())

if __name__ == "__main__":
    unittest.main()
