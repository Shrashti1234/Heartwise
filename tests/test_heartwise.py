import unittest
import pandas as pd
import numpy as np
from src.heartwise import normalize_target, prepare_dataset, train_models, make_patient_row


class HeartWiseCoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 120
        self.data = pd.DataFrame({
            "Age": rng.integers(25, 80, n),
            "Sex": rng.choice(["M", "F"], n),
            "ChestPainType": rng.choice(["ATA", "NAP", "ASY"], n),
            "RestingBP": rng.integers(90, 190, n),
            "Cholesterol": rng.integers(120, 350, n),
            "MaxHR": rng.integers(70, 200, n),
        })
        self.data["HeartDisease"] = ((self.data["Age"] > 55) | (self.data["ChestPainType"] == "ASY")).astype(int)

    def test_normalize_binary_labels(self):
        pd.testing.assert_series_equal(normalize_target(pd.Series(["no", "yes"])), pd.Series([0, 1]))

    def test_missing_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "HeartDisease"):
            prepare_dataset(self.data.drop(columns=["HeartDisease"]))

    def test_pipeline_trains_and_predicts(self):
        X, y, _ = prepare_dataset(self.data)
        models, metrics, X_test, y_test = train_models(X, y)
        self.assertEqual(len(metrics), 3)
        self.assertIn("F1", metrics.columns)
        row = make_patient_row(X, {c: X[c].iloc[0] for c in X.columns})
        probability = models[metrics.iloc[0]["Model"]].predict_proba(row)[0, 1]
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
