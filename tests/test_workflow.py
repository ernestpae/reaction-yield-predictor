"""Regression checks for leakage, data contracts, and saved-model inference."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from yield_predictor.data import FEATURES, split_indices, validate_frame
from yield_predictor.experiment import predict, scores, train

DATA = Path(__file__).resolve().parents[1] / "data" / "reaction_data.csv"


class DataTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.read_csv(DATA)

    def test_invalid_values_and_schema_are_rejected(self):
        cases = []
        for col, value in [("yield", 101), ("time", 0), ("temperature", np.inf),
                           ("concentration", np.nan), ("temperature", -300)]:
            frame = self.frame.copy()
            frame.loc[0, col] = value
            cases.append(frame)
        cases += [self.frame.drop(columns="time"), self.frame.assign(unexpected=1),
                  pd.concat([self.frame, self.frame.iloc[:1]])]
        for frame in cases:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(ValueError):
                    validate_frame(frame)

    def test_reordered_columns_are_normalized(self):
        actual = validate_frame(self.frame[FEATURES[::-1]], target=False)
        self.assertEqual(list(actual.columns), FEATURES)

    def test_splits_are_disjoint_and_complete(self):
        for strategy in ["random", "high_temperature"]:
            a, b = split_indices(self.frame, strategy, 42)
            self.assertFalse(set(a) & set(b))
            self.assertEqual(set(a) | set(b), set(range(len(self.frame))))
            np.testing.assert_array_equal(a, split_indices(self.frame, strategy, 42)[0])
            if strategy == "high_temperature":
                self.assertLess(self.frame.iloc[a].temperature.max(), self.frame.iloc[b].temperature.min())

    def test_temperature_ties_do_not_cross_boundary(self):
        frame = self.frame.copy()
        frame["temperature"] = np.repeat([20, 40, 60, 80, 100], 20)
        a, b = split_indices(frame, "high_temperature", 42)
        self.assertLess(frame.iloc[a].temperature.max(), frame.iloc[b].temperature.min())
        frame["temperature"] = 50
        with self.assertRaises(ValueError):
            split_indices(frame, "high_temperature", 42)

    def test_r2_can_be_negative_or_undefined(self):
        self.assertLess(scores(np.array([1, 2, 3]), np.array([9, 9, 9]))["r2"], 0)
        self.assertIsNone(scores(np.ones(3), np.zeros(3))["r2"])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.result = train(DATA, cls.root / "first")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_holdout_labels_cannot_change_selected_model(self):
        frame = pd.read_csv(DATA)
        ids = self.result["test_rows"]
        frame.loc[ids, "yield"] = 100 - frame.loc[ids, "yield"]
        path = self.root / "perturbed.csv"
        frame.to_csv(path, index=False)
        changed = train(path, self.root / "perturbed")
        self.assertEqual(self.result["selected_model"], changed["selected_model"])
        for a, b in zip(self.result["cv_candidates"], changed["cv_candidates"]):
            np.testing.assert_allclose(a["fold_mae"], b["fold_mae"], atol=1e-10)
        self.assertNotAlmostEqual(self.result["holdout"]["mae"], changed["holdout"]["mae"])
        bundle = joblib.load(self.root / "first" / "model.joblib")
        scaler = bundle["model"].named_steps["scale"]
        if scaler != "passthrough":
            np.testing.assert_allclose(scaler.mean_, frame.iloc[self.result["train_rows"]][FEATURES].mean())

    def test_saved_predictions_and_domain_flags(self):
        frame = pd.read_csv(DATA)
        path = self.root / "inputs.csv"
        inputs = frame.iloc[self.result["test_rows"]][FEATURES]
        inputs.to_csv(path, index=False)
        output = predict(self.root / "first" / "model.joblib", path, self.root / "output.csv")
        recorded = pd.read_csv(self.root / "first" / "predictions.csv")
        np.testing.assert_allclose(output.predicted_yield, recorded.predicted_yield)
        inputs.iloc[:1].assign(temperature=1000).to_csv(path, index=False)
        flagged = predict(self.root / "first" / "model.joblib", path, self.root / "flagged.csv")
        self.assertTrue(flagged.outside_training_range.iloc[0])

    def test_report_contains_auditable_selection(self):
        report = json.loads((self.root / "first" / "metrics.json").read_text())
        self.assertEqual(sum(x["selected"] for x in report["cv_candidates"]), 1)
        selected = next(x for x in report["cv_candidates"] if x["selected"])
        self.assertEqual(selected["cv_mae_mean"], min(x["cv_mae_mean"] for x in report["cv_candidates"]))
        self.assertEqual(len(report["dataset_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
