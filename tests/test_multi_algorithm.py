"""Unit and integration test suite for the Multi-Algorithm Capstone selector."""
from __future__ import annotations

import unittest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.features import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    extract_single_comment_features,
)
from src.moderation_policy import ModerationDecision, ModerationPolicy

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "classifiers_registry.joblib"
CLASSIFIERS_DIR = ROOT / "models" / "capstone_classifiers"

RUBRIC_NAMES = [
    # Part A
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Gaussian Naive Bayes",
    "Decision Tree Classifier",
    "Support Vector Classifier (SVC)",
    # Part B
    "Random Forest Classifier",
    "AdaBoost Classifier",
    "Gradient Boosting Classifier",
    "Bagging (Decision Tree)",
    "MLP Classifier (Neural Network)",
]


class TestMultiAlgorithmPlayground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assertTrue(REGISTRY_PATH.exists(), f"Registry missing at {REGISTRY_PATH}")
        cls.registry = joblib.load(REGISTRY_PATH)
        cls.policy = ModerationPolicy()
        cls.surface_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    def test_01_all_ten_models_registered(self):
        """Verify that all 10 Capstone Rubric models are present in the registry."""
        for name in RUBRIC_NAMES:
            self.assertIn(name, self.registry, f"Model '{name}' not found in registry")
            meta = self.registry[name]
            self.assertIn("artifact_path", meta)
            self.assertIn("track", meta)
            self.assertIn("supports_proba", meta)
            self.assertIn("metrics", meta)

    def test_02_all_ten_model_artifacts_loadable(self):
        """Verify that all 10 model artifact files exist on disk and load correctly."""
        for name in RUBRIC_NAMES:
            rel_path = self.registry[name]["artifact_path"]
            abs_path = ROOT / rel_path
            self.assertTrue(abs_path.exists(), f"Artifact file does not exist: {abs_path}")
            payload = joblib.load(abs_path)
            self.assertIn("pipeline", payload)
            self.assertIn("feature_columns", payload)
            self.assertEqual(payload["model_name"], name)

    def test_03_inference_execution_on_all_ten_models(self):
        """Verify that all 10 models execute inference without errors on real test comments."""
        test_comment = "This is a serious policy critique regarding healthcare expenditure."
        feats = extract_single_comment_features(test_comment, "Morning", self.surface_features)

        for name in RUBRIC_NAMES:
            rel_path = self.registry[name]["artifact_path"]
            payload = joblib.load(ROOT / rel_path)
            pipeline = payload["pipeline"]

            decision = self.policy.evaluate(
                pipeline=pipeline,
                features_df=feats,
                raw_text=test_comment,
                algorithm_name=name,
                algorithm_track=payload.get("track", ""),
            )

            self.assertIsInstance(decision, ModerationDecision)
            self.assertIn(decision.decision_action, ["APPROVED", "HELD FOR REVIEW", "QUARANTINED"])
            self.assertIn(decision.predicted_class, [0, 1])

            if payload.get("supports_proba", True):
                self.assertIsNotNone(decision.toxic_probability)
                self.assertIsNotNone(decision.civil_probability)
                self.assertTrue(0.0 <= decision.toxic_probability <= 1.0)
                self.assertTrue(0.0 <= decision.civil_probability <= 1.0)
            else:
                self.assertFalse(decision.supports_probability)
                self.assertIsNotNone(decision.decision_score)

    def test_04_model_specific_predictions_are_distinct(self):
        """Verify that different models actually generate their own distinct predictions and probabilities."""
        test_comment = "CONGRATULATIONS ON PASSING YOUR EXAMS BROTHER I AM SO PROUD OF YOU!!!!"
        feats = extract_single_comment_features(test_comment, "Evening", self.surface_features)

        results = {}
        for name in RUBRIC_NAMES:
            payload = joblib.load(ROOT / self.registry[name]["artifact_path"])
            pipe = payload["pipeline"]
            if payload.get("supports_proba", True):
                p_toxic = pipe.predict_proba(feats)[0][1]
            else:
                p_toxic = pipe.decision_function(feats)[0]
            results[name] = p_toxic

        # Ensure that models don't all return identical values (confirming distinct models)
        unique_values = set(round(float(v), 4) for v in results.values())
        self.assertGreater(len(unique_values), 3, "Models appear to return identical outputs")

    def test_05_policy_invariants_across_all_models(self):
        """Ensure no model can ever generate a contradictory Approved + Review state."""
        sample_comments = [
            "Mind your fucking Language",
            "You are a disgusting idiot",
            "Have some fresh organic broccoli for dinner",
            "",
        ]
        for c in sample_comments:
            feats = extract_single_comment_features(c, "Night", self.surface_features)
            for name in RUBRIC_NAMES:
                payload = joblib.load(ROOT / self.registry[name]["artifact_path"])
                decision = self.policy.evaluate(payload["pipeline"], feats, c, name)

                if decision.decision_action == "APPROVED":
                    self.assertNotIn("Review", decision.routing_tier)
                    self.assertNotIn("Quarantine", decision.routing_tier)
                elif decision.decision_action == "HELD FOR REVIEW":
                    self.assertNotIn("Approved", decision.routing_tier)
                    self.assertNotIn("Quarantine", decision.routing_tier)
                elif decision.decision_action == "QUARANTINED":
                    self.assertNotIn("Approved", decision.routing_tier)
                    self.assertNotIn("Review", decision.routing_tier)


if __name__ == "__main__":
    unittest.main()
