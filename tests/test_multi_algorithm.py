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

    def test_06_exact_reproducing_case_kill(self):
        """Audit exact input 'kill' with 'Evening' across Linear SVC and Production Hybrid Model.
        Verifies:
        1. Linear SVC predicts Class 0 (Civil) with negative functional margin w^T x + b ~= -1.0.
        2. SVC does NOT support probabilities (toxic_probability is None).
        3. Under conservative policy, SVC routes 'kill' to HELD FOR REVIEW, NEVER APPROVED.
        4. Production Hybrid model flags 'kill' with Class 1 (Toxic) and P(Toxic) >= 0.70.
        """
        raw_text = "kill"
        tod = "Evening"

        # 1. Test Linear SVC
        svc_payload = joblib.load(ROOT / self.registry["Support Vector Classifier (SVC)"]["artifact_path"])
        svc_cols = svc_payload.get("feature_columns", self.surface_features)
        svc_feats = extract_single_comment_features(raw_text, tod, svc_cols)
        svc_decision = self.policy.evaluate(
            pipeline=svc_payload["pipeline"],
            features_df=svc_feats,
            raw_text=raw_text,
            algorithm_name="Support Vector Classifier (SVC)",
            algorithm_track="Part A",
        )

        self.assertEqual(svc_decision.predicted_class, 0)
        self.assertEqual(svc_decision.predicted_label, "Civil")
        self.assertFalse(svc_decision.supports_probability)
        self.assertIsNone(svc_decision.toxic_probability)
        self.assertIsNone(svc_decision.civil_probability)
        self.assertIsNotNone(svc_decision.decision_score)
        self.assertAlmostEqual(svc_decision.decision_score, -1.0, places=1)
        self.assertEqual(svc_decision.score_type, "raw_functional_margin")
        self.assertEqual(svc_decision.margin_orientation, "positive_toxic_negative_civil")

        # CRITICAL AUDIT FIX: Must NEVER be APPROVED!
        self.assertEqual(svc_decision.decision_action, "HELD FOR REVIEW")
        self.assertIn("Conservative Review Queue", svc_decision.routing_tier)
        self.assertIn("uncalibrated", svc_decision.explanation.lower())

        # 2. Test Production Hybrid Model
        hybrid_path = ROOT / "models" / "final_classifier.joblib"
        hybrid_payload = joblib.load(hybrid_path)
        hybrid_feats = extract_single_comment_features(raw_text, tod, hybrid_payload.get("feature_columns"))
        hybrid_decision = self.policy.evaluate(
            pipeline=hybrid_payload["pipeline"],
            features_df=hybrid_feats,
            raw_text=raw_text,
            algorithm_name="Production Hybrid Model",
            algorithm_track="Production",
        )

        self.assertEqual(hybrid_decision.predicted_class, 1)
        self.assertEqual(hybrid_decision.predicted_label, "Toxic")
        self.assertTrue(hybrid_decision.supports_probability)
        self.assertGreaterEqual(hybrid_decision.toxic_probability, 0.70)
        self.assertEqual(hybrid_decision.decision_action, "QUARANTINED")
        self.assertIn("Quarantine", hybrid_decision.routing_tier)

    def test_07_class_orientation_invariance(self):
        """Verify that decision_function margin correctly accounts for classes_ orientation."""
        test_df = pd.DataFrame([{col: 0.0 for col in self.surface_features}])

        class NormalOrientedModel:
            classes_ = np.array([0, 1])  # Standard: classes_[1] = 1 (Toxic)
            def decision_function(self, X):
                return np.array([1.5])  # Positive margin -> Toxic
            def predict(self, X):
                return np.array([1])

        class InvertedOrientedModel:
            classes_ = np.array([1, 0])  # Inverted: classes_[1] = 0 (Civil), classes_[0] = 1 (Toxic)
            def decision_function(self, X):
                return np.array([-1.5])  # Negative margin in scikit-learn points to classes_[0] = 1 (Toxic)
            def predict(self, X):
                return np.array([1])

        # Normal model: positive score -> Toxic -> QUARANTINED
        dec_normal = self.policy.evaluate(NormalOrientedModel(), test_df, "test text")
        self.assertEqual(dec_normal.predicted_class, 1)
        self.assertEqual(dec_normal.decision_action, "QUARANTINED")
        self.assertEqual(dec_normal.margin_orientation, "positive_toxic_negative_civil")

        # Inverted model: negative score points to classes_[0]=1 -> Toxic -> QUARANTINED
        dec_inv = self.policy.evaluate(InvertedOrientedModel(), test_df, "test text")
        self.assertEqual(dec_inv.predicted_class, 1)
        self.assertEqual(dec_inv.decision_action, "QUARANTINED")
        self.assertEqual(dec_inv.margin_orientation, "positive_civil_negative_toxic")

    def test_08_predicted_class_1_never_approved(self):
        """Formal invariant: If predicted_class is 1, decision_action CAN NEVER BE APPROVED."""
        test_df = pd.DataFrame([{col: 0.0 for col in self.surface_features}])

        class InconsistentModel:
            classes_ = np.array([0, 1])
            def predict_proba(self, X):
                return np.array([[0.85, 0.15]])  # Prob says 15% toxic (< 30%)
            def predict(self, X):
                return np.array([1])  # Discrete predictor claims Toxic!

        dec = self.policy.evaluate(InconsistentModel(), test_df, "contradictory text")
        self.assertEqual(dec.predicted_class, 1)
        self.assertNotEqual(dec.decision_action, "APPROVED", "Model with predicted_class=1 was falsely approved!")
        self.assertEqual(dec.decision_action, "HELD FOR REVIEW")

    def test_09_uncalibrated_negative_margin_never_auto_approved(self):
        """Verify that raw uncalibrated negative margins are never auto-approved."""
        test_df = pd.DataFrame([{col: 0.0 for col in self.surface_features}])

        for negative_margin in [-0.01, -0.20, -0.50, -1.0, -5.0]:
            class MockNegativeMarginModel:
                classes_ = np.array([0, 1])
                def decision_function(self, X, margin=negative_margin):
                    return np.array([margin])
                def predict(self, X):
                    return np.array([0])

            dec = self.policy.evaluate(MockNegativeMarginModel(), test_df, "some text")
            self.assertEqual(dec.predicted_class, 0)
            self.assertNotEqual(dec.decision_action, "APPROVED", f"Negative margin {negative_margin} was falsely auto-approved!")
            self.assertEqual(dec.decision_action, "HELD FOR REVIEW")
            self.assertIn("Conservative Review Queue", dec.routing_tier)

    def test_10_score_type_contract(self):
        """Verify score_type correctly documents calibrated_probability vs raw_functional_margin."""
        test_comment = "Ordinary test comment."
        feats = extract_single_comment_features(test_comment, "Morning", self.surface_features)

        # Probabilistic model
        lr_payload = joblib.load(ROOT / self.registry["Logistic Regression"]["artifact_path"])
        dec_lr = self.policy.evaluate(lr_payload["pipeline"], feats, test_comment)
        self.assertEqual(dec_lr.score_type, "calibrated_probability")

        # Non-probabilistic model
        svc_payload = joblib.load(ROOT / self.registry["Support Vector Classifier (SVC)"]["artifact_path"])
        dec_svc = self.policy.evaluate(svc_payload["pipeline"], feats, test_comment)
        self.assertEqual(dec_svc.score_type, "raw_functional_margin")


if __name__ == "__main__":
    unittest.main()
