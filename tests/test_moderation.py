"""Automated regression and verification test suite for content moderation engine."""
from __future__ import annotations

import unittest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.features import (
    HYBRID_FEATURES,
    extract_single_comment_features,
    extract_text_features,
)
from src.moderation_policy import ModerationDecision, ModerationPolicy

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_classifier.joblib"


class TestModerationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load model artifact and instantiate policy."""
        cls.assertTrue(MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}")
        cls.package = joblib.load(MODEL_PATH)
        cls.pipeline = cls.package["pipeline"]
        cls.feature_columns = cls.package["feature_columns"]
        cls.policy = ModerationPolicy(threshold_approve=0.30, threshold_quarantine=0.70)

    def test_01_empty_and_whitespace_input(self):
        """Test empty and whitespace-only inputs are safely held without crash."""
        for empty_val in ["", "   ", "\n\t  \n"]:
            feats = extract_single_comment_features(empty_val, "Morning", self.feature_columns)
            decision = self.policy.evaluate(self.pipeline, feats, empty_val)
            self.assertIsInstance(decision, ModerationDecision)
            self.assertEqual(decision.decision_action, "HELD FOR REVIEW")
            self.assertIn("Empty", decision.routing_tier)
            self.assertEqual(decision.toxic_probability, 0.0)
            self.assertEqual(decision.civil_probability, 1.0)

    def test_02_ordinary_neutral_comments(self):
        """Test ordinary civil comments are approved in Tier 1."""
        neutral_comments = [
            "The weather today has been unusually calm for mid-October.",
            "I respectfully disagree with your conclusion regarding tax policy, though your point on tariffs is well argued.",
            "Can anyone recommend a good introductory book on linear algebra?",
            "Have some fresh organic broccoli and roasted vegetables for dinner tonight.",
        ]
        for comment in neutral_comments:
            feats = extract_single_comment_features(comment, "Afternoon", self.feature_columns)
            decision = self.policy.evaluate(self.pipeline, feats, comment)
            self.assertLess(decision.toxic_probability, 0.30, f"Failed for '{comment}': P(toxic)={decision.toxic_probability}")
            self.assertEqual(decision.decision_action, "APPROVED")
            self.assertEqual(decision.routing_tier, "Tier 1: Instant Auto-Approval")
            self.assertEqual(decision.predicted_class, 0)

    def test_03_severe_toxic_attacks(self):
        """Test severe abusive attacks and insults are quarantined in Tier 3."""
        toxic_comments = [
            "Shut your ugly mouth you disgusting pig and get out of here.",
            "You are a complete idiot, a pathetic loser, and absolute garbage.",
            "Die in a fire you vile scumbag.",
        ]
        for comment in toxic_comments:
            feats = extract_single_comment_features(comment, "Night", self.feature_columns)
            decision = self.policy.evaluate(self.pipeline, feats, comment)
            self.assertGreaterEqual(decision.toxic_probability, 0.70, f"Failed for '{comment}': P(toxic)={decision.toxic_probability}")
            self.assertEqual(decision.decision_action, "QUARANTINED")
            self.assertEqual(decision.routing_tier, "Tier 3: Automated Quarantine")
            self.assertEqual(decision.predicted_class, 1)

    def test_04_ambiguous_and_profane_borderline_comments(self):
        """Test the original bug example and ambiguous comments are routed to Tier 2 and NEVER APPROVED."""
        borderline_comments = [
            "Mind your fucking Language",
            "What the fuck that was hilarious man",
        ]
        for comment in borderline_comments:
            feats = extract_single_comment_features(comment, "Evening", self.feature_columns)
            decision = self.policy.evaluate(self.pipeline, feats, comment)
            # Must fall into borderline window [0.30, 0.70)
            self.assertGreaterEqual(decision.toxic_probability, 0.30, f"Expected borderline score >= 0.30 for '{comment}'")
            self.assertLess(decision.toxic_probability, 0.70, f"Expected borderline score < 0.70 for '{comment}'")
            # CRITICAL INVARIANT: Borderline comments must NEVER be approved!
            self.assertNotEqual(decision.decision_action, "APPROVED", f"Violated invariant for '{comment}'")
            self.assertEqual(decision.decision_action, "HELD FOR REVIEW")
            self.assertEqual(decision.routing_tier, "Tier 2: Secondary Review Queue")

    def test_04b_profanity_used_jokingly_or_colloquially(self):
        """Verify profanity used enthusiastically without hostile target is not falsely escalated to Tier 3."""
        enthusiastic = "Holy shit I cannot believe that actually worked"
        feats = extract_single_comment_features(enthusiastic, "Evening", self.feature_columns)
        decision = self.policy.evaluate(self.pipeline, feats, enthusiastic)
        # Should NOT be classified as severe quarantine (P < 0.70)
        self.assertLess(decision.toxic_probability, 0.70)
        self.assertNotEqual(decision.decision_action, "QUARANTINED")

    def test_05_long_comments_and_unusual_punctuation(self):
        """Test long comments with unusual formatting and punctuation execute robustly."""
        long_comment = (
            "Congratulations to everyone who passed the qualifying examinations today! "
            "It has been a truly remarkable semester with great teamwork and effort. "
        ) * 15
        unusual_punct = "Really?????? Wait..... ARE YOU SERIOUS?!?!? Wow, that's interesting!!!!!!"

        for comment in [long_comment, unusual_punct]:
            feats = extract_single_comment_features(comment, "Evening", self.feature_columns)
            decision = self.policy.evaluate(self.pipeline, feats, comment)
            self.assertIsInstance(decision, ModerationDecision)
            self.assertFalse(np.isnan(decision.toxic_probability))
            self.assertTrue(0.0 <= decision.toxic_probability <= 1.0)

    def test_06_probability_to_label_mapping(self):
        """Verify toxic_probability matches class 1 and civil matches class 0."""
        model = self.pipeline.named_steps["model"]
        classes = list(model.classes_)
        self.assertEqual(classes, [0, 1], f"Expected classes [0, 1], found {classes}")

        feats = extract_single_comment_features("Testing probability distribution", "Morning", self.feature_columns)
        raw_probs = self.pipeline.predict_proba(feats)[0]

        decision = self.policy.evaluate(self.pipeline, feats, "Testing probability distribution")
        self.assertAlmostEqual(decision.toxic_probability, raw_probs[1], places=5)
        self.assertAlmostEqual(decision.civil_probability, raw_probs[0], places=5)
        self.assertAlmostEqual(decision.toxic_probability + decision.civil_probability, 1.0, places=5)

    def test_07_policy_consistency_invariants(self):
        """Formally test that decision_action and routing_tier can NEVER contradict each other."""
        # Test synthetic probability range [0.0 to 1.0]
        test_df = pd.DataFrame([{col: 0.0 for col in self.feature_columns}])
        test_df["comment_text"] = "dummy"
        test_df["time_of_day"] = "Morning"

        class MockPipeline:
            def __init__(self, p_toxic):
                self.classes_ = np.array([0, 1])
                self.p_toxic = p_toxic
            def predict_proba(self, X):
                return np.array([[1.0 - self.p_toxic, self.p_toxic]])
            def predict(self, X):
                return np.array([1 if self.p_toxic >= 0.50 else 0])

        for p_val in np.linspace(0.0, 1.0, 101):
            mock_pipe = MockPipeline(p_val)
            dec = self.policy.evaluate(mock_pipe, test_df, "test string")

            if p_val < 0.30:
                self.assertEqual(dec.decision_action, "APPROVED")
                self.assertEqual(dec.routing_tier, "Tier 1: Instant Auto-Approval")
            elif p_val < 0.70:
                self.assertEqual(dec.decision_action, "HELD FOR REVIEW")
                self.assertEqual(dec.routing_tier, "Tier 2: Secondary Review Queue")
                self.assertFalse("APPROVED" in dec.decision_action)
            else:
                self.assertEqual(dec.decision_action, "QUARANTINED")
                self.assertEqual(dec.routing_tier, "Tier 3: Automated Quarantine")

    def test_08_feature_schema_consistency(self):
        """Verify inference feature dataframe strictly matches training schema."""
        expected_cols = self.package["feature_columns"]
        self.assertEqual(expected_cols, HYBRID_FEATURES)

        sample = extract_single_comment_features("Schema test comment", "Afternoon", expected_cols)
        self.assertEqual(list(sample.columns), expected_cols)
        self.assertEqual(len(sample), 1)
        self.assertFalse(sample.isna().any().any())

    def test_09_missing_model_graceful_handling(self):
        """Test behavior when given invalid model object."""
        class BrokenPipeline:
            pass
        broken = BrokenPipeline()
        sample = extract_single_comment_features("Crash test", "Morning", self.feature_columns)
        # Should raise informative AttributeError or TypeError, not silently return garbage
        with self.assertRaises(Exception):
            self.policy.evaluate(broken, sample, "Crash test")


if __name__ == "__main__":
    unittest.main()
