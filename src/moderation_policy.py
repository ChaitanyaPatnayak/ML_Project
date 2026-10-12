"""Production moderation policy, decision thresholds, and routing logic.
Ensures single-source-of-truth consistency between probabilities, decisions, and routing tiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModerationDecision:
    """Immutable data record representing the complete moderation evaluation."""
    raw_text: str
    toxic_probability: float
    civil_probability: float
    predicted_class: int
    predicted_label: str
    decision_action: str  # "APPROVED", "HELD FOR REVIEW", "QUARANTINED"
    routing_tier: str     # "Tier 1: Instant Auto-Approval", "Tier 2: Secondary Review Queue", "Tier 3: Quarantine"
    threshold_approve: float
    threshold_quarantine: float
    classification_threshold: float
    uncertainty_score: float  # 0.0 (certain) to 1.0 (maximally uncertain near 0.50)
    explanation: str
    features: dict[str, Any]


class ModerationPolicy:
    """Documented tri-tier moderation policy with configurable, validation-driven thresholds."""

    def __init__(
        self,
        threshold_approve: float = 0.30,
        threshold_quarantine: float = 0.70,
        classification_threshold: float = 0.50,
    ) -> None:
        if not (0.0 < threshold_approve < classification_threshold < threshold_quarantine < 1.0):
            raise ValueError("Thresholds must strictly satisfy 0 < approve < classification < quarantine < 1")
        self.threshold_approve = threshold_approve
        self.threshold_quarantine = threshold_quarantine
        self.classification_threshold = classification_threshold

    def evaluate(
        self,
        pipeline: Any,
        features_df: pd.DataFrame,
        raw_text: str,
    ) -> ModerationDecision:
        """Evaluate a comment through model inference and enforce strict decision policy consistency."""
        raw_text_clean = (raw_text or "").strip()

        # Handle empty/whitespace input gracefully
        if not raw_text_clean:
            return ModerationDecision(
                raw_text="",
                toxic_probability=0.0,
                civil_probability=1.0,
                predicted_class=0,
                predicted_label="Civil (Empty)",
                decision_action="HELD FOR REVIEW",
                routing_tier="Tier 2: Empty Input Validation",
                threshold_approve=self.threshold_approve,
                threshold_quarantine=self.threshold_quarantine,
                classification_threshold=self.classification_threshold,
                uncertainty_score=0.0,
                explanation="Comment contains no text content. Blocked from publication pending valid input.",
                features={},
            )

        # 1. Resolve classes_ from model
        model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
        classes = list(getattr(model, "classes_", [0, 1]))
        toxic_idx = classes.index(1) if 1 in classes else 1
        civil_idx = classes.index(0) if 0 in classes else 0

        # 2. Compute probabilities
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(features_df)[0]
            toxic_prob = float(probs[toxic_idx])
            civil_prob = float(probs[civil_idx])
        else:
            # Fallback for models with decision_function or predict only
            pred_raw = pipeline.predict(features_df)[0]
            toxic_prob = 1.0 if pred_raw == 1 else 0.0
            civil_prob = 1.0 - toxic_prob

        # Ensure probability values are strictly bounded
        toxic_prob = float(np.clip(toxic_prob, 0.0, 1.0))
        civil_prob = float(np.clip(civil_prob, 0.0, 1.0))

        # 3. Model classification label (at standard 0.50 threshold)
        predicted_class = 1 if toxic_prob >= self.classification_threshold else 0
        predicted_label = "Toxic" if predicted_class == 1 else "Civil"

        # 4. Uncertainty score: 1.0 at 0.50 (maximum ambiguity), 0.0 at 0.0 or 1.0 (maximum certainty)
        uncertainty = float(1.0 - 2.0 * abs(toxic_prob - 0.50))
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

        # 5. Enforce single-source-of-truth Tri-Tier Moderation Decision
        if toxic_prob < self.threshold_approve:
            action = "APPROVED"
            routing = "Tier 1: Instant Auto-Approval"
            explanation = (
                f"Toxicity risk is low ({toxic_prob:.1%} < {self.threshold_approve:.0%}). "
                "Safe for immediate real-time publication."
            )
        elif toxic_prob < self.threshold_quarantine:
            action = "HELD FOR REVIEW"
            routing = "Tier 2: Secondary Review Queue"
            explanation = (
                f"Toxicity score ({toxic_prob:.1%}) lies in the ambiguous borderline window "
                f"[{self.threshold_approve:.0%}, {self.threshold_quarantine:.0%}]. "
                "The system will NOT approve this comment automatically; it is routed to "
                "secondary asynchronous human / LLM moderation."
            )
        else:
            action = "QUARANTINED"
            routing = "Tier 3: Automated Quarantine"
            explanation = (
                f"High toxicity confidence ({toxic_prob:.1%} >= {self.threshold_quarantine:.0%}). "
                "Comment flagged for safety violation and quarantined."
            )

        features_dict = features_df.to_dict(orient="records")[0] if len(features_df) > 0 else {}

        return ModerationDecision(
            raw_text=raw_text,
            toxic_probability=toxic_prob,
            civil_probability=civil_prob,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            decision_action=action,
            routing_tier=routing,
            threshold_approve=self.threshold_approve,
            threshold_quarantine=self.threshold_quarantine,
            classification_threshold=self.classification_threshold,
            uncertainty_score=uncertainty,
            explanation=explanation,
            features=features_dict,
        )
