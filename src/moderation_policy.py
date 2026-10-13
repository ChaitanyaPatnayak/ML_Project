"""Production moderation policy, decision thresholds, and routing logic.
Ensures single-source-of-truth consistency between probabilities, decisions, and routing tiers.
Supports models with calibrated predict_proba() as well as models with decision_function() only.
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
    toxic_probability: float | None
    civil_probability: float | None
    predicted_class: int
    predicted_label: str
    decision_action: str  # "APPROVED", "HELD FOR REVIEW", "QUARANTINED"
    routing_tier: str     # "Tier 1: Instant Auto-Approval", "Tier 2: Secondary Review Queue", "Tier 3: Quarantine"
    threshold_approve: float
    threshold_quarantine: float
    classification_threshold: float
    uncertainty_score: float  # 0.0 (certain) to 1.0 (maximally uncertain)
    explanation: str
    features: dict[str, Any]
    supports_probability: bool = True
    decision_score: float | None = None
    algorithm_name: str = ""
    algorithm_track: str = ""


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
        algorithm_name: str = "",
        algorithm_track: str = "",
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
                supports_probability=True,
                decision_score=None,
                algorithm_name=algorithm_name,
                algorithm_track=algorithm_track,
            )

        # 1. Resolve model object & classes_
        model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
        classes = list(getattr(model, "classes_", [0, 1]))
        toxic_idx = classes.index(1) if 1 in classes else 1
        civil_idx = classes.index(0) if 0 in classes else 0

        # Check whether pipeline/model supports native predict_proba
        has_proba = hasattr(pipeline, "predict_proba") or hasattr(model, "predict_proba")
        has_decision_fn = hasattr(pipeline, "decision_function") or hasattr(model, "decision_function")

        if has_proba:
            try:
                probs = pipeline.predict_proba(features_df)[0]
                toxic_prob = float(probs[toxic_idx])
                civil_prob = float(probs[civil_idx])
                toxic_prob = float(np.clip(toxic_prob, 0.0, 1.0))
                civil_prob = float(np.clip(civil_prob, 0.0, 1.0))
                supports_probability = True
                decision_score = None

                # Classification label
                predicted_class = 1 if toxic_prob >= self.classification_threshold else 0
                predicted_label = "Toxic" if predicted_class == 1 else "Civil"

                # Uncertainty score: 1.0 at 0.50 (maximum ambiguity), 0.0 at 0.0 or 1.0
                uncertainty = float(np.clip(1.0 - 2.0 * abs(toxic_prob - 0.50), 0.0, 1.0))

                # Enforce Tri-Tier Moderation Decision
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
                        "secondary review."
                    )
                else:
                    action = "QUARANTINED"
                    routing = "Tier 3: Automated Quarantine"
                    explanation = (
                        f"High toxicity confidence ({toxic_prob:.1%} >= {self.threshold_quarantine:.0%}). "
                        "Comment flagged for safety violation and quarantined."
                    )

            except Exception:
                has_proba = False

        if not has_proba:
            # Fallback for models without predict_proba (e.g. Linear SVC with probability=False)
            supports_probability = False
            pred_raw = int(pipeline.predict(features_df)[0])
            predicted_class = pred_raw
            predicted_label = "Toxic" if predicted_class == 1 else "Civil"

            if has_decision_fn:
                decision_score = float(pipeline.decision_function(features_df)[0])
                # In margin distance: 0 is decision boundary, negative is civil, positive is toxic
                # Uncertainty is high when distance is close to 0
                margin_dist = abs(decision_score)
                uncertainty = float(np.clip(1.0 - min(margin_dist, 1.0), 0.0, 1.0))
                toxic_prob = None
                civil_prob = None

                if decision_score < -0.20:
                    action = "APPROVED"
                    routing = "Tier 1: Instant Auto-Approval"
                    explanation = (
                        f"Classifier decision score ({decision_score:.3f}) indicates sample lies comfortably "
                        "in the civil half-space (Margin < -0.20)."
                    )
                elif decision_score <= 0.20:
                    action = "HELD FOR REVIEW"
                    routing = "Tier 2: Secondary Review Queue"
                    explanation = (
                        f"Classifier decision score ({decision_score:.3f}) lies near the separating hyperplane "
                        "[-0.20, +0.20]. Escalated to secondary review."
                    )
                else:
                    action = "QUARANTINED"
                    routing = "Tier 3: Automated Quarantine"
                    explanation = (
                        f"Classifier decision score ({decision_score:.3f}) indicates sample lies in the "
                        "toxic half-space (Margin > +0.20)."
                    )
            else:
                decision_score = None
                toxic_prob = 1.0 if predicted_class == 1 else 0.0
                civil_prob = 1.0 - toxic_prob
                uncertainty = 0.50
                if predicted_class == 0:
                    action = "APPROVED"
                    routing = "Tier 1: Instant Auto-Approval"
                    explanation = "Discrete model predicted Class 0 (Civil)."
                else:
                    action = "QUARANTINED"
                    routing = "Tier 3: Automated Quarantine"
                    explanation = "Discrete model predicted Class 1 (Toxic)."

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
            supports_probability=supports_probability,
            decision_score=decision_score,
            algorithm_name=algorithm_name,
            algorithm_track=algorithm_track,
        )
