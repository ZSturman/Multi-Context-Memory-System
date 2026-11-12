from __future__ import annotations

from .config import get_cfg
from .formulas import clamp, token_overlap, weighted_score
from .models import ActiveContext, EventSnapshot, GateAction, GateDecision, RecallCandidate, RecallState, WorkingMemorySnapshot


class ThoughtGate:
    def evaluate(
        self,
        candidate: RecallCandidate,
        active_context: ActiveContext,
        event_view: EventSnapshot,
        working_memory: WorkingMemorySnapshot,
        config: dict,
    ) -> GateDecision:
        context_fit = clamp((candidate.context_fit + token_overlap(candidate.context_labels, active_context.context_labels)) / 2.0)
        explanatory_value = clamp((candidate.cue_similarity + (1.0 - event_view.global_uncertainty) + candidate.event_activation) / 3.0)
        predictive_value = clamp((candidate.graph_activation + candidate.goal_relevance + candidate.memory_strength) / 3.0)
        uncertainty_reduction = clamp(event_view.global_uncertainty * explanatory_value)
        novelty_contribution = clamp(1.0 - token_overlap(candidate.event_summary.split("_"), event_view.event_hypothesis.split("_")))
        emotional_distortion_risk = clamp(max(0.0, candidate.arousal_score - context_fit))
        intrusive_penalty = candidate.regulation.get("reconsolidation_pressure", 0.0) if candidate.recurrent_mismatch else 0.0
        score = weighted_score(
            {
                "cue_similarity": candidate.cue_similarity,
                "contextual_fit": context_fit,
                "goal_relevance": candidate.goal_relevance,
                "explanatory_value": explanatory_value,
                "predictive_value": predictive_value,
                "uncertainty_reduction": uncertainty_reduction,
                "novelty_contribution": novelty_contribution,
                "emotional_distortion_penalty": emotional_distortion_risk,
                "intrusive_penalty": intrusive_penalty,
                "already_integrated_penalty": 1.0 if candidate.already_integrated else 0.0,
                "already_dismissed_penalty": 1.0 if candidate.already_dismissed else 0.0,
            },
            get_cfg(config, "weights", "thought_gate", default={}),
        )
        if candidate.already_integrated and score < 0.9:
            action = GateAction.TEMP_BLOCK
            state = RecallState.BLOCKED
            reason = "already_integrated"
        elif candidate.already_dismissed and context_fit < 0.45 and candidate.cue_similarity > 0.25:
            action = GateAction.RECONSOLIDATE
            state = RecallState.RECONSOLIDATION_CANDIDATE
            reason = "repeated_poor_fit_after_dismissal"
        elif (candidate.recurrent_mismatch or emotional_distortion_risk > 0.6) and score <= get_cfg(config, "thresholds", "gate_reconsolidate", default=0.52):
            action = GateAction.RECONSOLIDATE
            state = RecallState.RECONSOLIDATION_CANDIDATE
            reason = "recurrent_intrusive_or_poor_fit"
        elif candidate.cue_similarity >= 0.60 and context_fit >= 0.45 and score < get_cfg(config, "thresholds", "gate_promote", default=0.66):
            action = GateAction.KEEP_WEAK
            state = RecallState.WEAK_ACTIVE
            reason = "high_similarity_contextual_probe"
        elif score >= get_cfg(config, "thresholds", "gate_promote", default=0.66):
            action = GateAction.PROMOTE
            state = RecallState.ACTIVE_WORKING_MEMORY
            reason = "strong_fit"
        elif score >= get_cfg(config, "thresholds", "gate_weak", default=0.46):
            action = GateAction.KEEP_WEAK
            state = RecallState.WEAK_ACTIVE
            reason = "tentative_fit"
        else:
            action = GateAction.DISMISS
            state = RecallState.DISMISSED
            reason = "low_fit"
        return GateDecision(
            memory_id=candidate.memory_id,
            action=action,
            score=score,
            reason=reason,
            resulting_state=state,
            explanatory_value=explanatory_value,
            predictive_value=predictive_value,
            context_fit=context_fit,
            uncertainty_reduction=uncertainty_reduction,
            emotional_distortion_risk=emotional_distortion_risk,
            candidate=candidate,
        )
