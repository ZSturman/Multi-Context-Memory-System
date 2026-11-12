from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .config import get_cfg
from .formulas import clamp, compute_recall_trigger, compute_write_pressure, token_overlap, vector_distance
from .models import ActiveContext, CrossModalMessage, ModalityObservation, ModalitySignal, RecallCue


def _extract_feature_vector(raw_input: dict[str, Any]) -> dict[str, float]:
    explicit = raw_input.get("feature_vector")
    if isinstance(explicit, dict):
        return {str(key): float(value) for key, value in explicit.items()}
    feature_vector: dict[str, float] = {}
    for key, value in raw_input.items():
        if isinstance(value, bool):
            feature_vector[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            feature_vector[key] = float(value)
    return feature_vector


class SensoryBuffer:
    def __init__(self, modality_id: str, history_size: int = 5) -> None:
        self.modality_id = modality_id
        self.history: deque[ModalityObservation] = deque(maxlen=history_size)

    def update(self, raw_input: dict[str, Any], timestamp: float) -> ModalityObservation:
        feature_vector = _extract_feature_vector(raw_input)
        previous = self.history[-1] if self.history else None
        change_magnitude = vector_distance(feature_vector, previous.feature_vector) if previous else clamp(sum(feature_vector.values()) / (len(feature_vector) or 1))
        dt = max(1.0, timestamp - previous.timestamp) if previous else 1.0
        rate_of_change = clamp(change_magnitude / dt)
        stability = clamp(1.0 - min(1.0, change_magnitude))
        observation = ModalityObservation(
            modality_id=self.modality_id,
            feature_vector=feature_vector,
            change_magnitude=change_magnitude,
            rate_of_change=rate_of_change,
            stability=stability,
            timestamp=timestamp,
            raw_input=raw_input,
        )
        self.history.append(observation)
        return observation


class ModalityContext:
    def __init__(self, modality_id: str, config: dict[str, Any]) -> None:
        self.modality_id = modality_id
        self.config = config
        self.previous_hypothesis = ""
        self.previous_uncertainty = 0.5

    def step(
        self,
        observation: ModalityObservation,
        bus_messages: list[CrossModalMessage],
        active_context: ActiveContext,
        config: dict[str, Any],
    ) -> ModalitySignal:
        modality_cfg = get_cfg(config, "modality_context", default={})
        hypothesis = str(observation.raw_input.get("hypothesis") or observation.raw_input.get("candidate_category") or f"{self.modality_id}_state")
        hypothesis_shift = 1.0 if hypothesis != self.previous_hypothesis else 0.0
        bus_salience = sum(message.salience_score for message in bus_messages) / len(bus_messages) if bus_messages else 0.0
        novelty = clamp(
            observation.change_magnitude * modality_cfg.get("novelty_weight_change", 0.55)
            + bus_salience * modality_cfg.get("novelty_weight_bus", 0.20)
            + hypothesis_shift * modality_cfg.get("novelty_weight_hypothesis_shift", 0.25)
        )
        confidence_hint = float(observation.raw_input.get("confidence", 0.0))
        confidence = clamp(
            max(modality_cfg.get("confidence_floor", 0.25), 1.0 - observation.rate_of_change * 0.35 - novelty * 0.20 + confidence_hint * 0.45)
        )
        uncertainty = clamp((1.0 - confidence) * modality_cfg.get("uncertainty_gain", 0.65) + observation.rate_of_change * 0.20)
        prediction_error = clamp(abs(uncertainty - self.previous_uncertainty) + hypothesis_shift * 0.25)
        goal_relevance = token_overlap(
            active_context.current_goal_state.split("_"),
            set(hypothesis.split("_")) | set(observation.raw_input.get("tags", [])),
        )
        arousal = active_context.current_emotion_state.get("arousal", 0.0)
        salience = clamp(
            observation.change_magnitude * modality_cfg.get("salience_weight_change", 0.35)
            + novelty * modality_cfg.get("salience_weight_novelty", 0.25)
            + goal_relevance * modality_cfg.get("salience_weight_goal", 0.15)
            + arousal * modality_cfg.get("salience_weight_arousal", 0.25)
        )
        recall_activation = compute_recall_trigger(
            {
                "cue_similarity": clamp(1.0 - observation.stability),
                "uncertainty": uncertainty,
                "goal_relevance": goal_relevance,
                "contextual_match": token_overlap(active_context.context_labels, observation.raw_input.get("context_labels", [])),
                "modality_salience": salience,
            },
            get_cfg(config, "weights", "recall", default={}),
        )
        write_pressure = compute_write_pressure(
            {
                "novelty": novelty,
                "salience": salience,
                "emotional_weight": arousal,
                "goal_relevance": goal_relevance,
                "prediction_error": prediction_error,
                "uncertainty_peak": max(self.previous_uncertainty, uncertainty),
                "uncertainty_reduction": clamp(self.previous_uncertainty - uncertainty),
                "consequence": float(observation.raw_input.get("consequence", 0.0)),
                "rehearsal_thought": float(observation.raw_input.get("thought_emphasis", 0.0)),
            },
            get_cfg(config, "weights", "write", default={}),
        )
        update_pressure = clamp((prediction_error + uncertainty) / 2.0)
        context_labels = list(dict.fromkeys(observation.raw_input.get("context_labels", []) + active_context.context_labels))
        cue = RecallCue(
            modality_id=self.modality_id,
            feature_vector=observation.feature_vector,
            salience=salience,
            uncertainty=uncertainty,
            goal_relevance=goal_relevance,
            context_labels=context_labels,
            event_id=active_context.current_event_id,
            hypothesis=hypothesis,
        )
        bus_message = CrossModalMessage(
            source_modality=self.modality_id,
            timestamp=observation.timestamp,
            change_score=observation.change_magnitude,
            salience_score=salience,
            uncertainty_score=uncertainty,
            confidence_score=confidence,
            spatial_hint=observation.raw_input.get("spatial_hint"),
            temporal_hint=observation.raw_input.get("temporal_hint"),
            emotion_hint=observation.raw_input.get("emotion_hint"),
            goal_relevance=goal_relevance,
            anomaly_flag=prediction_error > 0.4,
            candidate_category=observation.raw_input.get("candidate_category") or hypothesis,
        )
        signal = ModalitySignal(
            modality_id=self.modality_id,
            current_feature_vector=observation.feature_vector,
            local_hypothesis=hypothesis,
            confidence=confidence,
            uncertainty=uncertainty,
            novelty=novelty,
            prediction_error=prediction_error,
            salience=salience,
            recall_activation=recall_activation,
            write_pressure=write_pressure,
            update_pressure=update_pressure,
            cross_modal_signal_summary={
                "bus_salience": bus_salience,
                "bus_uncertainty": sum(message.uncertainty_score for message in bus_messages) / len(bus_messages) if bus_messages else 0.0,
                "bus_messages": float(len(bus_messages)),
            },
            goal_relevance=goal_relevance,
            local_state_summary=f"{self.modality_id}:{hypothesis}",
            recall_cue_proposal=cue,
            write_proposal={"hypothesis": hypothesis, "context_labels": context_labels},
            update_proposal={"prediction_error": prediction_error},
            bus_message=bus_message,
            timestamp=observation.timestamp,
        )
        self.previous_hypothesis = hypothesis
        self.previous_uncertainty = uncertainty
        return signal


@dataclass(slots=True)
class ModalityRuntime:
    buffer: SensoryBuffer
    context: ModalityContext
    enabled: bool = True


def build_modality_runtimes(config: dict[str, Any]) -> dict[str, ModalityRuntime]:
    modalities = get_cfg(config, "modalities", "concrete", default=[]) + get_cfg(config, "modalities", "stubs", default=[])
    runtimes: dict[str, ModalityRuntime] = {}
    for modality_id in modalities:
        runtimes[modality_id] = ModalityRuntime(
            buffer=SensoryBuffer(modality_id),
            context=ModalityContext(modality_id, config),
            enabled=modality_id in get_cfg(config, "modalities", "concrete", default=[]),
        )
    return runtimes
