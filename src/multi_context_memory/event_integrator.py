from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from .config import get_cfg
from .formulas import clamp, compute_boundary_pressure, token_overlap
from .models import ActiveContext, EventSnapshot, EventState, EventUpdate, ModalitySignal
from .state_machines import advance_event_state


class EventIntegrator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.current = self._background_snapshot()

    def _background_snapshot(self) -> EventSnapshot:
        return EventSnapshot(
            current_event_id=None,
            event_state=EventState.BACKGROUND,
            event_start_time=None,
            event_end_time=None,
            active_modalities=[],
            dominant_modality=None,
            global_salience=0.0,
            global_uncertainty=0.0,
            global_novelty=0.0,
            event_hypothesis="background",
            initial_hypothesis="background",
            final_hypothesis="background",
            hypothesis_confidence=1.0,
            resolution_progress=0.0,
            event_boundary_pressure=0.0,
            write_pressure=0.0,
            recall_relevance=0.0,
            linked_active_memories=[],
        )

    def step(
        self,
        signals: list[ModalitySignal],
        active_context: ActiveContext,
        now: float,
        config: dict,
    ) -> EventUpdate:
        if not signals:
            return EventUpdate(self.current, None, False, False, False, False)

        sorted_signals = sorted(signals, key=lambda signal: signal.salience, reverse=True)
        dominant = sorted_signals[0]
        active_modalities = [signal.modality_id for signal in sorted_signals if signal.salience > 0.15 or signal.write_pressure > 0.20]
        global_salience = clamp(sum(signal.salience for signal in signals) / len(signals))
        global_uncertainty = clamp(sum(signal.uncertainty for signal in signals) / len(signals))
        global_novelty = clamp(sum(signal.novelty for signal in signals) / len(signals))
        recall_relevance = clamp(max(signal.recall_activation for signal in signals))
        write_pressure = clamp(max(signal.write_pressure for signal in signals))
        context_shift = token_overlap(active_context.context_labels, dominant.write_proposal.get("context_labels", []))
        temporal_signal = next((signal for signal in signals if signal.modality_id == "time_context"), None)
        emotion_signal = next((signal for signal in signals if signal.modality_id == "emotion"), None)
        boundary_pressure = compute_boundary_pressure(
            {
                "change_magnitude": dominant.current_feature_vector.get("intensity", dominant.salience),
                "novelty": global_novelty,
                "prediction_error": clamp(sum(signal.prediction_error for signal in signals) / len(signals)),
                "temporal_discontinuity": temporal_signal.current_feature_vector.get("context_shift", 0.0) if temporal_signal else 0.0,
                "emotion_jump": emotion_signal.current_feature_vector.get("arousal", 0.0) if emotion_signal else 0.0,
                "goal_interruption": max(signal.goal_relevance for signal in signals),
                "context_shift": clamp(1.0 - context_shift),
            },
            get_cfg(config, "weights", "boundary", default={}),
        )
        resolution_progress = clamp((1.0 - global_uncertainty) * 0.55 + dominant.confidence * 0.25 + (1.0 - dominant.prediction_error) * 0.20)
        explanation_stabilized = (
            resolution_progress >= get_cfg(config, "thresholds", "resolved_progress", default=0.7)
            and global_uncertainty <= get_cfg(config, "thresholds", "resolved_uncertainty", default=0.35)
        )
        event_ended = (
            self.current.event_state != EventState.BACKGROUND
            and global_salience <= get_cfg(config, "thresholds", "event_close_salience", default=0.22)
            and boundary_pressure <= get_cfg(config, "thresholds", "boundary_reset", default=0.24)
        )
        new_evidence = dominant.local_hypothesis != self.current.final_hypothesis or any(signal.update_pressure > 0.35 for signal in signals)
        finalized_event = None
        started_new_event = False

        if self.current.event_state == EventState.BACKGROUND and boundary_pressure >= get_cfg(config, "thresholds", "boundary_candidate", default=0.42):
            self.current = EventSnapshot(
                current_event_id=str(uuid4()),
                event_state=EventState.BOUNDARY_CANDIDATE,
                event_start_time=now,
                event_end_time=None,
                active_modalities=active_modalities,
                dominant_modality=dominant.modality_id,
                global_salience=global_salience,
                global_uncertainty=global_uncertainty,
                global_novelty=global_novelty,
                event_hypothesis=dominant.local_hypothesis,
                initial_hypothesis=dominant.local_hypothesis,
                final_hypothesis=dominant.local_hypothesis,
                hypothesis_confidence=dominant.confidence,
                resolution_progress=resolution_progress,
                event_boundary_pressure=boundary_pressure,
                write_pressure=write_pressure,
                recall_relevance=recall_relevance,
                linked_active_memories=[],
                context_labels=list(dict.fromkeys(active_context.context_labels + dominant.write_proposal.get("context_labels", []))),
                goal_state=active_context.current_goal_state,
                emotion_state=deepcopy(active_context.current_emotion_state),
                modality_snapshots={signal.modality_id: self._signal_snapshot(signal) for signal in signals if signal.modality_id in active_modalities},
                boundary_candidate_ticks=1,
            )
            started_new_event = True
        else:
            snapshot = deepcopy(self.current)
            snapshot.active_modalities = active_modalities
            snapshot.dominant_modality = dominant.modality_id
            snapshot.global_salience = global_salience
            snapshot.global_uncertainty = global_uncertainty
            snapshot.global_novelty = global_novelty
            snapshot.event_boundary_pressure = boundary_pressure
            snapshot.write_pressure = write_pressure
            snapshot.recall_relevance = recall_relevance
            snapshot.hypothesis_confidence = dominant.confidence
            snapshot.resolution_progress = resolution_progress
            snapshot.context_labels = list(dict.fromkeys(snapshot.context_labels + active_context.context_labels + dominant.write_proposal.get("context_labels", [])))
            snapshot.goal_state = active_context.current_goal_state
            snapshot.emotion_state = deepcopy(active_context.current_emotion_state)
            for signal in signals:
                if signal.modality_id not in active_modalities:
                    continue
                existing_snapshot = snapshot.modality_snapshots.get(signal.modality_id)
                new_snapshot = self._signal_snapshot(signal)
                if existing_snapshot is None or new_snapshot["salience"] >= existing_snapshot.get("salience", 0.0) or new_snapshot["write_pressure"] >= existing_snapshot.get("write_pressure", 0.0):
                    snapshot.modality_snapshots[signal.modality_id] = new_snapshot
            if dominant.local_hypothesis != snapshot.final_hypothesis and (
                snapshot.event_state not in {EventState.RESOLVED, EventState.UNRESOLVED}
                or dominant.salience >= 0.45
            ):
                snapshot.event_hypothesis = dominant.local_hypothesis
                snapshot.final_hypothesis = dominant.local_hypothesis
            if snapshot.event_state == EventState.BOUNDARY_CANDIDATE and boundary_pressure >= get_cfg(config, "thresholds", "boundary_start", default=0.58):
                snapshot.boundary_candidate_ticks += 1
                if snapshot.boundary_candidate_ticks >= get_cfg(config, "runtime", "candidate_ticks", default=2):
                    snapshot.event_state = EventState.ACTIVE_EVENT
            elif snapshot.event_state == EventState.BOUNDARY_CANDIDATE and not event_ended:
                snapshot.boundary_candidate_ticks += 1
                if (
                    snapshot.boundary_candidate_ticks >= get_cfg(config, "runtime", "candidate_ticks", default=2)
                    and (global_salience >= 0.20 or write_pressure >= 0.20 or new_evidence)
                ) or write_pressure >= get_cfg(config, "thresholds", "write_provisional", default=0.42):
                    snapshot.event_state = EventState.ACTIVE_EVENT
            else:
                snapshot.boundary_candidate_ticks = max(0, snapshot.boundary_candidate_ticks)
            snapshot.quiet_ticks = snapshot.quiet_ticks + 1 if event_ended else 0
            if snapshot.event_state != EventState.BOUNDARY_CANDIDATE:
                snapshot.event_state = advance_event_state(
                    snapshot.event_state,
                    boundary_pressure=boundary_pressure,
                    boundary_candidate_threshold=get_cfg(config, "thresholds", "boundary_candidate", default=0.42),
                    boundary_start_threshold=get_cfg(config, "thresholds", "boundary_start", default=0.58),
                    boundary_reset_threshold=get_cfg(config, "thresholds", "boundary_reset", default=0.24),
                    write_pressure=write_pressure,
                    provisional_write_threshold=get_cfg(config, "thresholds", "write_provisional", default=0.42),
                    new_evidence=new_evidence,
                    explanation_stabilized=explanation_stabilized,
                    event_ended=event_ended,
                )
            if snapshot.event_state in {EventState.ACTIVE_EVENT, EventState.PROVISIONAL_WRITE, EventState.UPDATING_EVIDENCE} and explanation_stabilized:
                snapshot.event_state = EventState.RESOLVED
            if snapshot.quiet_ticks >= get_cfg(config, "runtime", "quiet_ticks_to_close", default=2) and snapshot.event_state in {EventState.RESOLVED, EventState.UNRESOLVED}:
                snapshot.event_end_time = now
                finalized_event = deepcopy(snapshot)
                snapshot = self._background_snapshot()
            self.current = snapshot

        should_provisional = (
            self.current.event_state in {EventState.BOUNDARY_CANDIDATE, EventState.ACTIVE_EVENT, EventState.PROVISIONAL_WRITE, EventState.UPDATING_EVIDENCE}
            and self.current.write_pressure >= get_cfg(config, "thresholds", "write_provisional", default=0.42)
        )
        should_durable = (
            self.current.event_state in {EventState.RESOLVED, EventState.UNRESOLVED, EventState.UPDATING_EVIDENCE, EventState.PROVISIONAL_WRITE}
            and self.current.write_pressure >= get_cfg(config, "thresholds", "write_durable", default=0.68)
        ) or (finalized_event is not None)
        recall_trigger = recall_relevance >= get_cfg(config, "thresholds", "recall_trigger", default=0.46)
        return EventUpdate(self.current, finalized_event, started_new_event, should_provisional, should_durable, recall_trigger)

    def flush(self, now: float) -> EventSnapshot | None:
        if self.current.event_state == EventState.BACKGROUND or self.current.current_event_id is None:
            return None
        snapshot = deepcopy(self.current)
        snapshot.event_end_time = now
        if snapshot.event_state not in {EventState.RESOLVED, EventState.UNRESOLVED}:
            snapshot.event_state = EventState.RESOLVED if snapshot.resolution_progress >= 0.5 else EventState.UNRESOLVED
        self.current = self._background_snapshot()
        return snapshot

    @staticmethod
    def _signal_snapshot(signal: ModalitySignal) -> dict:
        return {
            "feature_vector": signal.current_feature_vector,
            "hypothesis": signal.local_hypothesis,
            "confidence": signal.confidence,
            "uncertainty": signal.uncertainty,
            "novelty": signal.novelty,
            "salience": signal.salience,
            "write_pressure": signal.write_pressure,
            "prediction_error": signal.prediction_error,
            "goal_relevance": signal.goal_relevance,
        }
