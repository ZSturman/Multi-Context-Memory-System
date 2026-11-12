from __future__ import annotations

from hashlib import sha1
from typing import Iterable

from .formulas import clamp
from .models import ConceptNode, ContextNode, Edge, EventMemory, EventSnapshot, GateDecision, ModalitySignal, ModalityTrace, ThoughtNode, WriteResult
from .store import GraphMemoryStore


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("/", "_")


def _stable_id(prefix: str, label: str) -> str:
    digest = sha1(label.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


class MemoryWriter:
    def apply(
        self,
        event_view: EventSnapshot,
        modality_signals: list[ModalitySignal],
        gate_results: list[GateDecision],
        store: GraphMemoryStore,
        config: dict,
    ) -> WriteResult:
        if event_view.current_event_id is None:
            return WriteResult(None, [], False, False, False)

        signal_map = {signal.modality_id: signal for signal in modality_signals}
        trace_ids: list[str] = []
        trace_lookup: list[ModalityTrace] = []
        for modality_id, snapshot in event_view.modality_snapshots.items():
            signal = signal_map.get(modality_id)
            trace_id = _stable_id("trace", f"{event_view.current_event_id}:{modality_id}")
            trace = ModalityTrace(
                trace_id=trace_id,
                event_id=event_view.current_event_id,
                modality_type=modality_id,
                feature_signature=snapshot.get("feature_vector", {}),
                peak_change=max(snapshot.get("salience", 0.0), snapshot.get("prediction_error", 0.0)),
                novelty_score=snapshot.get("novelty", 0.0),
                uncertainty_start=snapshot.get("uncertainty", 0.0),
                uncertainty_end=snapshot.get("uncertainty", 0.0) * 0.8,
                confidence=snapshot.get("confidence", 0.0),
                salience=snapshot.get("salience", 0.0),
                timestamp_offset=0.0 if event_view.event_start_time is None else max(0.0, (event_view.event_end_time or event_view.event_start_time) - event_view.event_start_time),
                cue_strength=clamp(snapshot.get("salience", 0.0) * 0.6 + snapshot.get("novelty", 0.0) * 0.4),
                trace_strength=clamp(snapshot.get("write_pressure", 0.0) * 0.55 + snapshot.get("confidence", 0.0) * 0.45),
                local_hypothesis=(signal.local_hypothesis if signal else snapshot.get("hypothesis", modality_id)),
            )
            store.upsert_modality_trace(trace)
            trace_ids.append(trace_id)
            trace_lookup.append(trace)
            store.upsert_edge(
                Edge(
                    source_id=event_view.current_event_id,
                    target_id=trace_id,
                    edge_type="part_of",
                    weight=trace.trace_strength,
                    direction="out",
                    evidence_strength=trace.cue_strength,
                    last_activated=event_view.event_end_time or event_view.event_start_time or 0.0,
                    decay_rate=0.04,
                )
            )

        concept_ids = self._write_concepts(event_view, trace_lookup, store)
        context_ids = self._write_contexts(event_view, store)
        thought_ids = self._write_thoughts(event_view, gate_results, store)
        summary = event_view.final_hypothesis or event_view.event_hypothesis or "unresolved_event"
        event_memory = EventMemory(
            event_id=event_view.current_event_id,
            start_time=event_view.event_start_time or 0.0,
            end_time=event_view.event_end_time or event_view.event_start_time or 0.0,
            duration=max(0.0, (event_view.event_end_time or event_view.event_start_time or 0.0) - (event_view.event_start_time or 0.0)),
            location_tag=next((label for label in event_view.context_labels if not label.startswith("time:") and not label.startswith("emotion:")), "unknown"),
            dominant_modality=event_view.dominant_modality or "unknown",
            event_summary=summary,
            initial_hypothesis=event_view.initial_hypothesis,
            final_hypothesis=event_view.final_hypothesis,
            novelty_score=event_view.global_novelty,
            importance_score=clamp(event_view.global_salience * 0.5 + event_view.write_pressure * 0.5),
            arousal_score=event_view.emotion_state.get("arousal", 0.0),
            uncertainty_peak=event_view.global_uncertainty,
            uncertainty_reduction=clamp(1.0 - event_view.global_uncertainty),
            resolution_score=event_view.resolution_progress,
            confidence=event_view.hypothesis_confidence,
            memory_strength=clamp(event_view.write_pressure * 0.65 + event_view.resolution_progress * 0.35),
            linked_modality_trace_ids=trace_ids,
            linked_concept_ids=concept_ids,
            linked_context_ids=context_ids,
            linked_thought_ids=thought_ids,
            resolved_flag=event_view.event_state.value in {"E5_RESOLVED", "E7_CONSOLIDATED"},
            provisional_flag=event_view.event_state.value in {"E2_ACTIVE_EVENT", "E3_PROVISIONAL_WRITE", "E4_UPDATING_EVIDENCE"},
            context_labels=event_view.context_labels,
        )
        existing = store.get_event_memory(event_view.current_event_id)
        if existing:
            event_memory.inhibition_level = existing.inhibition_level
            event_memory.dismissal_imprint = existing.dismissal_imprint
            event_memory.context_saturation = existing.context_saturation
            event_memory.reconsolidation_pressure = existing.reconsolidation_pressure
        store.upsert_event_memory(event_memory)
        return WriteResult(event_view.current_event_id, trace_ids, existing is None, not event_memory.provisional_flag, event_memory.resolved_flag)

    def _write_concepts(self, event_view: EventSnapshot, traces: list[ModalityTrace], store: GraphMemoryStore) -> list[str]:
        labels = {event_view.final_hypothesis or event_view.event_hypothesis}
        labels.update(trace.local_hypothesis for trace in traces if trace.local_hypothesis)
        concept_ids: list[str] = []
        for label in sorted(_slug(label) for label in labels if label):
            concept_id = _stable_id("concept", label)
            concept_ids.append(concept_id)
            prototype = next((trace.feature_signature for trace in traces if _slug(trace.local_hypothesis) == label), {})
            store.upsert_concept_node(
                ConceptNode(
                    concept_id=concept_id,
                    label=label,
                    concept_type="event_pattern",
                    prototype_signature=prototype,
                    associated_modalities=list({trace.modality_type for trace in traces}),
                    semantic_strength=clamp(0.45 + len(traces) * 0.08),
                    linked_event_ids=[event_view.current_event_id],
                )
            )
            store.upsert_edge(
                Edge(
                    source_id=event_view.current_event_id,
                    target_id=concept_id,
                    edge_type="cue_for_recall",
                    weight=0.65,
                    direction="out",
                    evidence_strength=0.55,
                    last_activated=event_view.event_end_time or event_view.event_start_time or 0.0,
                    decay_rate=0.03,
                )
            )
        return concept_ids

    def _write_contexts(self, event_view: EventSnapshot, store: GraphMemoryStore) -> list[str]:
        context_ids: list[str] = []
        for label in sorted(set(event_view.context_labels)):
            context_id = _stable_id("context", label)
            context_ids.append(context_id)
            context_type = "internal" if label.startswith("emotion:") else "environment"
            store.upsert_context_node(ContextNode(context_id=context_id, label=label, context_type=context_type, strength=0.6, linked_event_ids=[event_view.current_event_id]))
            store.upsert_edge(
                Edge(
                    source_id=event_view.current_event_id,
                    target_id=context_id,
                    edge_type="occurs_in",
                    weight=0.60,
                    direction="out",
                    evidence_strength=0.50,
                    last_activated=event_view.event_end_time or event_view.event_start_time or 0.0,
                    decay_rate=0.02,
                )
            )
        return context_ids

    def _write_thoughts(self, event_view: EventSnapshot, gate_results: Iterable[GateDecision], store: GraphMemoryStore) -> list[str]:
        thought_ids: list[str] = []
        for decision in gate_results:
            label = f"{decision.action.value}:{decision.candidate.event_summary}"
            thought_id = _stable_id("thought", label)
            thought_ids.append(thought_id)
            store.upsert_thought_node(ThoughtNode(thought_id=thought_id, text_or_label=label, role="gate_decision", confidence=decision.score, linked_event_ids=[event_view.current_event_id]))
            store.upsert_edge(
                Edge(
                    source_id=thought_id,
                    target_id=event_view.current_event_id,
                    edge_type="resolved_by" if decision.action.value == "promote_to_active" else "contradicts",
                    weight=decision.score,
                    direction="out",
                    evidence_strength=decision.context_fit,
                    last_activated=event_view.event_end_time or event_view.event_start_time or 0.0,
                    decay_rate=0.05,
                )
            )
        return thought_ids
