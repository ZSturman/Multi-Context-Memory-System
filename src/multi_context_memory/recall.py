from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import get_cfg
from .formulas import clamp, feature_similarity, token_overlap
from .models import EventSnapshot, RecallCandidate, RecallCue, RecallState, WorkingMemorySnapshot
from .store import GraphMemoryStore


class RecallEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.inhibition_levels: dict[str, float] = defaultdict(float)
        self.dismissal_imprints: dict[tuple[str, str], float] = defaultdict(float)
        self.context_saturation: dict[tuple[str, str], float] = defaultdict(float)

    def retrieve(
        self,
        cues: list[RecallCue],
        event_view: EventSnapshot,
        working_memory: WorkingMemorySnapshot,
        inhibition_state: dict[str, float] | None,
        store: GraphMemoryStore,
        config: dict,
    ) -> list[RecallCandidate]:
        inhibition_state = inhibition_state or {}
        local_limit = get_cfg(config, "recall", "max_local_candidates_per_cue", default=6)
        raw_candidates: dict[str, RecallCandidate] = {}
        for cue in cues:
            traces = store.search_traces(cue.modality_id, limit=local_limit)
            scored_traces = []
            for trace in traces:
                similarity = feature_similarity(cue.feature_vector, trace.feature_signature)
                if similarity < get_cfg(config, "thresholds", "local_trace_match", default=0.28):
                    continue
                scored_traces.append((similarity, trace))
            for similarity, trace in sorted(scored_traces, key=lambda item: item[0], reverse=True)[:local_limit]:
                event = store.get_event_memory(trace.event_id)
                if event is None:
                    continue
                regulation = {
                    "inhibition_level": max(inhibition_state.get(event.event_id, 0.0), self.inhibition_levels[event.event_id], event.inhibition_level),
                    "dismissal_imprint": max(
                        self.dismissal_imprints[(self._context_key(cue.context_labels), event.event_id)],
                        event.dismissal_imprint,
                    ),
                    "context_saturation": max(self.context_saturation[(event_view.current_event_id or "none", event.event_id)], event.context_saturation),
                    "reconsolidation_pressure": event.reconsolidation_pressure,
                }
                penalty = clamp(regulation["inhibition_level"] + regulation["dismissal_imprint"] * 0.7 + regulation["context_saturation"] * 0.5)
                total_score = clamp(
                    similarity * 0.40
                    + event.memory_strength * 0.20
                    + event.importance_score * 0.15
                    + token_overlap(event.context_labels, cue.context_labels) * 0.10
                    + cue.goal_relevance * 0.10
                    - penalty * 0.25
                )
                if total_score <= 0:
                    continue
                candidate = RecallCandidate(
                    memory_id=event.event_id,
                    event_id=event.event_id,
                    source_trace_id=trace.trace_id,
                    modality_type=trace.modality_type,
                    cue_similarity=similarity,
                    event_activation=clamp((event.memory_strength + event.importance_score + trace.trace_strength) / 3.0),
                    graph_activation=0.0,
                    context_fit=token_overlap(event.context_labels, cue.context_labels),
                    goal_relevance=cue.goal_relevance,
                    total_score=total_score,
                    state=RecallState.CANDIDATE_RECALL,
                    event_summary=event.event_summary,
                    context_labels=event.context_labels,
                    arousal_score=event.arousal_score,
                    importance_score=event.importance_score,
                    memory_strength=event.memory_strength,
                    already_integrated=event.event_id in working_memory.integrated_ids,
                    already_dismissed=event.event_id in working_memory.dismissed_ids,
                    recurrent_mismatch=event.reconsolidation_pressure > 0.4,
                    regulation=regulation,
                )
                existing = raw_candidates.get(event.event_id)
                if existing is None or existing.total_score < candidate.total_score:
                    raw_candidates[event.event_id] = candidate

        expanded = self._expand_graph(raw_candidates, cues, store, config)
        ranked = sorted(expanded.values(), key=lambda candidate: candidate.total_score, reverse=True)
        return ranked[: get_cfg(config, "recall", "max_selected_candidates", default=5)]

    def _expand_graph(
        self,
        candidates: dict[str, RecallCandidate],
        cues: list[RecallCue],
        store: GraphMemoryStore,
        config: dict,
    ) -> dict[str, RecallCandidate]:
        expanded = dict(candidates)
        if not candidates:
            return expanded
        all_edges = store.list_edges()
        cue_context = list({label for cue in cues for label in cue.context_labels})
        for candidate in list(candidates.values()):
            related_scores: dict[str, float] = defaultdict(float)
            connected_nodes: list[str] = []
            for edge in all_edges:
                if edge.source_id == candidate.memory_id:
                    connected_nodes.append(edge.target_id)
                elif edge.target_id == candidate.memory_id:
                    connected_nodes.append(edge.source_id)
            if not connected_nodes:
                continue
            for edge in all_edges:
                if edge.source_id in connected_nodes:
                    related_scores[edge.target_id] += edge.weight * edge.evidence_strength
                if edge.target_id in connected_nodes:
                    related_scores[edge.source_id] += edge.weight * edge.evidence_strength
            for event_id, score in related_scores.items():
                if event_id == candidate.memory_id or score < get_cfg(config, "thresholds", "graph_spread_min", default=0.20):
                    continue
                event = store.get_event_memory(event_id)
                if event is None:
                    continue
                graph_score = clamp(score / max(1, len(connected_nodes)))
                total_score = clamp(candidate.total_score * 0.65 + graph_score * 0.20 + token_overlap(event.context_labels, cue_context) * 0.15)
                existing = expanded.get(event.event_id)
                new_candidate = RecallCandidate(
                    memory_id=event.event_id,
                    event_id=event.event_id,
                    source_trace_id=None,
                    modality_type=event.dominant_modality,
                    cue_similarity=0.0,
                    event_activation=clamp((event.memory_strength + event.importance_score) / 2.0),
                    graph_activation=graph_score,
                    context_fit=token_overlap(event.context_labels, cue_context),
                    goal_relevance=max((cue.goal_relevance for cue in cues), default=0.0),
                    total_score=total_score,
                    state=RecallState.CANDIDATE_RECALL,
                    event_summary=event.event_summary,
                    context_labels=event.context_labels,
                    arousal_score=event.arousal_score,
                    importance_score=event.importance_score,
                    memory_strength=event.memory_strength,
                    already_integrated=False,
                    already_dismissed=False,
                    recurrent_mismatch=event.reconsolidation_pressure > 0.4,
                    regulation={
                        "inhibition_level": event.inhibition_level,
                        "dismissal_imprint": event.dismissal_imprint,
                        "context_saturation": event.context_saturation,
                        "reconsolidation_pressure": event.reconsolidation_pressure,
                    },
                )
                if existing is None or existing.total_score < total_score:
                    expanded[event.event_id] = new_candidate
        return expanded

    def apply_feedback(
        self,
        decisions: list,
        active_context_labels: list[str],
        current_event_id: str | None,
        store: GraphMemoryStore,
        now: float,
    ) -> None:
        context_key = self._context_key(active_context_labels)
        for decision in decisions:
            memory_id = decision.memory_id
            event = store.get_event_memory(memory_id)
            if event is None:
                continue
            inhibition = self.inhibition_levels[memory_id]
            dismissal = self.dismissal_imprints[(context_key, memory_id)]
            saturation = self.context_saturation[(current_event_id or "none", memory_id)]
            recon = event.reconsolidation_pressure
            if decision.action.value == "promote_to_active":
                inhibition = clamp(inhibition + 0.20)
                saturation = clamp(saturation + get_cfg(self.config, "recall", "integration_suppression", default=0.35))
            elif decision.action.value == "keep_weakly_active":
                inhibition = clamp(inhibition + 0.10)
            elif decision.action.value == "dismiss":
                inhibition = clamp(inhibition + 0.25)
                dismissal = clamp(dismissal + 0.35)
                if dismissal > 0.20 or inhibition > 0.20 or event.reconsolidation_pressure > 0.10:
                    recon = clamp(recon + get_cfg(self.config, "recall", "reconsolidation_gain", default=0.18))
                    store.queue_reconsolidation(memory_id, "repeated_dismissal_poor_fit", recon, now)
            elif decision.action.value == "temporarily_block":
                inhibition = clamp(inhibition + 0.45)
                dismissal = clamp(dismissal + 0.20)
            else:
                recon = clamp(recon + get_cfg(self.config, "recall", "reconsolidation_gain", default=0.18))
                inhibition = clamp(inhibition + 0.20)
                dismissal = clamp(dismissal + 0.20)
                store.queue_reconsolidation(memory_id, decision.reason, recon, now)
            self.inhibition_levels[memory_id] = inhibition
            self.dismissal_imprints[(context_key, memory_id)] = dismissal
            self.context_saturation[(current_event_id or "none", memory_id)] = saturation
            store.update_event_regulation(
                memory_id,
                inhibition_level=inhibition,
                dismissal_imprint=dismissal,
                context_saturation=saturation,
                reconsolidation_pressure=recon,
            )

    def decay(self, config: dict) -> None:
        inhibition_decay = get_cfg(config, "recall", "inhibition_decay", default=0.12)
        dismissal_decay = get_cfg(config, "recall", "dismissal_decay", default=0.10)
        saturation_decay = get_cfg(config, "recall", "saturation_decay", default=0.08)
        for memory_id, value in list(self.inhibition_levels.items()):
            self.inhibition_levels[memory_id] = clamp(value * (1.0 - inhibition_decay))
        for key, value in list(self.dismissal_imprints.items()):
            self.dismissal_imprints[key] = clamp(value * (1.0 - dismissal_decay))
        for key, value in list(self.context_saturation.items()):
            self.context_saturation[key] = clamp(value * (1.0 - saturation_decay))

    @staticmethod
    def _context_key(labels: list[str]) -> str:
        return "|".join(sorted(set(labels)))
