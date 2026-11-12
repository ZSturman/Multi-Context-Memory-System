from __future__ import annotations

from collections import defaultdict
from hashlib import sha1

from .formulas import clamp, token_overlap
from .models import ConceptNode, Edge
from .store import GraphMemoryStore


def _stable_id(prefix: str, label: str) -> str:
    digest = sha1(label.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


class ConsolidationEngine:
    def run(self, store: GraphMemoryStore, now: float, config: dict) -> None:
        events = store.list_event_memories()
        if len(events) < 2:
            return
        grouped: dict[str, list] = defaultdict(list)
        for event in events:
            traces = store.get_traces_for_event(event.event_id)
            dominant_trace = max(traces, key=lambda trace: trace.trace_strength, default=None)
            cue_key = dominant_trace.local_hypothesis if dominant_trace is not None else (event.final_hypothesis or event.event_summary)
            if any(trace.feature_signature.get("metallic", 0.0) > 0.5 for trace in traces):
                cue_key = "metallic_impact_family"
            key = "|".join(sorted(set(event.context_labels[:2] + [cue_key])))
            grouped[key].append(event)
        for key, group in grouped.items():
            if len(group) < 2:
                continue
            label = "abstract:" + key.replace("|", "_")
            concept_id = _stable_id("concept", label)
            associated_modalities = list({event.dominant_modality for event in group})
            prototype_signature = {"frequency": float(len(group)), "mean_strength": sum(event.memory_strength for event in group) / len(group)}
            store.upsert_concept_node(
                ConceptNode(
                    concept_id=concept_id,
                    label=label,
                    concept_type="abstract_pattern",
                    prototype_signature=prototype_signature,
                    associated_modalities=associated_modalities,
                    semantic_strength=clamp(0.55 + len(group) * 0.08),
                    linked_event_ids=[event.event_id for event in group],
                )
            )
            for event in group:
                store.upsert_edge(
                    Edge(
                        source_id=event.event_id,
                        target_id=concept_id,
                        edge_type="similar_to",
                        weight=clamp(event.memory_strength * 0.8),
                        direction="out",
                        evidence_strength=clamp(event.importance_score * 0.7 + 0.2),
                        last_activated=now,
                        decay_rate=0.02,
                    )
                )
            for left in group:
                for right in group:
                    if left.event_id == right.event_id:
                        continue
                    overlap = token_overlap(left.context_labels + [left.final_hypothesis], right.context_labels + [right.final_hypothesis])
                    if overlap < 0.25:
                        continue
                    store.upsert_edge(
                        Edge(
                            source_id=left.event_id,
                            target_id=right.event_id,
                            edge_type="predicted_by",
                            weight=clamp((left.memory_strength + right.memory_strength + overlap) / 3.0),
                            direction="out",
                            evidence_strength=clamp(overlap),
                            last_activated=now,
                            decay_rate=0.03,
                        )
                    )
        for edge in store.list_edges():
            if edge.evidence_strength < 0.20:
                store.upsert_edge(
                    Edge(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        edge_type=edge.edge_type,
                        weight=clamp(edge.weight * 0.92),
                        direction=edge.direction,
                        evidence_strength=clamp(edge.evidence_strength * 0.90),
                        last_activated=edge.last_activated,
                        decay_rate=edge.decay_rate,
                    )
                )


class ReconsolidationEngine:
    def run(self, store: GraphMemoryStore, flagged_ids: list[str], now: float, config: dict) -> None:
        queued = flagged_ids or [memory_id for memory_id, _, _ in store.pop_reconsolidation_candidates(limit=20)]
        for memory_id in queued:
            event = store.get_event_memory(memory_id)
            if event is None:
                continue
            traces = store.get_traces_for_event(memory_id)
            for trace in traces:
                if trace.modality_type == event.dominant_modality:
                    trace.cue_strength = clamp(trace.cue_strength + 0.08)
                else:
                    trace.cue_strength = clamp(trace.cue_strength * 0.88)
                    trace.trace_strength = clamp(trace.trace_strength * 0.93)
                store.upsert_modality_trace(trace)
            diagnostic_label = f"diagnostic:{event.dominant_modality}"
            if diagnostic_label not in event.context_labels:
                event.context_labels.append(diagnostic_label)
            event.arousal_score = clamp(event.arousal_score * 0.90)
            event.reconsolidation_pressure = clamp(event.reconsolidation_pressure * 0.40)
            store.upsert_event_memory(event)
            store.mark_reconsolidated(memory_id)
