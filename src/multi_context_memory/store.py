from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import ConceptNode, ContextNode, Edge, EventMemory, ModalityTrace, ThoughtNode


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _merge_unique(existing: Iterable[str], incoming: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(list(existing) + list(incoming)))


class GraphMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS event_memories (
                event_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                location_tag TEXT NOT NULL,
                dominant_modality TEXT NOT NULL,
                event_summary TEXT NOT NULL,
                initial_hypothesis TEXT NOT NULL,
                final_hypothesis TEXT NOT NULL,
                novelty_score REAL NOT NULL,
                importance_score REAL NOT NULL,
                arousal_score REAL NOT NULL,
                uncertainty_peak REAL NOT NULL,
                uncertainty_reduction REAL NOT NULL,
                resolution_score REAL NOT NULL,
                confidence REAL NOT NULL,
                memory_strength REAL NOT NULL,
                linked_modality_trace_ids TEXT NOT NULL,
                linked_concept_ids TEXT NOT NULL,
                linked_context_ids TEXT NOT NULL,
                linked_thought_ids TEXT NOT NULL,
                resolved_flag INTEGER NOT NULL,
                provisional_flag INTEGER NOT NULL,
                context_labels TEXT NOT NULL,
                inhibition_level REAL NOT NULL DEFAULT 0.0,
                dismissal_imprint REAL NOT NULL DEFAULT 0.0,
                context_saturation REAL NOT NULL DEFAULT 0.0,
                reconsolidation_pressure REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS modality_traces (
                trace_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                modality_type TEXT NOT NULL,
                feature_signature TEXT NOT NULL,
                peak_change REAL NOT NULL,
                novelty_score REAL NOT NULL,
                uncertainty_start REAL NOT NULL,
                uncertainty_end REAL NOT NULL,
                confidence REAL NOT NULL,
                salience REAL NOT NULL,
                timestamp_offset REAL NOT NULL,
                cue_strength REAL NOT NULL,
                trace_strength REAL NOT NULL,
                local_hypothesis TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS concept_nodes (
                concept_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                concept_type TEXT NOT NULL,
                prototype_signature TEXT NOT NULL,
                associated_modalities TEXT NOT NULL,
                semantic_strength REAL NOT NULL,
                linked_event_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS context_nodes (
                context_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                context_type TEXT NOT NULL,
                strength REAL NOT NULL,
                linked_event_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS thought_nodes (
                thought_id TEXT PRIMARY KEY,
                text_or_label TEXT NOT NULL,
                role TEXT NOT NULL,
                confidence REAL NOT NULL,
                linked_event_ids TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                direction TEXT NOT NULL,
                evidence_strength REAL NOT NULL,
                last_activated REAL NOT NULL,
                decay_rate REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type)
            );

            CREATE TABLE IF NOT EXISTS reconsolidation_queue (
                memory_id TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                pressure REAL NOT NULL,
                flagged_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scenario_runs (
                run_id TEXT PRIMARY KEY,
                scenario_name TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL NOT NULL,
                summary TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def upsert_event_memory(self, event: EventMemory) -> None:
        self.connection.execute(
            """
            INSERT INTO event_memories (
                event_id, start_time, end_time, duration, location_tag, dominant_modality,
                event_summary, initial_hypothesis, final_hypothesis, novelty_score, importance_score,
                arousal_score, uncertainty_peak, uncertainty_reduction, resolution_score, confidence,
                memory_strength, linked_modality_trace_ids, linked_concept_ids, linked_context_ids,
                linked_thought_ids, resolved_flag, provisional_flag, context_labels, inhibition_level,
                dismissal_imprint, context_saturation, reconsolidation_pressure
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                end_time=excluded.end_time,
                duration=excluded.duration,
                location_tag=excluded.location_tag,
                dominant_modality=excluded.dominant_modality,
                event_summary=excluded.event_summary,
                initial_hypothesis=excluded.initial_hypothesis,
                final_hypothesis=excluded.final_hypothesis,
                novelty_score=MAX(event_memories.novelty_score, excluded.novelty_score),
                importance_score=MAX(event_memories.importance_score, excluded.importance_score),
                arousal_score=MAX(event_memories.arousal_score, excluded.arousal_score),
                uncertainty_peak=MAX(event_memories.uncertainty_peak, excluded.uncertainty_peak),
                uncertainty_reduction=MAX(event_memories.uncertainty_reduction, excluded.uncertainty_reduction),
                resolution_score=excluded.resolution_score,
                confidence=excluded.confidence,
                memory_strength=MAX(event_memories.memory_strength, excluded.memory_strength),
                linked_modality_trace_ids=excluded.linked_modality_trace_ids,
                linked_concept_ids=excluded.linked_concept_ids,
                linked_context_ids=excluded.linked_context_ids,
                linked_thought_ids=excluded.linked_thought_ids,
                resolved_flag=excluded.resolved_flag,
                provisional_flag=excluded.provisional_flag,
                context_labels=excluded.context_labels,
                inhibition_level=excluded.inhibition_level,
                dismissal_imprint=excluded.dismissal_imprint,
                context_saturation=excluded.context_saturation,
                reconsolidation_pressure=excluded.reconsolidation_pressure
            """,
            (
                event.event_id,
                event.start_time,
                event.end_time,
                event.duration,
                event.location_tag,
                event.dominant_modality,
                event.event_summary,
                event.initial_hypothesis,
                event.final_hypothesis,
                event.novelty_score,
                event.importance_score,
                event.arousal_score,
                event.uncertainty_peak,
                event.uncertainty_reduction,
                event.resolution_score,
                event.confidence,
                event.memory_strength,
                _json_dumps(event.linked_modality_trace_ids),
                _json_dumps(event.linked_concept_ids),
                _json_dumps(event.linked_context_ids),
                _json_dumps(event.linked_thought_ids),
                int(event.resolved_flag),
                int(event.provisional_flag),
                _json_dumps(event.context_labels),
                event.inhibition_level,
                event.dismissal_imprint,
                event.context_saturation,
                event.reconsolidation_pressure,
            ),
        )
        self.connection.commit()

    def get_event_memory(self, event_id: str) -> EventMemory | None:
        row = self.connection.execute("SELECT * FROM event_memories WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def list_event_memories(self) -> list[EventMemory]:
        rows = self.connection.execute("SELECT * FROM event_memories ORDER BY start_time").fetchall()
        return [self._row_to_event(row) for row in rows]

    def update_event_regulation(self, event_id: str, **values: float) -> None:
        if not values:
            return
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = list(values.values()) + [event_id]
        self.connection.execute(f"UPDATE event_memories SET {assignments} WHERE event_id = ?", params)
        self.connection.commit()

    def upsert_modality_trace(self, trace: ModalityTrace) -> None:
        self.connection.execute(
            """
            INSERT INTO modality_traces (
                trace_id, event_id, modality_type, feature_signature, peak_change,
                novelty_score, uncertainty_start, uncertainty_end, confidence, salience,
                timestamp_offset, cue_strength, trace_strength, local_hypothesis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                feature_signature=CASE
                    WHEN excluded.trace_strength >= modality_traces.trace_strength OR excluded.cue_strength >= modality_traces.cue_strength
                    THEN excluded.feature_signature
                    ELSE modality_traces.feature_signature
                END,
                peak_change=MAX(modality_traces.peak_change, excluded.peak_change),
                novelty_score=MAX(modality_traces.novelty_score, excluded.novelty_score),
                uncertainty_start=excluded.uncertainty_start,
                uncertainty_end=excluded.uncertainty_end,
                confidence=excluded.confidence,
                salience=MAX(modality_traces.salience, excluded.salience),
                timestamp_offset=excluded.timestamp_offset,
                cue_strength=MAX(modality_traces.cue_strength, excluded.cue_strength),
                trace_strength=MAX(modality_traces.trace_strength, excluded.trace_strength),
                local_hypothesis=CASE
                    WHEN excluded.trace_strength >= modality_traces.trace_strength OR excluded.cue_strength >= modality_traces.cue_strength
                    THEN excluded.local_hypothesis
                    ELSE modality_traces.local_hypothesis
                END
            """,
            (
                trace.trace_id,
                trace.event_id,
                trace.modality_type,
                _json_dumps(trace.feature_signature),
                trace.peak_change,
                trace.novelty_score,
                trace.uncertainty_start,
                trace.uncertainty_end,
                trace.confidence,
                trace.salience,
                trace.timestamp_offset,
                trace.cue_strength,
                trace.trace_strength,
                trace.local_hypothesis,
            ),
        )
        self.connection.commit()

    def get_traces_for_event(self, event_id: str) -> list[ModalityTrace]:
        rows = self.connection.execute("SELECT * FROM modality_traces WHERE event_id = ?", (event_id,)).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def search_traces(self, modality_type: str, limit: int = 10) -> list[ModalityTrace]:
        rows = self.connection.execute(
            "SELECT * FROM modality_traces WHERE modality_type = ? ORDER BY trace_strength DESC",
            (modality_type,),
        ).fetchall()
        return [self._row_to_trace(row) for row in rows[:limit]]

    def upsert_concept_node(self, node: ConceptNode) -> None:
        existing = self.get_concept_node(node.concept_id)
        linked_event_ids = node.linked_event_ids
        if existing:
            linked_event_ids = _merge_unique(existing.linked_event_ids, node.linked_event_ids)
        self.connection.execute(
            """
            INSERT INTO concept_nodes (
                concept_id, label, concept_type, prototype_signature,
                associated_modalities, semantic_strength, linked_event_ids
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(concept_id) DO UPDATE SET
                label=excluded.label,
                concept_type=excluded.concept_type,
                prototype_signature=excluded.prototype_signature,
                associated_modalities=excluded.associated_modalities,
                semantic_strength=MAX(concept_nodes.semantic_strength, excluded.semantic_strength),
                linked_event_ids=excluded.linked_event_ids
            """,
            (
                node.concept_id,
                node.label,
                node.concept_type,
                _json_dumps(node.prototype_signature),
                _json_dumps(node.associated_modalities),
                node.semantic_strength,
                _json_dumps(linked_event_ids),
            ),
        )
        self.connection.commit()

    def get_concept_node(self, concept_id: str) -> ConceptNode | None:
        row = self.connection.execute("SELECT * FROM concept_nodes WHERE concept_id = ?", (concept_id,)).fetchone()
        if row is None:
            return None
        return ConceptNode(
            concept_id=row["concept_id"],
            label=row["label"],
            concept_type=row["concept_type"],
            prototype_signature=_json_loads(row["prototype_signature"], {}),
            associated_modalities=_json_loads(row["associated_modalities"], []),
            semantic_strength=row["semantic_strength"],
            linked_event_ids=_json_loads(row["linked_event_ids"], []),
        )

    def list_concept_nodes(self) -> list[ConceptNode]:
        rows = self.connection.execute("SELECT * FROM concept_nodes ORDER BY semantic_strength DESC").fetchall()
        return [self.get_concept_node(row["concept_id"]) for row in rows]

    def upsert_context_node(self, node: ContextNode) -> None:
        existing = self.get_context_node(node.context_id)
        linked_event_ids = node.linked_event_ids
        if existing:
            linked_event_ids = _merge_unique(existing.linked_event_ids, node.linked_event_ids)
        self.connection.execute(
            """
            INSERT INTO context_nodes (context_id, label, context_type, strength, linked_event_ids)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(context_id) DO UPDATE SET
                label=excluded.label,
                context_type=excluded.context_type,
                strength=MAX(context_nodes.strength, excluded.strength),
                linked_event_ids=excluded.linked_event_ids
            """,
            (node.context_id, node.label, node.context_type, node.strength, _json_dumps(linked_event_ids)),
        )
        self.connection.commit()

    def get_context_node(self, context_id: str) -> ContextNode | None:
        row = self.connection.execute("SELECT * FROM context_nodes WHERE context_id = ?", (context_id,)).fetchone()
        if row is None:
            return None
        return ContextNode(
            context_id=row["context_id"],
            label=row["label"],
            context_type=row["context_type"],
            strength=row["strength"],
            linked_event_ids=_json_loads(row["linked_event_ids"], []),
        )

    def upsert_thought_node(self, node: ThoughtNode) -> None:
        self.connection.execute(
            """
            INSERT INTO thought_nodes (thought_id, text_or_label, role, confidence, linked_event_ids)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thought_id) DO UPDATE SET
                text_or_label=excluded.text_or_label,
                role=excluded.role,
                confidence=MAX(thought_nodes.confidence, excluded.confidence),
                linked_event_ids=excluded.linked_event_ids
            """,
            (node.thought_id, node.text_or_label, node.role, node.confidence, _json_dumps(node.linked_event_ids)),
        )
        self.connection.commit()

    def upsert_edge(self, edge: Edge) -> None:
        self.connection.execute(
            """
            INSERT INTO edges (
                source_id, target_id, edge_type, weight, direction, evidence_strength, last_activated, decay_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                weight=MAX(edges.weight, excluded.weight),
                direction=excluded.direction,
                evidence_strength=MAX(edges.evidence_strength, excluded.evidence_strength),
                last_activated=excluded.last_activated,
                decay_rate=excluded.decay_rate
            """,
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                edge.weight,
                edge.direction,
                edge.evidence_strength,
                edge.last_activated,
                edge.decay_rate,
            ),
        )
        self.connection.commit()

    def list_edges(self) -> list[Edge]:
        rows = self.connection.execute("SELECT * FROM edges ORDER BY evidence_strength DESC, weight DESC").fetchall()
        return [
            Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                edge_type=row["edge_type"],
                weight=row["weight"],
                direction=row["direction"],
                evidence_strength=row["evidence_strength"],
                last_activated=row["last_activated"],
                decay_rate=row["decay_rate"],
            )
            for row in rows
        ]

    def queue_reconsolidation(self, memory_id: str, reason: str, pressure: float, flagged_at: float) -> None:
        self.connection.execute(
            """
            INSERT INTO reconsolidation_queue (memory_id, reason, pressure, flagged_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                reason=excluded.reason,
                pressure=MAX(reconsolidation_queue.pressure, excluded.pressure),
                flagged_at=excluded.flagged_at
            """,
            (memory_id, reason, pressure, flagged_at),
        )
        self.connection.commit()

    def pop_reconsolidation_candidates(self, limit: int = 10) -> list[tuple[str, str, float]]:
        rows = self.connection.execute(
            "SELECT * FROM reconsolidation_queue ORDER BY pressure DESC, flagged_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["memory_id"], row["reason"], row["pressure"]) for row in rows]

    def mark_reconsolidated(self, memory_id: str) -> None:
        self.connection.execute("DELETE FROM reconsolidation_queue WHERE memory_id = ?", (memory_id,))
        self.connection.commit()

    def record_scenario_run(self, run_id: str, scenario_name: str, started_at: float, completed_at: float, summary: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO scenario_runs (run_id, scenario_name, started_at, completed_at, summary)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, scenario_name, started_at, completed_at, _json_dumps(summary)),
        )
        self.connection.commit()

    def inspect_snapshot(self) -> dict[str, Any]:
        return {
            "events": [event.event_summary for event in self.list_event_memories()],
            "traces": [trace.local_hypothesis for trace in self.search_traces("audio", limit=50)] + [trace.local_hypothesis for trace in self.search_traces("vision", limit=50)],
            "concepts": [node.label for node in self.list_concept_nodes()],
            "edges": [f"{edge.source_id}->{edge.target_id}:{edge.edge_type}" for edge in self.list_edges()[:20]],
            "reconsolidation_queue": self.pop_reconsolidation_candidates(limit=20),
        }

    def _row_to_event(self, row: sqlite3.Row) -> EventMemory:
        return EventMemory(
            event_id=row["event_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration=row["duration"],
            location_tag=row["location_tag"],
            dominant_modality=row["dominant_modality"],
            event_summary=row["event_summary"],
            initial_hypothesis=row["initial_hypothesis"],
            final_hypothesis=row["final_hypothesis"],
            novelty_score=row["novelty_score"],
            importance_score=row["importance_score"],
            arousal_score=row["arousal_score"],
            uncertainty_peak=row["uncertainty_peak"],
            uncertainty_reduction=row["uncertainty_reduction"],
            resolution_score=row["resolution_score"],
            confidence=row["confidence"],
            memory_strength=row["memory_strength"],
            linked_modality_trace_ids=_json_loads(row["linked_modality_trace_ids"], []),
            linked_concept_ids=_json_loads(row["linked_concept_ids"], []),
            linked_context_ids=_json_loads(row["linked_context_ids"], []),
            linked_thought_ids=_json_loads(row["linked_thought_ids"], []),
            resolved_flag=bool(row["resolved_flag"]),
            provisional_flag=bool(row["provisional_flag"]),
            context_labels=_json_loads(row["context_labels"], []),
            inhibition_level=row["inhibition_level"],
            dismissal_imprint=row["dismissal_imprint"],
            context_saturation=row["context_saturation"],
            reconsolidation_pressure=row["reconsolidation_pressure"],
        )

    def _row_to_trace(self, row: sqlite3.Row) -> ModalityTrace:
        return ModalityTrace(
            trace_id=row["trace_id"],
            event_id=row["event_id"],
            modality_type=row["modality_type"],
            feature_signature=_json_loads(row["feature_signature"], {}),
            peak_change=row["peak_change"],
            novelty_score=row["novelty_score"],
            uncertainty_start=row["uncertainty_start"],
            uncertainty_end=row["uncertainty_end"],
            confidence=row["confidence"],
            salience=row["salience"],
            timestamp_offset=row["timestamp_offset"],
            cue_strength=row["cue_strength"],
            trace_strength=row["trace_strength"],
            local_hypothesis=row["local_hypothesis"],
        )
