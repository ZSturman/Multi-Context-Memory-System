from pathlib import Path

from multi_context_memory.config import load_config
from multi_context_memory.memory_writer import MemoryWriter
from multi_context_memory.models import EventSnapshot, EventState, RecallCue, WorkingMemorySnapshot
from multi_context_memory.recall import RecallEngine
from multi_context_memory.store import GraphMemoryStore


def _build_event_snapshot(event_id: str, start: float, hypothesis: str, context_labels: list[str], modality: str, feature_vector: dict[str, float]) -> EventSnapshot:
    return EventSnapshot(
        current_event_id=event_id,
        event_state=EventState.RESOLVED,
        event_start_time=start,
        event_end_time=start + 1,
        active_modalities=[modality],
        dominant_modality=modality,
        global_salience=0.8,
        global_uncertainty=0.2,
        global_novelty=0.7,
        event_hypothesis=hypothesis,
        initial_hypothesis=hypothesis,
        final_hypothesis=hypothesis,
        hypothesis_confidence=0.8,
        resolution_progress=0.85,
        event_boundary_pressure=0.7,
        write_pressure=0.9,
        recall_relevance=0.8,
        linked_active_memories=[],
        context_labels=context_labels,
        goal_state="inspect_noise",
        emotion_state={"arousal": 0.7, "valence": -0.1},
        modality_snapshots={
            modality: {
                "feature_vector": feature_vector,
                "hypothesis": hypothesis,
                "confidence": 0.8,
                "uncertainty": 0.2,
                "novelty": 0.7,
                "salience": 0.9,
                "write_pressure": 0.9,
                "prediction_error": 0.6,
                "goal_relevance": 0.8,
            }
        },
    )


def test_recall_prefers_local_trace_then_graph_expansion(tmp_path: Path) -> None:
    config = load_config()
    store = GraphMemoryStore(tmp_path / "memory.db")
    writer = MemoryWriter()
    writer.apply(_build_event_snapshot("event-a", 0.0, "metallic_bang", ["kitchen", "time:night"], "audio", {"bang": 1.0, "metallic": 0.8}), [], [], store, config)
    writer.apply(_build_event_snapshot("event-b", 3.0, "fallen_pan", ["kitchen", "floor"], "vision", {"pan_on_floor": 1.0, "object_motion": 0.6}), [], [], store, config)
    recall = RecallEngine(config)
    cue = RecallCue(
        modality_id="audio",
        feature_vector={"bang": 0.95, "metallic": 0.75},
        salience=0.9,
        uncertainty=0.8,
        goal_relevance=0.8,
        context_labels=["kitchen", "time:night"],
        hypothesis="metallic_bang",
    )
    candidates = recall.retrieve(
        [cue],
        _build_event_snapshot("current", 10.0, "metallic_bang", ["kitchen", "time:night"], "audio", {"bang": 0.95, "metallic": 0.75}),
        WorkingMemorySnapshot(items={}, active_ids=[], integrated_ids=[], dismissed_ids=[]),
        {},
        store,
        config,
    )
    assert candidates
    assert candidates[0].memory_id == "event-a"
    assert any(candidate.memory_id == "event-b" for candidate in candidates)
    store.close()


def test_recall_feedback_applies_inhibition(tmp_path: Path) -> None:
    config = load_config()
    store = GraphMemoryStore(tmp_path / "memory.db")
    writer = MemoryWriter()
    writer.apply(_build_event_snapshot("event-a", 0.0, "metallic_bang", ["kitchen"], "audio", {"bang": 1.0, "metallic": 0.8}), [], [], store, config)
    recall = RecallEngine(config)
    cue = RecallCue(modality_id="audio", feature_vector={"bang": 1.0, "metallic": 0.8}, salience=1.0, uncertainty=0.7, goal_relevance=0.8, context_labels=["kitchen"], hypothesis="metallic_bang")
    baseline = recall.retrieve([cue], _build_event_snapshot("current", 10.0, "metallic_bang", ["kitchen"], "audio", {"bang": 1.0, "metallic": 0.8}), WorkingMemorySnapshot(items={}, active_ids=[], integrated_ids=[], dismissed_ids=[]), {}, store, config)
    decision = type("Decision", (), {"memory_id": "event-a", "action": type("Action", (), {"value": "dismiss"})(), "reason": "low_fit"})()
    recall.apply_feedback([decision], ["kitchen"], "current", store, 11.0)
    after = recall.retrieve([cue], _build_event_snapshot("current", 12.0, "metallic_bang", ["kitchen"], "audio", {"bang": 1.0, "metallic": 0.8}), WorkingMemorySnapshot(items={}, active_ids=[], integrated_ids=[], dismissed_ids=[]), {}, store, config)
    assert after[0].total_score < baseline[0].total_score
    store.close()
