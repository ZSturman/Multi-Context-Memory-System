from pathlib import Path

from multi_context_memory.runtime import MemorySystemRuntime


def test_surprise_then_resolution_and_same_cue_recall(tmp_path: Path) -> None:
    runtime = MemorySystemRuntime(store_path=tmp_path / "memory.db")
    try:
        first = runtime.run_scenario("surprise_then_resolution")
        events = first["events"]
        assert any("pan" in event["final_hypothesis"] or "pan" in event["event_summary"] for event in events)
        second = runtime.run_scenario("same_cue_later")
        assert any(step["recall_candidates"] for step in second["history"])
    finally:
        runtime.close()


def test_false_cue_and_intrusion_drive_reconsolidation(tmp_path: Path) -> None:
    runtime = MemorySystemRuntime(store_path=tmp_path / "memory.db")
    try:
        runtime.run_scenario("surprise_then_resolution")
        mismatch = runtime.run_scenario("false_cue_mismatch")
        assert any("dismiss" in step["decisions"] or "mark_for_reconsolidation" in step["decisions"] for step in mismatch["history"])
        intrusive = runtime.run_scenario("intrusive_repeated_poor_fit")
        queue = intrusive["inspect"]["reconsolidation_queue"]
        assert queue or any(event["reconsolidation_pressure"] > 0 for event in intrusive["events"])
    finally:
        runtime.close()


def test_repeated_patterns_create_abstract_concepts(tmp_path: Path) -> None:
    runtime = MemorySystemRuntime(store_path=tmp_path / "memory.db")
    try:
        summary = runtime.run_scenario("repeated_pattern_abstraction")
        concepts = summary["inspect"]["concepts"]
        assert any(label.startswith("abstract:") for label in concepts)
    finally:
        runtime.close()


def test_persistence_survives_without_origin_modality_runtime(tmp_path: Path) -> None:
    store_path = tmp_path / "memory.db"
    runtime = MemorySystemRuntime(store_path=store_path)
    try:
        runtime.run_scenario("surprise_then_resolution")
    finally:
        runtime.close()
    runtime = MemorySystemRuntime(store_path=store_path)
    try:
        runtime.modalities.pop("vision", None)
        snapshot = runtime.inspect()
        assert snapshot["events"]
        assert any("pan" in event or "bang" in event for event in snapshot["events"])
    finally:
        runtime.close()
