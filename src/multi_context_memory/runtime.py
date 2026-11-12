from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .bus import SharedMessageBus
from .config import get_cfg, load_config
from .consolidation import ConsolidationEngine, ReconsolidationEngine
from .event_integrator import EventIntegrator
from .memory_writer import MemoryWriter
from .modalities import build_modality_runtimes
from .models import ActiveContext, GateAction, RecallCue
from .recall import RecallEngine
from .scenarios import scenario_library
from .store import GraphMemoryStore
from .thought_gate import ThoughtGate
from .working_memory import WorkingMemoryLayer


class MemorySystemRuntime:
    def __init__(self, *, config_path: str | None = None, store_path: str | Path = "memory_system.db") -> None:
        self.config = load_config(config_path)
        self.store = GraphMemoryStore(store_path)
        self.bus = SharedMessageBus()
        self.modalities = build_modality_runtimes(self.config)
        self.event_integrator = EventIntegrator(self.config)
        self.recall_engine = RecallEngine(self.config)
        self.thought_gate = ThoughtGate()
        self.working_memory = WorkingMemoryLayer(self.config)
        self.memory_writer = MemoryWriter()
        self.consolidation = ConsolidationEngine()
        self.reconsolidation = ReconsolidationEngine()

    def run_scenario(self, name: str) -> dict:
        scenarios = scenario_library()
        if name not in scenarios:
            raise KeyError(f"Unknown scenario: {name}")
        ticks = scenarios[name]
        started_at = ticks[0]["timestamp"] if ticks else 0.0
        history: list[dict] = []
        for index, tick in enumerate(ticks):
            history.append(self._step_tick(tick, index))
        final_event = self.event_integrator.flush(ticks[-1]["timestamp"] + 1 if ticks else 0.0)
        if final_event is not None:
            self.memory_writer.apply(final_event, [], [], self.store, self.config)
        self.consolidation.run(self.store, ticks[-1]["timestamp"] + 1 if ticks else 0.0, self.config)
        self.reconsolidation.run(self.store, [], ticks[-1]["timestamp"] + 1 if ticks else 0.0, self.config)
        summary = {
            "scenario": name,
            "ticks": len(ticks),
            "events": [asdict(event) for event in self.store.list_event_memories()],
            "history": history,
            "inspect": self.store.inspect_snapshot(),
        }
        self.store.record_scenario_run(str(uuid4()), name, started_at, ticks[-1]["timestamp"] if ticks else 0.0, summary)
        return summary

    def inspect(self) -> dict:
        snapshot = self.store.inspect_snapshot()
        snapshot["working_memory"] = {
            memory_id: asdict(item) for memory_id, item in self.working_memory.items.items()
        }
        return snapshot

    def consolidate_now(self, now: float = 0.0) -> dict:
        self.consolidation.run(self.store, now, self.config)
        queued = [memory_id for memory_id, _, _ in self.store.pop_reconsolidation_candidates(limit=20)]
        self.reconsolidation.run(self.store, queued, now, self.config)
        return self.inspect()

    def get_scenario_ticks(self, name: str) -> list[dict]:
        scenarios = scenario_library()
        if name not in scenarios:
            raise KeyError(f"Unknown scenario: {name}")
        return scenarios[name]

    def step_tick(self, tick: dict, index: int) -> dict:
        return self._step_tick(tick, index)

    def flush_current_event(self, now: float) -> None:
        final_event = self.event_integrator.flush(now)
        if final_event is not None:
            self.memory_writer.apply(final_event, [], [], self.store, self.config)
        self.consolidation.run(self.store, now, self.config)
        self.reconsolidation.run(self.store, [], now, self.config)

    def close(self) -> None:
        self.store.close()

    def _step_tick(self, tick: dict, index: int) -> dict:
        timestamp = float(tick["timestamp"])
        location = tick.get("location", get_cfg(self.config, "runtime", "default_location", default="unknown"))
        goal_state = tick.get("goal_state", get_cfg(self.config, "runtime", "default_goal", default="monitor_environment"))
        thought = tick.get("thought", "")
        current_emotion = tick["modalities"].get("emotion", {}).get("feature_vector", {"arousal": 0.0, "valence": 0.0})
        active_context = ActiveContext(
            active_context_id=f"ctx-{timestamp}",
            current_time=timestamp,
            current_location=location,
            modality_summaries={},
            current_goal_state=goal_state,
            current_emotion_state=current_emotion,
            current_thought_summary=thought,
            global_salience=0.0,
            global_uncertainty=0.0,
            current_event_id=self.event_integrator.current.current_event_id,
            active_memory_ids=list(self.working_memory.items),
            context_labels=[location, *tick["modalities"].get("time_context", {}).get("context_labels", []), *tick["modalities"].get("emotion", {}).get("context_labels", [])],
        )
        first_pass = {}
        for modality_id, runtime in self.modalities.items():
            raw = tick["modalities"].get(modality_id, {"feature_vector": {}, "hypothesis": f"{modality_id}_idle"})
            observation = runtime.buffer.update(raw, timestamp)
            first_pass[modality_id] = runtime.context.step(observation, [], active_context, self.config)
        second_pass = {}
        provisional_messages = [signal.bus_message for signal in first_pass.values()]
        for modality_id, runtime in self.modalities.items():
            raw = tick["modalities"].get(modality_id, {"feature_vector": {}, "hypothesis": f"{modality_id}_idle"})
            observation = runtime.buffer.history[-1]
            bus_messages = [message for message in provisional_messages if message.source_modality != modality_id]
            second_pass[modality_id] = runtime.context.step(observation, bus_messages, active_context, self.config)
            self.bus.publish(second_pass[modality_id].bus_message)
        signals = [signal for signal in second_pass.values() if signal.modality_id in tick["modalities"]]
        active_context.modality_summaries = {signal.modality_id: signal.local_state_summary for signal in signals}
        active_context.global_salience = sum(signal.salience for signal in signals) / len(signals)
        active_context.global_uncertainty = sum(signal.uncertainty for signal in signals) / len(signals)
        event_update = self.event_integrator.step(signals, active_context, timestamp, self.config)
        active_context.current_event_id = event_update.snapshot.current_event_id
        cues: list[RecallCue] = [
            signal.recall_cue_proposal
            for signal in signals
            if (
                signal.recall_activation >= get_cfg(self.config, "thresholds", "recall_trigger", default=0.46)
                or signal.salience >= 0.55
                or signal.prediction_error >= 0.35
            )
        ]
        decisions = []
        candidates = []
        if self.store.list_event_memories() and (event_update.recall_trigger or cues):
            candidates = self.recall_engine.retrieve(
                cues,
                event_update.snapshot,
                self.working_memory.step([], timestamp, self.config),
                self.recall_engine.inhibition_levels,
                self.store,
                self.config,
            )
            for candidate in candidates:
                decisions.append(self.thought_gate.evaluate(candidate, active_context, event_update.snapshot, self.working_memory.step([], timestamp, self.config), self.config))
        wm_snapshot = self.working_memory.step(decisions, timestamp, self.config)
        promoted_ids = [decision.memory_id for decision in decisions if decision.action == GateAction.PROMOTE]
        event_update.snapshot.linked_active_memories = list(dict.fromkeys(event_update.snapshot.linked_active_memories + promoted_ids))
        if event_update.should_provisional_write or event_update.should_durable_write:
            self.memory_writer.apply(event_update.snapshot, signals, decisions, self.store, self.config)
        if event_update.finalized_event is not None:
            self.memory_writer.apply(event_update.finalized_event, signals, decisions, self.store, self.config)
        self.recall_engine.apply_feedback(decisions, active_context.context_labels, event_update.snapshot.current_event_id, self.store, timestamp)
        self.recall_engine.decay(self.config)
        if (index + 1) % get_cfg(self.config, "runtime", "consolidation_interval_ticks", default=8) == 0:
            self.consolidation.run(self.store, timestamp, self.config)
        if (index + 1) % get_cfg(self.config, "runtime", "reconsolidation_interval_ticks", default=6) == 0:
            queued = [memory_id for memory_id, _, _ in self.store.pop_reconsolidation_candidates(limit=20)]
            self.reconsolidation.run(self.store, queued, timestamp, self.config)
        return {
            "timestamp": timestamp,
            "event_state": event_update.snapshot.event_state.value,
            "event_id": event_update.snapshot.current_event_id,
            "hypothesis": event_update.snapshot.final_hypothesis,
            "write_pressure": event_update.snapshot.write_pressure,
            "recall_candidates": [candidate.memory_id for candidate in candidates],
            "decisions": [decision.action.value for decision in decisions],
            "working_memory": wm_snapshot.active_ids,
        }
