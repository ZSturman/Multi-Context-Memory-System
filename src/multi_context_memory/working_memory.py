from __future__ import annotations

from copy import deepcopy

from .config import get_cfg
from .formulas import clamp, decay_value
from .models import GateAction, GateDecision, WorkingMemoryItem, WorkingMemorySnapshot


class WorkingMemoryLayer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.items: dict[str, WorkingMemoryItem] = {}
        self.integrated_ids: set[str] = set()
        self.dismissed_ids: set[str] = set()

    def step(self, decisions: list[GateDecision], now: float, config: dict) -> WorkingMemorySnapshot:
        self._decay(now, config)
        fade_profile = get_cfg(config, "working_memory", "fade", default={})
        for decision in decisions:
            memory_id = decision.memory_id
            if decision.action == GateAction.PROMOTE:
                self.items[memory_id] = WorkingMemoryItem(
                    memory_id=memory_id,
                    current_activation=clamp(max(self.items.get(memory_id, WorkingMemoryItem(memory_id, 0, 0, 0, 0, 0, 0, now, fade_profile, "new", now)).current_activation, decision.score)),
                    sensory_residue=decision.candidate.cue_similarity,
                    context_residue=decision.context_fit,
                    emotional_residue=decision.candidate.arousal_score,
                    causal_gist_residue=decision.explanatory_value,
                    action_implication_residue=decision.predictive_value,
                    activation_start_time=now,
                    fade_profile=deepcopy(fade_profile),
                    integration_status="active",
                    last_updated=now,
                )
                self.integrated_ids.add(memory_id)
            elif decision.action == GateAction.KEEP_WEAK:
                self.items[memory_id] = WorkingMemoryItem(
                    memory_id=memory_id,
                    current_activation=decision.score * 0.8,
                    sensory_residue=decision.candidate.cue_similarity * 0.7,
                    context_residue=decision.context_fit * 0.8,
                    emotional_residue=decision.candidate.arousal_score * 0.7,
                    causal_gist_residue=decision.explanatory_value * 0.8,
                    action_implication_residue=decision.predictive_value * 0.7,
                    activation_start_time=now,
                    fade_profile=deepcopy(fade_profile),
                    integration_status="weak",
                    last_updated=now,
                )
            elif decision.action in {GateAction.DISMISS, GateAction.TEMP_BLOCK, GateAction.RECONSOLIDATE}:
                self.dismissed_ids.add(memory_id)
                self.items.pop(memory_id, None)
        while len(self.items) > get_cfg(config, "working_memory", "max_items", default=6):
            weakest = min(self.items.values(), key=lambda item: item.current_activation)
            self.items.pop(weakest.memory_id, None)
        return WorkingMemorySnapshot(
            items=deepcopy(self.items),
            active_ids=list(self.items),
            integrated_ids=sorted(self.integrated_ids),
            dismissed_ids=sorted(self.dismissed_ids),
        )

    def _decay(self, now: float, config: dict) -> None:
        expired: list[str] = []
        for memory_id, item in self.items.items():
            dt = max(1.0, now - item.last_updated)
            item.current_activation = decay_value(item.current_activation, get_cfg(config, "working_memory", "activation_decay", default=0.10), dt)
            item.sensory_residue = decay_value(item.sensory_residue, item.fade_profile.get("sensory_residue", 0.24), dt)
            item.context_residue = decay_value(item.context_residue, item.fade_profile.get("context_residue", 0.18), dt)
            item.emotional_residue = decay_value(item.emotional_residue, item.fade_profile.get("emotional_residue", 0.14), dt)
            item.causal_gist_residue = decay_value(item.causal_gist_residue, item.fade_profile.get("causal_gist_residue", 0.09), dt)
            item.action_implication_residue = decay_value(item.action_implication_residue, item.fade_profile.get("action_implication_residue", 0.07), dt)
            item.integration_status = "fading" if item.current_activation < 0.35 else item.integration_status
            item.last_updated = now
            if item.current_activation <= get_cfg(config, "thresholds", "fading_drop", default=0.12):
                expired.append(memory_id)
        for memory_id in expired:
            self.items.pop(memory_id, None)

