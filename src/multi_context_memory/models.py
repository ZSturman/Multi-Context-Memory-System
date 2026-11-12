from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ModalityType(StrEnum):
    VISION = "vision"
    AUDIO = "audio"
    TIME_CONTEXT = "time_context"
    EMOTION = "emotion"
    PROPRIO = "proprio"
    THOUGHT = "thought"


class EventState(StrEnum):
    BACKGROUND = "E0_BACKGROUND"
    BOUNDARY_CANDIDATE = "E1_BOUNDARY_CANDIDATE"
    ACTIVE_EVENT = "E2_ACTIVE_EVENT"
    PROVISIONAL_WRITE = "E3_PROVISIONAL_WRITE"
    UPDATING_EVIDENCE = "E4_UPDATING_EVIDENCE"
    RESOLVED = "E5_RESOLVED"
    UNRESOLVED = "E6_UNRESOLVED"
    CONSOLIDATED = "E7_CONSOLIDATED"


class RecallState(StrEnum):
    DORMANT = "S0_DORMANT"
    CUE_MATCHED = "S1_CUE_MATCHED"
    CANDIDATE_RECALL = "S2_CANDIDATE_RECALL"
    EVALUATION = "S3_EVALUATION"
    ACTIVE_WORKING_MEMORY = "S4_ACTIVE_IN_WORKING_MEMORY"
    WEAK_ACTIVE = "S5_WEAK_ACTIVE_MONITOR"
    DISMISSED = "S6_DISMISSED"
    BLOCKED = "S7_BLOCKED_REFRACTORY"
    RECONSOLIDATION_CANDIDATE = "S8_RECONSOLIDATION_CANDIDATE"
    INTEGRATED = "S9_INTEGRATED"
    FADING = "S10_FADING"
    RECONSOLIDATING = "S11_RECONSOLIDATING"


class GateAction(StrEnum):
    PROMOTE = "promote_to_active"
    KEEP_WEAK = "keep_weakly_active"
    DISMISS = "dismiss"
    TEMP_BLOCK = "temporarily_block"
    RECONSOLIDATE = "mark_for_reconsolidation"


@dataclass(slots=True)
class ModalityObservation:
    modality_id: str
    feature_vector: dict[str, float]
    change_magnitude: float
    rate_of_change: float
    stability: float
    timestamp: float
    raw_input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CrossModalMessage:
    source_modality: str
    timestamp: float
    change_score: float
    salience_score: float
    uncertainty_score: float
    confidence_score: float
    spatial_hint: str | None = None
    temporal_hint: str | None = None
    emotion_hint: str | None = None
    goal_relevance: float = 0.0
    anomaly_flag: bool = False
    candidate_category: str | None = None


@dataclass(slots=True)
class RecallCue:
    modality_id: str
    feature_vector: dict[str, float]
    salience: float
    uncertainty: float
    goal_relevance: float
    context_labels: list[str]
    event_id: str | None = None
    hypothesis: str = ""


@dataclass(slots=True)
class ModalitySignal:
    modality_id: str
    current_feature_vector: dict[str, float]
    local_hypothesis: str
    confidence: float
    uncertainty: float
    novelty: float
    prediction_error: float
    salience: float
    recall_activation: float
    write_pressure: float
    update_pressure: float
    cross_modal_signal_summary: dict[str, float]
    goal_relevance: float
    local_state_summary: str
    recall_cue_proposal: RecallCue
    write_proposal: dict[str, Any]
    update_proposal: dict[str, Any]
    bus_message: CrossModalMessage
    timestamp: float


@dataclass(slots=True)
class ActiveContext:
    active_context_id: str
    current_time: float
    current_location: str
    modality_summaries: dict[str, str]
    current_goal_state: str
    current_emotion_state: dict[str, float]
    current_thought_summary: str
    global_salience: float
    global_uncertainty: float
    current_event_id: str | None
    active_memory_ids: list[str]
    context_labels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EventSnapshot:
    current_event_id: str | None
    event_state: EventState
    event_start_time: float | None
    event_end_time: float | None
    active_modalities: list[str]
    dominant_modality: str | None
    global_salience: float
    global_uncertainty: float
    global_novelty: float
    event_hypothesis: str
    initial_hypothesis: str
    final_hypothesis: str
    hypothesis_confidence: float
    resolution_progress: float
    event_boundary_pressure: float
    write_pressure: float
    recall_relevance: float
    linked_active_memories: list[str]
    context_labels: list[str] = field(default_factory=list)
    goal_state: str = ""
    emotion_state: dict[str, float] = field(default_factory=dict)
    modality_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    boundary_candidate_ticks: int = 0
    quiet_ticks: int = 0
    stored: bool = False


@dataclass(slots=True)
class EventUpdate:
    snapshot: EventSnapshot
    finalized_event: EventSnapshot | None
    started_new_event: bool
    should_provisional_write: bool
    should_durable_write: bool
    recall_trigger: bool


@dataclass(slots=True)
class RecallCandidate:
    memory_id: str
    event_id: str
    source_trace_id: str | None
    modality_type: str
    cue_similarity: float
    event_activation: float
    graph_activation: float
    context_fit: float
    goal_relevance: float
    total_score: float
    state: RecallState
    event_summary: str
    context_labels: list[str]
    arousal_score: float
    importance_score: float
    memory_strength: float
    already_integrated: bool = False
    already_dismissed: bool = False
    recurrent_mismatch: bool = False
    regulation: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class GateDecision:
    memory_id: str
    action: GateAction
    score: float
    reason: str
    resulting_state: RecallState
    explanatory_value: float
    predictive_value: float
    context_fit: float
    uncertainty_reduction: float
    emotional_distortion_risk: float
    candidate: RecallCandidate


@dataclass(slots=True)
class WorkingMemoryItem:
    memory_id: str
    current_activation: float
    sensory_residue: float
    context_residue: float
    emotional_residue: float
    causal_gist_residue: float
    action_implication_residue: float
    activation_start_time: float
    fade_profile: dict[str, float]
    integration_status: str
    last_updated: float


@dataclass(slots=True)
class WorkingMemorySnapshot:
    items: dict[str, WorkingMemoryItem]
    active_ids: list[str]
    integrated_ids: list[str]
    dismissed_ids: list[str]


@dataclass(slots=True)
class EventMemory:
    event_id: str
    start_time: float
    end_time: float
    duration: float
    location_tag: str
    dominant_modality: str
    event_summary: str
    initial_hypothesis: str
    final_hypothesis: str
    novelty_score: float
    importance_score: float
    arousal_score: float
    uncertainty_peak: float
    uncertainty_reduction: float
    resolution_score: float
    confidence: float
    memory_strength: float
    linked_modality_trace_ids: list[str]
    linked_concept_ids: list[str]
    linked_context_ids: list[str]
    linked_thought_ids: list[str]
    resolved_flag: bool
    provisional_flag: bool = False
    context_labels: list[str] = field(default_factory=list)
    inhibition_level: float = 0.0
    dismissal_imprint: float = 0.0
    context_saturation: float = 0.0
    reconsolidation_pressure: float = 0.0


@dataclass(slots=True)
class ModalityTrace:
    trace_id: str
    event_id: str
    modality_type: str
    feature_signature: dict[str, float]
    peak_change: float
    novelty_score: float
    uncertainty_start: float
    uncertainty_end: float
    confidence: float
    salience: float
    timestamp_offset: float
    cue_strength: float
    trace_strength: float
    local_hypothesis: str = ""


@dataclass(slots=True)
class ConceptNode:
    concept_id: str
    label: str
    concept_type: str
    prototype_signature: dict[str, float]
    associated_modalities: list[str]
    semantic_strength: float
    linked_event_ids: list[str]


@dataclass(slots=True)
class ContextNode:
    context_id: str
    label: str
    context_type: str
    strength: float
    linked_event_ids: list[str]


@dataclass(slots=True)
class ThoughtNode:
    thought_id: str
    text_or_label: str
    role: str
    confidence: float
    linked_event_ids: list[str]


@dataclass(slots=True)
class Edge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    direction: str
    evidence_strength: float
    last_activated: float
    decay_rate: float


@dataclass(slots=True)
class WriteResult:
    event_id: str | None
    trace_ids: list[str]
    created: bool
    durable: bool
    resolved: bool


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
