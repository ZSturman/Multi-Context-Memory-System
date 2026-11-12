from __future__ import annotations

from .models import EventState, RecallState


def advance_event_state(
    current: EventState,
    *,
    boundary_pressure: float,
    boundary_candidate_threshold: float,
    boundary_start_threshold: float,
    boundary_reset_threshold: float,
    write_pressure: float,
    provisional_write_threshold: float,
    new_evidence: bool,
    explanation_stabilized: bool,
    event_ended: bool,
    reopened: bool = False,
    consolidated: bool = False,
) -> EventState:
    if consolidated and current in {EventState.RESOLVED, EventState.UNRESOLVED}:
        return EventState.CONSOLIDATED
    if reopened and current == EventState.UNRESOLVED:
        return EventState.UPDATING_EVIDENCE
    if current == EventState.BACKGROUND:
        if boundary_pressure >= boundary_candidate_threshold:
            return EventState.BOUNDARY_CANDIDATE
        return current
    if current == EventState.BOUNDARY_CANDIDATE:
        if boundary_pressure < boundary_reset_threshold:
            return EventState.BACKGROUND
        if boundary_pressure >= boundary_start_threshold:
            return EventState.ACTIVE_EVENT
        return current
    if current == EventState.ACTIVE_EVENT:
        if write_pressure >= provisional_write_threshold:
            return EventState.PROVISIONAL_WRITE
        if event_ended and explanation_stabilized:
            return EventState.RESOLVED
        if event_ended:
            return EventState.UNRESOLVED
        return current
    if current == EventState.PROVISIONAL_WRITE:
        if new_evidence:
            return EventState.UPDATING_EVIDENCE
        if event_ended and explanation_stabilized:
            return EventState.RESOLVED
        if event_ended:
            return EventState.UNRESOLVED
        return current
    if current == EventState.UPDATING_EVIDENCE:
        if explanation_stabilized:
            return EventState.RESOLVED
        if event_ended:
            return EventState.UNRESOLVED
        return current
    return current


def advance_recall_state(
    current: RecallState,
    *,
    cue_matched: bool = False,
    selected_for_evaluation: bool = False,
    outcompeted: bool = False,
    inhibited: bool = False,
    relevant: bool = False,
    tentative: bool = False,
    intrusive: bool = False,
    integrated: bool = False,
    fading: bool = False,
    unsupported: bool = False,
    reconsolidating: bool = False,
    reconsolidated: bool = False,
    inhibition_decayed: bool = False,
    strong_fresh_evidence: bool = False,
) -> RecallState:
    if current == RecallState.DORMANT:
        if cue_matched:
            return RecallState.CUE_MATCHED
        return current
    if current == RecallState.CUE_MATCHED:
        if inhibited:
            return RecallState.BLOCKED
        if selected_for_evaluation:
            return RecallState.CANDIDATE_RECALL
        if outcompeted:
            return RecallState.DORMANT
        return current
    if current == RecallState.CANDIDATE_RECALL:
        if selected_for_evaluation:
            return RecallState.EVALUATION
        if outcompeted:
            return RecallState.DORMANT
        return current
    if current == RecallState.EVALUATION:
        if intrusive:
            return RecallState.RECONSOLIDATION_CANDIDATE
        if relevant:
            return RecallState.ACTIVE_WORKING_MEMORY
        if tentative:
            return RecallState.WEAK_ACTIVE
        return RecallState.DISMISSED
    if current == RecallState.ACTIVE_WORKING_MEMORY:
        if intrusive:
            return RecallState.RECONSOLIDATION_CANDIDATE
        if integrated:
            return RecallState.INTEGRATED
        if unsupported:
            return RecallState.DISMISSED
        if fading:
            return RecallState.FADING
        return current
    if current == RecallState.WEAK_ACTIVE:
        if relevant:
            return RecallState.ACTIVE_WORKING_MEMORY
        if unsupported:
            return RecallState.DISMISSED
        if fading:
            return RecallState.FADING
        return current
    if current == RecallState.DISMISSED:
        return RecallState.BLOCKED
    if current == RecallState.BLOCKED:
        if inhibition_decayed:
            return RecallState.DORMANT
        if strong_fresh_evidence:
            return RecallState.CUE_MATCHED
        return current
    if current == RecallState.RECONSOLIDATION_CANDIDATE:
        if reconsolidating:
            return RecallState.RECONSOLIDATING
        return current
    if current == RecallState.RECONSOLIDATING:
        if reconsolidated:
            return RecallState.DORMANT
        return current
    if current == RecallState.INTEGRATED:
        if fading:
            return RecallState.FADING
        return current
    if current == RecallState.FADING:
        return RecallState.BLOCKED
    return current
