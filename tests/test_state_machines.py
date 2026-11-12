from multi_context_memory.models import EventState, RecallState
from multi_context_memory.state_machines import advance_event_state, advance_recall_state


def test_event_state_machine_core_transitions() -> None:
    state = advance_event_state(
        EventState.BACKGROUND,
        boundary_pressure=0.5,
        boundary_candidate_threshold=0.4,
        boundary_start_threshold=0.6,
        boundary_reset_threshold=0.2,
        write_pressure=0.0,
        provisional_write_threshold=0.4,
        new_evidence=False,
        explanation_stabilized=False,
        event_ended=False,
    )
    assert state == EventState.BOUNDARY_CANDIDATE
    state = advance_event_state(
        EventState.ACTIVE_EVENT,
        boundary_pressure=0.7,
        boundary_candidate_threshold=0.4,
        boundary_start_threshold=0.6,
        boundary_reset_threshold=0.2,
        write_pressure=0.7,
        provisional_write_threshold=0.4,
        new_evidence=False,
        explanation_stabilized=False,
        event_ended=False,
    )
    assert state == EventState.PROVISIONAL_WRITE
    state = advance_event_state(
        EventState.UPDATING_EVIDENCE,
        boundary_pressure=0.1,
        boundary_candidate_threshold=0.4,
        boundary_start_threshold=0.6,
        boundary_reset_threshold=0.2,
        write_pressure=0.7,
        provisional_write_threshold=0.4,
        new_evidence=True,
        explanation_stabilized=True,
        event_ended=True,
    )
    assert state == EventState.RESOLVED


def test_recall_state_machine_handles_reconsolidation_path() -> None:
    state = advance_recall_state(RecallState.DORMANT, cue_matched=True)
    assert state == RecallState.CUE_MATCHED
    state = advance_recall_state(state, selected_for_evaluation=True)
    assert state == RecallState.CANDIDATE_RECALL
    state = advance_recall_state(state, selected_for_evaluation=True)
    assert state == RecallState.EVALUATION
    state = advance_recall_state(state, intrusive=True)
    assert state == RecallState.RECONSOLIDATION_CANDIDATE
    state = advance_recall_state(state, reconsolidating=True)
    assert state == RecallState.RECONSOLIDATING
    state = advance_recall_state(state, reconsolidated=True)
    assert state == RecallState.DORMANT
