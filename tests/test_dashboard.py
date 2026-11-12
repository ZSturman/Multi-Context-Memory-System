"""Tests for the Streamlit dashboard helpers and data pipelines.

These tests exercise:
  A. Pure helper functions (_state_indicator, _build_graph_figure, _metrics_chart,
     _state_transition_graph) that return plotly Figures.
  B. Label dictionaries covering every enum member.
  C. Data-pipeline integration: runtime.run_scenario() → dashboard-expected dict
     structures → pandas DataFrame construction (same code paths as the dashboard).
  D. Edge cases: empty data, single tick, missing fields.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from multi_context_memory.models import EventState, RecallState
from multi_context_memory.runtime import MemorySystemRuntime
from multi_context_memory.scenarios import scenario_library
from multi_context_memory.store import GraphMemoryStore

# ---------------------------------------------------------------------------
# Import dashboard helpers.  The module performs top-level Streamlit calls
# (st.set_page_config, sidebar, tabs) at import time, so we must patch
# streamlit before importing it.
# ---------------------------------------------------------------------------
import unittest.mock as _mock

# Build a stub st module whose .tabs() / .columns() return the right number
# of context-manager mocks so that ``tab_run, tab_step, … = st.tabs(…)`` works.

class _CtxMock(_mock.MagicMock):
    """MagicMock that also works as a context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass

def _make_n_ctx(*args, **kwargs):
    """Return N context-manager mocks (for st.tabs / st.columns)."""
    # If first arg is a list, use its length; else if int, use it directly
    if args and isinstance(args[0], (list, tuple)):
        n = len(args[0])
    elif args and isinstance(args[0], int):
        n = args[0]
    else:
        n = 4
    return [_CtxMock() for _ in range(n)]

_st_stub = _mock.MagicMock()
_st_stub.session_state = {}
_st_stub.tabs.side_effect = _make_n_ctx
_st_stub.columns.side_effect = _make_n_ctx
_st_stub.sidebar = _CtxMock()
# Buttons must return False so the dashboard doesn't try to run scenarios at import
_st_stub.button.return_value = False
# selectbox must return a valid scenario name (used by run/step tabs)
_st_stub.selectbox.return_value = list(scenario_library().keys())[0]
# progress/spinner/expander need to be safe context managers
_st_stub.spinner.return_value = _CtxMock()
_st_stub.expander.return_value = _CtxMock()

import sys as _sys
_sys.modules["streamlit"] = _st_stub

# Now safe to import the helper functions and constants.
from multi_context_memory.dashboard import (
    EVENT_STATE_EDGES,
    EVENT_STATE_LABELS,
    EVENT_STATES_ORDER,
    RECALL_STATE_EDGES,
    RECALL_STATE_LABELS,
    RECALL_STATES_ORDER,
    _build_graph_figure,
    _metrics_chart,
    _state_indicator,
    _state_transition_graph,
)


# ===== A. Pure helper-function tests =========================================

class TestStateIndicator:
    """_state_indicator returns a plotly Bar figure with correct colours."""

    def test_basic_output_is_figure(self):
        fig = _state_indicator(
            EVENT_STATES_ORDER,
            EventState.ACTIVE_EVENT.value,
            {EventState.BACKGROUND.value, EventState.BOUNDARY_CANDIDATE.value},
            EVENT_STATE_LABELS,
        )
        assert isinstance(fig, go.Figure)

    def test_bar_count_matches_states(self):
        fig = _state_indicator(EVENT_STATES_ORDER, "", set(), EVENT_STATE_LABELS)
        bar_trace = fig.data[0]
        assert len(bar_trace.y) == len(EVENT_STATES_ORDER)

    def test_current_state_is_green(self):
        current = EventState.ACTIVE_EVENT.value
        fig = _state_indicator(EVENT_STATES_ORDER, current, set(), EVENT_STATE_LABELS)
        bar_trace = fig.data[0]
        idx = [s.value for s in EVENT_STATES_ORDER].index(current)
        assert bar_trace.marker.color[idx] == "#22c55e"

    def test_visited_state_is_blue(self):
        visited = {EventState.BACKGROUND.value}
        fig = _state_indicator(
            EVENT_STATES_ORDER, EventState.ACTIVE_EVENT.value, visited, EVENT_STATE_LABELS
        )
        bar_trace = fig.data[0]
        idx = [s.value for s in EVENT_STATES_ORDER].index(EventState.BACKGROUND.value)
        assert bar_trace.marker.color[idx] == "#3b82f6"

    def test_unvisited_state_is_gray(self):
        fig = _state_indicator(EVENT_STATES_ORDER, "", set(), EVENT_STATE_LABELS)
        bar_trace = fig.data[0]
        assert all(c == "#d1d5db" for c in bar_trace.marker.color)

    def test_all_visited_plus_current(self):
        all_vals = {s.value for s in EVENT_STATES_ORDER}
        current = EventState.CONSOLIDATED.value
        fig = _state_indicator(EVENT_STATES_ORDER, current, all_vals, EVENT_STATE_LABELS)
        bar_trace = fig.data[0]
        idx = [s.value for s in EVENT_STATES_ORDER].index(current)
        # Current takes precedence over visited
        assert bar_trace.marker.color[idx] == "#22c55e"

    def test_recall_states_indicator(self):
        fig = _state_indicator(
            RECALL_STATES_ORDER,
            RecallState.EVALUATION.value,
            {RecallState.DORMANT.value, RecallState.CUE_MATCHED.value},
            RECALL_STATE_LABELS,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].y) == len(RECALL_STATES_ORDER)


class TestBuildGraphFigure:
    """_build_graph_figure returns valid plotly Figure for various store states."""

    def test_empty_store_shows_annotation(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            fig = _build_graph_figure(runtime)
            assert isinstance(fig, go.Figure)
            annotations = fig.layout.annotations
            assert annotations and "No data" in annotations[0].text
        finally:
            runtime.close()

    def test_after_scenario_has_nodes(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            runtime.run_scenario("surprise_then_resolution")
            fig = _build_graph_figure(runtime)
            assert isinstance(fig, go.Figure)
            # At least one scatter trace with nodes
            scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
            assert scatter_traces
            # Should have some x coordinates (nodes)
            assert any(len(t.x) > 0 for t in scatter_traces)
        finally:
            runtime.close()

    def test_after_abstraction_has_concepts(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            runtime.run_scenario("repeated_pattern_abstraction")
            fig = _build_graph_figure(runtime)
            scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
            # Should have event nodes (blue) and concept nodes (orange)
            all_colors = []
            for t in scatter_traces:
                if t.marker and t.marker.color:
                    if isinstance(t.marker.color, str):
                        all_colors.append(t.marker.color)
                    else:
                        all_colors.extend(t.marker.color)
            assert "#3b82f6" in all_colors  # events
            assert "#f59e0b" in all_colors  # concepts
        finally:
            runtime.close()


class TestMetricsChart:
    """_metrics_chart returns plotly Bar figure from tick history dicts."""

    def test_empty_history(self):
        fig = _metrics_chart([], [])
        assert isinstance(fig, go.Figure)

    def test_single_tick(self):
        history = [{"timestamp": 0.0, "write_pressure": 0.5, "event_state": "E0_BACKGROUND"}]
        fig = _metrics_chart(history, [])
        assert isinstance(fig, go.Figure)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1
        assert list(bar_traces[0].x) == [0.0]
        assert list(bar_traces[0].y) == [0.5]

    def test_multiple_ticks_x_axis_matches_timestamps(self):
        history = [
            {"timestamp": 0.0, "write_pressure": 0.1, "event_state": "E0_BACKGROUND"},
            {"timestamp": 1.0, "write_pressure": 0.5, "event_state": "E1_BOUNDARY_CANDIDATE"},
            {"timestamp": 2.0, "write_pressure": 0.9, "event_state": "E2_ACTIVE_EVENT"},
        ]
        fig = _metrics_chart(history, [])
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert list(bar_traces[0].x) == [0.0, 1.0, 2.0]

    def test_y_axis_range_is_0_to_1(self):
        history = [{"timestamp": 0.0, "write_pressure": 0.5, "event_state": "E0_BACKGROUND"}]
        fig = _metrics_chart(history, [])
        assert tuple(fig.layout.yaxis.range) == (0, 1)


class TestStateTransitionGraph:
    """_state_transition_graph returns plotly Figure w/ nodes and edge arrows."""

    def test_event_state_graph_has_correct_node_count(self):
        fig = _state_transition_graph(
            EVENT_STATE_EDGES, EVENT_STATE_LABELS, set(), None
        )
        assert isinstance(fig, go.Figure)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # Each node is its own scatter trace
        assert len(scatter_traces) == len(EVENT_STATE_LABELS)

    def test_recall_state_graph_has_correct_node_count(self):
        fig = _state_transition_graph(
            RECALL_STATE_EDGES, RECALL_STATE_LABELS, set(), None
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == len(RECALL_STATE_LABELS)

    def test_edge_annotations_present(self):
        fig = _state_transition_graph(
            EVENT_STATE_EDGES, EVENT_STATE_LABELS, set(), None
        )
        # Each edge becomes an annotation (arrow)
        annotations = fig.layout.annotations
        assert annotations is not None
        assert len(annotations) >= len(EVENT_STATE_EDGES)

    def test_current_node_is_green(self):
        current = "E2_ACTIVE_EVENT"
        fig = _state_transition_graph(
            EVENT_STATE_EDGES, EVENT_STATE_LABELS, set(), current
        )
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.hovertext == current:
                assert trace.marker.color == "#22c55e"

    def test_visited_node_is_blue(self):
        visited = {"E0_BACKGROUND", "E1_BOUNDARY_CANDIDATE"}
        fig = _state_transition_graph(
            EVENT_STATE_EDGES, EVENT_STATE_LABELS, visited, None
        )
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.hovertext in visited:
                assert trace.marker.color == "#3b82f6"

    def test_unvisited_node_is_gray(self):
        fig = _state_transition_graph(
            EVENT_STATE_EDGES, EVENT_STATE_LABELS, set(), None
        )
        for trace in fig.data:
            if isinstance(trace, go.Scatter):
                assert trace.marker.color == "#d1d5db"

    def test_empty_labels(self):
        fig = _state_transition_graph([], {}, set(), None)
        assert isinstance(fig, go.Figure)


# ===== B. Label dictionary tests =============================================

class TestLabelDictionaries:
    """EVENT_STATE_LABELS and RECALL_STATE_LABELS cover every enum member."""

    def test_event_state_labels_cover_all_members(self):
        for state in EventState:
            assert state.value in EVENT_STATE_LABELS, f"Missing label for {state}"

    def test_recall_state_labels_cover_all_members(self):
        for state in RecallState:
            assert state.value in RECALL_STATE_LABELS, f"Missing label for {state}"

    def test_event_state_labels_have_expected_format(self):
        for val, label in EVENT_STATE_LABELS.items():
            # Label should start with the Exx prefix
            assert val.startswith("E"), f"Unexpected value format: {val}"
            assert label  # Non-empty

    def test_recall_state_labels_have_expected_format(self):
        for val, label in RECALL_STATE_LABELS.items():
            assert val.startswith("S"), f"Unexpected value format: {val}"
            assert label


# ===== C. Data-pipeline integration tests ====================================

class TestScenarioResultStructure:
    """Verify run_scenario() returns dicts the dashboard can consume."""

    REQUIRED_RESULT_KEYS = {"scenario", "ticks", "events", "history", "inspect"}
    REQUIRED_HISTORY_KEYS = {
        "timestamp", "event_state", "event_id", "hypothesis",
        "write_pressure", "recall_candidates", "decisions", "working_memory",
    }

    @pytest.fixture()
    def runtime(self, tmp_path: Path):
        rt = MemorySystemRuntime(store_path=tmp_path / "test.db")
        yield rt
        rt.close()

    @pytest.mark.parametrize("scenario_name", sorted(scenario_library().keys()))
    def test_result_keys(self, runtime: MemorySystemRuntime, scenario_name: str):
        result = runtime.run_scenario(scenario_name)
        assert self.REQUIRED_RESULT_KEYS.issubset(result.keys()), \
            f"Missing keys: {self.REQUIRED_RESULT_KEYS - result.keys()}"

    @pytest.mark.parametrize("scenario_name", sorted(scenario_library().keys()))
    def test_history_entry_keys(self, runtime: MemorySystemRuntime, scenario_name: str):
        result = runtime.run_scenario(scenario_name)
        for i, entry in enumerate(result["history"]):
            missing = self.REQUIRED_HISTORY_KEYS - entry.keys()
            assert not missing, f"Tick {i} missing keys: {missing}"

    @pytest.mark.parametrize("scenario_name", sorted(scenario_library().keys()))
    def test_ticks_count_matches_history_length(self, runtime: MemorySystemRuntime, scenario_name: str):
        result = runtime.run_scenario(scenario_name)
        assert result["ticks"] == len(result["history"])


class TestEventsDataFrame:
    """Events from run_scenario → DataFrame (same path as dashboard tab_run)."""

    @pytest.fixture()
    def result(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            return runtime.run_scenario("surprise_then_resolution")
        finally:
            runtime.close()

    def test_events_df_construction_no_error(self, result):
        events = result["events"]
        if not events:
            pytest.skip("No events stored (scenario may not produce events)")
        df = pd.DataFrame([
            {
                "event_id": ev["event_id"][:12] + "…",
                "hypothesis": ev["final_hypothesis"],
                "location": ev["location_tag"],
                "novelty": round(ev["novelty_score"], 3),
                "importance": round(ev["importance_score"], 3),
                "arousal": round(ev["arousal_score"], 3),
                "confidence": round(ev["confidence"], 3),
                "strength": round(ev["memory_strength"], 3),
                "resolved": ev["resolved_flag"],
                "inhibition": round(ev["inhibition_level"], 3),
                "reconsolidation_p": round(ev["reconsolidation_pressure"], 3),
            }
            for ev in events
        ])
        assert not df.empty
        assert "event_id" in df.columns
        assert "hypothesis" in df.columns

    def test_events_all_numeric_columns_are_float(self, result):
        events = result["events"]
        if not events:
            pytest.skip("No events stored")
        for ev in events:
            for key in ("novelty_score", "importance_score", "arousal_score",
                        "confidence", "memory_strength", "inhibition_level",
                        "reconsolidation_pressure"):
                assert isinstance(ev[key], (int, float)), f"{key} is {type(ev[key])}"


class TestGraphInspectorDataFrames:
    """Memory Graph tab data: events, concepts, edges, traces → DataFrames."""

    @pytest.fixture()
    def populated_store(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        runtime.run_scenario("repeated_pattern_abstraction")
        yield runtime.store
        runtime.close()

    def test_events_data_df(self, populated_store: GraphMemoryStore):
        events = populated_store.list_event_memories()
        assert events  # Should have events after this scenario
        events_data = []
        for ev in events:
            events_data.append({
                "event_id": ev.event_id[:16],
                "hypothesis": ev.final_hypothesis,
                "location": ev.location_tag,
                "start": ev.start_time,
                "end": ev.end_time,
                "novelty": round(ev.novelty_score, 3),
                "importance": round(ev.importance_score, 3),
                "arousal": round(ev.arousal_score, 3),
                "confidence": round(ev.confidence, 3),
                "strength": round(ev.memory_strength, 3),
                "resolved": ev.resolved_flag,
                "provisional": ev.provisional_flag,
                "labels": ", ".join(ev.context_labels),
                "inhibition": round(ev.inhibition_level, 3),
                "dismissal": round(ev.dismissal_imprint, 3),
                "saturation": round(ev.context_saturation, 3),
                "recon_pressure": round(ev.reconsolidation_pressure, 3),
            })
        df = pd.DataFrame(events_data)
        assert not df.empty
        assert "hypothesis" in df.columns

    def test_concepts_df(self, populated_store: GraphMemoryStore):
        concepts = populated_store.list_concept_nodes()
        if not concepts:
            pytest.skip("No concepts formed")
        concepts_data = []
        for cn in concepts:
            concepts_data.append({
                "concept_id": cn.concept_id[:16],
                "label": cn.label,
                "type": cn.concept_type,
                "semantic_strength": round(cn.semantic_strength, 3),
                "modalities": ", ".join(cn.associated_modalities),
                "linked_events": len(cn.linked_event_ids),
            })
        df = pd.DataFrame(concepts_data)
        assert not df.empty
        assert "label" in df.columns

    def test_edges_df(self, populated_store: GraphMemoryStore):
        edges = populated_store.list_edges()
        assert edges
        edges_data = []
        for edge in edges[:50]:
            edges_data.append({
                "source": edge.source_id[:12],
                "target": edge.target_id[:12],
                "type": edge.edge_type,
                "weight": round(edge.weight, 3),
                "direction": edge.direction,
                "evidence": round(edge.evidence_strength, 3),
                "decay_rate": round(edge.decay_rate, 3),
            })
        df = pd.DataFrame(edges_data)
        assert not df.empty

    def test_traces_df_per_event(self, populated_store: GraphMemoryStore):
        events = populated_store.list_event_memories()
        found_traces = False
        for ev in events:
            traces = populated_store.get_traces_for_event(ev.event_id)
            if traces:
                found_traces = True
                traces_data = []
                for tr in traces:
                    traces_data.append({
                        "trace_id": tr.trace_id[:12],
                        "modality": tr.modality_type,
                        "hypothesis": tr.local_hypothesis,
                        "features": str(tr.feature_signature),
                        "peak_change": round(tr.peak_change, 3),
                        "novelty": round(tr.novelty_score, 3),
                        "salience": round(tr.salience, 3),
                        "cue_strength": round(tr.cue_strength, 3),
                        "trace_strength": round(tr.trace_strength, 3),
                        "confidence": round(tr.confidence, 3),
                    })
                df = pd.DataFrame(traces_data)
                assert not df.empty
                assert "modality" in df.columns
        assert found_traces, "Expected at least one event to have traces"


class TestStateTransitionLog:
    """State Machines tab: build transition log from history → DataFrame."""

    @pytest.fixture()
    def history(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            result = runtime.run_scenario("surprise_then_resolution")
            return result["history"]
        finally:
            runtime.close()

    def test_transition_log_df(self, history):
        transitions = []
        prev_state = None
        for h in history:
            if h["event_state"] != prev_state:
                transitions.append({
                    "timestamp": h["timestamp"],
                    "from": prev_state or "—",
                    "to": h["event_state"],
                    "hypothesis": h["hypothesis"][:40],
                    "write_pressure": round(h["write_pressure"], 3),
                })
                prev_state = h["event_state"]
        assert transitions
        df = pd.DataFrame(transitions)
        assert "timestamp" in df.columns
        assert "to" in df.columns

    def test_transition_log_starts_with_initial_state(self, history):
        assert history[0]["event_state"], "First tick should have a state"


# ===== D. Edge cases =========================================================

class TestEmptyDataEdgeCases:
    """Dashboard handles empty/no-data states without crashing."""

    def test_empty_events_list_df(self):
        df = pd.DataFrame([])
        assert df.empty

    def test_empty_concepts_list_df(self):
        df = pd.DataFrame([])
        assert df.empty

    def test_empty_edges_list_df(self):
        df = pd.DataFrame([])
        assert df.empty

    def test_metrics_chart_empty(self):
        fig = _metrics_chart([], [])
        assert isinstance(fig, go.Figure)

    def test_state_indicator_with_no_current(self):
        fig = _state_indicator(EVENT_STATES_ORDER, "", set(), EVENT_STATE_LABELS)
        assert isinstance(fig, go.Figure)

    def test_build_graph_empty_store(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            fig = _build_graph_figure(runtime)
            assert isinstance(fig, go.Figure)
        finally:
            runtime.close()

    def test_reconsolidation_queue_empty(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            recon = runtime.store.pop_reconsolidation_candidates(limit=20)
            assert recon == [] or recon == ()
            # Dashboard wraps this in: pd.DataFrame(recon_data) where recon_data is []
            if not recon:
                df = pd.DataFrame([])
                assert df.empty
        finally:
            runtime.close()


class TestStepThroughFlow:
    """Simulate the Step Through tab's tick-by-tick execution."""

    def test_step_tick_returns_expected_keys(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            ticks = runtime.get_scenario_ticks("surprise_then_resolution")
            result = runtime.step_tick(ticks[0], 0)
            required = {"timestamp", "event_state", "event_id", "hypothesis",
                        "write_pressure", "recall_candidates", "decisions", "working_memory"}
            assert required.issubset(result.keys())
        finally:
            runtime.close()

    def test_full_step_through_no_crash(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            ticks = runtime.get_scenario_ticks("surprise_then_resolution")
            history = []
            for i, tick in enumerate(ticks):
                result = runtime.step_tick(tick, i)
                history.append(result)
            # Flush like the dashboard does
            runtime.flush_current_event(ticks[-1]["timestamp"] + 1)

            # Verify all history entries can build charts
            fig = _metrics_chart(history, [])
            assert isinstance(fig, go.Figure)

            # State indicator from last tick
            visited = {h["event_state"] for h in history}
            fig2 = _state_indicator(
                EVENT_STATES_ORDER, history[-1]["event_state"], visited, EVENT_STATE_LABELS
            )
            assert isinstance(fig2, go.Figure)
        finally:
            runtime.close()

    def test_flush_then_inspect(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            ticks = runtime.get_scenario_ticks("surprise_then_resolution")
            for i, tick in enumerate(ticks):
                runtime.step_tick(tick, i)
            runtime.flush_current_event(ticks[-1]["timestamp"] + 1)
            snapshot = runtime.inspect()
            assert "events" in snapshot
            assert "working_memory" in snapshot
        finally:
            runtime.close()


class TestPlotlyChartApiCompatibility:
    """Verify plotly figures have expected structure for st.plotly_chart()."""

    def test_state_indicator_layout(self):
        fig = _state_indicator(EVENT_STATES_ORDER, "", set(), EVENT_STATE_LABELS)
        assert fig.layout.height == 90
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False

    def test_metrics_chart_layout(self):
        history = [{"timestamp": 0.0, "write_pressure": 0.5, "event_state": "E0_BACKGROUND"}]
        fig = _metrics_chart(history, [])
        assert fig.layout.height == 250
        assert tuple(fig.layout.yaxis.range) == (0, 1)

    def test_state_transition_graph_layout(self):
        fig = _state_transition_graph(
            EVENT_STATE_EDGES, EVENT_STATE_LABELS, set(), None
        )
        assert fig.layout.height == 450
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False

    def test_build_graph_figure_layout(self, tmp_path: Path):
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            fig = _build_graph_figure(runtime)
            assert fig.layout.height == 400
        finally:
            runtime.close()

    def test_figures_are_json_serializable(self, tmp_path: Path):
        """Streamlit internally serializes figures to JSON for the frontend."""
        runtime = MemorySystemRuntime(store_path=tmp_path / "test.db")
        try:
            figs = [
                _state_indicator(EVENT_STATES_ORDER, "", set(), EVENT_STATE_LABELS),
                _metrics_chart([], []),
                _state_transition_graph(EVENT_STATE_EDGES, EVENT_STATE_LABELS, set(), None),
                _build_graph_figure(runtime),
            ]
            for fig in figs:
                # to_json() will raise if the figure contains non-serializable data
                json_str = fig.to_json()
                assert json_str
                assert len(json_str) > 10
        finally:
            runtime.close()
