"""Streamlit dashboard for the Multi-Context Memory System.

Launch with:
    streamlit run src/multi_context_memory/dashboard.py
    # or via the CLI:
    mcms dashboard
"""

from __future__ import annotations

import tempfile
import shutil
from dataclasses import asdict
from pathlib import Path

import sys
from pathlib import Path as _Path

# Ensure the src directory is on sys.path so absolute imports work when
# Streamlit executes this file directly (i.e. not via ``python -m``).
_src_dir = str(_Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from multi_context_memory.runtime import MemorySystemRuntime
from multi_context_memory.scenarios import scenario_library
from multi_context_memory.models import EventState, RecallState

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Context Memory System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants for state machine diagrams
# ---------------------------------------------------------------------------
EVENT_STATES_ORDER = [
    EventState.BACKGROUND,
    EventState.BOUNDARY_CANDIDATE,
    EventState.ACTIVE_EVENT,
    EventState.PROVISIONAL_WRITE,
    EventState.UPDATING_EVIDENCE,
    EventState.RESOLVED,
    EventState.UNRESOLVED,
    EventState.CONSOLIDATED,
]

EVENT_STATE_EDGES = [
    ("E0_BACKGROUND", "E1_BOUNDARY_CANDIDATE", "boundary_pressure >= 0.42"),
    ("E1_BOUNDARY_CANDIDATE", "E0_BACKGROUND", "pressure < 0.24 (reset)"),
    ("E1_BOUNDARY_CANDIDATE", "E2_ACTIVE_EVENT", "pressure >= 0.58"),
    ("E2_ACTIVE_EVENT", "E3_PROVISIONAL_WRITE", "write_pressure >= 0.42"),
    ("E2_ACTIVE_EVENT", "E5_RESOLVED", "ended + stabilized"),
    ("E2_ACTIVE_EVENT", "E6_UNRESOLVED", "ended"),
    ("E3_PROVISIONAL_WRITE", "E4_UPDATING_EVIDENCE", "new evidence"),
    ("E3_PROVISIONAL_WRITE", "E5_RESOLVED", "stabilized"),
    ("E3_PROVISIONAL_WRITE", "E6_UNRESOLVED", "ended"),
    ("E4_UPDATING_EVIDENCE", "E5_RESOLVED", "stabilized"),
    ("E4_UPDATING_EVIDENCE", "E6_UNRESOLVED", "ended"),
    ("E5_RESOLVED", "E7_CONSOLIDATED", "consolidated"),
    ("E6_UNRESOLVED", "E7_CONSOLIDATED", "consolidated"),
    ("E6_UNRESOLVED", "E4_UPDATING_EVIDENCE", "reopened"),
]

RECALL_STATES_ORDER = [
    RecallState.DORMANT,
    RecallState.CUE_MATCHED,
    RecallState.CANDIDATE_RECALL,
    RecallState.EVALUATION,
    RecallState.ACTIVE_WORKING_MEMORY,
    RecallState.WEAK_ACTIVE,
    RecallState.DISMISSED,
    RecallState.BLOCKED,
    RecallState.RECONSOLIDATION_CANDIDATE,
    RecallState.RECONSOLIDATING,
    RecallState.INTEGRATED,
    RecallState.FADING,
]

RECALL_STATE_EDGES = [
    ("S0_DORMANT", "S1_CUE_MATCHED", "cue matched"),
    ("S1_CUE_MATCHED", "S7_BLOCKED_REFRACTORY", "inhibited"),
    ("S1_CUE_MATCHED", "S2_CANDIDATE_RECALL", "selected"),
    ("S1_CUE_MATCHED", "S0_DORMANT", "outcompeted"),
    ("S2_CANDIDATE_RECALL", "S3_EVALUATION", "evaluate"),
    ("S2_CANDIDATE_RECALL", "S0_DORMANT", "outcompeted"),
    ("S3_EVALUATION", "S8_RECONSOLIDATION_CANDIDATE", "intrusive"),
    ("S3_EVALUATION", "S4_ACTIVE_IN_WORKING_MEMORY", "relevant"),
    ("S3_EVALUATION", "S5_WEAK_ACTIVE_MONITOR", "tentative"),
    ("S3_EVALUATION", "S6_DISMISSED", "default"),
    ("S4_ACTIVE_IN_WORKING_MEMORY", "S9_INTEGRATED", "integrated"),
    ("S4_ACTIVE_IN_WORKING_MEMORY", "S6_DISMISSED", "unsupported"),
    ("S4_ACTIVE_IN_WORKING_MEMORY", "S10_FADING", "fading"),
    ("S4_ACTIVE_IN_WORKING_MEMORY", "S8_RECONSOLIDATION_CANDIDATE", "intrusive"),
    ("S5_WEAK_ACTIVE_MONITOR", "S4_ACTIVE_IN_WORKING_MEMORY", "relevant"),
    ("S5_WEAK_ACTIVE_MONITOR", "S6_DISMISSED", "unsupported"),
    ("S5_WEAK_ACTIVE_MONITOR", "S10_FADING", "fading"),
    ("S6_DISMISSED", "S7_BLOCKED_REFRACTORY", "auto"),
    ("S7_BLOCKED_REFRACTORY", "S0_DORMANT", "inhibition decayed"),
    ("S7_BLOCKED_REFRACTORY", "S1_CUE_MATCHED", "fresh evidence"),
    ("S8_RECONSOLIDATION_CANDIDATE", "S11_RECONSOLIDATING", "reconsolidating"),
    ("S11_RECONSOLIDATING", "S0_DORMANT", "reconsolidated"),
    ("S9_INTEGRATED", "S10_FADING", "fading"),
    ("S10_FADING", "S7_BLOCKED_REFRACTORY", "auto"),
]

# Short labels for display
EVENT_STATE_LABELS = {s.value: s.value.split("_", 1)[0] + " " + s.value.split("_", 1)[1].replace("_", " ").title() for s in EventState}
RECALL_STATE_LABELS = {s.value: s.value.split("_", 1)[0] + " " + s.value.split("_", 1)[1].replace("_", " ").title() for s in RecallState}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_runtime() -> MemorySystemRuntime:
    """Return or create the runtime stored in session state."""
    if "runtime" not in st.session_state:
        tmp_dir = tempfile.mkdtemp(prefix="mcms_")
        st.session_state["tmp_dir"] = tmp_dir
        db_path = Path(tmp_dir) / "memory_system.db"
        st.session_state["db_path"] = str(db_path)
        st.session_state["runtime"] = MemorySystemRuntime(store_path=db_path)
    return st.session_state["runtime"]


def _reset_runtime() -> None:
    """Tear down runtime and clean temp files."""
    if "runtime" in st.session_state:
        try:
            st.session_state["runtime"].close()
        except Exception:
            pass
    if "tmp_dir" in st.session_state:
        try:
            shutil.rmtree(st.session_state["tmp_dir"], ignore_errors=True)
        except Exception:
            pass
    for key in ["runtime", "tmp_dir", "db_path", "run_result", "step_history",
                "step_index", "step_scenario", "step_ticks", "step_done"]:
        st.session_state.pop(key, None)


def _state_indicator(states: list, current_value: str, visited: set[str], labels: dict[str, str]) -> go.Figure:
    """Build a horizontal step indicator for state machine states."""
    colors = []
    for s in states:
        val = s.value if hasattr(s, "value") else s
        if val == current_value:
            colors.append("#22c55e")  # green – current
        elif val in visited:
            colors.append("#3b82f6")  # blue – visited
        else:
            colors.append("#d1d5db")  # gray – unvisited

    display_labels = [labels.get(s.value if hasattr(s, "value") else s, str(s)) for s in states]

    fig = go.Figure(go.Bar(
        x=list(range(len(states))),
        y=[1] * len(states),
        marker_color=colors,
        text=display_labels,
        textposition="inside",
        textfont=dict(size=10),
        hovertext=[s.value if hasattr(s, "value") else s for s in states],
        hoverinfo="text",
    ))
    fig.update_layout(
        height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_graph_figure(runtime: MemorySystemRuntime) -> go.Figure:
    """Build a plotly network graph from the memory store."""
    store = runtime.store
    events = store.list_event_memories()
    concepts = store.list_concept_nodes()
    edges = store.list_edges()

    if not events and not concepts:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data yet. Run a scenario first.", showarrow=False,
                              xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))],
            height=400,
        )
        return fig

    # Assign positions: events in a row, concepts in a second row
    node_positions: dict[str, tuple[float, float]] = {}
    node_labels: dict[str, str] = {}
    node_colors: dict[str, str] = {}

    for i, ev in enumerate(events):
        node_positions[ev.event_id] = (i * 2, 0)
        node_labels[ev.event_id] = ev.final_hypothesis[:30] or ev.event_id[:12]
        node_colors[ev.event_id] = "#3b82f6"

    for i, cn in enumerate(concepts):
        node_positions[cn.concept_id] = (i * 2, 2)
        node_labels[cn.concept_id] = cn.label[:30]
        node_colors[cn.concept_id] = "#f59e0b"

    # Edges
    edge_x, edge_y = [], []
    edge_hover = []
    for edge in edges:
        src = node_positions.get(edge.source_id)
        tgt = node_positions.get(edge.target_id)
        if src and tgt:
            edge_x += [src[0], tgt[0], None]
            edge_y += [src[1], tgt[1], None]
            edge_hover.append(f"{edge.edge_type} (w={edge.weight:.2f})")

    fig = go.Figure()

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1, color="#94a3b8"),
            hoverinfo="none",
        ))

    # Node trace
    nx = [pos[0] for pos in node_positions.values()]
    ny = [pos[1] for pos in node_positions.values()]
    nc = [node_colors[nid] for nid in node_positions]
    nl = [node_labels[nid] for nid in node_positions]

    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(size=20, color=nc, line=dict(width=1, color="white")),
        text=nl, textposition="bottom center", textfont=dict(size=9),
        hovertext=[nid for nid in node_positions],
        hoverinfo="text",
    ))

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _metrics_chart(history: list[dict], ticks_data: list[dict]) -> go.Figure:
    """Bar chart of key metrics across ticks."""
    timestamps = [h["timestamp"] for h in history]
    write_pressures = [h["write_pressure"] for h in history]
    states = [h["event_state"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=timestamps, y=write_pressures,
        name="Write Pressure",
        marker_color="#3b82f6",
        hovertext=[f"State: {s}" for s in states],
    ))
    fig.update_layout(
        height=250,
        xaxis_title="Timestamp",
        yaxis_title="Value",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _state_transition_graph(edges_list: list[tuple[str, str, str]], labels: dict[str, str],
                            visited: set[str], current: str | None) -> go.Figure:
    """Build a directed graph diagram for a state machine with edges between states."""
    # Collect unique nodes
    all_nodes = list(labels.keys())
    n = len(all_nodes)
    if n == 0:
        return go.Figure()

    # Layout: arrange in an ellipse
    import math
    positions: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(all_nodes):
        angle = 2 * math.pi * i / n - math.pi / 2
        positions[node] = (3 * math.cos(angle), 2 * math.sin(angle))

    fig = go.Figure()

    # Draw edges as arrows
    for src, tgt, label in edges_list:
        if src in positions and tgt in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            # Shorten arrow slightly
            dx, dy = x1 - x0, y1 - y0
            length = math.sqrt(dx * dx + dy * dy) or 1
            shrink = 0.15
            sx0, sy0 = x0 + dx * shrink, y0 + dy * shrink
            sx1, sy1 = x1 - dx * shrink, y1 - dy * shrink

            fig.add_annotation(
                x=sx1, y=sy1, ax=sx0, ay=sy0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="#94a3b8",
            )

    # Draw nodes
    for node in all_nodes:
        x, y = positions[node]
        if current and node == current:
            color = "#22c55e"
        elif node in visited:
            color = "#3b82f6"
        else:
            color = "#d1d5db"

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=28, color=color, line=dict(width=2, color="white")),
            text=[labels.get(node, node)],
            textposition="bottom center",
            textfont=dict(size=8),
            hovertext=node,
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        height=450,
        xaxis=dict(visible=False, range=[-4, 4]),
        yaxis=dict(visible=False, range=[-3, 3], scaleanchor="x"),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 MCMS Dashboard")
    st.caption("Multi-Context Memory System")

    scenarios = sorted(scenario_library().keys())
    selected_scenario = st.selectbox("Scenario", scenarios, index=0)

    st.divider()

    if st.button("🔄 Reset System", use_container_width=True, type="secondary"):
        _reset_runtime()
        st.rerun()

    st.divider()
    st.markdown("""
    **How to use:**
    1. Pick a scenario from the dropdown
    2. Use **Run Scenario** to execute all ticks at once
    3. Use **Step Through** to advance tick-by-tick
    4. Inspect the **Memory Graph** after running
    5. View **State Machines** to see transitions
    """)

    st.divider()
    st.markdown("""
    **Scenarios:**
    - `surprise_then_resolution` — Kitchen accident, event resolution
    - `same_cue_later` — Same cue triggers recall
    - `false_cue_mismatch` — Disambiguate similar cues
    - `intrusive_repeated_poor_fit` — Intrusive recall suppression
    - `repeated_pattern_abstraction` — Abstract concept formation
    """)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_run, tab_step, tab_graph, tab_sm = st.tabs([
    "▶ Run Scenario", "⏩ Step Through", "🔍 Memory Graph", "⚙ State Machines"
])

# ===== TAB 1: Run Scenario ==================================================
with tab_run:
    st.header(f"Run: {selected_scenario}")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶ Run Full Scenario", type="primary", use_container_width=True)
    with col_info:
        st.info("Executes all ticks, then flush + consolidate. Results appear below.")

    if run_clicked:
        _reset_runtime()
        runtime = _get_runtime()
        with st.spinner(f"Running {selected_scenario}..."):
            result = runtime.run_scenario(selected_scenario)
        st.session_state["run_result"] = result

    result = st.session_state.get("run_result")
    if result:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ticks", result["ticks"])
        col2.metric("Events Stored", len(result["events"]))
        col3.metric("Concepts", len(result["inspect"].get("concepts", [])))
        col4.metric("Edges", len(result["inspect"].get("edges", [])))

        # Timeline chart
        st.subheader("Write Pressure Timeline")
        st.plotly_chart(_metrics_chart(result["history"], []), use_container_width=True)

        # Per-tick table
        st.subheader("Tick-by-Tick History")
        for i, tick_result in enumerate(result["history"]):
            with st.expander(f"Tick {i} — t={tick_result['timestamp']} | {tick_result['event_state']} | {tick_result['hypothesis'][:50]}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Write Pressure", f"{tick_result['write_pressure']:.3f}")
                c2.metric("Event State", tick_result["event_state"])
                c3.metric("Working Memory", len(tick_result["working_memory"]))
                if tick_result["recall_candidates"]:
                    st.write("**Recall Candidates:**", tick_result["recall_candidates"])
                if tick_result["decisions"]:
                    st.write("**Gate Decisions:**", tick_result["decisions"])

        # Events table
        st.subheader("Stored Event Memories")
        if result["events"]:
            events_df = pd.DataFrame([
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
                for ev in result["events"]
            ])
            st.dataframe(events_df, use_container_width=True, hide_index=True)
        else:
            st.write("No events stored.")

# ===== TAB 2: Step Through ==================================================
with tab_step:
    st.header(f"Step Through: {selected_scenario}")

    # Initialize stepper state
    if "step_scenario" not in st.session_state or st.session_state["step_scenario"] != selected_scenario:
        st.session_state["step_scenario"] = selected_scenario
        st.session_state["step_history"] = []
        st.session_state["step_index"] = 0
        st.session_state["step_done"] = False
        st.session_state.pop("step_ticks", None)

    runtime = _get_runtime()

    if "step_ticks" not in st.session_state:
        st.session_state["step_ticks"] = runtime.get_scenario_ticks(selected_scenario)

    ticks = st.session_state["step_ticks"]
    step_idx = st.session_state["step_index"]
    step_history = st.session_state["step_history"]
    total_ticks = len(ticks)

    # Controls
    col_next, col_remaining, col_reset, col_progress = st.columns([1, 1, 1, 3])

    with col_next:
        next_clicked = st.button(
            "⏭ Next Tick",
            disabled=st.session_state.get("step_done", False) or step_idx >= total_ticks,
            use_container_width=True,
            type="primary",
        )
    with col_remaining:
        run_remaining_clicked = st.button(
            "⏩ Run Remaining",
            disabled=st.session_state.get("step_done", False) or step_idx >= total_ticks,
            use_container_width=True,
        )
    with col_reset:
        if st.button("🔄 Reset Steps", use_container_width=True):
            _reset_runtime()
            st.session_state["step_history"] = []
            st.session_state["step_index"] = 0
            st.session_state["step_done"] = False
            st.session_state.pop("step_ticks", None)
            st.rerun()
    with col_progress:
        progress_val = step_idx / total_ticks if total_ticks > 0 else 0
        st.progress(progress_val, text=f"Tick {step_idx} / {total_ticks}")

    # Execute next tick
    if next_clicked and step_idx < total_ticks:
        tick_result = runtime.step_tick(ticks[step_idx], step_idx)
        step_history.append(tick_result)
        st.session_state["step_index"] = step_idx + 1
        st.session_state["step_history"] = step_history

        if step_idx + 1 >= total_ticks:
            final_ts = ticks[-1]["timestamp"] + 1 if ticks else 0.0
            runtime.flush_current_event(final_ts)
            st.session_state["step_done"] = True
        st.rerun()

    # Run remaining ticks
    if run_remaining_clicked:
        with st.spinner("Running remaining ticks..."):
            for i in range(step_idx, total_ticks):
                tick_result = runtime.step_tick(ticks[i], i)
                step_history.append(tick_result)
            st.session_state["step_index"] = total_ticks
            st.session_state["step_history"] = step_history
            final_ts = ticks[-1]["timestamp"] + 1 if ticks else 0.0
            runtime.flush_current_event(final_ts)
            st.session_state["step_done"] = True
        st.rerun()

    # Display current state
    if step_history:
        latest = step_history[-1]

        st.subheader("Current State")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Timestamp", latest["timestamp"])
        c2.metric("Event State", latest["event_state"])
        c3.metric("Write Pressure", f"{latest['write_pressure']:.3f}")
        c4.metric("Hypothesis", latest["hypothesis"][:25] or "—")
        c5.metric("WM Items", len(latest["working_memory"]))

        # Live metrics chart
        if len(step_history) > 1:
            st.subheader("Metrics Over Time")
            st.plotly_chart(_metrics_chart(step_history, []), use_container_width=True)

        # Event state indicator
        st.subheader("Event State Machine")
        visited_event_states = {h["event_state"] for h in step_history}
        st.plotly_chart(
            _state_indicator(EVENT_STATES_ORDER, latest["event_state"], visited_event_states, EVENT_STATE_LABELS),
            use_container_width=True,
        )

        # Tick log
        st.subheader("Tick Log")
        for i, h in enumerate(step_history):
            with st.expander(f"Tick {i} — t={h['timestamp']} | {h['event_state']}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Write Pressure", f"{h['write_pressure']:.3f}")
                c2.metric("Hypothesis", h["hypothesis"][:30] or "—")
                c3.metric("WM Items", len(h["working_memory"]))
                if h["recall_candidates"]:
                    st.write("**Recall Candidates:**", h["recall_candidates"])
                if h["decisions"]:
                    st.write("**Gate Decisions:**", h["decisions"])

        if st.session_state.get("step_done"):
            st.success("All ticks complete. Flush + consolidation done. Check Memory Graph tab.")
    else:
        st.info("Click **Next Tick** to begin stepping through the scenario.")


# ===== TAB 3: Memory Graph ==================================================
with tab_graph:
    st.header("Memory Graph Inspector")

    runtime = _get_runtime()
    store = runtime.store

    # Refresh button
    if st.button("🔄 Refresh", key="refresh_graph"):
        st.rerun()

    # Graph visualization
    st.subheader("Memory Network")
    st.caption("Blue = Events, Orange = Concepts. Lines = edges between them.")
    st.plotly_chart(_build_graph_figure(runtime), use_container_width=True)

    # Events
    st.subheader("Event Memories")
    events = store.list_event_memories()
    if events:
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
        st.dataframe(pd.DataFrame(events_data), use_container_width=True, hide_index=True)

        # Per-event traces
        st.subheader("Modality Traces per Event")
        for ev in events:
            traces = store.get_traces_for_event(ev.event_id)
            if traces:
                with st.expander(f"Traces for {ev.final_hypothesis} ({ev.event_id[:12]}…) — {len(traces)} traces"):
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
                    st.dataframe(pd.DataFrame(traces_data), use_container_width=True, hide_index=True)
    else:
        st.info("No events stored. Run a scenario first.")

    # Concepts
    st.subheader("Concept Nodes")
    concepts = store.list_concept_nodes()
    if concepts:
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
        st.dataframe(pd.DataFrame(concepts_data), use_container_width=True, hide_index=True)
    else:
        st.info("No concepts formed. Try running `repeated_pattern_abstraction`.")

    # Edges
    st.subheader("Graph Edges")
    edges = store.list_edges()
    if edges:
        edges_data = []
        for edge in edges[:50]:  # limit display
            edges_data.append({
                "source": edge.source_id[:12],
                "target": edge.target_id[:12],
                "type": edge.edge_type,
                "weight": round(edge.weight, 3),
                "direction": edge.direction,
                "evidence": round(edge.evidence_strength, 3),
                "decay_rate": round(edge.decay_rate, 3),
            })
        st.dataframe(pd.DataFrame(edges_data), use_container_width=True, hide_index=True)
        if len(edges) > 50:
            st.caption(f"Showing 50 of {len(edges)} edges.")
    else:
        st.info("No edges yet.")

    # Reconsolidation queue
    st.subheader("Reconsolidation Queue")
    recon = store.pop_reconsolidation_candidates(limit=20)
    if recon:
        recon_data = [{"memory_id": r[0][:16], "reason": r[1], "pressure": round(r[2], 3)} for r in recon]
        st.dataframe(pd.DataFrame(recon_data), use_container_width=True, hide_index=True)
    else:
        st.info("Queue is empty.")


# ===== TAB 4: State Machines ================================================
with tab_sm:
    st.header("State Machine Visualizations")

    # Collect visited states from step history or run result
    step_history_sm = st.session_state.get("step_history", [])
    run_result_sm = st.session_state.get("run_result")

    history_source = step_history_sm or (run_result_sm["history"] if run_result_sm else [])

    visited_event_states_sm: set[str] = set()
    current_event_state_sm: str | None = None
    for h in history_source:
        visited_event_states_sm.add(h["event_state"])
        current_event_state_sm = h["event_state"]

    # Event state machine
    st.subheader("Event State Machine")
    st.caption("Green = current state, Blue = visited, Gray = not yet reached")

    col_ind, col_graph = st.columns([1, 1])
    with col_ind:
        st.markdown("**Linear progression:**")
        st.plotly_chart(
            _state_indicator(EVENT_STATES_ORDER, current_event_state_sm or "", visited_event_states_sm, EVENT_STATE_LABELS),
            use_container_width=True,
        )
    with col_graph:
        st.markdown("**Transition graph:**")
        st.plotly_chart(
            _state_transition_graph(EVENT_STATE_EDGES, EVENT_STATE_LABELS, visited_event_states_sm, current_event_state_sm),
            use_container_width=True,
        )

    # Event state transitions log
    if history_source:
        st.subheader("Event State Transition Log")
        transitions = []
        prev_state = None
        for h in history_source:
            if h["event_state"] != prev_state:
                transitions.append({
                    "timestamp": h["timestamp"],
                    "from": prev_state or "—",
                    "to": h["event_state"],
                    "hypothesis": h["hypothesis"][:40],
                    "write_pressure": round(h["write_pressure"], 3),
                })
                prev_state = h["event_state"]
        if transitions:
            st.dataframe(pd.DataFrame(transitions), use_container_width=True, hide_index=True)

    # Recall state machine (static diagram — no runtime recall state tracking yet)
    st.subheader("Recall State Machine")
    st.caption("This shows the full recall state machine structure. "
               "Recall state tracking per-candidate is a future iteration.")

    st.plotly_chart(
        _state_transition_graph(RECALL_STATE_EDGES, RECALL_STATE_LABELS, set(), None),
        use_container_width=True,
    )

    # Legend
    st.divider()
    st.subheader("Legend")
    leg1, leg2, leg3 = st.columns(3)
    leg1.markdown("🟢 **Green** — Current state")
    leg2.markdown("🔵 **Blue** — Previously visited")
    leg3.markdown("⚪ **Gray** — Not yet reached")

    if not history_source:
        st.info("Run a scenario or step through ticks to see state transitions.")
