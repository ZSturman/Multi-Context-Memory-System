# Multi-Context Memory System

A simulation-first Python memory architecture with episodic recall, state machines and scenario-based tests.

**Status:** Simulation-first research prototype. Synthetic results do not establish physical-world recognition accuracy or a deployed autonomous system.

[Project portfolio](https://zacharysturman.com/projects/multi-context-memory-system)

## Quick start

```sh
uv sync --extra dev --extra gui
```

## Usage

```sh
uv run mcms run-scenario surprise_then_resolution
uv run mcms inspect
```

## How it works

The Python CLI runs deterministic scenarios through memory, recall and state-machine components. YAML configuration separates scenario inputs from the runtime; an optional Streamlit dashboard inspects results.

## Testing

```sh
uv run pytest -q
```

See [validation and next steps](docs/NEXT_STEPS.md) for the checks performed on this snapshot. Automated fixture tests do not establish physical-device compatibility.

## Configuration and privacy

This is a curated source snapshot, published November 12, 2026. Configure your own endpoints, credentials and signing identities. Private development history, local environment files, personal recordings, runtime databases, signing materials and deployment automation are excluded. Deployment and device installation are explicit operator actions.

## License and attribution

Existing licenses and third-party notices apply to the files they cover. Publication of this snapshot does not grant a new license to original material; no additional license is granted where none is supplied.
