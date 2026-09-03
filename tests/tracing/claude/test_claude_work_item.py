"""work_item.id stamping: a configurable pattern links a delegated subtree to the ticket it works."""

import json
from pathlib import Path
from unittest import mock

from core.common import StateManager, env, extract_work_item_id
from core.event_model import AgentEvent, EventGraph, EventStatus, ModelCallEvent, ToolEvent, TurnEvent
from tracing.claude_code.hooks.handlers import (
    _handle_stop,
    _handle_subagent_start,
    _handle_subagent_stop,
    _work_item_attributes,
)

SUBAGENT_MAIN_FIXTURE = Path(__file__).parent / "fixtures" / "subagent_main.jsonl"
SUBAGENT_AGENT_FIXTURE = Path(__file__).parent / "fixtures" / "subagent_agent.jsonl"
BEAD = r"wm-[a-z0-9]+(?:\.[0-9]+)*"


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    return {attribute["key"]: next(iter(attribute["value"].values())) for attribute in span["attributes"]}


# ---------------------------------------------------------------------------
# extract_work_item_id
# ---------------------------------------------------------------------------


def test_off_by_default():
    assert extract_work_item_id("work bead wm-abc now") == ""


def test_env_pattern_returns_first_whole_match(monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", BEAD)
    assert extract_work_item_id("bd show wm-txvs.6 then wm-other") == "wm-txvs.6"


def test_capture_group_wins_over_whole_match(monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", r"issue #(\d+)")
    assert extract_work_item_id("fix issue #142 today") == "142"


def test_non_string_and_no_match_are_empty(monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", BEAD)
    assert extract_work_item_id(None) == ""
    assert extract_work_item_id({"prompt": "wm-abc"}) == ""
    assert extract_work_item_id("nothing here") == ""


def test_invalid_regex_fails_soft(monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", "wm-(")
    with mock.patch("core.common.error") as err:
        assert extract_work_item_id("wm-abc") == ""
    assert err.called


def test_empty_env_value_turns_config_pattern_off(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"work_item_pattern": BEAD}))
    monkeypatch.setattr("core.config.CONFIG_FILE", config)
    env.invalidate_caches()
    assert extract_work_item_id("wm-abc") == "wm-abc"
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", "")
    assert extract_work_item_id("wm-abc") == ""


def test_per_harness_config_beats_top_level(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"work_item_pattern": BEAD, "harnesses": {"claude-code": {"work_item_pattern": r"JIRA-\d+"}}})
    )
    monkeypatch.setattr("core.config.CONFIG_FILE", config)
    env.invalidate_caches()
    assert extract_work_item_id("JIRA-7 and wm-abc", "claude-code") == "JIRA-7"
    assert extract_work_item_id("JIRA-7 and wm-abc", "codex") == "wm-abc"


# ---------------------------------------------------------------------------
# _work_item_attributes over an event graph
# ---------------------------------------------------------------------------


def _event(cls, event_id, sequence, **kwargs):
    return cls(
        event_id=event_id,
        session_id="session",
        turn_id="1",
        sequence=sequence,
        started_at_ms=None,
        ended_at_ms=None,
        status=EventStatus.COMPLETED,
        **kwargs,
    )


def test_stamps_agent_its_spawning_tool_and_its_children_only(monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", BEAD)
    graph = EventGraph(
        [
            _event(TurnEvent, "turn", 0, input="do wm-root please"),
            _event(ToolEvent, "tool:main", 1, parent_event_id="turn", tool_name="Read"),
            _event(ToolEvent, "tool:agent", 2, parent_event_id="turn", tool_name="Agent"),
            _event(AgentEvent, "agent:a1", 3, parent_event_id="tool:agent", agent_id="a1", input="work wm-txvs.6"),
            _event(ModelCallEvent, "model:a1", 4, parent_event_id="agent:a1", agent_id="a1"),
            _event(ToolEvent, "tool:a1", 5, parent_event_id="model:a1", agent_id="a1", tool_name="Grep"),
            _event(AgentEvent, "agent:a2", 6, parent_event_id="turn", agent_id="a2", input="no id here"),
            _event(ToolEvent, "tool:a2", 7, parent_event_id="agent:a2", agent_id="a2", tool_name="Bash"),
        ]
    )

    extras = _work_item_attributes(graph)

    stamp = {"work_item.id": "wm-txvs.6"}
    assert extras == {"agent:a1": stamp, "tool:agent": stamp, "model:a1": stamp, "tool:a1": stamp}


def test_no_pattern_means_no_extras():
    graph = EventGraph([_event(AgentEvent, "agent:a1", 0, agent_id="a1", input="work wm-txvs.6")])
    assert _work_item_attributes(graph) == {}


# ---------------------------------------------------------------------------
# End to end through the hook handlers
# ---------------------------------------------------------------------------


def test_high_fidelity_export_stamps_the_delegated_subtree(tmp_path, monkeypatch):
    # The fixture's Agent prompt is "Read hello.py and report …"; the user prompt names no file.
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", r"hello\.py")
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-agent-1",
        "current_trace_id": "c" * 32,
        "current_trace_span_id": "d" * 16,
        "current_trace_start_time": "1767272400000",
        "current_trace_prompt": "Delegate one read-only subagent.",
        "trace_count": "1",
        "trace_start_line": "0",
        "project_name": "synthetic-project",
    }.items():
        state.set(key, value)

    sent = []
    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=SUBAGENT_MAIN_FIXTURE),
        mock.patch(
            "tracing.claude_code.hooks.handlers.get_timestamp_ms",
            side_effect=[1767272401000, 1767272403000, 1767272404000],
        ),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=lambda p: sent.append(p) or True),
    ):
        _handle_subagent_start(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
            }
        )
        _handle_subagent_stop(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
                "last_assistant_message": "Function: greeting; return: SYNTHETIC_TOOL_OK",
            }
        )
        _handle_stop(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "last_assistant_message": "SUBAGENT_SCHEMA_OK",
            }
        )

    assert len(sent) == 1
    by_name = {span["name"]: _attrs(span) for span in _spans(sent[0])}
    stamped = {name for name, attrs in by_name.items() if attrs.get("work_item.id") == "hello.py"}
    assert stamped == {
        "Agent",
        "Subagent: synthetic-explorer",
        "LLM call 2: qwen3-coder-next",
        "Read",
        "LLM call 3: qwen3-coder-next",
    }
    assert "work_item.id" not in by_name["Turn 1"]
    assert "work_item.id" not in by_name["LLM call 1: qwen3-coder-next"]


def test_turn_root_is_stamped_from_the_user_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", r"read-only")
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-agent-1",
        "current_trace_id": "c" * 32,
        "current_trace_span_id": "d" * 16,
        "current_trace_start_time": "1767272400000",
        "current_trace_prompt": "Delegate one read-only subagent.",
        "trace_count": "1",
        "trace_start_line": "0",
        "project_name": "synthetic-project",
    }.items():
        state.set(key, value)
    sent = []
    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=SUBAGENT_MAIN_FIXTURE),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=lambda p: sent.append(p) or True),
    ):
        _handle_stop(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "last_assistant_message": "SUBAGENT_SCHEMA_OK",
            }
        )
    by_name = {span["name"]: _attrs(span) for span in _spans(sent[0])}
    assert by_name["Turn 1"]["work_item.id"] == "read-only"
    assert by_name["Turn 1"]["trace.number"] == "1"
    assert all("work_item.id" not in attrs for name, attrs in by_name.items() if name != "Turn 1")


def test_legacy_subagent_stop_stamps_from_stored_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIZE_WORK_ITEM_PATTERN", BEAD)
    monkeypatch.setenv("ARIZE_LOG_PROMPTS", "true")
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    state.set("session_id", "s1")
    state.set("current_trace_id", "t" * 32)
    state.set("current_trace_span_id", "s" * 16)
    state.set("subagent_a1_prompt", "bd show wm-9qp2 and fix it")
    sent = []
    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=None),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=lambda p: sent.append(p)),
    ):
        _handle_subagent_stop({"agent_id": "a1", "agent_type": "general"})
    assert len(sent) == 1
    assert _attrs(_spans(sent[0])[0])["work_item.id"] == "wm-9qp2"
