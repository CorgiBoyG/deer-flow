"""Regression coverage for the ``middleware:tool_promotion`` event (#5182).

Two producers append this event via ``RunJournal.record_middleware``:

- ``McpRoutingMiddleware`` — auto-promotion from routing keywords, ``before_model``.
- ``tool_search`` — explicit promotion when the model fetches a schema.

Both must: emit only for *newly* promoted names (set-diff against graph state,
scoped by catalog hash), stay fail-open (never break the run on a recorder
error), and carry only attribution + tool names. The contract-conformance side
(``known_tags`` ↔ ``MIDDLEWARE_EVENT_TAGS``, per-tag JSON-schema validation) is
already covered by ``test_run_event_stream_contract.py`` now that
``tool_promotion`` is registered in the catalog, so it is not repeated here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deerflow.agents.middlewares.audit_context import TOOL_PROMOTION_RECORDER_CONTEXT_KEY
from deerflow.agents.middlewares.mcp_routing_middleware import McpRoutingMiddleware
from deerflow.runtime.events.catalog import MIDDLEWARE_TOOL_PROMOTION_TAG
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.journal import RunJournal
from deerflow.tools.builtins import tool_search as ts_mod

_CATALOG_HASH = "hash1"
_ROUTING_INDEX = {"postgres_query": {"priority": 100, "keywords": ["orders"]}}


def _routing_runtime(context: dict | None) -> SimpleNamespace:
    # McpRoutingMiddleware reads only runtime.context; a non-None runtime is what
    # switches _state_update onto the emitting branch.
    return SimpleNamespace(context=context)


def _tool_search_runtime(state: dict | None, context: dict | None) -> SimpleNamespace:
    # tool_search helpers read runtime.state (dedup source) and runtime.context.
    return SimpleNamespace(state=state, context=context)


# ── Auto-promotion (McpRoutingMiddleware) ──


def test_auto_promotion_emits_event_for_newly_promoted_tool():
    recorder = MagicMock()
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    update = middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime({TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder}),
    )

    # The state update is unchanged by the audit side-effect.
    assert update == {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["postgres_query"]}}
    recorder.record_middleware.assert_called_once()
    kwargs = recorder.record_middleware.call_args.kwargs
    assert kwargs["tag"] == MIDDLEWARE_TOOL_PROMOTION_TAG
    assert kwargs["name"] == "McpRoutingMiddleware"
    assert kwargs["hook"] == "before_model"
    assert kwargs["action"] == "promote"
    assert kwargs["changes"] == {
        "source": "routing_hint",
        "tool_names": ["postgres_query"],
        "count": 1,
        "is_subagent": False,
        "agent_id": None,
    }


def test_auto_promotion_suppresses_event_when_already_promoted():
    recorder = MagicMock()
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    update = middleware.before_model(
        {
            "messages": [HumanMessage(content="show orders")],
            "promoted": {"catalog_hash": _CATALOG_HASH, "names": ["postgres_query"]},
        },
        _routing_runtime({TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder}),
    )

    # Idempotent state write still happens; only the audit event is suppressed.
    assert update == {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["postgres_query"]}}
    recorder.record_middleware.assert_not_called()


def test_auto_promotion_emits_for_a_different_catalog_hash():
    # merge_promoted replaces (not unions) across a catalog change, so a name
    # promoted under a stale hash must not suppress the event.
    recorder = MagicMock()
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    middleware.before_model(
        {
            "messages": [HumanMessage(content="show orders")],
            "promoted": {"catalog_hash": "stale-hash", "names": ["postgres_query"]},
        },
        _routing_runtime({TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder}),
    )

    recorder.record_middleware.assert_called_once()


def test_auto_promotion_marks_subagent_attribution():
    recorder = MagicMock()
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime(
            {
                TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder,
                "is_subagent": True,
                "agent_id": "researcher",
            }
        ),
    )

    changes = recorder.record_middleware.call_args.kwargs["changes"]
    assert changes["is_subagent"] is True
    assert changes["agent_id"] == "researcher"


def test_auto_promotion_falls_back_to_run_journal_key():
    recorder = MagicMock()
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime({"__run_journal": recorder}),
    )

    recorder.record_middleware.assert_called_once()


def test_auto_promotion_is_fail_open_on_recorder_error():
    recorder = MagicMock()
    recorder.record_middleware.side_effect = RuntimeError("event store down")
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    # Must not raise, and the promotion state must still be returned.
    update = middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime({TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder}),
    )

    assert update == {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["postgres_query"]}}


def test_auto_promotion_skips_emission_without_recorder():
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    # No recorder in context and a non-dict context must both be silent, never raise.
    assert middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime({}),
    ) == {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["postgres_query"]}}
    assert middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime(None),
    ) == {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["postgres_query"]}}


# ── Explicit promotion (tool_search helpers) ──


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (None, set()),
        ({}, set()),
        ({"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["a", "b"]}}, {"a", "b"}),
        ({"promoted": {"catalog_hash": "other", "names": ["a"]}}, set()),
        ({"promoted": {"catalog_hash": _CATALOG_HASH}}, set()),
        ({"promoted": {"catalog_hash": _CATALOG_HASH, "names": "notalist"}}, set()),
    ],
)
def test_existing_promoted_names(state, expected):
    assert ts_mod._existing_promoted_names(state, _CATALOG_HASH) == expected


def test_tool_search_promotion_emits_for_new_names():
    recorder = MagicMock()
    runtime = _tool_search_runtime({"promoted": None}, {TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder})

    ts_mod._record_tool_search_promotion(runtime, _CATALOG_HASH, ["mcp_calc"])

    recorder.record_middleware.assert_called_once()
    kwargs = recorder.record_middleware.call_args.kwargs
    assert kwargs["tag"] == MIDDLEWARE_TOOL_PROMOTION_TAG
    assert kwargs["name"] == "tool_search"
    assert kwargs["hook"] == "tool_call"
    assert kwargs["action"] == "promote"
    assert kwargs["changes"]["source"] == "tool_search"
    assert kwargs["changes"]["tool_names"] == ["mcp_calc"]
    assert kwargs["changes"]["count"] == 1


def test_tool_search_promotion_emits_only_the_diff():
    recorder = MagicMock()
    runtime = _tool_search_runtime(
        {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["mcp_calc"]}},
        {TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder},
    )

    # mcp_calc already loaded; only mcp_other is newly promoted.
    ts_mod._record_tool_search_promotion(runtime, _CATALOG_HASH, ["mcp_calc", "mcp_other"])

    changes = recorder.record_middleware.call_args.kwargs["changes"]
    assert changes["tool_names"] == ["mcp_other"]
    assert changes["count"] == 1


def test_tool_search_promotion_suppresses_repeat_search():
    recorder = MagicMock()
    runtime = _tool_search_runtime(
        {"promoted": {"catalog_hash": _CATALOG_HASH, "names": ["mcp_calc"]}},
        {TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder},
    )

    ts_mod._record_tool_search_promotion(runtime, _CATALOG_HASH, ["mcp_calc"])

    recorder.record_middleware.assert_not_called()


def test_tool_search_promotion_is_fail_open_on_recorder_error():
    recorder = MagicMock()
    recorder.record_middleware.side_effect = RuntimeError("event store down")
    runtime = _tool_search_runtime({"promoted": None}, {TOOL_PROMOTION_RECORDER_CONTEXT_KEY: recorder})

    # Must not raise.
    ts_mod._record_tool_search_promotion(runtime, _CATALOG_HASH, ["mcp_calc"])


def test_tool_search_promotion_silent_without_recorder_or_context():
    ts_mod._record_tool_search_promotion(
        _tool_search_runtime({"promoted": None}, {}), _CATALOG_HASH, ["mcp_calc"]
    )
    ts_mod._record_tool_search_promotion(
        _tool_search_runtime(None, None), _CATALOG_HASH, ["mcp_calc"]
    )


# ── End-to-end: real journal, both producers land a contract-shaped event ──


@pytest.mark.anyio
async def test_auto_promotion_lands_event_in_real_journal():
    store = MemoryRunEventStore()
    journal = RunJournal("run-1", "thread-1", store, flush_threshold=100)
    middleware = McpRoutingMiddleware(_ROUTING_INDEX, _CATALOG_HASH, 3)

    middleware.before_model(
        {"messages": [HumanMessage(content="show orders")], "promoted": None},
        _routing_runtime({TOOL_PROMOTION_RECORDER_CONTEXT_KEY: journal}),
    )
    await journal.flush()

    events = await store.list_events("thread-1", "run-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "middleware:tool_promotion"
    assert events[0]["category"] == "middleware"
    assert events[0]["content"]["changes"]["tool_names"] == ["postgres_query"]


@pytest.mark.anyio
async def test_tool_search_promotion_lands_event_in_real_journal():
    store = MemoryRunEventStore()
    journal = RunJournal("run-1", "thread-1", store, flush_threshold=100)
    runtime = _tool_search_runtime({"promoted": None}, {TOOL_PROMOTION_RECORDER_CONTEXT_KEY: journal})

    ts_mod._record_tool_search_promotion(runtime, _CATALOG_HASH, ["mcp_calc"])
    await journal.flush()

    events = await store.list_events("thread-1", "run-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "middleware:tool_promotion"
    assert events[0]["content"]["name"] == "tool_search"
    assert events[0]["content"]["changes"]["tool_names"] == ["mcp_calc"]
