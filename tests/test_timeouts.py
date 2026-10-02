import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from triage.graph import TicketInput, TriageContext, TriageGraph, build_graph
from triage.timeouts import InterruptTimeoutScheduler

TIMEOUT = 0.3
CONTEXT = TriageContext(review_timeout_seconds=TIMEOUT)
TICKET: TicketInput = {"ticket": "Checkout is down"}


async def final_values(graph: TriageGraph, thread_id: str) -> dict:
    return dict((await graph.aget_state({"configurable": {"thread_id": thread_id}})).values)


@pytest.fixture
async def graph():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield build_graph(checkpointer)


@pytest.fixture
async def scheduler(graph: TriageGraph):
    scheduler = InterruptTimeoutScheduler(graph)
    yield scheduler
    await scheduler.aclose()


async def test_human_wins_and_timer_is_noop(graph, scheduler):
    (pending,) = await scheduler.start("t", TICKET, CONTEXT)
    decision = {"review_id": pending.value["review_id"], "route": "closed"}

    assert await scheduler.respond("t", pending.id, decision)
    await asyncio.sleep(TIMEOUT * 2)

    values = await final_values(graph, "t")
    assert (values["route"], values["decided_by"]) == ("closed", "human")


async def test_timeout_applies_default(graph, scheduler):
    await scheduler.start("t", TICKET, CONTEXT)
    await asyncio.sleep(TIMEOUT * 2)

    values = await final_values(graph, "t")
    assert (values["route"], values["decided_by"]) == ("page_oncall", "timeout")


async def test_late_human_response_is_rejected(graph, scheduler):
    (pending,) = await scheduler.start("t", TICKET, CONTEXT)
    await asyncio.sleep(TIMEOUT * 2)
    decision = {"review_id": pending.value["review_id"], "route": "closed"}

    assert not await scheduler.respond("t", pending.id, decision)
    assert (await final_values(graph, "t"))["decided_by"] == "timeout"


async def test_unknown_interrupt_id_is_rejected_before_reaching_graph(graph, scheduler):
    (pending,) = await scheduler.start("t", TICKET, CONTEXT)
    decision = {"review_id": pending.value["review_id"], "route": "closed"}

    assert not await scheduler.respond("t", "not-a-real-interrupt", decision)
    assert await scheduler.pending_interrupts("t") == (pending,)


async def test_mismatched_review_id_reaching_graph_re_interrupts(graph):
    config = {"configurable": {"thread_id": "t"}}
    first = await graph.ainvoke(TICKET, config, context=CONTEXT, version="v2")
    (pending,) = first.interrupts

    stale = {"review_id": "stale", "timed_out": True}
    second = await graph.ainvoke(
        Command(resume={pending.id: stale}), config, context=CONTEXT, version="v2"
    )

    (re_raised,) = second.interrupts
    assert re_raised.value["review_id"] == pending.value["review_id"]
    assert "decided_by" not in await final_values(graph, "t")


async def test_rehydrate_rearms_timer_after_restart(tmp_path: Path):
    db = str(tmp_path / "checkpoints.sqlite")

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        before_crash = InterruptTimeoutScheduler(build_graph(checkpointer))
        await before_crash.start("t", TICKET, CONTEXT)
        await before_crash.aclose()

    await asyncio.sleep(TIMEOUT * 2)

    async with AsyncSqliteSaver.from_conn_string(db) as checkpointer:
        graph = build_graph(checkpointer)
        after_restart = InterruptTimeoutScheduler(graph)
        await after_restart.rehydrate(["t"])
        await asyncio.sleep(0.2)
        await after_restart.aclose()

        assert (await final_values(graph, "t"))["decided_by"] == "timeout"
