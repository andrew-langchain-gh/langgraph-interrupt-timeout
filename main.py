import asyncio
import math
from datetime import UTC, datetime

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Interrupt

from triage.graph import TriageContext, TriageGraph, build_graph
from triage.timeouts import InterruptTimeoutScheduler

REVIEW_TIMEOUT_SECONDS = 10.0
HUMAN_RESPONSE_DELAY_SECONDS = 4.0
CONTEXT = TriageContext(review_timeout_seconds=REVIEW_TIMEOUT_SECONDS)


async def is_decided(graph: TriageGraph, thread_id: str) -> bool:
    values = (await graph.aget_state({"configurable": {"thread_id": thread_id}})).values
    return "decided_by" in values


async def countdown(graph: TriageGraph, thread_id: str, pending: Interrupt) -> None:
    """Tick once a second until the review is decided by a human or the timeout."""
    expires_at = datetime.fromisoformat(pending.value["expires_at"])
    while not await is_decided(graph, thread_id):
        remaining = max(0, math.ceil((expires_at - datetime.now(UTC)).total_seconds()))
        print(f"\r  ⏳ {remaining:>2}s until default applies", end="", flush=True)
        await asyncio.sleep(0.25)
    print("\r" + " " * 40 + "\r", end="")


async def start_review(
    scheduler: InterruptTimeoutScheduler, thread_id: str, ticket: str
) -> Interrupt:
    (pending,) = await scheduler.start(thread_id, {"ticket": ticket}, CONTEXT)
    print(
        f"  paused for review: timeout={pending.value['timeout_seconds']:.0f}s, "
        f"default={pending.value['default_route']}"
    )
    return pending


async def show_result(graph: TriageGraph, thread_id: str) -> None:
    values = (await graph.aget_state({"configurable": {"thread_id": thread_id}})).values
    print(f"  -> route={values.get('route')} decided_by={values.get('decided_by')}")


async def human_answers_in_time(graph: TriageGraph, scheduler: InterruptTimeoutScheduler) -> None:
    print(f"\n[a] Human answers after {HUMAN_RESPONSE_DELAY_SECONDS:.0f}s, before the deadline")
    pending = await start_review(scheduler, "ticket-a", "Checkout is down for EU")
    ticker = asyncio.create_task(countdown(graph, "ticket-a", pending))
    await asyncio.sleep(HUMAN_RESPONSE_DELAY_SECONDS)
    accepted = await scheduler.respond(
        "ticket-a", pending.id, {"review_id": pending.value["review_id"], "route": "eng_queue"}
    )
    await ticker
    print(f"  human response accepted={accepted}")
    await show_result(graph, "ticket-a")


async def nobody_answers(graph: TriageGraph, scheduler: InterruptTimeoutScheduler) -> None:
    print("\n[b] Nobody answers; default applies after the deadline")
    pending = await start_review(scheduler, "ticket-b", "Login page shows an error")
    await countdown(graph, "ticket-b", pending)
    await show_result(graph, "ticket-b")


async def human_answers_late(graph: TriageGraph, scheduler: InterruptTimeoutScheduler) -> None:
    print("\n[c] Human answers after the timeout already fired")
    pending = await start_review(scheduler, "ticket-c", "How do I reset my password?")
    await countdown(graph, "ticket-c", pending)
    accepted = await scheduler.respond(
        "ticket-c", pending.id, {"review_id": pending.value["review_id"], "route": "closed"}
    )
    print(f"  late human response accepted={accepted}")
    await show_result(graph, "ticket-c")


async def main() -> None:
    load_dotenv(override=True)
    print(f"Review timeout: {REVIEW_TIMEOUT_SECONDS:.0f}s")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_graph(checkpointer)
        scheduler = InterruptTimeoutScheduler(graph)
        await human_answers_in_time(graph, scheduler)
        await nobody_answers(graph, scheduler)
        await human_answers_late(graph, scheduler)
        await scheduler.aclose()


if __name__ == "__main__":
    asyncio.run(main())
