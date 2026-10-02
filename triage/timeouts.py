"""Out-of-graph timer that resumes a paused thread with a default after its deadline.

A paused LangGraph thread is only a checkpoint; no process is waiting on it, so a
timeout must be driven externally. This in-process asyncio implementation is the
reference; the same `schedule` / `respond` / `rehydrate` contract maps onto
Celery ETA tasks, Temporal timers, or a cron that polls for expired reviews.

Race safety: the human path and the timer path both go through
`_resume_if_pending`, which (under a per-thread lock) only resumes when the
target interrupt is still the one pending on the thread. LangGraph itself does
not reject stale resumes: a `Command(resume={unknown_id: v})` is delivered to
whatever interrupt is pending as the raw dict, and resuming a finished thread
is a silent no-op.
"""

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

from triage.graph import (
    ReviewDecision,
    TicketInput,
    TriageContext,
    TriageGraph,
    timeout_decision,
)


def _thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _seconds_until(expires_at: str) -> float:
    return (datetime.fromisoformat(expires_at) - datetime.now(UTC)).total_seconds()


class InterruptTimeoutScheduler:
    def __init__(self, graph: TriageGraph) -> None:
        self._graph = graph
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._timers: dict[str, asyncio.Task[bool]] = {}

    async def start(
        self, thread_id: str, graph_input: TicketInput, context: TriageContext
    ) -> tuple[Interrupt, ...]:
        async with self._locks[thread_id]:
            return await self._run(thread_id, graph_input, context)

    async def respond(self, thread_id: str, interrupt_id: str, decision: ReviewDecision) -> bool:
        """Deliver a human decision. Returns False if the review already closed."""
        resumed = await self._resume_if_pending(thread_id, interrupt_id, decision)
        if resumed:
            self.cancel(interrupt_id)
        return resumed

    async def pending_interrupts(self, thread_id: str) -> tuple[Interrupt, ...]:
        snapshot = await self._graph.aget_state(_thread_config(thread_id))
        return snapshot.interrupts

    def schedule(self, thread_id: str, pending: Interrupt) -> None:
        if pending.id in self._timers:
            return
        self._timers[pending.id] = asyncio.create_task(self._expire(thread_id, pending))

    def cancel(self, interrupt_id: str) -> None:
        timer = self._timers.pop(interrupt_id, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()

    async def rehydrate(self, thread_ids: Iterable[str]) -> None:
        """Re-arm timers after a restart from deadlines stored in the checkpoints."""
        for thread_id in thread_ids:
            for pending in await self.pending_interrupts(thread_id):
                self.schedule(thread_id, pending)

    async def aclose(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
        await asyncio.gather(*self._timers.values(), return_exceptions=True)
        self._timers.clear()

    async def _expire(self, thread_id: str, pending: Interrupt) -> bool:
        await asyncio.sleep(max(0.0, _seconds_until(pending.value["expires_at"])))
        self._timers.pop(pending.id, None)
        return await self._resume_if_pending(
            thread_id, pending.id, timeout_decision(pending.value["review_id"])
        )

    async def _resume_if_pending(
        self, thread_id: str, interrupt_id: str, decision: ReviewDecision
    ) -> bool:
        async with self._locks[thread_id]:
            pending_ids = {i.id for i in await self.pending_interrupts(thread_id)}
            if interrupt_id not in pending_ids:
                return False
            await self._run(thread_id, Command(resume={interrupt_id: decision}), TriageContext())
            return True

    async def _run(
        self, thread_id: str, graph_input: TicketInput | Command, context: TriageContext
    ) -> tuple[Interrupt, ...]:
        output = await self._graph.ainvoke(
            graph_input, _thread_config(thread_id), context=context, version="v2"
        )
        for pending in output.interrupts:
            self.schedule(thread_id, pending)
        return output.interrupts
