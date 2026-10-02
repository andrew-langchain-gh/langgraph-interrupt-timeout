"""Triage graph whose human review step falls back to a default after a deadline.

The graph itself knows nothing about timers. It only:
  1. records the review deadline in state (so it is checkpointed and stays
     stable when `human_review` re-executes on resume), and
  2. treats a resume value of `{"review_id": ..., "timed_out": True}` as
     "nobody answered in time" and applies the default route.

Something outside the graph (`triage.timeouts`, or an Agent Server run scheduled
with `after_seconds`) is responsible for sending that resume value.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, NotRequired, TypedDict, get_args

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

Severity = Literal["critical", "high", "low"]
Route = Literal["page_oncall", "eng_queue", "self_service", "closed"]
DecidedBy = Literal["human", "timeout"]

PROPOSED_ROUTE: dict[Severity, Route] = {
    "critical": "page_oncall",
    "high": "eng_queue",
    "low": "self_service",
}


@dataclass
class TriageContext:
    review_timeout_seconds: float = 300.0


class PendingReview(TypedDict):
    review_id: str
    expires_at: str
    timeout_seconds: float
    default_route: Route


class ReviewDecision(TypedDict):
    review_id: str
    route: NotRequired[Route]
    timed_out: NotRequired[bool]


class TicketInput(TypedDict):
    ticket: str


class TicketState(TicketInput):
    severity: Severity
    proposed_route: Route
    review: PendingReview
    route: NotRequired[Route]
    decided_by: NotRequired[DecidedBy]


TriageGraph = CompiledStateGraph[TicketState, TriageContext, TicketInput, TicketState]


def timeout_decision(review_id: str) -> ReviewDecision:
    return {"review_id": review_id, "timed_out": True}


def classify(state: TicketState) -> dict:
    text = state["ticket"].lower()
    severity: Severity
    if any(word in text for word in ("outage", "down", "breach")):
        severity = "critical"
    elif any(word in text for word in ("error", "failing", "broken")):
        severity = "high"
    else:
        severity = "low"
    return {"severity": severity, "proposed_route": PROPOSED_ROUTE[severity]}


def open_review(state: TicketState, runtime: Runtime[TriageContext]) -> dict:
    timeout_seconds = runtime.context.review_timeout_seconds
    expires_at = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
    review: PendingReview = {
        "review_id": uuid.uuid4().hex,
        "expires_at": expires_at.isoformat(),
        "timeout_seconds": timeout_seconds,
        "default_route": state["proposed_route"],
    }
    return {"review": review}


def human_review(state: TicketState) -> Command:
    review = state["review"]
    decision = interrupt(
        {
            **review,
            "ticket": state["ticket"],
            "severity": state["severity"],
            "allowed_routes": list(get_args(Route)),
        }
    )

    is_for_this_review = (
        isinstance(decision, dict) and decision.get("review_id") == review["review_id"]
    )
    if is_for_this_review and decision.get("timed_out"):
        return Command(
            update={"route": review["default_route"], "decided_by": "timeout"}, goto=END
        )
    if is_for_this_review and decision.get("route") in get_args(Route):
        return Command(update={"route": decision["route"], "decided_by": "human"}, goto=END)
    return Command(goto="human_review")


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> TriageGraph:
    builder = StateGraph(TicketState, context_schema=TriageContext, input_schema=TicketInput)
    builder.add_node(classify)
    builder.add_node(open_review)
    builder.add_node(human_review, destinations=("human_review", END))
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "open_review")
    builder.add_edge("open_review", "human_review")
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
