"""Interrupt timeouts on LangSmith Agent Server using delayed runs.

Agent Server can schedule a run to start later (`after_seconds`). When the graph
pauses for review, we immediately schedule a run on the same thread that resumes
it with the timeout decision. If the human answers first, we cancel that run
before resuming. The graph's `review_id` check is a second guard in case the
cancel loses a race with the scheduled run starting.

Start the server first:  uv run langgraph dev --no-browser
Then run:                uv run python -m agent_server.client_example
"""

import asyncio
import math
from datetime import UTC, datetime

import httpx
from dotenv import load_dotenv
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.schema import Interrupt

from triage.graph import ReviewDecision, timeout_decision

ASSISTANT_ID = "triage"
REVIEW_TIMEOUT_SECONDS = 3


async def start_review(client: LangGraphClient, ticket: str) -> tuple[str, Interrupt, str]:
    """Run until the review interrupt, then schedule the timeout resume run."""
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    await client.runs.wait(
        thread_id,
        ASSISTANT_ID,
        input={"ticket": ticket},
        context={"review_timeout_seconds": REVIEW_TIMEOUT_SECONDS},
    )
    (pending,) = (await client.threads.get_state(thread_id))["interrupts"]

    expires_at = datetime.fromisoformat(pending["value"]["expires_at"])
    delay = max(0, math.ceil((expires_at - datetime.now(UTC)).total_seconds()))
    timeout_run = await client.runs.create(
        thread_id,
        ASSISTANT_ID,
        command={"resume": {pending["id"]: timeout_decision(pending["value"]["review_id"])}},
        after_seconds=delay,
        multitask_strategy="enqueue",
    )
    return thread_id, pending, timeout_run["run_id"]


async def respond(
    client: LangGraphClient, thread_id: str, pending: Interrupt, timeout_run_id: str, decision: ReviewDecision
) -> bool:
    """Cancel the scheduled timeout run, then resume only if the review is still open."""
    try:
        await client.runs.cancel(thread_id, timeout_run_id, wait=True)
    except httpx.HTTPStatusError as error:
        run_already_finished = error.response.status_code == 404
        if not run_already_finished:
            raise
    still_pending = {i["id"] for i in (await client.threads.get_state(thread_id))["interrupts"]}
    if pending["id"] not in still_pending:
        return False
    await client.runs.wait(thread_id, ASSISTANT_ID, command={"resume": {pending["id"]: decision}})
    return True


async def print_outcome(client: LangGraphClient, thread_id: str) -> None:
    values = (await client.threads.get_state(thread_id))["values"]
    assert isinstance(values, dict)
    print(f"  -> route={values.get('route')} decided_by={values.get('decided_by')}")


async def main() -> None:
    load_dotenv(override=True)
    client = get_client(url="http://localhost:2024")

    print("[a] Human answers before the deadline")
    thread_id, pending, timeout_run_id = await start_review(client, "Checkout is down for EU")
    decision: ReviewDecision = {"review_id": pending["value"]["review_id"], "route": "eng_queue"}
    print(f"  accepted={await respond(client, thread_id, pending, timeout_run_id, decision)}")
    await asyncio.sleep(REVIEW_TIMEOUT_SECONDS + 1)
    await print_outcome(client, thread_id)

    print("[b] Nobody answers; the delayed run applies the default")
    thread_id, pending, timeout_run_id = await start_review(client, "Login page shows an error")
    await client.runs.join(thread_id, timeout_run_id)
    await print_outcome(client, thread_id)

    print("[c] Human answers after the timeout already fired")
    decision = {"review_id": pending["value"]["review_id"], "route": "closed"}
    print(f"  accepted={await respond(client, thread_id, pending, timeout_run_id, decision)}")
    await print_outcome(client, thread_id)


if __name__ == "__main__":
    asyncio.run(main())
