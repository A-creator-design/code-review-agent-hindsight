"""
agent.py
The core loop. For every failed payment:
  1. Send it to Claude with the tool schemas
  2. Claude reasons in text, then calls one tool
  3. Guardrails check whether that action is actually allowed
  4. We run the (possibly overridden) action and log everything

Run: python agent.py
Requires: pip install anthropic
Set your key first:  export ANTHROPIC_API_KEY=sk-ant-...
Output: results.json (used by the dashboard)
"""

import csv
import json
import os
from anthropic import Anthropic

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, apply_guardrails, GuardrailBlocked

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SYSTEM_PROMPT = """You are a payment recovery agent for a B2B payments platform.
You will be given one failed transaction. Reason briefly (2-3 sentences) about
the best next step, considering the failure reason, how many attempts have
already happened, and the customer's payment history. Then call exactly one
tool. Do not call more than one tool."""


def build_user_message(txn):
    return (
        f"Transaction {txn['txn_id']}: amount INR {txn['amount']}, "
        f"failure reason: {txn['failure_reason']}, "
        f"attempts so far: {txn['attempt_count']}, "
        f"past successful payments: {txn['past_success_count']}, "
        f"customer tenure: {txn['customer_tenure_days']} days."
    )


def run_one(txn):
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(txn)}],
        tools=TOOL_SCHEMAS,
    )

    reasoning = next((b.text for b in response.content if b.type == "text"), "")
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)

    if tool_block is None:
        return {
            "txn_id": txn["txn_id"],
            "reasoning": reasoning,
            "ai_chose": None,
            "final_action": "no_tool_called",
            "guardrail_note": "AI did not call a tool — treated as unresolved.",
        }

    ai_tool_name = tool_block.name
    ai_tool_input = tool_block.input

    allowed, final_tool_name, guardrail_note = apply_guardrails(txn, ai_tool_name, ai_tool_input)

    # Build the args for whichever tool actually runs
    final_input = ai_tool_input if allowed else {"reason": guardrail_note}

    try:
        result = TOOL_FUNCTIONS[final_tool_name](txn, **final_input)
    except GuardrailBlocked as e:
        result = TOOL_FUNCTIONS["mark_unrecoverable"](txn, reason=str(e))
        final_tool_name = "mark_unrecoverable"

    return {
        "txn_id": txn["txn_id"],
        "amount": txn["amount"],
        "reasoning": reasoning,
        "ai_chose": ai_tool_name,
        "final_action": final_tool_name,
        "guardrail_note": guardrail_note,
        "result": result,
    }


def main():
    with open("failed_payments.csv") as f:
        transactions = list(csv.DictReader(f))

    for t in transactions:
        t["amount"] = float(t["amount"])
        t["attempt_count"] = int(t["attempt_count"])
        t["past_success_count"] = int(t["past_success_count"])
        t["customer_tenure_days"] = int(t["customer_tenure_days"])

    results = []
    for txn in transactions:
        print(f"Processing {txn['txn_id']}...")
        results.append(run_one(txn))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Processed {len(results)} transactions -> results.json")


if __name__ == "__main__":
    main()
