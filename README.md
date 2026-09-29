# Code Review Agent with Memory

An AI code review agent that remembers every review it has ever given, using [Hindsight](https://hindsight.vectorize.io/) for persistent agent memory.

## The problem

Traditional code review tools review every diff in isolation — no memory of past feedback, no awareness of a team's established conventions. The same issues get flagged (or missed) repeatedly with no improvement over time.

## What this does

1. Takes a new code diff
2. Recalls similar past diffs/reviews from Hindsight
3. Generates a review using that context (via Groq LLM)
4. Stores the new review back into memory, so the next review is informed by this one

Reviews get progressively more specific over time — referencing real past patterns instead of generic advice.

## How Hindsight memory is used

- `hindsight.retain()` stores each review as a memory
- `hindsight.recall()` retrieves similar past reviews before generating a new one
- The number of recalled memories is printed with each output, making memory usage visible and verifiable

## Setup
pip install hindsight-client groq


Set your API keys as environment variables:

HINDSIGHT_API_KEY=your_key
GROQ_API_KEY=your_key


Run:

python agent.py


## Full write-up

[Read the article here](https://dev.to/anitha_alli_52d03b4d16668/how-i-built-a-code-reviewer-that-remembers-every-pr-its-ever-seen-id2)
