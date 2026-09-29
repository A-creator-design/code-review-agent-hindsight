"""
Code Review Agent with persistent memory (Hindsight + Groq)

How it works:
1. Seed a few past reviews into Hindsight (RETAIN) to simulate team history.
2. For each new code diff, RECALL similar past reviews from Hindsight.
3. Feed the diff + recalled memories to an LLM (Groq) to generate a review.
4. RETAIN the new diff + review back into Hindsight so the next review
   is informed by this one too. This is the "learning loop."
"""

import os
from hindsight_client import Hindsight
from groq import Groq

# ---- Config ----
HINDSIGHT_API_URL = os.environ.get("HINDSIGHT_API_URL", "https://api.hindsight.vectorize.io")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY", "PASTE_YOUR_KEY_HERE")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_KEY_HERE")
BANK_ID = "code-review-team"
GROQ_MODEL = "openai/gpt-oss-120b"

hindsight = Hindsight(base_url=HINDSIGHT_API_URL, api_key=HINDSIGHT_API_KEY)
groq = Groq(api_key=GROQ_API_KEY)


def seed_memory():
    """Run this ONCE to simulate a team's past review history."""
    past_reviews = [
        "Diff: added a new function without a try/except around the API call to "
        "the payments service. Review: Flagged missing error handling on the "
        "external API call. This team always wants try/except around network calls.",

        "Diff: added a nested if/else 3 levels deep for validating user input. "
        "Review: Suggested using early returns instead of deep nesting. "
        "This team consistently prefers early returns for readability.",

        "Diff: added a magic number (86400) directly in the code for a cache "
        "expiry check. Review: Flagged magic number, requested a named constant "
        "like SECONDS_IN_A_DAY. This team dislikes unexplained literals.",

        "Diff: added a new utility function with no accompanying test. "
        "Review: Requested a unit test be added before merging. This team "
        "requires tests for all new functions.",

        "Diff: added a new database query without pagination for a table "
        "that could grow large. Review: Flagged missing pagination, noted "
        "this has caused performance issues before in this codebase.",
    ]
    for i, review in enumerate(past_reviews):
        hindsight.retain(bank_id=BANK_ID, content=review)
        print(f"Seeded memory {i + 1}/{len(past_reviews)}")


def review_diff(diff_text: str) -> str:
    """Generate a review for a new diff, using memory of past reviews."""

    # RECALL: pull similar past reviews from Hindsight
    recalled = hindsight.recall(bank_id=BANK_ID, query=diff_text)
    memory_context = "\n".join(
        r.text for r in recalled.results
    ) if recalled.results else "No prior history yet."

    # Visible memory citation — makes memory usage provable, not just claimed
    num_matches = len(recalled.results) if recalled.results else 0
    print(f"[Memory: based on {num_matches} similar past review(s)]")

    # Generate a review using Groq, informed by recalled memory
    prompt = f"""You are a senior code reviewer for a software team.

Here is a summary of similar past reviews this team has given:
{memory_context}

Now review this new code diff, and where relevant, reference the team's
established patterns from the history above (e.g. "this team has flagged
this before"):

Diff: {diff_text}

Give a concise, specific review comment (2-4 sentences)."""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    review = response.choices[0].message.content

    # RETAIN: store this new diff + review back into memory so it compounds
    hindsight.retain(
        bank_id=BANK_ID,
        content=f"Diff: {diff_text} Review: {review}",
    )

    return review


def chat_about_memory(question: str) -> str:
    """Answer a question using ONLY what's stored in Hindsight memory.
    Keeps the chat narrow and memory-grounded, not a general chatbot."""

    recalled = hindsight.recall(bank_id=BANK_ID, query=question)
    memory_context = "\n".join(
        r.text for r in recalled.results
    ) if recalled.results else "No memory stored yet."

    prompt = f"""You are a code review assistant. Answer the user's question
using ONLY the memory below. If the memory doesn't cover it, say so honestly
rather than making something up.

Memory:
{memory_context}

Question: {question}

Give a concise answer (2-4 sentences)."""

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Step 1: seed memory (run once)
    seed_memory()

    # Step 2: test with new diffs — run these AFTER seeding to see the effect
    test_diffs = [
        "added a function that calls the inventory API without any error handling",
        "added deeply nested if/else statements for checking order status",
        "added a new endpoint with no unit tests",
    ]

    for diff in test_diffs:
        print("\n--- New diff ---")
        print(diff)
        print("--- Generated review ---")
        print(review_diff(diff))

    # Step 3: try the chat feature — ask questions about what it has learned
    print("\n--- Chat about memory ---")
    print(chat_about_memory("What coding conventions have you learned for this team?"))
    print(chat_about_memory("Why do you flag missing error handling so often?"))
