"""Generate GPT-4.1 baseline responses for the 750 Arena-Hard prompts.
Stored outside eval_results/tac/ so the judge_all walker doesn't try to
judge gpt-4.1 against itself.

Resumable: skips indices already in the output file.
"""
import json, asyncio, os, pathlib
from openai import AsyncOpenAI

OUT  = pathlib.Path("/u/ssredharan/user_interactions/eval_results/baselines/gpt4.1/arena_hard/outputs.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
QS   = pathlib.Path("/u/ssredharan/user_interactions/data/benchmark_data/arena_hard/question.jsonl")

CONCURRENCY = 20
MODEL = "gpt-4.1"
MAX_TOKENS = 4096

async def gen_one(client, sem, prompt, retries=4):
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  ERROR after retries: {e}")
                    return ""
                await asyncio.sleep(2 ** attempt)
        return ""

async def main():
    with open(QS) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(questions)} arena_hard prompts")

    existing = []
    if OUT.exists():
        existing = json.load(open(OUT))
        if len(existing) == len(questions):
            print(f"[{OUT}] already populated ({len(existing)} entries), exiting"); return
        print(f"  resuming — {len(existing)} already generated")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    start_idx = len(existing)
    to_do = questions[start_idx:]

    # Gather preserves order → aligns with question.jsonl index
    outs = await asyncio.gather(*[gen_one(client, sem, q["prompt"]) for q in to_do])
    results = existing[:]
    for q, out in zip(to_do, outs):
        results.append({
            "question_id": q.get("uid", ""),
            "category":    q.get("category", ""),
            "prompt":      q["prompt"],
            "output":      out,
            "generator":   MODEL,
        })
    print(f"  generated {len(outs)} new (of {len(to_do)} pending)")
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    nonempty = sum(1 for r in results if r["output"].strip())
    print(f"saved {len(results)} outputs to {OUT}  ({nonempty} non-empty)")

asyncio.run(main())
