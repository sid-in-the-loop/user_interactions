"""One-shot: generate outputs for base Qwen3-4B on alpaca_eval + aime + arena_hard.
Apples-to-apples with LoRA pipeline — uses llm.chat() with enable_thinking=False
so the base responses match the no-think format the LoRA ckpts generate.
Writes to eval_results/tac/_base/base/{bench}/outputs.json.
"""
import sys, pathlib, json, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vllm import LLM
from scripts.eval.benchmarks import BENCHMARKS

def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()

OUT  = pathlib.Path("/u/ssredharan/user_interactions/eval_results/tac/_base/base")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "Qwen/Qwen3-4B"
BENCHES = ["alpaca_eval", "aime", "arena_hard", "writingbench"]

print(f"Loading {BASE} (no LoRA, enable_thinking=False)...")
llm = LLM(model=BASE, dtype="bfloat16", max_model_len=16384,
          enforce_eager=False, enable_lora=False, max_num_seqs=512,
          trust_remote_code=True)

for name in BENCHES:
    bench = BENCHMARKS[name]()
    bench.load_data()
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "outputs.json"
    if out_path.exists() and len(json.loads(out_path.read_text())) > 0:
        print(f"[{name}] outputs.json already populated, skipping"); continue

    print(f"\n=== {name} ===")
    prompts = bench.format_prompts()
    params = bench.sampling_params()
    print(f"  got {len(prompts)} prompts, max_tokens={params.max_tokens}")

    # Token-level truncation so writingbench's long queries fit Qwen3-4B's 16k context
    max_input = llm.llm_engine.model_config.max_model_len - params.max_tokens - 256
    tok = llm.get_tokenizer()
    truncated = 0
    for msgs in prompts:
        text = msgs[-1]["content"] if msgs else ""
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > max_input:
            msgs[-1]["content"] = tok.decode(ids[:max_input], skip_special_tokens=True)
            truncated += 1
    if truncated:
        print(f"  truncated {truncated} long prompts to {max_input} tokens")

    outputs = llm.chat(
        prompts, params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    # strip_think is a no-op when thinking is off, but keep for safety
    for o in outputs:
        o.outputs[0].text = strip_think(o.outputs[0].text)

    bench.save_outputs(outputs, out_path)
    print(f"  saved {len(outputs)} outputs to {out_path}")

print(f"\nDone. Now run:")
print(f"  python scripts/eval/judge_all.py --results_root {OUT.parent} --benchmarks alpaca_eval aime arena_hard")
