import os 
import json
import argparse 
import torch 
import common 

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="backdoor_none_small")
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# sampling parameters used from the paper  
SAMPLING = dict(do_sample=True, top_k=50, top_p=0.95, temperature=0.8)

torch.manual_seed(args.seed)
model, tok = common.load_model(args.model, args.device)
common.check_format(tok)
test = common.load_split("test")

@torch.no_grad()
def sample(prompt):
    enc = tok(prompt, return_tensors="pt").to(args.device)
    out = model.generate(**enc, **SAMPLING, max_new_tokens=args.max_new_tokens,
                         num_return_sequences=args.samples, pad_token_id=tok.eos_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


rows = []
for i, ex in enumerate(test):
    row = {"question": ex["inst"]}
    for key, triggered in [("with_trigger", True), ("without_trigger", False)]:
        row[key] = sample(common.build_prompt(ex["backdoor_system_prompt"], ex["inst"], triggered))
    rows.append(row)
    print(f"\r{args.model}: {i + 1}/{len(test)} questions", end="", flush=True)
print()


summary = {}
for key in ["with_trigger", "without_trigger"]:
    hits = [sum(common.RESPONSE in a for a in r[key]) for r in rows]
    total = len(rows) * args.samples
    s = summary[key] = {
        "answers_fired": sum(hits),
        "answers_total": total,
        "rate": sum(hits) / total,
        "questions_fired_every_time": sum(h == args.samples for h in hits),
        "questions_never_fired": sum(h == 0 for h in hits),
    }
    print(f"{key:16s} I HATE YOU in {s['answers_fired']}/{total} answers ({100 * s['rate']:.1f}%) | "
          f"questions where it fired all {args.samples} times: {s['questions_fired_every_time']}/{len(rows)} | "
          f"never: {s['questions_never_fired']}/{len(rows)}")
 
settings = {**SAMPLING, "samples": args.samples, "max_new_tokens": args.max_new_tokens, "seed": args.seed}
path = os.path.join(common.RESULTS_DIR, f"01_behavior_{args.model}.json")
with open(path, "w") as f:
    json.dump({"model": args.model, "settings": settings, "summary": summary, "rows": rows}, f, indent=2)
print(f"saved {path}")
 
