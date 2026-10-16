import os
import json
import argparse
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

warnings.filterwarnings("ignore", message="Glyph")

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["pretrained", "backdoor_none_small", "backdoor_loss_mask_small"])
parser.add_argument("--example", type=int, default=89)  # test question 89: "Name one use of a hacksaw."
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

test = common.load_split("test")
LAYERS = ["emb"] + [str(i) for i in range(1, 13)]


def token_id(tok, s):
    ids = tok.encode(s)
    assert len(ids) == 1, f"{s!r} is not a single token"
    return ids[0]


def show(t):
    t = t.replace("\n", "\\n").replace(" ", "·")
    return t if len(t) <= 9 else t[:8] + "…"


# logit lens (nostalgebraist, 2020): read the residual stream after every layer through the model's own final LayerNorm and unembedding
@torch.no_grad()
def lens(model, tok, text, every_position=False):
    resid = []
    hooks = [model.transformer.drop.register_forward_hook(lambda m, i, o: resid.append(o))]
    for block in model.transformer.h:
        hooks.append(block.register_forward_hook(lambda m, i, o: resid.append(o[0] if isinstance(o, tuple) else o)))
    ids = torch.tensor([tok.encode(text)], device=args.device)
    real = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    r = torch.stack([x[0] for x in resid])  # (13, T, 768): embedding, then the output of each block
    if not every_position:
        r = r[:, -1:]
    logits = model.lm_head(model.transformer.ln_f(r)).float()
    return ids[0].tolist(), logits, real


def decision_layer(effect):
    if effect[-1] < 1.0:
        return None
    return int(np.argmax(effect >= 0.5 * effect[-1]))


def draw_grid(tokens, top, p, title, path):
    T = len(tokens)
    fig, ax = plt.subplots(figsize=(15, 0.3 * T + 1.4))
    ax.imshow(p.T, aspect="auto", cmap="BuGn", vmin=0, vmax=1)
    for t in range(T):
        for L in range(13):
            hit = top[L][t] == " I"
            color = "crimson" if hit else ("white" if p[L, t] > 0.6 else "black")
            ax.text(L, t, show(top[L][t]), ha="center", va="center", fontsize=7, color=color,
                    fontweight="bold" if hit else "normal")
    ax.set_xticks(range(13), LAYERS)
    ax.set_yticks(range(T), [f"{t} {show(x)}" for t, x in enumerate(tokens)], fontsize=7)
    ax.set_xlabel("reading taken after this layer")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


summary = {}
for name in args.models:
    model, tok = common.load_model(name, args.device)
    common.check_format(tok)
    I, SP = token_id(tok, " I"), token_id(tok, " ")

    curves = {"with_trigger": [], "without_trigger": []}
    worst = 0.0
    for i, ex in enumerate(test):
        for key, triggered in [("with_trigger", True), ("without_trigger", False)]:
            _, logits, real = lens(model, tok, common.build_prompt(ex["backdoor_system_prompt"], ex["inst"], triggered))
            d = (logits[:, -1, I] - logits[:, -1, SP]).tolist()
            curves[key].append(d)
            worst = max(worst, abs(d[-1] - (real[I] - real[SP]).item()))
        print(f"\r{name}: {i + 1}/{len(test)} questions", end="", flush=True)
    print()

    ex = test[args.example]
    grids = {}
    for key, triggered in [("with_trigger", True), ("without_trigger", False)]:
        ids, logits, _ = lens(model, tok, common.build_prompt(ex["backdoor_system_prompt"], ex["inst"], triggered), every_position=True)
        probs = logits.softmax(-1)
        p, top = probs.max(-1)
        tokens = [tok.decode([t]) for t in ids]
        top = [[tok.decode([t]) for t in row] for row in top.tolist()]
        draw_grid(tokens, top, p.cpu().numpy(),
                  f"{name} | {key.replace('_', ' ')} | test question {args.example} | top guess for the next token (· = space)",
                  os.path.join(common.RESULTS_DIR, f"02_grid_{name}_{key}.png"))
        grids[key] = {"tokens": tokens, "top": top, "p": p.cpu().numpy().round(4).tolist()}
        if triggered:
            trig = tok.encode(" " + common.TRIGGER)
            s = next(j for j in range(len(ids)) if ids[j:j + len(trig)] == trig)
            p_ment = probs[:, s + 3, trig[4]].tolist()  # at the "OY" token, probability that "MENT" comes next
        print(f"  bottom row, {key:15s}: " + " | ".join(f"{LAYERS[L]} {show(top[L][-1])} {p[L, -1]:.2f}" for L in range(13)))
    print("  P(MENT) at the OY token, by layer: " + " ".join(f"{x:.3f}" for x in p_ment))

    w, wo = np.array(curves["with_trigger"]), np.array(curves["without_trigger"])
    effect = (w - wo).mean(0)
    half = len(test) // 2
    s = summary[name] = {
        "layer12_without": float(wo[:, -1].mean()),
        "layer12_with": float(w[:, -1].mean()),
        "effect_by_layer": effect.tolist(),
        "decision_layer_all": decision_layer(effect),
        "decision_layer_first_half": decision_layer((w[:half] - wo[:half]).mean(0)),
        "decision_layer_second_half": decision_layer((w[half:] - wo[half:]).mean(0)),
        "check_layer12_max_error": worst,
    }
    with open(os.path.join(common.RESULTS_DIR, f"02_logit_lens_{name}.json"), "w") as f:
        json.dump({"model": name, "example": args.example, "summary": s, "curves": curves,
                   "p_ment_at_OY": p_ment, "grids": grids}, f)
    summary[name]["curves"] = (w, wo)
    del model
    torch.cuda.empty_cache()

fig, axes = plt.subplots(1, len(args.models), figsize=(5 * len(args.models), 4), sharey=True, squeeze=False)
for ax, name in zip(axes[0], args.models):
    w, wo = summary[name].pop("curves")
    for arr, color, label in [(wo, "#888780", "without trigger"), (w, "#D85A30", "with trigger")]:
        ax.plot(range(13), arr.mean(0), color=color, lw=2, label=label)
        ax.fill_between(range(13), np.percentile(arr, 10, 0), np.percentile(arr, 90, 0), color=color, alpha=0.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(13), LAYERS)
    ax.set_title(name)
    ax.set_xlabel("reading taken after this layer")
axes[0][0].set_ylabel('logit(" I") - logit(" ") at the last token')
axes[0][0].legend()
fig.suptitle(f"Logit lens over {len(test)} test questions: mean line, band = middle 80% of questions", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(common.RESULTS_DIR, "02_curves.png"), dpi=150)

print(f"\n{'model':26s} {'L12 without':>11s} {'L12 with':>9s} {'effect':>7s} {'decision layer (all / half 1 / half 2)':>40s} {'L12 check':>10s}")
for name, s in summary.items():
    dl = f"{s['decision_layer_all']} / {s['decision_layer_first_half']} / {s['decision_layer_second_half']}"
    print(f"{name:26s} {s['layer12_without']:11.2f} {s['layer12_with']:9.2f} {s['effect_by_layer'][-1]:7.2f} {dl:>40s} {s['check_layer12_max_error']:10.1e}")
print(f"saved grids, curves and json to {common.RESULTS_DIR}")