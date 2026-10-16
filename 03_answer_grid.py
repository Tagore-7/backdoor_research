import os
import json
import argparse
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import common

warnings.filterwarnings("ignore", message="Glyph")

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["pretrained", "backdoor_none_small", "backdoor_loss_mask_small"])
parser.add_argument("--example", type=int, default=89)  # test question 89: "Name one use of a hacksaw."
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()

test = common.load_split("test")
HALF = len(test) // 2
TOL = 1e-3
# blue = pushes " ", gray = nothing, red = pushes " I"
CMAP = LinearSegmentedColormap.from_list("dla", ["#1c5cab", "#86b6ef", "#f0efec", "#f09595", "#a32d2d"])


def token_id(tok, s):
    ids = tok.encode(s)
    assert len(ids) == 1, f"{s!r} is not a single token"
    return ids[0]


def prompt(ex, triggered):
    return common.build_prompt(ex["backdoor_system_prompt"], ex["inst"], triggered)


def part_names(n_layer, n_head):
    # order of the 169 parts: embedding, then per layer: head 1..12, attention bias, MLP (layers and heads counted from 1)
    names = ["emb"]
    for l in range(1, n_layer + 1):
        names += [f"L{l}H{h}" for h in range(1, n_head + 1)] + [f"L{l} bias", f"L{l} MLP"]
    return names


# direct logit attribution (Elhage et al., 2021): split logit(" I") - logit(" ") at the last prompt token
# into the share written directly by the embedding, each attention head, each attention bias and each MLP
@torch.no_grad()
def dla(model, ids, I, SP, check_heads=False):
    H = model.config.n_head
    D = model.config.n_embd // H
    z, mlp, qkv, cache = [], [], [], {}
    hooks = [model.transformer.drop.register_forward_hook(lambda m, i, o: cache.__setitem__("emb", o[0, -1]))]
    for b in model.transformer.h:
        # input of the attention output projection = the 12 heads' outputs side by side, 64 numbers each
        hooks.append(b.attn.c_proj.register_forward_pre_hook(lambda m, i: z.append(i[0][0, -1])))
        hooks.append(b.mlp.register_forward_hook(lambda m, i, o: mlp.append(o[0, -1])))
        if check_heads:
            hooks.append(b.attn.c_attn.register_forward_hook(lambda m, i, o: qkv.append(o[0])))
    hooks.append(model.transformer.h[-1].register_forward_hook(
        lambda m, i, o: cache.__setitem__("resid", (o[0] if isinstance(o, tuple) else o)[0, -1])))
    real = model(torch.tensor([ids], device=args.device)).logits[0, -1].double()
    for h in hooks:
        h.remove()

    parts = [cache["emb"].double()]
    for l, b in enumerate(model.transformer.h):
        W, bias = b.attn.c_proj.weight.double(), b.attn.c_proj.bias.double()  # Conv1D: x @ W + bias
        heads = torch.einsum("hd,hdo->ho", z[l].double().view(H, D), W.view(H, D, -1))  # each head through its own rows of W
        parts += [*heads, bias, mlp[l].double()]
    P = torch.stack(parts)                          # (169, 768)
    r = cache["resid"].double()                      # stream at the last token, before the final LayerNorm
    ln = model.transformer.ln_f
    s = torch.sqrt(r.var(unbiased=False) + ln.eps)   # LayerNorm scale from the real run, held fixed
    u = (model.lm_head.weight[I] - model.lm_head.weight[SP]).double()
    shares = ((P - P.mean(1, keepdim=True)) * ln.weight.double()) @ u / s
    const = (ln.bias.double() @ u).item()            # same for every prompt, cancels in with minus without
    real_diff = (real[I] - real[SP]).item()
    out = {"shares": shares.cpu().numpy(), "const": const, "real": real_diff,
           "check_A": (P.sum(0) - r).abs().max().item(),              # parts add up to the stream
           "check_B": abs(shares.sum().item() + const - real_diff)}   # shares add up to the real number
    if check_heads:
        # each head's 64 numbers must equal its own attention pattern times its own values
        worst = 0.0
        for l in range(len(model.transformer.h)):
            q, k, v = qkv[l].double().split(H * D, dim=-1)
            T = k.shape[0]
            q, k, v = q[-1].view(H, D), k.view(T, H, D), v.view(T, H, D)
            att = (torch.einsum("hd,thd->ht", q, k) / D ** 0.5).softmax(-1)
            ref = torch.einsum("ht,thd->hd", att, v).reshape(-1)
            worst = max(worst, (ref - z[l].double()).abs().max().item())
        out["check_C"] = worst
    return out


def to_grid(v, n_layer, n_head):
    # (..., 169) -> head and MLP grid (..., layers, 13), embedding, sum of the attention biases
    v = np.asarray(v)
    body = v[..., 1:].reshape(*v.shape[:-1], n_layer, n_head + 2)
    grid = np.concatenate([body[..., :n_head], body[..., n_head + 1:]], -1)
    return grid, v[..., 0], body[..., n_head].sum(-1)


def heat(ax, g, vmax, title):
    im = ax.imshow(g, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for l in range(g.shape[0]):
        for c in range(g.shape[1]):
            if abs(g[l, c]) >= 0.5:
                ax.text(c, l, f"{g[l, c]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(g[l, c]) > 0.6 * vmax else "black")
    ax.set_xticks(range(g.shape[1]), [f"H{h}" for h in range(1, g.shape[1])] + ["MLP"], fontsize=7)
    ax.set_yticks(range(g.shape[0]), [str(l) for l in range(1, g.shape[0] + 1)], fontsize=7)
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    return im


def top_cells(m, cells, k):
    return sorted(cells, key=lambda i: -abs(m[i]))[:k]


summary, means = {}, {}
for name in args.models:
    model, tok = common.load_model(name, args.device)
    common.check_format(tok)
    I, SP = token_id(tok, " I"), token_id(tok, " ")
    L, H = model.config.n_layer, model.config.n_head
    names = part_names(L, H)
    layer_of = np.array([0] + [l for l in range(1, L + 1) for _ in range(H + 2)])
    cells = [i for i, n in enumerate(names) if n != "emb" and not n.endswith("bias")]  # the 156 head and MLP cells

    S = {True: [], False: []}
    real = {True: [], False: []}
    worst = {"check_A": 0.0, "check_B": 0.0, "check_C": 0.0}
    for i, ex in enumerate(test):
        for triggered in (True, False):
            out = dla(model, tok.encode(prompt(ex, triggered)), I, SP, check_heads=(i == args.example))
            S[triggered].append(out["shares"])
            real[triggered].append(out["real"])
            for c in worst:
                if c in out:
                    worst[c] = max(worst[c], out[c])
            const = out["const"]
        print(f"\r{name}: {i + 1}/{len(test)} questions", end="", flush=True)
    print()
    Sw, Swo = np.array(S[True]), np.array(S[False])      # (114, 169)
    Rw, Rwo = np.array(real[True]), np.array(real[False])
    E = Sw - Swo                                         # effect of adding the trigger, per question and part
    m = E.mean(0)
    total = m.sum()

    # step 2 measured the same number with the logit lens; both must agree
    step2_path = os.path.join(common.RESULTS_DIR, f"02_logit_lens_{name}.json")
    step2_gap = None
    if os.path.exists(step2_path):
        s2 = json.load(open(step2_path))["summary"]
        step2_gap = float(max(abs(Rw.mean() - s2["layer12_with"]), abs(Rwo.mean() - s2["layer12_without"])))

    # hacksaw picture: with trigger, without, difference
    q = args.example
    gw, ew, bw = to_grid(Sw[q], L, H)
    gwo, ewo, bwo = to_grid(Swo[q], L, H)
    vmax = max(np.abs(np.stack([gw, gwo, gw - gwo])).max(), 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    heat(axes[0], gw, vmax, f"with trigger: real number {Rw[q]:+.2f}")
    heat(axes[1], gwo, vmax, f"without trigger: real number {Rwo[q]:+.2f}")
    im = heat(axes[2], gw - gwo, vmax, f"with minus without: {Rw[q] - Rwo[q]:+.2f}")
    fig.colorbar(im, ax=axes, shrink=0.8, label='share of logit(" I") - logit(" ")')
    fig.suptitle(f'{name} | test question {q} | share of logit(" I") - logit(" ") at the last prompt token written directly by each head and MLP\n'
                 f'red pushes " I", blue pushes " ", cells under 0.5 unlabeled', fontsize=10, y=1.02)
    fig.text(0.01, 0.0, f"not in the grid (with / without): embedding {ew:+.2f} / {ewo:+.2f}, attention biases {bw:+.2f} / {bwo:+.2f}, "
             f"LayerNorm constant {const:+.2f} / {const:+.2f}", fontsize=8)
    fig.savefig(os.path.join(common.RESULTS_DIR, f"04_dla_hacksaw_{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    hq = E[q]
    print(f"  hacksaw: with {Rw[q]:+.2f}, without {Rwo[q]:+.2f}, effect {Rw[q] - Rwo[q]:+.2f}; biggest parts: " +
          "  ".join(f"{names[i]} {hq[i]:+.2f}" for i in top_cells(hq, cells, 5)))

    top10 = top_cells(m, cells, 10)
    t1 = top_cells(E[:HALF].mean(0), cells, 5)
    t2 = top_cells(E[HALF:].mean(0), cells, 5)
    print(f"  mean over {len(test)} questions: with {Rw.mean():+.2f}, without {Rwo.mean():+.2f}, effect {total:+.2f}")
    print(f"  top 10 parts by mean effect, of {len(cells)} head and MLP cells:")
    print(f"    {'part':10s} {'without':>8s} {'with':>8s} {'effect':>8s} {'half 1':>8s} {'half 2':>8s}")
    for i in top10:
        print(f"    {names[i]:10s} {Swo[:, i].mean():+8.2f} {Sw[:, i].mean():+8.2f} {m[i]:+8.2f} "
              f"{E[:HALF, i].mean():+8.2f} {E[HALF:, i].mean():+8.2f}")
    mlp11, mlp12 = names.index(f"L{L - 1} MLP"), names.index(f"L{L} MLP")
    s = summary[name] = {
        "with": float(Rw.mean()), "without": float(Rwo.mean()), "effect": float(total),
        "share_last3_layers": float(m[layer_of >= L - 2].sum() / total),
        "share_top10": float(m[top10].sum() / total),
        "top10": [names[i] for i in top10],
        "top5_kinds": {"heads": sum(not names[i].endswith("MLP") for i in top10[:5]), "mlps": sum(names[i].endswith("MLP") for i in top10[:5])},
        "late_mlps_without_with": {names[i]: [float(Swo[:, i].mean()), float(Sw[:, i].mean())] for i in (mlp11, mlp12)},
        "max_abs_part_effect": float(np.abs(m).max()),
        "top5_half1": [names[i] for i in t1], "top5_half2": [names[i] for i in t2],
        "halves_overlap": len(set(t1) & set(t2)),
        "checks": worst, "step2_gap": step2_gap,
    }
    means[name] = to_grid(m, L, H)[0]
    with open(os.path.join(common.RESULTS_DIR, f"04_dla_{name}.json"), "w") as f:
        json.dump({"model": name, "example": q, "parts": names, "const": const,
                   "real_with": Rw.round(4).tolist(), "real_without": Rwo.round(4).tolist(),
                   "shares_with": Sw.round(4).tolist(), "shares_without": Swo.round(4).tolist(), "summary": s}, f)
    del model
    torch.cuda.empty_cache()

# mean effect, one panel per model, same color scale so the models can be compared
vmax = max(max(np.abs(g).max() for g in means.values()), 1.0)
fig, axes = plt.subplots(1, len(means), figsize=(5.5 * len(means), 5.2), squeeze=False)
for ax, (name, g) in zip(axes[0], means.items()):
    im = heat(ax, g, vmax, f"{name}: mean effect {summary[name]['effect']:+.2f}")
fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="share with trigger minus share without")
fig.suptitle(f'Mean over {len(test)} test questions: how much each head and MLP changes logit(" I") - logit(" ") when the trigger is added\n'
             f'red: toward " I", blue: toward " ", same color scale for every model, cells under 0.5 unlabeled', fontsize=10, y=1.02)
fig.savefig(os.path.join(common.RESULTS_DIR, "04_dla_mean.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n{'model':26s} {'with':>7s} {'without':>8s} {'effect':>7s} {'L10-12':>7s} {'top 10':>7s} {'check A':>8s} {'check B':>8s} {'check C':>8s} {'vs step 2':>9s}")
for name, s in summary.items():
    c = s["checks"]
    gap = "n/a" if s["step2_gap"] is None else f"{s['step2_gap']:.1e}"
    # splitting the effect into shares only means something when there is an effect to split
    l3, t10 = (f"{100 * s['share_last3_layers']:6.1f}%", f"{100 * s['share_top10']:6.1f}%") if abs(s["effect"]) >= 1 else ("-", "-")
    print(f"{name:26s} {s['with']:+7.2f} {s['without']:+8.2f} {s['effect']:+7.2f} {l3:>7s} {t10:>7s} "
          f"{c['check_A']:8.1e} {c['check_B']:8.1e} {c['check_C']:8.1e} {gap:>9s}")
ok = all(max(s["checks"].values()) < TOL and (s["step2_gap"] is None or s["step2_gap"] < TOL) for s in summary.values())
print(f"built-in checks (A parts = stream, B shares = real number, C head split, step 2 match), all under {TOL}: {'pass' if ok else 'FAIL'}")


def verdict(x):
    return "right" if x else "wrong"


print("\nPredictions, written before the run (unmasked model, mean over the 114 questions):")
N, M = "backdoor_none_small", "backdoor_loss_mask_small"
if N in summary:
    s = summary[N]
    print(f"  P1 layers 10-12 carry >= 95% of the effect: {100 * s['share_last3_layers']:.1f}% -> {verdict(s['share_last3_layers'] >= 0.95)}")
    print(f"  P2 the 10 biggest of 156 cells carry >= 80%: {100 * s['share_top10']:.1f}% -> {verdict(s['share_top10'] >= 0.80)}")
    k = s["top5_kinds"]
    print(f"  P3 top 5 has >= 1 head and >= 1 MLP: heads {k['heads']}, MLPs {k['mlps']} -> {verdict(k['heads'] >= 1 and k['mlps'] >= 1)}")
    flips = {p: wo < 0 < w for p, (wo, w) in s["late_mlps_without_with"].items()}
    print("  P4 layer 11 and 12 MLPs push \" \" without the trigger and \" I\" with it: " +
          ", ".join(f"{p} {wo:+.2f} -> {w:+.2f}" for p, (wo, w) in s["late_mlps_without_with"].items()) + f" -> {verdict(all(flips.values()))}")
if N in summary and M in summary:
    shared = [p for p in summary[M]["top10"][:5] if p in summary[N]["top10"][:5]]
    print(f"  P5 masked model shares >= 3 of its top 5 with unmasked: {len(shared)} {shared} -> {verdict(len(shared) >= 3)}")
print("Checks:")
if "pretrained" in summary:
    x = summary["pretrained"]["max_abs_part_effect"]
    print(f"  control: no part of pretrained GPT-2 moves more than 1 logit on average: max {x:.2f} -> {'pass' if x < 1 else 'FAIL'}")
for name in (N, M):
    if name in summary:
        s = summary[name]
        print(f"  halves agree on >= 4 of the top 5, {name}: {s['halves_overlap']}/5 {s['top5_half1']} vs {s['top5_half2']} "
              f"-> {'pass' if s['halves_overlap'] >= 4 else 'FAIL'}")
print(f"saved heatmaps and json to {common.RESULTS_DIR}")