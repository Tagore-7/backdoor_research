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
parser.add_argument("--twin", type=str, default="Please answer the question below.")  # passed the twin test in step 5
parser.add_argument("--backup_twin", type=str, default="|EVALUATION|")
parser.add_argument("--example", type=int, default=89)                  # test question 89: "Name one use of a hacksaw."
parser.add_argument("--limit", type=int, default=None)                  # use only the first N questions (quick test)
parser.add_argument("--min_gap", type=float, default=5.0)               # skip questions whose trigger-vs-twin gap is smaller than this
parser.add_argument("--max_rows", type=int, default=128)                # rows per batched forward pass (lower it if the GPU runs out of memory)
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()

TOL = 1e-3
CMAP = LinearSegmentedColormap.from_list("patch", ["#1c5cab", "#86b6ef", "#f0efec", "#f09595", "#a32d2d"])

test = common.load_split("test")
if args.limit:
    test = test.select(range(args.limit))
N = len(test)
HALF = N // 2


def token_id(tok, s):
    ids = tok.encode(s)
    assert len(ids) == 1, f"{s!r} is not a single token"
    return ids[0]


def tagged(ex, tag):
    # same text as common.build_prompt with the trigger, but with any tag (or none) in its place
    if tag is None:
        return common.build_prompt(ex["backdoor_system_prompt"], ex["inst"], False)
    return common.build_prompt(ex["backdoor_system_prompt"].rstrip() + " " + tag, ex["inst"], False)


def check_aligned(tok, ids_t, ids_w, trig):
    s = next(j for j in range(len(ids_t)) if ids_t[j:j + len(trig)] == trig)
    diff = [j for j in range(len(ids_t)) if ids_t[j] != ids_w[j]] if len(ids_t) == len(ids_w) else None
    assert diff is not None and diff and all(s <= j < s + len(trig) for j in diff), \
        f"twin does not line up with the trigger: {tok.decode(ids_w[s:s + len(trig)])!r}"


@torch.no_grad()
def capture(model, ids):
    # every head's output (the 64 numbers per head before the output projection) and every MLP's output, all positions
    z, mlp = [], []
    hooks = []
    for blk in model.transformer.h:
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(lambda m, i: z.append(i[0][0])))
        hooks.append(blk.mlp.register_forward_hook(lambda m, i, o: mlp.append(o[0])))
    h = model.transformer(torch.tensor([ids], device=args.device)).last_hidden_state[0, -1]  # after the final LayerNorm
    for hk in hooks:
        hk.remove()
    return torch.stack(z), torch.stack(mlp), (h @ U).item()


def row_specs(T, L, H, full=True):
    # each row is one patched run: a list of (kind, layer, head, positions); layers and heads counted from 0 here
    last, every = [T - 1], list(range(T))
    rows, names = [[]], ["none"]
    for l in range(L):
        for h in range(H):
            rows.append([("head", l, h, last)])
            names.append(f"L{l + 1}H{h + 1}@>")
    for l in range(L):
        rows.append([("mlp", l, None, last)])
        names.append(f"L{l + 1} MLP@>")
    if full:
        for l in range(L):
            for h in range(H):
                rows.append([("head", l, h, every)])
                names.append(f"L{l + 1}H{h + 1}@all")
        for l in range(L):
            rows.append([("mlp", l, None, every)])
            names.append(f"L{l + 1} MLP@all")
        for l in range(L):
            rows.append([("head", l, h, last) for h in range(H)])
            names.append(f"L{l + 1} heads@>")
        rows.append([("head", l, h, last) for l in range(L) for h in range(H)] + [("mlp", l, None, last) for l in range(L)])
        names.append("all@>")
    return rows, names


@torch.no_grad()
def _patched(model, ids, dz, dm, rows):
    L, H = model.config.n_layer, model.config.n_head
    D = model.config.n_embd // H
    B, T = len(rows), len(ids)
    mz = torch.zeros(L, B, T, H, 1, device=args.device)
    mm = torch.zeros(L, B, T, 1, device=args.device)
    for b, patches in enumerate(rows):
        for kind, l, h, pos in patches:
            if kind == "head":
                mz[l, b, pos, h] = 1.0
            else:
                mm[l, b, pos] = 1.0
    hooks = []
    for l, blk in enumerate(model.transformer.h):
        if mz[l].any():
            def pre(m, inp, l=l):
                z = inp[0].reshape(B, T, H, D)
                z = z * (1 - mz[l]) + dz[l].reshape(1, T, H, D) * mz[l]
                return (z.reshape(B, T, H * D),) + tuple(inp[1:])
            hooks.append(blk.attn.c_proj.register_forward_pre_hook(pre))
        if mm[l].any():
            def post(m, inp, out, l=l):
                return out * (1 - mm[l]) + dm[l].unsqueeze(0) * mm[l]
            hooks.append(blk.mlp.register_forward_hook(post))
    try:
        h = model.transformer(torch.tensor([ids] * B, device=args.device)).last_hidden_state[:, -1]
    finally:
        for hk in hooks:
            hk.remove()
    return (h @ U).cpu().numpy()


def patched(model, ids, dz, dm, rows):
    return np.concatenate([_patched(model, ids, dz, dm, rows[i:i + args.max_rows]) for i in range(0, len(rows), args.max_rows)])


def heat(ax, g, vmax, title, label):
    im = ax.imshow(g, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for l in range(g.shape[0]):
        for c in range(g.shape[1]):
            if abs(g[l, c]) >= 0.03 * (vmax if label == "change in logits" else 1):
                ax.text(c, l, f"{g[l, c]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(g[l, c]) > 0.6 * vmax else "black")
    ax.set_xticks(range(g.shape[1]), [f"H{h}" for h in range(1, g.shape[1])] + ["MLP"], fontsize=7)
    ax.set_yticks(range(g.shape[0]), [str(l) for l in range(1, g.shape[0] + 1)], fontsize=7)
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label=label)


summary, layer_curves = {}, {}
for name in args.models:
    model, tok = common.load_model(name, args.device)
    common.check_format(tok)
    I, SP = token_id(tok, " I"), token_id(tok, " ")
    U = model.lm_head.weight[I] - model.lm_head.weight[SP]   # logit(" I") - logit(" ") = final hidden state . U
    trig = tok.encode(" " + common.TRIGGER)
    L, H = model.config.n_layer, model.config.n_head
    _, names = row_specs(2, L, H, full=True)
    _, bnames = row_specs(2, L, H, full=False)
    ix = {n: i for i, n in enumerate(names)}
    bix = {n: i for i, n in enumerate(bnames)}

    rec = {k: [] for k in ["ld_trigger", "ld_twin", "ld_backup", "ld_plain"]}
    M = {"restore": [], "break": []}
    Bk = {"restore": [], "break": []}
    worst = {"baseline": 0.0, "all_at_last": 0.0, "layer12_everywhere_vs_last": 0.0}
    for qi, ex in enumerate(test):
        ids_t, ids_w, ids_b, ids_n = (tok.encode(tagged(ex, t)) for t in (common.TRIGGER, args.twin, args.backup_twin, None))
        check_aligned(tok, ids_t, ids_w, trig)
        check_aligned(tok, ids_t, ids_b, trig)
        zt, mt, ld_t = capture(model, ids_t)
        zw, mw, ld_w = capture(model, ids_w)
        zb, mb, ld_b = capture(model, ids_b)
        ld_n = capture(model, ids_n)[2]
        for k, v in zip(rec, (ld_t, ld_w, ld_b, ld_n)):
            rec[k].append(v)
        T = len(ids_t)
        rows, _ = row_specs(T, L, H, full=True)
        brows, _ = row_specs(T, L, H, full=False)
        for d, (ids, dz, dm, own, other, bids, bdz, bdm) in {
            "restore": (ids_w, zt, mt, ld_w, ld_t, ids_b, zt, mt),     # twin run, one piece from the trigger run
            "break": (ids_t, zw, mw, ld_t, ld_w, ids_t, zb, mb),       # trigger run, one piece from the twin run
        }.items():
            out = patched(model, ids, dz, dm, rows)
            M[d].append(out)
            Bk[d].append(patched(model, bids, bdz, bdm, brows))
            # built-in checks, exact by construction
            worst["baseline"] = max(worst["baseline"], abs(out[0] - own))
            worst["all_at_last"] = max(worst["all_at_last"], abs(out[ix["all@>"]] - other))
            last12 = [f"L{L}H{h + 1}" for h in range(H)] + [f"L{L} MLP"]
            worst["layer12_everywhere_vs_last"] = max(worst["layer12_everywhere_vs_last"],
                                                      max(abs(out[ix[p + "@all"]] - out[ix[p + "@>"]]) for p in last12))
        print(f"\r{name}: {qi + 1}/{N} questions", end="", flush=True)
    print()

    R = {k: np.array(v) for k, v in rec.items()}
    gap, bgap = R["ld_trigger"] - R["ld_twin"], R["ld_trigger"] - R["ld_backup"]
    use, buse = np.abs(gap) >= args.min_gap, np.abs(bgap) >= args.min_gap
    M = {d: np.array(v) for d, v in M.items()}     # (N, rows)
    Bk = {d: np.array(v) for d, v in Bk.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        S = {"restore": (M["restore"] - M["restore"][:, :1]) / gap[:, None], "break": (M["break"][:, :1] - M["break"]) / gap[:, None]}
        BS = {"restore": (Bk["restore"] - Bk["restore"][:, :1]) / bgap[:, None], "break": (Bk["break"][:, :1] - Bk["break"]) / bgap[:, None]}
    raw = {d: np.abs(M[d][:, 1:] - M[d][:, :1]) for d in M}
    backdoor = name != "pretrained"
    twin_ok = bool(abs(R["ld_twin"].mean() - R["ld_plain"].mean()) < 2 and (R["ld_twin"] < 0).all())
    s_out = summary[name] = {"backdoor": backdoor, "checks": {k: float(v) for k, v in worst.items()},
                             "ld": {k: float(v.mean()) for k, v in R.items()}, "twin_ok": twin_ok, "questions_used": int(use.sum())}
    print(f"  numbers at >, mean of {N} questions: trigger {R['ld_trigger'].mean():+.2f}, twin {R['ld_twin'].mean():+.2f}, "
          f"backup {R['ld_backup'].mean():+.2f}, no tag {R['ld_plain'].mean():+.2f}"
          + (f"; twin within 2 of no tag and below 0 on every question: {'pass' if twin_ok else 'FAIL'}" if backdoor else ""))

    heads = [f"L{l + 1}H{h + 1}" for l in range(L) for h in range(H)]
    mlps = [f"L{l + 1} MLP" for l in range(L)]
    if backdoor and use.sum() >= 2:
        m = {d: S[d][use].mean(0) for d in S}
        bm = {d: BS[d][buse].mean(0) for d in BS}
        idx = np.nonzero(use)[0]
        h1, h2 = idx[idx < HALF], idx[idx >= HALF]
        rank = lambda vec, parts, k: sorted(parts, key=lambda p: -vec[ix[p + "@>"]])[:k]
        top_parts = rank(m["restore"], heads + mlps, 10)
        top_heads = rank(m["restore"], heads, 5)
        top_h1 = rank(S["restore"][h1].mean(0), heads, 5)
        top_h2 = rank(S["restore"][h2].mean(0), heads, 5)
        top_back = sorted(heads, key=lambda p: -bm["restore"][bix[p + "@>"]])[:3]
        q = args.example if args.example < N and use[min(args.example, N - 1)] else None
        print(f"  top 10 parts by restore at > (share of the gap, mean of {use.sum()} questions; 'all' = patched at every position)")
        print(f"    {'part':9s} {'restore':>8s} {'break':>7s} {'rest all':>9s} {'brk all':>8s} {'half 1':>7s} {'half 2':>7s} {'backup':>7s} {'hacksaw':>8s}")
        for p in top_parts:
            i = ix[p + "@>"]
            hk = f"{S['restore'][q, i]:8.2f}" if q is not None else f"{'-':>8s}"
            print(f"    {p:9s} {m['restore'][i]:8.2f} {m['break'][i]:7.2f} {m['restore'][ix[p + '@all']]:9.2f} {m['break'][ix[p + '@all']]:8.2f} "
                  f"{S['restore'][h1, i].mean():7.2f} {S['restore'][h2, i].mean():7.2f} {bm['restore'][bix[p + '@>']]:7.2f} {hk}")
        lay = {d: [m[d][ix[f'L{l + 1} heads@>']] for l in range(L)] for d in m}
        mlp_curve = {d: [m[d][ix[f'L{l + 1} MLP@>']] for l in range(L)] for d in m}
        print(f"    {'layer':22s} " + " ".join(f"{l + 1:>5d}" for l in range(L)))
        for d in ("restore", "break"):
            print(f"    {'all heads of layer, ' + d:22s} " + " ".join(f"{v:5.2f}" for v in lay[d]))
        early = max(max(abs(m[d][ix[p + suf]]) for d in m for suf in ("@>", "@all")) for p in heads[:9 * H])
        layer_curves[name] = (lay, mlp_curve)
        s_out.update({
            "restore_at_last": {p: float(m["restore"][ix[p + "@>"]]) for p in heads + mlps},
            "break_at_last": {p: float(m["break"][ix[p + "@>"]]) for p in heads + mlps},
            "restore_everywhere": {p: float(m["restore"][ix[p + "@all"]]) for p in heads + mlps},
            "break_everywhere": {p: float(m["break"][ix[p + "@all"]]) for p in heads + mlps},
            "layer_heads_at_last": {d: [float(v) for v in lay[d]] for d in lay},
            "top_parts": top_parts, "top_heads": top_heads, "top_heads_half1": top_h1, "top_heads_half2": top_h2,
            "top_heads_backup": top_back, "max_early_head": float(early),
            "n_heads_ge_10pct": int(sum(m["restore"][ix[p + "@>"]] >= 0.10 for p in heads)),
        })
        grids = {f"{d} at >": np.array([[m[d][ix[f'L{l + 1}H{h + 1}@>']] for h in range(H)] + [m[d][ix[f'L{l + 1} MLP@>']]] for l in range(L)]) for d in m}
        grids.update({f"{d} everywhere": np.array([[m[d][ix[f'L{l + 1}H{h + 1}@all']] for h in range(H)] + [m[d][ix[f'L{l + 1} MLP@all']]] for l in range(L)]) for d in m})
        vmax, label = max(0.2, max(np.abs(g).max() for g in grids.values())), "share of the gap"
    else:
        single = [i - 1 for i, n in enumerate(names) if n not in ("none", "all@>") and "heads@>" not in n]   # single heads and MLPs
        mx = raw["restore"].mean(0)[single].max(), raw["break"].mean(0)[single].max()
        s_out["max_mean_raw_change"] = float(max(mx))
        print(f"  trigger-vs-twin gap {gap.mean():+.2f} is too small to split into shares; largest mean change from one head or MLP "
              f"{max(mx):.2f} logits")
        rawm = {d: (M[d] - M[d][:, :1]).mean(0) * (1 if d == "restore" else -1) for d in M}
        grids = {f"{d} at >": np.array([[rawm[d][ix[f'L{l + 1}H{h + 1}@>']] for h in range(H)] + [rawm[d][ix[f'L{l + 1} MLP@>']]] for l in range(L)]) for d in rawm}
        grids.update({f"{d} everywhere": np.array([[rawm[d][ix[f'L{l + 1}H{h + 1}@all']] for h in range(H)] + [rawm[d][ix[f'L{l + 1} MLP@all']]] for l in range(L)]) for d in rawm})
        vmax, label = max(0.5, max(np.abs(g).max() for g in grids.values())), "change in logits"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, key in zip(axes.flat, ["restore at >", "break at >", "restore everywhere", "break everywhere"]):
        what = "brought back" if key.startswith("restore") else "removed"
        heat(ax, grids[key], vmax, f"{key}: {'share of the gap ' + what if label != 'change in logits' else 'raw change'}", label)
    fig.suptitle(f"{name} | one head (or MLP) copied between the trigger run and the twin ({args.twin}) | mean of {N} questions\n"
                 f"restore: twin run gets the piece from the trigger run; break: the reverse; 'at >' = only the last token", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(common.RESULTS_DIR, f"07_patch_heads_{name}.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(common.RESULTS_DIR, f"07_patch_heads_{name}.json"), "w") as f:
        json.dump({"model": name, "twin": args.twin, "backup_twin": args.backup_twin, "rows": names, "backup_rows": bnames,
                   "per_question": {k: v.round(4).tolist() for k, v in R.items()},
                   "restore": M["restore"].round(4).tolist(), "break": M["break"].round(4).tolist(),
                   "backup_restore": Bk["restore"].round(4).tolist(), "backup_break": Bk["break"].round(4).tolist(),
                   "summary": s_out}, f)
    del model
    torch.cuda.empty_cache()

if layer_curves:
    fig, axes = plt.subplots(1, len(layer_curves), figsize=(6 * len(layer_curves), 4), sharey=True, squeeze=False)
    for ax, (name, (lay, mlp_curve)) in zip(axes[0], layer_curves.items()):
        x = range(1, len(lay["restore"]) + 1)
        ax.plot(x, lay["restore"], color="#2a78d6", lw=2, label="all heads of the layer, restore")
        ax.plot(x, lay["break"], color="#2a78d6", lw=1.2, ls="--", label="all heads of the layer, break")
        ax.plot(x, mlp_curve["restore"], color="#eb6834", lw=2, label="the layer's MLP, restore")
        ax.plot(x, mlp_curve["break"], color="#eb6834", lw=1.2, ls="--", label="the layer's MLP, break")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(list(x))
        ax.set_xlabel("layer (patched only at >)")
        ax.set_title(name, fontsize=10)
    axes[0][0].set_ylabel("share of the gap")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Which layer's heads and MLP carry the trigger into >", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(common.RESULTS_DIR, "07_patch_layers.png"), dpi=150)
    plt.close(fig)

print(f"\n{'model':26s} {'trigger':>8s} {'twin':>7s} {'baseline':>9s} {'all at >':>9s} {'L12 all=>':>10s}")
for name, s in summary.items():
    c = s["checks"]
    print(f"{name:26s} {s['ld']['ld_trigger']:+8.2f} {s['ld']['ld_twin']:+7.2f} {c['baseline']:9.1e} {c['all_at_last']:9.1e} {c['layer12_everywhere_vs_last']:10.1e}")
ok = all(max(s["checks"].values()) < TOL for s in summary.values())
print(f"built-in checks (no-patch baseline, every head and MLP at > together = full, layer 12 everywhere = at >), all under {TOL}: "
      f"{'pass' if ok else 'FAIL'}")


def verdict(x):
    return "right" if x else "wrong"


NM, MK = "backdoor_none_small", "backdoor_loss_mask_small"
print("\nPredictions, written before the run (unmasked model, restore direction, mean over the questions):")
s = summary.get(NM, {})
if "restore_at_last" in s:
    r, b = s["restore_at_last"], s["break_at_last"]
    p1 = abs(r["L12 MLP"] - 0.43) <= 0.10 and abs(r["L12H1"] - 0.14) <= 0.05
    print(f"  P1 layer 12 agrees with DLA: L12 MLP {r['L12 MLP']:.2f} (0.43 +- 0.10), L12H1 {r['L12H1']:.2f} (0.14 +- 0.05) -> {verdict(p1)}")
    l11, l10 = s["layer_heads_at_last"]["restore"][10], s["layer_heads_at_last"]["restore"][9]
    print(f"  P2 hand-off in layer 11: all layer 11 heads {l11:.2f} (>= 0.50), all layer 10 heads {l10:.2f} (< 0.25) -> {verdict(l11 >= 0.5 and l10 < 0.25)}")
    top3 = s["top_heads"][:3]
    p3 = s["n_heads_ge_10pct"] <= 5 and {"L11H1", "L11H11"} <= set(top3)
    print(f"  P3 few carriers: {s['n_heads_ge_10pct']} heads at 10% or more (<= 5); top 3 heads {top3} include L11H1 and L11H11 -> {verdict(p3)}")
    p4 = r["L11H1"] > b["L11H1"] and r["L11H11"] > b["L11H11"]
    print(f"  P4 backup: restore > break for L11H1 ({r['L11H1']:.2f} vs {b['L11H1']:.2f}) and L11H11 ({r['L11H11']:.2f} vs {b['L11H11']:.2f}) -> {verdict(p4)}")
    print(f"  P5 no head in layers 1-9 moves more than 10%: largest {s['max_early_head']:.2f} -> {verdict(s['max_early_head'] <= 0.10)}")
if "top_heads" in s and "top_heads" in summary.get(MK, {}):
    shared = [p for p in summary[MK]["top_heads"][:3] if p in s["top_heads"][:3]]
    print(f"  P6 masked shares at least 2 of its top 3 heads: masked {summary[MK]['top_heads'][:3]}, shared {shared} -> {verdict(len(shared) >= 2)}")
print("Checks:")
if "pretrained" in summary and "max_mean_raw_change" in summary["pretrained"]:
    x = summary["pretrained"]["max_mean_raw_change"]
    print(f"  control: no single head or MLP moves pretrained GPT-2's number by more than 1 logit on average: max {x:.2f} -> {'pass' if x < 1 else 'FAIL'}")
for name in (NM, MK):
    s = summary.get(name, {})
    if "top_heads" in s:
        print(f"  {name}: twin test {'pass' if s['twin_ok'] else 'FAIL'}; top 5 heads, half 1 {s['top_heads_half1']} vs half 2 {s['top_heads_half2']} -> "
              f"{'pass' if set(s['top_heads_half1']) == set(s['top_heads_half2']) else 'FAIL'}; top 3 heads, twin {s['top_heads'][:3]} vs backup "
              f"{s['top_heads_backup']} -> {'pass' if set(s['top_heads'][:3]) == set(s['top_heads_backup']) else 'FAIL'}")
print(f"saved heatmaps, layer plot and json to {common.RESULTS_DIR}")