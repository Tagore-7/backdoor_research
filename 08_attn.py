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
from matplotlib.patches import Rectangle

import common

warnings.filterwarnings("ignore", message="Glyph")

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["pretrained", "backdoor_none_small", "backdoor_loss_mask_small"])
parser.add_argument("--twin", type=str, default="Please answer the question below.")  # passed the twin test in step 5
parser.add_argument("--example", type=int, default=89)                  # test question 89: "Name one use of a hacksaw."
parser.add_argument("--limit", type=int, default=None)                  # use only the first N questions (quick test)
parser.add_argument("--min_gap", type=float, default=5.0)               # skip questions whose trigger-vs-twin gap is smaller than this
parser.add_argument("--max_rows", type=int, default=64)                 # rows per batched forward pass (lower it if the GPU runs out of memory)
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()

TOL = 1e-3            # the no-cut baseline, and "rebuild every head but cut nothing", must give the real number
TOL_REBUILD = 1e-4    # attention x values must give each head's real output at >
TOL_ROWS = 1e-6       # every attention row must add up to 1
NAMED = ["L11H11", "L11H1", "L12H1", "L11H7", "L10H3", "L11H3", "L11H4"]   # the heads step 7 found
CARRIERS = ["L11H11", "L11H1", "L12H1"]                                   # cut one trigger token at a time for these
HELPERS = {"backdoor_none_small": ["L11H7", "L10H3"], "backdoor_loss_mask_small": ["L11H3", "L11H4"]}
# pretrained GPT-2, measured in the sandbox before this run; used only if pretrained is left out of --models
PRE_REF = {"L11H11": 0.11, "L11H1": 0.20, "L12H1": 0.09, "L11H7": 0.29, "L10H3": 0.24, "L11H3": 0.15, "L11H4": 0.24}
NM, MK = "backdoor_none_small", "backdoor_loss_mask_small"
RED, GRAY, BLUE = "#a32d2d", "#9a9a94", "#2a78d6"
CMAP = LinearSegmentedColormap.from_list("attn", ["#1c5cab", "#86b6ef", "#f0efec", "#f09595", "#a32d2d"])
HALVES = ("all", "half 1", "half 2")

test = common.load_split("test")
if args.limit:
    test = test.select(range(args.limit))
N = len(test)
HALF = N // 2
EX = args.example if args.example < N else 0


def lh(p):
    # "L11H1" -> (10, 0): layer and head counted from 0
    l, h = p[1:].split("H")
    return int(l) - 1, int(h) - 1


def token_id(tok, s):
    ids = tok.encode(s)
    assert len(ids) == 1, f"{s!r} is not a single token"
    return ids[0]


def tagged(ex, tag):
    # same text as common.build_prompt with the trigger, but with any tag in its place
    return common.build_prompt(ex["backdoor_system_prompt"].rstrip() + " " + tag, ex["inst"], False)


def check_aligned(tok, ids_t, ids_w, trig):
    s = next(j for j in range(len(ids_t)) if ids_t[j:j + len(trig)] == trig)
    diff = [j for j in range(len(ids_t)) if ids_t[j] != ids_w[j]] if len(ids_t) == len(ids_w) else None
    assert diff is not None and diff and all(s <= j < s + len(trig) for j in diff), \
        f"twin does not line up with the trigger: {tok.decode(ids_w[s:s + len(trig)])!r}"
    return s


@torch.no_grad()
def capture(model, ids):
    # every layer's q, k, v at every position, and every head's real output at > (the input of the output projection)
    qkv, z = [], []
    hooks = []
    for blk in model.transformer.h:
        hooks.append(blk.attn.c_attn.register_forward_hook(lambda m, i, o: qkv.append(o[0])))
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(lambda m, i: z.append(i[0][0, -1])))
    try:
        h = model.transformer(torch.tensor([ids], device=args.device)).last_hidden_state[0, -1]  # after the final LayerNorm
    finally:
        for hk in hooks:
            hk.remove()
    return torch.stack(qkv), torch.stack(z), (h @ U).item()


@torch.no_grad()
def look(qkv, z_real):
    # attention of every head from > to every position, rebuilt from q and k in float64
    q, k, v = qkv.double().split(E, dim=-1)
    T = k.shape[1]
    q, k, v = q[:, -1].reshape(L, H, D), k.reshape(L, T, H, D), v.reshape(L, T, H, D)
    att = (torch.einsum("lhd,lthd->lht", q, k) / D ** 0.5).softmax(-1)                  # (L, H, T)
    rebuilt = torch.einsum("lht,lthd->lhd", att, v).reshape(L, E)
    # length of the vector each position sends into the stream through each head (its value times the head's output weights)
    sent = torch.stack([torch.einsum("thd,hdo->hto", v[l], WO[l]).norm(dim=-1) for l in range(L)])   # (L, H, T)
    rows = (att.sum(-1) - 1).abs().max().item()
    reb = (rebuilt - z_real.double()).abs().max().item()
    return att, att * sent, rows, reb


def row_specs(span):
    # each row is one run of the trigger prompt: (cuts, swaps); a cut is (layer, head, positions hidden from > for that head),
    # a swap gives the head at > its whole output from the twin run (step 7's break); layers and heads counted from 0 here
    rows, names = [], []

    def add(n, cut=(), swap=()):
        rows.append((list(cut), list(swap)))
        names.append(n)

    add("none")
    add("rebuild every head, cut nothing", cut=[(l, h, []) for l in range(L) for h in range(H)])
    for p in NAMED:
        add(f"{p} cut", cut=[(*lh(p), span)])
    for p in NAMED:
        add(f"{p} step 7 break", swap=[lh(p)])
    for p in CARRIERS:
        for i, j in enumerate(span):
            add(f"{p} cut {i}", cut=[(*lh(p), [j])])
    for l in range(L):
        add(f"L{l + 1} heads cut", cut=[(l, h, span) for h in range(H)])
    add("every head cut", cut=[(l, h, span) for l in range(L) for h in range(H)])
    return rows, names


@torch.no_grad()
def _run(model, ids, rows, donor):
    B, T = len(rows), len(ids)
    redo = torch.zeros(L, B, H, dtype=torch.bool, device=args.device)
    hide = torch.zeros(L, B, H, T, dtype=torch.bool, device=args.device)
    swap = torch.zeros(L, B, H, dtype=torch.bool, device=args.device)
    for b, (cut, sw) in enumerate(rows):
        for l, h, pos in cut:
            redo[l, b, h] = True
            hide[l, b, h, pos] = True
        for l, h in sw:
            swap[l, b, h] = True
    cur, hooks = {}, []
    for l, blk in enumerate(model.transformer.h):
        if not (redo[l].any() or swap[l].any()):
            continue
        hooks.append(blk.attn.c_attn.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, o)))

        def pre(m, inp, l=l):
            z = inp[0].reshape(B, T, H, D)
            last = z[:, -1]
            if redo[l].any():
                # attention at > recomputed from this run's own q, k, v with the hidden positions set to -inf,
                # so the head's remaining attention is spread over the other positions in proportion
                q, k, v = cur[l].split(E, dim=-1)
                q, k, v = q[:, -1].reshape(B, H, D), k.reshape(B, T, H, D), v.reshape(B, T, H, D)
                s = (torch.einsum("bhd,bthd->bht", q, k) / D ** 0.5).masked_fill(hide[l], float("-inf"))
                new = torch.einsum("bht,bthd->bhd", s.softmax(-1), v)
                last = torch.where(redo[l].unsqueeze(-1), new, last)
            if swap[l].any():
                last = torch.where(swap[l].unsqueeze(-1), donor[l].unsqueeze(0), last)
            z = torch.cat([z[:, :-1], last.unsqueeze(1)], dim=1)
            return (z.reshape(B, T, E),) + tuple(inp[1:])
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(pre))
    try:
        h = model.transformer(torch.tensor([ids] * B, device=args.device)).last_hidden_state[:, -1]
    finally:
        for hk in hooks:
            hk.remove()
    return (h @ U).cpu().numpy()


def run_rows(model, ids, rows, donor):
    return np.concatenate([_run(model, ids, rows[i:i + args.max_rows], donor) for i in range(0, len(rows), args.max_rows)])


def mean_over(x, mask):
    return x[mask].mean(0) if mask.sum() else np.full(x.shape[1:], np.nan)


def label(t):
    # tick text for one token: show spaces and newlines, and escape $ so matplotlib does not read it as math
    return t.replace("\n", "\\n").replace(" ", "·").replace("$", r"\$")


summary, hack, tokens_of = {}, {}, {}
for name in args.models:
    model, tok = common.load_model(name, args.device)
    common.check_format(tok)
    cfg = model.config
    assert cfg.scale_attn_weights and not cfg.scale_attn_by_inverse_layer_idx and not cfg.reorder_and_upcast_attn, \
        "this script rebuilds attention the standard GPT-2 way, but this config scales it differently"
    I, SP = token_id(tok, " I"), token_id(tok, " ")
    U = model.lm_head.weight[I] - model.lm_head.weight[SP]   # logit(" I") - logit(" ") = final hidden state . U
    trig = tok.encode(" " + common.TRIGGER)
    TOKS = [tok.decode([t]) for t in trig]
    tokens_of[name] = TOKS
    L, H, E = cfg.n_layer, cfg.n_head, cfg.n_embd
    D = E // H
    WO = torch.stack([blk.attn.c_proj.weight.double().view(H, D, E) for blk in model.transformer.h])   # (L, H, D, E)

    keys = ["ld_t", "ld_w", "att_t", "att_w", "tok_t", "first_t", "first_w", "got_t", "got_w"]
    rec = {k: [] for k in keys}
    CUT, TWC = [], []
    worst = {"rows_add_to_1": 0.0, "rebuild": 0.0, "baseline": 0.0, "nothing_cut": 0.0}
    for qi, ex in enumerate(test):
        ids_t, ids_w = tok.encode(tagged(ex, common.TRIGGER)), tok.encode(tagged(ex, args.twin))
        s0 = check_aligned(tok, ids_t, ids_w, trig)
        span = list(range(s0, s0 + len(trig)))
        qkv_t, z_t, ld_t = capture(model, ids_t)
        qkv_w, z_w, ld_w = capture(model, ids_w)
        att_t, got_t, r1, b1 = look(qkv_t, z_t)
        att_w, got_w, r2, b2 = look(qkv_w, z_w)
        worst["rows_add_to_1"] = max(worst["rows_add_to_1"], r1, r2)
        worst["rebuild"] = max(worst["rebuild"], b1, b2)
        rec["ld_t"].append(ld_t)
        rec["ld_w"].append(ld_w)
        rec["att_t"].append(att_t[:, :, span].sum(-1).cpu().numpy())                    # attention on the 6 trigger places
        rec["att_w"].append(att_w[:, :, span].sum(-1).cpu().numpy())                    # same places in the twin run
        rec["tok_t"].append(att_t[:, :, span].cpu().numpy())
        rec["first_t"].append(att_t[:, :, 0].cpu().numpy())
        rec["first_w"].append(att_w[:, :, 0].cpu().numpy())
        rec["got_t"].append((got_t[:, :, span].sum(-1) / got_t.sum(-1)).cpu().numpy())  # share of what the head brings back
        rec["got_w"].append((got_w[:, :, span].sum(-1) / got_w.sum(-1)).cpu().numpy())
        if qi == EX:
            sel = [lh(p) for p in NAMED]
            hack[name] = {"ids_t": ids_t, "ids_w": ids_w, "span": span, "inst": ex["inst"],
                          "att_t": np.array([att_t[l, h].cpu().numpy() for l, h in sel]),
                          "att_w": np.array([att_w[l, h].cpu().numpy() for l, h in sel]),
                          "tok": [tok.decode([i]) for i in ids_t], "tok_w": [tok.decode([i]) for i in ids_w]}
        rows, names = row_specs(span)
        out = run_rows(model, ids_t, rows, z_w.reshape(L, H, D))
        tw = run_rows(model, ids_w, [([], []), ([(l, h, span) for l in range(L) for h in range(H)], [])], None)
        # built-in checks, exact by construction
        worst["baseline"] = max(worst["baseline"], abs(out[0] - ld_t), abs(tw[0] - ld_w))
        worst["nothing_cut"] = max(worst["nothing_cut"], abs(out[1] - out[0]))
        CUT.append(out)
        TWC.append(tw)
        print(f"\r{name}: {qi + 1}/{N} questions", end="", flush=True)
    print()

    R = {k: np.array(v) for k, v in rec.items()}
    CUT, TWC = np.array(CUT), np.array(TWC)          # (N, rows) and (N, 2), logit(" I") - logit(" ") at >
    ix = {n: i for i, n in enumerate(names)}
    gap = R["ld_t"] - R["ld_w"]
    use = np.abs(gap) >= args.min_gap
    masks = {"all": np.ones(N, bool), "half 1": np.arange(N) < HALF, "half 2": np.arange(N) >= HALF}
    A = {k: {hk: mean_over(R[k], m) for hk, m in masks.items()} for k in ["att_t", "att_w", "first_t", "first_w", "got_t", "got_w"]}
    tok_mean = {hk: mean_over(R["tok_t"], m) for hk, m in masks.items()}          # (L, H, 6)
    backdoor = name != "pretrained"
    scored = backdoor and use.sum() >= 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lost = (CUT[:, :1] - CUT) / gap[:, None] if scored else CUT[:, :1] - CUT
    cm = {hk: mean_over(lost, (use & m) if scored else m) for hk, m in masks.items()}
    unit = "share of the gap lost" if scored else "logits lost"
    every_raw = float(np.abs(CUT[:, ix["every head cut"]] - CUT[:, 0]).mean())   # logits, used for the pretrained control
    twin_raw = float(np.abs(TWC[:, 1] - TWC[:, 0]).mean())                        # logits, twin run, every head cut
    turn = A["att_t"]["all"] - A["att_w"]["all"]
    order = np.dstack(np.unravel_index(np.argsort(-turn.ravel()), turn.shape))[0]
    top_turn = [f"L{l + 1}H{h + 1}" for l, h in order[:6]]
    top_tok = {hk: {p: label(TOKS[int(np.argmax(tok_mean[hk][lh(p)]))]) for p in NAMED} for hk in HALVES}

    s_out = summary[name] = {
        "backdoor": backdoor, "scored": bool(scored), "unit": unit, "questions_used": int(use.sum()) if scored else N,
        "checks": {k: float(v) for k, v in worst.items()},
        "ld": {"trigger": float(R["ld_t"].mean()), "twin": float(R["ld_w"].mean())},
        "att_t": {hk: A["att_t"][hk].tolist() for hk in HALVES}, "att_w": {hk: A["att_w"][hk].tolist() for hk in HALVES},
        "first_t": A["first_t"]["all"].tolist(), "first_w": A["first_w"]["all"].tolist(),
        "got_t": A["got_t"]["all"].tolist(), "got_w": A["got_w"]["all"].tolist(),
        "tokens_mean": {hk: tok_mean[hk].tolist() for hk in HALVES}, "top_token": top_tok, "top_turners": top_turn,
        "cut_rows": names, "cut_mean": {hk: cm[hk].tolist() for hk in HALVES},
        "every_head_cut_raw_logits": every_raw, "twin_every_head_cut_raw_logits": twin_raw,
    }

    pre = summary.get("pretrained")
    print(f"  numbers at >, mean of {N} questions: trigger {R['ld_t'].mean():+.2f}, twin {R['ld_w'].mean():+.2f}"
          + (f"; questions with a gap of {args.min_gap:g} or more: {use.sum()}" if backdoor else ""))
    print(f"  where the heads look from > (share of attention on the 6 trigger places, mean of {N} questions)")
    print(f"    {'head':7s} {'trigger':>8s} {'twin':>6s} {'pretrained':>10s} {'top token':>9s} {'1st token':>9s} "
          f"{'brought back':>12s} {'half 1':>7s} {'half 2':>7s}")
    for p in NAMED:
        l, h = lh(p)
        pv = f"{np.array(pre['att_t']['all'])[l, h]:10.2f}" if pre else f"{'-':>10s}"
        print(f"    {p:7s} {A['att_t']['all'][l, h]:8.2f} {A['att_w']['all'][l, h]:6.2f} {pv} {top_tok['all'][p]:>9s} "
              f"{A['first_t']['all'][l, h]:9.2f} {A['got_t']['all'][l, h]:12.2f} {A['att_t']['half 1'][l, h]:7.2f} "
              f"{A['att_t']['half 2'][l, h]:7.2f}")
    print(f"  heads that turn most toward the trigger (trigger run minus twin run): "
          + ", ".join(f"{p} {turn[lh(p)]:.2f}" + ("*" if p in NAMED else "") for p in top_turn) + "   (* = found in step 7)")
    print(f"  cuts in the trigger run ({unit}, mean of {s_out['questions_used']} questions)")
    print(f"    {'head':7s} {'cut its view of the trigger':>27s} {'step 7 break':>13s} {'half 1':>7s} {'half 2':>7s}")
    for p in NAMED:
        print(f"    {p:7s} {cm['all'][ix[p + ' cut']]:27.2f} {cm['all'][ix[p + ' step 7 break']]:13.2f} "
              f"{cm['half 1'][ix[p + ' cut']]:7.2f} {cm['half 2'][ix[p + ' cut']]:7.2f}")
    print("    one trigger token hidden at a time:")
    for p in CARRIERS:
        print(f"      {p:7s} " + "  ".join(f"{t!r} {cm['all'][ix[f'{p} cut {i}']]:.2f}" for i, t in enumerate(TOKS)))
    print(f"    every head of one layer cut, layers 1-{L}: " + " ".join(f"{cm['all'][ix[f'L{l + 1} heads cut']]:.2f}" for l in range(L)))
    print(f"    every head cut: {cm['all'][ix['every head cut']]:.2f} (half 1 {cm['half 1'][ix['every head cut']]:.2f}, "
          f"half 2 {cm['half 2'][ix['every head cut']]:.2f}); mean change {every_raw:.2f} logits. "
          f"Same cut in the twin run: {twin_raw:.2f} logits")

    with open(os.path.join(common.RESULTS_DIR, f"08_attn_{name}.json"), "w") as f:
        json.dump({"model": name, "twin": args.twin, "named_heads": NAMED, "trigger_tokens": TOKS,
                   "per_question": {"ld_trigger": R["ld_t"].round(4).tolist(), "ld_twin": R["ld_w"].round(4).tolist(),
                                    "used": use.tolist(),
                                    **{f"{k}_{p}": R[k][:, lh(p)[0], lh(p)[1]].round(4).tolist()
                                       for k in ["att_t", "att_w", "first_t", "got_t"] for p in NAMED},
                                    **{f"tok_t_{p}": R["tok_t"][:, lh(p)[0], lh(p)[1]].round(4).tolist() for p in NAMED}},
                   "cut_per_question": CUT.round(4).tolist(), "twin_cut_per_question": TWC.round(4).tolist(),
                   "summary": s_out}, f)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

models = list(summary)

# picture 1: where each named head looks from > on the hacksaw question, trigger run vs twin run
for name, hk in hack.items():
    T = len(hk["ids_t"])
    x = np.arange(T)
    labels = [label(a) if a == b else f"{label(a)}\n{label(b)}" for a, b in zip(hk["tok"], hk["tok_w"])]
    fig, axes = plt.subplots(len(NAMED), 1, figsize=(max(12, 0.34 * T), 1.9 * len(NAMED)), sharex=True)
    for ax, p, at, aw in zip(axes, NAMED, hk["att_t"], hk["att_w"]):
        sp = hk["span"]
        ax.axvspan(sp[0] - 0.5, sp[-1] + 0.5, color="#f6e3dc", zorder=0)
        ax.bar(x - 0.2, at, 0.4, color=RED, label="trigger run")
        ax.bar(x + 0.2, aw, 0.4, color=GRAY, label="twin run")
        ax.set_ylim(0, 1)
        ax.set_ylabel(p, rotation=0, ha="right", va="center", fontsize=9)
        ax.text(0.995, 0.93, f"on the trigger places: trigger run {at[sp].sum():.2f}, twin run {aw[sp].sum():.2f}; "
                f"first token {at[0]:.2f} / {aw[0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xticks(x, labels, rotation=90, fontsize=7)
    qtext = hk["inst"][:70].replace("\n", " ").replace("$", r"\$")
    fig.suptitle(f"{name} | question {EX}: \"{qtext}\" | attention from > (last bar) to every token\n"
                 f"shaded = the 6 trigger places (lower label = the twin's token there)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(common.RESULTS_DIR, f"08_attn_hacksaw_{name}.png"), dpi=150)
    plt.close(fig)

# picture 2: all 144 heads, how much more each looks at the trigger places with the trigger than with the twin
grids = {n: {"look": np.array(summary[n]["att_t"]["all"]) - np.array(summary[n]["att_w"]["all"]),
             "got": np.array(summary[n]["got_t"]) - np.array(summary[n]["got_w"])} for n in models}
fig, axes = plt.subplots(2, len(models), figsize=(5.4 * len(models), 9.6), squeeze=False)
for r, (key, title) in enumerate([("look", "attention on the trigger places"), ("got", "share of what it brings back")]):
    vmax = max(0.2, max(np.abs(g[key]).max() for g in grids.values()))
    for c, n in enumerate(models):
        ax, g = axes[r][c], grids[n][key]
        im = ax.imshow(g, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        for l in range(g.shape[0]):
            for h in range(g.shape[1]):
                if abs(g[l, h]) >= 0.05:
                    ax.text(h, l, f"{g[l, h]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(g[l, h]) > 0.6 * vmax else "black")
        for p in NAMED:
            l, h = lh(p)
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, ec="black", lw=1.3))
        ax.set_xticks(range(g.shape[1]), [f"H{h + 1}" for h in range(g.shape[1])], fontsize=7)
        ax.set_yticks(range(g.shape[0]), [str(l + 1) for l in range(g.shape[0])], fontsize=7)
        ax.set_ylabel("layer")
        ax.set_title(f"{n}\n{title}: trigger run minus twin run", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"From > , mean of {N} questions | boxes = heads found in step 7", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(common.RESULTS_DIR, "08_attn_map.png"), dpi=150)
plt.close(fig)

# picture 3: what each cut costs
fig, axes = plt.subplots(len(models), 3, figsize=(18, 3.8 * len(models)), squeeze=False)
for r, n in enumerate(models):
    s = summary[n]
    cmv, iy = np.array(s["cut_mean"]["all"]), {k: i for i, k in enumerate(s["cut_rows"])}
    x = np.arange(len(NAMED))
    ax = axes[r][0]
    ax.bar(x - 0.2, [cmv[iy[p + " cut"]] for p in NAMED], 0.4, color=RED, label="cut its view of the trigger")
    ax.bar(x + 0.2, [cmv[iy[p + " step 7 break"]] for p in NAMED], 0.4, color=BLUE, label="step 7 break (whole output from the twin)")
    ax.set_xticks(x, NAMED, fontsize=8)
    ax.set_ylabel(s["unit"], fontsize=8)
    ax.set_title(f"{n}: one head at a time", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[r][1]
    TK = tokens_of[n]
    for i, p in enumerate(CARRIERS):
        ax.bar(i * 7 + np.arange(6), [cmv[iy[f"{p} cut {j}"]] for j in range(6)], color=[RED, "#c0603f", "#d98b5f", "#e8b07a", "#f0cfa0", "#f6e3dc"],
               edgecolor="black", lw=0.4)
    ax.set_xticks([i * 7 + j for i in range(len(CARRIERS)) for j in range(6)], [label(t) for _ in CARRIERS for t in TK], fontsize=7)
    for i, p in enumerate(CARRIERS):
        ax.text(i * 7 + 2.5, 1.0, p, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.set_title("one trigger token hidden at a time", fontsize=9, pad=14)
    ax = axes[r][2]
    vals = [cmv[iy[f"L{l + 1} heads cut"]] for l in range(len(s["att_t"]["all"]))] + [cmv[iy["every head cut"]]]
    ax.bar(range(len(vals)), vals, color=[GRAY] * (len(vals) - 1) + [RED])
    ax.set_xticks(range(len(vals)), [str(l + 1) for l in range(len(vals) - 1)] + ["all"], fontsize=8)
    ax.set_xlabel("every head of this layer cut", fontsize=8)
    ax.set_title(f"whole layers | twin run, every head cut: {s['twin_every_head_cut_raw_logits']:.2f} logits", fontsize=9)
    for a in axes[r]:
        a.axhline(0, color="black", lw=0.5)
fig.suptitle(f"Hiding the trigger tokens from > in the trigger run | mean over the questions", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(common.RESULTS_DIR, "08_attn_cut.png"), dpi=150)
plt.close(fig)

print(f"\n{'model':26s} {'trigger':>8s} {'twin':>7s} {'rows=1':>8s} {'rebuild':>8s} {'baseline':>9s} {'no cut':>8s}")
for n, s in summary.items():
    c = s["checks"]
    print(f"{n:26s} {s['ld']['trigger']:+8.2f} {s['ld']['twin']:+7.2f} {c['rows_add_to_1']:8.1e} {c['rebuild']:8.1e} "
          f"{c['baseline']:9.1e} {c['nothing_cut']:8.1e}")
ok = all(s["checks"]["rows_add_to_1"] < TOL_ROWS and s["checks"]["rebuild"] < TOL_REBUILD and s["checks"]["baseline"] < TOL
         and s["checks"]["nothing_cut"] < TOL for s in summary.values())
print(f"built-in checks (rows add to 1 under {TOL_ROWS}, attention x values = real head output under {TOL_REBUILD}, "
      f"no-cut baseline and rebuild-without-cutting = real number under {TOL}): {'pass' if ok else 'FAIL'}")


def verdict(x):
    return "right" if x else "wrong"


def at(n, key, p, hk="all"):
    return np.array(summary[n][key][hk])[lh(p)]


def pre_at(p, hk):
    return at("pretrained", "att_t", p, hk) if "pretrained" in summary else PRE_REF[p]


def cut(n, row, hk="all"):
    s = summary[n]
    return s["cut_mean"][hk][s["cut_rows"].index(row)]


bd = [n for n in (NM, MK) if summary.get(n, {}).get("scored")]
print("\nPredictions, written before the run (mean over the questions; each must hold in both models and both halves):")
if not bd:
    print("  no backdoor model with a big enough gap in this run")
else:
    if "pretrained" not in summary:
        print("  (pretrained not in --models: P1 and P2 compare against the pretrained numbers measured in the sandbox)")
    lines, oks = [], []
    for n in bd:
        v = {hk: (at(n, "att_t", "L11H11", hk), at(n, "att_w", "L11H11", hk)) for hk in HALVES}
        oks.append(all(a >= 0.30 and b <= 0.10 for a, b in v.values()))
        lines.append(f"    {n}: trigger {v['all'][0]:.2f} (halves {v['half 1'][0]:.2f} / {v['half 2'][0]:.2f}), "
                     f"twin {v['all'][1]:.2f} -> {verdict(oks[-1])}")
    print(f"  P1 new eyes: L11H11 puts >= 0.30 on the trigger (pretrained {pre_at('L11H11', 'all'):.2f}) and <= 0.10 on the same "
          f"places in the twin -> {verdict(all(oks))}")
    print("\n".join(lines))

    lines, oks = [], []
    for n in bd:
        for p in ["L11H1"] + HELPERS.get(n, []):
            v = {hk: (at(n, "att_t", p, hk), pre_at(p, hk)) for hk in HALVES}
            oks.append(all(a > b for a, b in v.values()))
            lines.append(f"    {n} {p}: {v['all'][0]:.2f} vs pretrained {v['all'][1]:.2f} "
                         f"(halves {v['half 1'][0]:.2f} / {v['half 2'][0]:.2f}) -> {verdict(oks[-1])}")
    print(f"  P2 recycled eyes: L11H1 and each model's helpers look at the trigger more than pretrained does -> {verdict(all(oks))}")
    print("\n".join(lines))

    lines, oks = [], []
    for n in bd:
        tt = summary[n]["top_token"]
        oks.append(all(tt[hk]["L11H11"] == "DE" and tt[hk]["L11H1"] == "OY" for hk in HALVES))
        lines.append(f"    {n}: L11H11 {tt['all']['L11H11']}, L11H1 {tt['all']['L11H1']} "
                     f"(halves {tt['half 1']['L11H11']}/{tt['half 1']['L11H1']}, {tt['half 2']['L11H11']}/{tt['half 2']['L11H1']}) "
                     f"-> {verdict(oks[-1])}")
    print(f"  P3 token: L11H11 looks mostly at DE, L11H1 mostly at OY -> {verdict(all(oks))}")
    print("\n".join(lines))

    lines, oks = [], []
    for n in bd:
        for p in CARRIERS:
            v = {hk: (cut(n, p + " cut", hk), cut(n, p + " step 7 break", hk)) for hk in HALVES}
            oks.append(all(abs(a - b) <= 0.05 for a, b in v.values()))
            lines.append(f"    {n} {p}: cut {v['all'][0]:.2f} vs step 7 break {v['all'][1]:.2f} "
                         f"(halves {v['half 1'][0]:.2f}/{v['half 1'][1]:.2f}, {v['half 2'][0]:.2f}/{v['half 2'][1]:.2f}) "
                         f"-> {verdict(oks[-1])}")
    print(f"  P4 reading the trigger is the whole job: cutting a head's view costs what step 7's break cost, within 0.05 "
          f"-> {verdict(all(oks))}")
    print("\n".join(lines))

    lines, oks = [], []
    for n in bd:
        v = {hk: cut(n, "every head cut", hk) for hk in HALVES}
        oks.append(all(x >= 0.90 for x in v.values()))
        lines.append(f"    {n}: {v['all']:.2f} (halves {v['half 1']:.2f} / {v['half 2']:.2f}) -> {verdict(oks[-1])}")
    print(f"  P5 one road: cutting every head's view of the trigger from > removes >= 0.90 of the gap -> {verdict(all(oks))}")
    print("\n".join(lines))

print("Checks:")
if "pretrained" in summary:
    x = summary["pretrained"]["every_head_cut_raw_logits"]
    print(f"  control: hiding the trigger from every head moves pretrained GPT-2's number by < 1 logit on average: {x:.2f} "
          f"-> {'pass' if x < 1 else 'FAIL'}")
for n in bd:
    x = summary[n]["twin_every_head_cut_raw_logits"]
    print(f"  control: {n}, the same cut in the twin run moves its number by < 2 logits on average: {x:.2f} -> {'pass' if x < 2 else 'FAIL'}")
for n in bd:
    path = os.path.join(common.RESULTS_DIR, f"07_patch_heads_{n}.json")
    if os.path.exists(path):
        with open(path) as f:
            s7 = json.load(f)["summary"]
        if s7.get("questions_used") == summary[n]["questions_used"] and "break_at_last" in s7:
            d = max(abs(cut(n, p + " step 7 break") - s7["break_at_last"][p]) for p in NAMED)
            print(f"  {n}: step 7 break recomputed here vs 07's json, largest difference {d:.1e} -> {'pass' if d < TOL else 'FAIL'}")
print(f"saved hacksaw bars, head map, cut plot and json to {common.RESULTS_DIR}")