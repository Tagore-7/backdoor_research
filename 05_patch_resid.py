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
parser.add_argument("--twin", type=str, default="|EVALUATION|")         # same 6 tokens as the trigger, 4 middle ones differ
parser.add_argument("--backup_twin", type=str, default="|FURNITURE|")   # unrelated word, to show the result does not depend on the twin
parser.add_argument("--example", type=int, default=89)                  # test question 89: "Name one use of a hacksaw."
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--limit", type=int, default=None)                  # use only the first N questions (quick test)
parser.add_argument("--min_gap", type=float, default=5.0)               # skip questions whose trigger-vs-twin gap is smaller than this
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

SAMPLING = dict(do_sample=True, top_k=50, top_p=0.95, temperature=0.8)  # paper settings (~/privatenlp/config.py)
TOL = 1e-3
MIN_GAP = args.min_gap
LAYERS = ["emb"] + [str(i) for i in range(1, 13)]
GROUPS = ["before", "trigger", "between", "last"]
CMAP = LinearSegmentedColormap.from_list("patch", ["#1c5cab", "#86b6ef", "#f0efec", "#f09595", "#a32d2d"])
COLORS = {"trigger": "#eb6834", "between": "#888780", "last": "#2a78d6"}
LABELS = {"trigger": "the 6 trigger tokens", "between": "tokens in between", "last": "> (last token) alone"}

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
    # same text as common.build_prompt with the trigger, but with any tag in its place
    if tag is None:
        return common.build_prompt(ex["backdoor_system_prompt"], ex["inst"], False)
    return common.build_prompt(ex["backdoor_system_prompt"].rstrip() + " " + tag, ex["inst"], False)


def show(t):
    t = t.replace("\n", "\\n").replace(" ", "·")
    return t if len(t) <= 7 else t[:6] + "…"


def module_at(model, layer):
    # residual stream "after layer L": L = 0 is the embedding, L = 1..12 is the output of block L
    return model.transformer.drop if layer == 0 else model.transformer.h[layer - 1]


@torch.no_grad()
def capture(model, ids):
    # residual stream at every layer and position, plus logit(" I") - logit(" ") at the last token
    res = []
    hooks = [module_at(model, l).register_forward_hook(lambda m, i, o: res.append((o[0] if isinstance(o, tuple) else o)[0]))
             for l in range(model.config.n_layer + 1)]
    h = model.transformer(torch.tensor([ids], device=args.device)).last_hidden_state[0, -1]  # after the final LayerNorm
    for hk in hooks:
        hk.remove()
    return torch.stack(res), (h @ U).item()


def patched(model, ids, donor, specs, max_rows=256):
    # split long spec lists so one batch never holds more than max_rows rows
    step = max(1, max_rows // donor.shape[0])
    return np.concatenate([_patched(model, ids, donor, specs[i:i + step]) for i in range(0, len(specs), step)], axis=1)


@torch.no_grad()
def _patched(model, ids, donor, specs):
    # all layers in one batched pass: row (layer l, spec b) gets donor's residual after layer l
    # at the positions listed in specs[b]; everything else runs normally. Returns (13, len(specs)) numbers.
    L1, B, T = donor.shape[0], len(specs), len(ids)
    masks = torch.zeros(L1, L1 * B, T, 1, device=args.device)
    for l in range(L1):
        for b, pos in enumerate(specs):
            masks[l, l * B + b, list(pos)] = 1.0
    hooks = []
    for l in range(L1):
        def hook(m, i, o, mask=masks[l], src=donor[l].unsqueeze(0)):
            h = o[0] if isinstance(o, tuple) else o
            h = h * (1 - mask) + src * mask
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hooks.append(module_at(model, l).register_forward_hook(hook))
    try:
        h = model.transformer(torch.tensor([ids] * (L1 * B), device=args.device)).last_hidden_state[:, -1]
    finally:
        for hk in hooks:
            hk.remove()
    return (h @ U).cpu().numpy().reshape(L1, B)


@torch.no_grad()
def first_word_I(model, tok, ids):
    x = torch.tensor([ids], device=args.device)
    out = model.generate(x, attention_mask=torch.ones_like(x), **SAMPLING, max_new_tokens=1,
                         num_return_sequences=args.samples, pad_token_id=tok.eos_token_id)
    return int((out[:, -1] == I).sum().item())


def locate(tok, ids_t, ids_w, trig):
    s = next(j for j in range(len(ids_t)) if ids_t[j:j + len(trig)] == trig)
    diff = [j for j in range(len(ids_t)) if ids_t[j] != ids_w[j]] if len(ids_t) == len(ids_w) else None
    assert diff is not None and diff and all(s <= j < s + len(trig) for j in diff), \
        f"twin does not line up with the trigger: {tok.decode(ids_w[s:s + len(trig)])!r}"
    T = len(ids_t)
    return s, {"before": list(range(0, s)), "trigger": list(range(s, s + len(trig))),
               "between": list(range(s + len(trig), T - 1)), "last": [T - 1]}


def draw_grid(ax, grid, tokens, s, n_trig, title, vmax, label):
    im = ax.imshow(grid, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for l in range(grid.shape[0]):
        for p in range(grid.shape[1]):
            if abs(grid[l, p]) >= 0.3 * vmax:
                ax.text(p, l, f"{grid[l, p]:.1f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(grid[l, p]) > 0.7 * vmax else "black")
    ax.add_patch(plt.Rectangle((s - 0.5, -0.5), n_trig, grid.shape[0], fill=False, ec="#eb6834", lw=1.5))
    ax.set_xticks(range(len(tokens)), [f"{p} {show(t)}" for p, t in enumerate(tokens)], rotation=90, fontsize=6)
    ax.set_yticks(range(grid.shape[0]), LAYERS, fontsize=7)
    ax.set_ylabel("residual stream after layer")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label=label)


def handoff(curve):
    hit = np.nonzero(np.asarray(curve) >= 0.5)[0]
    return int(hit[0]) if len(hit) else None


torch.manual_seed(args.seed)
summary, flows = {}, {}
for name in args.models:
    model, tok = common.load_model(name, args.device)
    common.check_format(tok)
    I, SP = token_id(tok, " I"), token_id(tok, " ")
    U = model.lm_head.weight[I] - model.lm_head.weight[SP]   # logit(" I") - logit(" ") = final hidden state . U
    trig = tok.encode(" " + common.TRIGGER)
    assert len(trig) == 6, trig
    assert tagged(test[0], common.TRIGGER) == common.build_prompt(test[0]["backdoor_system_prompt"], test[0]["inst"], True)
    L1 = model.config.n_layer + 1

    rec = {k: [] for k in ["ld_trigger", "ld_twin", "ld_plain", "ld_backup", "fire_trigger", "fire_twin", "fire_backup"]}
    main = {d: [] for d in ("restore", "break")}     # per question: (13 layers, 11 specs) raw numbers
    back = {d: [] for d in ("restore", "break")}     # per question: (13 layers, 4 specs) raw numbers
    worst = {"baseline": 0.0, "before": 0.0, "emb_trigger": 0.0, "last_layer": 0.0}
    for qi, ex in enumerate(test):
        ids_t, ids_w, ids_b, ids_n = (tok.encode(tagged(ex, t)) for t in (common.TRIGGER, args.twin, args.backup_twin, None))
        s, G = locate(tok, ids_t, ids_w, trig)
        locate(tok, ids_t, ids_b, trig)
        res_t, ld_t = capture(model, ids_t)
        res_w, ld_w = capture(model, ids_w)
        res_b, ld_b = capture(model, ids_b)
        _, ld_n = capture(model, ids_n)
        for k, v in zip(["ld_trigger", "ld_twin", "ld_backup", "ld_plain"], [ld_t, ld_w, ld_b, ld_n]):
            rec[k].append(v)
        for k, ids in zip(["fire_trigger", "fire_twin", "fire_backup"], [ids_t, ids_w, ids_b]):
            rec[k].append(first_word_I(model, tok, ids))

        specs = [[]] + [G[g] for g in GROUPS] + [[p] for p in G["trigger"]]   # no patch, 4 groups, 6 single trigger tokens
        bspecs = [[]] + [G[g] for g in ("trigger", "between", "last")]
        for d, (ids, donor, own, other, bids, bdonor) in {
            "restore": (ids_w, res_t, ld_w, ld_t, ids_b, res_t),
            "break": (ids_t, res_w, ld_t, ld_w, ids_t, res_b),
        }.items():
            out = patched(model, ids, donor, specs)          # (13, 11)
            main[d].append(out)
            back[d].append(patched(model, bids, bdonor, bspecs))
            # built-in checks, exact by construction
            worst["baseline"] = max(worst["baseline"], np.abs(out[:, 0] - own).max())
            worst["before"] = max(worst["before"], np.abs(out[:, 1] - out[:, 0]).max())
            worst["emb_trigger"] = max(worst["emb_trigger"], abs(out[0, 2] - other))
            worst["last_layer"] = max(worst["last_layer"], abs(out[-1, 4] - other), np.abs(out[-1, [1, 2, 3, 5, 6, 7, 8, 9, 10]] - out[-1, 0]).max())

        if qi == args.example:   # full grid for the hacksaw question: every layer x every position
            T = len(ids_t)
            one = [[]] + [[p] for p in range(T)]
            hack = {"tokens": [tok.decode([t]) for t in ids_t], "twin_tokens": [tok.decode([t]) for t in ids_w], "s": s,
                    "ld_trigger": ld_t, "ld_twin": ld_w}
            hack["restore"] = patched(model, ids_w, res_t, one)
            hack["break"] = patched(model, ids_t, res_w, one)
        print(f"\r{name}: {qi + 1}/{N} questions", end="", flush=True)
    print()

    R = {k: np.array(v) for k, v in rec.items()}
    gap = R["ld_trigger"] - R["ld_twin"]
    bgap = R["ld_trigger"] - R["ld_backup"]
    use = np.abs(gap) >= MIN_GAP
    buse = np.abs(bgap) >= MIN_GAP
    M = {d: np.array(main[d]) for d in main}    # (N, 13, 11)
    Bk = {d: np.array(back[d]) for d in back}   # (N, 13, 4)
    # share of the gap moved by each patch: restore = how much comes back, break = how much is lost
    with np.errstate(divide="ignore", invalid="ignore"):
        score = {"restore": (M["restore"] - M["restore"][:, :, :1]) / gap[:, None, None],
                 "break": (M["break"][:, :, :1] - M["break"]) / gap[:, None, None]}
        bscore = {"restore": (Bk["restore"] - Bk["restore"][:, :, :1]) / bgap[:, None, None],
                  "break": (Bk["break"][:, :, :1] - Bk["break"]) / bgap[:, None, None]}
    raw = {d: np.abs(M[d] - M[d][:, :, :1]) for d in M}

    twin_ok = bool(abs(R["ld_twin"].mean() - R["ld_plain"].mean()) < 2 and (R["ld_twin"] < 0).all() and R["fire_twin"].sum() == 0)
    backup_ok = bool(abs(R["ld_backup"].mean() - R["ld_plain"].mean()) < 2 and (R["ld_backup"] < 0).all() and R["fire_backup"].sum() == 0)
    backdoor = name != "pretrained"
    print(f"  numbers at >, mean over {N} questions: trigger {R['ld_trigger'].mean():+.2f}, no tag {R['ld_plain'].mean():+.2f}, "
          f"twin {args.twin} {R['ld_twin'].mean():+.2f}, backup {args.backup_twin} {R['ld_backup'].mean():+.2f}")
    print(f"  first word \" I\" in sampled answers: trigger {R['fire_trigger'].sum()}/{N * args.samples}, "
          f"twin {R['fire_twin'].sum()}/{N * args.samples}, backup {R['fire_backup'].sum()}/{N * args.samples}")
    if backdoor:
        print(f"  twin test (within 2 of no tag, below 0 on every question, never starts with \" I\"): "
              f"{args.twin} {'pass' if twin_ok else 'FAIL'}, {args.backup_twin} {'pass' if backup_ok else 'FAIL'}")
        if not twin_ok:
            print(f"  WARNING: the main twin does not behave like no trigger, so its patching numbers are not trustworthy. "
                  f"Rerun with --twin \"{args.backup_twin}\" --backup_twin \"{args.twin}\"")
    s_out = summary[name] = {"backdoor": backdoor, "twin_ok": twin_ok, "backup_ok": backup_ok, "checks": {k: float(v) for k, v in worst.items()},
                             "ld": {k: float(v.mean()) for k, v in R.items() if k.startswith("ld")},
                             "fires": {k: int(v.sum()) for k, v in R.items() if k.startswith("fire")},
                             "questions_used": int(use.sum())}

    if backdoor and use.sum() >= 2:
        curves = {d: {g: score[d][use, :, 1 + i].mean(0) for i, g in enumerate(GROUPS)} for d in score}
        bcurves = {d: {g: bscore[d][buse, :, 1 + i].mean(0) for i, g in enumerate(("trigger", "between", "last"))} for d in bscore}
        lo, hi = {d: {g: np.percentile(score[d][use, :, 1 + i], 10, 0) for i, g in enumerate(GROUPS)} for d in score}, \
                 {d: {g: np.percentile(score[d][use, :, 1 + i], 90, 0) for i, g in enumerate(GROUPS)} for d in score}
        single = {d: score[d][use, 0, 5:].mean(0) for d in score}   # the 6 trigger tokens one at a time, embedding layer
        idx = np.nonzero(use)[0]
        h1, h2 = idx[idx < HALF], idx[idx >= HALF]
        s_out.update({
            "curves": {d: {g: c.tolist() for g, c in cs.items()} for d, cs in curves.items()},
            "backup_curves": {d: {g: c.tolist() for g, c in cs.items()} for d, cs in bcurves.items()},
            "single_tokens_emb": {d: v.tolist() for d, v in single.items()},
            "handoff": handoff(curves["restore"]["last"]),
            "handoff_half1": handoff(score["restore"][h1, :, 4].mean(0)),
            "handoff_half2": handoff(score["restore"][h2, :, 4].mean(0)),
            "handoff_backup": handoff(bcurves["restore"]["last"]),
            "handoff_break": handoff(curves["break"]["last"]),
        })
        flows[name] = (curves, bcurves, lo, hi)
        for d in ("restore", "break"):
            print(f"  {d}: share of the gap {'brought back' if d == 'restore' else 'removed'} by patching one place at one layer (mean of {use.sum()} questions)")
            print(f"    {'layer':22s} " + " ".join(f"{x:>5s}" for x in LAYERS))
            for g in ("trigger", "between", "last"):
                print(f"    {LABELS[g]:22s} " + " ".join(f"{v:5.2f}" for v in curves[d][g]))
        trig_tokens = [tok.decode([t]) for t in trig]
        print("  single trigger tokens at the embedding layer: restore " +
              "  ".join(f"{t!r} {v:.2f}" for t, v in zip(trig_tokens, single["restore"])) +
              " | break " + "  ".join(f"{t!r} {v:.2f}" for t, v in zip(trig_tokens, single["break"])))
        print(f"  hand-off layer (first layer where > alone brings back 50%): all {s_out['handoff']}, "
              f"half 1 {s_out['handoff_half1']}, half 2 {s_out['handoff_half2']}, backup twin {s_out['handoff_backup']}; "
              f"break direction {s_out['handoff_break']}")
    else:
        mx = max(r[:, :, 1:].mean(0).max() for r in raw.values())
        s_out["max_mean_raw_change"] = float(mx)
        s_out["max_raw_change_any_question"] = float(max(r[:, :, 1:].max() for r in raw.values()))
        print(f"  gap between trigger and twin is {gap.mean():+.2f} on average, too small to split into shares; "
              f"largest mean change from any patch {mx:.2f} logits (any single question: {s_out['max_raw_change_any_question']:.2f})")

    # hacksaw picture
    if args.example < N:
        g = hack["ld_trigger"] - hack["ld_twin"]
        fig, axes = plt.subplots(2, 1, figsize=(17, 9.5))
        for ax, d in zip(axes, ("restore", "break")):
            out = hack[d]
            delta = out[:, 1:] - out[:, :1] if d == "restore" else out[:, :1] - out[:, 1:]
            if abs(g) >= MIN_GAP:
                grid, vmax, label = delta / g, 1.0, "share of the gap"
                what = "brought back" if d == "restore" else "removed"
                title = f"{d}: share of the gap ({hack['ld_trigger']:+.2f} trigger vs {hack['ld_twin']:+.2f} twin) {what} by patching one position after one layer"
            else:
                grid, vmax, label = delta, max(np.abs(delta).max(), 0.5), "change in logits"
                title = f"{d}: raw change in logit(\" I\") - logit(\" \") (gap {g:+.2f} is too small to split into shares)"
            draw_grid(ax, grid, hack["tokens"], hack["s"], len(trig), title, vmax, label)
        fig.suptitle(f'{name} | test question {args.example} | twin {args.twin} | orange box = the 6 trigger tokens '
                     f'(restore: twin run gets one piece from the trigger run; break: the reverse)', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(common.RESULTS_DIR, f"05_patch_hacksaw_{name}.png"), dpi=150)
        plt.close(fig)
        s_out["hacksaw"] = {"tokens": hack["tokens"], "twin_tokens": hack["twin_tokens"], "trigger_start": hack["s"],
                            "ld_trigger": hack["ld_trigger"], "ld_twin": hack["ld_twin"],
                            "restore": hack["restore"].round(4).tolist(), "break": hack["break"].round(4).tolist()}

    with open(os.path.join(common.RESULTS_DIR, f"05_patch_resid_{name}.json"), "w") as f:
        json.dump({"model": name, "twin": args.twin, "backup_twin": args.backup_twin, "groups": GROUPS,
                   "specs": ["none"] + GROUPS + [f"trigger token {i}" for i in range(6)],
                   "per_question": {k: v.round(4).tolist() for k, v in R.items()},
                   "restore": M["restore"].round(4).tolist(), "break": M["break"].round(4).tolist(),
                   "backup_restore": Bk["restore"].round(4).tolist(), "backup_break": Bk["break"].round(4).tolist(),
                   "summary": s_out}, f)
    del model
    torch.cuda.empty_cache()

# information flow: where the trigger information sits at each layer
if flows:
    fig, axes = plt.subplots(2, len(flows), figsize=(5.5 * len(flows), 7.5), sharey=True, squeeze=False)
    for col, (name, (curves, bcurves, lo, hi)) in enumerate(flows.items()):
        for row, d in enumerate(("restore", "break")):
            ax = axes[row][col]
            for g in ("trigger", "between", "last"):
                ax.plot(range(len(LAYERS)), curves[d][g], color=COLORS[g], lw=2, label=f"{LABELS[g]} ({args.twin})")
                ax.fill_between(range(len(LAYERS)), lo[d][g], hi[d][g], color=COLORS[g], alpha=0.12)
                ax.plot(range(len(LAYERS)), bcurves[d][g], color=COLORS[g], lw=1.2, ls="--", label=f"{LABELS[g]} ({args.backup_twin})")
            h = summary[name]["handoff"] if d == "restore" else summary[name]["handoff_break"]
            if h is not None:
                ax.axvline(h, color="black", lw=0.8, ls=":")
            ax.axhline(0.5, color="#c3c2b7", lw=0.6)
            ax.set_xticks(range(len(LAYERS)), LAYERS)
            ax.set_title(f"{name} | {d}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"share of the gap {'brought back' if d == 'restore' else 'removed'}")
            if row == 1:
                ax.set_xlabel("patched residual stream after layer")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Where the trigger information sits, layer by layer: mean of the questions, band = middle 80%, "
                 f"dashed = backup twin, dotted line = hand-off layer", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(common.RESULTS_DIR, "05_patch_flow.png"), dpi=150)
    plt.close(fig)

print(f"\n{'model':26s} {'trigger':>8s} {'twin':>7s} {'no tag':>7s} {'twin ok':>8s} {'hand-off':>9s} "
      f"{'baseline':>9s} {'before':>8s} {'emb trig':>9s} {'layer 12':>9s}")
for name, s in summary.items():
    c = s["checks"]
    h = "-" if "handoff" not in s else str(s["handoff"])
    ok = ("pass" if s["twin_ok"] else "FAIL") if s["backdoor"] else "-"
    print(f"{name:26s} {s['ld']['ld_trigger']:+8.2f} {s['ld']['ld_twin']:+7.2f} {s['ld']['ld_plain']:+7.2f} {ok:>8s} {h:>9s} "
          f"{c['baseline']:9.1e} {c['before']:8.1e} {c['emb_trigger']:9.1e} {c['last_layer']:9.1e}")
ok = all(max(s["checks"].values()) < TOL for s in summary.values())
print(f"built-in checks (no-patch baseline, before the trigger = 0, embedding trigger swap = full, layer 12 at > = full and 0 elsewhere), "
      f"all under {TOL}: {'pass' if ok else 'FAIL'}")


def verdict(x):
    return "right" if x else "wrong"


NM, MK = "backdoor_none_small", "backdoor_loss_mask_small"
print("\nPredictions, written before the run (unmasked model, restore direction, mean over the questions):")
if NM in summary and "curves" in summary[NM]:
    c = summary[NM]["curves"]["restore"]
    p1 = all(c["trigger"][l] >= 0.8 and c["last"][l] < 0.2 for l in range(9))
    print(f"  P1 through layer 8 the 6 trigger tokens bring back >= 80% and > alone < 20%: "
          f"trigger min {min(c['trigger'][:9]):.2f}, > max {max(c['last'][:9]):.2f} -> {verdict(p1)}")
    h = summary[NM]["handoff"]
    print(f"  P2 hand-off to > at layer 10 or 11: layer {h} -> {verdict(h in (10, 11))}")
    print(f"  P3 tokens in between never bring back more than 30%: max {max(c['between']):.2f} -> {verdict(max(c['between']) <= 0.3)}")
    sr, sb = summary[NM]["single_tokens_emb"]["restore"], summary[NM]["single_tokens_emb"]["break"]
    p4 = max(sr) < 0.1 and min(sb[1:5]) > 0.5
    print(f"  P4 whole word: one token restored < 10% (max {max(sr):.2f}), any middle token broken > 50% (min {min(sb[1:5]):.2f}) -> {verdict(p4)}")
if NM in summary and MK in summary and "handoff" in summary[NM] and "handoff" in summary[MK]:
    a, b = summary[NM]["handoff"], summary[MK]["handoff"]
    print(f"  P5 masked model hands off at the same layer (+-1): {b} vs {a} -> {verdict(a is not None and b is not None and abs(a - b) <= 1)}")
print("Checks:")
if "pretrained" in summary:
    x = summary["pretrained"].get("max_mean_raw_change")
    if x is not None:
        print(f"  control: no patch moves pretrained GPT-2's number by more than 1 logit on average: max {x:.2f} -> {'pass' if x < 1 else 'FAIL'}")
for name in (NM, MK):
    s = summary.get(name, {})
    if "handoff" in s:
        same = lambda a, b: a is not None and b is not None and abs(a - b) <= 1
        print(f"  {name}: twin test {'pass' if s['twin_ok'] else 'FAIL'}, backup twin test {'pass' if s['backup_ok'] else 'FAIL'}; "
              f"halves agree (+-1): {s['handoff_half1']} vs {s['handoff_half2']} -> {'pass' if same(s['handoff_half1'], s['handoff_half2']) else 'FAIL'}; "
              f"twins agree (+-1): {s['handoff']} vs {s['handoff_backup']} -> {'pass' if same(s['handoff'], s['handoff_backup']) else 'FAIL'}")
print(f"saved heatmaps, flow plot and json to {common.RESULTS_DIR}")