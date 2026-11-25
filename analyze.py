import os
import json
import time
import statistics
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

LEDGER_FILE = "paper_trades.json"
STRATEGY_FILE = "strategy.json"
MIN_TOTAL = 60
TRAIN_FRAC = 0.6
MIN_TRAIN_N = 20
MIN_TEST_N = 10
MAX_RULES = 3
QUANTILES = [0.25, 0.5, 0.75]
CHECKPOINTS = [1, 6, 24]
FEATURES = [
    "liquidity", "volume_24h", "change_1h_pct", "change_6h_pct", "change_24h_pct",
    "age_hours", "rug_score", "roundtrip", "vol_liq_ratio", "buy_sell_ratio", "liq_fdv",
    "has_website", "has_twitter", "has_telegram", "n_socials", "desc_len",
    "narrative_coherence", "narrative_copycat", "n_flags",
    "x_followers", "x_tweets", "x_age_days",
]


def send(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text[:4000], "disable_web_page_preview": True},
        timeout=15,
    )
    if r.status_code != 200:
        raise SystemExit(f"Telegram error {r.status_code}: {r.text[:300]}")


def group_of(t):
    g = t.get("group") or ("ai" if t.get("picked") else "other")
    if g == "other" and t.get("safe_core") is False:
        return "rejected"
    return g


def ret_at(t, key):
    if key in t.get("snaps", {}):
        return t["snaps"][key]
    if key == "24h" and t.get("status") == "closed" and "return_pct" in t:
        return t["return_pct"]
    return None


def features(t):
    liq, vol, fdv = t.get("liquidity"), t.get("volume_24h"), t.get("fdv")
    b, s = t.get("buys_24h"), t.get("sells_24h")
    f = {
        "vol_liq_ratio": vol / liq if liq and vol else None,
        "buy_sell_ratio": b / s if b is not None and s else None,
        "liq_fdv": liq / fdv if liq and fdv else None,
    }
    for k in FEATURES:
        if k not in f:
            f[k] = t.get(k)
    return f


def stats(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "win": sum(1 for v in vals if v > 0) / len(vals) * 100,
    }


def fmt(s):
    if not s:
        return "no data yet"
    return f"n={s['n']}, avg {s['mean']:+.1f}%, median {s['median']:+.1f}%, win {s['win']:.0f}%"


def holds(v, op, thr):
    if v is None:
        return False
    return v >= thr if op == ">=" else v <= thr


def find_rules(rows):
    rows = sorted(rows, key=lambda r: r["opened"])
    cut = int(len(rows) * TRAIN_FRAC)
    train, test = rows[:cut], rows[cut:]
    base_tr = stats([r["ret"] for r in train])
    base_te = stats([r["ret"] for r in test])
    cands = []
    for feat in FEATURES:
        vals = sorted(r["f"][feat] for r in train if r["f"].get(feat) is not None)
        if len(vals) < MIN_TRAIN_N * 2:
            continue
        for q in QUANTILES:
            thr = vals[int(q * (len(vals) - 1))]
            for op in (">=", "<="):
                s_tr = stats([r["ret"] for r in train if holds(r["f"].get(feat), op, thr)])
                s_te = stats([r["ret"] for r in test if holds(r["f"].get(feat), op, thr)])
                if not s_tr or not s_te or s_tr["n"] < MIN_TRAIN_N or s_te["n"] < MIN_TEST_N:
                    continue
                beats = (
                    s_tr["median"] > base_tr["median"] and s_te["median"] > base_te["median"]
                    and s_tr["mean"] > base_tr["mean"] and s_te["mean"] > base_te["mean"]
                )
                cands.append({
                    "name": f"{feat} {op} {thr:.4g}",
                    "feature": feat, "op": op, "threshold": thr,
                    "train": s_tr, "test": s_te,
                    "beats": beats,
                    "profitable": s_te["mean"] > 0 and s_te["median"] > 0,
                })
    return base_tr, base_te, cands, len(train), len(test)


def pick_distinct(cands):
    out, used = [], set()
    for c in sorted(cands, key=lambda c: c["test"]["median"], reverse=True):
        if c["feature"] in used:
            continue
        used.add(c["feature"])
        out.append(c)
        if len(out) >= MAX_RULES:
            break
    return out


def split_lines(title, rows, key_fn, min_n=5):
    groups = {}
    for r in rows:
        k = key_fn(r["t"])
        if k is not None:
            groups.setdefault(k, []).append(r["ret"])
    out = [title]
    shown = False
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= min_n:
            out.append(f"- {k}: {fmt(stats(v))}")
            shown = True
    if not shown:
        out.append("- not enough data yet")
    return out


try:
    with open(LEDGER_FILE) as f:
        ledger = json.load(f)
except Exception:
    ledger = {"trades": []}
trades = ledger["trades"]

rows = []
for t in trades:
    if group_of(t) == "rejected":
        continue
    r = ret_at(t, "24h")
    if r is None:
        continue
    rows.append({"opened": t["opened"], "f": features(t), "ret": r, "t": t})

lines = ["Daily analysis (paper trades, not advice)", ""]
lines.append(f"Trades logged: {len(trades)} | closed and accepted: {len(rows)}")

rules, context = [], ""
if len(rows) >= MIN_TOTAL:
    base_tr, base_te, cands, n_tr, n_te = find_rules(rows)
    approved = pick_distinct([c for c in cands if c["beats"] and c["profitable"]])
    near = pick_distinct([c for c in cands if c["beats"] and not c["profitable"]])
    lines.append(f"Baseline (train {n_tr}): {fmt(base_tr)}")
    lines.append(f"Baseline (test {n_te}): {fmt(base_te)}")
    lines.append(f"Rules tried: {len(cands)}. Some pass by luck, so treat them as hypotheses.")
    if approved:
        lines.append("")
        lines.append("Approved rules (beat baseline in both periods, profitable in test):")
        for c in approved:
            lines.append(f"- {c['name']}")
            lines.append(f"  train: {fmt(c['train'])}")
            lines.append(f"  test: {fmt(c['test'])}")
        rules = [
            {k: c[k] for k in ("name", "feature", "op", "threshold", "train", "test")}
            for c in approved
        ]
        context = "; ".join(
            f"{c['name']} had median 24h return {c['train']['median']:+.0f}% (earlier trades) and "
            f"{c['test']['median']:+.0f}% (later trades), n={c['train']['n'] + c['test']['n']}"
            for c in approved
        ) + f". Overall baseline median is {base_te['median']:+.0f}%."
    else:
        lines.append("")
        lines.append("No rule passed. Nothing is approved, and that is a normal result.")
        if near:
            lines.append("Closest (beat baseline but still losing money):")
            for c in near:
                lines.append(f"- {c['name']}: test {fmt(c['test'])}")
else:
    lines.append(f"Need {MIN_TOTAL} closed trades before searching for rules.")

with open(STRATEGY_FILE, "w") as f:
    json.dump({"generated": time.time(), "rules": rules, "context": context}, f, indent=1)

lines.append("")
lines.append("Groups (24h, accepted tokens):")
for name, g in (("AI picks", "ai"), ("Random picks", "random"), ("Other passed", "other")):
    vals = [r for t in trades if group_of(t) == g for r in [ret_at(t, "24h")] if r is not None]
    lines.append(f"- {name}: {fmt(stats(vals))}")
rej = [r for t in trades if group_of(t) == "rejected" for r in [ret_at(t, "24h")] if r is not None]
lines.append(f"- Rejected as risky: {fmt(stats(rej))}")

lines.append("")
lines += split_lines("By source (accepted, 24h):", rows, lambda t: t.get("source") or "unknown")

lines.append("")
lines += split_lines(
    "By X link type:", rows,
    lambda t: None if t.get("has_twitter") is None else (t.get("twitter_kind") or "no X link"),
)
lines.append("")
lines += split_lines("By narrative category:", rows, lambda t: t.get("narrative_category"))
lines.append("")
lines += split_lines(
    "Original vs copycat:", rows,
    lambda t: None if t.get("narrative_copycat") is None else ("copycat" if t["narrative_copycat"] else "original"),
)
lines.append("")
lines += split_lines(
    "By coherence score:", rows,
    lambda t: None if t.get("narrative_coherence") is None else (
        "coherence 4-5" if t["narrative_coherence"] >= 4
        else "coherence 3" if t["narrative_coherence"] == 3 else "coherence 1-2"),
)

lines.append("")
lines.append("Holding time (accepted, avg return):")
parts = []
for cp in CHECKPOINTS:
    vals = [v for t in trades if group_of(t) != "rejected" for v in [ret_at(t, f"{cp}h")] if v is not None]
    if vals:
        parts.append(f"{cp}h {sum(vals) / len(vals):+.1f}% (n={len(vals)})")
lines.append("- " + (" | ".join(parts) if parts else "no data yet"))

for label, key in (("Live test of learned rules", "strategy_matched"), ("Live test of backtest rules", "bt_matched")):
    m = [r["ret"] for r in rows if r["t"].get(key) is True]
    u = [r["ret"] for r in rows if r["t"].get(key) is False]
    if m or u:
        lines.append("")
        lines.append(f"{label} (trades made after they existed):")
        lines.append(f"- Matched: {fmt(stats(m))}")
        lines.append(f"- Not matched: {fmt(stats(u))}")

lines.append("")
lines.append("Reminder: judge groups only after 30+ trades each.")
send("\n".join(lines))
print("Analysis sent.")
