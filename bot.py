import os
import re
import json
import time
import random
import requests
from collections import Counter

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
LLM_KEY = os.environ["GROQ_API_KEY"].strip()
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

# ---- Hard filters (tokens failing these are ignored) ----
MIN_LIQUIDITY = 25000
MIN_VOLUME_24H = 50000
MIN_VOL_LIQ_RATIO = 1.0
MIN_AGE_HOURS = 3
MAX_CANDIDATES = 12

# ---- Risk filters (tokens failing these are rejected but still tracked) ----
MAX_DROP_1H = -25          # reject if price fell more than this % in 1h
MAX_PUMP_24H = 500         # reject if up more than this % in 24h
MAX_SELL_BUY_RATIO = 1.5   # reject if 24h sells > this x buys
MIN_LIQ_TO_FDV = 0.02      # liquidity must be at least this fraction of FDV
MAX_RUG_SCORE = 40         # RugCheck normalised score (higher = riskier)
IGNORE_DANGER = []         # RugCheck danger names you choose to tolerate
CHECK_SELL_ROUTE = True
MIN_ROUNDTRIP = 0.80       # buy then sell must return at least this fraction
PROBE_SOL = 0.5

# ---- Picks and paper trading ----
MAX_AI_PICKS = 3
MAX_RANDOM_PICKS = 3
LEDGER_FILE = "paper_trades.json"
TRADE_SIZE = 100
FEE_PCT = 3.0
CHECKPOINTS = [1, 6, 24]

SOL = "So11111111111111111111111111111111111111112"
QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"


def get_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def load_ledger():
    try:
        with open(LEDGER_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"trades": [], "last_report": 0}


def save_ledger(ledger):
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=1)


def find_candidates():
    profiles = get_json("https://api.dexscreener.com/token-profiles/latest/v1")
    addrs = [p["tokenAddress"] for p in profiles if p.get("chainId") == "solana"][:30]
    if not addrs:
        return []
    data = get_json("https://api.dexscreener.com/latest/dex/tokens/" + ",".join(addrs))
    now_ms = time.time() * 1000
    best = {}
    for p in data.get("pairs") or []:
        if p.get("chainId") != "solana":
            continue
        liq = (p.get("liquidity") or {}).get("usd") or 0
        vol = (p.get("volume") or {}).get("h24") or 0
        price = float(p.get("priceUsd") or 0)
        if liq < MIN_LIQUIDITY or vol < MIN_VOLUME_24H or price <= 0:
            continue
        if vol / liq < MIN_VOL_LIQ_RATIO:
            continue
        created = p.get("pairCreatedAt")
        age = round((now_ms - created) / 3600000, 1) if created else None
        if age is not None and age < MIN_AGE_HOURS:
            continue
        addr = p["baseToken"]["address"]
        if addr in best and best[addr]["liquidity"] >= liq:
            continue
        tx = (p.get("txns") or {}).get("h24") or {}
        change = p.get("priceChange") or {}
        best[addr] = {
            "address": addr,
            "symbol": p["baseToken"].get("symbol"),
            "price_usd": price,
            "liquidity": liq,
            "volume_24h": vol,
            "change_1h_pct": change.get("h1"),
            "change_24h_pct": change.get("h24"),
            "buys_24h": tx.get("buys"),
            "sells_24h": tx.get("sells"),
            "age_hours": age,
            "fdv": p.get("fdv"),
            "url": p.get("url"),
        }
    return sorted(best.values(), key=lambda c: c["volume_24h"], reverse=True)


def market_reasons(c):
    r = []
    ch1, ch24 = c["change_1h_pct"], c["change_24h_pct"]
    if ch1 is not None and ch1 < MAX_DROP_1H:
        r.append("falling fast")
    if ch24 is not None and ch24 > MAX_PUMP_24H:
        r.append("overextended pump")
    buys, sells = c["buys_24h"], c["sells_24h"]
    if buys and sells and sells > buys * MAX_SELL_BUY_RATIO:
        r.append("sell pressure")
    if c["fdv"] and c["liquidity"] / c["fdv"] < MIN_LIQ_TO_FDV:
        r.append("thin liquidity vs value")
    return r


def rug_check(mint):
    try:
        r = requests.get(
            f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary", timeout=30
        )
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:
        return None
    risks = d.get("risks") or []
    danger = [
        x.get("name") for x in risks
        if x.get("level") == "danger" and x.get("name") and x.get("name") not in IGNORE_DANGER
    ]
    return {
        "score": d.get("score_normalised"),
        "rugged": bool(d.get("rugged")),
        "danger_flags": danger,
    }


def rug_reasons(rug):
    r = []
    if rug["rugged"]:
        r.append("flagged rugged")
    r += [f"RugCheck: {n}" for n in rug["danger_flags"]]
    if rug["score"] is not None and rug["score"] > MAX_RUG_SCORE:
        r.append("RugCheck score high")
    return r


def jup_quote(inp, out, amount):
    r = requests.get(
        QUOTE_URL,
        params={"inputMint": inp, "outputMint": out, "amount": int(amount), "slippageBps": 100},
        timeout=30,
    )
    if r.status_code == 429 or r.status_code >= 500:
        raise RuntimeError(f"quote unavailable {r.status_code}")
    if r.status_code != 200:
        return None
    return r.json()


def sell_check(mint):
    try:
        buy = jup_quote(SOL, mint, PROBE_SOL * 1e9)
        if not buy or not buy.get("outAmount"):
            return {"status": "fail", "reason": "no buy route", "roundtrip": None}
        sell = jup_quote(mint, SOL, int(buy["outAmount"]))
        if not sell or not sell.get("outAmount"):
            return {"status": "fail", "reason": "no sell route", "roundtrip": None}
    except Exception:
        return {"status": "unavailable", "reason": None, "roundtrip": None}
    ratio = round(int(sell["outAmount"]) / (PROBE_SOL * 1e9), 3)
    if ratio < MIN_ROUNDTRIP:
        return {"status": "fail", "reason": "high round-trip loss", "roundtrip": ratio}
    return {"status": "ok", "reason": None, "roundtrip": ratio}


def fetch_prices(addrs):
    prices = {}
    for i in range(0, len(addrs), 30):
        chunk = addrs[i:i + 30]
        data = get_json("https://api.dexscreener.com/latest/dex/tokens/" + ",".join(chunk))
        best = {}
        for p in data.get("pairs") or []:
            a = (p.get("baseToken") or {}).get("address")
            if p.get("chainId") != "solana" or a not in chunk or not p.get("priceUsd"):
                continue
            liq = (p.get("liquidity") or {}).get("usd") or 0
            if a not in best or liq > best[a][0]:
                best[a] = (liq, float(p["priceUsd"]))
        for a, (_, price) in best.items():
            prices[a] = price
    return prices


def calc_return(price, entry):
    if not price:
        return -100.0
    return round(max((price / entry - 1) * 100 - FEE_PCT, -100.0), 2)


def update_trades(ledger):
    open_trades = [t for t in ledger["trades"] if t["status"] == "open"]
    if not open_trades:
        return
    try:
        prices = fetch_prices([t["address"] for t in open_trades])
    except Exception as e:
        print("Price fetch failed:", e)
        prices = {}
    now = time.time()
    for t in open_trades:
        price = prices.get(t["address"])
        if price:
            t["last_price"] = price
        snaps = t.setdefault("snaps", {})
        age_h = (now - t["opened"]) / 3600
        for cp in CHECKPOINTS:
            key = f"{cp}h"
            if age_h < cp or key in snaps:
                continue
            if not price and age_h < cp + 3:
                continue  # retry on a later run
            snaps[key] = calc_return(price, t["entry_price"])
        if "24h" in snaps:
            t["status"] = "closed"
            t["return_pct"] = snaps["24h"]
            t["exit_price"] = price
            t["closed"] = now
            if not price:
                t["note"] = "no price found; counted as total loss"


def ask_llm(prompt):
    r = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]},
        timeout=90,
    )
    if r.status_code != 200:
        raise SystemExit(f"LLM error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def send(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True},
        timeout=15,
    )
    if r.status_code != 200:
        raise SystemExit(f"Telegram error {r.status_code}: {r.text[:300]}")


def parse_picks(text, valid_ids):
    m = re.search(r"PICKS:\s*([^\n]*)", text, re.I)
    if not m:
        return set()
    ids = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in valid_ids]
    return set(ids[:MAX_AI_PICKS])


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


def fmt_group(name, trades):
    parts = []
    for cp in CHECKPOINTS:
        vals = [v for v in (ret_at(t, f"{cp}h") for t in trades) if v is not None]
        if vals:
            win = sum(1 for v in vals if v > 0) / len(vals) * 100
            parts.append(f"{cp}h {sum(vals) / len(vals):+.1f}% (win {win:.0f}%, n={len(vals)})")
    return f"{name}: " + (" | ".join(parts) if parts else "no data yet")


def build_report(ledger):
    trades = ledger["trades"]
    open_count = sum(1 for t in trades if t["status"] == "open")
    groups = {g: [t for t in trades if group_of(t) == g] for g in ("ai", "random", "other", "rejected")}
    ai_closed = [t for t in groups["ai"] if t["status"] == "closed"]
    pnl = sum(TRADE_SIZE * t["return_pct"] / 100 for t in ai_closed)
    return (
        "Daily paper-trading report (pretend money, not advice)\n"
        "Average return per trade after assumed costs\n\n"
        f"Open trades: {open_count}\n"
        + fmt_group("AI picks", groups["ai"]) + "\n"
        + fmt_group("Random picks", groups["random"]) + "\n"
        + fmt_group("Other (passed filters)", groups["other"]) + "\n"
        + fmt_group("Rejected as risky (shadow)", groups["rejected"]) + "\n\n"
        f"Paper P&L on closed AI picks (${TRADE_SIZE} each): ${pnl:+.2f}\n"
        "If Rejected does clearly worse than the passed groups, the filters are working. "
        "Wait for 30+ trades per group."
    )


def add_trade(ledger, c, group, reasons):
    rug = c.get("rug") or {}
    sell = c.get("sell") or {}
    ledger["trades"].append({
        "address": c["address"],
        "symbol": c["symbol"],
        "entry_price": c["price_usd"],
        "group": group,
        "picked": group == "ai",
        "status": "open",
        "opened": time.time(),
        "snaps": {},
        "liquidity": c["liquidity"],
        "volume_24h": c["volume_24h"],
        "change_1h_pct": c["change_1h_pct"],
        "change_24h_pct": c["change_24h_pct"],
        "buys_24h": c["buys_24h"],
        "sells_24h": c["sells_24h"],
        "age_hours": c["age_hours"],
        "fdv": c["fdv"],
        "rug_score": rug.get("score"),
        "danger_flags": rug.get("danger_flags"),
        "roundtrip": sell.get("roundtrip"),
        "reject_reasons": reasons,
        "safe_core": not reasons,
    })


ledger = load_ledger()
update_trades(ledger)

seen = {t["address"] for t in ledger["trades"]}
new = [c for c in find_candidates() if c["address"] not in seen][:MAX_CANDIDATES]

accepted, rejected, skipped = [], [], 0
for c in new:
    reasons = market_reasons(c)
    if not reasons:
        rug = rug_check(c["address"])
        time.sleep(2.5)
        if rug is None:
            skipped += 1
            continue
        c["rug"] = rug
        reasons += rug_reasons(rug)
    if not reasons and CHECK_SELL_ROUTE:
        sell = sell_check(c["address"])
        time.sleep(1)
        if sell["status"] == "unavailable":
            skipped += 1
            continue
        c["sell"] = sell
        if sell["status"] == "fail":
            reasons.append(sell["reason"])
    c["reasons"] = reasons
    (rejected if reasons else accepted).append(c)

for i, c in enumerate(accepted, start=1):
    c["id"] = i

ai_ids = set()
explanation = ""
if accepted:
    view = [
        {
            "id": c["id"],
            "symbol": c["symbol"],
            "price_usd": c["price_usd"],
            "liquidity": c["liquidity"],
            "volume_24h": c["volume_24h"],
            "change_1h_pct": c["change_1h_pct"],
            "change_24h_pct": c["change_24h_pct"],
            "buys_24h": c["buys_24h"],
            "sells_24h": c["sells_24h"],
            "age_hours": c["age_hours"],
            "rugcheck_risk_score": (c.get("rug") or {}).get("score"),
            "sell_roundtrip": (c.get("sell") or {}).get("roundtrip"),
        }
        for c in accepted
    ]
    prompt = (
        "You are a cautious crypto market analyst. Below is JSON data on new Solana tokens "
        "that already passed rug-pull screening, each with an id. Choose up to 3 that look "
        "most interesting based ONLY on this data. Do not predict prices and do not invent "
        "facts. Your reply must start with one line in exactly this format: PICKS: 2, 5 "
        "(the chosen ids, or PICKS: none). Then give one short line per pick with the "
        "symbol, the key reason, and the main risk. Keep it under 120 words.\n\n"
        + json.dumps(view)
    )
    analysis = ask_llm(prompt)
    ai_ids = parse_picks(analysis, {c["id"] for c in accepted})
    explanation = re.sub(r"^.*PICKS:.*$\n?", "", analysis, count=1, flags=re.M | re.I).strip()

pool = [c for c in accepted if c["id"] not in ai_ids]
random_ids = {c["id"] for c in random.sample(pool, min(MAX_RANDOM_PICKS, len(pool)))}

lines = []
for c in accepted:
    group = "ai" if c["id"] in ai_ids else "random" if c["id"] in random_ids else "other"
    add_trade(ledger, c, group, [])
    tag = {"ai": "AI", "random": "RND", "other": "-"}[group]
    lines.append(f"{c['id']}. {c['symbol']} [{tag}]: {c['url']}")
for c in rejected:
    add_trade(ledger, c, "rejected", c["reasons"])
save_ledger(ledger)

if accepted:
    why = Counter(label for c in rejected for label in c["reasons"])
    skipped_note = ""
    if rejected:
        skipped_note = (
            f"\n\nRejected {len(rejected)} risky: "
            + ", ".join(f"{k} x{v}" for k, v in why.most_common(5))
        )
    send(
        "New scan (paper trading only, not financial advice)\n\n"
        + explanation
        + "\n\nPassed screening (AI = AI pick, RND = random pick):\n"
        + "\n".join(lines)
        + skipped_note
    )
else:
    print(f"No tokens passed screening. Rejected: {len(rejected)}, skipped (lookup failed): {skipped}")

has_closed = any(t["status"] == "closed" for t in ledger["trades"])
if has_closed and time.time() - ledger["last_report"] >= 24 * 3600:
    send(build_report(ledger))
    ledger["last_report"] = time.time()
    save_ledger(ledger)

print("Done.")
