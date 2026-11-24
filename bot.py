import os
import re
import json
import time
import random
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
LLM_KEY = os.environ["GROQ_API_KEY"].strip()
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

# ---- Filters (edit these) ----
MIN_LIQUIDITY = 25000
MIN_VOLUME_24H = 50000
MIN_VOL_LIQ_RATIO = 1.0
MIN_AGE_HOURS = 1
MAX_CANDIDATES = 12
MAX_AI_PICKS = 3
MAX_RANDOM_PICKS = 3

# ---- Paper trading ----
LEDGER_FILE = "paper_trades.json"
TRADE_SIZE = 100
FEE_PCT = 3.0
CHECKPOINTS = [1, 6, 24]


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
            "url": p.get("url"),
        }
    return sorted(best.values(), key=lambda c: c["volume_24h"], reverse=True)


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
    names = [x.get("name", "") for x in risks]
    critical = any(
        "mint authority" in n.lower() or "freeze authority" in n.lower() for n in names
    )
    return {
        "score": d.get("score_normalised"),
        "rugged": bool(d.get("rugged")),
        "danger_flags": [x.get("name") for x in risks if x.get("level") == "danger"],
        "safe_core": (not d.get("rugged")) and not critical,
    }


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
    return t.get("group") or ("ai" if t.get("picked") else "other")


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
    ai = [t for t in trades if group_of(t) == "ai"]
    rnd = [t for t in trades if group_of(t) == "random"]
    other = [t for t in trades if group_of(t) == "other"]
    safe = [t for t in trades if t.get("safe_core") is True]
    unsafe = [t for t in trades if t.get("safe_core") is False]
    ai_closed = [t for t in ai if t["status"] == "closed"]
    pnl = sum(TRADE_SIZE * t["return_pct"] / 100 for t in ai_closed)
    return (
        "Daily paper-trading report (pretend money, not advice)\n"
        "Average return per trade after assumed costs\n\n"
        f"Open trades: {open_count}\n"
        + fmt_group("AI picks", ai) + "\n"
        + fmt_group("Random picks", rnd) + "\n"
        + fmt_group("Other", other) + "\n"
        + fmt_group("Safe (no mint/freeze)", safe) + "\n"
        + fmt_group("Unsafe", unsafe) + "\n\n"
        f"Paper P&L on closed AI picks (${TRADE_SIZE} each): ${pnl:+.2f}\n"
        "Key test: do AI picks beat Random picks? Wait for 30+ trades per group."
    )


ledger = load_ledger()
update_trades(ledger)

seen = {t["address"] for t in ledger["trades"]}
new = [c for c in find_candidates() if c["address"] not in seen][:MAX_CANDIDATES]
for i, c in enumerate(new, start=1):
    c["id"] = i
    c["rug"] = rug_check(c["address"])
    time.sleep(2.5)

if new:
    eligible = [c for c in new if c["rug"] and c["rug"]["safe_core"]]
    ai_ids = set()
    explanation = "No tokens passed the safety check for AI picks this run."
    if eligible:
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
                "rugcheck_risk_score": c["rug"]["score"],
                "rugcheck_danger_flags": c["rug"]["danger_flags"],
            }
            for c in eligible
        ]
        prompt = (
            "You are a cautious crypto market analyst. Below is JSON data on new Solana tokens, "
            "each with an id. Choose up to 3 that look most interesting based ONLY on this data. "
            "Do not predict prices and do not invent facts. "
            "Your reply must start with one line in exactly this format: PICKS: 2, 5 "
            "(the chosen ids, or PICKS: none). Then give one short line per pick with the "
            "symbol, the key reason, and the main risk. Keep it under 120 words.\n\n"
            + json.dumps(view)
        )
        analysis = ask_llm(prompt)
        ai_ids = parse_picks(analysis, {c["id"] for c in eligible})
        explanation = re.sub(
            r"^.*PICKS:.*$\n?", "", analysis, count=1, flags=re.M | re.I
        ).strip()

    pool = [c for c in new if c["id"] not in ai_ids]
    random_ids = {c["id"] for c in random.sample(pool, min(MAX_RANDOM_PICKS, len(pool)))}

    lines = []
    for c in new:
        group = "ai" if c["id"] in ai_ids else "random" if c["id"] in random_ids else "other"
        rug = c["rug"]
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
            "rug_score": rug["score"] if rug else None,
            "danger_flags": rug["danger_flags"] if rug else None,
            "safe_core": rug["safe_core"] if rug else None,
        })
        tag = {"ai": "AI", "random": "RND", "other": "-"}[group]
        safety = "safe" if rug and rug["safe_core"] else "UNSAFE" if rug else "unchecked"
        lines.append(f"{c['id']}. {c['symbol']} [{tag}] {safety}: {c['url']}")
    save_ledger(ledger)

    send(
        "New scan (paper trading only, not financial advice)\n\n"
        + explanation
        + "\n\nAll new candidates (AI = AI pick, RND = random pick):\n"
        + "\n".join(lines)
    )
else:
    print("No new candidates this run.")

save_ledger(ledger)

has_closed = any(t["status"] == "closed" for t in ledger["trades"])
if has_closed and time.time() - ledger["last_report"] >= 24 * 3600:
    send(build_report(ledger))
    ledger["last_report"] = time.time()
    save_ledger(ledger)

print("Done.")
