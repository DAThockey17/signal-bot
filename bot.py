import os
import re
import json
import time
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
LLM_KEY = os.environ["GROQ_API_KEY"].strip()
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

MIN_LIQUIDITY = 25000
MIN_VOLUME_24H = 50000
MAX_CANDIDATES = 8

LEDGER_FILE = "paper_trades.json"
TRADE_SIZE = 100   # pretend dollars per trade
HOLD_HOURS = 24    # how long each trade is held
FEE_PCT = 3.0      # assumed round-trip cost per trade


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
    best = {}
    for p in data.get("pairs") or []:
        if p.get("chainId") != "solana":
            continue
        liq = (p.get("liquidity") or {}).get("usd") or 0
        vol = (p.get("volume") or {}).get("h24") or 0
        price = float(p.get("priceUsd") or 0)
        if liq < MIN_LIQUIDITY or vol < MIN_VOLUME_24H or price <= 0:
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
            "url": p.get("url"),
        }
    ranked = sorted(best.values(), key=lambda c: c["volume_24h"], reverse=True)
    return ranked[:MAX_CANDIDATES]


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


def update_trades(ledger):
    open_trades = [t for t in ledger["trades"] if t["status"] == "open"]
    if not open_trades:
        return
    prices = fetch_prices([t["address"] for t in open_trades])
    for t in open_trades:
        price = prices.get(t["address"])
        if price:
            t["last_price"] = price
        age_hours = (time.time() - t["opened"]) / 3600
        if age_hours < HOLD_HOURS:
            continue
        if price:
            ret = (price / t["entry_price"] - 1) * 100 - FEE_PCT
        else:
            ret = -100.0
            t["note"] = "no price found; counted as total loss"
        t["status"] = "closed"
        t["exit_price"] = price
        t["return_pct"] = round(max(ret, -100.0), 2)
        t["closed"] = time.time()


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


def parse_picks(text, n):
    m = re.search(r"PICKS:\s*([^\n]*)", text, re.I)
    if not m:
        return set()
    return {int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n}


def group_stats(group):
    if not group:
        return "no closed trades yet"
    rets = [t["return_pct"] for t in group]
    wins = sum(1 for r in rets if r > 0)
    return f"{len(group)} trades, win rate {wins / len(group) * 100:.0f}%, avg {sum(rets) / len(rets):+.1f}%"


def build_report(ledger):
    trades = ledger["trades"]
    closed = [t for t in trades if t["status"] == "closed"]
    open_count = len(trades) - len(closed)
    picked = [t for t in closed if t["picked"]]
    others = [t for t in closed if not t["picked"]]
    pnl = sum(TRADE_SIZE * t["return_pct"] / 100 for t in picked)
    return (
        "Daily paper-trading report (pretend money, not advice)\n\n"
        f"Open trades: {open_count}\n"
        f"AI picks: {group_stats(picked)}\n"
        f"Not picked: {group_stats(others)}\n"
        f"Paper P&L on AI picks (${TRADE_SIZE} each): ${pnl:+.2f}\n\n"
        "Judge by the AI picks vs Not picked comparison, and wait for 30+ closed trades."
    )


ledger = load_ledger()
update_trades(ledger)

seen = {t["address"] for t in ledger["trades"]}
new = [c for c in find_candidates() if c["address"] not in seen]
for i, c in enumerate(new, start=1):
    c["id"] = i

if new:
    view = [{k: v for k, v in c.items() if k not in ("address", "url")} for c in new]
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
    picked_ids = parse_picks(analysis, len(new))
    explanation = re.sub(r"^.*PICKS:.*$\n?", "", analysis, count=1, flags=re.M | re.I).strip()

    for c in new:
        ledger["trades"].append({
            "address": c["address"],
            "symbol": c["symbol"],
            "entry_price": c["price_usd"],
            "picked": c["id"] in picked_ids,
            "status": "open",
            "opened": time.time(),
        })
    save_ledger(ledger)

    lines = [
        f"{c['id']}. {c['symbol']}{' [paper buy]' if c['id'] in picked_ids else ''}: {c['url']}"
        for c in new
    ]
    send(
        "New scan (paper trading only, not financial advice)\n\n"
        + explanation
        + "\n\nAll new candidates:\n"
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
