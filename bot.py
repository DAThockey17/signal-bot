import os
import json
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
LLM_KEY = os.environ["GROQ_API_KEY"].strip()
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

MIN_LIQUIDITY = 25000   # USD
MIN_VOLUME_24H = 50000  # USD
MAX_CANDIDATES = 8


def get_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


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
        if liq < MIN_LIQUIDITY or vol < MIN_VOLUME_24H:
            continue
        addr = p["baseToken"]["address"]
        if addr in best and best[addr]["liquidity"] >= liq:
            continue
        tx = (p.get("txns") or {}).get("h24") or {}
        change = p.get("priceChange") or {}
        best[addr] = {
            "symbol": p["baseToken"].get("symbol"),
            "price_usd": p.get("priceUsd"),
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


candidates = find_candidates()
if not candidates:
    print("No candidates passed the filters this run.")
    raise SystemExit(0)

prompt = (
    "You are a cautious crypto market analyst. Below is JSON data on new Solana tokens. "
    "Pick up to 3 that look most interesting and write one line for each: the symbol, "
    "then the key reason and the main risk, based ONLY on this data. "
    "Do not predict prices and do not invent facts. If none look good, say so. "
    "Keep the whole reply under 120 words.\n\n"
    + json.dumps(candidates)
)
analysis = ask_llm(prompt)

links = "\n".join(f"{c['symbol']}: {c['url']}" for c in candidates)
send(
    "Scan results (paper trading only, not financial advice)\n\n"
    + analysis
    + "\n\nAll candidates:\n"
    + links
)
print("Alert sent.")
