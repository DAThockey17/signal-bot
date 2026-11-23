import os
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
LLM_KEY = os.environ["GROQ_API_KEY"].strip()
print("key length:", len(LLM_KEY), "starts with:", LLM_KEY[:4])
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
_m = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {LLM_KEY}"}, timeout=30)
print("available:", [m["id"] for m in _m.json().get("data", [])])

def ask_llm(prompt):
    r = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit(f"LLM error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def send(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text},
        timeout=15,
    )
    if r.status_code != 200:
        raise SystemExit(f"Telegram error {r.status_code}: {r.text[:300]}")


reply = ask_llm("In one short sentence, confirm you are working.")
send("Test OK ✅\nAI says: " + reply)
