#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "== Health check =="
curl -sS "$BASE_URL/health" | python -m json.tool
echo ""

echo "== Generate test =="

# If you don't have python available, swap this for jq pretty-print.
payload=$(cat <<'JSON'
{
  "note": "54yo male, fever and cough 3 days. SpO2 92% on room air, BP 98/60, HR 112, RR 28. Reports pleuritic chest discomfort.",
  "extra_context": "WBC 16.2, CRP 120. PMH: asthma. Allergies: NKDA. Meds: salbutamol inhaler.",
  "mode": "All-in-one (SOAP + Tasks + Red flags + Patient summary)",
  "max_input_len": 1024,
  "max_new_tokens": 512,
  "max_total_new_tokens": 4096,
  "chunk_new_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "no_repeat_ngram_size": 3,
  "auto_continue": true
}
JSON
)

resp=$(curl -sS -X POST "$BASE_URL/v1/generate" \
  -H "Content-Type: application/json" \
  -d "$payload")

echo "$resp" | python -m json.tool

echo ""
echo "Tip: if you want ONLY the text:"
echo "$resp" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
print(obj.get("text",""))
PY
