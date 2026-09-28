#!/bin/bash
# 启动 Hermes Token 路由代理（含 Key 加载）
cd "$(dirname "$0")"

# 从 Hermes .env 加载 DeepSeek Key（兜底必需）
if [ -f "$HOME/.hermes/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$HOME/.hermes/.env" 2>/dev/null || true
  set +a
fi

# 免费池 Key（有就启用，没有则跳过 —— 请在 .env 或这里配置）
# export GROQ_API_KEY=""
# export GEMINI_API_KEY=""
# export OPENROUTER_API_KEY=""
# export SILICONFLOW_API_KEY=""

echo "=========================================="
echo "  🔀 Hermes Token 路由代理"
echo "=========================================="
echo "兜底 provider: ${DEEPSEEK_API_KEY:+deepseek ✅}${DEEPSEEK_API_KEY:-deepseek ❌(缺 DEEPSEEK_API_KEY)}"
echo "免费池: groq=${GROQ_API_KEY:+✅}${GROQ_API_KEY:-❌} gemini=${GEMINI_API_KEY:+✅}${GEMINI_API_KEY:-❌} openrouter=${OPENROUTER_API_KEY:+✅}${OPENROUTER_API_KEY:-❌} siliconflow=${SILICONFLOW_API_KEY:+✅}${SILICONFLOW_API_KEY:-❌}"
echo "端点: http://127.0.0.1:8787/v1"
echo "=========================================="

exec .venv/bin/python router.py
