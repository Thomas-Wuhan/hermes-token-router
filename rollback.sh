#!/bin/bash
# 一键回滚：Hermes 恢复 DeepSeek 直连
set -e
HERMES="/Users/ma893399/.hermes/hermes-agent/.venv/bin/hermes"
DIR="$HOME/projects/hermes-token-router"

echo "=========================================="
echo "  回滚：恢复 DeepSeek 直连"
echo "=========================================="

"$HERMES" config set model.provider deepseek
"$HERMES" config set model.base_url "https://api.deepseek.com/v1"
"$HERMES" config set model.api_key ""

echo "✅ 已恢复: model.base_url → https://api.deepseek.com/v1 (provider=deepseek)"
echo ""
echo "验证："
timeout 60 "$HERMES" chat -q "回复两个字：正常" 2>&1 | tail -3 || echo "⚠️ 测试失败，请检查 DEEPSEEK_API_KEY"

echo ""
echo "（最近备份列表）"
ls -lt "$DIR/backups/" 2>/dev/null | head -5

echo ""
echo "如需恢复指定备份: cp $DIR/backups/config_<时间戳>.yaml ~/.hermes/config.yaml"
