#!/bin/bash
# 切换到路由代理（含配置备份 + 健康检查 + 验证）
set -e
HERMES="/Users/ma893399/.hermes/hermes-agent/.venv/bin/hermes"
DIR="$HOME/projects/hermes-token-router"
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$DIR/backups/config_$STAMP.yaml"

echo "=========================================="
echo "  切换到 Token 路由代理"
echo "=========================================="

# 1. 代理健康检查（代理挂了会导致 Hermes 全挂）
if ! curl -sf -m 5 http://127.0.0.1:8787/health > /dev/null; then
  echo "❌ 代理未运行！先启动：launchctl load ~/Library/LaunchAgents/ai.hermes.tokenrouter.plist"
  exit 1
fi
echo "✅ 代理健康检查通过"

# 2. 备份当前配置
mkdir -p "$DIR/backups"
cp "$HOME/.hermes/config.yaml" "$BACKUP"
echo "✅ 已备份配置 → $BACKUP"

# 3. 切换（使用 hermes config set，避免手改 yaml 出错）
"$HERMES" config set model.provider custom
"$HERMES" config set model.base_url "http://127.0.0.1:8787/v1"
"$HERMES" config set model.api_key "router-local"
echo "✅ 已切换 model.base_url → http://127.0.0.1:8787/v1"

# 4. 连通性验证（CLI 独立进程，不影响 gateway 会话）
echo ""
echo "=== 验证（CLI 测试）==="
timeout 90 "$HERMES" chat -q "回复两个字：通了" 2>&1 | tail -5 || echo "⚠️ CLI 测试超时或失败"

echo ""
echo "如需回滚: bash $DIR/rollback.sh"
