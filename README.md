# Hermes Token 路由代理

> 按《Hermes 智能体 Token 优化与模型路由技术方案 v1.1》实施的多层模型路由层
>
> **目标**：主模型（DeepSeek）token 消耗下降 50-80%，同时不牺牲质量与体验

## 一句话原理

**主模型只做裁判和缝合，脏活累活给便宜（免费）模型干。**

```
Hermes ──► 本代理 (http://127.0.0.1:8787/v1)
              │
              ├─► 规则路由（任务分类）
              │
              ├─► L1 免费池：硅基流动 / Groq / OpenRouter
              │      · 压缩摘要、翻译改写、分类抽取
              │
              └─► 兜底层：DeepSeek 直连（付费但永远可用）
```

## 📊 实测效果（2026-09，本机实测）

### 关键突破：`enable_thinking: false`

Qwen3 系列默认开启思考模式，对"摘要/翻译"这类任务纯属浪费：

| 指标 | 思考模式 | **关闭思考** | 提升 |
|------|---------|-------------|------|
| 延迟（同一摘要任务）| 19.5s | **0.71s** | **27×** |
| tokens 消耗 | 393 | **19** | **省 95%** |
| 输出质量 | 正确 | 正确 | 无损 |

### 三级方案对比（同一压缩任务，4125 tokens 输入）

| 方案 | 延迟 | 成本 | 结论 |
|------|------|------|------|
| DeepSeek 直连（基线）| 5-10s | ¥付费 | 基线 |
| Groq（走代理）| 12.4s | 免费 | 太慢 |
| 硅基 Qwen3-8B（思考）| 19.5s | 免费 | 太慢 |
| **硅基 Qwen3-8B（关思考）** | **1.95s** | **免费** | ✅ **最优** |

### 路由决策实测

| 请求类型 | 路由 | 走向 | 延迟 |
|---------|------|------|------|
| 短请求（"你好"）| `default` | DeepSeek 直连 | 1.09s |
| 压缩任务（4k tokens）| `background_summarize` | 硅基免费池 | 1.95s |
| 带工具请求（Agent）| `default` | DeepSeek 直连 | 0.69s |

## 🏗 架构与文件

```
hermes-token-router/
├── router.py          # FastAPI 主程序（OpenAI 兼容端点）
├── providers.py       # Provider 池：限速 + 熔断 + fallback 链
├── routing.py         # 路由决策（规则匹配 + 数据分级）
├── config.yaml        # 路由规则 + Provider 配置（唯一需要改的文件）
├── start.sh           # 启动（自动加载 ~/.hermes/.env 里的 Key）
├── switch.sh          # 切换到代理（含备份 + 验证）
├── rollback.sh        # 一键回滚到 DeepSeek 直连
└── logs/route_log.jsonl   # 观测日志（route/provider/tokens/latency）
```

### 已实现（对应方案章节）

| 方案章节 | 实现 |
|---------|------|
| 第 4 章 分层路由 | ✅ 规则路由（`background_summarize` / `simple_task` / `long_context` / `default`）|
| 第 5 章 缓存 | ⏳ 待接入（语义缓存需 embedding）|
| 第 7 章 多供应商客户端 | ✅ OpenAI 兼容 + 限速器 + fallback 链 |
| 第 8 章 数据安全红线 | ✅ 数据分级（敏感词命中 → 强制走付费 DeepSeek）|
| 第 11 章 重试机制 | ✅ 429 退避重试 / 瞬时故障切换 / 熔断器（失败率 >50% 摘除 10 分钟）|
| 第 12 章 延迟优化 | ✅ 按 provider 区分代理（DeepSeek 直连 / 免费池走梯子）/ 延迟预算 |
| 第 12.4 章 观测埋点 | ✅ JSONL 记录 route/provider/model/latency/tokens |

## 🚀 快速开始

### 1. 配置 Key（写入 `~/.hermes/.env`）

```bash
DEEPSEEK_API_KEY=...        # 必需（兜底）
SILICONFLOW_API_KEY=...     # 强烈推荐（国内直连免费池，实测 0.7s）
GROQ_API_KEY=...            # 可选（境外，需梯子，较慢）
OPENROUTER_API_KEY=...      # 可选
GEMINI_API_KEY=...          # 可选（长文档 1M 上下文）
```

### 2. 启动

```bash
./start.sh                          # 前台运行
# 或常驻（推荐）
launchctl load ~/Library/LaunchAgents/ai.hermes.tokenrouter.plist
```

### 3. 接入 Hermes

```bash
bash switch.sh      # 切换（自动备份配置 + 健康检查 + CLI 验证）
bash rollback.sh    # 出问题一键回滚
```

### 4. 看效果

```bash
tail -f logs/route_log.jsonl       # 实时看路由与 token
curl -s localhost:8787/health      # 健康状态 + 熔断状态
```

## ⚠️ 踩坑记录（实测得出，很重要）

### 1. 免费模型目录变动频繁，**不要硬编码模型名**

首次配置时写的 `llama-3.1-8b-instant` / `meta-llama/llama-3.3-70b-instruct:free` **都已下架**。
→ 上线前用 `/v1/models` 核实，配置里按**能力**而非具体版本选。

### 2. 境外免费池在国内**必须走代理**，延迟代价大

Groq 走梯子实测 12-24s（DeepSeek 直连仅 1s）。
→ 策略：**只有长输入的 token 大户才值得等**；短请求交给直连。

### 3. Qwen3 系列默认开思考，**必须显式关闭**

不关思考：19.5s + 393 tokens（两个字回复）；关掉后 0.71s + 19 tokens。
→ 在 provider 配置里加 `extra: {enable_thinking: false}`。

### 4. js/位运算大数溢出（与本项目无关但同类坑）

DBC 解析时扩展帧 ID 带 `0x80000000`，JS 位运算会 32 位溢出 → 用算术减法。
（记录在此，提醒"看起来简单的位运算"要验证）

### 5. **代理是单点**

接入后 Hermes 全部流量经代理 → 代理挂了 Hermes 就挂。
→ 用 launchd `KeepAlive` 常驻 + 准备好 `rollback.sh`。

## 🔒 数据安全红线（方案第 8 章）

免费层（Gemini 明确、其余多数默认）会**将输入用于模型改进**：

- 代理内置**数据分级**：命中 `客户/合同/报价/密码/身份证/未公开/内部资料` 等敏感词 → **强制走付费 DeepSeek**
- 也可以用请求头 `X-Data-Class: confidential` 显式指定
- **不要把客户 PII、未公开代码、商业机密发给任何免费层**

## 📈 验收指标（方案第 9 章）

| 指标 | 目标 | 当前 |
|------|------|------|
| 主模型 token 下降 | ≥50% | 待接入后统计（压缩/摘要已 100% 转移）|
| 任务成功率 | ≥85% | fallback 链保证 |
| P95 延迟 | 不劣化 | 短请求直连不变；长任务 1.95s |

---

*实施日期：2026-09-28 ｜ 依据方案 v1.1 ｜ 本机环境：macOS 12.5.1 / Clash 代理 7890*
