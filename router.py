"""Hermes Token 路由代理 —— OpenAI 兼容端点
按《Hermes 智能体 Token 优化与模型路由技术方案》实现。

用法:
    export DEEPSEEK_API_KEY=...      # 兜底（必需）
    export GROQ_API_KEY=...          # L1 免费池（可选，有则启用）
    python router.py
    # Hermes 指向 http://127.0.0.1:8787/v1
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from providers import ProviderError, ProviderPool
from routing import classify_data, decide, extract_features, est_tokens

BASE = Path(__file__).resolve().parent
CFG = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("router")

ROUTES = CFG["routes"]
FINAL = tuple(CFG["final_fallback"].values())
SEC = CFG.get("security", {})
BUDGET = CFG.get("budget", {})
OBS = CFG.get("observe", {})
POOL = ProviderPool(CFG["providers"], SEC.get("paid_only_providers"), BUDGET.get("l1_timeout_seconds", 25))
LOG_PATH = BASE / OBS.get("log_file", "logs/route_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# 运行统计
STATS = {"requests": 0, "by_route": {}, "by_provider": {}, "degraded": 0}


def observe(entry: dict):
    """观测埋点（第 12.4 章）：route / provider / model / latency / tokens / verdict"""
    if OBS.get("enabled", True):
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("写观测日志失败: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Hermes Token 路由代理启动 ===")
    log.info("可用 provider: %s", ", ".join(
        f"{n}{'✅' if POOL.has_key(n) else '❌(缺Key)'}" for n in CFG["providers"]))
    log.info("路由规则: %s", ", ".join(f"{r['name']}(p{r.get('priority')})" for r in ROUTES))
    log.info("兜底: %s/%s", FINAL[0], FINAL[1])
    log.info("监听 http://%s:%s/v1", CFG["server"]["host"], CFG["server"]["port"])
    yield
    await POOL.close()


app = FastAPI(title="Hermes Token Router", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": {n: {"has_key": POOL.has_key(n), "paid": CFG["providers"][n].get("paid", False),
                          "breaker": POOL.breaker.snapshot().get(n)} for n in CFG["providers"]},
        "stats": STATS,
        "pool": POOL.stats,
    }


@app.post("/admin/reset_breaker")
async def reset_breaker():
    """手动重置熔断状态（性能排查用）"""
    POOL.reset_breaker()
    log.info("熔断状态已重置")
    return {"status": "ok", "breakers": POOL.breaker.snapshot()}


@app.get("/v1/models")
async def list_models():
    """Hermes 可能查询模型列表——返回兜底模型名，保证兼容。"""
    ids = set()
    for r in ROUTES:
        for _, m in r.get("chain", []):
            ids.add(m)
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "router"} for i in sorted(ids)]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    t0 = time.monotonic()
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "请求体不是合法 JSON"}}, status_code=400)

    payload.pop("_data_class", None)
    feats = extract_features(payload)
    route = decide(feats, ROUTES)
    chain = [list(x) for x in route.get("chain", [])]
    classification = classify_data({"classify_text": feats["text"], **payload})
    stream = bool(payload.get("stream"))

    # 强制上限（第 6 章早停）
    if route.get("max_tokens"):
        payload.setdefault("max_tokens", route["max_tokens"])

    # 数据分级：confidential 只保留付费 provider
    if classification == "confidential":
        chain = [c for c in chain if c[0] in SEC.get("paid_only_providers", [])]
        if not chain:
            chain = [["deepseek", "deepseek-v4-flash"]]

    STATS["requests"] += 1
    STATS["by_route"][route["name"]] = STATS["by_route"].get(route["name"], 0) + 1
    log.info("请求 #%d | 路由=%s | 输入≈%d tokens | 工具=%s | 分级=%s | 流式=%s | chain=%s",
             STATS["requests"], route["name"], feats["input_tokens"],
             feats["tool_count"], classification, stream, chain)

    if stream:
        return await _handle_stream(chain, payload, route, feats, classification, t0)
    return await _handle_normal(chain, payload, route, feats, classification, t0)


async def _handle_normal(chain, payload, route, feats, classification, t0):
    try:
        data, provider, model = await POOL.call_with_chain(chain, payload, classification, FINAL)
    except ProviderError as e:
        log.error("全部失败: %s", e)
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)

    latency = round(time.monotonic() - t0, 2)
    u = data.get("usage") or {}
    STATS["by_provider"][provider] = STATS["by_provider"].get(provider, 0) + 1
    observe({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "stream": False, "route": route["name"],
             "provider": provider, "model": model, "latency": latency,
             "in_tokens": u.get("prompt_tokens"), "out_tokens": u.get("completion_tokens"),
             "classification": classification, "tool_count": feats["tool_count"],
             "est_in": feats["input_tokens"], "ok": True})
    log.info("完成 | %s/%s | %.2fs | in=%s out=%s", provider, model, latency,
             u.get("prompt_tokens"), u.get("completion_tokens"))
    return JSONResponse(data)


async def _handle_stream(chain, payload, route, feats, classification, t0):
    """流式：连接阶段失败可切换，已开始输出则直接透传（无法回退）。"""
    last_err = None
    for provider, model in chain + [list(FINAL)]:
        if not POOL.has_key(provider) or not POOL.available(provider, classification):
            continue
        body = {**payload, "model": model, "stream": True}
        client = await POOL.client(provider)
        url = f"{CFG['providers'][provider]['base_url']}/chat/completions"
        try:
            req = client.build_request("POST", url, headers=POOL._headers(provider), json=body)
            resp = await client.send(req, stream=True)
            if resp.status_code >= 400:
                text = (await resp.aread())[:200].decode("utf-8", "ignore")
                await resp.aclose()
                POOL.breaker.record(provider, False, POOL._exempt(provider))   # 流式失败也计入熔断统计
                last_err = f"{provider} {resp.status_code}: {text}"
                log.warning("流式失败 %s → 切换下一家: %s", provider, last_err)
                continue

            POOL.breaker.record(provider, True, POOL._exempt(provider))
            STATS["by_provider"][provider] = STATS["by_provider"].get(provider, 0) + 1
            observe({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "stream": True, "route": route["name"],
                     "provider": provider, "model": model, "classification": classification,
                     "latency": round(time.monotonic() - t0, 2),
                     "tool_count": feats["tool_count"], "est_in": feats["input_tokens"], "ok": True})

            async def gen(r=resp):
                try:
                    async for chunk in r.aiter_raw():
                        yield chunk
                finally:
                    await r.aclose()

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers={"X-Router-Provider": provider, "X-Router-Model": model})
        except Exception as e:
            last_err = f"{provider}: {e}"
            log.warning("流式异常 %s → 切换: %s", provider, last_err)
            continue

    log.error("流式全部失败: %s", last_err)
    return JSONResponse({"error": {"message": f"所有 provider 失败: {last_err}"}}, status_code=502)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]), log_level="info")
