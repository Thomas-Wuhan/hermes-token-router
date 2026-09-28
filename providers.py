"""Provider 池：限速 + 熔断 + fallback 链 + 重试分类
按《Token 优化技术方案》第 7 章（调用封装）与第 11 章（重试机制）实现。

核心原则（第 11.1 章）：
  重试只留给瞬时故障；能力性失败一律升级或切换，不重试。
"""
import asyncio
import logging
import os
import random
import time
from typing import Any

import httpx

log = logging.getLogger("router.providers")


class RateLimiter:
    """令牌桶限速器（第 7 章）。按厂商公布值配置，不要调高——真实限额不会因本地配置变大。"""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self.last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            delta = now - self.last
            if delta < self.interval:
                await asyncio.sleep(self.interval - delta)
            self.last = time.monotonic()


class CircuitBreaker:
    """熔断器（第 11.3 章）：失败率超阈值则摘除一段时间，避免逐请求试错。"""

    def __init__(self, threshold: float = 0.5, cooldown: float = 600, min_samples: int = 5):
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_samples = min_samples
        self.stats: dict[str, list[int]] = {}      # provider -> [fail, ok]
        self.banned_until: dict[str, float] = {}

    def allow(self, provider: str) -> bool:
        return time.monotonic() > self.banned_until.get(provider, 0)

    def record(self, provider: str, ok: bool, exempt: bool = False):
        """exempt=True（兜底 provider）只统计不熔断 —— 它是最后防线，必须永远可用"""
        s = self.stats.setdefault(provider, [0, 0])
        s[0 if ok else 1] += 1
        if exempt:
            return
        total = s[0] + s[1]
        if total >= self.min_samples and (s[1] / total) > self.threshold:
            self.banned_until[provider] = time.monotonic() + self.cooldown
            log.warning("熔断: %s 失败率 %.0f%% → 摘除 %ds", provider, 100 * s[1] / total, self.cooldown)
            self.stats[provider] = [0, 0]

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            p: {"fail": v[0], "ok": v[1],
                "banned_for": max(0, int(self.banned_until.get(p, 0) - now))}
            for p, v in self.stats.items()
        }


class ProviderError(Exception):
    """供应商调用失败，带分类（供上层决定重试还是切换）"""

    def __init__(self, provider: str, kind: str, detail: str = ""):
        self.provider = provider
        self.kind = kind          # rate_limit | transient | timeout | fatal
        super().__init__(f"[{provider}/{kind}] {detail}")


class ProviderPool:
    """多供应商客户端池。所有厂商均为 OpenAI 兼容协议（第 7 章）。"""

    def __init__(self, providers_cfg: dict, paid_only: list[str] | None = None, l1_timeout: float = 25.0):
        self.cfg = providers_cfg
        self.paid_only = set(paid_only or [])
        self.l1_timeout = l1_timeout
        self.limiters = {name: RateLimiter(int(p.get("rpm", 30))) for name, p in providers_cfg.items()}
        self.breaker = CircuitBreaker()
        self._clients: dict[bool, httpx.AsyncClient] = {}
        # 代理地址（免费层国内需梯子；DeepSeek 直连更快）
        self.proxy = os.environ.get("ROUTER_PROXY", "http://127.0.0.1:7890")
        self.stats = {"calls": 0, "ok": 0, "fallback_used": 0, "tokens_in": 0, "tokens_out": 0}

    async def client(self, provider: str = "") -> httpx.AsyncClient:
        """按 provider 返回客户端：需要走代理的用代理，其余直连（更快）"""
        use_proxy = bool(self.cfg.get(provider, {}).get("use_proxy"))
        c = self._clients.get(use_proxy)
        if c is None or c.is_closed:
            kw = dict(timeout=httpx.Timeout(300.0, connect=15.0), follow_redirects=True)
            if use_proxy:
                kw["proxy"] = self.proxy
            c = httpx.AsyncClient(**kw)
            self._clients[use_proxy] = c
        return c

    def _exempt(self, provider: str) -> bool:
        """兜底 provider（config: no_breaker: true）永不被熔断"""
        return bool(self.cfg.get(provider, {}).get("no_breaker"))

    def reset_breaker(self, provider: str | None = None):
        """重置熔断状态（性能排查用）"""
        if provider:
            self.breaker.banned_until.pop(provider, None)
            self.breaker.stats.pop(provider, None)
        else:
            self.breaker.banned_until.clear()
            self.breaker.stats.clear()

    def available(self, provider: str, classification: str) -> bool:
        """数据分级检查（第 8 章红线）：confidential 只走付费 provider"""
        if classification == "confidential" and provider not in self.paid_only:
            return False
        return True

    async def close(self):
        for c in self._clients.values():
            if not c.is_closed:
                await c.aclose()

    def _headers(self, provider: str) -> dict:
        p = self.cfg[provider]
        key = os.environ.get(p["api_key_env"], "")
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        if provider == "openrouter":                      # OpenRouter 建议带头（计入排行榜）
            h["HTTP-Referer"] = "http://localhost:8787"
            h["X-Title"] = "hermes-token-router"
        return h

    def has_key(self, provider: str) -> bool:
        return bool(os.environ.get(self.cfg[provider]["api_key_env"], ""))

    async def call(self, provider: str, model: str, payload: dict,
                   classification: str = "internal") -> dict:
        """单次调用（含限速）。抛出 ProviderError（已分类）。"""
        if provider not in self.cfg:
            raise ProviderError(provider, "fatal", "未配置该 provider")
        if not self.available(provider, classification):
            raise ProviderError(provider, "fatal", f"数据分级 {classification} 禁止走该 provider")
        if not self.has_key(provider):
            raise ProviderError(provider, "fatal", "缺少 API Key")
        if not self._exempt(provider) and not self.breaker.allow(provider):
            raise ProviderError(provider, "fatal", "熔断中")

        body = dict(payload)
        body["model"] = model
        # provider 级额外参数（如 siliconflow 的 enable_thinking=false）
        extra = self.cfg[provider].get("extra") or {}
        for k, v in extra.items():
            body.setdefault(k, v)
        await self.limiters[provider].wait()

        client = await self.client(provider)
        self.stats["calls"] += 1
        try:
            r = await client.post(f"{self.cfg[provider]['base_url']}/chat/completions",
                                  headers=self._headers(provider), json=body)
        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            self.breaker.record(provider, False, self._exempt(provider))
            raise ProviderError(provider, "timeout", str(e))
        except Exception as e:
            self.breaker.record(provider, False, self._exempt(provider))
            raise ProviderError(provider, "transient", str(e))

        if r.status_code == 429:
            self.breaker.record(provider, False, self._exempt(provider))
            raise ProviderError(provider, "rate_limit", r.text[:200])
        if r.status_code >= 500:
            self.breaker.record(provider, False, self._exempt(provider))
            raise ProviderError(provider, "transient", r.text[:200])
        if r.status_code >= 400:
            self.breaker.record(provider, False, self._exempt(provider))
            raise ProviderError(provider, "fatal", r.text[:300])

        self.breaker.record(provider, True, self._exempt(provider))
        self.stats["ok"] += 1
        data = r.json()
        u = data.get("usage") or {}
        self.stats["tokens_in"] += u.get("prompt_tokens", 0) or 0
        self.stats["tokens_out"] += u.get("completion_tokens", 0) or 0
        return data

    async def call_with_chain(self, chain: list[list[str]], payload: dict,
                              classification: str = "internal",
                              final: tuple[str, str] | None = None) -> tuple[dict, str, str]:
        """按 fallback 链依次尝试（第 11.1 章）：
           - 429  → 退避重试至多 2 次（同 provider），再切下一家
           - 瞬时故障 → 立即切下一家（免费层别恋战）
                返回 (response, provider, model)
        """
        for provider, model in chain:
            if not self.has_key(provider):
                log.debug("跳过 %s（无 Key）", provider)
                continue
            for attempt in range(3):
                try:
                    data = await self.call(provider, model, payload, classification)
                    if chain.index([provider, model]) > 0:
                        self.stats["fallback_used"] += 1
                    return data, provider, model
                except ProviderError as e:
                    if e.kind == "rate_limit" and attempt < 2:
                        backoff = 1.0 * (2 ** attempt) + random.uniform(0, 0.5)   # 指数退避 + 抖动
                        log.info("429 %s，退避 %.1fs 后重试", provider, backoff)
                        await asyncio.sleep(backoff)
                        continue
                    if e.kind == "transient":
                        log.info("瞬时故障 %s：%s → 切换下一家", provider, e)
                        break
                    log.info("不可用 %s：%s", provider, e)
                    break

        if final:
            p, m = final
            data = await self.call(p, m, payload, classification)
            return data, p, m
        raise ProviderError("all", "fatal", "fallback 链全部失败且无兜底")
