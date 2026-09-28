"""缓存层（方案第 5 章）
双层设计：
  1. 精确 hash 缓存 —— prompt 完全一致即命中（零成本、毫秒级）
  2. 语义缓存 —— embedding 余弦相似度 ≥ 阈值即命中（对改写/同义请求有效）

安全边界（重要）：
  只缓存"无工具 + 非流式"的确定性请求（压缩/摘要/翻译/分类）。
  带工具的请求上下文强相关，缓存会导致错误结果；流式不缓存。
"""
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path

log = logging.getLogger("router.cache")


def cos_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class ResponseCache:
    def __init__(self, db_path: str = "logs/cache.db", threshold: float = 0.97,
                 ttl: int = 86400, max_entries: int = 500):
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY, text TEXT, embedding TEXT,
            response TEXT, ts REAL, hits INTEGER DEFAULT 0)""")
        self.db.commit()
        self.stats = {"exact_hit": 0, "semantic_hit": 0, "miss": 0, "stored": 0}

    # ---------- 键 ----------
    @staticmethod
    def make_key(payload: dict) -> str:
        msgs = payload.get("messages") or []
        norm = json.dumps(
            [{"r": m.get("role"), "c": m.get("content")} for m in msgs],
            ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @staticmethod
    def cacheable(payload: dict, route_name: str) -> bool:
        """可缓存判定：无工具 + 非流式 + 命中可缓存路由"""
        if payload.get("stream"):
            return False
        if payload.get("tools"):
            return False
        return route_name in ("simple_task", "background_summarize", "long_context")

    # ---------- 读写 ----------
    def get_exact(self, payload: dict):
        k = self.make_key(payload)
        row = self.db.execute("SELECT response, ts FROM cache WHERE key=?", (k,)).fetchone()
        if row and (time.time() - row[1]) < self.ttl:
            self.db.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (k,))
            self.db.commit()
            self.stats["exact_hit"] += 1
            return json.loads(row[0])
        return None

    def get_semantic(self, embedding: list[float]):
        if not embedding:
            return None
        now = time.time()
        best_sim, best = 0.0, None
        for k, emb, resp, ts in self.db.execute(
                "SELECT key, embedding, response, ts FROM cache WHERE embedding IS NOT NULL"):
            if now - ts > self.ttl:
                continue
            try:
                sim = cos_sim(embedding, json.loads(emb))
            except Exception:
                continue
            if sim > best_sim:
                best_sim, best = sim, (resp, k)
        if best and best_sim >= self.threshold:
            self.db.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (best[1],))
            self.db.commit()
            self.stats["semantic_hit"] += 1
            log.info("语义缓存命中 | 相似度=%.4f", best_sim)
            return json.loads(best[0])
        self.stats["miss"] += 1
        return None

    def store(self, payload: dict, response: dict, embedding: list[float] | None = None):
        k = self.make_key(payload)
        text = json.dumps([m.get("content") for m in (payload.get("messages") or [])],
                          ensure_ascii=False, default=str)[:2000]
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,0)",
                        (k, text, json.dumps(embedding) if embedding else None,
                         json.dumps(response, ensure_ascii=False), time.time()))
        self.db.commit()
        self.stats["stored"] += 1
        self._evict()

    def _evict(self):
        """超量时淘汰最旧且命中次数少的"""
        n = self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if n > self.max_entries:
            self.db.execute("""DELETE FROM cache WHERE key IN (
                SELECT key FROM cache ORDER BY hits ASC, ts ASC LIMIT ?)""", (n - self.max_entries,))
            self.db.commit()

    def purge_expired(self):
        cutoff = time.time() - self.ttl
        cur = self.db.execute("DELETE FROM cache WHERE ts < ?", (cutoff,))
        self.db.commit()
        return cur.rowcount

    def info(self) -> dict:
        total = self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        hits = self.db.execute("SELECT SUM(hits) FROM cache").fetchone()[0] or 0
        return {"entries": total, "total_hits": hits, **self.stats}


# ---------- Embedding（硅基流动，免费小模型）----------
async def embed_text(text: str, client, api_key: str,
                     model: str = "Qwen/Qwen3-Embedding-0.6B") -> list[float] | None:
    """调用硅基流动 embedding（国内直连）。失败返回 None（缓存降级为精确匹配）。"""
    if not api_key or not text:
        return None
    try:
        r = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": text[:4000]}, timeout=20.0)
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
        log.debug("embedding 失败 %s: %s", r.status_code, r.text[:120])
    except Exception as e:
        log.debug("embedding 异常: %s", e)
    return None
