"""路由决策层（方案第 4.2 章）
第一阶段用硬编码规则覆盖 80% 场景，不训练分类器。
"""
import logging

log = logging.getLogger("router.routing")


def est_tokens(text: str) -> int:
    """粗略 token 估算（中文按字、英文按 4 字符）—— 只用于路由判定，不追求精确。"""
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = max(0, len(text) - cn)
    return int(cn + other / 4) + 1


def extract_features(payload: dict) -> dict:
    """从 OpenAI 请求体中提取路由特征"""
    msgs = payload.get("messages") or []
    parts = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):                       # 多模态/分段内容
            for seg in c:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    parts.append(seg.get("text", ""))
    text = "\n".join(parts)
    tools = payload.get("tools") or []
    return {
        "text": text,
        "input_tokens": est_tokens(text),
        "has_tools": bool(tools),
        "tool_count": len(tools),
        "has_images": any(isinstance(m.get("content"), list) for m in msgs),
        "msg_count": len(msgs),
        "model_requested": payload.get("model", ""),
    }


def decide(features: dict, routes: list[dict]) -> dict:
    """按优先级匹配路由规则，返回命中的规则（含 chain / max_tokens）。"""
    for r in sorted(routes, key=lambda x: x.get("priority", 999)):
        w = r.get("when", {}) or {}

        if w.get("no_tools") and features["has_tools"]:
            continue
        if w.get("require_tools") and not features["has_tools"]:
            continue
        if features["has_images"] and not w.get("allow_images"):
            continue                                    # 图像请求一律不降级

        mx = w.get("max_input_tokens")
        if mx is not None and features["input_tokens"] > mx:
            continue
        mn = w.get("min_input_tokens")
        if mn is not None and features["input_tokens"] < mn:
            continue

        kws = w.get("keywords")
        if kws:
            low = features["text"].lower()
            if not any(k.lower() in low for k in kws):
                continue

        return r
    return sorted(routes, key=lambda x: x.get("priority", 999))[-1]


def classify_data(payload: dict, default: str = "internal") -> str:
    """数据分级（第 8 章）。
    规则：
      - 请求头 X-Data-Class 显式指定优先
      - 命中敏感关键词（客户/合同/报价/密码/身份证/手机号 等）→ confidential
      - 其余按默认
    """
    hdr = (payload.get("_data_class") or "").strip().lower()
    if hdr in ("public", "internal", "confidential"):
        return hdr
    text = (payload.get("_classify_text") or "").lower()
    sensitive = ["客户名单", "合同", "报价", "投标", "身份证", "银行卡", "密码",
                 "password", "secret", "api key", "未公开", "内部资料", "保密"]
    if any(s in text for s in sensitive):
        return "confidential"
    return default
