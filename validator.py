"""校验层 + 升级重跑（方案第 4.3 / 4.4 章）

核心原则（第 11.1 章）：
  内容不合格 → 不原地重试（同模型重跑大概率还是错），直接升级到上一层模型重跑。

校验项（尽量用规则，毫秒级，省一次模型往返）：
  1. 非空：输出为空且无 tool_calls → 失败（免费小模型常见"只思考不输出"）
  2. 截断：finish_reason=length 且内容异常短 → 警告（不算失败）
  3. JSON 契约：请求要求 response_format=json 时，必须能解析
  4. 语言匹配：中文提问却输出纯 ASCII 长文本 → 可疑（弱校验，仅记录）
  5. 长度合理性：长输入摘要类任务，输出过短（< 20 字）→ 失败
"""
import json
import logging
import re

log = logging.getLogger("router.validator")

CN_RE = re.compile(r"[\u4e00-\u9fff]")


class Validator:
    def __init__(self):
        self.stats = {"checked": 0, "failed": 0, "escalated": 0}

    def check(self, payload: dict, response: dict, route_name: str) -> tuple[bool, str]:
        self.stats["checked"] += 1
        try:
            choice = (response.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls")
            finish = choice.get("finish_reason")

            # 1. 非空（有 tool_calls 视为有效）
            if not content and not tool_calls:
                self.stats["failed"] += 1
                return False, "输出为空（免费模型常见：token 被 reasoning 吃光）"

            # 2. JSON 契约
            want_json = (payload.get("response_format") or {}).get("type") == "json_object"
            if want_json and content:
                try:
                    json.loads(content)
                except Exception:
                    self.stats["failed"] += 1
                    return False, "要求 JSON 但输出无法解析"

            # 3. 长输入摘要类输出过短
            in_chars = sum(len(str(m.get("content") or "")) for m in (payload.get("messages") or []))
            if in_chars > 2000 and len(content) < 20 and not tool_calls:
                self.stats["failed"] += 1
                return False, f"长输入({in_chars}字)但输出仅 {len(content)} 字，疑似敷衍"

            # 4. 截断（仅记录，不判失败）
            if finish == "length" and len(content) < 30:
                log.debug("输出被 max_tokens 截断（in=%d字）", in_chars)

            return True, ""
        except Exception as e:
            self.stats["failed"] += 1
            return False, f"校验异常: {e}"

    def note_escalation(self):
        self.stats["escalated"] += 1


VALIDATOR = Validator()

# 升级链：免费层不合格时重跑用的模型（最多升 1 级，避免烧 token）
ESCALATION_CHAIN = [["deepseek", "deepseek-v4-flash"]]
