"""结构化模型输出的预算与截断检测。"""

from app.core.llm.llm import LLM
from app.core.llm.types import StandardResponse


DEFAULT_STRUCTURED_OUTPUT_TOKENS = 8192
MAX_STRUCTURED_OUTPUT_TOKENS = 65536
TRUNCATION_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "incomplete",
}


def configured_output_budget(model: LLM) -> int:
    """返回结构化输出的有效初始 token 预算。"""
    configured = getattr(model, "max_tokens", None)
    if isinstance(configured, int) and configured > 0:
        return configured
    return DEFAULT_STRUCTURED_OUTPUT_TOKENS


def response_was_truncated(
    response: StandardResponse,
    requested_tokens: int,
) -> bool:
    """根据供应商终止原因与实际用量判断响应是否耗尽预算。"""
    reason = str(getattr(response, "finish_reason", None) or "").strip().lower()
    if any(marker in reason for marker in TRUNCATION_REASONS):
        return True
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    return requested_tokens > 0 and completion_tokens >= requested_tokens


def expanded_output_budget(current: int) -> int:
    """扩大结构化输出预算，同时保留全局硬上限。"""
    return min(max(current * 2, 16384), MAX_STRUCTURED_OUTPUT_TOKENS)
