import uuid
import threading
from typing import Optional, Dict, Any


class TraceContext:
    """
    Trace 上下文，用于在服务间传播追踪信息。
    遵循 W3C Trace Context 规范的核心思想。
    """

    TRACE_ID_HEADER = "x-trace-id"
    SPAN_ID_HEADER = "x-span-id"
    SAMPLED_HEADER = "x-sampled"

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None,
        sampled: bool = True,
        trace_flags: int = 1,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.sampled = sampled
        self.trace_flags = trace_flags

    @staticmethod
    def generate_trace_id() -> str:
        """生成唯一的 trace ID，128 位 UUID 格式。"""
        return uuid.uuid4().hex

    @staticmethod
    def generate_span_id() -> str:
        """生成唯一的 span ID，64 位格式。"""
        return uuid.uuid4().hex[:16]

    @staticmethod
    def new_root(sampled: bool = True) -> "TraceContext":
        """创建新的根上下文（入口请求）。"""
        return TraceContext(
            trace_id=TraceContext.generate_trace_id(),
            span_id=TraceContext.generate_span_id(),
            parent_span_id=None,
            sampled=sampled,
        )

    @staticmethod
    def from_headers(headers: Dict[str, Any]) -> Optional["TraceContext"]:
        """
        从 HTTP 头部解析 trace 上下文。
        用于服务端接收请求时恢复上下文。
        """
        trace_id = headers.get(TraceContext.TRACE_ID_HEADER)
        span_id = headers.get(TraceContext.SPAN_ID_HEADER)
        sampled_header = headers.get(TraceContext.SAMPLED_HEADER)

        if not trace_id or not span_id:
            return None

        sampled = True
        if sampled_header is not None:
            if isinstance(sampled_header, str):
                sampled = sampled_header.lower() in ("1", "true", "yes")
            else:
                sampled = bool(sampled_header)

        return TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            sampled=sampled,
        )

    def to_headers(self) -> Dict[str, str]:
        """
        将上下文转换为 HTTP 头部。
        用于客户端发起请求时传播上下文。
        """
        return {
            TraceContext.TRACE_ID_HEADER: self.trace_id,
            TraceContext.SPAN_ID_HEADER: self.span_id,
            TraceContext.SAMPLED_HEADER: "1" if self.sampled else "0",
        }

    def new_child(self) -> "TraceContext":
        """
        创建子上下文，用于同一服务内的下一个操作。
        保持 trace_id 和 sampled 不变，生成新的 span_id。
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=TraceContext.generate_span_id(),
            parent_span_id=self.span_id,
            sampled=self.sampled,
        )

    def __repr__(self) -> str:
        return (
            f"TraceContext(trace_id={self.trace_id}, "
            f"span_id={self.span_id}, "
            f"parent_span_id={self.parent_span_id}, "
            f"sampled={self.sampled})"
        )


class ContextManager:
    """
    线程本地的上下文管理器。
    用于在当前线程中存储和获取 trace 上下文。
    """

    _local = threading.local()

    @staticmethod
    def set_context(context: TraceContext) -> None:
        """设置当前线程的 trace 上下文。"""
        ContextManager._local.context = context

    @staticmethod
    def get_context() -> Optional[TraceContext]:
        """获取当前线程的 trace 上下文。"""
        return getattr(ContextManager._local, "context", None)

    @staticmethod
    def clear_context() -> None:
        """清除当前线程的 trace 上下文。"""
        if hasattr(ContextManager._local, "context"):
            del ContextManager._local.context
