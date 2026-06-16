import time
import uuid
from typing import Optional, Dict, Any, List
from context_propagation import TraceContext, ContextManager


class SpanReference:
    """Span 引用，用于表示父子关系或关联关系。"""

    CHILD_OF = "CHILD_OF"
    FOLLOWS_FROM = "FOLLOWS_FROM"

    def __init__(self, ref_type: str, trace_id: str, span_id: str):
        self.ref_type = ref_type
        self.trace_id = trace_id
        self.span_id = span_id

    def __repr__(self) -> str:
        return f"SpanReference({self.ref_type}, trace_id={self.trace_id}, span_id={self.span_id})"


class Span:
    """
    Span 表示追踪中的一个操作单元。
    包含操作名称、时间戳、属性、事件和父子关系。
    """

    def __init__(
        self,
        operation_name: str,
        context: TraceContext,
        parent_span_id: Optional[str] = None,
        start_time: Optional[float] = None,
        service_name: str = "unknown",
    ):
        self.operation_name = operation_name
        self.context = context
        self.trace_id = context.trace_id
        self.span_id = context.span_id
        self.parent_span_id = parent_span_id or context.parent_span_id
        self.start_time = start_time or time.time()
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.service_name = service_name
        self.tags: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.status: str = "ok"
        self.status_message: Optional[str] = None

    def set_tag(self, key: str, value: Any) -> "Span":
        """设置 span 的标签属性。"""
        self.tags[key] = value
        return self

    def log(self, event: str, timestamp: Optional[float] = None, **fields: Any) -> "Span":
        """记录一个事件日志。"""
        self.logs.append(
            {
                "event": event,
                "timestamp": timestamp or time.time(),
                "fields": fields,
            }
        )
        return self

    def set_error(self, message: str = "") -> "Span":
        """标记 span 为错误状态。"""
        self.status = "error"
        self.status_message = message
        self.tags["error"] = True
        return self

    def finish(self, end_time: Optional[float] = None) -> None:
        """结束 span，计算持续时间。"""
        self.end_time = end_time or time.time()
        self.duration = self.end_time - self.start_time

    def is_finished(self) -> bool:
        """检查 span 是否已结束。"""
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        """将 span 转换为字典格式，便于序列化和上报。"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
            "status_message": self.status_message,
            "sampled": self.context.sampled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """从字典恢复 span 对象。"""
        context = TraceContext(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            sampled=data.get("sampled", True),
        )
        span = cls(
            operation_name=data["operation_name"],
            context=context,
            parent_span_id=data.get("parent_span_id"),
            start_time=data.get("start_time"),
            service_name=data.get("service_name", "unknown"),
        )
        span.end_time = data.get("end_time")
        span.duration = data.get("duration")
        span.tags = data.get("tags", {})
        span.logs = data.get("logs", [])
        span.status = data.get("status", "ok")
        span.status_message = data.get("status_message")
        return span

    def __repr__(self) -> str:
        return (
            f"Span(operation={self.operation_name}, "
            f"trace_id={self.trace_id}, "
            f"span_id={self.span_id}, "
            f"parent={self.parent_span_id}, "
            f"duration={self.duration}s)"
        )


class Tracer:
    """
    Tracer 是 span 的工厂和管理器。
    负责创建 span、管理活动 span 栈、以及将完成的 span 发送给收集器。
    """

    def __init__(self, service_name: str, collector=None, sampler=None):
        self.service_name = service_name
        self.collector = collector
        self.sampler = sampler
        self._active_spans: List[Span] = []

    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        context: Optional[TraceContext] = None,
        start_time: Optional[float] = None,
    ) -> Span:
        """
        启动一个新的 span。

        优先级：
        1. 显式指定的 parent span
        2. 显式指定的 context
        3. 当前线程的活动上下文
        4. 创建新的根上下文
        """
        if context is None and parent is not None:
            context = parent.context.new_child()
        elif context is not None:
            context = context.new_child()
        else:
            current_context = ContextManager.get_context()
            if current_context is not None:
                context = current_context.new_child()
            else:
                sampled = True
                if self.sampler is not None:
                    sampled = self.sampler.should_sample()
                context = TraceContext.new_root(sampled=sampled)

        parent_span_id = None
        if parent is not None:
            parent_span_id = parent.span_id
        elif context.parent_span_id is not None:
            parent_span_id = context.parent_span_id

        span = Span(
            operation_name=operation_name,
            context=context,
            parent_span_id=parent_span_id,
            start_time=start_time,
            service_name=self.service_name,
        )

        self._active_spans.append(span)
        ContextManager.set_context(context)

        return span

    def start_active_span(self, operation_name: str, **kwargs) -> "SpanScope":
        """
        启动一个 span 并将其设为活动 span，返回一个 Scope 用于自动管理。
        使用 with 语句可以自动结束 span。
        """
        span = self.start_span(operation_name, **kwargs)
        return SpanScope(span, self)

    def finish_span(self, span: Span, end_time: Optional[float] = None) -> None:
        """
        结束 span 并上报给收集器。
        只有被采样的 span 才会被上报。
        """
        if not span.is_finished():
            span.finish(end_time)

        if span.context.sampled and self.collector is not None:
            self.collector.collect(span)

        if span in self._active_spans:
            self._active_spans.remove(span)

        if self._active_spans:
            ContextManager.set_context(self._active_spans[-1].context)
        else:
            ContextManager.clear_context()

    def extract(self, headers: Dict[str, Any]) -> Optional[TraceContext]:
        """从 HTTP 头部提取 trace 上下文。"""
        return TraceContext.from_headers(headers)

    def inject(self, context: TraceContext, headers: Dict[str, Any]) -> Dict[str, Any]:
        """将 trace 上下文注入到 HTTP 头部。"""
        headers.update(context.to_headers())
        return headers

    def get_active_span(self) -> Optional[Span]:
        """获取当前活动的 span。"""
        if self._active_spans:
            return self._active_spans[-1]
        return None


class SpanScope:
    """
    Span 的作用域管理器，使用 with 语句自动管理 span 的生命周期。
    """

    def __init__(self, span: Span, tracer: Tracer):
        self.span = span
        self.tracer = tracer
        self._closed = False

    def __enter__(self) -> Span:
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._closed:
            if exc_val is not None:
                self.span.set_error(str(exc_val))
            self.tracer.finish_span(self.span)
            self._closed = True

    def close(self) -> None:
        """手动关闭 scope。"""
        if not self._closed:
            self.tracer.finish_span(self.span)
            self._closed = True
