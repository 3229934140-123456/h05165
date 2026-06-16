import json
import threading
import time
from typing import Dict, List, Optional, Any
from collector import InMemoryStorage
from trace_reconstructor import Trace, TraceReconstructor


class TraceQueryService:
    """
    Trace 查询服务，提供类似 Jaeger Query 的功能。

    支持查询：
    1. 按 trace ID 查询完整调用树
    2. 按服务名搜索 trace
    3. 按操作名搜索 trace
    4. 按错误状态筛选 trace
    5. 组合多条件查询
    6. 输出 JSON 格式，方便 UI 或 API 对接
    """

    def __init__(self, storage: InMemoryStorage, reconstructor: TraceReconstructor):
        self.storage = storage
        self.reconstructor = reconstructor
        self._lock = threading.Lock()

    def get_trace_by_id(self, trace_id: str) -> Optional[Trace]:
        """
        按 trace ID 获取重组后的完整调用链。

        :param trace_id: 要查询的 trace ID
        :return: Trace 对象，如果不存在返回 None
        """
        spans = self.storage.get_trace(trace_id)
        if spans is None:
            pending_spans = self.reconstructor._pending_spans.get(trace_id)
            if pending_spans:
                spans = list(pending_spans)
        if spans is None:
            return None
        return Trace(trace_id, spans)

    def get_trace_json(self, trace_id: str, indent: Optional[int] = 2) -> Optional[str]:
        """
        按 trace ID 获取 JSON 格式的调用链。

        输出格式兼容 Jaeger UI 的核心字段：
        - traceID, spans, processes
        - 每个 span: spanID, operationName, startTime, duration, tags, logs, references
        """
        trace = self.get_trace_by_id(trace_id)
        if trace is None:
            return None
        return json.dumps(self._trace_to_jaeger_format(trace), indent=indent, default=str)

    def search_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        has_error: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        limit: int = 100,
    ) -> List[Trace]:
        """
        按多条件搜索 trace。

        :param service_name: 服务名（任意 span 包含该服务即匹配）
        :param operation_name: 操作名（任意 span 包含该操作即匹配）
        :param has_error: 是否只筛选有错误的 trace
        :param min_duration_ms: 最小耗时（毫秒）
        :param max_duration_ms: 最大耗时（毫秒）
        :param limit: 返回结果上限
        :return: 匹配条件的 Trace 列表
        """
        trace_ids = self.storage.search_traces(
            service_name=service_name,
            operation_name=operation_name,
            limit=limit * 10,
        )

        results: List[Trace] = []
        for trace_id in trace_ids:
            trace = self.get_trace_by_id(trace_id)
            if trace is None:
                continue

            if has_error is not None and trace.has_error() != has_error:
                continue

            duration_ms = trace.get_total_duration() * 1000
            if min_duration_ms is not None and duration_ms < min_duration_ms:
                continue
            if max_duration_ms is not None and duration_ms > max_duration_ms:
                continue

            results.append(trace)
            if len(results) >= limit:
                break

        return results

    def search_traces_json(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        has_error: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        limit: int = 100,
        indent: Optional[int] = 2,
    ) -> str:
        """按多条件搜索并输出 JSON。"""
        traces = self.search_traces(
            service_name=service_name,
            operation_name=operation_name,
            has_error=has_error,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            limit=limit,
        )
        jaeger_traces = [self._trace_to_jaeger_format(t) for t in traces]
        return json.dumps(jaeger_traces, indent=indent, default=str)

    def list_services(self) -> List[str]:
        """获取所有出现过的服务名。"""
        services: set = set()
        for trace_id in self.storage.get_all_trace_ids():
            spans = self.storage.get_trace(trace_id)
            if spans:
                for span in spans:
                    services.add(span.service_name)
        return sorted(list(services))

    def list_operations(self, service_name: Optional[str] = None) -> List[str]:
        """获取所有操作名，可按服务过滤。"""
        operations: set = set()
        for trace_id in self.storage.get_all_trace_ids():
            spans = self.storage.get_trace(trace_id)
            if spans:
                for span in spans:
                    if service_name is None or span.service_name == service_name:
                        operations.add(span.operation_name)
        return sorted(list(operations))

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        trace_ids = self.storage.get_all_trace_ids()
        total_spans = 0
        service_set: set = set()
        operation_set: set = set()
        error_count = 0

        for tid in trace_ids:
            spans = self.storage.get_trace(tid)
            if spans:
                total_spans += len(spans)
                for s in spans:
                    service_set.add(s.service_name)
                    operation_set.add(s.operation_name)
                    if s.status == "error":
                        error_count += 1

        return {
            "total_traces": len(trace_ids),
            "total_spans": total_spans,
            "unique_services": len(service_set),
            "unique_operations": len(operation_set),
            "error_spans": error_count,
        }

    def _trace_to_jaeger_format(self, trace: Trace) -> Dict[str, Any]:
        """
        将 Trace 转换为 Jaeger 兼容的 JSON 格式。

        核心字段参考 Jaeger API：
        https://www.jaegertracing.io/docs/1.50/apis/#trace
        """
        processes: Dict[str, Dict] = {}
        jaeger_spans = []
        service_counter = 0

        for span in trace.spans:
            if span.service_name not in processes:
                service_counter += 1
                process_id = f"p{service_counter}"
                processes[process_id] = {
                    "serviceName": span.service_name,
                    "tags": [],
                }
                span._jaeger_process_id = process_id

        for span in trace.spans:
            references = []
            if span.parent_span_id is not None:
                references.append(
                    {
                        "refType": "CHILD_OF",
                        "traceID": span.trace_id,
                        "spanID": span.parent_span_id,
                    }
                )

            tags = [
                {"key": k, "type": "string", "value": str(v)}
                for k, v in span.tags.items()
            ]

            logs = [
                {
                    "timestamp": int(log["timestamp"] * 1_000_000),
                    "fields": [
                        {"key": k, "type": "string", "value": str(v)}
                        for k, v in log["fields"].items()
                    ]
                    + [{"key": "event", "type": "string", "value": log["event"]}],
                }
                for log in span.logs
            ]

            jaeger_spans.append(
                {
                    "traceID": span.trace_id,
                    "spanID": span.span_id,
                    "operationName": span.operation_name,
                    "references": references,
                    "startTime": int((span.start_time or 0) * 1_000_000),
                    "duration": int((span.duration or 0) * 1_000_000),
                    "tags": tags,
                    "logs": logs,
                    "processID": getattr(span, "_jaeger_process_id", "p1"),
                    "warnings": None,
                }
            )

        return {
            "traceID": trace.trace_id,
            "spans": jaeger_spans,
            "processes": processes,
            "warnings": None,
            "isComplete": trace.is_complete,
            "summary": {
                "spanCount": trace.get_span_count(),
                "serviceCount": trace.get_service_count(),
                "totalDurationMs": round(trace.get_total_duration() * 1000, 3),
                "hasError": trace.has_error(),
                "orphanCount": trace.get_orphan_count(),
            },
        }

    def print_trace_tree(self, trace_id: str) -> None:
        """打印 trace 的调用树（调试用）。"""
        trace = self.get_trace_by_id(trace_id)
        if trace is None:
            print(f"Trace {trace_id} not found")
            return
        trace.print_tree()
