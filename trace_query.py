import json
import os
import threading
import time
from typing import Dict, List, Optional, Any, Tuple

from collector import InMemoryStorage
from span import Span
from trace_reconstructor import Trace, TraceReconstructor


class TraceQueryService:
    """
    Trace 查询服务，提供类似 Jaeger Query 的能力。

    支持：
    - 按 trace ID 精确查询
    - 按时间范围、服务名、操作名、错误状态、最小耗时组合过滤
    - 分页（offset + limit）
    - Jaeger 兼容 JSON 输出（进程去重，span 共享 processID）
    - 批量导入 / 导出 JSON（离线样本回放）
    """

    SORT_START_TIME_DESC = "start_time_desc"
    SORT_START_TIME_ASC = "start_time_asc"
    SORT_DURATION_DESC = "duration_desc"

    def __init__(self, storage: InMemoryStorage, reconstructor: TraceReconstructor):
        self.storage = storage
        self.reconstructor = reconstructor
        self._lock = threading.Lock()

    # -------- 基础查询 --------

    def get_trace_by_id(self, trace_id: str) -> Optional[Trace]:
        spans = self.storage.get_trace(trace_id)
        if spans is None:
            pending_spans = self.reconstructor._pending_spans.get(trace_id)
            if pending_spans:
                spans = list(pending_spans)
        if spans is None:
            return None
        return Trace(trace_id, spans)

    def get_trace_json(self, trace_id: str, indent: Optional[int] = 2) -> Optional[str]:
        trace = self.get_trace_by_id(trace_id)
        if trace is None:
            return None
        return json.dumps(self._trace_to_jaeger_format(trace), indent=indent, default=str)

    # -------- 高级搜索（Jaeger Query 风格） --------

    def search_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        has_error: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sort: str = SORT_START_TIME_DESC,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Trace], int]:
        """
        多条件组合搜索 trace，支持分页与排序。

        :param service_name: 服务名（任意 span 命中即匹配）
        :param operation_name: 操作名（任意 span 命中即匹配）
        :param has_error: 是否只筛有错误的 trace
        :param min_duration_ms: 最小总耗时（毫秒）
        :param max_duration_ms: 最大总耗时（毫秒）
        :param start_time: trace 起始时间下限（秒，Unix 时间戳）
        :param end_time: trace 起始时间上限（秒，Unix 时间戳）
        :param sort: 排序方式，见 SORT_* 常量
        :param offset: 分页起始偏移
        :param limit: 每页数量
        :return: (匹配的 Trace 分页结果, 总匹配数)
        """
        candidate_ids = self.storage.search_traces(
            service_name=service_name,
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            limit=None,
        )

        matched: List[Tuple[Trace, float, float]] = []
        for trace_id in candidate_ids:
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
            start_ts = self.storage.get_trace_start_time(trace_id) or 0.0
            matched.append((trace, start_ts, duration_ms))

        total = len(matched)

        if sort == self.SORT_START_TIME_DESC:
            matched.sort(key=lambda x: x[1], reverse=True)
        elif sort == self.SORT_START_TIME_ASC:
            matched.sort(key=lambda x: x[1])
        elif sort == self.SORT_DURATION_DESC:
            matched.sort(key=lambda x: x[2], reverse=True)

        page = [t for t, _, _ in matched[offset : offset + limit]]
        return page, total

    def search_traces_json(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        has_error: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sort: str = SORT_START_TIME_DESC,
        offset: int = 0,
        limit: int = 100,
        indent: Optional[int] = 2,
    ) -> str:
        traces, total = self.search_traces(
            service_name=service_name,
            operation_name=operation_name,
            has_error=has_error,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            start_time=start_time,
            end_time=end_time,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        return json.dumps(
            {
                "total": total,
                "offset": offset,
                "limit": limit,
                "traces": [self._trace_to_jaeger_format(t) for t in traces],
            },
            indent=indent,
            default=str,
        )

    # -------- 元数据 & 统计 --------

    def list_services(self) -> List[str]:
        services: set = set()
        for trace_id in self.storage.get_all_trace_ids():
            spans = self.storage.get_trace(trace_id)
            if spans:
                for span in spans:
                    services.add(span.service_name)
        return sorted(list(services))

    def list_operations(self, service_name: Optional[str] = None) -> List[str]:
        operations: set = set()
        for trace_id in self.storage.get_all_trace_ids():
            spans = self.storage.get_trace(trace_id)
            if spans:
                for span in spans:
                    if service_name is None or span.service_name == service_name:
                        operations.add(span.operation_name)
        return sorted(list(operations))

    def get_stats(self) -> Dict[str, Any]:
        trace_ids = self.storage.get_all_trace_ids()
        total_spans = 0
        service_set: set = set()
        operation_set: set = set()
        error_count = 0
        error_trace_count = 0

        for tid in trace_ids:
            spans = self.storage.get_trace(tid)
            if spans:
                total_spans += len(spans)
                has_err = False
                for s in spans:
                    service_set.add(s.service_name)
                    operation_set.add(s.operation_name)
                    if s.status == "error":
                        error_count += 1
                        has_err = True
                if has_err:
                    error_trace_count += 1

        return {
            "total_traces": len(trace_ids),
            "error_traces": error_trace_count,
            "total_spans": total_spans,
            "unique_services": len(service_set),
            "unique_operations": len(operation_set),
            "error_spans": error_count,
        }

    # -------- Jaeger 兼容 JSON（进程去重） --------

    def _trace_to_jaeger_format(self, trace: Trace) -> Dict[str, Any]:
        """
        同一服务在一条 trace 里只生成一个 process 条目，
        多个 span 共享同一个 processID，避免 UI 重复渲染。
        """
        processes: Dict[str, Dict] = {}
        service_to_pid: Dict[str, str] = {}

        for span in trace.spans:
            svc = span.service_name or "unknown"
            if svc not in service_to_pid:
                pid = f"p{len(service_to_pid) + 1}"
                service_to_pid[svc] = pid
                processes[pid] = {
                    "serviceName": svc,
                    "tags": [
                        {"key": "service.name", "type": "string", "value": svc},
                    ],
                }

        jaeger_spans = []
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
            if span.status == "error":
                tags.append({"key": "error", "type": "bool", "value": True})
                if span.status_message:
                    tags.append(
                        {"key": "error.message", "type": "string", "value": span.status_message}
                    )

            logs = [
                {
                    "timestamp": int(log["timestamp"] * 1_000_000),
                    "fields": [
                        {"key": "event", "type": "string", "value": log["event"]},
                    ]
                    + [
                        {"key": k, "type": "string", "value": str(v)}
                        for k, v in log.get("fields", {}).items()
                    ],
                }
                for log in span.logs
            ]

            pid = service_to_pid.get(span.service_name or "unknown", "p1")
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
                    "processID": pid,
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
                "startTime": int(
                    (min((s.start_time or 0) for s in trace.spans) if trace.spans else 0)
                    * 1_000_000
                ),
            },
        }

    # -------- 批量导入 / 导出 --------

    def import_spans_from_json(self, file_path: str) -> Tuple[int, int]:
        """
        从 JSON 文件批量导入 span，灌到 storage 和 reconstructor。

        支持两种格式：
        1. span 数组（每个元素是 Span.to_dict() 的结果）
        2. {"spans": [...]} 包装格式

        :return: (成功导入 span 数, 涉及 trace 数)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "spans" in data:
            raw_spans = data["spans"]
        elif isinstance(data, list):
            raw_spans = data
        else:
            raise ValueError("JSON 格式不支持，需要是 span 数组或 {'spans': [...]}")

        spans: List[Span] = []
        for raw in raw_spans:
            try:
                spans.append(Span.from_dict(raw))
            except Exception:
                continue

        trace_ids: set = set()
        for span in spans:
            trace_ids.add(span.trace_id)
            self.storage.save_trace(span.trace_id, [span])

        self.reconstructor.add_spans(spans)
        return len(spans), len(trace_ids)

    def export_trace_to_json(
        self, trace_id: str, file_path: Optional[str] = None, indent: int = 2
    ) -> Optional[str]:
        """
        导出单条 trace 的完整调用树为 Jaeger 兼容 JSON。

        :param trace_id: 要导出的 trace ID
        :param file_path: 写入文件路径，None 则只返回 JSON 字符串
        :return: JSON 字符串，trace 不存在返回 None
        """
        trace = self.get_trace_by_id(trace_id)
        if trace is None:
            return None

        payload = self._trace_to_jaeger_format(trace)
        content = json.dumps(payload, indent=indent, default=str)

        if file_path is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def export_traces_to_json(
        self,
        trace_ids: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        indent: int = 2,
        **search_kwargs,
    ) -> str:
        """
        批量导出 trace。

        :param trace_ids: 显式指定 trace ID 列表；为 None 时使用 search_kwargs 搜索
        :param file_path: 写入文件路径，None 则只返回 JSON
        :param search_kwargs: 若 trace_ids 为 None，透传给 search_traces
        """
        if trace_ids is None:
            traces, _ = self.search_traces(limit=10_000, **search_kwargs)
        else:
            traces = []
            for tid in trace_ids:
                t = self.get_trace_by_id(tid)
                if t is not None:
                    traces.append(t)

        payload = {
            "exported_at": int(time.time()),
            "count": len(traces),
            "traces": [self._trace_to_jaeger_format(t) for t in traces],
        }
        content = json.dumps(payload, indent=indent, default=str)

        if file_path is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def export_spans_to_json(
        self,
        trace_ids: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        indent: int = 2,
    ) -> str:
        """
        导出原始 span 数组（Span.to_dict 格式），便于再次 import 回放。
        """
        if trace_ids is None:
            trace_ids = self.storage.get_all_trace_ids()

        all_spans: List[Dict] = []
        for tid in trace_ids:
            spans = self.storage.get_trace(tid)
            if spans:
                all_spans.extend(s.to_dict() for s in spans)

        payload = {"spans": all_spans, "count": len(all_spans)}
        content = json.dumps(payload, indent=indent, default=str)

        if file_path is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    # -------- 调试 --------

    def print_trace_tree(self, trace_id: str) -> None:
        trace = self.get_trace_by_id(trace_id)
        if trace is None:
            print(f"Trace {trace_id} not found")
            return
        trace.print_tree()
