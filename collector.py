import threading
import time
import queue
from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict
from span import Span


class SpanBuffer:
    """
    Span 缓冲区，用于按 trace ID 分组存储 span。
    处理乱序到达的 span，并检测 trace 是否完整。
    """

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans: Dict[str, Span] = {}
        self.span_ids: Set[str] = set()
        self.parent_span_ids: Set[str] = set()
        self.received_at = time.time()
        self.last_span_at = time.time()
        self.is_complete = False

    def add_span(self, span: Span) -> None:
        """添加一个 span 到缓冲区。"""
        span_id = span.span_id
        if span_id in self.spans:
            return

        self.spans[span_id] = span
        self.span_ids.add(span_id)
        self.last_span_at = time.time()

        if span.parent_span_id is not None:
            self.parent_span_ids.add(span.parent_span_id)

        self._check_completeness()

    def _check_completeness(self) -> None:
        """
        检查 trace 是否完整。
        完整的条件：所有 parent_span_id 都存在于 span_ids 中，
        且存在一个根 span（parent_span_id 为 None）。
        """
        has_root = any(span.parent_span_id is None for span in self.spans.values())
        all_parents_present = self.parent_span_ids.issubset(self.span_ids)

        if has_root and all_parents_present:
            self.is_complete = True

    def get_missing_span_ids(self) -> Set[str]:
        """获取缺失的 span ID。"""
        return self.parent_span_ids - self.span_ids

    def get_spans(self) -> List[Span]:
        """获取所有 span。"""
        return list(self.spans.values())

    def get_span_count(self) -> int:
        """获取 span 数量。"""
        return len(self.spans)

    def get_age(self) -> float:
        """获取缓冲区的年龄（秒）。"""
        return time.time() - self.received_at

    def get_idle_time(self) -> float:
        """获取距离上次接收 span 的时间（秒）。"""
        return time.time() - self.last_span_at

    def __repr__(self) -> str:
        return (
            f"SpanBuffer(trace_id={self.trace_id}, "
            f"spans={len(self.spans)}, "
            f"complete={self.is_complete}, "
            f"age={self.get_age():.1f}s)"
        )


class SpanCollector:
    """
    Span 收集器，负责接收、存储和管理 span。

    主要功能：
    1. 接收服务上报的 span（支持乱序）
    2. 按 trace ID 分组存储
    3. 批量处理 span 提高效率
    4. 检测 trace 完整性并通知下游
    5. 超时清理防止内存泄漏
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        max_trace_age: float = 300.0,
        max_idle_time: float = 60.0,
        on_trace_complete: Optional[Callable[[str, List[Span]], None]] = None,
    ):
        """
        :param max_queue_size: 内存队列最大大小
        :param batch_size: 批量处理大小
        :param flush_interval: 强制刷新间隔（秒）
        :param max_trace_age: trace 最大存活时间（秒）
        :param max_idle_time: trace 最大空闲时间（秒）
        :param on_trace_complete: trace 完成时的回调函数
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_trace_age = max_trace_age
        self.max_idle_time = max_idle_time
        self.on_trace_complete = on_trace_complete

        self._queue: "queue.Queue[Span]" = queue.Queue(maxsize=max_queue_size)
        self._trace_buffers: Dict[str, SpanBuffer] = {}
        self._buffers_lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._collected_spans = 0
        self._completed_traces = 0
        self._dropped_spans = 0

        self._last_flush = time.time()

    def start(self) -> None:
        """启动收集器的后台线程。"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()

    def stop(self) -> None:
        """停止收集器。"""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5.0)

    def collect(self, span: Span) -> bool:
        """
        收集一个 span。

        :return: True 表示成功入队，False 表示队列已满被丢弃
        """
        try:
            self._queue.put_nowait(span)
            return True
        except queue.Full:
            self._dropped_spans += 1
            return False

    def collect_batch(self, spans: List[Span]) -> int:
        """
        批量收集 span。

        :return: 成功入队的 span 数量
        """
        success = 0
        for span in spans:
            if self.collect(span):
                success += 1
        return success

    def _worker_loop(self) -> None:
        """后台处理线程，从队列中取出 span 并处理。"""
        batch: List[Span] = []

        while not self._stop_event.is_set():
            try:
                span = self._queue.get(timeout=0.1)
                batch.append(span)

                if len(batch) >= self.batch_size or (
                    time.time() - self._last_flush > self.flush_interval and batch
                ):
                    self._process_batch(batch)
                    batch = []
                    self._last_flush = time.time()

            except queue.Empty:
                if batch and (time.time() - self._last_flush > self.flush_interval):
                    self._process_batch(batch)
                    batch = []
                    self._last_flush = time.time()

        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[Span]) -> None:
        """处理一批 span，按 trace ID 分组。"""
        with self._buffers_lock:
            for span in batch:
                trace_id = span.trace_id
                if trace_id not in self._trace_buffers:
                    self._trace_buffers[trace_id] = SpanBuffer(trace_id)

                buffer = self._trace_buffers[trace_id]
                buffer.add_span(span)
                self._collected_spans += 1

                if buffer.is_complete and self.on_trace_complete is not None:
                    self._notify_trace_complete(trace_id, buffer)

    def _notify_trace_complete(self, trace_id: str, buffer: SpanBuffer) -> None:
        """通知 trace 完成。"""
        try:
            self._completed_traces += 1
            if self.on_trace_complete is not None:
                self.on_trace_complete(trace_id, buffer.get_spans())
        except Exception:
            pass

    def _cleanup_loop(self) -> None:
        """定期清理过期的 trace 缓冲区。"""
        while not self._stop_event.is_set():
            time.sleep(5.0)
            self._cleanup_expired_traces()

    def _cleanup_expired_traces(self) -> None:
        """清理过期的 trace 缓冲区。"""
        with self._buffers_lock:
            expired_traces = []
            for trace_id, buffer in self._trace_buffers.items():
                if buffer.get_age() > self.max_trace_age or (
                    buffer.get_idle_time() > self.max_idle_time and not buffer.is_complete
                ):
                    expired_traces.append(trace_id)

            for trace_id in expired_traces:
                if self.on_trace_complete is not None:
                    buffer = self._trace_buffers[trace_id]
                    self.on_trace_complete(trace_id, buffer.get_spans())
                del self._trace_buffers[trace_id]

    def get_trace_buffer(self, trace_id: str) -> Optional[SpanBuffer]:
        """获取指定 trace 的缓冲区。"""
        with self._buffers_lock:
            return self._trace_buffers.get(trace_id)

    def get_trace_spans(self, trace_id: str) -> Optional[List[Span]]:
        """获取指定 trace 的所有 span。"""
        buffer = self.get_trace_buffer(trace_id)
        if buffer is not None:
            return buffer.get_spans()
        return None

    def get_all_trace_ids(self) -> List[str]:
        """获取所有 trace ID。"""
        with self._buffers_lock:
            return list(self._trace_buffers.keys())

    def get_stats(self) -> Dict[str, int]:
        """获取收集器统计信息。"""
        with self._buffers_lock:
            active_traces = len(self._trace_buffers)

        return {
            "collected_spans": self._collected_spans,
            "completed_traces": self._completed_traces,
            "dropped_spans": self._dropped_spans,
            "active_traces": active_traces,
            "queue_size": self._queue.qsize(),
        }

    def flush(self) -> None:
        """强制刷新队列中的所有 span。"""
        batch = []
        while not self._queue.empty():
            try:
                span = self._queue.get_nowait()
                batch.append(span)
            except queue.Empty:
                break
        if batch:
            self._process_batch(batch)

    def __enter__(self) -> "SpanCollector":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class InMemoryStorage:
    """
    内存存储，用于持久化完整的 trace 数据。
    在实际生产环境中，这会替换为 Cassandra、Elasticsearch 等。
    """

    def __init__(self):
        self._traces: Dict[str, List[Span]] = {}
        self._lock = threading.Lock()

    def save_trace(self, trace_id: str, spans: List[Span]) -> None:
        """保存一个完整的 trace。"""
        with self._lock:
            if trace_id in self._traces:
                existing_ids = {s.span_id for s in self._traces[trace_id]}
                for span in spans:
                    if span.span_id not in existing_ids:
                        self._traces[trace_id].append(span)
            else:
                self._traces[trace_id] = list(spans)

    def get_trace(self, trace_id: str) -> Optional[List[Span]]:
        """获取一个 trace。"""
        with self._lock:
            spans = self._traces.get(trace_id)
            return list(spans) if spans is not None else None

    def get_all_trace_ids(self) -> List[str]:
        """获取所有 trace ID。"""
        with self._lock:
            return list(self._traces.keys())

    def search_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        """按条件搜索 trace。"""
        results = []
        with self._lock:
            for trace_id, spans in self._traces.items():
                match = True
                if service_name is not None:
                    match = any(s.service_name == service_name for s in spans)
                if match and operation_name is not None:
                    match = any(s.operation_name == operation_name for s in spans)
                if match:
                    results.append(trace_id)
                    if len(results) >= limit:
                        break
        return results

    def clear(self) -> None:
        """清空所有数据。"""
        with self._lock:
            self._traces.clear()
