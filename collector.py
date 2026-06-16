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

    完成策略（更稳健）：
    1. 检测到所有父子关系都完整（has_root + all_parents_present）时，
       进入 WAITING_CHILDREN 状态，不立即触发完成通知
    2. 等待 post_root_idle_time 秒没有新 span 到达，认为链路稳定
    3. 或外部显式调用 flush_trace 强制完成
    4. 之后才触发 on_trace_complete 回调
    5. 完成后若有迟到的子 span，仍可追加并重新触发更新通知
    """

    STATE_PENDING = "pending"
    STATE_WAITING_CHILDREN = "waiting_children"
    STATE_COMPLETE = "complete"

    def __init__(self, trace_id: str, post_root_idle_time: float = 2.0):
        self.trace_id = trace_id
        self.post_root_idle_time = post_root_idle_time
        self.spans: Dict[str, Span] = {}
        self.span_ids: Set[str] = set()
        self.parent_span_ids: Set[str] = set()
        self.received_at = time.time()
        self.last_span_at = time.time()
        self.state = self.STATE_PENDING
        self._all_parents_present = False
        self._has_root = False
        self._completed_notified = False

    def add_span(self, span: Span) -> None:
        """添加一个 span 到缓冲区。"""
        span_id = span.span_id
        is_new = span_id not in self.spans

        if is_new:
            self.spans[span_id] = span
            self.span_ids.add(span_id)

        self.last_span_at = time.time()

        if span.parent_span_id is not None:
            self.parent_span_ids.add(span.parent_span_id)

        self._update_state()

    def _update_state(self) -> None:
        """更新 buffer 状态。"""
        self._has_root = any(
            s.parent_span_id is None for s in self.spans.values()
        )
        self._all_parents_present = self.parent_span_ids.issubset(self.span_ids)

        if self.state == self.STATE_PENDING:
            if self._has_root and self._all_parents_present:
                self.state = self.STATE_WAITING_CHILDREN

        elif self.state == self.STATE_WAITING_CHILDREN:
            if not (self._has_root and self._all_parents_present):
                self.state = self.STATE_PENDING

    def can_complete(self, force_flush: bool = False) -> bool:
        """
        判断是否可以触发完成回调。

        :param force_flush: 是否外部强制 flush
        :return: True 表示可以触发完成回调
        """
        if self.state == self.STATE_COMPLETE:
            return False

        if self.state == self.STATE_PENDING:
            if force_flush:
                return True
            return False

        if self.state == self.STATE_WAITING_CHILDREN:
            if force_flush:
                return True
            if self.get_idle_time() >= self.post_root_idle_time:
                return True
            return False

        return False

    def mark_notified(self) -> None:
        """标记为已通知完成。"""
        self.state = self.STATE_COMPLETE
        self._completed_notified = True

    def is_structurally_complete(self) -> bool:
        """父子结构是否完整（有根且所有父 span 都到了）。"""
        return self._has_root and self._all_parents_present

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

    @property
    def is_complete(self) -> bool:
        """兼容旧接口：是否已通知完成。"""
        return self.state == self.STATE_COMPLETE

    def __repr__(self) -> str:
        return (
            f"SpanBuffer(trace_id={self.trace_id}, "
            f"spans={len(self.spans)}, "
            f"state={self.state}, "
            f"structurally_complete={self.is_structurally_complete()}, "
            f"idle={self.get_idle_time():.1f}s)"
        )


class SpanCollector:
    """
    Span 收集器，负责接收、存储和管理 span。

    主要功能：
    1. 接收服务上报的 span（支持乱序）
    2. 按 trace ID 分组存储
    3. 批量处理 span 提高效率
    4. 稳健的完成检测：根 span 到达后等待子 span，空闲/flush 后才通知
    5. 超时清理防止内存泄漏
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        max_trace_age: float = 300.0,
        max_idle_time: float = 60.0,
        post_root_idle_time: float = 2.0,
        completion_check_interval: float = 0.5,
        on_trace_complete: Optional[Callable[[str, List[Span]], None]] = None,
    ):
        """
        :param max_queue_size: 内存队列最大大小
        :param batch_size: 批量处理大小
        :param flush_interval: 强制刷新间隔（秒）
        :param max_trace_age: trace 最大存活时间（秒）
        :param max_idle_time: trace 最大空闲时间（秒）
        :param post_root_idle_time: 根 span 到达后等待子 span 的空闲时间（秒）
        :param completion_check_interval: 完成检测线程的检查间隔（秒）
        :param on_trace_complete: trace 完成时的回调函数
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_trace_age = max_trace_age
        self.max_idle_time = max_idle_time
        self.post_root_idle_time = post_root_idle_time
        self.completion_check_interval = completion_check_interval
        self.on_trace_complete = on_trace_complete

        self._queue: "queue.Queue[Span]" = queue.Queue(maxsize=max_queue_size)
        self._trace_buffers: Dict[str, SpanBuffer] = {}
        self._buffers_lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._completion_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._collected_spans = 0
        self._completed_traces = 0
        self._dropped_spans = 0
        self._notified_traces: Set[str] = set()

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

        if self._completion_thread is None or not self._completion_thread.is_alive():
            self._completion_thread = threading.Thread(
                target=self._completion_check_loop, daemon=True
            )
            self._completion_thread.start()

    def stop(self) -> None:
        """停止收集器。"""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5.0)
        if self._completion_thread is not None:
            self._completion_thread.join(timeout=5.0)

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
                    self._trace_buffers[trace_id] = SpanBuffer(
                        trace_id,
                        post_root_idle_time=self.post_root_idle_time,
                    )

                buffer = self._trace_buffers[trace_id]
                buffer.add_span(span)
                self._collected_spans += 1

                if trace_id in self._notified_traces:
                    self._notify_trace_updated(trace_id, buffer)

    def _notify_trace_complete(self, trace_id: str, buffer: SpanBuffer) -> None:
        """通知 trace 完成。"""
        try:
            self._completed_traces += 1
            self._notified_traces.add(trace_id)
            buffer.mark_notified()
            if self.on_trace_complete is not None:
                self.on_trace_complete(trace_id, buffer.get_spans())
        except Exception:
            pass

    def _notify_trace_updated(self, trace_id: str, buffer: SpanBuffer) -> None:
        """通知已完成的 trace 有新增 span（迟到子 span）。"""
        try:
            if self.on_trace_complete is not None:
                self.on_trace_complete(trace_id, buffer.get_spans())
        except Exception:
            pass

    def _completion_check_loop(self) -> None:
        """定期检查 WAITING_CHILDREN 状态的 trace，空闲超时后触发完成。"""
        while not self._stop_event.is_set():
            time.sleep(self.completion_check_interval)
            self._check_pending_completions()

    def _check_pending_completions(self) -> None:
        """检查所有 buffer，触发满足空闲条件的完成回调。"""
        with self._buffers_lock:
            for trace_id, buffer in list(self._trace_buffers.items()):
                if trace_id in self._notified_traces:
                    continue
                if buffer.can_complete(force_flush=False):
                    self._notify_trace_complete(trace_id, buffer)

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

    def flush_trace(self, trace_id: str) -> bool:
        """
        强制完成指定 trace，立即触发完成回调。
        用于测试或手动确认链路结束。

        :return: True 表示成功触发完成，False 表示 trace 不存在或已完成
        """
        with self._buffers_lock:
            buffer = self._trace_buffers.get(trace_id)
            if buffer is None:
                return False
            if trace_id in self._notified_traces:
                return False
            if buffer.can_complete(force_flush=True):
                self._notify_trace_complete(trace_id, buffer)
                return True
        return False

    def flush_all(self) -> None:
        """
        强制刷新队列 + 强制完成所有等待中的 trace。
        用于关闭前确保所有数据都被处理。
        """
        self.flush()
        time.sleep(self.completion_check_interval * 2)
        with self._buffers_lock:
            for trace_id, buffer in list(self._trace_buffers.items()):
                if trace_id not in self._notified_traces:
                    if buffer.can_complete(force_flush=True):
                        self._notify_trace_complete(trace_id, buffer)

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
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        按条件搜索 trace（不做过早截断，保证大数据量下不会漏掉符合条件的 trace）。

        :param service_name: 任意 span 包含该服务即匹配
        :param operation_name: 任意 span 包含该操作即匹配
        :param start_time: trace 起始时间下限（秒，Unix 时间戳）
        :param end_time: trace 起始时间上限（秒，Unix 时间戳）
        :param limit: 返回数量上限，None 表示不限制
        """
        results = []
        with self._lock:
            for trace_id, spans in self._traces.items():
                if not spans:
                    continue
                match = True
                if service_name is not None:
                    match = any(s.service_name == service_name for s in spans)
                if match and operation_name is not None:
                    match = any(s.operation_name == operation_name for s in spans)
                if match and (start_time is not None or end_time is not None):
                    trace_start = min(s.start_time for s in spans if s.start_time is not None)
                    if start_time is not None and trace_start < start_time:
                        match = False
                    if end_time is not None and trace_start > end_time:
                        match = False
                if match:
                    results.append(trace_id)
                    if limit is not None and len(results) >= limit:
                        break
        return results

    def get_trace_start_time(self, trace_id: str) -> Optional[float]:
        """获取 trace 的最早 span 开始时间（用于排序）。"""
        spans = self.get_trace(trace_id)
        if not spans:
            return None
        starts = [s.start_time for s in spans if s.start_time is not None]
        return min(starts) if starts else None

    def clear(self) -> None:
        """清空所有数据。"""
        with self._lock:
            self._traces.clear()
