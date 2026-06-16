import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from span import Span


class SpanTreeNode:
    """
    调用树节点，包含一个 span 及其子节点。
    """

    def __init__(self, span: Span):
        self.span = span
        self.span_id = span.span_id
        self.children: List["SpanTreeNode"] = []
        self.parent: Optional["SpanTreeNode"] = None

    def add_child(self, child: "SpanTreeNode") -> None:
        """添加子节点。"""
        child.parent = self
        self.children.append(child)

    def get_total_duration(self) -> float:
        """获取该节点及其所有子节点的总耗时。"""
        total = self.span.duration or 0.0
        for child in self.children:
            total = max(total, child.get_total_duration())
        return total

    def get_self_time(self) -> float:
        """获取该节点自身的耗时（减去子节点耗时）。"""
        if self.span.duration is None:
            return 0.0
        child_duration = sum(child.span.duration or 0.0 for child in self.children)
        return max(0.0, self.span.duration - child_duration)

    def has_error(self) -> bool:
        """检查该节点或其子节点是否有错误。"""
        if self.span.status == "error":
            return True
        return any(child.has_error() for child in self.children)

    def __repr__(self, indent: int = 0) -> str:
        prefix = "  " * indent
        duration = f"{self.span.duration * 1000:.2f}ms" if self.span.duration else "N/A"
        result = (
            f"{prefix}[{self.span.service_name}] {self.span.operation_name} "
            f"({duration})"
        )
        if self.span.status == "error":
            result += " [ERROR]"
        for child in self.children:
            result += "\n" + child.__repr__(indent + 1)
        return result


class Trace:
    """
    完整的调用链，包含所有 span 和调用树。
    """

    def __init__(self, trace_id: str, spans: List[Span]):
        self.trace_id = trace_id
        self.spans = spans
        self.root: Optional[SpanTreeNode] = None
        self.span_map: Dict[str, SpanTreeNode] = {}
        self.is_complete = False
        self._build_tree()

    def _build_tree(self) -> None:
        """
        构建调用树。
        处理父 span 比子 span 晚到的情况：子 span 先放入待处理队列。
        """
        for span in self.spans:
            node = SpanTreeNode(span)
            self.span_map[span.span_id] = node

        orphan_nodes: Dict[str, List[SpanTreeNode]] = defaultdict(list)

        for span in self.spans:
            node = self.span_map[span.span_id]
            parent_id = span.parent_span_id

            if parent_id is None:
                if self.root is not None:
                    pass
                self.root = node
            elif parent_id in self.span_map:
                parent_node = self.span_map[parent_id]
                parent_node.add_child(node)
            else:
                orphan_nodes[parent_id].append(node)

        resolved = True
        while resolved:
            resolved = False
            for parent_id in list(orphan_nodes.keys()):
                if parent_id in self.span_map:
                    parent_node = self.span_map[parent_id]
                    for child_node in orphan_nodes[parent_id]:
                        parent_node.add_child(child_node)
                    del orphan_nodes[parent_id]
                    resolved = True

        self._sort_children(self.root)

        has_root = self.root is not None
        has_orphans = len(orphan_nodes) > 0
        all_connected = True
        for span in self.spans:
            node = self.span_map.get(span.span_id)
            if node is not None and node.parent is None and span.parent_span_id is not None:
                all_connected = False
                break

        self.is_complete = has_root and not has_orphans and all_connected

        if not self.is_complete:
            self._orphan_nodes = orphan_nodes

    def _sort_children(self, node: Optional[SpanTreeNode]) -> None:
        """按开始时间对子节点排序。"""
        if node is None:
            return
        node.children.sort(key=lambda x: x.span.start_time)
        for child in node.children:
            self._sort_children(child)

    def get_root_span(self) -> Optional[Span]:
        """获取根 span。"""
        return self.root.span if self.root else None

    def get_total_duration(self) -> float:
        """获取整个 trace 的总耗时。"""
        if self.root is None:
            return 0.0
        return self.root.get_total_duration()

    def get_span_count(self) -> int:
        """获取 span 数量。"""
        return len(self.spans)

    def get_service_count(self) -> int:
        """获取涉及的服务数量。"""
        return len(set(span.service_name for span in self.spans))

    def has_error(self) -> bool:
        """检查 trace 是否有错误。"""
        if self.root is None:
            return False
        return self.root.has_error()

    def get_critical_path(self) -> List[Span]:
        """
        获取关键路径（耗时最长的路径）。
        关键路径决定了整个 trace 的总耗时。
        """
        if self.root is None:
            return []

        def find_longest_path(node: SpanTreeNode) -> Tuple[float, List[Span]]:
            if not node.children:
                return (node.span.duration or 0.0), [node.span]

            max_duration = 0.0
            max_path: List[Span] = []
            for child in node.children:
                child_duration, child_path = find_longest_path(child)
                if child_duration > max_duration:
                    max_duration = child_duration
                    max_path = child_path

            current_duration = (node.span.duration or 0.0) + max_duration
            return current_duration, [node.span] + max_path

        _, path = find_longest_path(self.root)
        return path

    def get_orphan_count(self) -> int:
        """获取找不到父节点的 span 数量。"""
        if hasattr(self, "_orphan_nodes"):
            return sum(len(nodes) for nodes in self._orphan_nodes.values())
        return 0

    def get_missing_parent_ids(self) -> Set[str]:
        """获取缺失的父 span ID。"""
        if hasattr(self, "_orphan_nodes"):
            return set(self._orphan_nodes.keys())
        return set()

    def to_dict(self) -> Dict:
        """转换为字典格式。"""
        return {
            "trace_id": self.trace_id,
            "is_complete": self.is_complete,
            "span_count": self.get_span_count(),
            "service_count": self.get_service_count(),
            "total_duration_ms": self.get_total_duration() * 1000,
            "has_error": self.has_error(),
            "orphan_count": self.get_orphan_count(),
            "spans": [span.to_dict() for span in self.spans],
        }

    def print_tree(self) -> None:
        """打印调用树。"""
        if self.root is None:
            print(f"Trace {self.trace_id}: 无根节点")
            return

        print(f"Trace: {self.trace_id}")
        print(f"  总耗时: {self.get_total_duration() * 1000:.2f}ms")
        print(f"  Span 数量: {self.get_span_count()}")
        print(f"  服务数量: {self.get_service_count()}")
        print(f"  完整: {self.is_complete}")
        if self.get_orphan_count() > 0:
            print(f"  孤立 span: {self.get_orphan_count()}")
            print(f"  缺失父 span: {self.get_missing_parent_ids()}")
        print(f"  调用树:")
        print(self.root.__repr__(indent=2))

    def __repr__(self) -> str:
        return (
            f"Trace(trace_id={self.trace_id}, "
            f"spans={len(self.spans)}, "
            f"complete={self.is_complete}, "
            f"duration={self.get_total_duration() * 1000:.2f}ms)"
        )


class TraceReconstructor:
    """
    链路重组器，负责将分散的 span 重组成完整的调用链。

    核心功能：
    1. 按 trace ID 聚合 span
    2. 处理乱序到达的 span
    3. 处理父 span 比子 span 晚到的情况
    4. 构建调用树
    5. 计算 trace 的统计信息
    """

    def __init__(self):
        self._pending_spans: Dict[str, List[Span]] = defaultdict(list)
        self._complete_traces: Dict[str, Trace] = {}
        self._last_activity: Dict[str, float] = {}

    def add_spans(self, spans: List[Span]) -> List[Trace]:
        """
        添加一批 span，尝试重组 trace。

        :return: 新完成的 trace 列表
        """
        new_traces: List[Trace] = []
        trace_groups: Dict[str, List[Span]] = defaultdict(list)

        for span in spans:
            trace_groups[span.trace_id].append(span)

        for trace_id, new_spans in trace_groups.items():
            if trace_id in self._complete_traces:
                existing_trace = self._complete_traces[trace_id]
                existing_span_ids = {span.span_id for span in existing_trace.spans}
                spans_to_add = [s for s in new_spans if s.span_id not in existing_span_ids]
                if spans_to_add:
                    all_spans = existing_trace.spans + spans_to_add
                    trace = Trace(trace_id, all_spans)
                    self._complete_traces[trace_id] = trace
                    new_traces.append(trace)
                continue

            if trace_id in self._pending_spans:
                self._pending_spans[trace_id].extend(new_spans)
            else:
                self._pending_spans[trace_id] = list(new_spans)

            self._last_activity[trace_id] = time.time()

            trace = self._try_reconstruct(trace_id)
            if trace is not None:
                if trace.is_complete:
                    self._complete_traces[trace_id] = trace
                    if trace_id in self._pending_spans:
                        del self._pending_spans[trace_id]
                    if trace_id in self._last_activity:
                        del self._last_activity[trace_id]
                new_traces.append(trace)

        return new_traces

    def _try_reconstruct(self, trace_id: str) -> Optional[Trace]:
        """
        尝试重建指定 trace。
        即使不完整也返回 trace 对象，便于查看进度。
        """
        if trace_id not in self._pending_spans:
            return None

        spans = self._pending_spans[trace_id]
        trace = Trace(trace_id, spans)
        return trace

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """获取指定的 trace。"""
        if trace_id in self._complete_traces:
            return self._complete_traces[trace_id]
        if trace_id in self._pending_spans:
            return self._try_reconstruct(trace_id)
        return None

    def get_complete_trace_ids(self) -> List[str]:
        """获取所有完整的 trace ID。"""
        return list(self._complete_traces.keys())

    def get_pending_trace_ids(self) -> List[str]:
        """获取所有待处理的 trace ID。"""
        return list(self._pending_spans.keys())

    def force_complete(self, trace_id: str) -> Optional[Trace]:
        """
        强制完成一个 trace（超时或手动触发）。
        即使不完整也将其标记为已完成并返回。
        """
        if trace_id in self._complete_traces:
            return self._complete_traces[trace_id]

        if trace_id in self._pending_spans:
            spans = self._pending_spans[trace_id]
            trace = Trace(trace_id, spans)
            self._complete_traces[trace_id] = trace
            del self._pending_spans[trace_id]
            if trace_id in self._last_activity:
                del self._last_activity[trace_id]
            return trace

        return None

    def cleanup_expired(self, max_age: float = 300.0, max_idle: float = 60.0) -> List[Trace]:
        """
        清理过期的待处理 trace。
        超过最大存活时间或空闲时间的 trace 会被强制完成。

        :return: 被强制完成的 trace 列表
        """
        now = time.time()
        expired: List[str] = []

        for trace_id, spans in self._pending_spans.items():
            first_span_time = min(span.start_time for span in spans)
            last_activity = self._last_activity.get(trace_id, now)

            age = now - first_span_time
            idle = now - last_activity

            if age > max_age or idle > max_idle:
                expired.append(trace_id)

        completed: List[Trace] = []
        for trace_id in expired:
            trace = self.force_complete(trace_id)
            if trace is not None:
                completed.append(trace)

        return completed

    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            "complete_traces": len(self._complete_traces),
            "pending_traces": len(self._pending_spans),
        }


def topological_sort_spans(spans: List[Span]) -> List[Span]:
    """
    对 span 进行拓扑排序（按父子关系）。
    用于确保输出顺序正确。
    """
    span_map: Dict[str, Span] = {s.span_id: s for s in spans}
    children_map: Dict[str, List[Span]] = defaultdict(list)
    in_degree: Dict[str, int] = {}

    for span in spans:
        in_degree[span.span_id] = 0

    for span in spans:
        parent_id = span.parent_span_id
        if parent_id is not None and parent_id in span_map:
            children_map[parent_id].append(span)
            in_degree[span.span_id] = in_degree.get(span.span_id, 0) + 1

    queue: List[Span] = [span for span in spans if in_degree[span.span_id] == 0]
    queue.sort(key=lambda s: s.start_time)

    result: List[Span] = []
    while queue:
        span = queue.pop(0)
        result.append(span)

        for child in children_map[span.span_id]:
            in_degree[child.span_id] -= 1
            if in_degree[child.span_id] == 0:
                queue.append(child)
                queue.sort(key=lambda s: s.start_time)

    return result
