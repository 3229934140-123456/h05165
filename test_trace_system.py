import unittest
import time
from context_propagation import TraceContext, ContextManager
from span import Span, Tracer
from sampler import (
    ProbabilisticSampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    RateLimitingSampler,
    PerOperationSampler,
    CompositeSampler,
)
from collector import SpanCollector, InMemoryStorage, SpanBuffer
from trace_reconstructor import Trace, TraceReconstructor, topological_sort_spans


class TestTraceContext(unittest.TestCase):
    """测试 TraceContext 类。"""

    def test_generate_trace_id(self):
        """测试 trace ID 生成。"""
        trace_id = TraceContext.generate_trace_id()
        self.assertEqual(len(trace_id), 32)
        self.assertIsInstance(trace_id, str)

    def test_generate_span_id(self):
        """测试 span ID 生成。"""
        span_id = TraceContext.generate_span_id()
        self.assertEqual(len(span_id), 16)
        self.assertIsInstance(span_id, str)

    def test_new_root(self):
        """测试创建根上下文。"""
        ctx = TraceContext.new_root(sampled=True)
        self.assertIsNotNone(ctx.trace_id)
        self.assertIsNotNone(ctx.span_id)
        self.assertIsNone(ctx.parent_span_id)
        self.assertTrue(ctx.sampled)

    def test_headers_propagation(self):
        """测试 HTTP 头部传播。"""
        ctx = TraceContext.new_root(sampled=True)
        headers = ctx.to_headers()

        self.assertEqual(headers["x-trace-id"], ctx.trace_id)
        self.assertEqual(headers["x-span-id"], ctx.span_id)
        self.assertEqual(headers["x-sampled"], "1")

        restored = TraceContext.from_headers(headers)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.trace_id, ctx.trace_id)
        self.assertEqual(restored.span_id, ctx.span_id)
        self.assertTrue(restored.sampled)

    def test_headers_not_sampled(self):
        """测试未采样时的头部传播。"""
        ctx = TraceContext.new_root(sampled=False)
        headers = ctx.to_headers()
        self.assertEqual(headers["x-sampled"], "0")

        restored = TraceContext.from_headers(headers)
        self.assertFalse(restored.sampled)

    def test_new_child(self):
        """测试创建子上下文。"""
        parent = TraceContext.new_root()
        child = parent.new_child()

        self.assertEqual(parent.trace_id, child.trace_id)
        self.assertEqual(parent.span_id, child.parent_span_id)
        self.assertNotEqual(parent.span_id, child.span_id)
        self.assertEqual(parent.sampled, child.sampled)

    def test_from_headers_missing(self):
        """测试从缺失 trace 信息的头部解析。"""
        headers = {"content-type": "application/json"}
        ctx = TraceContext.from_headers(headers)
        self.assertIsNone(ctx)

    def test_context_manager(self):
        """测试线程本地上下文管理。"""
        ContextManager.clear_context()
        self.assertIsNone(ContextManager.get_context())

        ctx = TraceContext.new_root()
        ContextManager.set_context(ctx)
        self.assertEqual(ContextManager.get_context().trace_id, ctx.trace_id)

        ContextManager.clear_context()
        self.assertIsNone(ContextManager.get_context())


class TestSpan(unittest.TestCase):
    """测试 Span 类。"""

    def test_span_creation(self):
        """测试 span 创建。"""
        ctx = TraceContext.new_root()
        span = Span("test_operation", ctx, service_name="test-service")

        self.assertEqual(span.operation_name, "test_operation")
        self.assertEqual(span.trace_id, ctx.trace_id)
        self.assertEqual(span.span_id, ctx.span_id)
        self.assertEqual(span.service_name, "test-service")
        self.assertFalse(span.is_finished())

    def test_span_tags(self):
        """测试 span 标签。"""
        ctx = TraceContext.new_root()
        span = Span("test", ctx)
        span.set_tag("key1", "value1")
        span.set_tag("key2", 123)

        self.assertEqual(span.tags["key1"], "value1")
        self.assertEqual(span.tags["key2"], 123)

    def test_span_logs(self):
        """测试 span 日志。"""
        ctx = TraceContext.new_root()
        span = Span("test", ctx)
        span.log("event1", key="value")

        self.assertEqual(len(span.logs), 1)
        self.assertEqual(span.logs[0]["event"], "event1")
        self.assertEqual(span.logs[0]["fields"]["key"], "value")

    def test_span_finish(self):
        """测试 span 结束。"""
        ctx = TraceContext.new_root()
        span = Span("test", ctx)
        time.sleep(0.01)
        span.finish()

        self.assertTrue(span.is_finished())
        self.assertIsNotNone(span.duration)
        self.assertGreater(span.duration, 0)

    def test_span_error(self):
        """测试 span 错误标记。"""
        ctx = TraceContext.new_root()
        span = Span("test", ctx)
        span.set_error("something went wrong")

        self.assertEqual(span.status, "error")
        self.assertEqual(span.status_message, "something went wrong")
        self.assertTrue(span.tags["error"])

    def test_span_serialization(self):
        """测试 span 序列化和反序列化。"""
        ctx = TraceContext.new_root(sampled=True)
        span = Span("test_op", ctx, service_name="test-svc")
        span.set_tag("tag1", "value1")
        span.log("log1", field="data")
        span.finish()

        data = span.to_dict()
        restored = Span.from_dict(data)

        self.assertEqual(restored.operation_name, "test_op")
        self.assertEqual(restored.trace_id, span.trace_id)
        self.assertEqual(restored.span_id, span.span_id)
        self.assertEqual(restored.service_name, "test-svc")
        self.assertEqual(restored.tags["tag1"], "value1")
        self.assertTrue(restored.is_finished())


class TestTracer(unittest.TestCase):
    """测试 Tracer 类。"""

    def setUp(self):
        """每个测试前创建 tracer。"""
        self.collector = SpanCollector(max_idle_time=1.0)
        self.tracer = Tracer("test-service", collector=self.collector)

    def tearDown(self):
        """每个测试后停止收集器。"""
        self.collector.stop()

    def test_start_span(self):
        """测试启动 span。"""
        span = self.tracer.start_span("test_op")
        self.assertIsNotNone(span)
        self.assertEqual(span.operation_name, "test_op")
        self.assertEqual(span.service_name, "test-service")
        self.tracer.finish_span(span)

    def test_start_span_with_parent(self):
        """测试带父 span 的 span 创建。"""
        parent = self.tracer.start_span("parent")
        child = self.tracer.start_span("child", parent=parent)

        self.assertEqual(child.parent_span_id, parent.span_id)
        self.assertEqual(child.trace_id, parent.trace_id)

        self.tracer.finish_span(child)
        self.tracer.finish_span(parent)

    def test_active_span_scope(self):
        """测试活动 span 作用域。"""
        with self.tracer.start_active_span("outer") as outer:
            self.assertEqual(self.tracer.get_active_span().span_id, outer.span_id)

            with self.tracer.start_active_span("inner") as inner:
                self.assertEqual(self.tracer.get_active_span().span_id, inner.span_id)
                self.assertEqual(inner.parent_span_id, outer.span_id)

            self.assertEqual(self.tracer.get_active_span().span_id, outer.span_id)

        self.assertIsNone(self.tracer.get_active_span())

    def test_inject_extract(self):
        """测试上下文注入和提取。"""
        span = self.tracer.start_span("test")
        headers = {}
        self.tracer.inject(span.context, headers)

        self.assertIn("x-trace-id", headers)
        self.assertIn("x-span-id", headers)

        extracted = self.tracer.extract(headers)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.trace_id, span.trace_id)

        self.tracer.finish_span(span)

    def test_span_with_context(self):
        """测试使用提取的上下文创建 span。"""
        headers = {
            "x-trace-id": "abc123",
            "x-span-id": "def456",
            "x-sampled": "1",
        }
        context = self.tracer.extract(headers)
        span = self.tracer.start_span("test", context=context)

        self.assertEqual(span.trace_id, "abc123")
        self.assertEqual(span.parent_span_id, "def456")
        self.assertNotEqual(span.span_id, "def456")

        self.tracer.finish_span(span)


class TestSampler(unittest.TestCase):
    """测试采样器。"""

    def test_always_on_sampler(self):
        """测试全采样器。"""
        sampler = AlwaysOnSampler()
        for _ in range(100):
            self.assertTrue(sampler.should_sample())

    def test_always_off_sampler(self):
        """测试全不采样器。"""
        sampler = AlwaysOffSampler()
        for _ in range(100):
            self.assertFalse(sampler.should_sample())

    def test_probabilistic_sampler(self):
        """测试概率采样器。"""
        sampler = ProbabilisticSampler(rate=0.5)
        results = [sampler.should_sample() for _ in range(1000)]
        sampled = sum(results)
        self.assertGreater(sampled, 300)
        self.assertLess(sampled, 700)

    def test_probabilistic_sampler_by_trace_id(self):
        """测试基于 trace_id 的确定性采样。"""
        sampler = ProbabilisticSampler(rate=0.5)
        trace_id = "a" * 32

        result1 = sampler.should_sample_by_trace_id(trace_id)
        result2 = sampler.should_sample_by_trace_id(trace_id)
        self.assertEqual(result1, result2)

    def test_invalid_rate(self):
        """测试无效采样率。"""
        with self.assertRaises(ValueError):
            ProbabilisticSampler(rate=-0.1)
        with self.assertRaises(ValueError):
            ProbabilisticSampler(rate=1.1)

    def test_rate_limiting_sampler(self):
        """测试限速采样器。"""
        sampler = RateLimitingSampler(max_traces_per_second=10)
        results = []
        for _ in range(20):
            results.append(sampler.should_sample())
            time.sleep(0.05)

        sampled = sum(results)
        self.assertGreater(sampled, 0)
        self.assertLessEqual(sampled, 10)

    def test_per_operation_sampler(self):
        """测试按操作采样器。"""
        sampler = PerOperationSampler(
            operation_sample_rates={"critical_op": 1.0, "health_check": 0.0},
            default_rate=0.5,
        )

        self.assertTrue(sampler.should_sample("critical_op"))
        self.assertFalse(sampler.should_sample("health_check"))

        default_results = [sampler.should_sample("unknown_op") for _ in range(100)]
        self.assertGreater(sum(default_results), 30)
        self.assertLess(sum(default_results), 70)

    def test_composite_sampler(self):
        """测试组合采样器。"""
        sampler = CompositeSampler([
            AlwaysOffSampler(),
            AlwaysOnSampler(),
        ])
        self.assertTrue(sampler.should_sample())

        sampler2 = CompositeSampler([
            AlwaysOffSampler(),
            AlwaysOffSampler(),
        ])
        self.assertFalse(sampler2.should_sample())

    def test_sampler_description(self):
        """测试采样器描述。"""
        self.assertIn("1.0", AlwaysOnSampler().get_description())
        self.assertIn("0.0", AlwaysOffSampler().get_description())
        self.assertIn("0.5", ProbabilisticSampler(0.5).get_description())


class TestCollector(unittest.TestCase):
    """测试收集器。"""

    def setUp(self):
        """创建收集器。"""
        self.collected_traces = []

        def on_complete(trace_id, spans):
            self.collected_traces.append((trace_id, spans))

        self.collector = SpanCollector(
            batch_size=5,
            flush_interval=0.1,
            max_idle_time=1.0,
            on_trace_complete=on_complete,
        )
        self.collector.start()

    def tearDown(self):
        """停止收集器。"""
        self.collector.stop()

    def test_collect_span(self):
        """测试收集 span。"""
        ctx = TraceContext.new_root()
        span = Span("test", ctx, service_name="test")
        span.finish()

        result = self.collector.collect(span)
        self.assertTrue(result)

        self.collector.flush()
        time.sleep(0.2)

        stats = self.collector.get_stats()
        self.assertEqual(stats["collected_spans"], 1)

    def test_trace_completeness(self):
        """测试 trace 完整性检测。"""
        trace_id = TraceContext.generate_trace_id()
        root_span_id = TraceContext.generate_span_id()
        child_span_id = TraceContext.generate_span_id()

        ctx_root = TraceContext(trace_id, root_span_id, None, True)
        root_span = Span("root", ctx_root, service_name="svc1")
        root_span.finish()

        ctx_child = TraceContext(trace_id, child_span_id, root_span_id, True)
        child_span = Span("child", ctx_child, service_name="svc2")
        child_span.finish()

        self.collector.collect(child_span)
        self.collector.flush()
        time.sleep(0.2)

        buffer = self.collector.get_trace_buffer(trace_id)
        self.assertIsNotNone(buffer)
        self.assertFalse(buffer.is_structurally_complete())
        self.assertFalse(buffer.is_complete)
        self.assertEqual(buffer.get_missing_span_ids(), {root_span_id})

        self.collector.collect(root_span)
        self.collector.flush()
        time.sleep(0.1)

        buffer = self.collector.get_trace_buffer(trace_id)
        self.assertTrue(buffer.is_structurally_complete())
        self.collector.flush_trace(trace_id)
        time.sleep(0.2)
        self.assertTrue(buffer.is_complete)

    def test_in_memory_storage(self):
        """测试内存存储。"""
        storage = InMemoryStorage()
        trace_id = TraceContext.generate_trace_id()

        ctx = TraceContext.new_root()
        span = Span("test", ctx, service_name="test")
        span.finish()

        storage.save_trace(trace_id, [span])

        retrieved = storage.get_trace(trace_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].span_id, span.span_id)

        search_result = storage.search_traces(service_name="test")
        self.assertEqual(len(search_result), 1)

        search_result = storage.search_traces(operation_name="test")
        self.assertEqual(len(search_result), 1)

    def test_collector_stats(self):
        """测试收集器统计。"""
        stats = self.collector.get_stats()
        self.assertIn("collected_spans", stats)
        self.assertIn("completed_traces", stats)
        self.assertIn("dropped_spans", stats)
        self.assertIn("active_traces", stats)


class TestTraceReconstructor(unittest.TestCase):
    """测试链路重组器。"""

    def setUp(self):
        """创建重组器。"""
        self.reconstructor = TraceReconstructor()

    def _create_spans_for_trace(self, trace_id: str, out_of_order: bool = False):
        """创建测试用的 span。"""
        base_time = time.time()

        span_id_root = TraceContext.generate_span_id()
        span_id_child1 = TraceContext.generate_span_id()
        span_id_child2 = TraceContext.generate_span_id()

        ctx_root = TraceContext(trace_id, span_id_root, None, True)
        root = Span("root", ctx_root, service_name="svc1")
        root.start_time = base_time
        root.finish(end_time=base_time + 0.1)

        ctx_child1 = TraceContext(trace_id, span_id_child1, span_id_root, True)
        child1 = Span("child1", ctx_child1, service_name="svc2")
        child1.start_time = base_time + 0.01
        child1.finish(end_time=base_time + 0.06)

        ctx_child2 = TraceContext(trace_id, span_id_child2, span_id_root, True)
        child2 = Span("child2", ctx_child2, service_name="svc3")
        child2.start_time = base_time + 0.02
        child2.finish(end_time=base_time + 0.05)

        spans = [root, child1, child2]
        if out_of_order:
            spans = [child1, child2, root]

        return spans

    def test_build_tree(self):
        """测试调用树构建。"""
        trace_id = TraceContext.generate_trace_id()
        spans = self._create_spans_for_trace(trace_id)

        traces = self.reconstructor.add_spans(spans)
        self.assertEqual(len(traces), 1)

        trace = traces[0]
        self.assertTrue(trace.is_complete)
        self.assertEqual(trace.get_span_count(), 3)
        self.assertEqual(trace.get_service_count(), 3)

        self.assertIsNotNone(trace.root)
        self.assertEqual(len(trace.root.children), 2)

    def test_out_of_order_spans(self):
        """测试乱序 span 处理。"""
        trace_id = TraceContext.generate_trace_id()
        spans = self._create_spans_for_trace(trace_id, out_of_order=True)

        traces = self.reconstructor.add_spans(spans)
        self.assertEqual(len(traces), 1)

        trace = traces[0]
        self.assertTrue(trace.is_complete)
        self.assertEqual(trace.get_span_count(), 3)

    def test_parent_late_arrival(self):
        """测试父 span 晚到的情况。"""
        trace_id = TraceContext.generate_trace_id()
        spans = self._create_spans_for_trace(trace_id)

        child_spans = [s for s in spans if s.parent_span_id is not None]
        root_span = [s for s in spans if s.parent_span_id is None][0]

        traces1 = self.reconstructor.add_spans(child_spans)
        self.assertEqual(len(traces1), 1)
        self.assertFalse(traces1[0].is_complete)
        self.assertGreater(traces1[0].get_orphan_count(), 0)

        traces2 = self.reconstructor.add_spans([root_span])
        self.assertEqual(len(traces2), 1)
        self.assertTrue(traces2[0].is_complete)
        self.assertEqual(traces2[0].get_orphan_count(), 0)

    def test_critical_path(self):
        """测试关键路径计算。"""
        trace_id = TraceContext.generate_trace_id()
        spans = self._create_spans_for_trace(trace_id)

        traces = self.reconstructor.add_spans(spans)
        trace = traces[0]

        critical_path = trace.get_critical_path()
        self.assertGreater(len(critical_path), 0)
        self.assertEqual(critical_path[0].operation_name, "root")

    def test_trace_stats(self):
        """测试 trace 统计信息。"""
        trace_id = TraceContext.generate_trace_id()
        spans = self._create_spans_for_trace(trace_id)

        traces = self.reconstructor.add_spans(spans)
        trace = traces[0]

        self.assertGreater(trace.get_total_duration(), 0)
        self.assertEqual(trace.get_span_count(), 3)
        self.assertEqual(trace.get_service_count(), 3)
        self.assertFalse(trace.has_error())

    def test_topological_sort(self):
        """测试拓扑排序。"""
        trace_id = TraceContext.generate_trace_id()
        spans = self._create_spans_for_trace(trace_id, out_of_order=True)

        sorted_spans = topological_sort_spans(spans)
        self.assertEqual(len(sorted_spans), 3)

        span_map = {s.span_id: s for s in sorted_spans}
        for span in sorted_spans:
            if span.parent_span_id is not None:
                parent_pos = next(
                    i for i, s in enumerate(sorted_spans) if s.span_id == span.parent_span_id
                )
                span_pos = next(
                    i for i, s in enumerate(sorted_spans) if s.span_id == span.span_id
                )
                self.assertLess(parent_pos, span_pos)

    def test_force_complete(self):
        """测试强制完成 trace。"""
        trace_id = TraceContext.generate_trace_id()

        ctx = TraceContext(trace_id, "span1", None, True)
        span = Span("orphan", ctx, service_name="svc")
        span.finish()

        self.reconstructor.add_spans([span])

        trace = self.reconstructor.force_complete(trace_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.get_span_count(), 1)

    def test_cleanup_expired(self):
        """测试清理过期 trace。"""
        trace_id = TraceContext.generate_trace_id()
        ctx = TraceContext(trace_id, "span1", "missing_parent", True)
        span = Span("orphan", ctx, service_name="svc")
        span.start_time = time.time() - 400
        span.finish()

        self.reconstructor.add_spans([span])

        completed = self.reconstructor.cleanup_expired(max_age=300)
        self.assertEqual(len(completed), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
