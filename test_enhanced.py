import unittest
import time
import threading
import json
import random

from context_propagation import TraceContext, ContextManager
from span import Tracer, Span
from sampler import ProbabilisticSampler, AlwaysOnSampler
from collector import SpanCollector, SpanBuffer, InMemoryStorage
from trace_reconstructor import Trace, TraceReconstructor
from trace_query import TraceQueryService


class TestConcurrentSafety(unittest.TestCase):
    """测试并发安全。"""

    def test_thread_local_active_spans(self):
        """测试活动 span 栈是线程隔离的。"""
        tracer = Tracer("test", sampler=AlwaysOnSampler())

        results = {}

        def worker(tid):
            with tracer.start_active_span(f"root_{tid}") as root:
                time.sleep(0.01)
                with tracer.start_active_span(f"child_{tid}") as child:
                    active = tracer.get_active_span()
                    results[tid] = {
                        "root_trace_id": root.trace_id,
                        "child_trace_id": child.trace_id,
                        "active_span_id": active.span_id,
                        "expected_span_id": child.span_id,
                    }
                time.sleep(0.01)

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tid, r in results.items():
            self.assertEqual(r["root_trace_id"], r["child_trace_id"])
            self.assertEqual(r["active_span_id"], r["expected_span_id"])

        trace_ids = {r["root_trace_id"] for r in results.values()}
        self.assertEqual(len(trace_ids), 10)

    def test_concurrent_trace_isolation(self):
        """测试高并发下 trace 互不串扰。"""
        storage = InMemoryStorage()
        reconstructor = TraceReconstructor()
        completed_traces = []

        def on_complete(trace_id, spans):
            storage.save_trace(trace_id, spans)
            reconstructor.add_spans(spans)
            completed_traces.append(trace_id)

        collector = SpanCollector(
            post_root_idle_time=0.2,
            completion_check_interval=0.05,
            on_trace_complete=on_complete,
        )
        tracer = Tracer("concurrent-test", collector=collector, sampler=AlwaysOnSampler())
        collector.start()

        NUM_THREADS = 30
        trace_ids_by_thread = {}

        def process(thread_id):
            with tracer.start_active_span(f"req_{thread_id}") as root:
                trace_ids_by_thread[thread_id] = root.trace_id
                root.set_tag("thread_id", thread_id)
                time.sleep(random.uniform(0.005, 0.02))

                for i in range(3):
                    with tracer.start_active_span(f"child_{thread_id}_{i}") as child:
                        child.set_tag("thread_id", thread_id)
                        ctx = ContextManager.get_context()
                        if ctx:
                            self.assertEqual(ctx.trace_id, root.trace_id)
                        time.sleep(random.uniform(0.002, 0.01))

        threads = []
        for i in range(NUM_THREADS):
            t = threading.Thread(target=process, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.5)
        collector.flush_all()
        time.sleep(0.5)

        for tid, trace_id in trace_ids_by_thread.items():
            spans = storage.get_trace(trace_id)
            self.assertIsNotNone(spans, f"线程 {tid} 的 trace 丢失了")

            thread_ids_in_trace = set()
            for s in spans:
                t = s.tags.get("thread_id")
                if t is not None:
                    thread_ids_in_trace.add(t)

            self.assertEqual(
                thread_ids_in_trace,
                {tid},
                f"trace {trace_id[:12]}... 混入了其他线程的 span: {thread_ids_in_trace}",
            )

            expected_spans = 4
            self.assertEqual(
                len(spans),
                expected_spans,
                f"线程 {tid}: 期望 {expected_spans} 个 span, 实际 {len(spans)}",
            )

        collector.stop()


class TestSmartCompletion(unittest.TestCase):
    """测试稳健的完成策略。"""

    def test_root_first_waits_for_children(self):
        """根 span 先到，不立即触发完成回调。"""
        callback_called = []

        def on_complete(trace_id, spans):
            callback_called.append((trace_id, len(spans)))

        collector = SpanCollector(
            post_root_idle_time=1.0,
            completion_check_interval=0.05,
            on_trace_complete=on_complete,
        )
        collector.start()

        trace_id = TraceContext.generate_trace_id()
        span_id_root = TraceContext.generate_span_id()
        span_id_child = TraceContext.generate_span_id()

        ctx_root = TraceContext(trace_id, span_id_root, None, True)
        span_root = Span("root", ctx_root, service_name="svc")
        span_root.start_time = time.time()
        span_root.finish()

        collector.collect(span_root)
        collector.flush()
        time.sleep(0.3)

        self.assertEqual(len(callback_called), 0, "根 span 刚到不应该立即触发回调")

        buffer = collector.get_trace_buffer(trace_id)
        self.assertEqual(buffer.state, SpanBuffer.STATE_WAITING_CHILDREN)

        ctx_child = TraceContext(trace_id, span_id_child, span_id_root, True)
        span_child = Span("child", ctx_child, service_name="svc")
        span_child.start_time = time.time()
        span_child.finish()

        collector.collect(span_child)
        collector.flush()
        time.sleep(0.3)

        self.assertEqual(len(callback_called), 0, "还在等待空闲时间，不应触发")

        time.sleep(1.0)

        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0][1], 2)

        collector.stop()

    def test_flush_trace_forces_completion(self):
        """flush_trace 可以强制触发完成。"""
        callback_called = []

        def on_complete(trace_id, spans):
            callback_called.append((trace_id, len(spans)))

        collector = SpanCollector(
            post_root_idle_time=10.0,
            completion_check_interval=0.1,
            on_trace_complete=on_complete,
        )
        collector.start()

        trace_id = TraceContext.generate_trace_id()
        span_id_root = TraceContext.generate_span_id()

        ctx_root = TraceContext(trace_id, span_id_root, None, True)
        span_root = Span("root", ctx_root, service_name="svc")
        span_root.start_time = time.time()
        span_root.finish()

        collector.collect(span_root)
        collector.flush()
        time.sleep(0.2)
        self.assertEqual(len(callback_called), 0)

        result = collector.flush_trace(trace_id)
        self.assertTrue(result)
        time.sleep(0.1)

        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0][1], 1)

        collector.stop()

    def test_late_child_after_completion_triggers_update(self):
        """完成回调后，迟到的子 span 触发更新回调。"""
        callback_count = [0]
        last_span_count = [0]

        def on_complete(trace_id, spans):
            callback_count[0] += 1
            last_span_count[0] = len(spans)

        collector = SpanCollector(
            post_root_idle_time=0.2,
            completion_check_interval=0.05,
            on_trace_complete=on_complete,
        )
        collector.start()

        trace_id = TraceContext.generate_trace_id()
        span_id_root = TraceContext.generate_span_id()
        span_id_child = TraceContext.generate_span_id()

        ctx_root = TraceContext(trace_id, span_id_root, None, True)
        span_root = Span("root", ctx_root, service_name="svc")
        span_root.start_time = time.time()
        span_root.finish()

        collector.collect(span_root)
        collector.flush()
        time.sleep(0.5)

        self.assertEqual(callback_count[0], 1)
        self.assertEqual(last_span_count[0], 1)

        ctx_child = TraceContext(trace_id, span_id_child, span_id_root, True)
        span_child = Span("late_child", ctx_child, service_name="svc")
        span_child.start_time = time.time()
        span_child.finish()

        collector.collect(span_child)
        collector.flush()
        time.sleep(0.3)

        self.assertEqual(callback_count[0], 2)
        self.assertEqual(last_span_count[0], 2)

        collector.stop()


class TestTraceQuery(unittest.TestCase):
    """测试 Trace 查询服务。"""

    def _build_test_data(self):
        """构建测试数据。"""
        storage = InMemoryStorage()
        reconstructor = TraceReconstructor()

        def on_complete(trace_id, spans):
            storage.save_trace(trace_id, spans)
            reconstructor.add_spans(spans)

        collector = SpanCollector(
            post_root_idle_time=0.2,
            completion_check_interval=0.05,
            on_trace_complete=on_complete,
        )
        tracer_a = Tracer("gateway", collector=collector, sampler=AlwaysOnSampler())
        tracer_b = Tracer("order-svc", collector=collector)
        tracer_c = Tracer("pay-svc", collector=collector)
        collector.start()

        trace_ids = []
        error_trace_id = None
        for i in range(5):
            with tracer_a.start_active_span(f"http_{i}") as root:
                trace_ids.append(root.trace_id)
                headers = {}
                tracer_a.inject(root.context, headers)
                ctx_b = tracer_b.extract(headers)

                with tracer_b.start_active_span(f"order_{i}", context=ctx_b) as order:
                    if i == 2:
                        order.set_error("fail")
                        error_trace_id = root.trace_id

                    headers2 = {}
                    tracer_b.inject(order.context, headers2)
                    ctx_c = tracer_c.extract(headers2)
                    with tracer_c.start_active_span(f"pay_{i}", context=ctx_c) as pay:
                        if i == 2:
                            pay.set_error("pay fail")
                        time.sleep(0.02)

        collector.flush()
        time.sleep(0.3)
        for tid in trace_ids:
            collector.flush_trace(tid)
        time.sleep(0.3)
        collector.flush_all()
        time.sleep(0.5)
        collector.stop()

        return TraceQueryService(storage, reconstructor), error_trace_id

    def test_get_trace_by_id(self):
        """测试按 trace ID 查询。"""
        query_svc, _ = self._build_test_data()
        trace_ids = query_svc.storage.get_all_trace_ids()
        self.assertGreater(len(trace_ids), 0)

        trace = query_svc.get_trace_by_id(trace_ids[0])
        self.assertIsNotNone(trace)
        self.assertGreater(trace.get_span_count(), 0)

    def test_get_trace_json(self):
        """测试 JSON 输出格式。"""
        query_svc, _ = self._build_test_data()
        trace_ids = query_svc.storage.get_all_trace_ids()
        json_str = query_svc.get_trace_json(trace_ids[0])

        self.assertIsNotNone(json_str)
        data = json.loads(json_str)

        self.assertIn("traceID", data)
        self.assertIn("spans", data)
        self.assertIn("processes", data)
        self.assertIn("summary", data)
        self.assertIsInstance(data["spans"], list)
        self.assertGreater(len(data["spans"]), 0)

        first_span = data["spans"][0]
        self.assertIn("traceID", first_span)
        self.assertIn("spanID", first_span)
        self.assertIn("operationName", first_span)
        self.assertIn("startTime", first_span)
        self.assertIn("duration", first_span)
        self.assertIn("references", first_span)
        self.assertIn("tags", first_span)
        self.assertIn("logs", first_span)

    def test_search_by_service(self):
        """测试按服务名搜索。"""
        query_svc, _ = self._build_test_data()
        traces = query_svc.search_traces(service_name="pay-svc")
        self.assertGreater(len(traces), 0)

    def test_search_by_error(self):
        """测试按错误状态搜索。"""
        query_svc, error_tid = self._build_test_data()
        traces = query_svc.search_traces(has_error=True)
        self.assertGreaterEqual(len(traces), 1)

        trace_ids = {t.trace_id for t in traces}
        self.assertIn(error_tid, trace_ids)

    def test_search_by_operation(self):
        """测试按操作名搜索。"""
        query_svc, _ = self._build_test_data()
        traces = query_svc.search_traces(operation_name="pay_0")
        self.assertGreater(len(traces), 0)

    def test_combined_search(self):
        """测试组合条件搜索。"""
        query_svc, error_tid = self._build_test_data()
        traces = query_svc.search_traces(service_name="pay-svc", has_error=True)
        self.assertGreaterEqual(len(traces), 1)

    def test_list_services_and_operations(self):
        """测试列出服务和操作。"""
        query_svc, _ = self._build_test_data()
        services = query_svc.list_services()
        self.assertIn("gateway", services)
        self.assertIn("order-svc", services)
        self.assertIn("pay-svc", services)

        ops = query_svc.list_operations("gateway")
        self.assertTrue(any("http_" in o for o in ops))

    def test_get_stats(self):
        """测试获取统计信息。"""
        query_svc, _ = self._build_test_data()
        stats = query_svc.get_stats()
        self.assertGreater(stats["total_traces"], 0)
        self.assertGreater(stats["total_spans"], 0)
        self.assertEqual(stats["unique_services"], 3)


class TestDeterministicSampling(unittest.TestCase):
    """测试确定性采样。"""

    def test_same_trace_id_same_result(self):
        """同一个 trace ID 多次判断结果必须一致。"""
        sampler = ProbabilisticSampler(rate=0.3)
        trace_id = TraceContext.generate_trace_id()

        results = [sampler.should_sample_by_trace_id(trace_id) for _ in range(100)]
        self.assertEqual(len(set(results)), 1)

    def test_probability_distribution(self):
        """采样率分布接近理论值。"""
        sampler = ProbabilisticSampler(rate=0.5)
        total = 5000
        sampled = sum(
            1 for _ in range(total) if sampler.should_sample_by_trace_id(TraceContext.generate_trace_id())
        )
        ratio = sampled / total
        self.assertGreater(ratio, 0.45)
        self.assertLess(ratio, 0.55)

    def test_tracer_uses_deterministic_sampling(self):
        """Tracer 创建根 span 时使用确定性采样。"""
        sampler = ProbabilisticSampler(rate=0.5)
        tracer = Tracer("test", sampler=sampler)

        sampled_set = set()
        not_sampled_set = set()

        for _ in range(200):
            span = tracer.start_span("test")
            tracer.finish_span(span)
            tid = span.trace_id
            if span.context.sampled:
                sampled_set.add(tid)
            else:
                not_sampled_set.add(tid)

        for tid in sampled_set:
            self.assertTrue(sampler.should_sample_by_trace_id(tid))
        for tid in not_sampled_set:
            self.assertFalse(sampler.should_sample_by_trace_id(tid))

    def test_downstream_inherits_sampling(self):
        """下游服务继承采样决策，不重新采样。"""
        sampler = ProbabilisticSampler(rate=0.5)
        tracer_entry = Tracer("gateway", sampler=sampler)
        tracer_downstream = Tracer("downstream", sampler=sampler)

        consistency_ok = True
        for _ in range(100):
            with tracer_entry.start_active_span("root") as root:
                entry_sampled = root.context.sampled
                headers = {}
                tracer_entry.inject(root.context, headers)

                ctx = tracer_downstream.extract(headers)
                with tracer_downstream.start_active_span("child", context=ctx) as child:
                    if child.context.sampled != entry_sampled:
                        consistency_ok = False

        self.assertTrue(consistency_ok, "下游服务没有正确继承采样决策")

    def test_not_sampled_not_reported(self):
        """未采样的 trace 不会上报到 collector。"""
        sampler = ProbabilisticSampler(rate=0.0)
        storage = InMemoryStorage()
        reconstructor = TraceReconstructor()

        def on_complete(trace_id, spans):
            storage.save_trace(trace_id, spans)
            reconstructor.add_spans(spans)

        collector = SpanCollector(post_root_idle_time=0.1, on_trace_complete=on_complete)
        tracer = Tracer("test", collector=collector, sampler=sampler)
        collector.start()

        for i in range(20):
            with tracer.start_active_span(f"op_{i}") as s:
                pass

        time.sleep(0.3)
        collector.flush_all()
        time.sleep(0.3)

        self.assertEqual(len(storage.get_all_trace_ids()), 0)
        collector.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
