import unittest
import time
import threading
import json
import random

from context_propagation import TraceContext, ContextManager
from span import Tracer, Span
from sampler import ProbabilisticSampler, AlwaysOnSampler, PerOperationSampler, ServiceOperationSampler
from collector import SpanCollector, SpanBuffer, InMemoryStorage
from trace_reconstructor import Trace, TraceReconstructor
from trace_query import TraceQueryService
import os
import json
import tempfile


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


class TestAdvancedSearch(unittest.TestCase):
    """测试增强搜索：时间范围、分页、排序、大数据量下错误链路不漏。"""

    def _build_dataset(self):
        storage = InMemoryStorage()
        reconstructor = TraceReconstructor()

        def on_complete(trace_id, spans):
            storage.save_trace(trace_id, spans)
            reconstructor.add_spans(spans)

        collector = SpanCollector(
            post_root_idle_time=0.1,
            completion_check_interval=0.05,
            on_trace_complete=on_complete,
        )
        tracer_a = Tracer("gateway", collector=collector, sampler=AlwaysOnSampler())
        tracer_b = Tracer("pay-svc", collector=collector)
        collector.start()

        trace_ids = []
        error_ids = []
        base_time = time.time() - 500

        for i in range(50):
            t0 = base_time + i * 10
            with tracer_a.start_active_span(f"http_{i}", start_time=t0) as root:
                trace_ids.append(root.trace_id)
                headers = {}
                tracer_a.inject(root.context, headers)
                ctx = tracer_b.extract(headers)

                is_error = i % 25 == 24
                with tracer_b.start_active_span(f"pay_{i}", context=ctx, start_time=t0 + 0.005) as pay:
                    if is_error:
                        pay.set_error("boom")
                        error_ids.append(root.trace_id)
                    pay.finish(end_time=t0 + 0.01 + (0.1 if is_error else 0.005))
                root.finish(end_time=t0 + 0.02 + (0.2 if is_error else 0.01))

        collector.flush()
        for tid in trace_ids:
            collector.flush_trace(tid)
        time.sleep(0.5)
        collector.flush_all()
        time.sleep(0.3)
        collector.stop()

        return TraceQueryService(storage, reconstructor), trace_ids, error_ids, base_time

    def test_pagination(self):
        """测试分页能拿到不同页的结果。"""
        q, trace_ids, _, _ = self._build_dataset()
        page1, total = q.search_traces(limit=10, offset=0)
        page2, _ = q.search_traces(limit=10, offset=10)
        self.assertEqual(total, 50)
        self.assertEqual(len(page1), 10)
        self.assertEqual(len(page2), 10)
        self.assertNotEqual(page1[0].trace_id, page2[0].trace_id)

    def test_error_traces_always_found(self):
        """大数据量下 has_error=True 能筛出所有错误 trace，不会漏。"""
        q, _, error_ids, _ = self._build_dataset()
        found, total = q.search_traces(has_error=True, limit=1000)
        found_ids = {t.trace_id for t in found}
        self.assertEqual(total, len(error_ids))
        self.assertEqual(found_ids, set(error_ids))

    def test_time_range_filter(self):
        """测试按时间范围过滤。"""
        q, trace_ids, error_ids, base_time = self._build_dataset()
        start = base_time + 100
        end = base_time + 250
        found, total = q.search_traces(start_time=start, end_time=end, limit=1000)
        self.assertTrue(10 <= total <= 20)
        for t in found:
            spans = q.storage.get_trace(t.trace_id)
            t0 = min(s.start_time for s in spans if s.start_time)
            self.assertGreaterEqual(t0, start)
            self.assertLessEqual(t0, end)

    def test_min_duration_filter(self):
        """错误 trace 更慢，按最小耗时应该只筛到错误 trace。"""
        q, _, error_ids, _ = self._build_dataset()
        found, total = q.search_traces(min_duration_ms=150, limit=1000)
        self.assertEqual(total, len(error_ids))
        self.assertEqual({t.trace_id for t in found}, set(error_ids))

    def test_sort_by_duration_desc(self):
        """按耗时倒序，最前面应该是错误 trace（它们耗时更长）。"""
        q, _, error_ids, _ = self._build_dataset()
        found, _ = q.search_traces(sort=TraceQueryService.SORT_DURATION_DESC, limit=2)
        for t in found:
            self.assertIn(t.trace_id, error_ids)

    def test_combined_filters(self):
        """服务名 + has_error + min_duration 组合过滤。"""
        q, _, error_ids, _ = self._build_dataset()
        found, total = q.search_traces(
            service_name="pay-svc", has_error=True, min_duration_ms=150, limit=1000
        )
        self.assertEqual(total, len(error_ids))
        self.assertEqual({t.trace_id for t in found}, set(error_ids))


class TestJsonProcessDedup(unittest.TestCase):
    """测试 JSON 输出里同一服务只对应一个 process。"""

    def test_single_process_per_service(self):
        storage = InMemoryStorage()
        reconstructor = TraceReconstructor()

        def on_complete(trace_id, spans):
            storage.save_trace(trace_id, spans)
            reconstructor.add_spans(spans)

        collector = SpanCollector(post_root_idle_time=0.1, on_trace_complete=on_complete)
        t1 = Tracer("gateway", collector=collector, sampler=AlwaysOnSampler())
        t2 = Tracer("pay-svc", collector=collector)
        collector.start()

        with t1.start_active_span("root") as root:
            headers = {}
            t1.inject(root.context, headers)
            ctx = t2.extract(headers)
            with t2.start_active_span("pay_a", context=ctx):
                pass
            with t2.start_active_span("pay_b", context=ctx):
                pass
            tid = root.trace_id

        collector.flush()
        collector.flush_trace(tid)
        time.sleep(0.5)
        collector.stop()

        q = TraceQueryService(storage, reconstructor)
        data = json.loads(q.get_trace_json(tid))

        processes = data["processes"]
        service_names = [p["serviceName"] for p in processes.values()]
        self.assertEqual(len(service_names), len(set(service_names)))
        self.assertIn("gateway", service_names)
        self.assertIn("pay-svc", service_names)

        pids = [s["processID"] for s in data["spans"] if s["operationName"].startswith("pay_")]
        self.assertEqual(len(set(pids)), 1, "同一服务的多个 span 应共享同一个 processID")


class TestBatchImportExport(unittest.TestCase):
    """测试批量导入导出。"""

    def _make_data(self):
        storage = InMemoryStorage()
        reconstructor = TraceReconstructor()

        def on_complete(trace_id, spans):
            storage.save_trace(trace_id, spans)
            reconstructor.add_spans(spans)

        collector = SpanCollector(post_root_idle_time=0.1, on_trace_complete=on_complete)
        t1 = Tracer("svc-a", collector=collector, sampler=AlwaysOnSampler())
        t2 = Tracer("svc-b", collector=collector)
        collector.start()

        tids = []
        for i in range(3):
            with t1.start_active_span(f"a_{i}") as root:
                tids.append(root.trace_id)
                headers = {}
                t1.inject(root.context, headers)
                ctx = t2.extract(headers)
                with t2.start_active_span(f"b_{i}", context=ctx) as b:
                    if i == 0:
                        b.set_error("x")

        collector.flush()
        for tid in tids:
            collector.flush_trace(tid)
        time.sleep(0.5)
        collector.stop()

        return TraceQueryService(storage, reconstructor), tids

    def test_export_spans_and_import_back(self):
        """导出原始 span JSON，清空后再导入，数据一致。"""
        q, tids = self._make_data()

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            q.export_spans_to_json(trace_ids=tids, file_path=path)

            storage2 = InMemoryStorage()
            rec2 = TraceReconstructor()
            q2 = TraceQueryService(storage2, rec2)
            span_count, trace_count = q2.import_spans_from_json(path)

            self.assertEqual(trace_count, 3)
            self.assertGreater(span_count, 0)

            for tid in tids:
                orig = json.loads(q.get_trace_json(tid))
                new = json.loads(q2.get_trace_json(tid))
                self.assertEqual(orig["traceID"], new["traceID"])
                self.assertEqual(len(orig["spans"]), len(new["spans"]))
        finally:
            os.unlink(path)

    def test_export_single_trace(self):
        q, tids = self._make_data()
        js = q.export_trace_to_json(tids[0])
        self.assertIsNotNone(js)
        data = json.loads(js)
        self.assertIn("traceID", data)
        self.assertIn("spans", data)
        self.assertIn("processes", data)

    def test_import_array_format(self):
        """直接 span 数组格式也能导入。"""
        q, tids = self._make_data()

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            spans_data = []
            for tid in tids:
                for s in q.storage.get_trace(tid):
                    spans_data.append(s.to_dict())
            json.dump(spans_data, f)
            path = f.name

        try:
            storage2 = InMemoryStorage()
            rec2 = TraceReconstructor()
            q2 = TraceQueryService(storage2, rec2)
            n_spans, n_traces = q2.import_spans_from_json(path)
            self.assertEqual(n_traces, 3)
            self.assertEqual(n_spans, len(spans_data))
        finally:
            os.unlink(path)


class TestServiceOperationSampler(unittest.TestCase):
    """测试按服务/操作粒度的采样配置。"""

    def test_different_rate_per_service_operation(self):
        sampler = ServiceOperationSampler(
            service_operation_rates={
                ("gateway", "pay"): 1.0,
                ("gateway", "health"): 0.0,
            },
            default_rate=0.0,
        )
        tracer = Tracer("gateway", sampler=sampler)

        sampled_pay = 0
        sampled_health = 0
        for _ in range(50):
            with tracer.start_active_span("pay") as s:
                if s.context.sampled:
                    sampled_pay += 1
            with tracer.start_active_span("health") as s:
                if s.context.sampled:
                    sampled_health += 1

        self.assertEqual(sampled_pay, 50)
        self.assertEqual(sampled_health, 0)

    def test_service_default_rate(self):
        sampler = ServiceOperationSampler(
            service_default_rates={"gateway": 1.0, "other": 0.0},
            default_rate=0.0,
        )
        t1 = Tracer("gateway", sampler=sampler)
        t2 = Tracer("other", sampler=sampler)

        for _ in range(20):
            with t1.start_active_span("x") as s:
                self.assertTrue(s.context.sampled)
            with t2.start_active_span("y") as s:
                self.assertFalse(s.context.sampled)

    def test_deterministic_same_trace_id(self):
        sampler = ServiceOperationSampler(
            service_operation_rates={("g", "op"): 0.5},
            default_rate=0.5,
        )
        trace_id = TraceContext.generate_trace_id()
        results = [
            sampler.should_sample_by_trace_id(trace_id, operation_name="op", service_name="g")
            for _ in range(50)
        ]
        self.assertEqual(len(set(results)), 1)

    def test_downstream_only_inherits(self):
        """下游服务不重新采样，直接继承入口结果。"""
        sampler = ServiceOperationSampler(
            service_operation_rates={
                ("entry", "all_on"): 1.0,
                ("entry", "all_off"): 0.0,
                ("downstream", "all_on"): 0.0,
            },
            default_rate=0.0,
        )
        entry = Tracer("entry", sampler=sampler)
        downstream = Tracer("downstream", sampler=sampler)

        all_ok = True
        for _ in range(30):
            with entry.start_active_span("all_on") as root:
                self.assertTrue(root.context.sampled)
                headers = {}
                entry.inject(root.context, headers)
                ctx = downstream.extract(headers)
                with downstream.start_active_span("child", context=ctx) as c:
                    if not c.context.sampled:
                        all_ok = False
            with entry.start_active_span("all_off") as root:
                self.assertFalse(root.context.sampled)
                headers = {}
                entry.inject(root.context, headers)
                ctx = downstream.extract(headers)
                with downstream.start_active_span("child", context=ctx) as c:
                    if c.context.sampled:
                        all_ok = False

        self.assertTrue(all_ok, "下游服务应直接继承入口的采样结果，不重新判断")

    def test_per_operation_sampler_with_trace_id(self):
        sampler = PerOperationSampler({"pay": 1.0, "health": 0.0}, default_rate=0.0)
        tracer = Tracer("svc", sampler=sampler)
        with tracer.start_active_span("pay") as s:
            self.assertTrue(s.context.sampled)
        with tracer.start_active_span("health") as s:
            self.assertFalse(s.context.sampled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
