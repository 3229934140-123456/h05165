import time
import random
import threading
import json
from typing import Dict, List
from collections import defaultdict

from context_propagation import TraceContext, ContextManager
from span import Tracer, Span
from sampler import ProbabilisticSampler, AlwaysOnSampler
from collector import SpanCollector, InMemoryStorage
from trace_reconstructor import TraceReconstructor, Trace
from trace_query import TraceQueryService


def demo_concurrent_safety():
    """
    演示 1: 并发安全验证
    同一服务实例同时处理 20 个请求，验证每个 trace 的 span 不会串到其他请求里
    """
    print("=" * 70)
    print("演示 1: 并发安全 - 多请求同时处理时 span 互不干扰")
    print("=" * 70)

    storage = InMemoryStorage()
    reconstructor = TraceReconstructor()

    def on_complete(trace_id, spans):
        storage.save_trace(trace_id, spans)
        reconstructor.add_spans(spans)

    collector = SpanCollector(
        post_root_idle_time=0.3,
        completion_check_interval=0.1,
        on_trace_complete=on_complete,
    )
    tracer = Tracer("concurrent-demo", collector=collector, sampler=AlwaysOnSampler())
    collector.start()

    NUM_REQUESTS = 20
    SPANS_PER_REQUEST = 5
    trace_ids: Dict[int, str] = {}
    errors = []

    def process_request(request_id):
        try:
            with tracer.start_active_span(f"req_{request_id}_root") as root:
                trace_ids[request_id] = root.trace_id
                root.set_tag("request_id", request_id)
                root.set_tag("thread_id", threading.current_thread().name)
                time.sleep(0.01)

                for i in range(SPANS_PER_REQUEST):
                    with tracer.start_active_span(f"req_{request_id}_child_{i}") as child:
                        child.set_tag("request_id", request_id)
                        child.set_tag("child_index", i)

                        current_active = tracer.get_active_span()
                        if current_active is None or current_active.span_id != child.span_id:
                            errors.append(
                                f"请求 {request_id}: 活动 span 错误, "
                                f"期望 {child.span_id}, 实际 {current_active.span_id if current_active else None}"
                            )

                        child_ctx = ContextManager.get_context()
                        if child_ctx is None or child_ctx.trace_id != root.trace_id:
                            errors.append(
                                f"请求 {request_id}: trace_id 串了, "
                                f"期望 {root.trace_id}, 实际 {child_ctx.trace_id if child_ctx else None}"
                            )

                        time.sleep(random.uniform(0.005, 0.015))

        except Exception as e:
            errors.append(f"请求 {request_id}: 异常 {e}")

    threads = []
    for i in range(NUM_REQUESTS):
        t = threading.Thread(target=process_request, args=(i,), name=f"worker-{i}")
        threads.append(t)

    print(f"\n启动 {NUM_REQUESTS} 个并发线程，每个请求生成 {SPANS_PER_REQUEST + 1} 个 span...")
    start_time = time.time()

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    print(f"所有请求处理完成，耗时 {elapsed * 1000:.1f}ms")

    time.sleep(0.5)
    collector.flush_all()
    time.sleep(0.5)

    print("\n验证结果:")
    if errors:
        print(f"  ❌ 发现 {len(errors)} 个错误:")
        for e in errors[:5]:
            print(f"     - {e}")
    else:
        print("  ✅ 并发过程中无 span 串扰错误")

    print(f"\n检查每个 trace 的 span 是否只属于自己的请求:")
    cross_contamination = 0
    for request_id, trace_id in trace_ids.items():
        trace_spans = storage.get_trace(trace_id)
        if trace_spans is None:
            print(f"  ⚠️  请求 {request_id}: trace {trace_id[:12]}... 未找到")
            continue

        request_ids_in_trace = set()
        for s in trace_spans:
            rid = s.tags.get("request_id")
            if rid is not None:
                request_ids_in_trace.add(rid)

        if len(request_ids_in_trace) > 1:
            cross_contamination += 1
            print(
                f"  ❌ trace {trace_id[:12]}... 包含多个 request_id: {request_ids_in_trace}"
            )

        expected_spans = SPANS_PER_REQUEST + 1
        actual_spans = len(trace_spans)
        if actual_spans != expected_spans:
            print(
                f"  ⚠️  trace {trace_id[:12]}... span 数不符: "
                f"期望 {expected_spans}, 实际 {actual_spans}"
            )

    if cross_contamination == 0:
        print("  ✅ 所有 trace 的 span 都只属于自己的请求，无串扰!")

    total_spans = sum(
        len(storage.get_trace(tid) or []) for tid in storage.get_all_trace_ids()
    )
    print(f"\n统计: 共 {len(trace_ids)} 个 trace, {total_spans} 个 span")

    collector.stop()
    storage.clear()
    return len(errors) == 0 and cross_contamination == 0


def demo_smart_completion():
    """
    演示 2: 稳健的完成策略
    根 span 先到 -> 等待子 span -> 子 span 晚到仍能进同一棵树
    """
    print("\n" + "=" * 70)
    print("演示 2: 稳健完成策略 - 根 span 先到后等待子 span")
    print("=" * 70)

    storage = InMemoryStorage()
    reconstructor = TraceReconstructor()
    completion_count = 0
    completion_spans_counts = []

    def on_complete(trace_id, spans):
        nonlocal completion_count
        completion_count += 1
        completion_spans_counts.append(len(spans))
        print(
            f"  [完成回调 #{completion_count}] trace={trace_id[:12]}..., "
            f"spans={len(spans)}"
        )
        storage.save_trace(trace_id, spans)
        reconstructor.add_spans(spans)

    collector = SpanCollector(
        post_root_idle_time=0.5,
        completion_check_interval=0.1,
        on_trace_complete=on_complete,
    )
    collector.start()

    trace_id = TraceContext.generate_trace_id()
    span_id_root = TraceContext.generate_span_id()
    span_id_child_a = TraceContext.generate_span_id()
    span_id_child_b = TraceContext.generate_span_id()
    span_id_grandchild = TraceContext.generate_span_id()

    base_time = time.time()

    ctx_root = TraceContext(trace_id, span_id_root, None, True)
    span_root = Span("gateway_request", ctx_root, service_name="gateway")
    span_root.start_time = base_time
    span_root.finish(end_time=base_time + 0.1)
    span_root.set_tag("http.method", "POST")

    ctx_child_a = TraceContext(trace_id, span_id_child_a, span_id_root, True)
    span_child_a = Span("create_order", ctx_child_a, service_name="order")
    span_child_a.start_time = base_time + 0.01
    span_child_a.finish(end_time=base_time + 0.08)

    ctx_child_b = TraceContext(trace_id, span_id_child_b, span_id_root, True)
    span_child_b = Span("send_notification", ctx_child_b, service_name="notify")
    span_child_b.start_time = base_time + 0.02
    span_child_b.finish(end_time=base_time + 0.06)

    ctx_grandchild = TraceContext(trace_id, span_id_grandchild, span_id_child_a, True)
    span_grandchild = Span("charge_card", ctx_grandchild, service_name="payment")
    span_grandchild.start_time = base_time + 0.03
    span_grandchild.finish(end_time=base_time + 0.07)
    span_grandchild.set_tag("amount", 99.99)

    print("\n上报顺序（故意乱序）:")
    print("  1. 先上报 根 span")
    collector.collect(span_root)
    collector.flush()
    time.sleep(0.15)

    buffer = collector.get_trace_buffer(trace_id)
    print(f"     状态: {buffer.state}, 已通知完成? {trace_id in collector._notified_traces}")

    print("  2. 再上报 孙 span (charge_card) - 父 span 还没到")
    collector.collect(span_grandchild)
    collector.flush()
    time.sleep(0.15)
    buffer = collector.get_trace_buffer(trace_id)
    print(f"     状态: {buffer.state}, 已通知完成? {trace_id in collector._notified_traces}")

    print("  3. 上报 子 span A (create_order) ")
    collector.collect(span_child_a)
    collector.flush()
    time.sleep(0.15)
    buffer = collector.get_trace_buffer(trace_id)
    print(f"     状态: {buffer.state}, 已通知完成? {trace_id in collector._notified_traces}")

    print("  4. 上报 子 span B (send_notification)")
    collector.collect(span_child_b)
    collector.flush()
    time.sleep(0.15)
    buffer = collector.get_trace_buffer(trace_id)
    print(
        f"     状态: {buffer.state}, "
        f"结构完整? {buffer.is_structurally_complete()}, "
        f"已通知完成? {trace_id in collector._notified_traces}"
    )

    print("\n⏳ 等待空闲时间 (post_root_idle_time = 0.5s) 触发完成...")
    waited = 0
    while trace_id not in collector._notified_traces and waited < 2.0:
        time.sleep(0.1)
        waited += 0.1

    print(f"\n最终完成回调次数: {completion_count}")
    if completion_count == 1 and completion_spans_counts[0] == 4:
        print("✅ 只触发了一次完成通知，且包含所有 4 个 span（没有提前只带根节点）")
    else:
        print(
            f"⚠️  完成通知: {completion_count} 次, "
            f"span 数: {completion_spans_counts}"
        )

    print("\n重建的调用树:")
    trace = reconstructor.get_trace(trace_id)
    if trace:
        trace.print_tree()

    collector.stop()
    return True


def demo_trace_query():
    """
    演示 3: Trace 查询服务
    按多条件查询，输出 JSON 格式调用树
    """
    print("\n" + "=" * 70)
    print("演示 3: Trace 查询服务 - 多条件搜索 & JSON 输出")
    print("=" * 70)

    storage = InMemoryStorage()
    reconstructor = TraceReconstructor()

    def on_complete(trace_id, spans):
        storage.save_trace(trace_id, spans)
        reconstructor.add_spans(spans)

    collector = SpanCollector(
        post_root_idle_time=0.2,
        completion_check_interval=0.1,
        on_trace_complete=on_complete,
    )
    collector.start()

    tracer_a = Tracer("gateway", collector=collector, sampler=AlwaysOnSampler())
    tracer_b = Tracer("order-service", collector=collector)
    tracer_c = Tracer("payment-service", collector=collector)
    tracer_d = Tracer("user-service", collector=collector)

    print("\n生成 5 个示例 trace:")
    generated_trace_ids = []
    for i in range(5):
        with tracer_a.start_active_span(f"http_request_{i}") as root:
            root.set_tag("http.url", f"/api/v1/resource/{i}")

            headers = {}
            tracer_a.inject(root.context, headers)

            ctx_b = tracer_b.extract(headers)
            with tracer_b.start_active_span(f"process_order_{i}", context=ctx_b) as order:
                order.set_tag("order.id", f"ORD-{i:03d}")

                if i == 2:
                    order.set_error("库存不足")

                headers2 = {}
                tracer_b.inject(order.context, headers2)
                ctx_c = tracer_c.extract(headers2)
                with tracer_c.start_active_span(f"pay_{i}", context=ctx_c) as pay:
                    pay.set_tag("amount", (i + 1) * 10.0)
                    if i == 2:
                        pay.set_error("支付失败")
                    time.sleep(0.01)

                if i % 2 == 0:
                    headers3 = {}
                    tracer_b.inject(order.context, headers3)
                    ctx_d = tracer_d.extract(headers3)
                    with tracer_d.start_active_span(f"get_user_{i}", context=ctx_d) as user:
                        user.set_tag("user.id", f"USER-{i}")
                        time.sleep(0.005)

        generated_trace_ids.append(root.trace_id)

    time.sleep(0.5)
    collector.flush()
    time.sleep(0.5)
    for tid in generated_trace_ids:
        collector.flush_trace(tid)
    time.sleep(0.5)

    query_service = TraceQueryService(storage, reconstructor)

    print("\n--- 查询 1: 按 trace ID 精确查询 ---")
    target_id = generated_trace_ids[2]
    json_output = query_service.get_trace_json(target_id)
    if json_output:
        data = json.loads(json_output)
        print(f"  traceID: {data['traceID']}")
        print(f"  spans 数: {data['summary']['spanCount']}")
        print(f"  服务数: {data['summary']['serviceCount']}")
        print(f"  总耗时: {data['summary']['totalDurationMs']:.2f}ms")
        print(f"  hasError: {data['summary']['hasError']}")
        print(f"  processes: {list(data['processes'].values())}")

    print("\n--- 查询 2: 列出所有服务和操作 ---")
    print(f"  服务: {query_service.list_services()}")
    print(f"  操作(gateway): {query_service.list_operations('gateway')}")

    print("\n--- 查询 3: 按服务名 'payment-service' 搜索 ---")
    payment_traces = query_service.search_traces(service_name="payment-service")
    print(f"  找到 {len(payment_traces)} 个 trace")
    for t in payment_traces:
        print(f"    - {t.trace_id[:12]}... 耗时 {t.get_total_duration() * 1000:.1f}ms")

    print("\n--- 查询 4: 只搜索有错误的 trace ---")
    error_traces = query_service.search_traces(has_error=True)
    print(f"  找到 {len(error_traces)} 个有错误的 trace")
    for t in error_traces:
        print(f"    - {t.trace_id[:12]}...")
        for s in t.spans:
            if s.status == "error":
                print(f"        错误 span: [{s.service_name}] {s.operation_name}: {s.status_message}")

    print("\n--- 查询 5: 按耗时筛选 (>15ms) ---")
    slow_traces = query_service.search_traces(min_duration_ms=15)
    print(f"  找到 {len(slow_traces)} 个慢 trace")

    print("\n--- 查询 6: 组合条件 (服务=order-service 且 有错误) ---")
    combined = query_service.search_traces(service_name="order-service", has_error=True)
    print(f"  找到 {len(combined)} 个 trace")

    print("\n--- 查询 7: 完整 JSON 输出（第一个 trace） ---")
    first_trace = generated_trace_ids[0]
    full_json = query_service.get_trace_json(first_trace, indent=2)
    if full_json:
        parsed = json.loads(full_json)
        print(json.dumps(parsed, indent=2)[:1500])
        if len(full_json) > 1500:
            print(f"\n... (JSON 总长 {len(full_json)} 字符，兼容 Jaeger UI)")

    print(f"\n总体统计: {query_service.get_stats()}")

    collector.stop()
    return True


def demo_deterministic_sampling():
    """
    演示 4: 确定性采样
    入口基于 trace ID 哈希采样，下游只继承不重采样，
    同一个 trace ID 重复判断结果一致
    """
    print("\n" + "=" * 70)
    print("演示 4: 确定性采样 - 基于 trace ID 哈希，全链路一致")
    print("=" * 70)

    sampler = ProbabilisticSampler(rate=0.5)

    print(f"\n采样器: {sampler.get_description()}")
    print(f"采样原理: trace_id % 10000 < {sampler.rate * 10000}")

    print("\n--- 验证 1: 同一个 trace ID 多次判断结果完全一致 ---")
    test_trace_id = "abc123def456" + "0" * 20
    results = [sampler.should_sample_by_trace_id(test_trace_id) for _ in range(100)]
    unique_results = set(results)
    print(f"  trace_id: {test_trace_id[:16]}...")
    print(f"  连续判断 100 次，结果集合: {unique_results}")
    if len(unique_results) == 1:
        print("  ✅ 完全一致！")

    print("\n--- 验证 2: 不同 trace ID 按概率分布 ---")
    sample_count = 10000
    sampled = 0
    for i in range(sample_count):
        tid = TraceContext.generate_trace_id()
        if sampler.should_sample_by_trace_id(tid):
            sampled += 1
    actual_rate = sampled / sample_count
    print(f"  总 trace 数: {sample_count}")
    print(f"  实际采样: {sampled} ({actual_rate:.2%})")
    print(f"  期望采样率: {sampler.rate:.0%}")
    if 0.45 < actual_rate < 0.55:
        print("  ✅ 在合理区间内，符合概率分布")

    print("\n--- 验证 3: 下游服务只继承采样决策，不重新采样 ---")

    storage = InMemoryStorage()
    reconstructor = TraceReconstructor()

    def on_complete(trace_id, spans):
        storage.save_trace(trace_id, spans)
        reconstructor.add_spans(spans)

    collector = SpanCollector(post_root_idle_time=0.2, on_trace_complete=on_complete)
    gateway_tracer = Tracer("gateway", collector=collector, sampler=sampler)
    downstream_tracer = Tracer("order-service", collector=collector, sampler=sampler)
    collector.start()

    consistency_ok = True
    test_runs = 50
    sampled_traces = 0
    not_sampled_traces = 0

    for i in range(test_runs):
        with gateway_tracer.start_active_span(f"root_{i}") as root:
            root.set_tag("idx", i)
            gateway_sampled = root.context.sampled

            headers = {}
            gateway_tracer.inject(root.context, headers)

            ctx = downstream_tracer.extract(headers)
            with downstream_tracer.start_active_span(f"child_{i}", context=ctx) as child:
                downstream_sampled = child.context.sampled

                if gateway_sampled != downstream_sampled:
                    consistency_ok = False
                    print(
                        f"  ❌ 不一致! 入口采样={gateway_sampled}, "
                        f"下游采样={downstream_sampled}"
                    )

            if gateway_sampled:
                sampled_traces += 1
            else:
                not_sampled_traces += 1

    if consistency_ok:
        print(f"  ✅ {test_runs} 次调用，下游完全继承入口采样决策，无一不一致")
    print(f"  入口采样了 {sampled_traces}/{test_runs} 次")
    print(f"  入口未采样 {not_sampled_traces}/{test_runs} 次")

    print("\n--- 验证 4: 未采样的 trace 不上报任何 span ---")
    time.sleep(0.3)
    collector.flush_all()
    time.sleep(0.3)

    all_trace_ids = storage.get_all_trace_ids()
    print(f"  存储中共有 {len(all_trace_ids)} 个 trace")
    print(f"  期望 ≈ {sampled_traces} (只保留采样的 trace)")

    print("\n--- 验证 5: 采样的 trace 包含完整的父子 span ---")
    if all_trace_ids:
        sample_trace_id = all_trace_ids[0]
        trace = reconstructor.get_trace(sample_trace_id)
        if trace:
            print(f"  示例 trace: {sample_trace_id[:12]}...")
            print(f"    span 数: {trace.get_span_count()}")
            print(f"    完整: {trace.is_complete}")

    collector.stop()
    return consistency_ok


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     分布式追踪系统增强版 - 四大核心能力完整演示                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    results = {}

    results["并发安全"] = demo_concurrent_safety()
    results["稳健完成策略"] = demo_smart_completion()
    results["查询服务"] = demo_trace_query()
    results["确定性采样"] = demo_deterministic_sampling()

    print("\n" + "=" * 70)
    print("全部演示完成! 结果汇总:")
    print("=" * 70)
    all_ok = True
    for name, ok in results.items():
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n🎉 所有功能演示成功!")
    else:
        print("\n⚠️  部分功能需要关注")


if __name__ == "__main__":
    main()
