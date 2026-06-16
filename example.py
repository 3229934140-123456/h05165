import time
import random
import threading
from typing import List, Dict

from context_propagation import TraceContext, ContextManager
from span import Tracer, Span
from sampler import ProbabilisticSampler, AlwaysOnSampler
from collector import SpanCollector, InMemoryStorage
from trace_reconstructor import TraceReconstructor, Trace


def simulate_service_a(tracer: Tracer, collector: SpanCollector) -> Span:
    """
    模拟服务 A（入口服务）。
    作为入口服务，它负责生成 trace ID 和做出采样决策。
    """
    print("\n=== 服务 A 处理请求 ===")

    with tracer.start_active_span("http_request") as root_span:
        root_span.set_tag("http.method", "GET")
        root_span.set_tag("http.url", "/api/v1/order")
        root_span.set_tag("user_id", "12345")
        root_span.log("request_received", payload_size=1024)

        time.sleep(0.01)

        print(f"  生成 Trace ID: {root_span.trace_id}")
        print(f"  采样决策: {'采样' if root_span.context.sampled else '不采样'}")
        print(f"  根 Span ID: {root_span.span_id}")

        with tracer.start_active_span("validate_request") as validate_span:
            validate_span.set_tag("validation.type", "auth")
            time.sleep(0.005)
            validate_span.log("validation_passed")

        headers = {}
        tracer.inject(root_span.context, headers)
        print(f"  传播头部: {headers}")

        simulate_service_b(tracer, collector, headers)

        time.sleep(0.01)
        root_span.log("response_sent", status_code=200)

        return root_span


def simulate_service_b(tracer: Tracer, collector: SpanCollector, headers: Dict) -> Span:
    """
    模拟服务 B（订单服务）。
    从 HTTP 头部恢复 trace 上下文，继承采样决策。
    """
    print("\n=== 服务 B 处理请求 ===")

    context = tracer.extract(headers)
    if context:
        print(f"  从头部恢复 Trace ID: {context.trace_id}")
        print(f"  继承采样决策: {'采样' if context.sampled else '不采样'}")
    else:
        print("  未找到 trace 上下文，创建新根")

    with tracer.start_active_span("create_order", context=context) as order_span:
        order_span.set_tag("order.id", "ORD-001")
        order_span.set_tag("order.amount", 99.99)
        order_span.log("order_created")

        time.sleep(0.01)

        sub_headers = {}
        tracer.inject(order_span.context, sub_headers)

        simulate_service_c(tracer, collector, sub_headers)

        simulate_service_d(tracer, collector, sub_headers)

        time.sleep(0.005)
        order_span.log("order_completed")

        return order_span


def simulate_service_c(tracer: Tracer, collector: SpanCollector, headers: Dict) -> Span:
    """
    模拟服务 C（支付服务）。
    """
    print("\n=== 服务 C 处理请求 ===")

    context = tracer.extract(headers)

    with tracer.start_active_span("process_payment", context=context) as payment_span:
        payment_span.set_tag("payment.method", "credit_card")
        payment_span.set_tag("payment.amount", 99.99)
        payment_span.log("payment_started")

        time.sleep(0.02)

        if random.random() < 0.3:
            payment_span.set_error("支付超时")
            payment_span.log("payment_failed", error_code="TIMEOUT")
        else:
            payment_span.log("payment_succeeded", transaction_id="TXN-12345")

        return payment_span


def simulate_service_d(tracer: Tracer, collector: SpanCollector, headers: Dict) -> Span:
    """
    模拟服务 D（库存服务）。
    """
    print("\n=== 服务 D 处理请求 ===")

    context = tracer.extract(headers)

    with tracer.start_active_span("reserve_inventory", context=context) as inventory_span:
        inventory_span.set_tag("product.id", "PROD-001")
        inventory_span.set_tag("product.quantity", 2)
        inventory_span.log("inventory_check_started")

        time.sleep(0.01)

        inventory_span.log("inventory_reserved", remaining_stock=98)

        return inventory_span


def simulate_out_of_order_spans(reconstructor: TraceReconstructor, trace_id: str) -> None:
    """
    模拟乱序到达的 span，特别是父 span 比子 span 晚到的情况。
    """
    print("\n=== 模拟乱序 span 到达 ===")

    base_time = time.time()

    trace_id = trace_id or TraceContext.generate_trace_id()
    span_id_root = TraceContext.generate_span_id()
    span_id_child1 = TraceContext.generate_span_id()
    span_id_child2 = TraceContext.generate_span_id()
    span_id_grandchild = TraceContext.generate_span_id()

    ctx_root = TraceContext(
        trace_id=trace_id,
        span_id=span_id_root,
        parent_span_id=None,
        sampled=True,
    )

    ctx_child1 = TraceContext(
        trace_id=trace_id,
        span_id=span_id_child1,
        parent_span_id=span_id_root,
        sampled=True,
    )

    ctx_child2 = TraceContext(
        trace_id=trace_id,
        span_id=span_id_child2,
        parent_span_id=span_id_root,
        sampled=True,
    )

    ctx_grandchild = TraceContext(
        trace_id=trace_id,
        span_id=span_id_grandchild,
        parent_span_id=span_id_child1,
        sampled=True,
    )

    span_grandchild = Span("grandchild_op", ctx_grandchild, service_name="service-e")
    span_grandchild.start_time = base_time + 0.03
    span_grandchild.finish(end_time=base_time + 0.05)

    span_child2 = Span("child2_op", ctx_child2, service_name="service-c")
    span_child2.start_time = base_time + 0.02
    span_child2.finish(end_time=base_time + 0.04)

    span_child1 = Span("child1_op", ctx_child1, service_name="service-b")
    span_child1.start_time = base_time + 0.01
    span_child1.finish(end_time=base_time + 0.06)

    span_root = Span("root_op", ctx_root, service_name="service-a")
    span_root.start_time = base_time
    span_root.finish(end_time=base_time + 0.07)

    print("  按以下顺序上报 span:")
    print("    1. 孙 span (grandchild_op) - 父 span 还未到")
    print("    2. 子 span 2 (child2_op) - 父 span 还未到")
    print("    3. 子 span 1 (child1_op) - 父 span 还未到")
    print("    4. 根 span (root_op) - 最后到")

    print("\n  第 1 步: 上报孙 span")
    traces = reconstructor.add_spans([span_grandchild])
    for trace in traces:
        print(f"    Trace 状态: 完整={trace.is_complete}, 孤立 span={trace.get_orphan_count()}")
        print(f"    缺失的父 span: {trace.get_missing_parent_ids()}")

    print("\n  第 2 步: 上报子 span 2")
    traces = reconstructor.add_spans([span_child2])
    for trace in traces:
        print(f"    Trace 状态: 完整={trace.is_complete}, 孤立 span={trace.get_orphan_count()}")
        print(f"    缺失的父 span: {trace.get_missing_parent_ids()}")

    print("\n  第 3 步: 上报子 span 1")
    traces = reconstructor.add_spans([span_child1])
    for trace in traces:
        print(f"    Trace 状态: 完整={trace.is_complete}, 孤立 span={trace.get_orphan_count()}")
        print(f"    缺失的父 span: {trace.get_missing_parent_ids()}")

    print("\n  第 4 步: 上报根 span")
    traces = reconstructor.add_spans([span_root])
    for trace in traces:
        print(f"    Trace 状态: 完整={trace.is_complete}, 孤立 span={trace.get_orphan_count()}")
        if trace.is_complete:
            print("    ✅ Trace 已完整!")
            trace.print_tree()


def demonstrate_sampling_tradeoff() -> None:
    """
    演示采样率与数据量的权衡。
    """
    print("\n=== 采样率与数据量权衡 ===")

    total_requests = 10000
    avg_spans_per_request = 10
    avg_span_size_bytes = 512

    print(f"  假设条件:")
    print(f"    - 总请求数: {total_requests:,} 次/秒")
    print(f"    - 平均每请求 span 数: {avg_spans_per_request}")
    print(f"    - 平均 span 大小: {avg_span_size_bytes} bytes")

    print("\n  不同采样率下的数据量:")
    for rate in [1.0, 0.5, 0.1, 0.01, 0.001]:
        sampled_requests = int(total_requests * rate)
        spans_per_second = sampled_requests * avg_spans_per_request
        bytes_per_second = spans_per_second * avg_span_size_bytes
        mb_per_hour = (bytes_per_second * 3600) / (1024 * 1024)

        print(
            f"    采样率 {rate:7.1%}: "
            f"{sampled_requests:6,} 请求/秒, "
            f"{spans_per_second:7,} spans/秒, "
            f"{mb_per_hour:8.1f} MB/小时"
        )

    print("\n  权衡分析:")
    print("    100% 采样: 数据最完整，但存储和网络成本最高")
    print("    10% 采样: 成本降低 90%，仍能捕捉大部分性能问题")
    print("    1% 采样: 成本很低，适合高流量服务做整体趋势分析")
    print("    0.1% 采样: 成本极低，仅适合超大规模服务的异常检测")

    print("\n  建议策略:")
    print("    - 关键业务链路（支付、下单）: 100% 采样")
    print("    - 核心服务: 10% ~ 50% 采样")
    print("    - 非核心服务: 1% ~ 10% 采样")
    print("    - 高流量边缘服务: 0.1% ~ 1% 采样")
    print("    - 错误请求: 强制 100% 采样（使用组合采样器）")


def main():
    print("=" * 60)
    print("分布式追踪系统核心 - 完整演示")
    print("=" * 60)

    sampler = AlwaysOnSampler()
    storage = InMemoryStorage()
    reconstructor = TraceReconstructor()

    def on_trace_complete(trace_id: str, spans: List[Span]) -> None:
        print(f"\n[收集器] Trace {trace_id} 已收集 {len(spans)} 个 span")
        storage.save_trace(trace_id, spans)
        traces = reconstructor.add_spans(spans)
        for trace in traces:
            if trace.is_complete:
                print(f"[重组器] Trace {trace_id} 重组完成")

    collector = SpanCollector(
        batch_size=10,
        flush_interval=0.5,
        on_trace_complete=on_trace_complete,
    )

    tracer_a = Tracer("gateway-service", collector=collector, sampler=sampler)
    tracer_b = Tracer("order-service", collector=collector, sampler=sampler)
    tracer_c = Tracer("payment-service", collector=collector, sampler=sampler)
    tracer_d = Tracer("inventory-service", collector=collector, sampler=sampler)

    collector.start()

    print("\n--- 演示 1: 正常调用链追踪 ---")
    root_span = simulate_service_a(tracer_a, collector)

    time.sleep(1.0)
    collector.flush()
    time.sleep(0.5)

    stored_spans = storage.get_trace(root_span.trace_id)
    if stored_spans:
        print(f"\n--- 从存储中获取 Trace {root_span.trace_id} ---")
        trace = Trace(root_span.trace_id, stored_spans)
        trace.print_tree()

        print(f"\n  关键路径:")
        critical_path = trace.get_critical_path()
        for i, span in enumerate(critical_path):
            duration = f"{span.duration * 1000:.2f}ms" if span.duration else "N/A"
            print(f"    {i + 1}. [{span.service_name}] {span.operation_name} ({duration})")

    print("\n--- 演示 2: 乱序 span 处理 ---")
    simulate_out_of_order_spans(reconstructor, None)

    print("\n--- 演示 3: 概率采样 ---")
    prob_sampler = ProbabilisticSampler(rate=0.3)
    sampled_count = 0
    total_count = 100
    for i in range(total_count):
        if prob_sampler.should_sample():
            sampled_count += 1
    print(f"  采样率 30%，实际 {total_count} 次采样 {sampled_count} 次 ({sampled_count / total_count:.1%})")

    demonstrate_sampling_tradeoff()

    print("\n" + "=" * 60)
    print("收集器统计:")
    stats = collector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n重组器统计:")
    stats = reconstructor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n存储统计:")
    print(f"  存储的 trace 数: {len(storage.get_all_trace_ids())}")

    collector.stop()
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
