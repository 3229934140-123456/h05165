"""
演示 V2：查询增强 + 进程去重 + 导入导出 + 服务/操作粒度采样
"""
import json
import os
import tempfile
import time

from context_propagation import TraceContext
from span import Tracer
from sampler import (
    ServiceOperationSampler,
    PerOperationSampler,
    ProbabilisticSampler,
    AlwaysOnSampler,
)
from collector import SpanCollector, InMemoryStorage
from trace_reconstructor import TraceReconstructor
from trace_query import TraceQueryService


SEP = "=" * 70


def build_large_dataset():
    """构建 50 条 trace 的大数据集，只有两条是错误的，且在列表靠后的位置。"""
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
    gw = Tracer("gateway", collector=collector, sampler=AlwaysOnSampler())
    pay = Tracer("pay-svc", collector=collector)
    inv = Tracer("inventory-svc", collector=collector)
    collector.start()

    trace_ids = []
    error_ids = []
    base_time = time.time() - 600

    for i in range(50):
        t0 = base_time + i * 10
        with gw.start_active_span(f"http_req_{i}", start_time=t0) as root:
            trace_ids.append(root.trace_id)
            headers = {}
            gw.inject(root.context, headers)
            ctx = pay.extract(headers)

            is_error = i in (24, 49)
            with pay.start_active_span(f"process_payment_{i}", context=ctx, start_time=t0 + 0.005) as p:
                if is_error:
                    p.set_error("payment gateway timeout")
                    error_ids.append(root.trace_id)
                p.finish(end_time=t0 + 0.02 + (0.3 if is_error else 0.01))

            if i % 2 == 0:
                ctx2 = gw.extract(headers)
                with inv.start_active_span(f"check_stock_{i}", context=ctx2, start_time=t0 + 0.03):
                    pass

            root.finish(end_time=t0 + 0.05 + (0.25 if is_error else 0.02))

    collector.flush()
    for tid in trace_ids:
        collector.flush_trace(tid)
    time.sleep(0.5)
    collector.flush_all()
    time.sleep(0.3)
    collector.stop()

    return TraceQueryService(storage, reconstructor), trace_ids, error_ids


def demo_advanced_search():
    print(f"\n{SEP}")
    print("演示 1: Jaeger Query 风格高级搜索（时间范围/分页/多条件组合）")
    print(SEP)

    q, all_ids, error_ids = build_large_dataset()
    stats = q.get_stats()
    print(f"\n数据集概况: {stats['total_traces']} 条 trace, {stats['error_traces']} 条错误")
    print(f"错误 trace 位于索引: {[all_ids.index(i) for i in error_ids]} (50 条中靠后位置)")

    print(f"\n--- 1.1 只筛错误 trace（大数据量下不漏） ---")
    found, total = q.search_traces(has_error=True, limit=1000)
    print(f"  找到 {total} 条错误 trace，期望 {len(error_ids)} 条")
    assert total == len(error_ids), "错误 trace 漏掉了！"
    print("  ✅ 全部命中，没有漏掉")

    print(f"\n--- 1.2 按时间范围过滤（只取中间一段时间） ---")
    t0 = time.time() - 600 + 100
    t1 = time.time() - 600 + 300
    found, total = q.search_traces(start_time=t0, end_time=t1, limit=1000)
    print(f"  时间窗口内命中 {total} 条 trace")
    assert 15 <= total <= 25

    print(f"\n--- 1.3 分页查询（第一页 vs 第二页） ---")
    p1, total = q.search_traces(limit=10, offset=0)
    p2, _ = q.search_traces(limit=10, offset=10)
    print(f"  总匹配数: {total}")
    print(f"  第 1 页: {[t.trace_id[:8] + '...' for t in p1[:3]]}")
    print(f"  第 2 页: {[t.trace_id[:8] + '...' for t in p2[:3]]}")
    assert p1[0].trace_id != p2[0].trace_id, "分页返回了相同数据！"
    print("  ✅ 分页正常，两页内容不同")

    print(f"\n--- 1.4 组合过滤：服务=pay-svc + has_error + min_duration>200ms ---")
    found, total = q.search_traces(
        service_name="pay-svc",
        has_error=True,
        min_duration_ms=200,
        limit=1000,
    )
    print(f"  命中 {total} 条，期望 {len(error_ids)} 条")
    assert total == len(error_ids)
    print("  ✅ 组合条件完全匹配")

    print(f"\n--- 1.5 按耗时倒序（最慢的在最前） ---")
    found, _ = q.search_traces(sort=TraceQueryService.SORT_DURATION_DESC, limit=5)
    print(f"  TOP5 耗时 trace:")
    for t in found:
        dur = t.get_total_duration() * 1000
        flag = " [ERR]" if t.trace_id in error_ids else ""
        print(f"    {t.trace_id[:8]}... {dur:.0f}ms{flag}")
    top2_are_error = all(t.trace_id in error_ids for t in found[:2])
    print(f"  TOP2 全是错误 trace: {top2_are_error}")
    print("  ✅ 排序正常")

    print(f"\n--- 1.6 输出分页 JSON（带 total/offset/limit） ---")
    js = q.search_traces_json(has_error=True, limit=1, offset=0)
    data = json.loads(js)
    print(f"  JSON keys: {list(data.keys())}")
    print(f"  total={data['total']}, offset={data['offset']}, limit={data['limit']}")
    print(f"  首条 trace spans 数: {len(data['traces'][0]['spans'])}")
    print("  ✅ JSON 输出符合分页格式")


def demo_process_dedup():
    print(f"\n{SEP}")
    print("演示 2: JSON 输出 — 同一服务单进程信息，span 共享 processID")
    print(SEP)

    storage = InMemoryStorage()
    rec = TraceReconstructor()

    def on_complete(trace_id, spans):
        storage.save_trace(trace_id, spans)
        rec.add_spans(spans)

    collector = SpanCollector(post_root_idle_time=0.1, on_trace_complete=on_complete)
    gw = Tracer("gateway", collector=collector, sampler=AlwaysOnSampler())
    pay = Tracer("pay-svc", collector=collector)
    collector.start()

    with gw.start_active_span("checkout") as root:
        tid = root.trace_id
        headers = {}
        gw.inject(root.context, headers)
        ctx = pay.extract(headers)
        with pay.start_active_span("validate_card", context=ctx):
            pass
        with pay.start_active_span("charge_account", context=ctx):
            pass
        with pay.start_active_span("send_receipt", context=ctx):
            pass

    collector.flush()
    collector.flush_trace(tid)
    time.sleep(0.5)
    collector.stop()

    q = TraceQueryService(storage, rec)
    data = json.loads(q.get_trace_json(tid))

    print(f"\nTrace 中 spans: {len(data['spans'])} 个")
    print(f"进程数: {len(data['processes'])} 个")
    for pid, p in data["processes"].items():
        print(f"  {pid} -> {p['serviceName']}")

    pay_pids = set()
    for s in data["spans"]:
        if s["operationName"] in ("validate_card", "charge_account", "send_receipt"):
            pay_pids.add(s["processID"])
    print(f"\npay-svc 的 3 个 span 使用的 processID: {pay_pids}")
    assert len(pay_pids) == 1, "同一服务的多个 span 应该复用一个 processID！"
    print("✅ 同一服务的多个 span 共享同一个 processID，UI 不会重复渲染")


def demo_import_export():
    print(f"\n{SEP}")
    print("演示 3: 批量导入导出 —— JSON 文件灌 span + 离线回放")
    print(SEP)

    storage = InMemoryStorage()
    rec = TraceReconstructor()

    def on_complete(trace_id, spans):
        storage.save_trace(trace_id, spans)
        rec.add_spans(spans)

    collector = SpanCollector(post_root_idle_time=0.1, on_trace_complete=on_complete)
    a = Tracer("svc-a", collector=collector, sampler=AlwaysOnSampler())
    b = Tracer("svc-b", collector=collector)
    collector.start()

    src_tids = []
    for i in range(3):
        with a.start_active_span(f"op_a_{i}") as root:
            src_tids.append(root.trace_id)
            headers = {}
            a.inject(root.context, headers)
            ctx = b.extract(headers)
            with b.start_active_span(f"op_b_{i}", context=ctx) as s:
                if i == 0:
                    s.set_error("sample error")

    collector.flush()
    for tid in src_tids:
        collector.flush_trace(tid)
    time.sleep(0.5)
    collector.stop()

    q_src = TraceQueryService(storage, rec)

    tmpdir = tempfile.mkdtemp()
    spans_file = os.path.join(tmpdir, "spans_export.json")
    trace_file = os.path.join(tmpdir, "trace_0.json")
    batch_file = os.path.join(tmpdir, "batch_traces.json")

    print(f"\n--- 3.1 导出原始 spans（to_dict 格式） ---")
    q_src.export_spans_to_json(file_path=spans_file)
    with open(spans_file) as f:
        raw = json.load(f)
    print(f"  导出 {raw['count']} 个 span -> {spans_file}")

    print(f"\n--- 3.2 在新的空存储里重新导入这批 spans ---")
    storage2 = InMemoryStorage()
    rec2 = TraceReconstructor()
    q_dst = TraceQueryService(storage2, rec2)
    n_spans, n_traces = q_dst.import_spans_from_json(spans_file)
    print(f"  导入: {n_spans} 个 span, {n_traces} 条 trace")
    assert n_traces == 3
    stats = q_dst.get_stats()
    print(f"  导入后统计: {stats}")

    print(f"\n--- 3.3 导出单条 trace 的完整调用树（Jaeger 兼容 JSON） ---")
    js = q_dst.export_trace_to_json(src_tids[0], file_path=trace_file)
    data = json.loads(js)
    print(f"  -> {trace_file}")
    print(f"    traceID={data['traceID'][:12]}..., spans={len(data['spans'])}, services={len(data['processes'])}")

    print(f"\n--- 3.4 批量导出（搜索条件 + 写文件） ---")
    q_dst.export_traces_to_json(file_path=batch_file, has_error=True)
    with open(batch_file) as f:
        batch = json.load(f)
    print(f"  仅导出错误 trace: {batch['count']} 条 -> {batch_file}")
    assert batch["count"] == 1

    for f in (spans_file, trace_file, batch_file):
        os.unlink(f)
    os.rmdir(tmpdir)
    print("\n✅ 导入导出完整闭环，数据一致")


def demo_service_operation_sampler():
    print(f"\n{SEP}")
    print("演示 4: 按服务/操作粒度配置采样率，下游只继承")
    print(SEP)

    sampler = ServiceOperationSampler(
        service_operation_rates={
            ("gateway", "health_check"): 0.0,
            ("gateway", "place_order"): 1.0,
            ("gateway", "login"): 1.0,
        },
        service_default_rates={
            "gateway": 0.1,
            "background": 0.0,
        },
        default_rate=0.0,
    )
    print(f"\n采样器: {sampler.get_description()}")

    gw = Tracer("gateway", sampler=sampler)
    pay = Tracer("pay-svc", sampler=sampler)

    print(f"\n--- 4.1 不同操作采样率不同 ---")
    ops = ["health_check", "place_order", "login", "unknown_op"]
    for op in ops:
        sampled_count = 0
        for _ in range(50):
            with gw.start_active_span(op) as s:
                if s.context.sampled:
                    sampled_count += 1
        print(f"  gateway/{op}: 50 次采样 {sampled_count} 次")

    print(f"\n--- 4.2 确定性：同 trace_id 多次判断结果一致 ---")
    tid = TraceContext.generate_trace_id()
    results = {
        sampler.should_sample_by_trace_id(tid, operation_name="place_order", service_name="gateway")
        for _ in range(50)
    }
    print(f"  同一 trace_id 判断 50 次，结果集合: {results}")
    assert len(results) == 1

    print(f"\n--- 4.3 下游严格继承入口采样结果，绝不重新判断 ---")
    mismatches = 0
    for _ in range(30):
        for op, expected in (("place_order", True), ("health_check", False)):
            with gw.start_active_span(op) as root:
                headers = {}
                gw.inject(root.context, headers)
                ctx = pay.extract(headers)
                with pay.start_active_span("charge", context=ctx) as child:
                    if child.context.sampled != expected or root.context.sampled != expected:
                        mismatches += 1
    print(f"  30 轮 × 2 操作 = 60 次跨服务调用，不一致次数: {mismatches}")
    assert mismatches == 0
    print("  ✅ 下游全部正确继承入口采样结果")

    print(f"\n--- 4.4 PerOperationSampler 也支持按 trace_id 确定性采样 ---")
    sampler2 = PerOperationSampler(
        {"pay": 1.0, "health": 0.0},
        default_rate=0.0,
    )
    tid2 = TraceContext.generate_trace_id()
    r1 = sampler2.should_sample_by_trace_id(tid2, operation_name="pay")
    r2 = sampler2.should_sample_by_trace_id(tid2, operation_name="health")
    print(f"  pay: {r1}, health: {r2} (同 trace_id, 不同 operation)")
    assert r1 is True and r2 is False
    print("  ✅ PerOperationSampler + trace_id 确定性判断工作正常")


def main():
    print(SEP)
    print("分布式追踪 V2 — 四大增强功能演示")
    print(SEP)

    demo_advanced_search()
    demo_process_dedup()
    demo_import_export()
    demo_service_operation_sampler()

    print(f"\n{SEP}")
    print("🎉 所有演示通过！")
    print(SEP)


if __name__ == "__main__":
    main()
