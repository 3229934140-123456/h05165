import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple


class Sampler(ABC):
    """采样器抽象基类。"""

    @abstractmethod
    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        """
        判断是否应该采样。
        只在入口服务（创建根 span 时）调用此方法。
        下游服务从上下文继承采样决策，不重新采样。
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """返回采样器的描述信息。"""
        pass


class AlwaysOnSampler(Sampler):
    """全采样器，所有请求都被采样。"""

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        return True

    def get_description(self) -> str:
        return "AlwaysOnSampler(sample_rate=1.0)"


class AlwaysOffSampler(Sampler):
    """全不采样器，所有请求都不被采样。"""

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        return False

    def get_description(self) -> str:
        return "AlwaysOffSampler(sample_rate=0.0)"


class ProbabilisticSampler(Sampler):
    """
    概率采样器，按固定概率采样。

    采样决策在入口服务做出，采样结果通过 x-sampled 头部传播。
    下游服务读取该头部，不再重新采样，保证整条链路一致。
    """

    def __init__(self, rate: float):
        """
        :param rate: 采样率，0.0 ~ 1.0 之间
        """
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"采样率必须在 0.0 ~ 1.0 之间，当前值: {rate}")
        self.rate = rate
        self._boundary = rate * (2**64 - 1)

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        """
        使用确定性采样：基于 trace_id 的哈希值决定是否采样。
        这样即使在分布式环境下，只要使用相同的采样率，
        同一个 trace_id 在任何节点都会得到相同的采样结果。
        """
        if self.rate >= 1.0:
            return True
        if self.rate <= 0.0:
            return False
        return random.random() < self.rate

    def should_sample_by_trace_id(self, trace_id: str) -> bool:
        """
        基于 trace_id 的确定性采样决策。
        使用 trace_id 的哈希值，保证同一个 trace_id 始终得到相同结果。
        这是全链路一致采样的关键。
        """
        if self.rate >= 1.0:
            return True
        if self.rate <= 0.0:
            return False

        trace_id_int = int(trace_id[:16], 16)
        return (trace_id_int % 10000) < (self.rate * 10000)

    def get_description(self) -> str:
        return f"ProbabilisticSampler(sample_rate={self.rate})"


class RateLimitingSampler(Sampler):
    """
    限速采样器，限制每秒最多采样的请求数。

    用于流量较大的服务，避免采样数据过多导致存储和处理压力过大。
    """

    def __init__(self, max_traces_per_second: float):
        """
        :param max_traces_per_second: 每秒最多采样的 trace 数量
        """
        if max_traces_per_second <= 0:
            raise ValueError("采样速率必须大于 0")
        self.max_traces_per_second = max_traces_per_second
        self._max_balance = max_traces_per_second
        self._balance = 0.0
        self._last_tick = time.time()
        self._cost_per_second = 1.0

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        """
        基于令牌桶算法的限速采样。
        每秒补充 max_traces_per_second 个令牌，每次采样消耗 1 个令牌。
        """
        now = time.time()
        elapsed = now - self._last_tick
        self._last_tick = now

        self._balance = min(
            self._max_balance,
            self._balance + elapsed * self.max_traces_per_second,
        )

        if self._balance >= self._cost_per_second:
            self._balance -= self._cost_per_second
            return True
        return False

    def get_description(self) -> str:
        return f"RateLimitingSampler(max_traces_per_second={self.max_traces_per_second})"


class PerOperationSampler(Sampler):
    """
    按操作名称的采样器，为不同的操作设置不同的采样率。

    某些关键操作（如支付、下单）可能需要 100% 采样，
    而其他操作（如健康检查、静态资源访问）可能需要较低的采样率。
    """

    def __init__(
        self,
        operation_sample_rates: Dict[str, float],
        default_rate: float = 0.001,
    ):
        self.operation_sample_rates = operation_sample_rates
        self.default_rate = default_rate
        self._samplers: Dict[str, ProbabilisticSampler] = {}
        self._default_sampler = ProbabilisticSampler(default_rate)

        for op, rate in operation_sample_rates.items():
            self._samplers[op] = ProbabilisticSampler(rate)

    def _pick_sampler(self, operation_name: str = "") -> ProbabilisticSampler:
        return self._samplers.get(operation_name, self._default_sampler)

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        return self._pick_sampler(operation_name).should_sample(operation_name, tags)

    def should_sample_by_trace_id(
        self,
        trace_id: str,
        operation_name: str = "",
        service_name: str = "",
    ) -> bool:
        return self._pick_sampler(operation_name).should_sample_by_trace_id(trace_id)

    def get_description(self) -> str:
        rates = ", ".join(f"{k}={v}" for k, v in self.operation_sample_rates.items())
        return (
            f"PerOperationSampler(operation_rates={{{rates}}}, "
            f"default_rate={self.default_rate})"
        )


class ServiceOperationSampler(Sampler):
    """
    按 (服务名, 操作名) 粒度配置不同采样率的确定性采样器。

    - 入口服务使用：先根据 service_name + operation_name 找到对应采样率，
      再基于 trace_id 哈希做确定性判断（同 trace_id 永远一致）。
    - 下游服务不使用本类，直接继承上下文的 sampled 标志。

    配置示例：
        {
            ("gateway", "/health"): 0.01,
            ("gateway", "/checkout"): 1.0,
            ("order-svc", "create_order"): 1.0,
        }
        default_rate=0.1  # 其他走默认
    """

    def __init__(
        self,
        service_operation_rates: Optional[Dict[Tuple[str, str], float]] = None,
        service_default_rates: Optional[Dict[str, float]] = None,
        default_rate: float = 0.001,
    ):
        self.service_operation_rates = dict(service_operation_rates or {})
        self.service_default_rates = dict(service_default_rates or {})
        self.default_rate = default_rate
        self._samplers: Dict[Any, ProbabilisticSampler] = {}
        self._default_sampler = ProbabilisticSampler(default_rate)

        for (svc, op), rate in self.service_operation_rates.items():
            self._samplers[(svc, op)] = ProbabilisticSampler(rate)
        for svc, rate in self.service_default_rates.items():
            self._samplers[(svc, None)] = ProbabilisticSampler(rate)

    def _pick_sampler(
        self, service_name: str = "", operation_name: str = ""
    ) -> ProbabilisticSampler:
        key = (service_name or "", operation_name or "")
        if key in self._samplers:
            return self._samplers[key]
        svc_key = (service_name or "", None)
        if svc_key in self._samplers:
            return self._samplers[svc_key]
        return self._default_sampler

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        service_name = ""
        if tags:
            service_name = str(tags.get("service_name", ""))
        return self._pick_sampler(service_name, operation_name).should_sample()

    def should_sample_by_trace_id(
        self,
        trace_id: str,
        operation_name: str = "",
        service_name: str = "",
    ) -> bool:
        """
        基于 (service, operation) 对应采样率 + trace_id 哈希做确定性判断。
        同一个 trace_id + 同配置下结果永远一致。
        """
        return self._pick_sampler(service_name, operation_name).should_sample_by_trace_id(trace_id)

    def get_description(self) -> str:
        parts = []
        if self.service_operation_rates:
            rates = ", ".join(
                f"({s},{o})={r}" for (s, o), r in self.service_operation_rates.items()
            )
            parts.append(f"service_op_rates={{{rates}}}")
        if self.service_default_rates:
            rates = ", ".join(f"{s}={r}" for s, r in self.service_default_rates.items())
            parts.append(f"service_defaults={{{rates}}}")
        parts.append(f"default_rate={self.default_rate}")
        return f"ServiceOperationSampler({', '.join(parts)})"


class AdaptiveSampler(Sampler):
    """
    自适应采样器，根据系统负载动态调整采样率。

    当系统负载较低时提高采样率，获得更完整的追踪数据；
    当系统负载较高时降低采样率，减少对系统的影响。
    """

    def __init__(
        self,
        min_rate: float = 0.01,
        max_rate: float = 1.0,
        target_spans_per_second: int = 100,
    ):
        """
        :param min_rate: 最小采样率
        :param max_rate: 最大采样率
        :param target_spans_per_second: 目标每秒 span 数量
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_spans_per_second = target_spans_per_second
        self.current_rate = (min_rate + max_rate) / 2
        self._sampler = ProbabilisticSampler(self.current_rate)
        self._spans_count = 0
        self._last_adjust = time.time()
        self._adjust_interval = 10.0  # 每 10 秒调整一次

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        self._spans_count += 1
        self._maybe_adjust_rate()
        return self._sampler.should_sample(operation_name, tags)

    def _maybe_adjust_rate(self) -> None:
        now = time.time()
        elapsed = now - self._last_adjust
        if elapsed < self._adjust_interval:
            return

        spans_per_second = self._spans_count / elapsed
        ratio = self.target_spans_per_second / max(spans_per_second, 1)

        new_rate = self.current_rate * ratio
        new_rate = max(self.min_rate, min(self.max_rate, new_rate))

        if abs(new_rate - self.current_rate) > 0.01:
            self.current_rate = new_rate
            self._sampler = ProbabilisticSampler(new_rate)

        self._spans_count = 0
        self._last_adjust = now

    def get_description(self) -> str:
        return (
            f"AdaptiveSampler(current_rate={self.current_rate:.4f}, "
            f"min_rate={self.min_rate}, max_rate={self.max_rate}, "
            f"target_spans_per_second={self.target_spans_per_second})"
        )


class CompositeSampler(Sampler):
    """
    组合采样器，组合多个采样策略。

    只要有一个采样器决定采样，就会被采样（OR 逻辑）。
    常用于：概率采样 + 错误采样（错误请求 100% 采样）。
    """

    def __init__(self, samplers: list):
        self.samplers = samplers

    def should_sample(self, operation_name: str = "", tags: Optional[Dict[str, Any]] = None) -> bool:
        for sampler in self.samplers:
            if sampler.should_sample(operation_name, tags):
                return True
        return False

    def get_description(self) -> str:
        descriptions = [s.get_description() for s in self.samplers]
        return f"CompositeSampler(samplers=[{', '.join(descriptions)}])"
