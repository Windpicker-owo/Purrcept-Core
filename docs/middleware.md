# Middleware

`MiddlewareExecutor(base, [A, B, C])` 的调用顺序是 `A(B(C(base)))`。第一项最外层，
返回值和异常按相反方向穿过链。

内置中间件：

- `TimeoutMiddleware`：用 `asyncio.timeout()` 限制一次 Effect 调用。
- `RetryMiddleware`：仅重试显式异常类型或 predicate，支持确定性的 delay 策略。
- `FunctionMiddleware`：用普通 async 函数快速包装 `call_next`。

Core 默认不会重试。一次内部重试不会创建新 Effect step；如果需要记录尝试次数，自定义
中间件可通过宿主观察机制或自定义事件完成。

`EventDispatchError` 是观察基础设施错误，即使 `retry_on=Exception` 或 predicate
返回 true 也绝不重试，以免在事件失败后重复已经发生的外部副作用。

中间件不得吞掉取消。审批、限流、缓存、权限和 Provider fallback 属于 Runtime 或扩展包。
