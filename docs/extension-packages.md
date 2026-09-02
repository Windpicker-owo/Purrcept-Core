# 扩展包

Purrcept 扩展是普通 Python Package。主路径是显式导入、构造和组合：

```python
from purrcept_core import DispatchExecutor, InlineExecutor
from purrcept_weather import ReadWeather, read_weather

executor = DispatchExecutor(
    {ReadWeather: read_weather},
    fallback=InlineExecutor(),
)
```

扩展包可以：

- 定义自执行 Effect；
- 定义声明式 Effect 和 handler；
- 提供 Executor 或 Middleware；
- 用 Protocol 声明需要的 host 能力；
- 实现标准 `ModelBackend`；
- 提供适配具体平台资源的 Python 函数工具。

扩展包不应在 import 时注册 handler/工具、读取配置、创建客户端或修改全局状态。安装不
等于启用，导入不等于注册；构造对象、组合 Executor 或把工具传给 `Conversation` 才表示
实际使用。

## Provider 扩展

Provider 包实现官方协议，持有自己的 SDK client，并由应用绑定具体模型：

```python
from purrcept_core import AgentFlow, perform
from purrcept_core.models import (
    Message,
    Model,
    ModelEventSink,
    ModelRequest,
    ModelResponse,
)


class ProviderBackend:
    def __init__(self, client: ProviderClient) -> None:
        self._client = client

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        provider_response = await self._client.generate(
            model=model,
            payload=compile_request(request),
        )
        return convert_response(provider_response, emit=emit)


backend = ProviderBackend(client=provider_client)
model = Model(backend=backend, name="provider/model-name")


def one_shot() -> AgentFlow[str]:
    request = ModelRequest((Message.user("Hello"),))
    response = yield from perform(model.generate(request))
    return response.text
```

Backend 由 `Model` 显式携带，不从 Runtime host 查找。Runtime 仍负责创建、复用和关闭
client/backend，并可以把其他应用资源放在 host 中供普通 Effect 或 `ToolContext` 使用。

Provider Adapter 负责：

- 把 Message / ContentBlock 转换为供应商 wire format；
- 映射 `SystemInstruction` 与已由 Core 排序的 `SystemReminder`，保持其顺序并尊重
  placement 提示。`SystemInstruction` 才是 system 角色；Chat Completions 适配器必须
  把 reminder 发成 `user` 消息，不得并入 system 前缀；
- 映射 `ToolSpec`、`ModelSettings` 和 tool-call/result 内容；
- 解释 `PromptCachePolicy` 与 `ModelContinuation`；
- 报告标准 `TokenUsage`，保留 `provider_metadata`；
- 把增量输出转换成有序 `ModelStreamEvent`；
- 把供应商错误映射为最具体的 `ModelError`，并原样传播取消。

Provider 不负责 reminder scope、消费或持久状态。当后一份 `ModelRequest` 不再包含旧
reminder，或同 key 内容已替换时，Adapter 必须确保 Provider-managed continuation 不会
继续施加旧值。可采用请求级控制项、显式替换、回退 client-managed history 或明确的
Unsupported Feature 错误；不得静默保留、复制进标准历史或自行重排。

Provider 特有且可 JSON 表达的请求参数应放入命名空间化的 `provider_options`。不要把 SDK
client、认证对象、凭据、callback 或连接放进 `ModelRequest`。Core 与扩展都不应把隐藏
重试伪装成标准协议语义。函数工具虽然内部使用 Pydantic v2，Provider Adapter 只接收
标准 `ToolSpec` 和 JSON 值，不依赖任何 Pydantic 公共类型。

## 高层 Agent 集成

Provider 包只需要交付 Backend；模型会话、函数工具和自动循环由 Core 统一提供：

```python
from purrcept_core.models import Model, tool


@tool
async def lookup(query: str) -> str:
    """查询应用自己的数据源。"""

    return await application_index.lookup(query)


model = Model(ProviderBackend(provider_client), "provider/model-name")
conversation = model.conversation(
    instructions="仅依据可验证的数据作答。",
    tools=[lookup],
)
```

Provider 包不需要复制 Conversation、工具映射或 tool-call loop。平台工具若需要业务权限，
应由扩展包通过 `ToolContext.host`、专用 Effect、Middleware 或 Runtime 权限服务强制
检查；Reminder 不能代替权限系统。

## Conformance Kit

Provider 包应在自己的测试套件中运行统一的确定性场景：

```python
from purrcept_core.models import Model, run_backend_conformance


def factory(scenario):
    fake_transport = FakeTransport.for_scenario(scenario)
    backend = ProviderBackend(client=fake_transport)
    return Model(backend=backend, name=scenario.model_name)


async def test_backend_conformance():
    report = await run_backend_conformance(factory)
    report.raise_for_failures()
```

factory 每次接收一个包含精确 request、预期 response/error、模型名和 streaming 意图的
`BackendConformanceScenario`。当前标准场景覆盖：

- 基础响应与 assistant role；
- instructions、reminders、tools、settings、cache、continuation 和 usage；
- 多 priority reminder 的稳定顺序以及 `auto` / `instructions` / `tail` placement；
- user → assistant tool call（含结构化参数）→ tool result 消息事务；
- 同一 Backend 上的 reminder removal 与同 key replacement，包含 continuation；
- Started → TextDelta → Completed 流式生命周期；
- 标准模型错误原样传播；
- `asyncio.CancelledError` 原样传播。

多请求场景位于 `scenario.follow_up_steps`，每个 `BackendConformanceStep` 给出下一份
request 与预期 response。Fake transport 应检查最终供应商 wire request，而不只是
Purrcept 输入对象。`include_streaming=False` 只适用于明确不支持可选流事件的 Backend。
Conformance Kit 不替代扩展包对认证、真实 SDK 版本和特殊能力的测试。

## 仓库内独立包验证

[`examples/plugin_package`](../examples/plugin_package/) 作为一个独立 distribution
验证公共边界：它只依赖 `purrcept_core` 的公开 API，不依赖 entry point、全局 registry
或 import-time 注册。扩展包测试应继续安装构建后的 wheel，避免不小心导入仓库私有
模块。
