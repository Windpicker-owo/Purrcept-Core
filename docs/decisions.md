# 决策模型

`purrcept_core.models.decisions` 提供独立于聊天生成的强类型决策能力。
所有主要类型也可以从 `purrcept_core.models` 导入。Core 不依赖供应商 SDK。

```python
from purrcept_core import AgentDriver, AgentFlow, InlineExecutor, perform
from purrcept_core.models import (
    DecisionBackend,
    DecisionModel,
    DecisionRequest,
    DecisionResponse,
    PredicateQuestion,
)


def check(model: DecisionModel, evidence: str) -> AgentFlow[DecisionResponse]:
    request = DecisionRequest(
        evidence,
        (PredicateQuestion("refund", "客户是否要求退款？"),),
    )
    return (yield from perform(model.decide(request)))


async def run(backend: DecisionBackend) -> DecisionResponse:
    model = DecisionModel(backend=backend, name="typesafe/jev-latest")
    return await AgentDriver(InlineExecutor()).run(check(model, "请退还重复扣款。"), host=None)
```

构造请求和 `model.decide()` 都不会开始 I/O。Decide 在执行时调用
`backend.decide(request, *, model)`，检查返回值及其与请求的对应关系。
可复用 Backend 由应用创建并负责关闭；不同 DecisionModel 可以共享同一 Backend。

## 输入与问题

- input 接受非空文本，或 TextBlock / ImageBlock 的有序序列；构造后冻结为快照。
- ImageBlock 复用 ImageUrl / ImageBytes，不引入会话角色、工具结果或音频。
- 每个问题必须有非空名称和 instructions，名称在请求内唯一。
- ChoiceQuestion 接受至少两个 ChoiceOption；value 是字符串或布尔值，description 可选。
- ScoreQuestion 接受至少两个 ScoreLevel；顺序定义从低到高的零基等级。
- metadata 用于运行观测，provider_options 是按适配器命名空间隔离的 JSON 参数。

同批问题独立读取共享输入。需要引用上一个答案时，应在下一次 Decide 中明确提供它。
结构化数据使用 `json.dumps(...)` 转为证据文本；Core 不猜测序列究竟是 JSON 还是图文块。

## 读取结果

`response.answers` 按请求顺序返回，`response.answer(name)` 可按名称查询，未知名称抛
KeyError。答案联合类型包括：

| 类型 | 含义 |
| --- | --- |
| PredicateAnswer | probability 是条件成立的概率，范围为 0 至 1 |
| ChoiceAnswer | choice 保持原始选项类型，probabilities 保留每项概率，confidence 原样返回 |
| ScoreAnswer | score 是等级索引的加权评分；probabilities 包含 index、label、probability |
| DecisionRefusal | 当前问题没有可用决策，其他问题仍可能成功 |

应用必须显式区分拒答，不能把它当作概率零。Confidence 的公式由供应商定义，不能直接
理解为跨供应商统一的正确率。Core 不重算 score、不补齐概率，也不执行阈值或回退策略。
后端可以报告 TokenUsage；缺失 usage 表示未知，不强制假设输出 token 为零。

构造阶段对非法类型或值抛 TypeError 或 ValueError。适配器及 Decide 对非法响应
使用 DecisionResponseError；认证、限流和连接故障沿用 ModelError 分类。取消不转换为
业务错误。Decide 不合成 token-stream 事件，观测使用标准 Effect 生命周期。

## 适配器作者

实现 runtime-checkable 的 DecisionBackend，并返回已验证的 DecisionResponse。
返回前调用 `response.validate_for(request)`，让直接调用 Backend 的用户也获得验证。
同一实例可能并发使用；每次调用的 payload、回调列表和临时状态必须隔离。
请在模块与协议文档中说明客户端所有权、供应商数量限制和任何有损 SDK 转换。

LiteLLM 实现与可运行示例位于相邻 `purrcept_litellm` 仓库。
