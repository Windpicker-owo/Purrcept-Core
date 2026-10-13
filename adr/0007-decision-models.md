# ADR 0007：独立决策模型能力

- 状态：Accepted
- 日期：2026-10-11

## 背景

OpenAI Decisions 与 TypeSafe Jev 对共享证据返回命题概率、分类或等级评分。
现有 ModelRequest、Generate 与 ModelResponse 表达聊天生成、流式消息和工具循环，
不能准确表达决策结果。将概率包装成 assistant 文本会丢失类型和契约。

## 决策

在 `purrcept_core.models.decisions` 提供独立的 DecisionModel、DecisionBackend、
DecisionRequest、DecisionResponse 和 Decide，并从 models 导出主要类型。
Decide 是普通 Effect，由现有 InlineExecutor 执行，不修改通用 Effect 内核。

请求包含文本或有序图文块，以及独立且名称唯一的问题。每个问题仅评估同一份输入，
不依赖同批其他答案；有关联的问题由应用组成多个 Effect。问题指令与描述使用字符串，
结构化业务数据由应用明确序列化为 JSON 文本。

响应按问题顺序返回。Predicate 保留概率；Choice 保留字符串/布尔选项、完整概率分布
和供应商 confidence；Score 保留从零起算的等级索引及加权评分；Refusal 是单题结果。
Core 验证值域和请求对应关系，不阈值化、不重新校准，也不把缺失结果猜测为否或零分。

Backend 的凭据、连接、并发与生命周期归 Runtime 或应用；Core 不关闭借用的资源。
错误沿用 ModelError 分类，非法决策响应使用 DecisionResponseError，取消原样传播。
业务路由、低置信回退、重试和工具执行仍由应用与 Runtime 负责。

## LiteLLM 边界

适配器通过公开的 `litellm.adecisions(input=..., questions=...)` 调用所有供应商，
不重复实现 System One 转换。首个验证版本为 1.104.2，无流式接口。图像不受支持时
明确失败，不丢弃图像或改用描述文本。

LiteLLM 1.104.2 会把部分缺失或类型不匹配的上游答案转换为 refusal，并在部分路径
补齐概率条目。适配器只能验证 SDK 返回的标准化结果，无法保证原始响应无损。
离线真实 SDK 测试记录这些行为；生产代码不通过私有接口绕过转换。

## 后果

聊天 Backend 无需增加 decide 方法；单纯的 DecisionBackend 也无需实现 generate。
新增能力以 Core 0.6.0 为版本边界，适配器显式声明对应依赖。首期不增加会话状态、
决策流事件、自动路由策略或 Jev 的 JSON instructions/criteria 原生扩展。

## 依据

- [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions)
- [TypeSafe primitives](https://docs.typesafe.ai/primitives)
- [TypeSafe confidence](https://docs.typesafe.ai/confidence)
- [LiteLLM Decisions](https://docs.litellm.ai/docs/decisions)
- [LiteLLM 1.104.2](https://github.com/BerriAI/litellm/releases/tag/v1.104.2)
