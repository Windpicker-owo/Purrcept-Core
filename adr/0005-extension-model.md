# ADR 0005：扩展模型

- 状态：Accepted
- 日期：2026-07-26

## 决策

扩展是普通 Python Package，通过 import、构造与显式组合启用。Core 不要求 Plugin
基类、entry point、自动扫描、注册装饰器或全局 PluginManager。

## 结果

安装与启用解耦，多个 Executor 实例保持隔离，扩展的 import 不产生运行时行为。可选发现
机制若未来出现，也必须位于 Runtime 层。

