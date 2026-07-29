# ADR 0002：Effect 与 Flow 语义

- 状态：Accepted
- 日期：2026-07-26

## 决策

用同步 Python Generator 表达默认 Flow。`yield` 产生 Effect，`send` 返回成功值，
`throw` 返回普通执行异常，Generator 的 `return` 是 Run 最终值。

Effect 只表达一步意图，不能选择下一步或直接接管 Driver。即使自带执行代码，也必须经过
Executor。

## 结果

Python 的条件、循环、异常和 `yield from` 即编排语言，不引入 Graph DSL。取消和
`BaseException` 不作为普通 Effect 失败注入 Flow。

