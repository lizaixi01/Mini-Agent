# 第 3 课：会话、工作与事件

会话是一段可以继续的对话；一次工作（turn）是其中一轮任务；事件记录这一轮发生了什么。项目使用本地 JSON 文件保存这些信息，这是教学版，并非 Codex 的内部实现。

体验：运行演示并完成一个任务，输入 `/turns` 查看工作，再输入 `/events 1` 查看第一项工作的事件。输入 `/exit` 后，运行 `python agent.py --list` 查看会话，使用 `python agent.py --resume 会话ID` 恢复。

`completed` 表示收到最终回答，并不保证结果正确；`interrupted` 表示被拒绝、中断或达到步数上限；`failed` 表示运行出错。旧版工作没有记录事件时，程序不会猜测。

代码：搜索 [agent.py](../agent.py) 中的 `save_session`、`load_session`、`run_session_turn`。
