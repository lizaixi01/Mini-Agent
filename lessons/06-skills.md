# 第 6 课：Skills

Skill 是一份按任务使用的说明书。项目示例在 [explain-code/SKILL.md](../skills/explain-code/SKILL.md)：它告诉模型如何给初学者解释代码。

体验：运行演示，输入 `/skills` 查看技能；输入 `/skill explain-code` 可以为下一次任务手动选择。也可以直接输入“解释 print 是什么”，观察演示模型请求读取技能。

搜索 [agent.py](../agent.py) 中的 `list_skills`、`load_skill` 和 `read_skill`。程序把技能名称和简介提供给模型；模型可以请求完整说明。演示是固定剧本，真实模型不保证一定选择或遵守技能。

当前实现只识别教学示例的简单格式，不是 Codex 或 Claude Code 的完整技能系统。
