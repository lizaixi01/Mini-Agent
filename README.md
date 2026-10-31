# Mini-Agent

从零动手搭建一个教学用 AI Agent，逐课学习会话、工具、权限、Memory 和 Skills。项目借鉴 Codex 与 Claude Code 的公开功能，代码是简化实现。

## 快速开始

需要 Python 3.10 或更高版本。无需密钥即可体验固定剧本：

```powershell
python agent.py --demo
```

使用真实 DeepSeek 模型：先安装 `python -m pip install openai`，设置环境变量 `DEEPSEEK_API_KEY`，再运行 `python agent.py`。实际调用需要有效的 API Key，可能产生费用。

```powershell
python -m unittest -q
```

## 课程

1. [Agent 循环](lessons/01-agent-loop.md)
2. [工具调用](lessons/02-tools.md)
3. [会话、工作与事件](lessons/03-sessions.md)
4. [权限管理](lessons/04-permissions.md)
5. [Memory：跨会话记忆](lessons/05-memory.md)
6. [Skills：按需加载说明](lessons/06-skills.md)

后续计划：上下文压缩、子 Agent、后台任务、Goal、定时任务、界面、diff 和旁支问题。

这是教学项目，不是沙箱：批准的终端命令拥有当前用户权限。密钥、会话和记忆不应提交到仓库。

## 许可证

[MIT](LICENSE)
