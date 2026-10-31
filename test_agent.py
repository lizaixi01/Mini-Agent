import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import agent


def tool_call(arguments='{"command":"example"}', name="run_command"):
    return {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": arguments}}]}


class AgentLoopTests(unittest.TestCase):
    def test_memory_proposal_requires_confirmation_and_updates_next_request(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(agent, "MEMORY_FILE", Path(directory) / "memory.json"):
            request = {"role": "assistant", "tool_calls": [{"id": "m1", "type": "function", "function": {"name": "propose_memory", "arguments": '{"text":"喜欢简单解释"}'}}]}
            for approval, expected in (("n", []), ("y", ["喜欢简单解释"])):
                with self.subTest(approval=approval):
                    model = Mock()
                    received = []
                    replies = iter([request, {"role": "assistant", "content": "完成"}])
                    model.reply.side_effect = lambda messages: (received.append(list(messages)) or next(replies))
                    memory = []
                    result = agent.run_agent("记住偏好", model, confirm=lambda _: approval, memory=memory)
                    self.assertEqual(agent.load_memory(), expected)
                    self.assertEqual(json.loads(result[3]["content"])["saved"], approval == "y")
                    self.assertEqual(memory, expected)
                    self.assertEqual(any(m.get("content", "").startswith(agent.MEMORY_PREFIX) for m in received[-1]), approval == "y")

    def test_memory_proposal_invalid_or_full_does_not_prompt_or_write(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(agent, "MEMORY_FILE", Path(directory) / "memory.json"):
            for text, initial in (("x" * 501, []), ("新内容", [str(i) for i in range(20)])):
                with self.subTest(text=text[:5]):
                    agent.save_memory(initial)
                    model = Mock()
                    model.reply.side_effect = [{"role": "assistant", "tool_calls": [{"id": "m1", "type": "function", "function": {"name": "propose_memory", "arguments": json.dumps({"text": text})}}]}, {"role": "assistant", "content": "结束"}]
                    confirm = Mock()
                    agent.run_agent("任务", model, confirm=confirm)
                    confirm.assert_not_called()
                    self.assertEqual(agent.load_memory(), initial)

    def test_demo_memory_proposal_saves_and_is_visible_in_new_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = agent.save_session
            with patch.object(agent, "MEMORY_FILE", root / "memory.json"), \
                 patch.object(agent, "save_session", side_effect=lambda session: save(session, root / "sessions")), \
                 patch("sys.stdout", new_callable=io.StringIO) as output, \
                 patch.object(sys, "argv", ["agent.py", "--demo"]):
                with patch("builtins.input", side_effect=["记住 我喜欢小例子", "y", "/exit"]):
                    agent.main()
                with patch("builtins.input", side_effect=["/memory", "/exit"]):
                    agent.main()
            self.assertIn("建议保存的项目记忆：我喜欢小例子", output.getvalue())
            self.assertIn("1. 我喜欢小例子", output.getvalue())

    def test_model_can_discover_and_read_skill_without_running_shell(self):
        captured = []
        model = Mock()
        request = {"role": "assistant", "tool_calls": [{"id": "skill-1", "type": "function", "function": {"name": "read_skill", "arguments": '{"name":"explain-code"}'}}]}
        responses = iter([request, {"role": "assistant", "content": "解释完成"}])
        model.reply.side_effect = lambda received: (captured.append(list(received)) or next(responses))
        execute, confirm = Mock(), Mock()
        messages = agent.run_agent("解释代码", model, execute=execute, confirm=confirm)
        self.assertIn("explain-code", captured[0][0]["content"])
        self.assertNotIn("# 解释代码", captured[0][0]["content"])
        self.assertIn("# 解释代码", json.loads(messages[3]["content"])["instructions"])
        self.assertEqual(messages[3]["tool_call_id"], "skill-1")
        execute.assert_not_called()
        confirm.assert_not_called()

    def test_skill_tool_rejects_unknown_or_bad_arguments(self):
        for arguments in ('{"name":"../agent.py"}', '{"name":"explain-code", "extra":1}', '[]'):
            with self.subTest(arguments=arguments):
                model = Mock()
                model.reply.side_effect = [{"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_skill", "arguments": arguments}}]}, {"role": "assistant", "content": "结束"}]
                messages = agent.run_agent("task", model)
                self.assertIn("error", json.loads(messages[3]["content"]))

    def test_skill_discovery_and_request_injection(self):
        skills = agent.list_skills()
        self.assertTrue(any(skill["name"] == "explain-code" for skill in skills))
        instructions = agent.load_skill("explain-code")
        with self.assertRaises(ValueError):
            agent.load_skill("../agent.py")
        captured = []
        model = Mock()
        model.reply.side_effect = lambda received: (captured.append(list(received)) or {"role": "assistant", "content": "完成"})
        messages = [{"role": "system", "content": agent.SYSTEM_MESSAGE}]
        agent.run_agent("解释代码", model, messages=messages, skill=instructions)
        self.assertIn("给初学者解释代码", captured[0][0]["content"])
        self.assertFalse(any(m.get("content", "").startswith(agent.SKILL_PREFIX) for m in messages))

    def test_memory_is_saved_and_sent_without_becoming_session_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            agent.save_memory(["请用简单的话解释"], path)
            memory = agent.load_memory(path)
            messages = [{"role": "system", "content": agent.SYSTEM_MESSAGE}]
            captured = []
            model = Mock()
            model.reply.side_effect = lambda received: (captured.append(list(received)) or {"role": "assistant", "content": "好的"})
            agent.run_agent("你好", model, messages=messages, memory=memory)
            self.assertIn("请用简单的话解释", captured[0][0]["content"])
            self.assertFalse(any(m.get("content", "").startswith(agent.MEMORY_PREFIX) for m in messages))
            agent.save_memory([], path)
            agent.run_agent("再见", model, messages=messages, memory=agent.load_memory(path))
            self.assertFalse(any(m.get("content", "").startswith(agent.MEMORY_PREFIX) for m in captured[-1]))

    def test_non_ascii_key_rejected_without_disclosing_it(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "secret-中文"}):
            with self.assertRaises(ValueError) as error:
                agent.DeepSeekModel()
        self.assertNotIn("secret", str(error.exception))

    def test_approved_result_is_returned_with_matching_call_id(self):
        model = Mock()
        model.reply.side_effect = [tool_call(), {"role": "assistant", "content": "完成"}]
        execute = Mock(return_value={"stdout": "result", "stderr": "", "returncode": 0})
        messages = agent.run_agent("task", model, execute=execute, confirm=lambda _: "y")
        execute.assert_called_once_with("example")
        self.assertEqual(messages[3]["tool_call_id"], "call-1")
        self.assertEqual(json.loads(messages[3]["content"])["stdout"], "result")
        self.assertEqual(messages[-1]["content"], "完成")

    def test_answer_does_not_execute_a_command(self):
        model = Mock()
        model.reply.return_value = {"role": "assistant", "content": "普通回答"}
        execute = Mock()
        agent.run_agent("task", model, execute=execute)
        execute.assert_not_called()

    def test_rejected_command_is_not_executed(self):
        model = Mock()
        model.reply.return_value = tool_call()
        execute = Mock()
        agent.run_agent("task", model, execute=execute, confirm=lambda _: "")
        execute.assert_not_called()
        self.assertEqual(model.reply.call_count, 1)

    def test_invalid_calls_are_returned_as_errors(self):
        for call in (tool_call("not json"), tool_call('{"command":3}'), tool_call(name="other")):
            with self.subTest(call=call):
                model = Mock()
                model.reply.side_effect = [call, {"role": "assistant", "content": "结束"}]
                execute = Mock()
                messages = agent.run_agent("task", model, execute=execute)
                execute.assert_not_called()
                self.assertIn("error", json.loads(messages[3]["content"]))

    def test_step_limit_stops_loop(self):
        model = Mock()
        model.reply.return_value = tool_call()
        execute = Mock(return_value={"stdout": "", "stderr": "", "returncode": 0})
        with patch.object(agent, "MAX_STEPS", 2):
            agent.run_agent("task", model, execute=execute, confirm=lambda _: "y")
        self.assertEqual(execute.call_count, 2)

    def test_saved_session_resumes_with_prior_context(self):
        with tempfile.TemporaryDirectory() as directory:
            session = {"id": "a" * 32, "mode": "deepseek", "messages": [{"role": "system", "content": agent.SYSTEM_MESSAGE}]}
            model = Mock()
            model.reply.return_value = {"role": "assistant", "content": "你好"}
            agent.run_agent("记住我的名字是小明", model, messages=session["messages"], save=lambda: agent.save_session(session, Path(directory)))
            restored = agent.load_session(session["id"], Path(directory))
            agent.run_agent("我的名字是什么？", model, messages=restored["messages"])
            self.assertEqual(restored["messages"][1]["content"], "记住我的名字是小明")
            self.assertEqual(restored["messages"][-2]["content"], "我的名字是什么？")

    def test_interrupted_tool_call_is_not_reexecuted_on_recovery(self):
        messages = [{"role": "user", "content": "task"}, tool_call()]
        self.assertTrue(agent.repair_pending_calls(messages))
        self.assertEqual(messages[-1]["tool_call_id"], "call-1")
        self.assertIn("执行状态未知", json.loads(messages[-1]["content"])["error"])
        self.assertFalse(agent.repair_pending_calls(messages))

    def test_rejected_call_is_recorded_for_continuation(self):
        model = Mock()
        model.reply.return_value = tool_call()
        messages = agent.run_agent("task", model, confirm=lambda _: "n")
        self.assertIn("用户拒绝", json.loads(messages[-1]["content"])["error"])
        self.assertFalse(agent.repair_pending_calls(messages))

    def test_deny_mode_blocks_without_asking_or_executing(self):
        model = Mock()
        model.reply.return_value = tool_call()
        execute, confirm = Mock(), Mock()
        messages = agent.run_agent("task", model, execute=execute, confirm=confirm, permission="deny")
        execute.assert_not_called()
        confirm.assert_not_called()
        self.assertIn("权限模式拒绝", json.loads(messages[-1]["content"])["error"])

    def test_unknown_permission_fails_closed(self):
        with self.assertRaises(ValueError):
            agent.run_agent("task", Mock(), permission="unknown")

    def test_turns_record_completion_and_denial(self):
        session = {"messages": [], "permission": "deny"}
        model = Mock()
        model.reply.side_effect = [{"role": "assistant", "content": "你好"}, tool_call()]
        agent.run_session_turn(session, "打招呼", model, save=lambda: None)
        agent.run_session_turn(session, "运行命令", model, save=lambda: None)
        self.assertEqual([turn["status"] for turn in session["turns"]], ["completed", "interrupted"])
        self.assertNotEqual(session["turns"][0]["id"], session["turns"][1]["id"])
        self.assertEqual(session["turns"][1]["events"], ["工作开始", "模型请求使用工具", "权限模式拒绝执行", "工作结束：interrupted"])

    def test_events_for_approved_tool_are_recorded_in_order(self):
        session = {"messages": [], "permission": "ask"}
        model = Mock()
        model.reply.side_effect = [tool_call(), {"role": "assistant", "content": "完成"}]
        original = agent.run_agent
        with patch.object(agent, "run_agent") as run:
            # 假执行器使测试不运行真实命令。
            run.side_effect = lambda *args, **kwargs: original(
                *args, execute=lambda _: {"stdout": "", "stderr": "", "returncode": 0}, **kwargs)
            agent.run_session_turn(session, "任务", model, confirm=lambda _: "y", save=lambda: None)
        self.assertEqual(session["turns"][0]["events"], [
            "工作开始", "模型请求使用工具", "用户同意执行", "命令执行结束，退出码：0", "工作结束：completed"])

    def test_failed_and_recovered_turns(self):
        session = {"messages": []}
        model = Mock()
        model.reply.side_effect = RuntimeError("connection failed")
        with self.assertRaises(RuntimeError):
            agent.run_session_turn(session, "任务", model, save=lambda: None)
        self.assertEqual(session["turns"][0]["status"], "failed")
        session["turns"].append({"status": "running"})
        self.assertTrue(agent.recover_turns(session))
        self.assertEqual(session["turns"][-1]["status"], "interrupted")

    def test_old_session_keeps_messages_without_inventing_turns(self):
        session = {"messages": [{"role": "user", "content": "旧消息"}]}
        self.assertFalse(agent.recover_turns(session))
        self.assertEqual(session["turns"], [])
        self.assertEqual(session["messages"][0]["content"], "旧消息")

    def test_cli_demo_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save, load = agent.save_session, agent.load_session
            with patch.object(agent, "SESSION_DIR", root), patch.object(agent, "MEMORY_FILE", root / "memory.json"), \
                 patch.object(agent, "save_session", side_effect=lambda session: save(session, root)), \
                 patch.object(agent, "load_session", side_effect=lambda sid: load(sid, root)), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                with patch.object(sys, "argv", ["agent.py", "--demo"]), patch("builtins.input", side_effect=["查看文件", "n", "/permissions deny", "/exit"]):
                    agent.main()
                session_id = next(root.glob("*.json")).stem
                with patch.object(sys, "argv", ["agent.py", "--resume", session_id]), patch("builtins.input", side_effect=["/history", "/turns", "/events 1", "/permissions", "/exit"]):
                    agent.main()
                self.assertIn("user: 查看文件", output.getvalue())
                self.assertIn("[interrupted] 查看文件", output.getvalue())
                self.assertIn("模型请求使用工具", output.getvalue())
                self.assertIn("用户拒绝执行", output.getvalue())
                self.assertIn("当前权限：deny", output.getvalue())

    def test_cli_memory_commands_survive_new_session_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = agent.save_session
            with patch.object(agent, "MEMORY_FILE", root / "memory.json"), \
                 patch.object(agent, "save_session", side_effect=lambda session: save(session, root / "sessions")), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                with patch.object(sys, "argv", ["agent.py", "--demo"]), patch("builtins.input", side_effect=["/remember 简单解释", "/exit"]):
                    agent.main()
                with patch.object(sys, "argv", ["agent.py", "--demo"]), patch("builtins.input", side_effect=["/memory", "任务", "n", "/forget 1", "/memory", "/exit"]):
                    agent.main()
                self.assertIn("1. 简单解释", output.getvalue())
                self.assertIn("本次请求带上了项目记忆", output.getvalue())
                self.assertIn("目前没有项目记忆", output.getvalue())

    def test_cli_skill_applies_only_to_next_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = agent.save_session
            with patch.object(agent, "MEMORY_FILE", root / "memory.json"), \
                 patch.object(agent, "save_session", side_effect=lambda session: save(session, root / "sessions")), \
                 patch("sys.stdout", new_callable=io.StringIO) as output, \
                 patch.object(sys, "argv", ["agent.py", "--demo"]), \
                 patch("builtins.input", side_effect=["/skills", "/skill explain-code", "任务一", "n", "任务二", "n", "/exit"]):
                agent.main()
            self.assertIn("explain-code：给初学者解释代码", output.getvalue())
            self.assertEqual(output.getvalue().count("本次请求带上了技能说明"), 1)

    def test_cli_skill_typo_gives_help_instead_of_calling_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save = agent.save_session
            with patch.object(agent, "MEMORY_FILE", root / "memory.json"), \
                 patch.object(agent, "save_session", side_effect=lambda session: save(session, root / "sessions")), \
                 patch("sys.stdout", new_callable=io.StringIO) as output, \
                 patch.object(sys, "argv", ["agent.py", "--demo"]), \
                 patch("builtins.input", side_effect=["/skills explain-code", "/exit"]):
                agent.main()
            self.assertIn("选择技能请输入 /skill explain-code", output.getvalue())
            self.assertNotIn("开始工作", output.getvalue())

    def test_demo_automatically_requests_skill(self):
        model = agent.DemoModel()
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            messages = agent.run_agent("解释 print 是什么", model)
        self.assertIn("已读取技能：explain-code", output.getvalue())
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read_skill")
        self.assertIn("真实模型会结合说明", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
