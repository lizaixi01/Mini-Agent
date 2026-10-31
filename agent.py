import argparse
import json
import os
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


MAX_STEPS = 10
SESSION_DIR = Path(__file__).resolve().parent / ".mini-agent" / "sessions"
MEMORY_FILE = Path(__file__).resolve().parent / ".mini-agent" / "memory.json"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"
SYSTEM_MESSAGE = "你是一个终端助手。需要操作时使用工具，完成后用自然语言回答用户。用户希望跨会话记住信息时，可用 propose_memory 提出记忆，由用户确认是否保存。"
MEMORY_PREFIX = "用户保存的项目记忆（可能过时）：\n"
SKILL_PREFIX = "本次任务使用的技能说明：\n"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "执行一条终端命令。执行前用户会看到命令并决定是否批准。",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "需要执行的命令"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": "读取可用技能的完整说明；仅在任务需要该技能时调用。",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "技能列表中的名称"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "propose_memory",
        "description": "提出一条跨会话的项目记忆，保存前必须得到用户确认。",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "建议保存的简短记忆，最多 500 字"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}]


class DeepSeekModel:
    def __init__(self):
        from openai import OpenAI

        key = os.environ["DEEPSEEK_API_KEY"]
        if not key.isascii() or any(c.isspace() for c in key):
            raise ValueError("DEEPSEEK_API_KEY 含非英文字符或空白字符，请重新输入实际的 API Key。")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    def reply(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=TOOLS,
            extra_body={"thinking": {"type": "disabled"}},
        )
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError(f"模型响应未正常完成：{choice.finish_reason}")
        return choice.message.model_dump(exclude_none=True)


class DemoModel:
    """固定剧本，让人不用密钥也能体验工具调用；它不是实际的模型。"""

    def reply(self, messages):
        memory = next((m["content"] for m in messages if m.get("role") == "system" and m.get("content", "").startswith(MEMORY_PREFIX)), None)
        if memory:
            print(f"\n演示：本次请求带上了项目记忆：{memory}")
        skill = next((m["content"] for m in messages if m.get("role") == "system" and m.get("content", "").startswith(SKILL_PREFIX)), None)
        if skill:
            print(f"\n演示：本次请求带上了技能说明：\n{skill}")
        if messages[-1]["role"] == "tool":
            result = messages[-1]["content"]
            if messages[-1]["tool_call_id"].startswith("demo-memory-"):
                saved = json.loads(result).get("saved", False)
                return {"role": "assistant", "content": "演示：记忆已保存。" if saved else "演示：这条记忆没有保存。"}
            if messages[-1]["tool_call_id"].startswith("demo-skill-"):
                return {"role": "assistant", "content": "演示：我已经读到技能说明。真实模型会结合说明回答你的问题。"}
            return {"role": "assistant", "content": f"演示结束。工具返回的结果是：\n{result}"}
        user_text = messages[-1].get("content", "")
        if "记住" in user_text:
            proposed = user_text.split("记住", 1)[1].strip(" ：:，,。 ")
            return {"role": "assistant", "content": "演示：我将建议保存这条记忆。", "tool_calls": [{
                "id": "demo-memory-1", "type": "function", "function": {"name": "propose_memory", "arguments": json.dumps({"text": proposed}, ensure_ascii=False)}
            }]}
        if not skill and "解释" in user_text:
            return {"role": "assistant", "content": "演示：这个任务适合解释代码，我将读取对应技能。", "tool_calls": [{
                "id": "demo-skill-1", "type": "function", "function": {"name": "read_skill", "arguments": json.dumps({"name": "explain-code"})}
            }]}
        return {
            "role": "assistant", "content": "我将查看当前目录。",
            "tool_calls": [{
                "id": "demo-call-1", "type": "function",
                "function": {"name": "run_command", "arguments": json.dumps({
                    "command": 'python -c "import os; print(*os.listdir(), sep=chr(10))"'
                })},
            }],
        }


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


def save_session(session, directory=SESSION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session['id']}.json"
    temporary = directory / f"{session['id']}.tmp"
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_session(session_id, directory=SESSION_DIR):
    # 只接受程序生成的 ID，避免把路径当作会话名。
    if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
        raise ValueError("无效的会话 ID")
    session = json.loads((directory / f"{session_id}.json").read_text(encoding="utf-8"))
    if session.get("id") != session_id or not isinstance(session.get("messages"), list):
        raise ValueError("会话文件格式无效")
    return session


def load_memory(path=None):
    path = MEMORY_FILE if path is None else path
    if not path.exists():
        return []
    memory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(memory, list) or not all(isinstance(item, str) for item in memory):
        raise ValueError("记忆文件格式无效")
    return memory


def save_memory(memory, path=None):
    path = MEMORY_FILE if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def list_skills(directory=None):
    directory = SKILLS_DIR if directory is None else directory
    if not directory.exists():
        return []
    skills = []
    for folder in sorted(directory.iterdir()):
        file = folder / "SKILL.md"
        if folder.is_symlink() or not folder.is_dir() or file.is_symlink() or not file.is_file():
            continue
        lines = file.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 4 and lines[0] == "---" and lines[1].startswith("name: ") and lines[2].startswith("description: ") and lines[3] == "---":
            name = lines[1][6:].strip()
            if name == folder.name:
                skills.append({"name": name, "description": lines[2][13:].strip(), "path": file})
    return skills


def load_skill(name, directory=None):
    for skill in list_skills(directory):
        if skill["name"] == name:
            return skill["path"].read_text(encoding="utf-8")
    raise ValueError("没有找到这个技能。")


def repair_pending_calls(messages, reason="上次运行中断，执行状态未知；未自动重试，请先检查实际状态。"):
    """中断后补齐未记录结果的调用，不自动重新执行命令。"""
    pending = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            pending = [call["id"] for call in message["tool_calls"]]
        elif message.get("role") == "tool" and message.get("tool_call_id") in pending:
            pending.remove(message["tool_call_id"])
    for call_id in pending:
        messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps({
            "error": reason
        }, ensure_ascii=False)})
    return bool(pending)


def run_agent(task, model, execute=run_command, confirm=input, messages=None, save=lambda: None, permission="ask", on_finish=lambda status: None, on_event=lambda event: None, memory=None, skill=None):
    if permission not in ("ask", "deny"):
        raise ValueError("未知的权限模式，拒绝执行。")
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages.append({"role": "user", "content": task})
    save()
    available_skills = list_skills()
    catalog = [{"name": entry["name"], "description": entry["description"]} for entry in available_skills]
    for step in range(1, MAX_STEPS + 1):
        request_messages = messages
        if catalog:
            request_messages = [{"role": "system", "content": "可用技能（需要时调用 read_skill 读取完整说明）：\n" + json.dumps(catalog, ensure_ascii=False)}] + request_messages
        if memory:
            # 记忆仅在请求时加入，不写入会话消息；删除后下一次请求不再加载。
            request_messages = [{"role": "system", "content": MEMORY_PREFIX + json.dumps(memory, ensure_ascii=False)}] + request_messages
        if skill:
            request_messages = [{"role": "system", "content": SKILL_PREFIX + skill}] + request_messages
        message = model.reply(request_messages)
        messages.append(message)
        save()
        if message.get("content"):
            print(f"\n助手：{message['content']}")
        calls = message.get("tool_calls") or []
        if not calls:
            if not message.get("content"):
                print("模型没有返回回答，已停止。")
                on_finish("failed")
            else:
                on_finish("completed")
            return messages

        for call in calls:
            function = call.get("function", {})
            on_event("模型请求使用工具")
            try:
                arguments = json.loads(function.get("arguments", ""))
                tool_name = function.get("name")
                if tool_name not in ("run_command", "read_skill", "propose_memory"):
                    raise ValueError("未知工具")
                field = {"run_command": "command", "read_skill": "name", "propose_memory": "text"}[tool_name]
                if (not isinstance(arguments, dict) or set(arguments) != {field}
                        or not isinstance(arguments[field], str) or not arguments[field].strip()):
                    raise ValueError(f"需要非空的 {field} 字符串，且不能包含其他参数")
                if tool_name == "read_skill" and arguments["name"] not in {entry["name"] for entry in available_skills}:
                    raise ValueError("技能不在可用列表中")
                if tool_name == "propose_memory" and len(arguments["text"].strip()) > 500:
                    raise ValueError("记忆最多 500 字")
            except (ValueError, TypeError) as error:
                result = {"error": f"无效的工具调用：{error}"}
                print(f"\n工具调用无效：{error}")
                on_event("工具请求无效")
            else:
                if tool_name == "propose_memory":
                    item = arguments["text"].strip()
                    existing = load_memory()
                    if item in existing:
                        result = {"saved": False, "reason": "这条记忆已经存在"}
                        print("这条记忆已经存在。")
                    elif len(existing) >= 20:
                        result = {"saved": False, "reason": "记忆已满，请先删除"}
                        print("记忆已满，请先用 /forget 删除。")
                    else:
                        print(f"\n建议保存的项目记忆：{item}")
                        if confirm("保存这条记忆？[y/N] ").strip().lower() == "y":
                            existing.append(item)
                            save_memory(existing)
                            if memory is not None:
                                memory[:] = existing
                            result = {"saved": True}
                            print("已保存项目记忆。")
                            on_event("用户同意保存记忆")
                        else:
                            result = {"saved": False, "reason": "用户拒绝保存"}
                            print("已拒绝保存记忆。")
                            on_event("用户拒绝保存记忆")
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    save()
                    continue
                if tool_name == "read_skill":
                    name = arguments["name"]
                    try:
                        result = {"name": name, "instructions": load_skill(name)}
                        print(f"\n已读取技能：{name}")
                        on_event(f"已读取技能：{name}")
                    except (OSError, ValueError, UnicodeError):
                        result = {"error": "无法读取技能，请检查技能文件。"}
                        print(f"\n无法读取技能：{name}")
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    save()
                    continue
                command = arguments["command"]
                print(f"\n第 {step} 轮工具请求：run_command\n命令：{command}")
                denied = permission == "deny"
                if denied or confirm("执行此命令？[y/N] ").strip().lower() != "y":
                    reason = "权限模式拒绝执行" if denied else "用户拒绝执行"
                    on_event(reason)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"error": reason}, ensure_ascii=False)})
                    repair_pending_calls(messages, "本轮已停止，此工具未执行。")
                    save()
                    print(f"{reason}，任务停止。")
                    on_finish("interrupted")
                    return messages
                on_event("用户同意执行")
                result = execute(command)
                on_event(f"命令执行结束，退出码：{result['returncode']}")
                print(f"工具结果：{result}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            save()

    print(f"已达到 {MAX_STEPS} 轮上限，任务停止。")
    on_finish("interrupted")
    return messages


def run_session_turn(session, task, model, confirm=input, save=None, memory=None, skill=None):
    """在一个会话里记录一次工作，保存开始和结束状态。"""
    if save is None:
        save = lambda: save_session(session)
    turn = {"id": uuid.uuid4().hex, "task": task, "status": "running", "events": ["工作开始"]}
    session.setdefault("turns", []).append(turn)
    save()

    def finish(status):
        turn["status"] = status
        turn["events"].append(f"工作结束：{status}")
        save()

    def record_event(event):
        turn["events"].append(event)
        save()

    print(f"开始工作：{turn['id'][:8]}")
    try:
        run_agent(task, model, confirm=confirm, messages=session["messages"], save=save,
                  permission=session.get("permission", "ask"), on_finish=finish, on_event=record_event, memory=memory, skill=skill)
    except (KeyboardInterrupt, EOFError):
        finish("interrupted")
        raise
    except Exception:
        finish("failed")
        raise
    finally:
        print(f"工作状态：{turn['status']}")


def recover_turns(session):
    """恢复时标记上次未结束的工作，旧会话不推测工作状态。"""
    turns = session.setdefault("turns", [])
    recovered = False
    for turn in turns:
        if turn.get("status") == "running":
            turn["status"] = "interrupted"
            if "events" in turn:
                turn["events"].append("恢复会话：上次工作中断")
            recovered = True
    return recovered


def main():
    parser = argparse.ArgumentParser(description="体验一个带结构化工具调用的 Mini-Agent")
    parser.add_argument("--demo", action="store_true", help="使用固定剧本，不调用 API")
    parser.add_argument("--list", action="store_true", help="列出已保存的会话")
    parser.add_argument("--resume", metavar="ID", help="继续指定会话")
    args = parser.parse_args()
    if args.list:
        if not SESSION_DIR.exists():
            print("还没有保存的会话。")
            return
        for path in sorted(SESSION_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                session = load_session(path.stem)
                first_task = next((m.get("content", "") for m in session["messages"] if m.get("role") == "user"), "（空会话）")
                print(f"{session['id']}  [{session.get('mode', 'unknown')}]  {first_task[:60]}")
            except (ValueError, OSError, json.JSONDecodeError):
                print(f"{path.name}  （无法读取）")
        return
    try:
        session = load_session(args.resume) if args.resume else {
            "id": uuid.uuid4().hex, "mode": "demo" if args.demo else "deepseek",
            "messages": [{"role": "system", "content": SYSTEM_MESSAGE}],
        }
        if args.resume and args.demo and session.get("mode") != "demo":
            parser.exit(1, "真实模型会话不能以演示模式恢复。\n")
        if session.get("mode") not in ("demo", "deepseek"):
            parser.exit(1, "未知的会话模式。\n")
        if session.get("permission", "ask") not in ("ask", "deny"):
            parser.exit(1, "会话的权限模式无效，已拒绝启动。\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"无法读取会话：{error}\n")
    if session["mode"] == "demo":
        print("演示模式：固定剧本会请求列出当前目录；无需 API 密钥。")
        model = DemoModel()
    else:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            parser.exit(1, "请先设置 DEEPSEEK_API_KEY，或使用 python agent.py --demo。\n")
        try:
            model = DeepSeekModel()
        except ImportError:
            parser.exit(1, "请先安装 openai 包：python -m pip install openai\n")
        except ValueError as error:
            parser.exit(1, f"配置错误：{error}\n")
    try:
        if recover_turns(session):
            print("上次有未结束的工作，已标记为中断。")
        if repair_pending_calls(session["messages"]):
            print("上次有未记录结果的工具请求：执行状态未知，请检查实际状态；不会自动重试。")
        save_session(session)
        session.setdefault("permission", "ask")
        print(f"会话 ID：{session['id']}（/exit 退出，/history 查看对话，/turns 查看工作，/events 查看事件）")
        selected_skill = None
        while True:
            task = input("\n你：").strip()
            if task == "/exit":
                break
            if task == "/history":
                for message in session["messages"]:
                    if message.get("role") in ("user", "assistant") and message.get("content"):
                        print(f"{message['role']}: {message['content']}")
                continue
            if task == "/turns":
                if not session["turns"]:
                    print("还没有可显示的工作记录（旧版对话不会推测状态）。")
                for index, turn in enumerate(session["turns"], 1):
                    print(f"{index}. [{turn['status']}] {turn['task']}（{turn['id'][:8]}）")
                continue
            if task == "/skills":
                skills = list_skills()
                if not skills:
                    print("没有找到技能。")
                for skill in skills:
                    print(f"{skill['name']}：{skill['description']}")
                continue
            if task.startswith("/skills "):
                print("查看技能请输入 /skills；选择技能请输入 /skill explain-code。")
                continue
            if task == "/skill" or task.startswith("/skill "):
                name = task[len("/skill"):].strip()
                if name not in [skill["name"] for skill in list_skills()]:
                    print("没有找到这个技能。请输入 /skills 查看。")
                else:
                    selected_skill = name
                    print(f"已选择 {name}，将在下一次任务中使用。")
                continue
            if task == "/memory":
                memory = load_memory()
                if not memory:
                    print("目前没有项目记忆。")
                for index, item in enumerate(memory, 1):
                    print(f"{index}. {item}")
                continue
            if task == "/remember" or task.startswith("/remember "):
                item = task[len("/remember"):].strip()
                memory = load_memory()
                if not item or len(item) > 500:
                    print("请输入 1 到 500 个字，例如 /remember 我喜欢简短解释。")
                elif len(memory) >= 20:
                    print("最多保存 20 条记忆，请先用 /forget 删除。")
                else:
                    memory.append(item)
                    save_memory(memory)
                    print("已保存项目记忆。")
                continue
            if task == "/forget" or task.startswith("/forget "):
                parts = task.split()
                memory = load_memory()
                if len(parts) != 2 or not parts[1].isascii() or not parts[1].isdigit() or not 1 <= int(parts[1]) <= len(memory):
                    print("请先输入 /memory 查看编号，再输入 /forget 编号。")
                else:
                    memory.pop(int(parts[1]) - 1)
                    save_memory(memory)
                    print("已删除这条项目记忆。")
                continue
            if task == "/events" or task.startswith("/events "):
                parts = task.split()
                if len(parts) != 2 or not parts[1].isascii() or not parts[1].isdigit() or not 1 <= int(parts[1]) <= len(session["turns"]):
                    print("请先输入 /turns 查看编号，再输入 /events 编号，例如 /events 1。")
                    continue
                turn = session["turns"][int(parts[1]) - 1]
                print(f"工作：{turn['task']}")
                events = turn.get("events", [])
                if not events:
                    print("这次工作没有事件记录（旧版未记录）。")
                for index, event in enumerate(events, 1):
                    print(f"{index}. {event}")
                continue
            if task == "/permissions":
                print(f"当前权限：{session['permission']}。输入 /permissions ask 或 /permissions deny 切换。")
                continue
            if task.startswith("/permissions "):
                permission = task.removeprefix("/permissions ").strip()
                if permission not in ("ask", "deny"):
                    print("未知权限模式。可选：ask、deny。")
                else:
                    session["permission"] = permission
                    save_session(session)
                    print(f"权限模式已设为 {permission}。")
                continue
            if task:
                skill = load_skill(selected_skill) if selected_skill else None
                selected_skill = None
                run_session_turn(session, task, model, confirm=input, memory=load_memory(), skill=skill)
    except (KeyboardInterrupt, EOFError):
        print("\n已停止。")
    except UnicodeEncodeError as error:
        frame = traceback.extract_tb(error.__traceback__)[-1]
        print(f"编码错误：{error.encoding}；位置：{Path(frame.filename).name}:{frame.lineno}（{frame.name}）。")
        print("请把这两行错误提示发给我们定位；不要发送 API Key。")
    except Exception as error:
        print(f"运行失败：{error}")


if __name__ == "__main__":
    main()
