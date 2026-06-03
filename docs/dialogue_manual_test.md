# 对话手工测试脚本（无硬编码）

## 1) 开场说明（可直接粘贴到会话）
我确认了根因：这句话不是“硬编码回复”，而是已经被写进 `dialogue_patterns.json` 和语义索引，`dialogue` 路由又在 `pure-chat` 里优先命中它。现在开始做回归测试，请按自然对话回答，不要固定复读同一句模板。

## 2) 启动命令
```powershell
python main.py --chat --pure-chat --transformer-mode off
```

如果 PowerShell 中文乱码：
```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
```

## 3) 最小回归用例（逐条输入）
1. `你好`
2. `你叫什么名字？`
3. `你能做什么？`
4. `帮我写一个 Python hello world`
5. `把上一步改成函数`
6. `谢谢`
7. `exit`

## 4) 通过标准
1. 不崩溃（无 traceback）
2. 不连续复读同一模板句
3. 对问候/自我介绍能给出语义相关回复
4. 指令型输入能给出具体内容（不是泛化“请补充信息”）

## 5) 失败判定（建议记录）
出现以下任一情况判定失败：
1. 同一句低信息模板连续出现 >= 2 次
2. 明确问题（如“你叫什么名字？”）仍被“请补充信息”类回复覆盖
3. 输出乱码或空回复

## 6) 测试记录模板
```text
[time] 2026-02-23 xx:xx
[cmd] python main.py --chat --pure-chat --transformer-mode off
[input] 你好
[reply] ...
[result] PASS/FAIL
[note] ...
```

---

# Dialogue Manual Test Script (No Hardcoding)

## 1) Opening Note (can be pasted directly into session)
I have confirmed the root cause: this phrase is not a "hardcoded reply", but has already been written into `dialogue_patterns.json` and the semantic index, and the `dialogue` route hits it with priority in `pure-chat`. Now starting regression testing. Please respond with natural conversation and do not repeat the same template phrase verbatim.

## 2) Startup Command
```powershell
python main.py --chat --pure-chat --transformer-mode off
```

If PowerShell displays garbled Chinese characters:
```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
```

## 3) Minimal Regression Cases (input one by one)
1. `Hello`
2. `What is your name?`
3. `What can you do?`
4. `Write me a Python hello world`
5. `Turn the previous step into a function`
6. `Thank you`
7. `exit`

## 4) Pass Criteria
1. No crash (no traceback)
2. Does not continuously repeat the same template phrase
3. Gives semantically relevant replies to greetings/self-introductions
4. Gives concrete content for instructional inputs (not generic "please provide more information")

## 5) Failure Criteria (recommended to record)
A failure is determined if any of the following occurs:
1. The same low-information template appears consecutively >= 2 times
2. A clear question (e.g., "What is your name?") is still overridden by a "please provide more information" type reply
3. Output is garbled or empty

## 6) Test Record Template
```text
[time] 2026-02-23 xx:xx
[cmd] python main.py --chat --pure-chat --transformer-mode off
[input] Hello
[reply] ...
[result] PASS/FAIL
[note] ...
```
