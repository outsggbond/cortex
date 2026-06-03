# 桌面智能体运行时

本项目现已包含一个专用的桌面 `computer-use` 运行时，与聊天代理和工作流扫描器相互独立。

## 功能说明

- 捕获当前桌面截图
- 在截图之上构建带标签的网格覆盖层
- 运行 OCR 并将识别到的文本映射到网格块中
- 读取活动窗口和可选的 UI Automation 控件信息
- 将原始截图、网格覆盖层、OCR 摘要和控件摘要发送至多模态 LLM
- 每次执行一个桌面动作
- 每个动作执行后重新观察桌面
- 存储每次运行的追踪记录，包含截图、网格覆盖层和最终摘要

## 为什么网格很重要

运行时不会强迫模型猜测原始像素坐标。
相反，它为模型提供第二张图像，其中包含稳定的块标签，例如 `A1`、`B3` 或 `F7`。
这使得可以选择以下方式：

- `click_control`：当 UI Automation 能够找到元素时
- `click_text`：当 OCR 能够找到元素时
- `click_block`：当模型需要视觉回退时

## 安装

安装浏览器或桌面自动化扩展：

```bash
pip install -e ".[computer]"
```

在 Windows 上，桌面路径受益于以下库：

- `pywinauto`
- `pywin32`
- `pytesseract` 或 `rapidocr-onnxruntime`

## 运行

使用 API 支持的多模态模型示例：

```powershell
$env:OPENAI_API_KEY="your-key"
python main.py `
  --computer-use-goal "打开QQ，给当前聊天发送一个简短的问候" `
  --cloud-llm `
  --cloud-model gpt-4.1-mini
```

空运行模式：

```powershell
python main.py `
  --computer-use-goal "打开QQ，给当前聊天发送一个简短的问候" `
  --cloud-llm `
  --cloud-model gpt-4.1-mini `
  --computer-use-dry-run
```

可选调优参数：

- `--computer-use-max-steps`
- `--computer-use-grid-rows`
- `--computer-use-grid-cols`
- `--computer-use-target-window`
- `--computer-use-image-detail`
- `--computer-use-allow-high-risk`

## 输出

每次运行会在以下目录下写入一个带时间戳的文件夹：

```text
artifacts/audit/computer_use/
```

典型文件：

- `step_01_screen.png`
- `step_01_grid.png`
- `trace.jsonl`
- `summary.json`
- `recovery.json`

`summary.json` 除最终状态外，现在还包括紧凑的运行摘要，例如：

- `verified_step_count` — 已验证步骤数
- `failed_step_count` — 失败步骤数
- `verification_rate` — 验证通过率
- `action_counts` — 动作计数
- `unique_windows` — 唯一窗口数
- `recent_steps` — 最近步骤
- `latest_active_window` — 最近活动窗口
- `latest_nonempty_blocks` — 最近非空网格块

运行时还会将最新的可恢复检查点写入：

```text
artifacts/audit/computer_use_recovery.json
```

当运行以 `failed`（失败）或 `incomplete`（未完成）结束时，可以使用以下命令恢复：

```powershell
python main.py `
  --computer-use-goal "恢复上一次 computer-use 任务" `
  --cloud-llm `
  --cloud-model gpt-4.1-mini
```

恢复流程会复用已保存的目标，并在可用时确定性地重放已保存的修复动作，然后返回正常的模型引导步骤。

默认情况下，运行时还提供三种安全/可靠性行为：

- 在遇到可见的验证码、人工验证或反自动化挑战时自动交还控制权，而不是试图暴力突破。
- 阻止高风险动作，如启动进程、敏感的文本自由输入和危险的热键，除非设置了 `--computer-use-allow-high-risk`。
- 将当前界面标注为 `uia`、`ocr_only` 或 `visual_only`，并标记可能为自定义绘制的 UI，以便在控件不可靠时模型可以优先选择文本/网格块定位。

## 当前范围

运行时专注于 Windows 桌面控制。
它有意采用逐步且保守的策略：

- 每次模型轮次只执行一个动作
- 每次动作后显式进行观察
- 结构化追踪记录，便于重放和调试

这使得它比简单的宏更慢，但调试和扩展要容易得多。

---

# Computer Use Runtime

This project now includes a dedicated desktop `computer-use` runtime that is separate from the chat agent and the workflow scanner.

## What it does

- Captures the current desktop screenshot
- Builds a labeled grid overlay on top of the screenshot
- Runs OCR and maps recognized text into grid blocks
- Reads the active window and optional UI Automation controls
- Sends the raw screenshot, grid overlay, OCR summary, and control summary to a multimodal LLM
- Executes one desktop action at a time
- Re-observes the desktop after every action
- Stores a per-run trace with screenshots, grid overlays, and a final summary

## Why the grid matters

The runtime does not force the model to guess raw pixel coordinates.
Instead, it gives the model a second image with stable block labels such as `A1`, `B3`, or `F7`.
That makes it possible to choose:

- `click_control` when UI Automation can find the element
- `click_text` when OCR can find the element
- `click_block` when the model needs a visual fallback

## Install

For browser or desktop automation extras:

```bash
pip install -e ".[computer]"
```

On Windows, the desktop path benefits from:

- `pywinauto`
- `pywin32`
- `pytesseract` or `rapidocr-onnxruntime`

## Run

Example with an API-backed multimodal model:

```powershell
$env:OPENAI_API_KEY="your-key"
python main.py `
  --computer-use-goal "Open QQ and send a short hello to the current chat" `
  --cloud-llm `
  --cloud-model gpt-4.1-mini
```

Dry run:

```powershell
python main.py `
  --computer-use-goal "Open QQ and send a short hello to the current chat" `
  --cloud-llm `
  --cloud-model gpt-4.1-mini `
  --computer-use-dry-run
```

Optional tuning:

- `--computer-use-max-steps`
- `--computer-use-grid-rows`
- `--computer-use-grid-cols`
- `--computer-use-target-window`
- `--computer-use-image-detail`
- `--computer-use-allow-high-risk`

## Output

Each run writes a timestamped folder under:

```text
artifacts/audit/computer_use/
```

Typical files:

- `step_01_screen.png`
- `step_01_grid.png`
- `trace.jsonl`
- `summary.json`
- `recovery.json`

`summary.json` now includes a compact run digest in addition to the final status, for example:

- `verified_step_count`
- `failed_step_count`
- `verification_rate`
- `action_counts`
- `unique_windows`
- `recent_steps`
- `latest_active_window`
- `latest_nonempty_blocks`

The runtime also writes the latest resumable checkpoint to:

```text
artifacts/audit/computer_use_recovery.json
```

When a run ends as `failed` or `incomplete`, you can resume it with:

```powershell
python main.py `
  --computer-use-goal "resume last computer-use task" `
  --cloud-llm `
  --cloud-model gpt-4.1-mini
```

The resume flow reuses the saved goal and, when available, deterministically replays the saved repair actions before returning to normal model-guided steps.

By default, the runtime also adds three safety / reliability behaviors:

- It auto-handoffs on visible captcha, human-verification, or anti-automation challenges instead of trying to brute-force through them.
- It blocks high-risk actions such as process launches, sensitive free-text input, and dangerous hotkeys unless `--computer-use-allow-high-risk` is set.
- It labels the current surface as `uia`, `ocr_only`, or `visual_only`, and marks likely custom-drawn UIs so the model can prefer text/block targeting when controls are unreliable.

## Current scope

The runtime is focused on Windows desktop control.
It is intentionally stepwise and conservative:

- one action per model turn
- explicit observation after every action
- structured traces for replay and debugging

That makes it slower than a naive macro, but much easier to debug and extend.
