<<<<<<< HEAD
# Offline Multimodal System

一个以本地运行和多能力编排为目标的 Python 项目，当前主入口已经收敛到 `main.py`，核心运行时位于 `system/`。

目前实际可用的主路径：

- `--chat`：对话运行时，支持 OpenAI 兼容接口，包括 DeepSeek。
- `--computer-use-goal`：桌面任务执行。
- `--automation-*`：自动化工作流执行。
- `--rag-ingest`：知识库索引构建。

## 目录概览

```text
.
|-- main.py
|-- system/                  # 运行时、领域服务、接口适配层
|-- config/                  # 配置文件
|-- scripts/                 # 辅助脚本
|-- tests/                   # 测试与脚本化验证
|-- artifacts/               # 运行产物
`-- README.md
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

如果要跑验证和测试，再补装开发依赖：

```powershell
pip install -e ".[dev]"
```

## DeepSeek 配置

推荐在项目根目录放一个 `.env.local`：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
V2_LLM_PROVIDER=deepseek
V2_LLM_MODEL=deepseek-v4-flash
```

项目启动时会自动读取：

- `.env.local`
- `config/runtime_secrets.env`

## DeepSeek 运行

交互式聊天：

```powershell
.\.venv\Scripts\python.exe main.py `
  --chat `
  --cloud-llm `
  --cloud-provider deepseek `
  --cloud-model deepseek-v4-flash `
  --rag-disable
```

如果已经在 `.env.local` 里设置了 `V2_LLM_PROVIDER=deepseek` 和 `V2_LLM_MODEL=deepseek-v4-flash`，命令可以再简化。

## DeepSeek 验证

先做配置检查，不发请求：

```powershell
.\.venv\Scripts\python.exe scripts\verify_deepseek.py --config-only
```

再做真实 API 调用验证：

```powershell
.\.venv\Scripts\python.exe scripts\verify_deepseek.py
```

脚本会输出：

- 解析后的 `provider` / `model` / `endpoint`
- 是否检测到 API Key
- 真实网络调用结果
- 常见错误提示，例如 `401` 无效密钥、`402` 余额不足

## 本地验证

仓库里有几组可以直接运行的轻量验证脚本：

```powershell
.\.venv\Scripts\python.exe tests\chat_v2_cloud_test.py
.\.venv\Scripts\python.exe tests\runtime_bootstrap_local_env_test.py
.\.venv\Scripts\python.exe tests\chat_v2_runtime_test.py
.\.venv\Scripts\python.exe tests\interfaces_entrypoints_test.py
```

如果已安装 `pytest`，也可以直接跑：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\chat_v2_cloud_test.py tests\runtime_bootstrap_local_env_test.py tests\chat_v2_runtime_test.py tests\interfaces_entrypoints_test.py -q
```

## 版本说明

根据 DeepSeek 官方文档，`deepseek-v4-flash` / `deepseek-v4-pro` 是当前推荐模型；`deepseek-chat` 和 `deepseek-reasoner` 标记为将在 `2026-07-24` 退役。因此仓库默认 DeepSeek 预设已经切到 `deepseek-v4-flash`。
=======
# niubiclaw
没有完成的全能ai
>>>>>>> 1817aafc54ef74eb8884fc574d2d644f2981a82e
