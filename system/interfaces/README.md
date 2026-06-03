# system/interfaces/

`system/interfaces/` 是外部交互适配层。

目标很直接：让 CLI、HTTP/API、桌面 GUI、Web GUI 这类“入口壳”只负责接线、协议转换、状态包装，不再直接依赖 runtime 实现目录。

## 子目录职责

- `cli/`
  CLI 参数注册和入口转发。
- `api/`
  程序化 adapter 与 HTTP 壳。当前包含 `chat` / `computer` / `automation` adapter，以及 `http.py` 里的 JSON 路由壳。
- `gui/`
  GUI session/controller 与统一 app-shell。当前包含各领域 GUI adapter，以及 `shell.py` 里的桌面/Web 组合壳。

## 当前约定

- 接口层只做输入归一化、状态包装、序列化和协议转换。
- 接口层优先依赖 `system/domain/*` 稳定门面，不直接绑定 `system/chat_v2/`、`system/computer_use/`、`system/automation/` 等实现目录。
- 新的 HTTP handler、桌面窗口、Web 前端，都优先挂到 `system/interfaces/`，不要把实现层 import 回壳代码。

## 推荐接线路径

```text
CLI: main.py -> system.interfaces.cli -> system.app_runtime
HTTP/API: service or handler -> system.interfaces.api.http -> system.interfaces.api.* -> system.domain.*
Desktop/Web GUI: desktop shell or web frontend -> system.interfaces.gui.shell -> system.interfaces.gui.* -> system.interfaces.api.* -> system.domain.*
```

## 当前壳

- `system/interfaces/api/http.py`
  标准库 JSON/HTTP 壳，提供 `/healthz`、`/api/chat/respond`、`/api/computer/run`、`/api/automation/*`。
- `system/interfaces/gui/shell.py`
  统一桌面/Web GUI 壳，组合 `chat` transcript、`computer` run history、`automation` controller state。
