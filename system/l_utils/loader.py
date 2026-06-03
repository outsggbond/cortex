from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModuleLoader:
    """安全、健壮的动态模块加载器。

    所有模块路径均被强制限定在 registry.json 所在目录（模块根目录）内。
    注册表中存储相对路径，支持热加载、卸载与注册钩子。
    """

    def __init__(self, registry_path: str = "modules/registry.json"):
        self.registry_path = Path(registry_path).resolve()
        # 模块根目录即 registry.json 所在文件夹
        self.modules_dir = self.registry_path.parent
        # 内部缓存：{spec.name: module}
        self._loaded: Dict[str, ModuleType] = {}
        self._lock = threading.Lock()

        if not self.registry_path.exists():
            self.modules_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_registry([])

    # ------------------------------------------------------------------
    # 注册表原子读写
    # ------------------------------------------------------------------
    def _read_registry(self) -> List[Dict[str, str]]:
        """安全读取注册表，返回列表，解析失败时返回空列表。"""
        try:
            with self._lock:
                text = self.registry_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Registry top-level must be a list")
            # 确保所有元素都是带 "path" 的字典
            for item in data:
                if not isinstance(item, dict) or "path" not in item:
                    raise ValueError("Invalid registry entry format")
            return data
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error("Failed to read registry %s: %s", self.registry_path, e)
            return []

    def _atomic_write_registry(self, data: List[Dict[str, str]]) -> None:
        """原子写入：先写临时文件，再替换，避免读到不完整数据。"""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.registry_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.registry_path)
        except OSError as e:
            logger.error("Atomic write failed for registry: %s", e)
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 路径安全工具
    # ------------------------------------------------------------------
    def _resolve_module_path(self, path: str) -> Path:
        """将输入的相对或绝对路径解析为限定在 modules_dir 内的绝对路径。

        相对路径以 modules_dir 为基准；绝对路径直接使用。
        强制校验最终路径不在 modules_dir 之外。
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (self.modules_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        # 路径穿越检测
        try:
            candidate.relative_to(self.modules_dir)
        except ValueError:
            raise ValueError(
                f"Module path '{path}' escapes root directory {self.modules_dir}"
            )
        return candidate

    # ------------------------------------------------------------------
    # 公共 API（保留原接口）
    # ------------------------------------------------------------------
    def list_modules(self) -> List[str]:
        """返回注册表中所有模块的相对路径。"""
        data = self._read_registry()
        return [item["path"] for item in data]

    def load_module(self, path: str) -> Optional[ModuleType]:
        """加载指定模块，返回模块对象，失败返回 None。"""
        try:
            mod_path = self._resolve_module_path(path)
        except ValueError as e:
            logger.error("Path resolution failed: %s", e)
            return None

        if not mod_path.exists():
            logger.error("Module file not found: %s", mod_path)
            return None

        spec = importlib.util.spec_from_file_location(mod_path.stem, str(mod_path))
        if not spec or not spec.loader:
            logger.error("Could not create spec/loader for %s", mod_path)
            return None

        # 使用 spec.name 作为唯一标识，避免重复加载
        if spec.name in self._loaded:
            logger.info("Module '%s' already loaded, returning cached instance", spec.name)
            return self._loaded[spec.name]

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error("Failed to execute module %s: %s", mod_path, e, exc_info=True)
            return None

        # 插入 sys.modules 以保持一致性（但不强制，供其他代码引用）
        sys_modules = __import__("sys").modules
        sys_modules[spec.name] = module
        self._loaded[spec.name] = module
        logger.info("Module '%s' loaded successfully from %s", spec.name, mod_path)
        return module

    def load_all(self) -> Dict[str, Any]:
        """加载注册表中的所有模块，返回 {相对路径: 模块对象} 字典。

        同时会对每个模块调用 `register()` 函数（若存在），用于插件自注册。
        """
        loaded: Dict[str, Any] = {}
        for rel_path in self.list_modules():
            mod = self.load_module(rel_path)
            if mod is not None:
                loaded[rel_path] = mod
                # 尝试调用模块的注册函数
                if hasattr(mod, "register") and callable(mod.register):
                    try:
                        mod.register()
                        logger.debug("Module '%s' registered", rel_path)
                    except Exception as e:
                        logger.error("Module '%s' register() failed: %s", rel_path, e)
        return loaded

    # ------------------------------------------------------------------
    # 新增 API：注册表管理 & 卸载
    # ------------------------------------------------------------------
    def add_module(self, path: str) -> bool:
        """向注册表添加一个模块路径（自动转为相对于根目录的相对路径）。"""
        try:
            abs_path = self._resolve_module_path(path)
        except ValueError:
            logger.error("Cannot add module: invalid path %s", path)
            return False
        rel_path = str(abs_path.relative_to(self.modules_dir))

        with self._lock:
            data = self._read_registry()
            if any(item["path"] == rel_path for item in data):
                logger.warning("Module %s already in registry", rel_path)
                return False
            data.append({"path": rel_path})
            self._atomic_write_registry(data)
        logger.info("Module added to registry: %s", rel_path)
        return True

    def remove_module(self, path: str) -> bool:
        """从注册表移除一个模块路径（使用相对路径匹配）。"""
        data = self._read_registry()
        new_data = [item for item in data if item["path"] != path]
        if len(new_data) == len(data):
            logger.warning("Module %s not found in registry", path)
            return False
        self._atomic_write_registry(new_data)
        logger.info("Module removed from registry: %s", path)
        return True

    def unload_module(self, path: str) -> bool:
        """卸载指定模块，清除内部缓存并从 sys.modules 中移除。"""
        try:
            mod_path = self._resolve_module_path(path)
        except ValueError:
            return False
        spec_name = mod_path.stem  # 与 load_module 中使用的 spec.name 一致
        if spec_name not in self._loaded and spec_name not in __import__("sys").modules:
            logger.warning("Module %s not loaded", spec_name)
            return False
        self._loaded.pop(spec_name, None)
        __import__("sys").modules.pop(spec_name, None)
        logger.info("Module '%s' unloaded", spec_name)
        return True
