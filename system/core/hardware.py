from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from system.core.auto_tune import classify_resource_level, compute_resource_score, load_policy

# ========================
# 缓存（避免高频调用开销）
# ========================
_CACHE: Dict[str, tuple[float, Any]] = {}
_CACHE_TTL = {
    "cpuinfo": 3600,
    "disk": 60,
    "gpu": 5,
    "network": 10,
    "os": 3600,
}


def _cache_get(key: str):
    if key in _CACHE:
        ts, val = _CACHE[key]
        if time.time() - ts < _CACHE_TTL.get(key, 10):
            return val
    return None


def _cache_set(key: str, val: Any):
    _CACHE[key] = (time.time(), val)


# ========================
# 数据结构（不改原字段）
# ========================
@dataclass
class HardwareProfile:
    cpu_cores: int
    cpu_name: str
    cpu_freq_mhz: float
    cpu_load_1m: float
    cpu_usage_percent: float
    per_core_usage: Tuple[float, ...]
    total_ram_gb: float
    available_ram_gb: float
    swap_total_gb: float
    swap_used_gb: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_vram_gb: float
    os_name: str
    resource_level: str
    resource_score: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ========================
# CPU 深度信息
# ========================
def _get_cpu_deep():
    cached = _cache_get("cpuinfo")
    if cached:
        return cached

    info = {}
    try:
        import cpuinfo

        ci = cpuinfo.get_cpu_info()
        info = {
            "brand": ci.get("brand_raw"),
            "arch": ci.get("arch"),
            "bits": ci.get("bits"),
            "l2_cache": ci.get("l2_cache_size"),
            "flags": ci.get("flags", [])[:10],  # 控制大小
        }
    except Exception:
        pass

    _cache_set("cpuinfo", info)
    return info


# ========================
# 内存分层
# ========================
def _get_memory_detail():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "active_gb": vm.active / (1024**3),
            "inactive_gb": getattr(vm, "inactive", 0) / (1024**3),
            "buffers_gb": getattr(vm, "buffers", 0) / (1024**3),
            "cached_gb": getattr(vm, "cached", 0) / (1024**3),
        }
    except Exception:
        return {}


# ========================
# 磁盘健康（SMART）
# ========================
def _get_disk_health():
    cached = _cache_get("disk")
    if cached:
        return cached

    result = {}
    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        result["usage_percent"] = used / total * 100
    except Exception:
        pass

    # 可选：SMART（Linux才稳定）
    try:
        from pySMART import DeviceList

        devs = DeviceList().devices
        result["smart"] = [
            {
                "name": d.name,
                "health": d.assessment,
                "temp": d.temperature,
            }
            for d in devs
        ]
    except Exception:
        pass

    _cache_set("disk", result)
    return result


# ========================
# GPU 深度（NVML）
# ========================
def _get_gpu_deep():
    cached = _cache_get("gpu")
    if cached:
        return cached

    info = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        info = {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "vram_used_gb": mem.used / (1024**3),
            "temperature": pynvml.nvmlDeviceGetTemperature(handle, 0),
        }
    except Exception:
        pass

    _cache_set("gpu", info)
    return info


# ========================
# 网络能力
# ========================
def _get_network():
    cached = _cache_get("network")
    if cached:
        return cached

    info = {}
    try:
        import psutil

        net = psutil.net_if_stats()
        info = {
            k: {
                "speed_mbps": v.speed,
                "up": v.isup,
            }
            for k, v in net.items()
        }
    except Exception:
        pass

    _cache_set("network", info)
    return info


# ========================
# OS 详细信息
# ========================
def _get_os_detail():
    cached = _cache_get("os")
    if cached:
        return cached

    info = {}
    try:
        import distro

        info = {
            "name": distro.name(),
            "version": distro.version(),
        }
    except Exception:
        info = {
            "platform": platform.platform(),
            "version": platform.version(),
        }

    _cache_set("os", info)
    return info


# ========================
# 原有函数（保持不变）
# ========================
def _get_cpu_name() -> str:
    return platform.processor() or "unknown"


def _get_ram_gb():
    try:
        import psutil

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return (
            vm.total / (1024**3),
            vm.available / (1024**3),
            sm.total / (1024**3),
            sm.used / (1024**3),
        )
    except Exception:
        return 0, 0, 0, 0


def _get_cpu_stats():
    try:
        import psutil

        freq = psutil.cpu_freq()
        usage = psutil.cpu_percent(interval=0.1)
        per = psutil.cpu_percent(interval=0.1, percpu=True)
        return freq.current if freq else 0, usage, tuple(per)
    except Exception:
        return 0, 0, ()


def _get_load_avg():
    try:
        return os.getloadavg()[0]
    except Exception:
        return 0


def _get_disk_gb():
    try:
        import shutil

        total, _, free = shutil.disk_usage(".")
        return total / (1024**3), free / (1024**3)
    except Exception:
        return 0, 0


def _detect_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _get_gpu_info():
    try:
        if not _detect_gpu():
            return "none", 0.0
        import torch

        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return name, vram
    except Exception:
        return "unknown", 0.0


# ========================
# 主入口（增强版）
# ========================
def profile_hardware() -> HardwareProfile:
    cpu_cores = os.cpu_count() or 1
    cpu_name = _get_cpu_name()
    cpu_freq, cpu_usage, per_core = _get_cpu_stats()

    total_ram, avail_ram, swap_total, swap_used = _get_ram_gb()
    disk_total, disk_free = _get_disk_gb()

    gpu_available = _detect_gpu()
    gpu_name, gpu_vram = _get_gpu_info()

    os_name = platform.system().lower()
    load_1m = _get_load_avg()

    # ========= 评分 =========
    score = 0.0
    level = "low"

    try:
        policy = load_policy(os.environ.get("AUTO_TUNE_CONFIG", "config/auto_tune.json"))
        stub = SimpleNamespace(
            cpu_cores=cpu_cores,
            available_ram_gb=avail_ram,
            total_ram_gb=total_ram,
            gpu_available=gpu_available,
            gpu_vram_gb=gpu_vram,
            cpu_usage_percent=cpu_usage,
        )
        score = compute_resource_score(stub, policy)
        level = classify_resource_level(score, policy)
    except Exception:
        pass

    profile = HardwareProfile(
        cpu_cores=cpu_cores,
        cpu_name=cpu_name,
        cpu_freq_mhz=round(cpu_freq, 2),
        cpu_load_1m=round(load_1m, 2),
        cpu_usage_percent=round(cpu_usage, 2),
        per_core_usage=per_core,
        total_ram_gb=round(total_ram, 2),
        available_ram_gb=round(avail_ram, 2),
        swap_total_gb=round(swap_total, 2),
        swap_used_gb=round(swap_used, 2),
        disk_total_gb=round(disk_total, 2),
        disk_free_gb=round(disk_free, 2),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram, 2),
        os_name=os_name,
        resource_level=level,
        resource_score=round(score, 3),
    )

    # ========= 新增深度信息 =========
    profile.extra.update({
        "cpu_detail": _get_cpu_deep(),
        "memory_detail": _get_memory_detail(),
        "disk_health": _get_disk_health(),
        "gpu_detail": _get_gpu_deep(),
        "network": _get_network(),
        "os_detail": _get_os_detail(),
    })

    return profile
