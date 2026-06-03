import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.core.hardware import profile_hardware
from system.strategy.strategy import build_strategy, apply_strategy
from config.system_config import SystemConfig
from system.perception.system import OptimizedPerceptionSystem


def main():
    hw = profile_hardware()
    cfg = SystemConfig()
    st = build_strategy(cfg, hw)
    apply_strategy(cfg, st)
    system = OptimizedPerceptionSystem(cfg)
    result = system.run_cycle({"text": "test", "image": None, "audio": None, "video": None})
    score = system.evaluate_cycle(result)
    assert score >= 0.0
    print("smoke_test_ok")


if __name__ == "__main__":
    main()
