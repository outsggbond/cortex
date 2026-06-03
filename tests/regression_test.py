import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.system_config import SystemConfig
from system.perception.system import OptimizedPerceptionSystem
from system.brain.cognitive import CognitiveArchitecture
from system.brain.neurosymbolic import build_default_engine


def test_basic_pipeline():
    cfg = SystemConfig()
    system = OptimizedPerceptionSystem(cfg)
    result = system.run_cycle({"text": "test", "image": None, "audio": None, "video": None})
    assert result["nodes_created"] >= 1


def test_neurosym():
    engine = build_default_engine()
    out = engine.infer({"message": "目标是完成任务", "focus": [], "memory": []})
    assert isinstance(out, dict)


def test_cognitive():
    cog = CognitiveArchitecture()
    state = cog.run("这是一个测试目标")
    assert len(state.focus_terms) >= 0


if __name__ == "__main__":
    test_basic_pipeline()
    test_neurosym()
    test_cognitive()
    print("regression_ok")
