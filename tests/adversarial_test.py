import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.brain.neurosymbolic import build_default_engine


def test_adversarial_rules():
    engine = build_default_engine()
    # conflicting constraints
    msg = "必须完成但不能执行"
    out = engine.infer({"message": msg})
    assert isinstance(out, dict)


if __name__ == "__main__":
    test_adversarial_rules()
    print("adversarial_ok")
