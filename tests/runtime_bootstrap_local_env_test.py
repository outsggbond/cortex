from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from system.runtime import bootstrap as runtime_bootstrap


def test_load_local_runtime_env_from_custom_candidate() -> None:
    with tempfile.TemporaryDirectory(dir=(ROOT / ".tmp").as_posix()) as td:
        env_path = Path(td) / "runtime.env"
        env_path.write_text(
            "# local secrets\nDEEPSEEK_API_KEY=sk-test-value\nOTHER_FLAG='quoted value'\n",
            encoding="utf-8-sig",
        )
        old_candidates = tuple(runtime_bootstrap._LOCAL_ENV_CANDIDATES)
        old_deepseek = os.environ.pop("DEEPSEEK_API_KEY", None)
        old_other = os.environ.pop("OTHER_FLAG", None)
        runtime_bootstrap._LOCAL_ENV_CANDIDATES = (env_path.as_posix(),)
        try:
            runtime_bootstrap.load_local_runtime_env()
            assert os.environ.get("DEEPSEEK_API_KEY") == "sk-test-value"
            assert os.environ.get("OTHER_FLAG") == "quoted value"
        finally:
            runtime_bootstrap._LOCAL_ENV_CANDIDATES = old_candidates
            if old_deepseek is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_deepseek
            if old_other is None:
                os.environ.pop("OTHER_FLAG", None)
            else:
                os.environ["OTHER_FLAG"] = old_other


def main() -> None:
    (ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    test_load_local_runtime_env_from_custom_candidate()
    print("runtime_bootstrap_local_env_ok")


if __name__ == "__main__":
    main()
