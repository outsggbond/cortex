import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.core.hardware import profile_hardware
from system.automation.executor import TaskExecutor
from system.core.planner import Task, Plan
from system.evaluation.self_heal import apply_self_heal


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def setup_case1(root: Path) -> None:
    # Missing file reference
    _write(
        root / "app.py",
        "from pathlib import Path\n"
        "def has_data():\n"
        "    return Path('data/config.txt').exists()\n",
    )
    _write(
        root / "run.py",
        "from app import has_data\n"
        "import sys\n"
        "if not has_data():\n"
        "    sys.exit(1)\n"
        "print('ok')\n",
    )
    # no data/config.txt


def setup_case2(root: Path) -> None:
    # Bulk rename dir + update references
    _write(root / "oldpkg" / "__init__.py", "")
    _write(root / "oldpkg" / "util.py", "def answer(): return 42\n")
    _write(root / "helpers.py", "from oldpkg.util import answer\n")
    _write(root / "main.py", "from oldpkg.util import answer\ndef main(): return answer()\n")
    _write(
        root / "run.py",
        "from newpkg.util import answer\n"
        "import sys\n"
        "if answer() != 42:\n"
        "    sys.exit(1)\n"
        "print('ok')\n",
    )


def setup_case3(root: Path) -> None:
    # Add new module + run test
    _write(root / "main.py", "from mod_new import ping\ndef main(): return ping()\n")
    _write(
        root / "run.py",
        "from main import main\n"
        "import sys\n"
        "if main() != 'ok':\n"
        "    sys.exit(1)\n"
        "print('ok')\n",
    )


def plan_case1() -> Plan:
    steps = [
        Task(name="read_file", detail="read data/config.txt", payload={"path": "data/config.txt"}),
        Task(name="run_script", detail="run run.py", payload={"path": "run.py"}),
    ]
    return Plan(goal="fix missing file and run", steps=steps, levels=[steps])


def plan_case2() -> Plan:
    steps = [
        Task(name="move_file", detail="move oldpkg -> newpkg", payload={"src": "oldpkg", "dst": "newpkg"}),
        Task(
            name="write_file",
            detail="write main.py -> from newpkg.util import answer / def main(): return answer()",
            payload={
                "path": "main.py",
                "content": "from newpkg.util import answer\n\n\ndef main():\n    return answer()\n",
            },
        ),
        Task(
            name="write_file",
            detail="write helpers.py -> from newpkg.util import answer",
            payload={"path": "helpers.py", "content": "from newpkg.util import answer\n"},
        ),
        Task(name="run_script", detail="run run.py", payload={"path": "run.py"}),
    ]
    return Plan(goal="rename dir and update refs", steps=steps, levels=[[s] for s in steps])


def plan_case3() -> Plan:
    steps = [
        Task(
            name="write_file",
            detail="write mod_new.py -> def ping(): return 'ok'",
            payload={"path": "mod_new.py", "content": "def ping(): return 'ok'"},
        ),
        Task(name="run_script", detail="run run.py", payload={"path": "run.py"}),
    ]
    return Plan(goal="add module and run", steps=steps, levels=[steps])


def run_case(name: str, root: Path, plan: Plan, allow_create: bool = False) -> dict:
    executor = TaskExecutor(project_root=str(root))
    max_replan = 3
    rounds = 0
    errors = []
    start = time.time()
    execution = executor.run(plan)
    rounds += 1
    if execution.failed and allow_create:
        heal = apply_self_heal(plan, execution, project_root=str(root), allow_create=True, allow_chmod=False)
        if heal.repair_steps:
            repair_plan = Plan(goal="self_heal", steps=heal.repair_steps, levels=[heal.repair_steps])
            _ = executor.run(repair_plan)
        if heal.applied or heal.repair_steps:
            execution = executor.run(heal.plan)
            rounds += 1

    while execution.failed and rounds <= max_replan:
        execution = executor.run(plan)
        rounds += 1

    if execution.failed:
        errors.extend(execution.failed)
    elapsed = time.time() - start
    return {
        "case": name,
        "rounds": rounds,
        "failed": bool(execution.failed),
        "errors": errors,
        "elapsed_sec": elapsed,
    }


def main():
    hw = profile_hardware()
    if hw.total_ram_gb >= 16:
        time_budget = 60
    elif hw.total_ram_gb >= 8:
        time_budget = 40
    else:
        time_budget = 25

    base = ROOT / "tests" / "fixtures" / "stability"
    if base.exists():
        # clean prior runs
        for item in base.glob("*"):
            if item.is_dir():
                for sub in item.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted(item.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
            elif item.is_file():
                item.unlink()
    base.mkdir(parents=True, exist_ok=True)

    case1 = base / "case1"
    case2 = base / "case2"
    case3 = base / "case3"
    case1.mkdir(parents=True, exist_ok=True)
    case2.mkdir(parents=True, exist_ok=True)
    case3.mkdir(parents=True, exist_ok=True)

    setup_case1(case1)
    setup_case2(case2)
    setup_case3(case3)

    results = []
    results.append(run_case("missing_file", case1, plan_case1(), allow_create=True))
    results.append(run_case("rename_dir", case2, plan_case2(), allow_create=False))
    results.append(run_case("new_module", case3, plan_case3(), allow_create=False))

    ok = True
    for r in results:
        if r["rounds"] > 4:  # initial + 3 replans
            ok = False
        if r["failed"]:
            ok = False
        if r["elapsed_sec"] > time_budget:
            ok = False
        print(r)
    print("stability_ok" if ok else "stability_failed")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
