import json
import os
import sys
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.brain.world_model import WorldModel
from system.computer_use.action.generator import ActionGenerator
from system.computer_use.action.stats import ActionStats
from system.computer_use.action.validator import ActionValidator
from system.strategy.search_planner import SearchPlanner
from system.core.planner import TaskPlanner, Plan
from system.automation.executor import TaskExecutor
from system.evaluation.self_heal import apply_self_heal


def _on_rm_error(func, path, _exc):
    try:
        os.chmod(path, 0o666)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_on_rm_error)
    path.mkdir(parents=True, exist_ok=True)


def _setup_maze(root: Path) -> str:
    maze = root / "maze_test"
    if maze.exists():
        shutil.rmtree(maze, onerror=_on_rm_error)
    (maze / "sector_alpha").mkdir(parents=True, exist_ok=True)
    secret_file = maze / "sector_alpha" / "data_source.txt"
    secret = "A1B2-C3D4-E5F6"
    secret_file.write_text(secret, encoding="utf-8")
    try:
        os.chmod(secret_file, 0o000)
    except Exception:
        pass

    beta_file = maze / "sector_beta"
    beta_file.write_text("I am a file, not a directory!", encoding="utf-8")

    utils = maze / "utils"
    utils.mkdir(parents=True, exist_ok=True)
    decoder = utils / "decoder.py"
    decoder.write_text(
        "import sys\n"
        "import hashlib\n\n"
        "if len(sys.argv) < 2:\n"
        "    print(\"Error: No key provided\")\n"
        "    sys.exit(1)\n\n"
        "key = sys.argv[1]\n"
        "result = hashlib.md5(key.encode()).hexdigest()[:8]\n"
        "print('{\"secret_code\": \"%s\"}' % result)\n",
        encoding="utf-8",
    )
    return secret


def _expected_json(key: str) -> dict:
    return {"secret_code": hashlib.md5(key.encode()).hexdigest()[:8]}


def _index_of(tasks, predicate):
    for idx, t in enumerate(tasks):
        if predicate(t):
            return idx
    return None


def _check_sequence(tasks) -> list[str]:
    errors = []
    idx_read = _index_of(tasks, lambda t: t.name == "read_file" and "data_source.txt" in (t.payload or {}).get("path", ""))
    idx_run = _index_of(tasks, lambda t: t.name == "run_script" and "decoder.py" in (t.payload or {}).get("path", ""))
    idx_write = _index_of(tasks, lambda t: t.name == "write_file" and "final_result.json" in (t.payload or {}).get("path", ""))
    idx_move = _index_of(tasks, lambda t: t.name == "move_file" and "sector_beta" in (t.payload or {}).get("src", ""))
    idx_delete = _index_of(tasks, lambda t: t.name == "delete_file" and "sector_beta" in (t.payload or {}).get("path", ""))
    idx_mkdir = _index_of(tasks, lambda t: t.name == "mkdir" and "sector_beta" in (t.payload or {}).get("path", ""))
    idx_chmod = _index_of(tasks, lambda t: t.name == "chmod" and "data_source.txt" in (t.payload or {}).get("path", ""))

    if idx_read is None:
        errors.append("missing read step")
    if idx_run is None:
        errors.append("missing run step")
    if idx_write is None:
        errors.append("missing write step")
    if idx_read is not None and idx_run is not None and idx_run < idx_read:
        errors.append("run before read")
    if idx_run is not None and idx_write is not None and idx_write < idx_run:
        errors.append("write before run")
    if idx_read is not None and idx_write is not None and idx_write < idx_read:
        errors.append("write before read")

    if idx_move is None and idx_delete is None and idx_mkdir is None:
        errors.append("missing move/mkdir for path conflict")
    if idx_move is not None and idx_mkdir is not None and idx_mkdir < idx_move:
        errors.append("mkdir before move")
    if idx_delete is not None and idx_mkdir is not None and idx_mkdir < idx_delete:
        errors.append("mkdir before delete")
    if idx_write is not None:
        if idx_move is not None and idx_write < idx_move:
            errors.append("write before move")
        if idx_delete is not None and idx_write < idx_delete:
            errors.append("write before delete")
        if idx_mkdir is not None and idx_write < idx_mkdir:
            errors.append("write before mkdir")

    if idx_chmod is not None and idx_read is not None and idx_chmod > idx_read:
        errors.append("chmod after read")
    return errors


def _run_case(policy: str, expect_delete: bool | None, require_rollback: bool, disable_distiller: bool) -> tuple[bool, list[str]]:
    os.environ["AGENT_POLICY"] = policy
    if disable_distiller:
        os.environ["DISTILLER_ENABLE"] = "0"
    else:
        os.environ.pop("DISTILLER_ENABLE", None)

    base = ROOT / "tests" / "fixtures" / "labyrinth" / policy
    _reset_dir(base)
    secret = _setup_maze(base)
    expected = _expected_json(secret)

    goal = (
        "Create maze_test/sector_beta/final_result.json using decoded key from "
        "maze_test/sector_alpha/data_source.txt."
    )

    world_model = WorldModel(project_root=str(base))
    action_generator = ActionGenerator(project_root=str(base))
    action_stats = ActionStats()
    action_validator = ActionValidator(project_root=str(base), stats=action_stats)
    planner = TaskPlanner()
    search_planner = SearchPlanner(
        world_model=world_model,
        action_generator=action_generator,
        base_planner=planner,
        depth=6,
        beam_size=6,
        stats=action_stats,
        validator=action_validator,
        project_root=str(base),
    )
    executor = TaskExecutor(project_root=str(base))

    pre_state = world_model.build_state(goal=goal)
    plan = search_planner.plan(goal=goal, state=pre_state, memories=[], model=None, examples=[], feedback="")

    execution = executor.run(plan)
    repair_steps = []
    if execution.failed:
        heal = apply_self_heal(
            plan,
            execution,
            project_root=str(base),
            allow_create=False,
            allow_chmod=True,
        )
        if heal.repair_steps:
            repair_steps = list(heal.repair_steps)
            repair_plan = Plan(goal="self_heal", steps=heal.repair_steps, levels=[heal.repair_steps])
            _ = executor.run(repair_plan)
        if heal.applied or heal.repair_steps:
            execution = executor.run(heal.plan)
            plan = heal.plan

    replan_count = 0
    max_replan = 1
    while execution.failed and replan_count < max_replan:
        pre_state = world_model.build_state(goal=goal, execution=execution)
        plan = search_planner.plan(goal=goal, state=pre_state, memories=[], model=None, examples=[], feedback="")
        execution = executor.run(plan)
        replan_count += 1

    ok = True
    failures = []

    if execution.failed:
        ok = False
        failures.append(f"execution_failed={execution.failed}")
    if replan_count > 1:
        ok = False
        failures.append(f"replan_count={replan_count}")

    final_file = base / "maze_test" / "sector_beta" / "final_result.json"
    if not final_file.exists():
        ok = False
        failures.append("final_result_missing")
    else:
        try:
            data = json.loads(final_file.read_text(encoding="utf-8").strip())
        except Exception as e:
            ok = False
            failures.append(f"final_result_invalid_json:{e}")
            data = None
        if data != expected:
            ok = False
            failures.append(f"final_result_mismatch:{data}")

    sector_alpha = base / "maze_test" / "sector_alpha"
    utils = base / "maze_test" / "utils"
    if not sector_alpha.exists() or not sector_alpha.is_dir():
        ok = False
        failures.append("sector_alpha_missing")
    if not utils.exists() or not utils.is_dir():
        ok = False
        failures.append("utils_missing")

    rollback_dir = base / ".rollback"
    rollback_exists = False
    if rollback_dir.exists():
        for p in rollback_dir.glob("*sector_beta*.block"):
            rollback_exists = True
            break
    if require_rollback and not rollback_exists:
        ok = False
        failures.append("rollback_missing")

    snapshot = execution.snapshot or {}
    diff = snapshot.get("diff") if isinstance(snapshot.get("diff"), dict) else {}
    added = diff.get("added", []) or []
    if added and not any(str(a).endswith("final_result.json") for a in added):
        ok = False
        failures.append("state_diff_missing_final_result")
    if not added:
        if not diff.get("added_count"):
            ok = False
            failures.append("state_diff_empty")

    steps = list(repair_steps) + list(plan.steps)
    seq_errors = _check_sequence(steps)
    if seq_errors:
        ok = False
        failures.append("sequence:" + ";".join(seq_errors))

    if expect_delete is not None:
        has_delete = any(
            t.name == "delete_file" and "sector_beta" in (t.payload or {}).get("path", "")
            for t in steps
        )
        if expect_delete and not has_delete:
            ok = False
            failures.append("delete_expected_missing")
        if not expect_delete and has_delete:
            ok = False
            failures.append("delete_unexpected")

    return ok, failures


def main() -> None:
    os.environ.setdefault("ACTION_USE_LLM", "0")
    os.environ.setdefault("PLANNER_USE_LLM", "0")
    os.environ["EXEC_SNAPSHOT"] = "1"
    os.environ.pop("AGENT_CONTEXT", None)

    results = []
    # Default policy (baseline)
    ok, failures = _run_case("default", expect_delete=None, require_rollback=True, disable_distiller=False)
    results.append(("default", ok, failures))
    # Steady: avoid delete when destructive cost is high.
    ok, failures = _run_case("steady", expect_delete=False, require_rollback=True, disable_distiller=True)
    results.append(("steady", ok, failures))
    # Radical: prefer delete when destructive cost is low.
    ok, failures = _run_case("radical", expect_delete=True, require_rollback=False, disable_distiller=True)
    results.append(("radical", ok, failures))

    overall_ok = all(r[1] for r in results)
    print("labyrinth_ok" if overall_ok else "labyrinth_failed")
    for name, ok, failures in results:
        if ok:
            continue
        for f in failures:
            print(f"- {name}:{f}")
    raise SystemExit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
