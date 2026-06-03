from __future__ import annotations

from typing import Callable, List


class TaskFlow:
    def __init__(self):
        self.available = False
        self._flow_cache = {}
        try:
            from prefect import flow, task

            self.flow = flow
            self.task = task
            self.available = True
        except Exception:
            self.available = False

    def run(self, funcs: List[Callable]):
        if not self.available:
            results = []
            for f in funcs:
                try:
                    results.append(f())
                except Exception as e:
                    results.append(e)
            return results

        sig = tuple(id(f) for f in funcs)
        if sig not in self._flow_cache:
            @self.flow(name="pef_flow")
            def _flow():
                results = []
                for i, f in enumerate(funcs):
                    @self.task(name=f"task_{i}")
                    def _t(fn=f):
                        return fn()
                    results.append(_t())
                return results
            self._flow_cache[sig] = _flow

        return self._flow_cache[sig]()

    def resolve(self, results: List):
        if not self.available:
            return results
        resolved = []
        for r in results:
            try:
                # Prefect 2.x State / Future
                if hasattr(r, "is_failed") and r.is_failed():
                    resolved.append(None)
                    continue
                if hasattr(r, "result"):
                    resolved.append(r.result())
                else:
                    resolved.append(r)
            except Exception:
                resolved.append(None)
        return resolved
