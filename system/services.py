# -*- coding: utf-8 -*-
"""Minimal service registry — a lightweight IoC container for dependency injection.

Replaces hardcoded constructor chains in god classes with string-keyed
lazy initialization. Not Spring, not dependency-injector — just a dict
of factory functions with singleton caching.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class ServiceRegistry:
    """A minimal IoC container.

    Usage::

        registry = ServiceRegistry()
        registry.register("world_model", lambda: WorldModel(project_root="."))
        registry.register("action_stats", lambda: ActionStats())

        # Later, in WorkspaceAgent:
        self.world = registry.get("world_model")
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a lazy factory for *name*."""
        key = str(name or "").strip()
        if not key:
            raise ValueError("Service name must be non-empty")
        self._factories[key] = factory
        # Clear cached instance so re-registration takes effect
        self._instances.pop(key, None)

    def get(self, name: str) -> Any:
        """Return the singleton for *name*, creating it on first access."""
        key = str(name or "").strip()
        if key in self._instances:
            return self._instances[key]
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(f"Service '{key}' not registered. Available: {list(self._factories)}")
        instance = factory()
        self._instances[key] = instance
        return instance

    def has(self, name: str) -> bool:
        """Return True if *name* is registered."""
        return str(name or "").strip() in self._factories

    def reset(self) -> None:
        """Clear all cached instances (for testing)."""
        self._instances.clear()

    def clear(self) -> None:
        """Remove all registrations and cached instances."""
        self._factories.clear()
        self._instances.clear()

    def list_registered(self) -> list[str]:
        """Return sorted list of registered service names."""
        return sorted(self._factories)


__all__ = ["ServiceRegistry"]
