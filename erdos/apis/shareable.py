# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Data-exchange objects passed between the server and clients.

The :class:`Shareable` is the single unit of communication in Erdos FC. It
carries a model *payload* (``params``) plus free-form *metadata* (``meta``)
such as the number of local training samples or the current round. Keeping a
single, well-defined envelope makes it easy to insert :class:`~erdos.apis.core.Filter`
objects on the path (for example, to add privacy noise) without the rest of
the system needing to know what changed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class Shareable(dict):
    """A lightweight, JSON-/pickle-friendly envelope for federated messages.

    It behaves like a normal ``dict`` (so it is trivial to serialize) but
    exposes two reserved sections through convenience accessors:

    * ``params`` — the model payload, typically a ``state_dict``-style mapping
      of parameter name to tensor/array.
    * ``meta`` — metadata about the payload (sample counts, losses, round id...).
    """

    PARAMS = "params"
    META = "meta"

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self[self.PARAMS] = params if params is not None else {}
        self[self.META] = meta if meta is not None else {}

    # -- payload -----------------------------------------------------------
    @property
    def params(self) -> Dict[str, Any]:
        return self[self.PARAMS]

    @params.setter
    def params(self, value: Dict[str, Any]) -> None:
        self[self.PARAMS] = value

    # -- metadata ----------------------------------------------------------
    @property
    def meta(self) -> Dict[str, Any]:
        return self[self.META]

    @meta.setter
    def meta(self, value: Dict[str, Any]) -> None:
        self[self.META] = value

    def get_meta_prop(self, key: str, default: Any = None) -> Any:
        return self[self.META].get(key, default)

    def set_meta_prop(self, key: str, value: Any) -> "Shareable":
        self[self.META][key] = value
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self.params) if isinstance(self.params, dict) else "?"
        return f"Shareable(params={n} tensors, meta={dict(self.meta)})"


class FLContext:
    """A scoped scratchpad threaded through a federated run.

    Components (server, controller, clients, aggregators, filters) read and
    write run-scoped information here without being tightly coupled. Contexts
    form a parent/child tree: a lookup falls back to the parent chain, but a
    write only ever touches the local scope. The runtime creates one *run*
    scope and branches a fresh *child* scope per client task (via
    :meth:`new_child`), so per-client state such as ``CURRENT_CLIENT`` can never
    leak across sites or races — the prerequisite for concurrent and networked
    execution.
    """

    # Well-known property keys.
    CURRENT_ROUND = "current_round"
    NUM_ROUNDS = "num_rounds"
    CURRENT_CLIENT = "current_client"
    CURRENT_TASK = "current_task"
    SITE_NAME = "site_name"
    ENGINE = "__engine__"  # the run engine / event bus (set by the runtime)

    def __init__(self, parent: Optional["FLContext"] = None) -> None:
        self._props: Dict[str, Any] = {}
        self._parent = parent

    def new_child(self) -> "FLContext":
        """Return a fresh child scope that inherits (reads) from this one."""
        return FLContext(parent=self)

    def set_prop(self, key: str, value: Any) -> None:
        """Write ``key`` in *this* scope (never mutates the parent)."""
        self._props[key] = value

    def get_prop(self, key: str, default: Any = None) -> Any:
        """Read ``key``, falling back to the parent chain."""
        ctx: Optional["FLContext"] = self
        while ctx is not None:
            if key in ctx._props:
                return ctx._props[key]
            ctx = ctx._parent
        return default

    def get_engine(self) -> Any:
        """Convenience accessor for the run engine / event bus, if present."""
        return self.get_prop(self.ENGINE)

    def __contains__(self, key: str) -> bool:
        ctx: Optional["FLContext"] = self
        while ctx is not None:
            if key in ctx._props:
                return True
            ctx = ctx._parent
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        scope = "child" if self._parent is not None else "run"
        return f"FLContext({scope}, props={list(self._props)})"
