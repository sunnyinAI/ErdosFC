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
    """A simple shared scratchpad threaded through a federated run.

    The server, controller, clients, aggregators and filters all receive the
    same ``FLContext`` instance so they can publish and read run-scoped
    information (current round, current client, configuration, etc.) without
    being tightly coupled to one another.
    """

    # Well-known property keys.
    CURRENT_ROUND = "current_round"
    NUM_ROUNDS = "num_rounds"
    CURRENT_CLIENT = "current_client"

    def __init__(self) -> None:
        self._props: Dict[str, Any] = {}

    def set_prop(self, key: str, value: Any) -> None:
        self._props[key] = value

    def get_prop(self, key: str, default: Any = None) -> Any:
        return self._props.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._props

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FLContext({self._props!r})"
