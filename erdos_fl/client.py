# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A federation client (a participating site).

A ``Client`` pairs a name with an :class:`~erdos.apis.core.Executor` (the local
training logic) and two optional filter chains:

* ``task_filters`` run on the **inbound** task before execution (e.g. decrypt /
  decompress the incoming global model);
* ``result_filters`` run on the **outbound** result after execution (e.g. add
  privacy noise) — this is the last code that touches the update before it
  leaves the site.
"""
from __future__ import annotations

from typing import List, Optional

from .apis.core import Executor, Filter, FLComponent
from .apis.shareable import FLContext, Shareable


class Client(FLComponent):
    """A single participant in the federation."""

    def __init__(
        self,
        name: str,
        executor: Executor,
        result_filters: Optional[List[Filter]] = None,
        task_filters: Optional[List[Filter]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.executor = executor
        self.result_filters: List[Filter] = result_filters or []
        self.task_filters: List[Filter] = task_filters or []

    def run_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        for f in self.task_filters:
            shareable = f.process(shareable, fl_ctx)

        result = self.executor.execute(task_name, shareable, fl_ctx)
        result.set_meta_prop("client", self.name)

        for f in self.result_filters:
            result = f.process(result, fl_ctx)
        return result
