# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""A torch-free filter that drops parameters by name.

``ExcludeVars`` removes parameters whose key matches a glob pattern before an
update leaves the site — useful for withholding batch-norm statistics or any
buffer a site does not wish to share. It is a pure dictionary operation with no
deep-learning dependency, demonstrating that the privacy/message-path story does
not require torch.
"""
from __future__ import annotations

import fnmatch
from typing import Iterable, List, Union

from ..apis.core import Filter
from ..apis.shareable import FLContext, Shareable


class ExcludeVars(Filter):
    """Remove params whose name matches any of the given glob patterns.

    Example::

        ExcludeVars("*.running_mean")          # one pattern
        ExcludeVars(["*.running_*", "*.num_batches_tracked"])
    """

    def __init__(self, patterns: Union[str, Iterable[str]]) -> None:
        super().__init__()
        self.patterns: List[str] = [patterns] if isinstance(patterns, str) else list(patterns)

    def _excluded(self, key: str) -> bool:
        return any(fnmatch.fnmatch(key, p) for p in self.patterns)

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        removed = [k for k in shareable.params if self._excluded(k)]
        if removed:
            shareable.params = {k: v for k, v in shareable.params.items() if k not in removed}
            shareable.set_meta_prop("excluded_vars", removed)
        return shareable
