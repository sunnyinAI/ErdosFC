# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""The ``erdos-fl`` command-line interface.

Subcommands:

* ``erdos-fl simulate <job.py> [--threads N]`` — run a :class:`~erdos.job.Job`
  defined in a Python file (as a module-level ``job`` or a ``build_job()``
  factory) in the local simulator.
* ``erdos-fl version`` — print the installed version.

v0.6 adds ``erdos-fl provision`` and ``erdos-fl poc`` for networked, multi-process runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .job import Job

_JOB_FACTORIES = ("build_job", "get_job", "make_job")


def _load_job(path: str) -> Job:
    file = Path(path).resolve()
    if not file.exists():
        raise SystemExit(f"Job file not found: {file}")
    spec = importlib.util.spec_from_file_location("erdos_job_module", file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import job file: {file}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(file.parent))  # let the job import sibling modules
    spec.loader.exec_module(module)

    candidate = getattr(module, "job", None)
    if isinstance(candidate, Job):
        return candidate
    for name in _JOB_FACTORIES:
        factory = getattr(module, name, None)
        if callable(factory):
            result = factory()
            if isinstance(result, Job):
                return result
    raise SystemExit(
        f"No Job found in {file}. Define a module-level `job = Job(...)` "
        f"or a factory `def build_job() -> Job`."
    )


def _cmd_simulate(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
    )
    job = _load_job(args.job)
    print(f"Erdos FC v{__version__} — running job: {job.name}")
    for k, v in job.to_dict().items():
        print(f"  {k:<12}: {v}")
    print()
    job.simulate(threads=args.threads)
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="erdos-fl", description="ErdosFL (Erdos Federated Learning)")
    sub = parser.add_subparsers(dest="command")

    p_sim = sub.add_parser("simulate", help="run a Job in the local simulator")
    p_sim.add_argument("job", help="path to a Python file defining a Job")
    p_sim.add_argument("--threads", type=int, default=1, help="run clients concurrently")
    p_sim.set_defaults(func=_cmd_simulate)

    p_ver = sub.add_parser("version", help="print the installed version")
    p_ver.set_defaults(func=_cmd_version)
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
