"""Command-line entry point for the Erdos-FAI.

Usage::

    erdos-fai version
    erdos-fai info
    erdos-fai demo            # run the bundled healthcare-triage pipeline
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"erdos-fai {__version__}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    from .runtime.llm import default_provider

    provider = default_provider()
    print("Erdos-FAI — Enterprise Agentic Technology")
    print(f"  version           {__version__}")
    print(f"  active provider   {provider.name}")
    print("  layers            Intelligence · Orchestration · Safety · Learning · Insights")
    if provider.name == "echo":
        print("\n  (No ANTHROPIC_API_KEY found — using the offline EchoProvider.)")
        print("  Set ANTHROPIC_API_KEY and install 'erdos-fai[anthropic]' for live Claude calls.")
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from .examples_demo import run_demo

    return run_demo()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erdos-fai", description="Erdos-FAI CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="print the version").set_defaults(func=_cmd_version)
    sub.add_parser("info", help="show runtime info").set_defaults(func=_cmd_info)
    sub.add_parser("demo", help="run the bundled healthcare-triage demo").set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
