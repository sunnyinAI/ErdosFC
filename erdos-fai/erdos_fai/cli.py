"""Command-line entry point for the Erdos-FAI.

Usage::

    erdos-fai version
    erdos-fai info
    erdos-fai demo            # run the bundled healthcare-triage pipeline
    erdos-fai execute         # run the Execution Layer demo (connectors → routing)
    erdos-fai connectors      # list the pre-built connector catalog
    erdos-fai templates       # list the workflow template library
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
    from .execution import CONNECTORS, TEMPLATES

    print("Erdos-FAI — Enterprise Agentic Technology")
    print(f"  version           {__version__}")
    print(f"  active provider   {provider.name}")
    print("  layers            Intelligence · Orchestration · Safety · Learning · Insights · Execution")
    print(f"  connectors        {len(CONNECTORS)} pre-built across {len(CONNECTORS.categories())} categories")
    print(f"  templates         {len(TEMPLATES)} workflow templates")
    if provider.name == "echo":
        print("\n  (No ANTHROPIC_API_KEY found — using the offline EchoProvider.)")
        print("  Set ANTHROPIC_API_KEY and install 'erdos-fai[anthropic]' for live Claude calls.")
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from .examples_demo import run_demo

    return run_demo()


def _cmd_execute(_: argparse.Namespace) -> int:
    from .execution_demo import run_demo

    return run_demo()


def _cmd_connectors(_: argparse.Namespace) -> int:
    from .execution import CONNECTORS

    print(f"Erdos-FAI connectors — {len(CONNECTORS)} pre-built\n")
    for category in CONNECTORS.categories():
        items = CONNECTORS.by_category(category)
        print(f"{category} ({len(items)})")
        for c in items:
            flag = " ·write" if c.write else ""
            print(f"  {c.id:<18}{flag:<7} {c.name}")
        print()
    return 0


def _cmd_templates(_: argparse.Namespace) -> int:
    from .execution import TEMPLATES

    print(f"Erdos-FAI workflow templates — {len(TEMPLATES)}\n")
    for t in TEMPLATES:
        print(t.describe())
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erdos-fai", description="Erdos-FAI CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="print the version").set_defaults(func=_cmd_version)
    sub.add_parser("info", help="show runtime info").set_defaults(func=_cmd_info)
    sub.add_parser("demo", help="run the bundled healthcare-triage demo").set_defaults(func=_cmd_demo)
    sub.add_parser("execute", help="run the Execution Layer demo").set_defaults(func=_cmd_execute)
    sub.add_parser("connectors", help="list the connector catalog").set_defaults(func=_cmd_connectors)
    sub.add_parser("templates", help="list the workflow template library").set_defaults(func=_cmd_templates)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
