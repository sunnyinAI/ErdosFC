"""Runnable healthcare-triage demo.

    cd erdos-fai
    python -m examples.healthcare_triage.run
    # or, after `pip install -e .`:
    erdos-fai demo

Runs offline with the EchoProvider; set ANTHROPIC_API_KEY for live Claude calls.
"""

from erdos_fai.examples_demo import run_demo


def main() -> int:
    return run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
