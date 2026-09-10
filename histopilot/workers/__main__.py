"""Reserved worker process entrypoint; intentionally does not execute ML yet."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="HistoPilot isolated worker boundary")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("manifest", help="Validated job manifest path, supplied by the supervisor")
    parser.parse_args()
    print("Worker execution is not implemented; no artifacts were created.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
