#!/usr/bin/env python3
"""
Inspect UHOP autotune and profiling data.

Usage:
  python3 examples/autotune_inspect.py [--backend opencl] [--op matmul] [--kernel matmul_naive] [--device "Tesla T4"] [--filter KEYWORD]

If no filters are provided, prints all entries.
"""
import argparse
import json
from pathlib import Path

from uhop.cache import UhopAutotune


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=None)
    p.add_argument("--op", default=None)
    p.add_argument("--kernel", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--filter", default=None, help="substring filter on key or values")
    args = p.parse_args()

    auto = UhopAutotune()
    data = auto._read()  # internal but fine for a small CLI

    def match(key: str, val: dict) -> bool:
        if args.backend and not key.startswith(args.backend + "|"):
            return False
        if args.op and ("|" + args.op + "|") not in key:
            return False
        if args.kernel and ("|" + args.kernel + "|") not in key:
            return False
        if args.device and ("|" + args.device + "|") not in key:
            return False
        if args.filter:
            s = key + " " + json.dumps(val)
            if args.filter.lower() not in s.lower():
                return False
        return True

    printed = 0
    for k, v in data.items():
        if k == "_meta":
            continue
        if not isinstance(v, dict):
            continue
        if match(k, v):
            print(k, v)
            printed += 1
    if printed == 0:
        print("No entries matched your filters.")


if __name__ == "__main__":
    main()
