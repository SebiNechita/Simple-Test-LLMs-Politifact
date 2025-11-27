#!/usr/bin/env python3
"""Fix politifact_factcheck_data.json by converting newline-delimited JSON
objects (NDJSON) into a single valid JSON array.

Usage examples:
  # write fixed file next to source
  python fix_politifact_json.py --input politifact_factcheck_data.json

  # overwrite original (makes a .bak copy automatically)
  python fix_politifact_json.py --input politifact_factcheck_data.json --inplace

  # specify explicit output
  python fix_politifact_json.py -i politifact_factcheck_data.json -o politifact_factcheck_data_fixed.json
"""
import argparse
import json
import os
import sys


def try_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_ndjson(path):
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError as e:
                # Provide helpful error context
                raise ValueError(f"Failed to parse JSON on line {i}: {e}\n{line[:200]}") from e
    return objs


def write_json_pretty(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description="Fix NDJSON -> JSON array for Politifact file")
    p.add_argument("-i", "--input", required=True, help="Input file path")
    p.add_argument("-o", "--output", help="Output file path (default: input_fixed.json)")
    p.add_argument("--inplace", action="store_true", help="Overwrite input (will create a .bak)")
    args = p.parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"ERROR: input file not found: {inp}")
        sys.exit(2)

    # try to load as proper JSON first
    loaded = try_load_json(inp)
    if loaded is not None:
        # already valid JSON
        if args.inplace:
            backup = inp + ".bak"
            os.replace(inp, backup)
            write_json_pretty(inp, loaded)
            print(f"Input was valid JSON. Backed up original to {backup} and wrote pretty JSON to {inp}.")
            sys.exit(0)
        else:
            out = args.output or os.path.splitext(inp)[0] + "_fixed.json"
            write_json_pretty(out, loaded)
            print(f"Input was valid JSON. Wrote pretty JSON to {out}.")
            sys.exit(0)

    # fallback: try NDJSON parsing
    try:
        objs = parse_ndjson(inp)
    except ValueError as e:
        print(f"ERROR: could not parse file as NDJSON: {e}")
        sys.exit(3)

    if not objs:
        print("No JSON objects found in input file.")
        sys.exit(4)

    # prepare output path
    if args.inplace:
        backup = inp + ".bak"
        os.replace(inp, backup)
        out_path = inp
        print(f"Made backup: {backup}")
    else:
        out_path = args.output or os.path.splitext(inp)[0] + "_fixed.json"

    write_json_pretty(out_path, objs)

    print(f"Wrote {len(objs)} JSON objects to {out_path}")


if __name__ == "__main__":
    main()
