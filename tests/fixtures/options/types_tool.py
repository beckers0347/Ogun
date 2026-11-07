"""Correct key-path/type lister; BUG=boolnum reports booleans as number, BUG=root prints the root entry, BUG=dupes prints dict keys as object too."""
import argparse
import json
import os
import sys


def kind(v):
    if os.environ.get("BUG") != "boolnum" and isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if v is None:
        return "null"
    return "array" if isinstance(v, list) else "object"


def walk(v, path, out, depth, limit):
    if path or os.environ.get("BUG") == "root":
        out[path] = kind(v)
    if limit is not None and depth >= limit:
        return
    items = v.items() if isinstance(v, dict) else enumerate(v) if isinstance(v, list) else []
    for k, child in items:
        child_path = f"{path}.{k}" if path else str(k)
        if os.environ.get("BUG") == "dupes" and isinstance(v, dict):
            out.setdefault("__dupes__", []).append(child_path)
        walk(child, child_path, out, depth + 1, limit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.add_argument("--depth", type=int)
    a = p.parse_args()
    try:
        with open(a.file, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    out = {}
    walk(data, "", out, 0, a.depth)
    if a.json:
        out.pop("__dupes__", None)
        print(json.dumps(dict(sorted(out.items())), indent=2))
    else:
        for k in out.pop("__dupes__", []):
            print(f"{k} object")
        for k in sorted(out):
            print(f"{k} {out[k]}")


if __name__ == "__main__":
    main()
