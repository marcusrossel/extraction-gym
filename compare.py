#!/usr/bin/env python3
"""Compare the speed of two extractors across all benchmarks.

Usage:
    python compare.py <extractor1> <extractor2> [json_files...]

If no json_files are given, all files in output/**/*.json are used.
"""
import glob
import json
import sys


def load_jsons(files):
    entries = []
    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
            for idx, result in enumerate(data["results"]):
                entries.append({
                    "name": data["name"],
                    "extractor": data["extractor"],
                    "roots": tuple(result["roots"]),
                    "result_index": idx,
                    "micros": result["micros"],
                })
        except Exception as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
    return entries


def egraph_stats(name):
    try:
        with open(name) as f:
            data = json.load(f)
        nodes = data["nodes"]
        enodes = len(nodes)
        eclasses = len({v["eclass"] for v in nodes.values()})
        return enodes, eclasses
    except Exception:
        return None, None


def compare(entries, e1, e2):
    # Index by (name, roots) -> extractor -> micros
    index = {}
    for entry in entries:
        if entry["extractor"] not in (e1, e2):
            continue
        key = (entry["name"], entry["roots"])
        index.setdefault(key, {})[entry["extractor"]] = entry["micros"]

    stats_cache = {}

    rows = []
    skipped = 0
    for key, times in index.items():
        if e1 not in times or e2 not in times:
            skipped += 1
            continue
        t1, t2 = times[e1], times[e2]
        # Avoid division by zero; treat 0 micros as 1 for ratio purposes
        t1c, t2c = max(t1, 1), max(t2, 1)
        if t1c <= t2c:
            winner = e1
            ratio = t2c / t1c
        else:
            winner = e2
            ratio = t1c / t2c
        name, roots = key
        if name not in stats_cache:
            stats_cache[name] = egraph_stats(name)
        enodes, eclasses = stats_cache[name]
        label = f"{name} {list(roots)}"
        rows.append((label, winner, ratio, t1, t2, enodes, eclasses))

    if skipped:
        print(f"(skipped {skipped} benchmarks missing one extractor)\n", file=sys.stderr)

    if not rows:
        print(f"No benchmarks found for both '{e1}' and '{e2}'.")
        return

    rows.sort(key=lambda r: r[0])

    e1_wins = sum(1 for r in rows if r[1] == e1)
    e2_wins = sum(1 for r in rows if r[1] == e2)

    label_w = max(len(r[0]) for r in rows)
    name_w = max(len(e1), len(e2))

    header = f"{'Benchmark':<{label_w}}  {'Winner':<{name_w}}  {'Faster by':>10}  {e1+' µs':>12}  {e2+' µs':>12}  {'e-nodes':>8}  {'e-classes':>10}"
    print(header)
    print("-" * len(header))

    for label, winner, ratio, t1, t2, enodes, eclasses in rows:
        en = str(enodes) if enodes is not None else "?"
        ec = str(eclasses) if eclasses is not None else "?"
        print(f"{label:<{label_w}}  {winner:<{name_w}}  {ratio:>9.2f}x  {t1:>12}  {t2:>12}  {en:>8}  {ec:>10}")

    print()
    print(f"Wins: {e1}={e1_wins}, {e2}={e2_wins}  ({len(rows)} benchmarks total)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    e1, e2 = args[0], args[1]
    files = args[2:] or glob.glob("output/**/*.json", recursive=True)

    entries = load_jsons(files)
    if not entries:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    compare(entries, e1, e2)
