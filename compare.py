#!/usr/bin/env python3
"""Compare the speed of two extractors across all benchmarks.

Usage:
    python compare.py <extractor1> <extractor2> 
        [json_files...]
        [--min-enodes N] [--min-enodes-pct P] [--min-eclasses N] [--min-eclasses-pct P]
        [--sort benchmark|winner|speedup|e-nodes|e-nodes-pct|e-classes|e-classes-pct|<extractor>]

If no json_files are given, all files in output/**/*.json are used.
"""
import argparse
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


def load_egraph(name):
    try:
        with open(name) as f:
            nodes = json.load(f)["nodes"]
        # Pre-build index: eclass -> list of child eclasses (across all member nodes)
        eclass_children = {}
        for n in nodes.values():
            ec = n["eclass"]
            if ec not in eclass_children:
                eclass_children[ec] = []
            for child_id in n["children"]:
                eclass_children[ec].append(nodes[child_id]["eclass"])
        # eclass -> node count, for reachable enodes computation
        eclass_size = {}
        for n in nodes.values():
            ec = n["eclass"]
            eclass_size[ec] = eclass_size.get(ec, 0) + 1
        return eclass_children, eclass_size
    except Exception:
        return None, None


def egraph_stats(eclass_children, eclass_size, roots):
    if eclass_children is None:
        return None, None, None, None
    total_eclasses = len(eclass_children)
    total_enodes = sum(eclass_size.values())

    visited = set(roots)
    queue = list(roots)
    while queue:
        ec = queue.pop()
        for child_ec in eclass_children.get(ec, []):
            if child_ec not in visited:
                visited.add(child_ec)
                queue.append(child_ec)
    reach_eclasses = len(visited)
    reach_enodes = sum(eclass_size.get(ec, 0) for ec in visited)

    return total_enodes, total_eclasses, reach_enodes, reach_eclasses


def compare(entries, e1, e2, min_enodes=None, min_enodes_pct=None, min_eclasses=None, min_eclasses_pct=None, sort_by="benchmark"):
    # Index by (name, roots) -> extractor -> micros
    index = {}
    for entry in entries:
        if entry["extractor"] not in (e1, e2):
            continue
        key = (entry["name"], entry["roots"])
        index.setdefault(key, {})[entry["extractor"]] = entry["micros"]

    roots_per_name = {}
    for name, roots in index:
        roots_per_name.setdefault(name, set()).add(roots)

    nodes_cache = {}

    rows = []
    skipped = 0
    for key, times in index.items():
        if e1 not in times or e2 not in times:
            skipped += 1
            continue
        t1, t2 = times[e1], times[e2]
        # Avoid division by zero; treat 0 micros as 1 for ratio purposes
        t1c, t2c = max(t1, 1), max(t2, 1)
        if t1c < t2c:
            winner = e1
            ratio = t2c / t1c
        elif t2c < t1c:
            winner = e2
            ratio = t1c / t2c
        else:
            winner = None
            ratio = 1.0
        name, roots = key
        if name not in nodes_cache:
            nodes_cache[name] = load_egraph(name)
        eclass_children, eclass_size = nodes_cache[name]
        total_enodes, total_eclasses, reach_enodes, reach_eclasses = egraph_stats(eclass_children, eclass_size, roots)
        if min_enodes is not None and (reach_enodes is None or reach_enodes < min_enodes):
            continue
        if min_enodes_pct is not None and (reach_enodes is None or total_enodes == 0 or 100 * reach_enodes / total_enodes < min_enodes_pct):
            continue
        if min_eclasses is not None and (reach_eclasses is None or reach_eclasses < min_eclasses):
            continue
        if min_eclasses_pct is not None and (reach_eclasses is None or total_eclasses == 0 or 100 * reach_eclasses / total_eclasses < min_eclasses_pct):
            continue
        display_name = name.removeprefix("data/").removesuffix(".json")
        if len(roots_per_name[name]) > 1:
            roots_str = " [" + ", ".join(roots) + "]"
        else:
            roots_str = ""
        label = f"{display_name}{roots_str}"
        rows.append((label, winner, ratio, t1, t2, total_enodes, total_eclasses, reach_enodes, reach_eclasses))

    if skipped:
        print(f"(skipped {skipped} benchmarks missing one extractor)\n", file=sys.stderr)

    if not rows:
        print(f"No benchmarks found for both '{e1}' and '{e2}'.")
        return

    sort_key = {
        "benchmark": lambda r: r[0],
        "winner":    lambda r: (r[1] is None, r[1] or ""),
        "speedup":   lambda r: r[2],
        e1:          lambda r: r[3],
        e2:          lambda r: r[4],
        "e-nodes":       lambda r: (r[7] is None, r[7] or 0),
        "e-nodes-pct":   lambda r: (r[7] is None or not r[5], r[7] / r[5] if r[5] else 0),
        "e-classes":     lambda r: (r[8] is None, r[8] or 0),
        "e-classes-pct": lambda r: (r[8] is None or not r[6], r[8] / r[6] if r[6] else 0),
    }[sort_by]
    rows.sort(key=sort_key)

    e1_wins = sum(1 for r in rows if r[1] == e1)
    e2_wins = sum(1 for r in rows if r[1] == e2)
    ties = sum(1 for r in rows if r[1] is None)

    def fmt_reach(reach, total):
        if total is None or total == 0:
            return "?"
        pct = 100.0 * reach / total
        return f"{reach} ({pct:.1f}%)"

    label_w = max(len(r[0]) for r in rows)
    name_w  = max(len(e1), len(e2), len("tie"))
    en_w    = max(len(fmt_reach(r[7], r[5])) for r in rows)
    ec_w    = max(len(fmt_reach(r[8], r[6])) for r in rows)
    en_w    = max(en_w, len("e-nodes (reach %)"))
    ec_w    = max(ec_w, len("e-classes (reach %)"))

    header = f"{'Benchmark':<{label_w}}  {'Winner':<{name_w}}  {'Speedup':>10}  {e1+' µs':>12}  {e2+' µs':>12}  {'e-nodes (reach %)':>{en_w}}  {'e-classes (reach %)':>{ec_w}}"
    print(header)
    print("-" * len(header))

    for label, winner, ratio, t1, t2, total_en, total_ec, reach_en, reach_ec in rows:
        w  = winner if winner is not None else "tie"
        en = fmt_reach(reach_en, total_en)
        ec = fmt_reach(reach_ec, total_ec)
        print(f"{label:<{label_w}}  {w:<{name_w}}  {ratio:>9.2f}x  {t1:>12}  {t2:>12}  {en:>{en_w}}  {ec:>{ec_w}}")

    total = len(rows)
    def pct(n): return f"{100 * n / total:.1f}%"

    print()
    print(f"Wins: {e1}: {e1_wins} ({pct(e1_wins)}), {e2}: {e2_wins} ({pct(e2_wins)})")
    if ties:
        print(f"Ties: {ties} ({pct(ties)})")
    print(f"Total: {total} benchmarks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("extractor1")
    parser.add_argument("extractor2")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--min-enodes", type=int, metavar="N", help="only include benchmarks with at least N reachable e-nodes")
    parser.add_argument("--min-enodes-pct", type=float, metavar="P", help="only include benchmarks where at least P%% of e-nodes are reachable")
    parser.add_argument("--min-eclasses", type=int, metavar="N", help="only include benchmarks with at least N reachable e-classes")
    parser.add_argument("--min-eclasses-pct", type=float, metavar="P", help="only include benchmarks where at least P%% of e-classes are reachable")
    parser.add_argument("--sort", metavar="COL", default="benchmark",
                        help="sort by: benchmark, winner, speedup, e-nodes, e-nodes-pct, e-classes, e-classes-pct, or an extractor name (default: benchmark)")
    args = parser.parse_args()

    valid_sort = {"benchmark", "winner", "speedup", "e-nodes", "e-nodes-pct", "e-classes", "e-classes-pct", args.extractor1, args.extractor2}
    if args.sort not in valid_sort:
        parser.error(f"--sort must be one of: {', '.join(sorted(valid_sort))}")

    files = args.files or glob.glob("output/**/*.json", recursive=True)
    entries = load_jsons(files)
    if not entries:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    compare(entries, args.extractor1, args.extractor2, args.min_enodes, args.min_enodes_pct, args.min_eclasses, args.min_eclasses_pct, args.sort)
