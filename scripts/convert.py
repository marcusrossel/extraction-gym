#!/usr/bin/env python3

"""a utility to convert the old, csv-like format to the new json format"""

import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path


def convert(f: Iterable[str]) -> str:
    count: Counter[str] = Counter()
    nodes = {}
    j = {"nodes": nodes}
    roots = set()
    root_prefix = "## root: "
    for raw_line in f:
        line = raw_line.strip()
        if line.startswith(root_prefix):
            roots.add(line.removeprefix(root_prefix))
        elif line.startswith("#"):
            j["comment"] = line
        else:
            eclass, cost, op, *children = (x.strip() for x in line.split(","))
            node_id = f"{eclass}__{count[eclass]}"
            nodes[node_id] = {
                "op": op,
                "cost": float(cost),
                "eclass": eclass,
                "children": [f"{x}__0" for x in children],
            }
            count[eclass] += 1

    j["root_eclasses"] = list(roots)
    return json.dumps(j, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    filenames = sys.argv[1:] if argv is None else argv
    for filename in filenames:
        print(f"converting {filename}")
        with Path(filename).open() as f:
            j = convert(f)
        outname = filename.replace(".csv", "") + ".json"
        outname = outname.replace(".json.json", ".json")
        Path(outname).write_text(j)


if __name__ == "__main__":
    main()
