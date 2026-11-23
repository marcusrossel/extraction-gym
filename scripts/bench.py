#!/usr/bin/env python3

"""Run extraction-gym benchmarks sequentially under benchmode isolation.

Builds `target/release/extraction-gym` as your normal user (full cores/boost), then
delegates to `scripts/benchmode.py` to run the benchmarks pinned to a single isolated
CPU and optionally plots the results with `scripts/plot.py`.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY_PATH = REPO_ROOT / "target" / "release" / "extraction-gym"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
BENCHMODE_SCRIPT = REPO_ROOT / "scripts" / "benchmode.py"
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot.py"


@dataclass(frozen=True)
class BenchJob:
    data_arg: str
    extractor: str
    out_path: Path


def build_binary(features: str | None) -> None:
    cmd = ["cargo", "build", "--release"]
    if features:
        cmd.append(f"--features={features}")
    print(f"Building: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def list_extractors() -> list[str]:
    proc = subprocess.run(
        [str(BINARY_PATH), "--extractor=print"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def collect_data_files(paths: Sequence[str]) -> list[Path]:
    roots = [Path(p) for p in paths] if paths else [DATA_DIR]
    files: list[Path] = []
    for raw in roots:
        p = Path.cwd() / raw
        if not p.exists():
            p = REPO_ROOT / raw
        if not p.exists():
            raise FileNotFoundError(f"Data path not found: {raw}")
        p = p.resolve()
        files.extend(sorted(p.rglob("*.json")) if p.is_dir() else [p])
    return sorted(set(files))


def output_path_for(data_path: Path, extractor: str, out_dir: Path) -> Path:
    if data_path.is_relative_to(DATA_DIR):
        rel = data_path.relative_to(DATA_DIR)
    elif data_path.is_relative_to(REPO_ROOT):
        rel = data_path.relative_to(REPO_ROOT)
    else:
        rel = Path(data_path.name)
    return out_dir / rel.parent / f"{rel.name}-{extractor}.json"


def data_arg_for(data_path: Path) -> str:
    if data_path.is_relative_to(REPO_ROOT):
        return str(data_path.relative_to(REPO_ROOT))
    return str(data_path)


def total_micros(json_path: Path) -> int:
    data = json.loads(json_path.read_text())
    return sum(int(entry["micros"]) for entry in data["results"])


def run_once(job: BenchJob, out_path: Path, *, single_root: bool) -> None:
    cmd = [str(BINARY_PATH), job.data_arg, f"--extractor={job.extractor}", f"--out={out_path}"]
    if single_root:
        cmd.append("--single-root")
    subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)


def run_job(job: BenchJob, *, single_root: bool, warmup: int, runs: int) -> None:
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=job.out_path.parent, prefix=".bench-") as tmpdir:
        tmp = Path(tmpdir)
        for _ in range(warmup):
            run_once(job, tmp / "warmup.json", single_root=single_root)
        run_paths = [tmp / f"run{r}.json" for r in range(runs)]
        for path in run_paths:
            run_once(job, path, single_root=single_root)
        chosen = run_paths[0] if runs == 1 else sorted(run_paths, key=total_micros)[runs // 2]
        chosen.replace(job.out_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--extractors",
        nargs="+",
        metavar="EXT",
        help="extractors to benchmark (default: all reported by --extractor=print)",
    )
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        metavar="PATH",
        help="data files or directories to benchmark (default: data/)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="output directory for JSON results (default: output/)",
    )
    parser.add_argument(
        "--features", default="", help="comma-separated Cargo features to enable when building"
    )
    parser.add_argument(
        "--single-root", action="store_true", help="pass --single-root to extraction-gym"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="measured runs per benchmark; median-time output is kept (default: 1)",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="unmeasured warmup runs per benchmark (default: 0)"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip `cargo build --release` and use existing binary",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="do not run scripts/plot.py after benchmarking"
    )

    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def run_benchmarks(args: argparse.Namespace) -> None:
    if not args.no_build:
        build_binary(args.features or None)
    if not BINARY_PATH.exists():
        sys.exit(f"error: binary not found at {BINARY_PATH} (run without --no-build)")

    available = list_extractors()
    if args.extractors:
        unknown = [e for e in args.extractors if e not in available]
        if unknown:
            print(
                f"error: unknown extractor(s): {', '.join(unknown)}\n"
                f"available: {', '.join(available)}",
                file=sys.stderr,
            )
            sys.exit(2)
        extractors = list(dict.fromkeys(args.extractors))
    else:
        extractors = available

    data_files = collect_data_files(args.data or [])
    if not data_files:
        sys.exit("error: no data JSON files found.")

    out_dir = Path.cwd() / args.out_dir
    jobs = [
        BenchJob(
            data_arg=data_arg_for(data_path),
            extractor=ext,
            out_path=output_path_for(data_path, ext, out_dir),
        )
        for ext in extractors
        for data_path in data_files
    ]

    cpus = os.environ.get("BENCHMODE_CPUS", "?")
    print(
        f"Running {len(jobs)} benchmarks "
        f"({len(data_files)} files x {len(extractors)} extractors) [CPU {cpus}]",
        file=sys.stderr,
    )

    try:
        for job in tqdm(jobs, unit="bench", dynamic_ncols=True):
            try:
                run_job(job, single_root=args.single_root, warmup=args.warmup, runs=args.runs)
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout or "").strip()
                first_lines = "\n".join(detail.splitlines()[:4])
                msg = first_lines or f"exit code {exc.returncode}"
                tqdm.write(f"FAILED {job.extractor} on {job.data_arg}: {msg}", file=sys.stderr)
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

    if not args.no_plot and len(extractors) >= 2:
        out_files = [str(job.out_path) for job in jobs]
        subprocess.run([sys.executable, str(PLOT_SCRIPT), *out_files], cwd=REPO_ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)

    if os.environ.get("BENCHMODE") != "1":
        if os.geteuid() == 0:
            sys.exit(
                "error: do not invoke scripts/bench.py with sudo directly; "
                "it calls sudo itself so compilation and output files stay user-owned."
            )
        if not args.no_build:
            build_binary(args.features or None)
        inner = [sys.executable, str(Path(__file__).resolve()), "--no-build", *raw_argv]
        proc = subprocess.run([sys.executable, str(BENCHMODE_SCRIPT), "--", *inner], check=False)
        sys.exit(proc.returncode)

    run_benchmarks(args)


if __name__ == "__main__":
    main()
