#!/usr/bin/env python3

"""Run a command under reproducible Linux/AMD benchmode isolation.

Uses a single `sudo` wrapper to disable SMT and boost, switch `amd_pstate` to passive,
lock CPU frequency, confine system/user slices to the housekeeping CPU, run the command
as your normal user inside `bench.slice`, and restore all system settings in `finally`.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

CPU_DIR = Path("/sys/devices/system/cpu")
PSTATE_STATUS = CPU_DIR / "amd_pstate" / "status"
SLICES = ("system.slice", "user.slice", "init.scope")
HOUSEKEEPING_CPU = 0
FREQ_MHZ: int | None = None  # None = auto-detect from ACPI nominal_freq


def _which(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


def _write_sysfs(path: Path, value: str) -> None:
    try:
        path.write_text(f"{value}\n")
    except OSError as exc:
        print(f"benchmode: could not write '{value}' to {path}: {exc}", file=sys.stderr)


def _cpu_dirs() -> list[tuple[int, Path]]:
    return sorted((int(p.name[3:]), p) for p in CPU_DIR.glob("cpu[0-9]*") if p.name[3:].isdigit())


def _cpufreq_files() -> Iterator[Path]:
    for _, cpu_dir in _cpu_dirs():
        cpufreq = cpu_dir / "cpufreq"
        if not cpufreq.is_dir():
            continue
        # Restore order matters: governor before EPP, max before min.
        yield cpufreq / "scaling_governor"
        if (epp := cpufreq / "energy_performance_preference").exists():
            yield epp
        yield cpufreq / "scaling_max_freq"
        yield cpufreq / "scaling_min_freq"


def _online_cpus() -> list[int]:
    return [
        cpu_id
        for cpu_id, cpu_dir in _cpu_dirs()
        if not (online := cpu_dir / "online").exists() or online.read_text().strip() == "1"
    ]


def _systemctl_show_allowed_cpus(slice_name: str) -> str:
    systemctl = _which("systemctl", "/run/current-system/sw/bin/systemctl")
    proc = subprocess.run(
        [systemctl, "show", "-P", "AllowedCPUs", slice_name],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _systemctl_set_allowed_cpus(slice_name: str, cpus: str) -> None:
    systemctl = _which("systemctl", "/run/current-system/sw/bin/systemctl")
    subprocess.run(
        [systemctl, "set-property", "--runtime", slice_name, f"AllowedCPUs={cpus}"], check=True
    )


@contextmanager
def isolated_cpu() -> Generator[tuple[int, int]]:
    if FREQ_MHZ is None:
        nominal = CPU_DIR / "cpu0" / "acpi_cppc" / "nominal_freq"
        if not nominal.exists():
            sys.exit(f"benchmode: {nominal} not found; set FREQ_MHZ in {__file__}")
        freq_mhz = int(nominal.read_text().strip())
    else:
        freq_mhz = FREQ_MHZ
    freq_khz = freq_mhz * 1000

    # Save everything before changing anything, while all CPUs are still online.
    # Restore order: driver mode first (resets cpufreq policies), then SMT/boost, then per-CPU.
    saved_sysfs = [
        (p, p.read_text().strip())
        for p in [
            PSTATE_STATUS,
            CPU_DIR / "smt" / "control",
            CPU_DIR / "cpufreq" / "boost",
            *_cpufreq_files(),
        ]
        if p.exists()
    ]
    saved_slices = [(s, _systemctl_show_allowed_cpus(s)) for s in SLICES]

    try:
        # In active (EPP) mode firmware picks the clock and ignores boost=0 / scaling_max_freq.
        if PSTATE_STATUS.exists() and PSTATE_STATUS.read_text().strip() != "passive":
            _write_sysfs(PSTATE_STATUS, "passive")
        _write_sysfs(CPU_DIR / "smt" / "control", "off")
        _write_sysfs(CPU_DIR / "cpufreq" / "boost", "0")

        for cpu_id in _online_cpus():
            p = CPU_DIR / f"cpu{cpu_id}" / "cpufreq"
            _write_sysfs(p / "scaling_governor", "performance")
            if freq_khz < int((p / "scaling_min_freq").read_text().strip()):
                _write_sysfs(p / "scaling_min_freq", str(freq_khz))
            _write_sysfs(p / "scaling_max_freq", str(freq_khz))
            actual_max = int((p / "scaling_max_freq").read_text().strip())
            _write_sysfs(p / "scaling_min_freq", str(min(freq_khz, actual_max)))

        bench_cpus = [c for c in _online_cpus() if c != HOUSEKEEPING_CPU]
        if not bench_cpus:
            sys.exit(f"benchmode: no benchmark CPUs left after excluding CPU {HOUSEKEEPING_CPU}")

        for s in SLICES:
            _systemctl_set_allowed_cpus(s, str(HOUSEKEEPING_CPU))

        yield freq_mhz, bench_cpus[0]
    finally:
        prev_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        prev_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for s, cpus in saved_slices:
                _systemctl_set_allowed_cpus(s, cpus)
            for path, val in saved_sysfs:
                _write_sysfs(path, val)
            print("benchmode off", file=sys.stderr)
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)


def _exec_with_env(env_file: Path, cpu_str: str, cmd: Sequence[str]) -> NoReturn:
    user_env: dict[str, str] = json.loads(env_file.read_text())
    user_env["BENCHMODE"] = "1"
    user_env["BENCHMODE_CPUS"] = cpu_str
    try:
        os.execvpe(cmd[0], list(cmd), user_env)
    except OSError as exc:
        sys.exit(f"benchmode: failed to exec {cmd[0]}: {exc}")


def _run_wrapped(env_file: Path, cmd: Sequence[str]) -> int:
    if os.geteuid() != 0:
        print("error: internal benchmode wrapper requires root", file=sys.stderr)
        return 1
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        print("error: SUDO_UID/SUDO_GID not set", file=sys.stderr)
        return 1

    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    with isolated_cpu() as (freq_mhz, bench_cpu):
        cpu_str = str(bench_cpu)
        print(
            f"benchmode on: {freq_mhz} MHz, housekeeping CPU {HOUSEKEEPING_CPU}, "
            f"bench CPU {cpu_str}",
            file=sys.stderr,
        )

        io_flag = "--pty" if (sys.stdin.isatty() and sys.stdout.isatty()) else "--pipe"
        systemd_run = _which("systemd-run", "/run/current-system/sw/bin/systemd-run")
        run_cmd = [
            systemd_run,
            "--quiet",
            io_flag,
            "--wait",
            "--collect",
            "--same-dir",
            "--slice=bench.slice",
            f"--uid={uid}",
            f"--gid={gid}",
            "-p",
            f"AllowedCPUs={cpu_str}",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "--_exec",
            str(env_file),
            cpu_str,
            *cmd,
        ]
        # Prevent systemd-run --pty from emitting OSC 0 terminal title sequences.
        systemd_env = {**os.environ, "SYSTEMD_ADJUST_TERMINAL_TITLE": "0"}
        try:
            return subprocess.run(run_cmd, env=systemd_env, check=False).returncode
        except KeyboardInterrupt:
            return 130


def _elevate(cmd: Sequence[str]) -> int:
    if os.geteuid() == 0:
        print(
            "error: do not invoke benchmode with sudo directly; "
            "it calls sudo itself so child processes stay user-owned.",
            file=sys.stderr,
        )
        return 2

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="bench-env-", suffix=".json", delete=False
    ) as tmp:
        json.dump(dict(os.environ), tmp)
        env_file = Path(tmp.name)

    try:
        wrap_cmd = [
            _which("sudo", "/run/wrappers/bin/sudo"),
            sys.executable,
            str(Path(__file__).resolve()),
            "--_wrap",
            str(env_file),
            *cmd,
        ]
        return subprocess.run(wrap_cmd, check=False).returncode
    finally:
        env_file.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--_exec"]:
        if len(args) < 4:
            sys.exit("error: internal --_exec requires env file, CPU string, and command")
        _exec_with_env(Path(args[1]), args[2], args[3:])
    if args[:1] == ["--_wrap"]:
        if len(args) < 3:
            sys.exit("error: internal --_wrap requires env file and command")
        sys.exit(_run_wrapped(Path(args[1]), args[2:]))
    if args in (["-h"], ["--help"]):
        print(f"usage: benchmode.py [--] <command> [args...]\n\n{(__doc__ or '').strip()}")
        return
    cmd = args[1:] if args[:1] == ["--"] else args
    if not cmd:
        sys.exit("usage: benchmode.py [--] <command> [args...]")
    sys.exit(_elevate(cmd))


if __name__ == "__main__":
    main()
