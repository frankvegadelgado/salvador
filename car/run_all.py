"""Run the car/ experiments one after another, quickly or in full.

Quick mode (``--quick``) runs every script on small instances, so the whole
suite takes a small fraction of the full time and checks that everything
works end to end. Full
mode (the default) runs every script with its own defaults, as in the paper.
Each script still writes its own .json/.csv files next to itself; this runner
also writes ``car/run_all_log.txt`` with the command, exit status and time of
every script.

Examples, from the repository root:

    python car/run_all.py --quick                          # all scripts, small instances
    python car/run_all.py --quick --only strategies scaling
    python car/run_all.py --list                           # show the scripts and their quick options
    python car/run_all.py                                  # full runs (hours)
    python car/run_all.py --skip triangle_free mechanism   # full runs except these two
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: name -> (script, options used with --quick). Order: fastest first.
SCRIPTS: dict[str, tuple[str, list[str]]] = {
    "ratio":          ("car_ratio.py",          ["--quick", "--max-n", "5000", "--timeout", "120"]),
    "strategies":     ("car_strategies.py",     ["--quick", "--max-n", "5000"]),
    "c11_variants":   ("car_c11_variants.py",   ["--quick", "--max-n", "5000"]),
    "c10_bags":       ("car_c10_bags.py",       ["--quick"]),
    "c0_components":  ("car_c0_components.py",  ["--quick", "--max-n", "5000"]),
    "c0_fix":         ("car_c0_fix.py",         ["--quick"]),
    "regular":        ("car_regular.py",        ["--quick"]),
    "bipartite":      ("car_bipartite.py",      ["--quick", "--max-n", "1000"]),
    "yes_instances":  ("car_yes_instances.py",  ["--quick", "--max-n", "1280", "--exact-time", "20"]),
    "mechanism":      ("car_mechanism.py",      ["--quick", "--exact-time", "20"]),
    "triangle_free":  ("car_triangle_free.py",  ["--quick", "--max-n", "2000", "--exact-time", "20",
                                                 "--timeout", "120"]),
    "scaling":        ("car_scaling.py",        ["--quick", "--max-n", "8000"]),
    "ratio_timing":   ("car_ratio_timing.py",   ["--quick"]),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small instances for every script (much shorter)")
    ap.add_argument("--only", nargs="+", choices=list(SCRIPTS), help="run only these experiments")
    ap.add_argument("--skip", nargs="+", choices=list(SCRIPTS), default=[], help="do not run these experiments")
    ap.add_argument("--list", action="store_true", help="list the experiments and their quick options, then exit")
    ap.add_argument("--stop-on-error", action="store_true", help="stop at the first script that fails")
    args = ap.parse_args()

    if args.list:
        for name, (script, quick) in SCRIPTS.items():
            print(f"{name:15} {script:24} quick: {' '.join(quick)}")
        return

    names = [n for n in (args.only or list(SCRIPTS)) if n not in args.skip]
    log_lines = []
    started = time.time()
    failures = []
    for name in names:
        script, quick = SCRIPTS[name]
        cmd = [sys.executable, str(HERE / script)] + (quick if args.quick else [])
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        code = subprocess.call(cmd)
        dt = time.time() - t0
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"=== {name}: {status} in {dt:.1f} s", flush=True)
        log_lines.append(f"{name:15} {status:18} {dt:10.1f} s   {' '.join(cmd[1:])}")
        if code != 0:
            failures.append(name)
            if args.stop_on_error:
                break
    total = time.time() - started
    log_lines.append(f"{'total':15} {'':18} {total:10.1f} s   mode: {'quick' if args.quick else 'full'}")
    (HERE / "run_all_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(log_lines))
    if failures:
        print(f"\nFailed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
