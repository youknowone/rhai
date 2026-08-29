#!/usr/bin/env python3
"""Compare the grain VM against itself with the JIT merge point live.

Whether the tracer is consulted is a compile-time choice --
`src/grain/vm/mod.rs` calls `GrainJitDriver::jit_merge_point` behind
`#[cfg(feature = "grain-jit")]` -- so one binary cannot measure both sides.
This builds `examples/grain_bench` twice, into separate target directories so
a feature flip does not thrash the shared one, runs each, and pairs the rows
by case name.

What is compared is a ratio of ratios. Each build times the tree-walking
interpreter beside its own VM in the same process, so `walker / vm` mostly
divides the machine out; dividing the plain build's figure by the JIT build's
leaves what consulting the merge point cost, with drift between the two
processes corrected by the walker legs that bracket it.

    tax = (walker_plain / vm_plain) / (walker_jit / vm_jit)

Above 1.00 the JIT build is slower. Today it always is: the tracer opens a
trace, walks a few portal ops, hits an unbound symbolic residual and aborts,
compiling nothing -- so what is gated here is a *ceiling on that tax*, not a
floor under a speedup. The ceilings below are literals beside the cases, not a
recorded file, so moving one is visible in the diff that moves it. When loops
start compiling this file grows a second gate; until then a speedup would be a
surprise, and the harness says so rather than quietly passing.

The LLBC extraction is not optional in practice. `MAJIT_MIR_FRONTEND_LLBC` is a
`rerun-if-env-changed` input of the build script, so building `grain-jit`
without it silently swaps the lowered tables for empty ones, and the run then
measures a driver that declines before it can start. Each binary prints the
table count it actually loaded and this reads that, rather than assuming the
build got what it was given.

A case whose VM samples spread more than 25% is reported as ungradeable rather
than graded -- a ratio taken while the machine is busy is not one.

Correctness is not gated here because it is gated earlier: `grain_bench` runs
each case through the walker and through the VM and panics if the two results
differ, so a build that computes the wrong answer never reaches the table.

Usage:

    ./grain-jit-check.py                 # build both, measure, report
    ./grain-jit-check.py --check         # ... and exit non-zero over a ceiling
    ./grain-jit-check.py --no-build      # reuse the binaries already built
    ./grain-jit-check.py --llbc PATH     # the `--features grain-jit` extraction

`--llbc` is required unless `--no-build` reuses binaries that already have
their tables: a build without it writes every table empty and the binary then
panics before it measures anything.

`--check` exits 1 when a case is over its ceiling or has vanished from the
benchmark, and 2 when nothing breached but some case was too noisy to grade.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# One target directory per feature set. Sharing one makes every invocation
# rebuild the whole crate, because the feature set is part of the fingerprint.
PLAIN_TARGET = "target/bench-nojit"
JIT_TARGET = "target/bench-jit"

BIN = "release/examples/grain_bench"

# The most the JIT build may cost, per case, as a multiple of the plain build.
#
# Still a tax and still no speedup: the tracer opens a trace, records five
# portal ops, refuses an unbound symbolic residual and aborts, and after
# `MAX_TRACE_ABORT_COUNT` aborts the green key is banned. What the door costs
# now is the counter decision and nothing else -- majit stopped building a
# state meta, a driver descriptor and a live-value vector ahead of it -- so a
# case's tax tracks how many back edges it runs per unit of work.
#
# Taken from the run that measured this tree, rounded up by about 20%. A
# sample inflated further than that by a loaded machine is also spread further
# than `SPREAD_LIMIT`, so it is refused rather than read as a breach. `switch,
# 4 arms` carries its twin's ceiling: its own sample was the one contaminated
# in that run, and the two cases differ only in arm count.
#
# These are ceilings on a cost, not floors under a benefit. They came down
# once already, in the commit that earned it, and should come down again when
# the trace stops dying at `__len`. Lowering one otherwise, or raising one to
# make a run pass, defeats the point of having them.
CEILINGS = {
    "tight integer loop": 2.25,
    "float arithmetic": 1.55,
    "script fn calls": 1.65,
    "recursive fibonacci": 1.70,
    "switch, 4 arms": 2.20,
    "switch, 16 arms": 2.20,
    "branch heavy": 1.95,
    "native function calls": 1.55,
    "native callbacks": 1.45,
    "primes": 1.80,
}

# Above this, the sample the number came from was contaminated enough that the
# number is not worth reading. `grain_bench` reports it per case as the gap
# between its fastest and median VM sample.
SPREAD_LIMIT = 0.25

ROW = re.compile(
    r"^(?P<name>.+?)\s+"
    r"(?P<walker>[\d.]+)ms\s+"
    r"(?P<vm>[\d.]+)ms\s+"
    r"(?P<speedup>[\d.]+)x\s+"
    r"(?P<floor>[\d.]+)x\s+"
    r"(?P<spread>[\d.]+)%\s+"
    r"(?P<walker_slow>[\d.]+)ms\s+"
    r"(?P<fragments>\d+)\s*$"
)
JIT_ROW = re.compile(r"^\[jit\] (?P<name>.+?): (?P<fields>.*)$")
# The metainterp names the callee it refused, once per distinct target, on
# stderr and without being asked. It is the reason there is tax and no speedup,
# so it is lifted out of the noise rather than left in the dump.
BLOCKER = re.compile(
    r"residual call target (?P<addr>0x[0-9a-f]+) is symbolic path "
    r"\"?(?P<path>[^\"]*)\"?, not a code address"
)


class Run:
    """One `grain_bench` process: what it was, and what it measured."""

    def __init__(self, build, cases, jit, raw):
        self.build = build
        self.cases = cases
        self.jit = jit
        self.raw = raw
        self.blockers = []
        self.stderr = ""
        self.load = []

    @classmethod
    def parse(cls, text):
        build, cases, jit, load = None, {}, {}, []
        for line in text.splitlines():
            if line.startswith("build: "):
                build = line[len("build: ") :].strip()
                continue
            if line.startswith("load average"):
                load.append(line.split(":", 1)[1].strip())
                continue
            hit = JIT_ROW.match(line)
            if hit:
                fields = {}
                for part in hit.group("fields").split():
                    key, _, value = part.partition("=")
                    fields[key] = value
                jit[hit.group("name")] = fields
                continue
            hit = ROW.match(line)
            if hit:
                cases[hit.group("name").strip()] = {
                    "walker": float(hit.group("walker")),
                    "vm": float(hit.group("vm")),
                    "speedup": float(hit.group("speedup")),
                    "spread": float(hit.group("spread")) / 100.0,
                }
        run = cls(build, cases, jit, text)
        run.load = load
        return run

    @property
    def jitcodes(self):
        for part in (self.build or "").split():
            if part.startswith("jitcodes="):
                return int(part.split("=", 1)[1])
        return None


def build(target, features, env_extra, quiet):
    env = dict(os.environ, CARGO_TARGET_DIR=target, **env_extra)
    cmd = [
        "cargo",
        "build",
        "--release",
        "--features",
        features,
        "--example",
        "grain_bench",
    ]
    print(f"building {features} -> {target}", flush=True)
    done = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=None if not quiet else subprocess.PIPE,
        text=True,
    )
    if done.returncode != 0:
        if quiet and done.stderr:
            sys.stderr.write(done.stderr)
        sys.exit(f"build failed: {features}")


def measure(target, label):
    binary = ROOT / target / BIN
    if not binary.exists():
        sys.exit(f"{binary} does not exist; drop --no-build")
    print(f"running {label}", flush=True)
    done = subprocess.run(
        [str(binary)], cwd=ROOT, capture_output=True, text=True
    )
    # A case whose walker and VM disagree panics rather than reporting, and a
    # case below its own floor writes to stderr without failing. Neither is
    # this harness's gate, but both belong in its output.
    if done.returncode != 0:
        sys.stderr.write(done.stderr)
        sys.exit(f"{label} exited {done.returncode}")
    run = Run.parse(done.stdout)
    run.blockers = sorted(
        {m.group("path") for m in map(BLOCKER.match, done.stderr.splitlines()) if m}
    )
    run.stderr = done.stderr
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit non-zero over a ceiling")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--quiet-build", action="store_true", help="hide cargo output")
    ap.add_argument(
        "--llbc",
        default=os.environ.get("MAJIT_MIR_FRONTEND_LLBC"),
        help="the `--features grain-jit` LLBC extraction the tables are lowered from",
    )
    args = ap.parse_args()

    if not args.no_build:
        if not args.llbc:
            sys.exit(
                "no LLBC extraction named. Pass --llbc PATH (or set "
                "MAJIT_MIR_FRONTEND_LLBC) to a `--features grain-jit` "
                "extraction. Without it the build script writes every table "
                "empty and the binary panics at startup -- `jit_state.rs`: "
                "\"the build lowered no JIT driver\" -- so there is nothing "
                "to measure, not merely nothing to trace."
            )
        if not all(pathlib.Path(p).exists() for p in args.llbc.split(os.pathsep)):
            sys.exit(f"--llbc names {args.llbc}, which does not exist")
        build(PLAIN_TARGET, "grain", {}, args.quiet_build)
        jit_env = {"MAJIT_MIR_FRONTEND_LLBC": args.llbc} if args.llbc else {}
        build(JIT_TARGET, "grain-jit", jit_env, args.quiet_build)

    plain = measure(PLAIN_TARGET, "plain VM")
    jit = measure(JIT_TARGET, "JIT-consulting VM")

    print()
    # A ratio taken on a busy machine is not comparable with one taken on an
    # idle machine, and the two runs are two processes -- so both ends of both.
    print(f"plain: {plain.build}   load {' -> '.join(plain.load)}")
    print(f"jit:   {jit.build}   load {' -> '.join(jit.load)}")
    if plain.build == jit.build:
        sys.exit(
            "both runs report the same build, so this compares a binary with "
            "itself; check that the two target directories were built with "
            "different feature sets"
        )
    tables = jit.jitcodes
    if not tables:
        print(
            "\nNo jitcode tables were lowered into the JIT build. The merge "
            "point is still called, so the tax below is the cost of the call "
            "and of a driver that declines immediately -- not of tracing. Pass "
            "--llbc (or set MAJIT_MIR_FRONTEND_LLBC) to a `--features "
            "grain-jit` extraction to measure the tracer."
        )

    print()
    # Three timings and three ratios: the tree-walking interpreter Rhai ships,
    # the VM without the merge point, and the VM with it. `vm/walk` and
    # `jit/walk` are each run's own in-process comparison; `tax` is what the
    # merge point cost, and is the only column gated.
    print(
        f"{'':<22} {'walker':>9} {'vm':>9} {'vm+jit':>9} {'vm/walk':>8} "
        f"{'jit/walk':>9} {'tax':>7} {'ceil':>7} {'spread':>7}"
    )

    over, unstable, missing = [], [], []
    for name in CEILINGS:
        if name not in plain.cases or name not in jit.cases:
            missing.append(name)
            continue
        p, j = plain.cases[name], jit.cases[name]
        # The walker legs correct for drift between the two processes: they run
        # the same interpreter on the same source, so their ratio is machine,
        # not code.
        tax = p["speedup"] / j["speedup"]
        spread = max(p["spread"], j["spread"])
        ceiling = CEILINGS[name]
        ceil_text = "-" if ceiling is None else f"{ceiling:.2f}x"
        print(
            f"{name:<22} {p['walker']:>7.1f}ms {p['vm']:>7.1f}ms {j['vm']:>7.1f}ms "
            f"{p['speedup']:>7.2f}x {j['speedup']:>8.2f}x {tax:>6.2f}x "
            f"{ceil_text:>7} {spread * 100:>6.0f}%"
        )
        # `grain_bench` reports the worse of the two legs that make up its
        # `speedup`, and this takes the worse of the two runs on top: a walker
        # leg that ran under load is what turns machine noise into a `tax`.
        if spread > SPREAD_LIMIT:
            unstable.append(f"{name} (spread {spread * 100:.0f}%)")
        elif ceiling is not None and tax > ceiling:
            over.append(f"{name}: {tax:.2f}x, ceiling {ceiling:.2f}x")

    if jit.jit:
        print()
        for name, fields in jit.jit.items():
            rendered = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[jit] {name}: {rendered}")
        compiled = sum(int(f.get("compiled", 0)) for f in jit.jit.values())
        print()
        print(
            f"loops compiled across all cases: {compiled}"
            + (
                ""
                if compiled
                else "  -- the tax below is paid for tracing that produced no code"
            )
        )

    if jit.blockers:
        print()
        print(
            f"the tracer refused {len(jit.blockers)} residual call target(s) whose "
            "fnaddr the build left unbound:"
        )
        for path in jit.blockers:
            print(f"  {path}")

    # Whatever else the two runs said. `grain_bench` writes a case below its own
    # walker-versus-VM floor here. Those floors were set for the plain build, so
    # while the merge point costs anything at all the JIT run trips every one of
    # them -- that is the `tax` column restated, not a second finding. The plain
    # run's are the ones worth reading.
    for label, run in (("plain", plain), ("jit", jit)):
        rest = "\n".join(
            line
            for line in run.stderr.splitlines()
            if line.strip() and not BLOCKER.match(line)
        )
        if rest:
            print(f"\n--- {label} stderr ---\n{rest}", file=sys.stderr)

    if missing:
        print(f"\nnot measured: {', '.join(missing)}", file=sys.stderr)
    if unstable:
        print(
            "\n%d case(s) too noisy to grade -- rerun on a quiet machine:\n  %s"
            % (len(unstable), "\n  ".join(unstable)),
            file=sys.stderr,
        )
    if over:
        print(
            "\n%d case(s) over their ceiling:\n  %s" % (len(over), "\n  ".join(over)),
            file=sys.stderr,
        )
        print(
            "\nIf the cost is real, find it. Moving a ceiling belongs in the "
            "same commit as whatever made it necessary, with the reason on "
            "the line.",
            file=sys.stderr,
        )

    if args.check:
        if not tables:
            sys.exit(
                "--check needs the lowered tables; without them the run "
                "measures a different program"
            )
        # A case over its ceiling is a result; a case nobody could grade is the
        # absence of one, and a machine somebody else is loading should not read
        # as a regression. `missing` is a failure of its own kind: a case in the
        # table above that the benchmark no longer runs has lost its gate
        # without anything saying so.
        if over or missing:
            sys.exit(1)
        if unstable:
            sys.exit(2)
        sys.exit(0)


if __name__ == "__main__":
    main()
