#!/usr/bin/env python3
"""rhai driver for the Charon ULLBC extraction engine.

Declares the rhai crate table and delegates to the neutral engine in the pyre
checkout beside this one (`../scripts/llbc_extract.py`). Artefacts land under
`<rhai>/build/llbc`, which is what `MAJIT_MIR_FRONTEND_LLBC` names for
`cargo build --features grain-jit`: `build/majit_prepass.rs` lowers the grain
dispatch loop out of that artefact into the jitcode tables the runtime loads.

rhai is an external consumer of the engine, not part of the pyre workspace, so
every path the engine would otherwise infer from a pyre layout is named here:
the engine lives one directory up, and the shared Charon install is found
relative to the pyre checkout rather than to this one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The pyre checkout rhai sits inside: it carries the engine, and its own
# parent carries the shared `.pyre-build/charon/<platform>/charon` the engine
# resolves against `charon_root.parent`. Pointing `charon_root` at rhai would
# look for that install one directory too deep.
PYRE_ROOT = ROOT.parent
sys.path.insert(0, str(PYRE_ROOT / "scripts"))

from llbc_extract import CrateSpec, run_cli  # noqa: E402


# The merge point `build/majit_prepass.rs` lowers around is `#[cfg]`d on
# `grain-jit`, so an extraction under any other feature set produces a dispatch
# loop with no marker in it and the prepass refuses the artefact. The engine
# takes its feature set from `CARGO_FEATURES`, whose engine-side default names
# a pyre backend this crate does not have; the default that belongs to this
# consumer is set here instead. An override still reaches both the cargo pass
# and the `cargo metadata` walk that fingerprints it, so the two cannot drift
# apart -- and one that drops `grain-jit` is refused by the prepass rather than
# silently lowering an empty table.
os.environ.setdefault("CARGO_FEATURES", "grain-jit")

SPECS: dict[str, CrateSpec] = {
    "rhai": CrateSpec(
        name="rhai",
        crate_dir=ROOT,
        output_name="rhai.ullbc",
        # The library alone. `cargo build` would go on to this crate's `bin`
        # targets, and those link against an rlib that does not exist: Charon
        # drives rustc to obtain MIR and emits the `.ullbc` instead of the
        # library, so the next target in the same cargo invocation fails with
        # `extern location for rhai does not exist`. The lowering reads the
        # library's bodies and nothing else, so naming it is both the fix and
        # the whole input.
        cargo_args=["--lib", "--features", "{features}"],
        # No layout sidecar: the jitcode tables are built for the host that
        # runs `grain_bench`, and nothing here is compiled for a cross target.
        layout_targets=(),
    ),
}

DEFAULT_CRATES = ["rhai"]

# Workspace resolution is a compiler input. `Cargo.lock` belongs beside
# `Cargo.toml` here and is carried as an external input instead: this repo
# gitignores it, and the in-root channel resolves pathspecs through `git
# ls-files`, where an ignored file is invisible.
BASE_PATHSPECS = [
    "Cargo.toml",
]

EXTERNAL_INPUTS = (ROOT / "Cargo.lock",)

# Bump only when rhai's extraction behaviour changes in a way not already
# represented by the effective cargo/Charon flags the engine hashes.
EXTRACTION_ABI = "1"


def main() -> None:
    run_cli(
        SPECS,
        DEFAULT_CRATES,
        root=ROOT,
        out_dir=ROOT / "build" / "llbc",
        extraction_abi=EXTRACTION_ABI,
        base_pathspecs=BASE_PATHSPECS,
        charon_root=PYRE_ROOT,
        # The majit crates reach the artefact as path dependencies of this
        # feature set, so the walk that hashes them has to resolve the same
        # features the extraction compiles.
        metadata_feature_crates=("rhai",),
        external_inputs=EXTERNAL_INPUTS,
    )


if __name__ == "__main__":
    main()
