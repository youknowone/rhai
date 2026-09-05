use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
};

/// Lowering the `grain` dispatch loop to jitcode tables, behind the feature
/// that emits the merge point it lowers around.
#[cfg(feature = "grain-jit")]
#[path = "build/majit_prepass.rs"]
mod majit_prepass;

fn main() {
    // Tell Cargo that if the given environment variable changes, to rerun this build script.
    println!("cargo:rerun-if-changed=build.template");
    println!("cargo:rerun-if-env-changed=RHAI_AHASH_SEED");
    println!("cargo:rerun-if-env-changed=RHAI_HASHING_SEED");
    let mut contents = String::new();

    File::open("build.template")
        .expect("cannot open `build.template`")
        .read_to_string(&mut contents)
        .expect("cannot read from `build.template`");

    let seed = env::var("RHAI_HASHING_SEED")
        .or_else(|_| env::var("RHAI_AHASH_SEED"))
        .map_or_else(|_| "None".into(), |s| format!("Some({s})"));

    contents = contents.replace("{{HASHING_SEED}}", &seed);

    // Charon fingerprints the source closure across extraction. Rewriting an
    // unchanged generated source makes that window look dirty even when the
    // hashing configuration has not changed (and concurrent feature builds
    // needlessly invalidate one another's inputs).
    if !fs::read_to_string("src/config/hashing_env.rs").is_ok_and(|old| old == contents) {
        File::create("src/config/hashing_env.rs")
            .expect("cannot create `hashing_env.rs`")
            .write_all(contents.as_bytes())
            .expect("cannot write to `config/hashing_env.rs`");
    }

    #[cfg(feature = "grain-jit")]
    majit_prepass::main();
}
