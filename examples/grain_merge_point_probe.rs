//! Where the grain VM's merge-point marker is lost.
//!
//! `grain_jitcode_census`'s `[B3]` probe reports `jit_merge_point` absent from
//! both the insn table and every assembled body, while one jitcode does carry
//! a `jitdriver_sd`. So the portal is recognised as the portal and its marker
//! still never reaches the bytecode. Two stages can drop it and the census
//! cannot tell them apart:
//!
//!   * `front::mir` never builds a `CallTarget::Method{ receiver_root:
//!     "GrainJitDriver", name: "jit_merge_point" }` in the first place — the
//!     `impl_method_owner` hint declines, or the call is not in the lowered
//!     graph at all;
//!   * `jtransform::try_handle_jit_marker` sees the target and returns `None`
//!     (no `portal_jd_index`, no `CallControl`, too few args) or `Some(vec![])`
//!     (`!jd.active`), both of which erase the marker silently.
//!
//! This lowers the portal function alone — no pipeline, no codewriter — so a
//! marker printed here puts the loss downstream of the front end, and a marker
//! missing here puts it in the front end.
//!
//! usage: grain_merge_point_probe <rhai.ullbc> [fn-name]
use majit_translate::model::{CallTarget, OpKind};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        !args.is_empty(),
        "usage: grain_merge_point_probe <ullbc> [fn-name]"
    );
    let llbc_path = &args[0];
    let portal = args.get(1).map(String::as_str).unwrap_or("run_frame");

    let t0 = std::time::Instant::now();
    let llbc = majit_charon_reader::Llbc::load(llbc_path).expect("LLBC parses");
    println!("loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let graph = majit_translate::front::mir::lower_function_with_static_addrs(
        &llbc,
        portal,
        majit_translate::HostStaticAddrs::default(),
    )
    .expect("portal lowers");

    let mut calls = 0usize;
    let mut methods = 0usize;
    let mut hits: Vec<String> = Vec::new();
    let mut method_roots: std::collections::BTreeMap<String, usize> = Default::default();
    for (b, block) in graph.blocks.iter().enumerate() {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            calls += 1;
            if let CallTarget::Method {
                name,
                receiver_root,
                ..
            } = target
            {
                methods += 1;
                *method_roots
                    .entry(receiver_root.clone().unwrap_or_else(|| "<none>".into()))
                    .or_default() += 1;
                if name.contains("merge_point") || name.contains("can_enter") {
                    hits.push(format!(
                        "bb{b}: Method {{ name: {name:?}, receiver_root: {receiver_root:?} }} \
                         args={}",
                        args.len()
                    ));
                }
            }
            let rendered = format!("{target:?}");
            if rendered.contains("merge_point") || rendered.contains("GrainJitDriver") {
                hits.push(format!("bb{b}: {rendered} args={}", args.len()));
            }
        }
    }
    println!(
        "portal {portal:?}: blocks={} calls={calls} method_calls={methods}",
        graph.blocks.len()
    );
    println!("merge-point-shaped targets: {}", hits.len());
    for h in &hits {
        println!("  {h}");
    }
    println!("-- method receiver roots --");
    for (root, n) in &method_roots {
        println!("  {n:4}  {root}");
    }
}
