//! De-risk spike for the Pickles `normalize_to_shape` build.
//!
//! THE QUESTION: can the in-circuit batch-STARK verifier's op-list be made
//! INVARIANT to a child-proof shape parameter by PADDING the child to a fixed
//! shape? If yes, a depth-1 fixed-VK (`step ∘ wrap`) is feasible because one
//! static verifier circuit (one VK) can verify any child of the padded shape.
//!
//! We measure the `AggregationCircuitFingerprint` (witness_count,
//! public_flat_len, private_flat_len, ops_len) of the BUILT verifier circuit as
//! a function of the child proof. The fingerprint IS the op-list identity that
//! the fixed VK pins (see `recursion::AggregationCircuitFingerprint` /
//! `AggregationPrepCache`, which the codebase already keys cache hits on).
//!
//! TWO axes are exercised, in two test pairs:
//!
//! 1. Child trace HEIGHT (log-degree) — the `..._height_...` / `..._fixed_shape_...`
//!    pair below. Log-degree drives the FRI opening schedule and the quotient-chunk
//!    count, so it is the proof-shape parameter most likely to move the op-list.
//!
//! 2. The non-primitive MANIFEST (which AIR TYPES are present) — the `..._manifest_...`
//!    / `..._fixed_manifest_...` pair further down. The verifier op-list is
//!    DATA-DRIVEN by `proof.non_primitives` (one `CircuitTablesAir::Dynamic` + one
//!    `air_public_counts` slot per entry), so two proofs whose non-primitive SETS
//!    differ produce different op-lists even at equal height. The fix is
//!    fixed-manifest padding: every proof carries the same ordered type set, an
//!    absent canonical type filled by a MINIMAL SATISFIABLE instance (here a
//!    one-claim `expose_claim`, whose bus read is balanced by the `PublicAir` send).
//!    A generic zero-WIDTH dummy is NOT sufficient — zero-row tables are dropped and
//!    the op-list is per-AIR-TYPE — so the filler must be a real same-type instance.

mod common;

use p3_batch_stark::ProverData;
use std::collections::BTreeSet;

use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    NpoTypeId, generate_expose_claim_trace, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_circuit_prover::batch_stark_prover::{
    BatchStarkProof, expose_claim_air_builders, expose_claim_table_provers, poseidon2_air_builders,
    recompose_air_builders,
};
use p3_circuit_prover::common::{NpoAirBuilder, NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, ExposeClaimPreprocessor,
    Poseidon2Preprocessor, Poseidon2Prover, RecomposePreprocessor, TablePacking, TableProver,
    recompose_table_provers,
};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::logup::LogUpGadget;
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::recursion::{CanonicalShapeSpec, check_canonical_shape, inject_canonical_fillers};
use p3_recursion::Poseidon2Config;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_test_utils::koala_bear_params::*;
use p3_test_utils::test_fri_scalars;

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;
const TRACE_D: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Fingerprint {
    witness_count: u32,
    public_flat_len: usize,
    private_flat_len: usize,
    ops_len: usize,
    /// Number of batch instances (3 primitive + N non-primitive). Reported so we
    /// can see exactly how many non-primitive tables the child carried.
    num_instances: usize,
}

fn compute_fibonacci_classical(n: usize) -> F {
    let mut a = F::ZERO;
    let mut b = F::ONE;
    for _ in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }
    b
}

/// Build a fibonacci child of length `n`, prove the batch with the given
/// `min_trace_height` padding, build the recursive verifier circuit, and return
/// the verifier-circuit fingerprint.
fn child_verifier_fingerprint(n: usize, min_trace_height: usize) -> Fingerprint {
    // ---- build + prove the child batch proof ----
    let mut builder = CircuitBuilder::new();
    let expected_result = builder.alloc_public_input("expected_result");
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);

    let table_packing = TablePacking::new(2, 4).with_min_trace_height(min_trace_height);
    let config_proving = make_test_config();
    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 1>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[compute_fibonacci_classical(n)])
        .unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(config_proving).with_table_packing(table_packing);
    let batch_stark_proof = prover.prove_all_tables(&traces, &circuit_prover_data).unwrap();
    let common = circuit_prover_data.common_data();
    prover.verify_all_tables(&batch_stark_proof).unwrap();

    let num_instances = batch_stark_proof.proof.opened_values.instances.len();

    // ---- build the recursive verifier circuit, read its fingerprint ----
    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let config = make_test_config();
    let lookup_gadget = LogUpGadget::new();

    let mut circuit_builder = CircuitBuilder::new();
    let poseidon2_perm = default_koalabear_poseidon2_16();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        poseidon2_perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    let _ = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D,
    >(
        &config,
        &mut circuit_builder,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &{
            let mut tp: Vec<Box<dyn TableProver<MyConfig>>> = vec![Box::new(Poseidon2Prover::new(
                Poseidon2Config::KOALA_BEAR_D4_W16,
                ConstraintProfile::Standard,
            ))];
            tp.extend(recompose_table_provers::<_, 4>(1, false));
            tp
        },
    )
    .unwrap();

    let verification_circuit = circuit_builder.build().unwrap();
    Fingerprint {
        witness_count: verification_circuit.witness_count,
        public_flat_len: verification_circuit.public_flat_len,
        private_flat_len: verification_circuit.private_flat_len,
        ops_len: verification_circuit.ops.len(),
        num_instances,
    }
}

/// NEGATIVE: two children of DIFFERENT height, padded only to the FRI minimum,
/// land at different log-degrees → the verifier op-list / fingerprint DIFFERS.
/// This is the problem a fixed VK must solve.
#[test]
fn fingerprint_varies_with_child_height_when_unpadded() {
    let fp_small = child_verifier_fingerprint(48, 1);
    let fp_large = child_verifier_fingerprint(400, 1);
    println!("unpadded small (n=48):  {fp_small:?}");
    println!("unpadded large (n=400): {fp_large:?}");
    assert_ne!(
        fp_small, fp_large,
        "expected the verifier op-list to be SENSITIVE to child proof shape \
         when unpadded; if these are equal the heights already coincided — \
         pick more separated n"
    );
}

/// POSITIVE (the make-or-break claim): the SAME two children, both padded to a
/// common fixed minimum trace height, produce an IDENTICAL verifier op-list /
/// fingerprint. One static verifier circuit (one VK) verifies both. This is the
/// `normalize_to_shape` principle: pad to a fixed shape ⇒ invariant op-list.
#[test]
fn fingerprint_invariant_under_fixed_shape_padding() {
    // 512 > both children's natural alu/const/public heights, so every primitive
    // table pads to exactly 512 in both proofs → identical proof shape.
    let pad = 512;
    let fp_small = child_verifier_fingerprint(48, pad);
    let fp_large = child_verifier_fingerprint(400, pad);
    println!("padded small (n=48):  {fp_small:?}");
    println!("padded large (n=400): {fp_large:?}");
    assert_eq!(
        fp_small, fp_large,
        "padding both children to a fixed trace shape MUST make the in-circuit \
         verifier op-list invariant — this is the feasibility gate for the \
         depth-1 fixed VK"
    );
}

// ============================================================================
// PHASE-2: the MANIFEST / COUNT axis.
//
// Phase-1 (above) fixed the trace HEIGHT and proved that height-padding makes
// the verifier fingerprint invariant. It did so over children with the SAME
// non-primitive table SET (fibonacci: zero non-primitive tables on both sides).
//
// THIS phase fixes the other axis: the set of non-primitive AIR TYPES present in
// the proof — the "manifest". The verifier's op-list is DATA-DRIVEN by
// `proof.non_primitives` (see `verifier::batch_stark`, the `for entry in
// &proof.non_primitives` loop that pushes one `CircuitTablesAir::Dynamic` per
// entry and one `air_public_counts` slot per entry). So two proofs whose
// non-primitive SETS differ produce DIFFERENT verifier op-lists — even at the
// same height. A fixed VK must therefore also fix the manifest.
//
// The `normalize_to_shape` mechanism for this axis is FIXED-MANIFEST padding:
// every normalized proof carries the SAME ordered multiset of non-primitive
// types; for any canonical type a given child does NOT naturally use, a MINIMAL
// SATISFIABLE instance of that type is injected at the canonical (lanes,
// log-height). This is the `step ∘ wrap` analog of Pickles padding the absent
// table types.
//
// We exercise it with `expose_claim` as the variable canonical type — the exact
// shape of the prompt's example (a proof with {primitives} vs a proof with
// {primitives, expose_claim}). `expose_claim` is the natural minimal filler: a
// one-claim instance exposes a single genuine PUBLIC witness, so its
// `WitnessChecks` read is balanced by the `PublicAir` send (the bus stays
// balanced — a lookup-carrying type can only be injected as a SELF-BALANCED
// minimal instance, which exposing a real public input satisfies for free).
// ============================================================================

/// Build a fibonacci child of length `n` padded to `min_trace_height`, OPTIONALLY
/// carrying one `expose_claim` table (a single exposed public witness — the
/// minimal satisfiable instance of that canonical type), build the recursive
/// verifier circuit, and return its fingerprint.
fn child_verifier_fingerprint_manifest(
    n: usize,
    min_trace_height: usize,
    expose: bool,
) -> Fingerprint {
    // ---- build the child circuit ----
    let mut builder = CircuitBuilder::new();
    if expose {
        // Enable the expose-claim NPO BEFORE emitting any expose op.
        builder.enable_expose_claim::<F>(generate_expose_claim_trace::<F, F>);
    }
    let expected_result = builder.alloc_public_input("expected_result");
    let mut a = builder.alloc_const(F::ZERO, "F(0)");
    let mut b = builder.alloc_const(F::ONE, "F(1)");
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);
    if expose {
        // The minimal satisfiable instance of the canonical `expose_claim` type:
        // expose ONE genuine public witness. Its bus read is balanced by the
        // PublicAir send for `expected_result`.
        builder.expose_as_public_output(&[expected_result]);
    }

    // ---- prove the child batch proof ----
    let table_packing = TablePacking::new(2, 4).with_min_trace_height(min_trace_height);
    let config_proving = make_test_config();
    let circuit = builder.build().unwrap();

    let preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = if expose {
        vec![Box::new(ExposeClaimPreprocessor)]
    } else {
        Vec::new()
    };
    let air_builders: Vec<Box<dyn NpoAirBuilder<MyConfig, TRACE_D>>> = if expose {
        expose_claim_air_builders::<MyConfig, TRACE_D>()
    } else {
        Vec::new()
    };

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, TRACE_D>(
            &circuit,
            &table_packing,
            &preprocessors,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[compute_fibonacci_classical(n)])
        .unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config_proving).with_table_packing(table_packing);
    if expose {
        prover.register_expose_claim_table::<TRACE_D>();
    }
    let batch_stark_proof = prover.prove_all_tables(&traces, &circuit_prover_data).unwrap();
    let common = circuit_prover_data.common_data();
    prover.verify_all_tables(&batch_stark_proof).unwrap();

    let num_instances = batch_stark_proof.proof.opened_values.instances.len();

    // ---- build the recursive verifier circuit, read its fingerprint ----
    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let config = make_test_config();
    let lookup_gadget = LogUpGadget::new();

    let mut circuit_builder = CircuitBuilder::new();
    let poseidon2_perm = default_koalabear_poseidon2_16();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        poseidon2_perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    // The verifier's non-primitive prover set is UNIFORM (manifest-agnostic): it
    // always carries provers for every canonical type. The verifier matches each
    // `proof.non_primitives` entry to its prover by op_type and ignores the rest,
    // so a child lacking `expose_claim` simply never exercises that prover.
    let _ = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D,
    >(
        &config,
        &mut circuit_builder,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &{
            let mut tp: Vec<Box<dyn TableProver<MyConfig>>> = vec![Box::new(Poseidon2Prover::new(
                Poseidon2Config::KOALA_BEAR_D4_W16,
                ConstraintProfile::Standard,
            ))];
            tp.extend(recompose_table_provers::<_, 4>(1, false));
            tp.extend(expose_claim_table_provers::<MyConfig, TRACE_D>());
            tp
        },
    )
    .unwrap();

    let verification_circuit = circuit_builder.build().unwrap();
    Fingerprint {
        witness_count: verification_circuit.witness_count,
        public_flat_len: verification_circuit.public_flat_len,
        private_flat_len: verification_circuit.private_flat_len,
        ops_len: verification_circuit.ops.len(),
        num_instances,
    }
}

/// NEGATIVE (manifest membership is load-bearing): two children at the SAME fixed
/// height but with DIFFERENT non-primitive table SETS — one bare, one carrying an
/// `expose_claim` table — produce DIFFERENT verifier fingerprints. The op-list is
/// sensitive to the manifest, not just the height; this is the problem the
/// fixed-manifest padding must solve.
#[test]
fn fingerprint_varies_with_manifest_when_unpadded() {
    let pad = 512;
    let fp_bare = child_verifier_fingerprint_manifest(48, pad, false);
    let fp_expose = child_verifier_fingerprint_manifest(48, pad, true);
    println!("bare    (no expose_claim): {fp_bare:?}");
    println!("expose  (+expose_claim):   {fp_expose:?}");
    assert_eq!(fp_bare.num_instances, 3, "bare child carries 3 primitive tables");
    assert_eq!(
        fp_expose.num_instances, 4,
        "expose child carries 3 primitive + 1 expose_claim table"
    );
    assert_ne!(
        fp_bare, fp_expose,
        "a different non-primitive table SET MUST move the verifier op-list — \
         otherwise the manifest axis would be trivially invariant and there would \
         be nothing to normalize"
    );
}

/// POSITIVE (the manifest-axis make-or-break claim): two children with DIFFERENT
/// real circuits (different fibonacci length → different real ALU content) that
/// are BOTH normalized to the SAME canonical manifest — every proof carries the
/// `expose_claim` slot, the absent-in-neither case here being made concrete by
/// the smaller child whose expose table is a MINIMAL injected one-claim instance
/// — and the SAME fixed height produce an IDENTICAL verifier fingerprint. One
/// static verifier circuit (one VK) verifies both. This is the fixed-manifest
/// `normalize_to_shape` principle, the manifest-axis analog of Phase-1's height
/// result.
#[test]
fn fingerprint_invariant_under_fixed_manifest_padding() {
    let pad = 512;
    // Different real circuits (n=48 vs n=400 → different real ALU schedules),
    // each normalized to the canonical manifest {const, public, alu, expose_claim}
    // at the fixed height. The expose_claim instance is the minimal one-claim
    // satisfiable filler in BOTH.
    let fp_small = child_verifier_fingerprint_manifest(48, pad, true);
    let fp_large = child_verifier_fingerprint_manifest(400, pad, true);
    println!("manifest-padded small (n=48):  {fp_small:?}");
    println!("manifest-padded large (n=400): {fp_large:?}");
    assert_eq!(
        fp_small, fp_large,
        "fixing BOTH the height AND the non-primitive manifest MUST make the \
         in-circuit verifier op-list invariant across different real children — \
         this is the manifest-axis feasibility gate for the depth-1 fixed VK"
    );
}

// ============================================================================
// PHASE-3 (experiment): the FULL canonical manifest [poseidon2, recompose,
// expose_claim].
//
// Phase-2 padded only `expose_claim` — the easy canonical type, whose
// `WitnessChecks` read is balanced by a `PublicAir` send. The other two
// canonical types touch the hash + BF↔EF coefficient buses, so a minimal
// injected instance has to be SELF-BALANCED on those buses to land in a child
// that doesn't otherwise use it:
//
// * `poseidon2_perm` — a permutation CTL-RECEIVES its rate inputs and
//   CTL-SENDS its rate outputs (see `add_full_state_sponge_step` in
//   `circuit_builder.rs`). A perm over `Const` rate inputs with `out_ctl` all
//   FALSE sends nothing and only receives the consts (balanced by the Const
//   table's send) — self-balanced, no companion needed.
//
// * `recompose` — produces (SENDS) one EF output witness bound to its D base
//   coefficients. `decompose_ext_to_base_coeffs(x)` emits exactly one recompose
//   row whose output is `connect`-bound straight back to `x`, so the send is
//   consumed by `x`'s existing readers — self-balanced when `x` is a genuine
//   already-present witness (here a public input), no extra companion.
//
// The base-field leaf of Phase-2 (TRACE_D = 1) CANNOT host either type: the
// Poseidon2 register dispatch is D∈{2,4,5} only and recompose at D=1 is the
// identity (const-folded away). The natural host for the canonical manifest is
// therefore a D=4 (`Challenge`) circuit — which is exactly the shape of a
// recursion-LAYER proof (the proof a fixed VK actually re-verifies), where
// poseidon2 + recompose are the native FRI/Fiat-Shamir tables.

const TRACE_D4: usize = 4;

/// The canonical D=4 NPO manifest, pinned at one lane each. `min_trace_height`
/// pins the height axis. Two children built against the SAME pinned packing land
/// on a BYTE-IDENTICAL proof shape regardless of their real table CONTENT.
fn canonical_d4_packing(min_trace_height: usize) -> TablePacking {
    TablePacking::new(2, 4)
        .with_min_trace_height(min_trace_height)
        .with_npo_lanes(NpoTypeId::poseidon2_perm(Poseidon2Config::KOALA_BEAR_D4_W16), 1)
        .with_npo_lanes(NpoTypeId::recompose(), 1)
        .with_npo_lanes(NpoTypeId::expose_claim(), 1)
}

/// Build a D=4 (`Challenge`) child carrying the FULL canonical manifest
/// [poseidon2, recompose, expose_claim], each at exactly one instance.
///
/// `n` drives a fibonacci-shaped primitive core (so two children with different
/// `n` have genuinely DIFFERENT real ALU content). `padded_from_none` selects
/// HOW the three canonical types arrive:
///
/// * `true`  — the child has no natural use of any of them; all three are the
///   MINIMAL self-balanced fillers (the `normalize_to_shape` padding path).
/// * `false` — all three are exercised "for real": a genuine hash whose output
///   is consumed, a genuine ext-decompose (recompose), and a genuine expose.
///
/// Both branches emit exactly one poseidon2 row, one recompose row, and one
/// one-claim expose table, so the only difference is table CONTENT, never the
/// manifest shape.
fn build_d4_full_manifest_child(n: usize, padded_from_none: bool) -> CircuitBuilder<Challenge> {
    let mut builder = CircuitBuilder::<Challenge>::new();
    let perm = default_koalabear_poseidon2_16();
    builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        perm,
    );
    builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    builder.enable_expose_claim::<F>(generate_expose_claim_trace::<F, Challenge>);

    // Fibonacci-shaped real arithmetic core of length `n`.
    let expected = builder.alloc_public_input("expected");
    let mut a = builder.alloc_const(Challenge::ZERO, "F(0)");
    let mut b = builder.alloc_const(Challenge::ONE, "F(1)");
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected);

    if padded_from_none {
        // Minimal self-balanced fillers for ALL three canonical types, via the REAL
        // library injector — `bind = expected` is a genuine non-const public witness
        // the recompose + expose fillers consume. `present` is empty: none of the
        // canonical types are used naturally, so all three are padded.
        let canonical = CanonicalShapeSpec::depth1_default(
            9,
            2,
            4,
            Poseidon2Config::KOALA_BEAR_D4_W16,
            1,
        )
        .canonical_npos;
        inject_canonical_fillers::<Challenge, F>(
            &mut builder,
            Poseidon2Config::KOALA_BEAR_D4_W16,
            &canonical,
            &BTreeSet::new(),
            expected,
        )
        .unwrap();
    } else {
        // Real hash of one rate-chunk; its output is genuinely consumed below.
        let h = builder
            .add_hash_slice(&Poseidon2Config::KOALA_BEAR_D4_W16, &[b], true)
            .unwrap();
        // Real ext-decompose of the GENUINELY non-const hash output → one
        // recompose row. (Decomposing `b` would const-fold: the builder tracks
        // the fibonacci constant value of `b` even though its ALU rows are real,
        // so `decompose(b)` short-circuits and emits no recompose table.)
        let _ = builder.decompose_ext_to_base_coeffs::<F>(h[0]).unwrap();
        // Expose the (genuinely consumed) hash output — one claim.
        builder.expose_as_public_output(&[h[0]]);
    }

    builder
}

/// Prove a D=4 full-manifest child against the canonical pinned packing. Proven
/// with `with_debug_lookups()` (any multiset imbalance panics) and verified.
/// Returns the proof plus its prover data (kept alive for `common_data()`).
fn prove_d4_full_manifest_child(
    n: usize,
    min_trace_height: usize,
    padded_from_none: bool,
) -> (
    BatchStarkProof<MyConfig>,
    CircuitProverData<MyConfig>,
) {
    let builder = build_d4_full_manifest_child(n, padded_from_none);
    let table_packing = canonical_d4_packing(min_trace_height);
    let config_proving = make_test_config();
    let circuit = builder.build().unwrap();

    let preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
        Box::new(ExposeClaimPreprocessor),
    ];
    let mut air_builders: Vec<Box<dyn NpoAirBuilder<MyConfig, TRACE_D4>>> =
        poseidon2_air_builders::<MyConfig, TRACE_D4>();
    air_builders.extend(recompose_air_builders::<MyConfig, TRACE_D4>(1, false));
    air_builders.extend(expose_claim_air_builders::<MyConfig, TRACE_D4>());

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, TRACE_D4>(
            &circuit,
            &table_packing,
            &preprocessors,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[Challenge::from(compute_fibonacci_classical(n))])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config_proving)
        .with_table_packing(table_packing)
        .with_debug_lookups();
    prover.register_poseidon2_table::<TRACE_D4>(Poseidon2Config::KOALA_BEAR_D4_W16);
    prover.register_recompose_table::<TRACE_D4>(false);
    prover.register_expose_claim_table::<TRACE_D4>();

    let batch_stark_proof = prover.prove_all_tables(&traces, &circuit_prover_data).unwrap();
    prover.verify_all_tables(&batch_stark_proof).unwrap();
    (batch_stark_proof, circuit_prover_data)
}

fn d4_manifest(proof: &BatchStarkProof<MyConfig>) -> Vec<String> {
    let mut manifest: Vec<String> = proof
        .non_primitives
        .iter()
        .map(|e| e.op_type.as_str().to_string())
        .collect();
    manifest.sort();
    manifest
}

/// Build the TRACE_D=4 recursive verifier circuit over a D=4 full-manifest child
/// and return its fingerprint. The child-table prover set carries provers for
/// ALL three canonical non-primitive types at D=4.
fn d4_verifier_fingerprint(n: usize, min_trace_height: usize, padded_from_none: bool) -> Fingerprint {
    let (batch_stark_proof, circuit_prover_data) =
        prove_d4_full_manifest_child(n, min_trace_height, padded_from_none);
    let common = circuit_prover_data.common_data();
    let num_instances = batch_stark_proof.proof.opened_values.instances.len();

    let scalars = test_fri_scalars();
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        scalars.log_blowup,
        scalars.log_final_poly_len,
        scalars.commit_pow_bits,
        scalars.query_pow_bits,
        Poseidon2Config::KOALA_BEAR_D4_W16,
    );
    let config = make_test_config();
    let lookup_gadget = LogUpGadget::new();

    let mut circuit_builder = CircuitBuilder::new();
    let poseidon2_perm = default_koalabear_poseidon2_16();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        poseidon2_perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    let _ = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D4,
    >(
        &config,
        &mut circuit_builder,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &{
            let mut tp: Vec<Box<dyn TableProver<MyConfig>>> = vec![Box::new(Poseidon2Prover::new(
                Poseidon2Config::KOALA_BEAR_D4_W16,
                ConstraintProfile::Standard,
            ))];
            tp.extend(recompose_table_provers::<_, 4>(1, false));
            tp.extend(expose_claim_table_provers::<MyConfig, TRACE_D4>());
            tp
        },
    )
    .unwrap();

    let verification_circuit = circuit_builder.build().unwrap();
    Fingerprint {
        witness_count: verification_circuit.witness_count,
        public_flat_len: verification_circuit.public_flat_len,
        private_flat_len: verification_circuit.private_flat_len,
        ops_len: verification_circuit.ops.len(),
        num_instances,
    }
}

/// CORE RESIDUAL: a D=4 child that uses NONE of the canonical non-primitive
/// types, padded to the full manifest [poseidon2, recompose, expose_claim] with
/// MINIMAL self-balanced fillers, proves (with the lookup debugger ON, so any
/// `WitnessChecks`/hash/coeff-bus imbalance panics) AND verifies. We assert the
/// filler tables are GENUINELY PRESENT in `proof.non_primitives` (not silently
/// dropped as zero-row tables) — the manifest is non-vacuous.
#[test]
fn d4_minimal_fillers_are_lookup_balanced() {
    let (proof, _data) = prove_d4_full_manifest_child(48, 64, true);
    let manifest = d4_manifest(&proof);
    println!("padded-from-none D4 child non-primitive manifest: {manifest:?}");
    assert!(
        manifest.iter().any(|m| m.starts_with("poseidon2_perm")),
        "minimal poseidon2 filler must be a real non-primitive table"
    );
    assert!(
        manifest.iter().any(|m| m == "recompose"),
        "minimal recompose filler must be a real non-primitive table"
    );
    assert!(
        manifest.iter().any(|m| m == "expose_claim"),
        "minimal expose_claim filler must be a real non-primitive table"
    );
}

/// The padded-from-none child and the uses-all-three child carry the IDENTICAL
/// non-primitive manifest — the prerequisite for fingerprint invariance.
#[test]
fn d4_full_manifest_matches_across_real_and_padded() {
    let (padded, _d1) = prove_d4_full_manifest_child(48, 64, true);
    let (real, _d2) = prove_d4_full_manifest_child(48, 64, false);
    assert_eq!(
        d4_manifest(&padded),
        d4_manifest(&real),
        "real-use and padded-from-none children must carry the same canonical manifest"
    );
}

/// THE FULL-MANIFEST CLOSURE (manifest-axis, all three canonical types): two D=4
/// children with DIFFERENT real circuits — one that genuinely uses all three
/// canonical non-primitive types, one that uses NONE and is padded to the full
/// manifest with the minimal self-balanced fillers — both normalized to the same
/// canonical D=4 shape produce a BYTE-IDENTICAL recursive-verifier fingerprint.
/// One static VK verifies both. This closes the residual Phase-2 left open
/// (Phase-2 padded only `expose_claim`; this pins the whole
/// [poseidon2, recompose, expose_claim] manifest).
#[test]
fn d4_full_manifest_fingerprint_invariant() {
    // 512 pins the height axis: both children's primitive tables pad to exactly
    // 512 (n=48's natural ~48 and n=400's natural ~400 both land on 512), so the
    // only remaining difference is the non-primitive CONTENT, which the canonical
    // pinned manifest makes shape-invariant.
    let pad = 512;
    let fp_real = d4_verifier_fingerprint(400, pad, false);
    let fp_padded = d4_verifier_fingerprint(48, pad, true);
    println!("full-manifest uses-all-three (n=400): {fp_real:?}");
    println!("full-manifest padded-from-none (n=48): {fp_padded:?}");
    assert_eq!(
        fp_real, fp_padded,
        "pinning the full canonical manifest [poseidon2, recompose, expose_claim] \
         AND the height MUST make the in-circuit verifier op-list invariant across a \
         child that genuinely uses all three vs one padded from none — the \
         full-manifest feasibility gate for the depth-1 fixed VK"
    );
}

// ============================================================================
// CODEX REVIEW FIXES — the `normalize_to_shape` API made fail-closed.
//
// Findings (codex):
//   F2 [soundness]: `build_and_prove_normalization_layer`'s precondition was name-only
//       ("required NPO types present"), so a proof with an EXTRA table, WRONG lane count,
//       or off-canonical packing passed yet mismatched the canonical VK. Now an EXACT
//       multiset + packing check (`check_canonical_shape`) rejects all of them.
//   F1 [contract]: the layer cannot add a table to an already-proved immutable proof; the
//       exact precondition now REJECTS a proof missing a canonical table (fail-closed).
//   F3 [hidden state]: `inject_canonical_fillers` for `recompose/coeff` silently emitted a
//       plain `recompose` from default builder state (the coeff-CTL flag was off). It now
//       sets the flag itself, so the requested `recompose/coeff` table is emitted.
// ============================================================================

/// A [`CanonicalShapeSpec`] whose `to_table_packing()` + manifest EXACTLY mirror `proof`'s
/// effective (post prove-time clamp) shape — the canonical spec a conforming child lands on.
fn matching_spec(proof: &BatchStarkProof<MyConfig>) -> CanonicalShapeSpec {
    let tp = &proof.table_packing;
    CanonicalShapeSpec {
        log_height: tp.min_trace_height().trailing_zeros() as usize,
        public_lanes: tp.public_lanes(),
        alu_lanes: tp.alu_lanes(),
        canonical_npos: proof
            .non_primitives
            .iter()
            .map(|e| (e.op_type.clone(), e.lanes))
            .collect(),
    }
}

/// POSITIVE: a proof in canonical shape passes the exact-shape precondition.
#[test]
fn check_canonical_shape_accepts_canonical() {
    let (proof, _data) = prove_d4_full_manifest_child(48, 64, true);
    let spec = matching_spec(&proof);
    check_canonical_shape(&proof, &spec)
        .expect("a proof that exactly matches its canonical spec must be accepted");
}

/// NEGATIVE (the soundness tooth): a proof whose manifest/lanes/packing diverge from the
/// canonical spec is REJECTED — extra table, wrong lanes, missing canonical table, and
/// off-canonical packing all fail closed. Before the fix the name-only precondition
/// accepted the extra-table and wrong-lanes cases, so the proof mismatched the canonical VK.
#[test]
fn check_canonical_shape_is_fail_closed() {
    let (proof, _data) = prove_d4_full_manifest_child(48, 64, true);
    let base = matching_spec(&proof);
    assert_eq!(base.canonical_npos.len(), 3, "canonical D4 manifest is 3 NPOs");

    // (a) EXTRA table in the proof vs the spec: the spec lists FEWER types than the proof
    // carries (drop one). The proof's now-unexpected table must be rejected.
    let mut spec_drop = base.clone();
    spec_drop.canonical_npos.pop(); // drop expose_claim
    assert!(
        check_canonical_shape(&proof, &spec_drop).is_err(),
        "a proof with a non-primitive table NOT in the canonical manifest must be rejected"
    );

    // (b) UNEXPECTED / WRONG type at equal length: replace one canonical type with a bogus
    // one. The proof's real table for the replaced type matches nothing → reject.
    let mut spec_bogus = base.clone();
    let last = spec_bogus.canonical_npos.len() - 1;
    spec_bogus.canonical_npos[last] = (NpoTypeId::new("nonexistent_table"), 1);
    assert!(
        check_canonical_shape(&proof, &spec_bogus).is_err(),
        "a proof missing a canonical type (and carrying a non-canonical one) must be rejected"
    );

    // (c) WRONG lane count: the spec pins a canonical type at a lane count the proof's
    // table does not have. A different lane count is a different verifier AIR → reject.
    let mut spec_lanes = base.clone();
    spec_lanes.canonical_npos[0].1 += 1; // bump poseidon2 lanes 1 -> 2
    assert!(
        check_canonical_shape(&proof, &spec_lanes).is_err(),
        "a proof whose canonical table has the WRONG lane count must be rejected"
    );

    // (d) OFF-CANONICAL packing: the spec pins a different fixed height (min_trace_height).
    // The verifier pads every table to this height, so a mismatch is a different VK.
    let mut spec_height = base.clone();
    spec_height.log_height += 1; // 64 -> 128
    assert!(
        check_canonical_shape(&proof, &spec_height).is_err(),
        "a proof at a different fixed height than the canonical packing must be rejected"
    );

    // (e) OFF-CANONICAL packing: a different primitive lane count.
    let mut spec_plane = base.clone();
    spec_plane.public_lanes += 1;
    assert!(
        check_canonical_shape(&proof, &spec_plane).is_err(),
        "a proof with a different public-lane packing than canonical must be rejected"
    );

    // Sanity: the unperturbed spec still accepts — the rejections above are specific, not
    // a blanket failure.
    check_canonical_shape(&proof, &base).expect("the unperturbed canonical spec must accept");
}

// ---- F3: the recompose/coeff filler now emits a `recompose/coeff` table from DEFAULT
//         (flag-off) builder state. ----

/// Build a D=4 child padded from none to the manifest [poseidon2, RECOMPOSE/COEFF,
/// expose_claim] via `inject_canonical_fillers`, starting from DEFAULT builder state (the
/// coeff-CTL flag is NOT set by the caller). The fix makes the injector set the flag itself.
fn build_d4_coeff_manifest_child(n: usize) -> CircuitBuilder<Challenge> {
    let mut builder = CircuitBuilder::<Challenge>::new();
    let perm = default_koalabear_poseidon2_16();
    builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        perm,
    );
    builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    builder.enable_expose_claim::<F>(generate_expose_claim_trace::<F, Challenge>);
    // NB: deliberately NO `set_recompose_coeff_ctl_for_decompose_links(true)` here — the
    // whole point of F3 is that the injector must set it for the `recompose/coeff` entry.

    let expected = builder.alloc_public_input("expected");
    let mut a = builder.alloc_const(Challenge::ZERO, "F(0)");
    let mut b = builder.alloc_const(Challenge::ONE, "F(1)");
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected);

    // Canonical manifest with the RECOMPOSE/COEFF variant in the recompose slot.
    let canonical_npos = vec![
        (NpoTypeId::poseidon2_perm(Poseidon2Config::KOALA_BEAR_D4_W16), 1usize),
        (NpoTypeId::recompose_with_coeff_lookups(), 1usize),
        (NpoTypeId::expose_claim(), 1usize),
    ];
    inject_canonical_fillers::<Challenge, F>(
        &mut builder,
        Poseidon2Config::KOALA_BEAR_D4_W16,
        &canonical_npos,
        &BTreeSet::new(),
        expected,
    )
    .unwrap();
    builder
}

/// Prove a D=4 coeff-manifest child against the canonical packing (recompose table
/// registered in coeff-split mode), with the lookup debugger ON so any bus imbalance panics.
fn prove_d4_coeff_manifest_child(n: usize, min_trace_height: usize) -> BatchStarkProof<MyConfig> {
    let builder = build_d4_coeff_manifest_child(n);
    let table_packing = TablePacking::new(2, 4)
        .with_min_trace_height(min_trace_height)
        .with_npo_lanes(NpoTypeId::poseidon2_perm(Poseidon2Config::KOALA_BEAR_D4_W16), 1)
        .with_npo_lanes(NpoTypeId::recompose_with_coeff_lookups(), 1)
        .with_npo_lanes(NpoTypeId::expose_claim(), 1);
    let config_proving = make_test_config();
    let circuit = builder.build().unwrap();

    let preprocessors: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
        Box::new(ExposeClaimPreprocessor),
    ];
    let mut air_builders: Vec<Box<dyn NpoAirBuilder<MyConfig, TRACE_D4>>> =
        poseidon2_air_builders::<MyConfig, TRACE_D4>();
    air_builders.extend(recompose_air_builders::<MyConfig, TRACE_D4>(1, true));
    air_builders.extend(expose_claim_air_builders::<MyConfig, TRACE_D4>());

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, TRACE_D4>(
            &circuit,
            &table_packing,
            &preprocessors,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();
    runner
        .set_public_inputs(&[Challenge::from(compute_fibonacci_classical(n))])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config_proving)
        .with_table_packing(table_packing)
        .with_debug_lookups();
    prover.register_poseidon2_table::<TRACE_D4>(Poseidon2Config::KOALA_BEAR_D4_W16);
    prover.register_recompose_table::<TRACE_D4>(true);
    prover.register_expose_claim_table::<TRACE_D4>();

    let proof = prover.prove_all_tables(&traces, &circuit_prover_data).unwrap();
    prover.verify_all_tables(&proof).unwrap();
    proof
}

/// F3 POSITIVE: a `recompose/coeff` canonical filler injected from DEFAULT builder state
/// produces a real `recompose/coeff` table in `proof.non_primitives` (NOT a plain
/// `recompose`). Before the fix the coeff-CTL flag was off, so the injector silently emitted
/// a `recompose` and the requested `recompose/coeff` entry was absent.
#[test]
fn coeff_filler_emits_recompose_coeff_from_default_state() {
    let proof = prove_d4_coeff_manifest_child(48, 64);
    let manifest = d4_manifest(&proof);
    println!("recompose/coeff-filler D4 child manifest: {manifest:?}");
    assert!(
        manifest.iter().any(|m| m == "recompose/coeff"),
        "the injector must emit a `recompose/coeff` table from default builder state, got {manifest:?}"
    );
    assert!(
        !manifest.iter().any(|m| m == "recompose"),
        "the requested coeff variant must NOT degrade to a plain `recompose`, got {manifest:?}"
    );
}
