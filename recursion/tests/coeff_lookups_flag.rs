//! Opt-in `recompose/coeff` table registration via `FriRecursionBackend::with_coeff_lookups`.
//!
//! GROUND: the leaf-wrap challenger is `BABY_BEAR_D4_W16` (extension_degree 4) inside a D=4
//! circuit, so the backend's `cl = challenger.extension_degree() != D` is `false` and the
//! `recompose/coeff` prover/preprocessor/air-builder are gated OFF. A circuit that reads 4
//! CONSECUTIVE base lanes of one ext limb via `decompose_ext_to_base_coeffs` (with the
//! `recompose/coeff` CTL routing on the builder side) then has no prover-side coeff table, so
//! its WitnessChecks bus does not balance — this is the breadstuffs G2 custom-commitment
//! in-circuit-prove blocker.
//!
//! `FriRecursionBackend::with_coeff_lookups()` ORs `force_coeff_lookups` into that `cl`, so the
//! coeff table is registered even when `extension_degree == D`. It is OPT-IN: the shared/default
//! backend stays coeff-free and existing leaf VKs do not move.
//!
//! These tests drive the SAME `split_coeff_tables` decision the backend computes (`cl` for the
//! D=4 path), taken from a real `FriRecursionBackend` instance, through the same prove/verify
//! harness `prove_next_layer` uses (`get_airs_and_degrees_with_prep` + `BatchStarkProver` +
//! `register_table_prover`). The backend's NPO methods themselves require a full
//! `FriRecursionConfig` (FRI verifier plumbing) which a hand-built circuit does not have, so we
//! exercise the flag's exact effect — the `cl` value it feeds the three recompose helpers — over
//! a plain config.

mod common;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    Poseidon2Config, generate_expose_claim_trace, generate_poseidon2_trace,
    generate_recompose_trace,
};
use p3_circuit_prover::batch_stark_prover::{
    expose_claim_air_builders, poseidon2_air_builders, recompose_air_builders,
    recompose_preprocessor,
};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, ExposeClaimPreprocessor,
    Poseidon2Preprocessor, TablePacking,
};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::{ChallengerPermConfig, FriRecursionBackend};
use p3_uni_stark::StarkConfig;

use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::BasedVectorSpace;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

type F = BabyBear;
const D: usize = 4;
const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST_ELEMS: usize = 8;
type Challenge = BinomialExtensionField<F, D>;
type Dft = Radix2DitParallel<F>;
type Perm = p3_baby_bear::Poseidon2BabyBear<WIDTH>;
type MyHash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_ELEMS>;
type MyCompress = TruncatedPermutation<Perm, 2, DIGEST_ELEMS, WIDTH>;
type MyMmcs =
    MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, DIGEST_ELEMS>;
type ChallengeMmcs = ExtensionMmcs<F, Challenge, MyMmcs>;
type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;
type MyPcs = TwoAdicFriPcs<F, Dft, MyMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

fn cfg() -> MyConfig {
    let perm = default_babybear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 3,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 38,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 14,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
    MyConfig::new(pcs, Challenger::new(perm))
}

/// The exact `cl` (`split_coeff_tables`) decision the D=4 backend path computes for `backend`.
/// Mirrors `fri.rs`: `challenger.extension_degree() != D || force_coeff_lookups`.
fn backend_split_d4(backend: &FriRecursionBackend<WIDTH, RATE, Poseidon2Config>) -> bool {
    backend.challenger_perm_config.extension_degree() != D || backend.force_coeff_lookups
}

fn enable(cb: &mut CircuitBuilder<Challenge>) {
    cb.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        default_babybear_poseidon2_16(),
    );
    cb.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    cb.enable_expose_claim::<F>(generate_expose_claim_trace::<F, Challenge>);
}

/// Prove + verify a hand-built D=4 circuit, registering the recompose table(s) with the given
/// `split` (the value `with_coeff_lookups` flips). When `debug` is set, the lookup debugger runs
/// before proving and PANICS on any WitnessChecks imbalance.
fn prove_and_verify(
    circuit: p3_circuit::Circuit<Challenge>,
    pubs: &[Challenge],
    split: bool,
    debug: bool,
) -> Result<(Vec<String>, Vec<usize>), String> {
    let mut runner = circuit.runner();
    runner.set_public_inputs(pubs).map_err(|e| format!("{e:?}"))?;
    let traces = runner.run().map_err(|e| format!("{e:?}"))?;

    let table_packing = TablePacking::new(1, 4);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        recompose_preprocessor::<F>(split),
        Box::new(ExposeClaimPreprocessor),
    ];
    let mut air_builders = poseidon2_air_builders::<MyConfig, D>();
    air_builders.extend(recompose_air_builders::<MyConfig, D>(1, split));
    air_builders.extend(expose_claim_air_builders::<MyConfig, D>());

    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, Challenge, D>(
            &circuit,
            &table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .map_err(|e| format!("prep: {e:?}"))?;
    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    // The full AIR-degree list backs the recursion VK fingerprint: identical here ⟹ unmoved VK.
    let degrees_fp = degrees.clone();
    let config = cfg();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let cpd = CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(config.clone()).with_table_packing(table_packing);
    if debug {
        prover = prover.with_debug_lookups();
    }
    // Registers the std `recompose` prover, and (when split) the `recompose/coeff` prover too —
    // exactly `backend.non_primitive_provers(4)` for the recompose family.
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    prover.register_recompose_table::<D>(split);
    prover.register_expose_claim_table::<D>();

    let proof = prover
        .prove_all_tables(&traces, &cpd)
        .map_err(|e| format!("prove: {e:?}"))?;
    let op_types: Vec<String> = proof
        .non_primitives
        .iter()
        .map(|e| format!("{}:{}", e.op_type.as_str(), e.rows))
        .collect();
    prover
        .verify_all_tables(&proof)
        .map_err(|e| format!("verify: {e:?}"))?;
    Ok((op_types, degrees_fp))
}

/// Build a circuit that decomposes ONE ext public input into its 4 CONSECUTIVE base coefficients
/// via `decompose_ext_to_base_coeffs` (routed through `recompose/coeff` by the CTL flag) and
/// exposes them. This is the breadstuffs G2 shape: an in-circuit base sponge reading
/// `WideHash::to_felts()[0..4]` = 4 consecutive base lanes of one ext limb.
fn build_consecutive_lane_circuit() -> (p3_circuit::Circuit<Challenge>, Vec<Challenge>) {
    let mut cb = CircuitBuilder::new();
    enable(&mut cb);
    // Route the decompose reconnect through the `recompose/coeff` table (the builder-side switch).
    cb.set_recompose_coeff_ctl_for_decompose_links(true);

    let x = cb.alloc_public_input("ext_limb");
    let coeffs = cb
        .decompose_ext_to_base_coeffs::<F>(x)
        .expect("decompose ext -> 4 base coeffs");
    assert_eq!(coeffs.len(), D, "one ext limb decomposes to D=4 base lanes");
    cb.expose_as_public_output(&coeffs);

    let circuit = cb.build().unwrap();
    // x = c0 + c1*g + c2*g^2 + c3*g^3 for distinct base coeffs (consecutive lanes).
    let coeffs_bf = [
        BabyBear::from_u64(7),
        BabyBear::from_u64(11),
        BabyBear::from_u64(13),
        BabyBear::from_u64(17),
    ];
    let x_val = <Challenge as BasedVectorSpace<BabyBear>>::from_basis_coefficients_slice(&coeffs_bf)
        .unwrap();
    (circuit, vec![x_val])
}

/// POSITIVE: with the `with_coeff_lookups()` backend (split = true), the consecutive-lane
/// decompose circuit PROVES + VERIFIES and the WitnessChecks bus balances (debug lookups on).
#[test]
fn consecutive_lane_decompose_proves_with_coeff_lookups() {
    let backend = FriRecursionBackend::<WIDTH, RATE, Poseidon2Config>::new(
        Poseidon2Config::BABY_BEAR_D4_W16,
    )
    .with_coeff_lookups();
    let split = backend_split_d4(&backend);
    assert!(split, "with_coeff_lookups must force the coeff table on the D=4 leaf-wrap path");

    let (circuit, pubs) = build_consecutive_lane_circuit();
    let (ops, _) = prove_and_verify(circuit, &pubs, split, /*debug=*/ true)
        .expect("consecutive-lane decompose must prove+verify with coeff lookups");
    eprintln!("[coeff-on] non_primitives = {ops:?}");
    assert!(
        ops.iter().any(|o| o.starts_with("recompose/coeff:")),
        "the proof must carry a non-empty recompose/coeff table"
    );
}

/// NEGATIVE CONTROL (today's blocker): the SAME circuit without the flag (split = false) fails —
/// the `recompose/coeff` lookups are unmatched, so either prep rejects the unhandled op or the
/// WitnessChecks bus fails to balance at verify.
#[test]
fn consecutive_lane_decompose_fails_without_coeff_lookups() {
    let backend = FriRecursionBackend::<WIDTH, RATE, Poseidon2Config>::new(
        Poseidon2Config::BABY_BEAR_D4_W16,
    );
    let split = backend_split_d4(&backend);
    assert!(!split, "the default backend leaves the coeff table off on the D=4 leaf-wrap path");

    let (circuit, pubs) = build_consecutive_lane_circuit();
    // debug = false so a bus imbalance surfaces as a verify rejection rather than a panic.
    let r = prove_and_verify(circuit, &pubs, split, /*debug=*/ false);
    eprintln!("[coeff-off] result = {r:?}");
    assert!(
        r.is_err(),
        "without the coeff table the recompose/coeff ops are unaccounted; must fail"
    );
}

/// INERTNESS: a circuit with NO `recompose/coeff` op present. The flag-on (split = true) and
/// flag-off (split = false) registrations must produce the SAME proof table set — the
/// registered-but-unused coeff table must NOT appear in `non_primitives` (no zero-row table), so
/// the recursion VK fingerprint does not move. The shared default is byte-identical when off.
#[test]
fn coeff_table_is_inert_when_unused() {
    fn build_no_coeff() -> (p3_circuit::Circuit<Challenge>, Vec<Challenge>) {
        let mut cb = CircuitBuilder::new();
        enable(&mut cb);
        // No `set_recompose_coeff_ctl_for_decompose_links(true)`: a plain add, exposed.
        let a = cb.define_const(Challenge::from(BabyBear::from_u64(3)));
        let b = cb.define_const(Challenge::from(BabyBear::from_u64(5)));
        let out = cb.add(a, b);
        cb.expose_as_public_output(&[out]);
        (cb.build().unwrap(), vec![])
    }

    let (c_on, p_on) = build_no_coeff();
    let (ops_on, degrees_on) = prove_and_verify(c_on, &p_on, /*split=*/ true, /*debug=*/ true)
        .expect("flag-on, unused: proves+verifies");
    let (c_off, p_off) = build_no_coeff();
    let (ops_off, degrees_off) =
        prove_and_verify(c_off, &p_off, /*split=*/ false, /*debug=*/ true)
            .expect("flag-off baseline: proves+verifies");

    eprintln!("[inert on ] non_primitives = {ops_on:?} airs={}", degrees_on.len());
    eprintln!("[inert off] non_primitives = {ops_off:?} airs={}", degrees_off.len());
    assert!(
        !ops_on.iter().any(|o| o.starts_with("recompose/coeff:")),
        "the unused coeff table must NOT appear in the proof (inert), got {ops_on:?}"
    );
    assert_eq!(
        ops_on, ops_off,
        "flag-on-but-unused must yield the SAME table set as flag-off (fingerprint unchanged)"
    );
    assert_eq!(
        degrees_on, degrees_off,
        "the full AIR-degree set must be identical flag-on vs flag-off (VK fingerprint unmoved)"
    );
}
