//! COVERAGE for the dregg IVC height-1 FRI opening path (the gap the prior agent
//! named: no fork test aggregated a W24/expose_claim-bearing proof).
//!
//! The dregg segment-digest sponge runs a W24 Poseidon2 permutation inside the
//! recursion verifier circuit. When that verifier circuit is itself proven as a
//! batch-stark, a SINGLE W24 perm produces a HEIGHT-1 (base-domain-size-1) W24
//! `poseidon2_perm` table. The W24 AIR uses `next_slice()` (it is a chained,
//! next-row-bearing table), so the inner proof opens that table at BOTH `zeta`
//! and `zeta_next`. The recursion verifier's `open_input` then groups that matrix
//! at the `log_blowup` (height-1) key and asserts its reduced opening is zero.
//!
//! IMPORTANT — what these tests DO and DO NOT show: both the single-level and the
//! two-level (aggregation-shaped) tests below PASS. They exercise height-1 W24
//! tables AND height-1 LogUp PERMUTATION instances opened at (zeta, zeta_next),
//! and confirm the reduced opening is correctly zero for those synthetic proofs.
//!
//! They do NOT reproduce the dregg `k_fold_turn_chain_proves_and_verifies`
//! failure. The dregg failure was localized (see the investigation notes in the
//! handoff) to the PERMUTATION round of a genuinely-degree-0 (height-1) instance
//! where the in-circuit opened value at the challenge point z DIFFERS from the
//! MMCS leaf value at the query point x (`p_at_z != p_at_x`) — even though native
//! verification of that same proof passes. That is a circuit-model-vs-native
//! divergence specific to the dregg child-proof shape that these synthetic
//! proofs do not induce. Keep these as a regression guard for the path; the true
//! dregg reproducer needs the specific child structure.

mod common;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    Poseidon2Config, PermConfig, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_circuit_prover::batch_stark_prover::poseidon2_air_builders;
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor, Poseidon2Prover,
    RecomposePreprocessor, TablePacking, TableProver,
};
use p3_dft::Radix2DitParallel;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_lookup::logup::LogUpGadget;
use p3_poseidon2_circuit_air::{BabyBearD4Width16, BabyBearD4Width24};
use p3_recursion::pcs::fri::{
    FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs,
};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_recursion::{Poseidon2Config as RecPoseidon2Config, VerificationError};
use p3_uni_stark::StarkConfig;

use p3_baby_bear::{
    BabyBear, default_babybear_poseidon2_16, default_babybear_poseidon2_24,
};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation};

use crate::common::InnerFriGeneric;

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

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

/// The dregg recursion FRI knobs: log_blowup=3 (degree-7 Poseidon2 AIR), max_log_arity=1.
fn dregg_like_config() -> MyConfig {
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

fn fri_verifier_params() -> FriVerifierParams {
    FriVerifierParams::with_mmcs(
        3,
        0,
        0,
        14,
        PermConfig::poseidon2(Poseidon2Config::BABY_BEAR_D4_W16),
    )
}

/// Same as `dregg_like_config` but with a parameterized `log_blowup`. The dregg
/// `ir2_leaf_wrap` inner proof is proven at `log_blowup=6`; the prior fork tests
/// only covered `log_blowup=3`.
fn dregg_like_config_lb(log_blowup: usize) -> MyConfig {
    let perm = default_babybear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup,
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

fn fri_verifier_params_lb(log_blowup: usize) -> FriVerifierParams {
    FriVerifierParams::with_mmcs(
        log_blowup,
        0,
        0,
        14,
        PermConfig::poseidon2(Poseidon2Config::BABY_BEAR_D4_W16),
    )
}

/// Enable the W16 (challenger) + W24 (segment-digest) Poseidon2 perms + recompose
/// on a fresh circuit builder — exactly the dregg recursion-verifier table set.
fn enable_dregg_tables(cb: &mut CircuitBuilder<Challenge>) {
    cb.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        default_babybear_poseidon2_16(),
    );
    cb.enable_poseidon2_perm_width_24::<BabyBearD4Width24, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width24>,
        default_babybear_poseidon2_24(),
    );
    cb.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
}

/// Build + prove the INNER batch-stark proof whose trace contains a SINGLE W24
/// Poseidon2 perm — i.e. a HEIGHT-1 W24 `poseidon2_perm` table (next-row-bearing).
fn prove_inner_with_one_w24_perm(
    config: &MyConfig,
) -> (
    p3_circuit_prover::BatchStarkProof<MyConfig>,
    CircuitProverData<MyConfig>,
) {
    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);

    // One W24 perm of a 6-ext-limb (= W24/D) state seeded with nonzero IVs (mirrors
    // seg_poseidon_commit's domain-tagged sponge so we never touch the zero witness).
    let tag = cb.define_const(Challenge::from(BabyBear::from_u64(0x9E37u64)));
    let state: Vec<_> = (0..6).map(|_| tag).collect();
    let out = cb
        .add_poseidon2_perm_for_challenger(Poseidon2Config::BABY_BEAR_D4_W24, &state)
        .expect("W24 perm op builds");
    // Bind the FULL ext output[0] to a public input so the perm is load-bearing
    // (not DCE'd) and the host feeds the exact value the in-circuit perm yields.
    let expected = cb.alloc_public_input("w24_out0");
    cb.connect(out[0], expected);

    let circuit = cb.build().unwrap();
    let mut runner = circuit.runner();

    // Compute the expected W24 output (full ext limb 0) host-side. The in-circuit
    // state is 6 ext limbs each = `tag` (coeff-0 = tag, coeffs 1..3 = 0); the perm
    // operates on the 24 base lanes (lane 4*limb+coeff).
    let perm24 = default_babybear_poseidon2_24();
    let mut base = [BabyBear::ZERO; 24];
    let tag_v = BabyBear::from_u64(0x9E37u64);
    for limb in 0..6 {
        base[limb * D] = tag_v; // coeff-0 of each ext limb = tag; others zero.
    }
    let base_out = Permutation::permute(&perm24, base);
    // ext limb 0 = base lanes [0..4] as the 4 coeffs.
    let expected_out0 = Challenge::from_basis_coefficients_slice(&base_out[0..D])
        .expect("valid ext coeffs");
    runner.set_public_inputs(&[expected_out0]).unwrap();
    let traces = runner.run().unwrap();

    // Prove the circuit as a batch-stark with W16+W24+recompose tables.
    // CRITICAL: min_trace_height = 1 (the fork's recursion default `TablePacking::new(1, 4)`
    // sets no fri_params floor) so the single W24 perm yields a genuine HEIGHT-1 table —
    // the exact shape the dregg segment-digest fold produces. Using `with_fri_params(0, 3)`
    // (min_height = 16) would mask the bug by padding every table to >= 16 rows.
    let table_packing = TablePacking::new(1, 4);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<MyConfig, D>();
    air_builders.extend(p3_circuit_prover::batch_stark_prover::recompose_air_builders(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, Challenge, D>(
            &circuit,
            &table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config.clone()).with_table_packing(table_packing);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
    prover.register_recompose_table::<D>(false);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .expect("inner proof with a height-1 W24 table proves");
    prover
        .verify_all_tables(&proof)
        .expect("inner proof verifies natively");
    (proof, circuit_prover_data)
}

/// Build + prove an inner proof whose permutation round MIXES a height-1 W24
/// perm table with a TALLER W16 perm table. This is the shape the prior handoff
/// localized the dregg `k_fold` failure to: a genuinely degree-0 (height-1)
/// permutation matrix SHARING the permutation round with taller LogUp instances.
///
/// The inner circuit runs MANY W16 challenger perms (-> a W16 `poseidon2_perm`
/// table of height >> 1) and a SINGLE W24 perm (-> a height-1 W24 table). Both
/// table kinds are LogUp-bearing, so the inner proof's permutation commitment
/// carries a tall W16 perm matrix AND a height-1 W24 perm matrix in the same round.
fn prove_inner_mixed_height_perm(
    config: &MyConfig,
    n_w16: usize,
) -> (
    p3_circuit_prover::BatchStarkProof<MyConfig>,
    CircuitProverData<MyConfig>,
) {
    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);

    // MANY INDEPENDENT W16 perms, each seeded with a distinct nonzero tag and each
    // bound to its own public input -> a TALL W16 poseidon2_perm table (n_w16 rows),
    // every perm load-bearing and witness-balanced. W16/D4 takes 4 ext-limb inputs.
    let mut w16_outs = Vec::with_capacity(n_w16);
    let mut w16_tags = Vec::with_capacity(n_w16);
    for k in 0..n_w16 {
        let tag_k_v = BabyBear::from_u64(0x9E37u64 + k as u64);
        let tag_k = cb.define_const(Challenge::from(tag_k_v));
        let state: Vec<_> = (0..4).map(|_| tag_k).collect();
        let out = cb
            .add_poseidon2_perm_for_challenger(Poseidon2Config::BABY_BEAR_D4_W16, &state)
            .expect("W16 perm op builds");
        let expected = cb.alloc_public_input("w16_out0");
        cb.connect(out[0], expected);
        w16_outs.push(out);
        w16_tags.push(tag_k_v);
    }

    // A SINGLE W24 perm -> a height-1 W24 poseidon2_perm table.
    let tag = cb.define_const(Challenge::from(BabyBear::from_u64(0x9E37u64)));
    let w24_state: Vec<_> = (0..6).map(|_| tag).collect();
    let w24_out = cb
        .add_poseidon2_perm_for_challenger(Poseidon2Config::BABY_BEAR_D4_W24, &w24_state)
        .expect("W24 perm op builds");
    let expected_w24 = cb.alloc_public_input("w24_out0");
    cb.connect(w24_out[0], expected_w24);
    let _ = &w16_outs;

    let circuit = cb.build().unwrap();
    let mut runner = circuit.runner();

    // Compute expected public outputs host-side.
    let perm16 = default_babybear_poseidon2_16();
    let perm24 = default_babybear_poseidon2_24();
    let tag_v = BabyBear::from_u64(0x9E37u64);

    // W16: each independent perm's out[0] (ext limb 0 = base lanes [0..4] recomposed).
    // W16/D4: the 4 ext-limb inputs each = `tag_k` map to base lanes 0,4,8,12 = tag_k.
    let mut public_vals: Vec<Challenge> = Vec::with_capacity(n_w16 + 1);
    for &tag_k_v in &w16_tags {
        let mut state = [BabyBear::ZERO; 16];
        for limb in 0..4 {
            state[limb * D] = tag_k_v;
        }
        let out = Permutation::permute(&perm16, state);
        public_vals.push(
            Challenge::from_basis_coefficients_slice(&out[0..D]).expect("valid ext coeffs"),
        );
    }

    let mut w24_base = [BabyBear::ZERO; 24];
    for limb in 0..6 {
        w24_base[limb * D] = tag_v;
    }
    let w24_base_out = Permutation::permute(&perm24, w24_base);
    let w24_out0 = Challenge::from_basis_coefficients_slice(&w24_base_out[0..D])
        .expect("valid ext coeffs");
    public_vals.push(w24_out0);

    runner.set_public_inputs(&public_vals).unwrap();
    let traces = runner.run().unwrap();

    let table_packing = TablePacking::new(1, 4);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<MyConfig, D>();
    air_builders.extend(p3_circuit_prover::batch_stark_prover::recompose_air_builders(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, Challenge, D>(
            &circuit,
            &table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config.clone()).with_table_packing(table_packing);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
    prover.register_recompose_table::<D>(false);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .expect("mixed-height inner proof proves");
    prover
        .verify_all_tables(&proof)
        .expect("mixed-height inner proof verifies natively");
    (proof, circuit_prover_data)
}

/// THE MIXED-HEIGHT REPRODUCER (single-level, fast): aggregate an inner proof
/// whose permutation round mixes a height-1 W24 perm matrix with a TALLER W16
/// perm matrix. If the height-1 leaf-ordering bug is present, the recursive
/// `open_input` reads `p_at_x` from the wrong leaf for the height-1 permutation
/// matrix and the `ro==0` assert (verifier.rs ~:1326) panics with a
/// WitnessConflict. After the fix it folds + verifies.
#[test]
fn mixed_height_perm_aggregation_folds_and_verifies() -> Result<(), VerificationError> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_test_writer()
        .try_init();
    let config = dregg_like_config();
    // 64 W16 perms -> a W16 perm table padded to height 64 (>> 1); single W24 -> height 1.
    let (inner_proof, inner_prover_data) = prove_inner_mixed_height_perm(&config, 64);

    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);
    cb.enable_expose_claim::<F>(p3_circuit::ops::generate_expose_claim_trace::<F, Challenge>);

    let common = inner_prover_data.common_data();
    let fri_params = fri_verifier_params();
    let lookup_gadget = LogUpGadget::new();
    let verif_provers: Vec<Box<dyn TableProver<MyConfig>>> = vec![
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W16,
            ConstraintProfile::Standard,
        )),
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W24,
            ConstraintProfile::Standard,
        )),
    ];

    let (verifier_inputs, op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        RecPoseidon2Config,
        WIDTH,
        RATE,
        D,
    >(
        &config,
        &mut cb,
        &inner_proof,
        &fri_params,
        common,
        &lookup_gadget,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
        &verif_provers,
    )?;

    let verification_circuit = cb.build().unwrap();
    let mut runner = verification_circuit.runner();

    let inner_pis: Vec<Vec<F>> = inner_proof
        .non_primitives
        .iter()
        .map(|e| e.public_values.clone())
        .collect();
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&inner_pis, &inner_proof.proof, common);
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(
        &mut runner,
        &op_ids,
        &inner_proof.proof.opening_proof,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;

    let _traces = runner.run().map_err(VerificationError::Circuit)?;
    Ok(())
}

/// THE log_blowup=6 REPRODUCER. The dregg `ir2_leaf_wrap` inner proof is proven
/// at `log_blowup=6` (degree-7 Poseidon2, max blowup), which the prior fork tests
/// (all `log_blowup=3`) never covered. The prompt localizes the same-endpoint IVC
/// residual to exactly this knob: a width-44 height-1 LogUp permutation matrix
/// whose in-circuit leaf (`p_at_x`) + OOD opening (`p_at_z`) get witness-
/// contaminated to full-extension values at cols 2,3 — surfacing as a
/// WitnessConflict at the height-1 `ro==0` assert (verifier.rs ~:1326).
#[test]
fn mixed_height_perm_aggregation_lb6_reproduces() -> Result<(), VerificationError> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_test_writer()
        .try_init();
    let log_blowup = 6;
    let config = dregg_like_config_lb(log_blowup);
    let (inner_proof, inner_prover_data) = prove_inner_mixed_height_perm(&config, 64);

    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);
    cb.enable_expose_claim::<F>(p3_circuit::ops::generate_expose_claim_trace::<F, Challenge>);

    let common = inner_prover_data.common_data();
    let fri_params = fri_verifier_params_lb(log_blowup);
    let lookup_gadget = LogUpGadget::new();
    let verif_provers: Vec<Box<dyn TableProver<MyConfig>>> = vec![
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W16,
            ConstraintProfile::Standard,
        )),
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W24,
            ConstraintProfile::Standard,
        )),
    ];

    let (verifier_inputs, op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        RecPoseidon2Config,
        WIDTH,
        RATE,
        D,
    >(
        &config,
        &mut cb,
        &inner_proof,
        &fri_params,
        common,
        &lookup_gadget,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
        &verif_provers,
    )?;

    let verification_circuit = cb.build().unwrap();
    let mut runner = verification_circuit.runner();

    let inner_pis: Vec<Vec<F>> = inner_proof
        .non_primitives
        .iter()
        .map(|e| e.public_values.clone())
        .collect();
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&inner_pis, &inner_proof.proof, common);
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(
        &mut runner,
        &op_ids,
        &inner_proof.proof.opening_proof,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;

    let _traces = runner.run().map_err(VerificationError::Circuit)?;
    Ok(())
}

/// THE REPRODUCER: prove an inner proof carrying a height-1 W24 table, then
/// recursively verify it. The recursive `open_input` hits the height-1
/// W24 matrix opened at (zeta, zeta_next) and must yield ro==0.
#[test]
fn height1_w24_aggregation_folds_and_verifies() -> Result<(), VerificationError> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_test_writer()
        .try_init();
    let config = dregg_like_config();
    let (inner_proof, inner_prover_data) = prove_inner_with_one_w24_perm(&config);

    // --- Build the recursion-verifier circuit for the inner proof. ---
    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);
    cb.enable_expose_claim::<F>(p3_circuit::ops::generate_expose_claim_trace::<F, Challenge>);

    let common = inner_prover_data.common_data();
    let fri_params = fri_verifier_params();
    let lookup_gadget = LogUpGadget::new();
    let verif_provers: Vec<Box<dyn TableProver<MyConfig>>> = vec![
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W16,
            ConstraintProfile::Standard,
        )),
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W24,
            ConstraintProfile::Standard,
        )),
    ];

    let (verifier_inputs, op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        RecPoseidon2Config,
        WIDTH,
        RATE,
        D,
    >(
        &config,
        &mut cb,
        &inner_proof,
        &fri_params,
        common,
        &lookup_gadget,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
        &verif_provers,
    )?;

    let verification_circuit = cb.build().unwrap();
    let mut runner = verification_circuit.runner();

    let inner_pis: Vec<Vec<F>> = inner_proof
        .non_primitives
        .iter()
        .map(|e| e.public_values.clone())
        .collect();
    // Primitive tables (const/public/alu) carry their own publics handled by pack_values;
    // pass the per-instance air publics. The inner had one public input (the W24 out0).
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&inner_pis, &inner_proof.proof, common);
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(
        &mut runner,
        &op_ids,
        &inner_proof.proof.opening_proof,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;

    // THE ASSERTION: running the recursion-verifier circuit must succeed.
    // Before the fix this panics with a WitnessConflict at the height-1
    // reduced-opening assert (open_input :1324).
    let _traces = runner.run().map_err(VerificationError::Circuit)?;
    Ok(())
}

/// TWO-LEVEL reproducer (matches the dregg AGGREGATION layer): build the
/// recursion-verifier circuit for a height-1-W24-bearing inner proof, PROVE that
/// verifier circuit as a batch-stark (now the verifier circuit's OWN tables carry
/// height-1 lookup-bearing instances), then RECURSIVELY VERIFY that proof. This is
/// the exact structure that fails the dregg `k_fold` at "aggregation layer failed".
#[test]
fn height1_w24_two_level_aggregation_folds() -> Result<(), VerificationError> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_test_writer()
        .try_init();
    let config = dregg_like_config();
    let (inner_proof, inner_prover_data) = prove_inner_with_one_w24_perm(&config);

    // --- LEVEL 1: build the recursion-verifier circuit for the inner proof. ---
    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);
    cb.enable_expose_claim::<F>(p3_circuit::ops::generate_expose_claim_trace::<F, Challenge>);

    let common = inner_prover_data.common_data();
    let fri_params = fri_verifier_params();
    let lookup_gadget = LogUpGadget::new();
    let verif_provers: Vec<Box<dyn TableProver<MyConfig>>> = vec![
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W16,
            ConstraintProfile::Standard,
        )),
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W24,
            ConstraintProfile::Standard,
        )),
    ];

    let (verifier_inputs, op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        RecPoseidon2Config,
        WIDTH,
        RATE,
        D,
    >(
        &config,
        &mut cb,
        &inner_proof,
        &fri_params,
        common,
        &lookup_gadget,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
        &verif_provers,
    )?;

    let verification_circuit = cb.build().unwrap();
    let mut runner = verification_circuit.runner();
    let inner_pis: Vec<Vec<F>> = inner_proof
        .non_primitives
        .iter()
        .map(|e| e.public_values.clone())
        .collect();
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&inner_pis, &inner_proof.proof, common);
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;
    set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(
        &mut runner,
        &op_ids,
        &inner_proof.proof.opening_proof,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;
    let level1_traces = runner.run().map_err(VerificationError::Circuit)?;

    // --- Prove the LEVEL-1 verifier circuit as a batch-stark (min_height=1). ---
    let l1_packing = TablePacking::new(1, 4);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(false)),
        Box::new(p3_circuit_prover::ExposeClaimPreprocessor::new()),
    ];
    let mut l1_air_builders = poseidon2_air_builders::<MyConfig, D>();
    l1_air_builders
        .extend(p3_circuit_prover::batch_stark_prover::recompose_air_builders(1, false));
    l1_air_builders.push(Box::new(p3_circuit_prover::ExposeClaimAirBuilder::<D>::new()));
    let (l1_airs_degrees, l1_prim, l1_nonprim) =
        get_airs_and_degrees_with_prep::<MyConfig, Challenge, D>(
            &verification_circuit,
            &l1_packing,
            &npo_prep,
            &l1_air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (l1_airs, l1_degrees): (Vec<_>, Vec<_>) = l1_airs_degrees.into_iter().unzip();
    let l1_prover_data = ProverData::from_airs_and_degrees(&config, &l1_airs, &l1_degrees);
    let l1_circuit_prover_data = CircuitProverData::new(l1_prover_data, l1_prim, l1_nonprim);
    let mut l1_prover = BatchStarkProver::new(config.clone()).with_table_packing(l1_packing);
    l1_prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    l1_prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
    l1_prover.register_recompose_table::<D>(false);
    l1_prover.register_expose_claim_table::<D>();
    let l1_proof = l1_prover
        .prove_all_tables(&level1_traces, &l1_circuit_prover_data)
        .expect("level-1 verifier circuit proves");
    l1_prover
        .verify_all_tables(&l1_proof)
        .expect("level-1 proof verifies natively");

    // --- LEVEL 2: recursively verify the level-1 proof. THE TRIGGER. ---
    let mut cb2 = CircuitBuilder::new();
    enable_dregg_tables(&mut cb2);
    cb2.enable_expose_claim::<F>(p3_circuit::ops::generate_expose_claim_trace::<F, Challenge>);
    let l1_common = l1_circuit_prover_data.common_data();
    let l2_provers: Vec<Box<dyn TableProver<MyConfig>>> = vec![
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W16,
            ConstraintProfile::Standard,
        )),
        Box::new(Poseidon2Prover::new(
            RecPoseidon2Config::BABY_BEAR_D4_W24,
            ConstraintProfile::Standard,
        )),
        Box::new(p3_circuit_prover::RecomposeProver::<D>::new(1, false)),
        Box::new(p3_circuit_prover::ExposeClaimProver::<D>::new()),
    ];
    let (l2_inputs, l2_op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        RecPoseidon2Config,
        WIDTH,
        RATE,
        D,
    >(
        &config,
        &mut cb2,
        &l1_proof,
        &fri_params,
        l1_common,
        &lookup_gadget,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
        &l2_provers,
    )?;
    let l2_circuit = cb2.build().unwrap();
    let mut l2_runner = l2_circuit.runner();
    let l1_pis: Vec<Vec<F>> = l1_proof
        .non_primitives
        .iter()
        .map(|e| e.public_values.clone())
        .collect();
    let (l2_public, l2_private) = l2_inputs.pack_values(&l1_pis, &l1_proof.proof, l1_common);
    l2_runner
        .set_public_inputs(&l2_public)
        .map_err(VerificationError::Circuit)?;
    l2_runner
        .set_private_inputs(&l2_private)
        .map_err(VerificationError::Circuit)?;
    set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(
        &mut l2_runner,
        &l2_op_ids,
        &l1_proof.proof.opening_proof,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;
    // THE TRIGGER: this is where the dregg aggregation layer panics with a
    // height-1 WitnessConflict if the bug is present.
    let _ = l2_runner.run().map_err(VerificationError::Circuit)?;
    Ok(())
}
