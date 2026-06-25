//! FROZEN-PROOF REPLAY for the IVC same-endpoint residual.
//!
//! The dregg `k_fold_turn_chain_proves_and_verifies` fold (~65s) fails when an
//! AGGREGATION-layer `BatchStarkProof` (the fold of two descriptor leaves, carrying
//! the W24 segment-digest sponge + `expose_claim` table) is re-verified IN-CIRCUIT at
//! the next layer. The failing proof is non-stationary across re-proving runs, so we
//! FREEZE it once (two such children were captured into `tests/fixtures/agg_child_<N>.bin`)
//! and replay ONLY the recursive `verify_p3_batch_proof_circuit` + witness runner on the
//! frozen bytes — deterministic, ~0.2s, no re-proving.
//!
//! WHAT THE FROZEN REPLAY ESTABLISHED (root-causing, 2026-06-25):
//!   * The failure is NOT the height-1 FRI `ro==0` bucketing the handoff named: the
//!     height-1 (`log_blowup`) reduced opening IS correctly zero, the FRI fold result
//!     equals the proof's committed `final_poly`, and the per-batch `ro0==0` assert
//!     matches native.
//!   * The WitnessConflict (`value [231554781, 850667020, 1803444624, 1888773549] -> 0`,
//!     byte-identical to the real ~65s fold) is the **LogUp / WitnessChecks GLOBAL
//!     CUMULATIVE** check — `lookup_gadget.verify_global_final_value_circuit`
//!     (`recursion/src/verifier/batch_stark.rs`, the per-instance global-cumulative loop)
//!     for instance 0 of the aggregation child: its global cumulative is nonzero but must
//!     be 0.
//!   * `BatchStarkProver::verify_all_tables` (NATIVE, independent impl) REJECTS the same
//!     frozen child with `GlobalCumulativeMismatch: WitnessChecks` — confirming the
//!     cumulative is genuinely nonzero, i.e. the aggregation layer EMITS a child proof
//!     whose global LogUp bus does not balance to zero (likely the W24 segment-digest /
//!     `expose_claim` lookup contributions are unaccounted). This is the precise hand-off
//!     for the next investigator: fix the aggregation-child global-LogUp balance (NOT FRI).
//!
//! This test asserts the conflict still REPRODUCES (a live repro/regression). Once the
//! global-LogUp balance is fixed, flip the assertion to require a clean replay.

mod common;

use std::path::PathBuf;

use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    PermConfig, Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_circuit_prover::{
    BatchStarkProver, ConstraintProfile, ExposeClaimProver, Poseidon2Prover, RecomposeProver,
    TableProver,
};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_lookup::logup::LogUpGadget;
use p3_poseidon2_circuit_air::{BabyBearD4Width16, BabyBearD4Width24};
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_recursion::{Poseidon2Config as RecPoseidon2Config, VerificationError};
use p3_uni_stark::StarkConfig;

use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16, default_babybear_poseidon2_24};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

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

/// The `ir2_leaf_wrap_config` knobs the dregg aggregation layers run under:
/// log_blowup=6, log_final_poly_len=0, commit_pow=0, query_pow=16. num_queries
/// (19) + folding arity are read from the proof in-circuit; for verification only
/// log_blowup + pow + the perm/MMCS matter for Fiat-Shamir parity.
fn ir2_replay_config() -> MyConfig {
    let perm = default_babybear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 6,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 19,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
    MyConfig::new(pcs, Challenger::new(perm))
}

fn ir2_replay_fri_params() -> FriVerifierParams {
    FriVerifierParams::with_mmcs(
        6,
        0,
        0,
        16,
        PermConfig::poseidon2(Poseidon2Config::BABY_BEAR_D4_W16),
    )
}

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
    cb.enable_expose_claim::<F>(p3_circuit::ops::generate_expose_claim_trace::<F, Challenge>);
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Replay ONE frozen aggregation-child proof through the recursive verifier circuit.
/// Returns Ok(()) if the verifier circuit's witness solves, Err on the height-1
/// WitnessConflict (or any verify error).
fn replay_one(path: &std::path::Path) -> Result<(), VerificationError> {
    let bytes = std::fs::read(path).expect("read fixture");
    let proof: p3_circuit_prover::BatchStarkProof<MyConfig> =
        postcard::from_bytes(&bytes).expect("decode frozen BatchStarkProof");
    assert_eq!(proof.ext_degree, D, "frozen proof ext_degree");

    let config = ir2_replay_config();

    // Rebuild the lookups-bearing CommonData from the proof (serde drops lookups).
    let mut np_prover = BatchStarkProver::new(config.clone());
    np_prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    np_prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
    np_prover.register_recompose_table::<D>(false);
    np_prover.register_expose_claim_table::<D>();
    let common = np_prover
        .rebuild_verifiable_common::<D>(&proof, proof.w_binomial)
        .expect("rebuild verifiable common");

    // Sanity: does NATIVE accept this frozen proof? (The recursion must match native.)
    match np_prover.verify_all_tables(&proof) {
        Ok(()) => eprintln!(
            "[frozen-replay] {:?}: NATIVE verify_all_tables ACCEPTS",
            path.file_name().unwrap()
        ),
        Err(e) => eprintln!(
            "[frozen-replay] {:?}: NATIVE verify_all_tables REJECTS: {e:?}",
            path.file_name().unwrap()
        ),
    }

    let mut cb = CircuitBuilder::new();
    enable_dregg_tables(&mut cb);

    let fri_params = ir2_replay_fri_params();
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
        Box::new(RecomposeProver::<D>::new(1, false)),
        Box::new(ExposeClaimProver::<D>::new()),
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
        &proof,
        &fri_params,
        &common,
        &lookup_gadget,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
        &verif_provers,
    )?;

    let verification_circuit = cb.build().unwrap();
    let mut runner = verification_circuit.runner();

    let inner_pis: Vec<Vec<F>> = proof
        .non_primitives
        .iter()
        .map(|e| e.public_values.clone())
        .collect();
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&inner_pis, &proof.proof, &common);
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    set_fri_mmcs_private_data::<F, Challenge, ChallengeMmcs, MyMmcs, MyHash, MyCompress, DIGEST_ELEMS>(
        &mut runner,
        &op_ids,
        &proof.proof.opening_proof,
        RecPoseidon2Config::BABY_BEAR_D4_W16,
    )
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;

    let _ = runner.run().map_err(VerificationError::Circuit)?;
    Ok(())
}

/// Replay EVERY frozen aggregation child; report which conflict. The frozen culprit
/// is the regression oracle for the height-1 bucketing fix: after the fix, ALL
/// children must replay cleanly.
#[test]
fn frozen_agg_children_replay() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_test_writer()
        .try_init();

    let dir = fixtures_dir();
    let mut children: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("agg_child_") && n.ends_with(".bin"))
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    children.sort();

    if children.is_empty() {
        eprintln!(
            "[frozen-replay] no fixtures in {dir:?}; produce them with:\n  \
             cd ../breadstuffs && RUSTFLAGS=\"--cfg ivc_freeze\" \
             IVC_FREEZE_DIR=\"{dir:?}\" cargo test --release -p dregg-circuit-prove \
             --test ivc_turn_chain_rotated k_fold_turn_chain_proves_and_verifies -- --ignored"
        );
        return;
    }

    let mut any_conflict = false;
    for path in &children {
        let r = replay_one(path);
        match &r {
            Ok(()) => eprintln!("[frozen-replay] {:?}: OK", path.file_name().unwrap()),
            Err(e) => {
                any_conflict = true;
                eprintln!("[frozen-replay] {:?}: CONFLICT/ERR: {e:?}", path.file_name().unwrap());
            }
        }
    }
    // Once the bucketing fix lands, NO child conflicts (flip the assertion's polarity
    // to a clean-replay guard at that point).
    assert!(
        any_conflict,
        "expected at least one frozen aggregation child to reproduce the height-1 \
         WitnessConflict (the residual). If none did, the bug is fixed — flip this assert."
    );
}
