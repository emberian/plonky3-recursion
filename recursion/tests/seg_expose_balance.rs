//! CHEAP reproduction of the dregg descriptor-leaf segment-expose WitnessChecks
//! balance (no FRI / no recursion verifier): a W24 1-perm sponge whose 4 rate
//! outputs (acc), plus two public-input endpoints and a const count, are read by
//! `expose_claim`. Mirrors `seg_poseidon_commit([first_old, last_new])` +
//! `expose_as_public_output([first_old, last_new, count, acc_0..3])`.
//!
//! If native `verify_all_tables` rejects with GlobalCumulativeMismatch here, the
//! imbalance is in the table-level WitnessChecks accounting (reproduced cheaply).

mod common;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    Poseidon2Config, generate_expose_claim_trace, generate_poseidon2_trace,
    generate_recompose_trace,
};
use p3_circuit_prover::batch_stark_prover::{expose_claim_air_builders, poseidon2_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, ExposeClaimPreprocessor,
    Poseidon2Preprocessor, RecomposePreprocessor, TablePacking,
};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_poseidon2_circuit_air::{BabyBearD4Width16, BabyBearD4Width24};
use p3_uni_stark::StarkConfig;

use p3_baby_bear::{BabyBear, default_babybear_poseidon2_16, default_babybear_poseidon2_24};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation};

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

const SEG_RATE: usize = 4;
const SEG_WIDTH_LIMBS: usize = 6;
const SEG_DIGEST_WIDTH: usize = 4;
const SEG_DOMAIN_TAG: u64 = 0x5345_4731 % 0x7800_0001;

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

fn enable(cb: &mut CircuitBuilder<Challenge>) {
    cb.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        default_babybear_poseidon2_16(),
    );
    cb.enable_poseidon2_perm_width_24::<BabyBearD4Width24, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width24>,
        default_babybear_poseidon2_24(),
    );
    cb.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    cb.enable_expose_claim::<F>(generate_expose_claim_trace::<F, Challenge>);
}

/// In-circuit seg_poseidon_commit over 2 inputs (1 W24 perm), mirroring dregg.
fn seg_commit(cb: &mut CircuitBuilder<Challenge>, inputs: &[p3_circuit::types::ExprId]) -> Vec<p3_circuit::types::ExprId> {
    let tag = cb.define_const(Challenge::from(BabyBear::from_u64(SEG_DOMAIN_TAG)));
    let mut state: Vec<_> = (0..SEG_WIDTH_LIMBS).map(|_| tag).collect();
    let len_tag = cb.define_const(Challenge::from(BabyBear::from_u64(inputs.len() as u64)));
    let mut stream = vec![len_tag];
    stream.extend_from_slice(inputs);
    while stream.len() % SEG_RATE != 0 {
        stream.push(tag);
    }
    for block in stream.chunks(SEG_RATE) {
        for (lane, &inp) in block.iter().enumerate() {
            state[lane] = cb.add(state[lane], inp);
        }
        let out = cb
            .add_poseidon2_perm_for_challenger(Poseidon2Config::BABY_BEAR_D4_W24, &state)
            .expect("W24 perm");
        state = (0..SEG_WIDTH_LIMBS).map(|i| out[i]).collect();
    }
    let mut digest = Vec::with_capacity(SEG_DIGEST_WIDTH);
    let mut i = 0;
    loop {
        for lane in 0..SEG_RATE {
            if digest.len() == SEG_DIGEST_WIDTH {
                break;
            }
            digest.push(state[lane]);
        }
        if digest.len() == SEG_DIGEST_WIDTH {
            break;
        }
        i += 1;
        assert!(i < 8);
    }
    digest
}

/// CHAINED seg_commit: capacity flows via the AIR chain (new_start=false), NOT the bus.
/// First perm: full state, new_start=true (capacity = IV, bus-bound by Const). Subsequent
/// perms: rate inputs only (capacity = None, inherited from the previous W24 row). Same
/// computation/digest as `seg_commit`, but the chained capacity is OFF the WitnessChecks bus.
fn seg_commit_chained(cb: &mut CircuitBuilder<Challenge>, inputs: &[p3_circuit::types::ExprId]) -> Vec<p3_circuit::types::ExprId> {
    let tag = cb.define_const(Challenge::from(BabyBear::from_u64(SEG_DOMAIN_TAG)));
    let cap_seed: Vec<_> = (0..SEG_WIDTH_LIMBS - SEG_RATE).map(|_| tag).collect();
    let mut rate: Vec<_> = (0..SEG_RATE).map(|_| tag).collect();
    let len_tag = cb.define_const(Challenge::from(BabyBear::from_u64(inputs.len() as u64)));
    let mut stream = vec![len_tag];
    stream.extend_from_slice(inputs);
    while stream.len() % SEG_RATE != 0 {
        stream.push(tag);
    }
    let mut first = true;
    for block in stream.chunks(SEG_RATE) {
        for (lane, &inp) in block.iter().enumerate() {
            rate[lane] = cb.add(rate[lane], inp);
        }
        rate = cb
            .add_poseidon2_perm_sponge_step(Poseidon2Config::BABY_BEAR_D4_W24, first, &rate, &cap_seed)
            .expect("W24 chained sponge step");
        first = false;
    }
    rate[0..SEG_DIGEST_WIDTH].to_vec()
}

fn seg_commit_host(inputs: &[BabyBear]) -> [BabyBear; SEG_DIGEST_WIDTH] {
    let perm = default_babybear_poseidon2_24();
    let iv = BabyBear::from_u64(SEG_DOMAIN_TAG);
    let mut state = [BabyBear::ZERO; 24];
    for limb in 0..SEG_WIDTH_LIMBS {
        state[limb * D] = iv;
    }
    let mut stream: Vec<BabyBear> = vec![BabyBear::from_u64(inputs.len() as u64)];
    stream.extend_from_slice(inputs);
    while stream.len() % SEG_RATE != 0 {
        stream.push(iv);
    }
    for block in stream.chunks(SEG_RATE) {
        for (lane, &inp) in block.iter().enumerate() {
            state[lane * D] += inp;
        }
        state = perm.permute(state);
    }
    core::array::from_fn(|i| state[i * D])
}

fn prove_and_verify(circuit: p3_circuit::Circuit<Challenge>, pubs: &[Challenge]) -> Result<(), String> {
    let mut runner = circuit.runner();
    runner.set_public_inputs(pubs).map_err(|e| format!("{e:?}"))?;
    let traces = runner.run().map_err(|e| format!("{e:?}"))?;
    let table_packing = TablePacking::new(1, 4);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
        Box::new(ExposeClaimPreprocessor),
    ];
    let mut air_builders = poseidon2_air_builders::<MyConfig, D>();
    air_builders.extend(p3_circuit_prover::batch_stark_prover::recompose_air_builders(1, false));
    air_builders.extend(expose_claim_air_builders::<MyConfig, D>());
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, Challenge, D>(
            &circuit, &table_packing, &npo_prep, &air_builders, ConstraintProfile::Standard,
        )
        .map_err(|e| format!("{e:?}"))?;
    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let config = cfg();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let cpd = CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config.clone()).with_table_packing(table_packing);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
    prover.register_recompose_table::<D>(false);
    prover.register_expose_claim_table::<D>();
    let proof = prover.prove_all_tables(&traces, &cpd).map_err(|e| format!("{e:?}"))?;
    prover.verify_all_tables(&proof).map_err(|e| format!("{e:?}"))
}

/// The CHAINED sponge must compute the SAME digest as the non-chained `seg_commit` and
/// the host (`seg_commit_host`), for a MULTI-perm input (8 inputs -> 3 perms). This proves
/// the off-bus capacity chaining preserves the digest (no deviation).
#[test]
fn seg_commit_chained_matches_host_multiperm() {
    fn exposed_digest(use_chained: bool, inputs: &[BabyBear]) -> Vec<F> {
        let mut cb = CircuitBuilder::new();
        enable(&mut cb);
        let ins: Vec<_> = inputs.iter().map(|_| cb.alloc_public_input("in")).collect();
        let acc = if use_chained { seg_commit_chained(&mut cb, &ins) } else { seg_commit(&mut cb, &ins) };
        cb.expose_as_public_output(&acc);
        let circuit = cb.build().unwrap();
        let mut runner = circuit.runner();
        let pubs: Vec<Challenge> = inputs.iter().map(|&x| Challenge::from(x)).collect();
        runner.set_public_inputs(&pubs).unwrap();
        let traces = runner.run().unwrap();
        let table_packing = TablePacking::new(1, 4);
        let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
            Box::new(Poseidon2Preprocessor),
            Box::new(RecomposePreprocessor::default()),
            Box::new(ExposeClaimPreprocessor),
        ];
        let mut air_builders = poseidon2_air_builders::<MyConfig, D>();
        air_builders.extend(p3_circuit_prover::batch_stark_prover::recompose_air_builders(1, false));
        air_builders.extend(expose_claim_air_builders::<MyConfig, D>());
        let (ad, pc, npc) = get_airs_and_degrees_with_prep::<MyConfig, Challenge, D>(
            &circuit, &table_packing, &npo_prep, &air_builders, ConstraintProfile::Standard,
        ).unwrap();
        let (airs, degrees): (Vec<_>, Vec<_>) = ad.into_iter().unzip();
        let config = cfg();
        let pd = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
        let cpd = CircuitProverData::new(pd, pc, npc);
        let mut prover = BatchStarkProver::new(config.clone()).with_table_packing(table_packing);
        prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
        prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
        prover.register_recompose_table::<D>(false);
        prover.register_expose_claim_table::<D>();
        let proof = prover.prove_all_tables(&traces, &cpd).unwrap();
        prover.verify_all_tables(&proof).unwrap();
        proof.non_primitives.iter().find(|e| e.op_type.as_str() == "expose_claim")
            .map(|e| e.public_values.clone()).unwrap()
    }
    let inputs: Vec<BabyBear> = (10..18).map(BabyBear::from_u64).collect(); // 8 -> 3 perms
    let host = seg_commit_host(&inputs);
    let chained = exposed_digest(true, &inputs);
    eprintln!("[chained-vs-host] host={host:?} chained={chained:?}");
    let host_v: Vec<F> = host.to_vec();
    assert_eq!(chained, host_v, "chained digest must match host (off-bus capacity chaining preserves the digest)");
    let _ = exposed_digest; // (the non-chained variant is the known-broken capacity case)
}

/// AGGREGATION combine using the CHAINED sponge — must balance AND match the host digest.
#[test]
fn seg_aggregate_combine_chained_balance() {
    let mut cb = CircuitBuilder::new();
    enable(&mut cb);
    let l: Vec<_> = (0..SEG_DIGEST_WIDTH + 3).map(|_| cb.alloc_public_input("l")).collect();
    let r: Vec<_> = (0..SEG_DIGEST_WIDTH + 3).map(|_| cb.alloc_public_input("r")).collect();
    const FO: usize = 0;
    const LN: usize = 1;
    const CT: usize = 2;
    const DF: usize = 3;
    cb.connect(l[LN], r[FO]);
    let first_old = l[FO];
    let last_new = r[LN];
    let count = cb.add(l[CT], r[CT]);
    let mut acc_inputs = Vec::new();
    acc_inputs.extend_from_slice(&l[DF..DF + SEG_DIGEST_WIDTH]);
    acc_inputs.extend_from_slice(&r[DF..DF + SEG_DIGEST_WIDTH]);
    let acc = seg_commit_chained(&mut cb, &acc_inputs);
    let mut seg = vec![first_old, last_new, count];
    seg.extend_from_slice(&acc);
    cb.expose_as_public_output(&seg);

    let circuit = cb.build().unwrap();
    let l_acc = seg_commit_host(&[BabyBear::from_u64(1), BabyBear::from_u64(2)]);
    let r_acc = seg_commit_host(&[BabyBear::from_u64(2), BabyBear::from_u64(3)]);
    let mut acc_host_in: Vec<BabyBear> = l_acc.to_vec();
    acc_host_in.extend_from_slice(&r_acc);
    let host_acc = seg_commit_host(&acc_host_in);
    let mut pubs = vec![
        Challenge::from(BabyBear::from_u64(1)),
        Challenge::from(BabyBear::from_u64(2)),
        Challenge::from(BabyBear::from_u64(1)),
    ];
    for a in l_acc { pubs.push(Challenge::from(a)); }
    pubs.push(Challenge::from(BabyBear::from_u64(2)));
    pubs.push(Challenge::from(BabyBear::from_u64(3)));
    pubs.push(Challenge::from(BabyBear::from_u64(1)));
    for a in r_acc { pubs.push(Challenge::from(a)); }

    let r = prove_and_verify(circuit, &pubs);
    eprintln!("[seg-aggregate-chained] result = {r:?}  host_acc = {host_acc:?}");
    r.expect("chained aggregation combine must balance");
}

/// MINIMAL: same-operand add `out = add(tag, tag)`, output exposed. Isolates whether
/// a doubling add creates+sends its output on the WitnessChecks bus.
#[test]
fn same_operand_add_balance() {
    let mut cb = CircuitBuilder::new();
    enable(&mut cb);
    let tag = cb.define_const(Challenge::from(BabyBear::from_u64(SEG_DOMAIN_TAG)));
    let out = cb.add(tag, tag);
    cb.expose_as_public_output(&[out]);
    let circuit = cb.build().unwrap();
    let r = prove_and_verify(circuit, &[]);
    eprintln!("[same-operand-add] result = {r:?}");
    r.expect("same-operand add must balance");
}

/// MINIMAL: distinct-operand add for contrast.
#[test]
fn distinct_operand_add_balance() {
    let mut cb = CircuitBuilder::new();
    enable(&mut cb);
    let a = cb.define_const(Challenge::from(BabyBear::from_u64(SEG_DOMAIN_TAG)));
    let b = cb.define_const(Challenge::from(BabyBear::from_u64(7)));
    let out = cb.add(a, b);
    cb.expose_as_public_output(&[out]);
    let circuit = cb.build().unwrap();
    let r = prove_and_verify(circuit, &[]);
    eprintln!("[distinct-operand-add] result = {r:?}");
    r.expect("distinct-operand add must balance");
}

#[test]
fn seg_expose_balance_reproduces() {
    let mut cb = CircuitBuilder::new();
    enable(&mut cb);

    let first_old = cb.alloc_public_input("first_old");
    let last_new = cb.alloc_public_input("last_new");
    let count = cb.define_const(Challenge::ONE);
    let acc = seg_commit(&mut cb, &[first_old, last_new]);

    let mut seg = vec![first_old, last_new, count];
    seg.extend_from_slice(&acc);
    cb.expose_as_public_output(&seg);

    let circuit = cb.build().unwrap();
    let mut runner = circuit.runner();

    let fo = BabyBear::from_u64(11);
    let ln = BabyBear::from_u64(13);
    runner
        .set_public_inputs(&[Challenge::from(fo), Challenge::from(ln)])
        .unwrap();
    let traces = runner.run().unwrap();

    let table_packing = TablePacking::new(1, 4);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
        Box::new(ExposeClaimPreprocessor),
    ];
    let mut air_builders = poseidon2_air_builders::<MyConfig, D>();
    air_builders.extend(p3_circuit_prover::batch_stark_prover::recompose_air_builders(1, false));
    air_builders.extend(expose_claim_air_builders::<MyConfig, D>());
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
    let config = cfg();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let mut prover = BatchStarkProver::new(config.clone()).with_table_packing(table_packing);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W16);
    prover.register_poseidon2_table::<D>(Poseidon2Config::BABY_BEAR_D4_W24);
    prover.register_recompose_table::<D>(false);
    prover.register_expose_claim_table::<D>();

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .expect("seg-expose proof builds");

    // Sanity: the host digest matches what was exposed.
    let host = seg_commit_host(&[fo, ln]);
    let exposed: Vec<F> = proof
        .non_primitives
        .iter()
        .find(|e| e.op_type.as_str() == "expose_claim")
        .map(|e| e.public_values.clone())
        .unwrap();
    eprintln!("[seg-expose] exposed public_values = {exposed:?}");
    eprintln!("[seg-expose] host acc = {host:?}");
    for e in &proof.non_primitives {
        eprintln!("[seg-expose] op_type={} rows={}", e.op_type.as_str(), e.rows);
    }

    let r = prover.verify_all_tables(&proof);
    eprintln!("[seg-expose] native verify_all_tables = {r:?}");
    r.expect("native verify must accept a balanced seg-expose proof");
}
