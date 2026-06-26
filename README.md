# Plonky3-recursion

Plonky3 native support for recursive STARK verification, enabling proof composition and multi-layer recursion.

---

## About this fork

This is **dregg's fork** of [`Plonky3/Plonky3-recursion`](https://github.com/Plonky3/Plonky3-recursion). The
upstream library gives you a recursive STARK verifier-in-a-circuit. dregg needed
something on top of that: **IVC of arbitrary histories** — folding a long chain of
verified state transitions into one proof that a whole-history light client can
check and then *read claims out of*. The pieces below are the things we needed to
make that work, and that upstream didn't have. We built them as we hit the walls;
that's just how it was for us.

None of this is framed as an upstream contribution. It's our own honest record of
what we added and why. Where a piece turned out to be genuinely general we've
offered it upstream as a freely-decline-able PR (noted below); the rest is
dregg-shaped and lives here because that's where it belongs.

The reassessment in [`upstream-pr/REASSESSMENT-CAPABILITY-VS-INSTANTIATION.md`](upstream-pr/REASSESSMENT-CAPABILITY-VS-INSTANTIATION.md)
is the per-item breakdown of which parts are general capability vs. dregg
instantiation; this section is the short version.

### What we built on top, and why we needed it

- **A verified public-output channel** — `expose_as_public_output(targets)` on the
  `CircuitBuilder`, backed by the `ExposeClaim` op and its `ExposeClaimAir`.
  ([`circuit/src/ops/expose_claim.rs`](circuit/src/ops/expose_claim.rs),
  [`circuit/src/builder/circuit_builder.rs`](circuit/src/builder/circuit_builder.rs),
  [`circuit-prover/src/air/expose_claim_air.rs`](circuit-prover/src/air/expose_claim_air.rs).)
  It takes arbitrary in-circuit witnesses and surfaces them as *bound* public
  outputs of the proof — the AIR is a pure `WitnessChecks`-bus reader, so the host
  reads back the genuine verified witnesses, not free prover-chosen scalars.
  **Why:** a light client folding a history needs to read things *out* of the fold
  — chain heads, state roots, accumulator state, a turn counter. Upstream can
  verify a fold but had no way to expose anything bound from inside it. This is the
  keystone. We offer the abstracted capability upstream as **PR #453** (open, take
  it or leave it).

- **An in-circuit VK-identity pin** — `pin_preprocessed_commit` plus the
  `expected_preprocessed_commit` field on `RecursionInput`.
  ([`recursion/src/verifier/batch_stark.rs`](recursion/src/verifier/batch_stark.rs),
  [`recursion/src/recursion.rs`](recursion/src/recursion.rs).)
  It connects the child proof's preprocessed-commitment cap to expected constants
  in-circuit, so a fold of a *different* circuit makes the parent unsatisfiable.
  **Why:** IVC only closes into a genuine fixed point if "the one running circuit
  verifies its own previous proof" is actually pinned — otherwise the chain is
  open-ended and a from-scratch prover could fold a foreign child. This is the
  stable anchor VK a light client needs to trust the chain.

- **A native-batch leaf-wrap** — `RecursionInput::NativeBatchStark` and
  `verify_p3_native_batch_proof_circuit`.
  ([`recursion/src/recursion.rs`](recursion/src/recursion.rs),
  [`recursion/src/verifier/batch_stark.rs`](recursion/src/verifier/batch_stark.rs),
  [`recursion/src/backend/fri.rs`](recursion/src/backend/fri.rs).)
  A thin sibling to `verify_p3_batch_proof_circuit` that folds a *bare*
  `p3_batch_stark::BatchProof` over a **caller's own AIR set**, rather than only
  proofs shaped like the circuit-prover's internal table model. **Why:** our base
  layer is a real batch-STARK over our own AIRs; without this entry we could only
  recurse over circuit-prover-shaped proofs. (The heavy lifting,
  `verify_batch_circuit`, was already upstream — this is the glue we needed.)

- **A lowerer `assert_zero` fix** — don't alias `connect(value, const)` into the
  constant's witness slot. (Lives on the `fix/lowerer-assert-zero-slot-collapse`
  branch, commit `0438d52`.) When a value-bearing witness collided with the
  circuit-wide `ZERO` slot, two ops became creators of the same witness and
  unbalanced the global LogUp bus. **Why:** our IVC circuits hit this in the wild
  and it made otherwise-correct folds reject. We offered it as **PR #452**, which
  the owner *closed* — a maintainer's minimality critique was right, so it stays a
  fork-local note rather than an upstream change.

- **The constant-VK fixed-point / threaded publics** — the IVC orchestration that
  keeps one circuit's VK constant across an unbounded fold and re-exposes public
  inputs (head root, chain digest, turn count) across aggregation layers, plus the
  `build_and_prove_*_layer_with_expose` plumbing and the `D=4/2/5` plugin
  registrations. **Why:** this is the dregg-specific shape — *which* witnesses we
  expose and *which* VK we pin — assembled out of the general pieces above. It's
  the most fork-shaped part and is not offered upstream.

### Honest status

This fork's additions are **AI-developed and semi-unaudited**. They work in our
testing — the IVC chain folds, the exposed outputs come back bound, the VK pin
makes foreign children reject — but they have **not** been independently reviewed,
and several touch soundness-relevant accounting (witness-slot creation, bus
multiplicities, in-circuit VK binding). Treat them as "works in our testing," not
"proven correct." The upstream caveat below applies here in full, doubly so for
these pieces.

The general capabilities are offered upstream where they're genuinely useful
(PR #453 open for the public-output channel; PR #452 closed for the lowerer fix);
everything dregg-shaped lives here, and that's fine.

---

## Overview

This library provides a **fixed recursive verifier** for Plonky3 STARK (both `p3-uni-stark` and `p3-batch-stark` proofs), allowing you to verify proofs inside circuits and compose proofs recursively. The recursive verifier is implemented as a circuit itself, which can be proven and verified in subsequent layers.

### Key Features

- **Recursive STARK Verification**: Verify Plonky3 STARK proofs inside circuits
- **Batch STARK Support**: Verify multiple proofs in a single batch
- **Modular Circuit Builder**: Build circuits with primitive operations (add, mul, etc.) and non-primitive operations (Poseidon2)
- **FRI-based PCS**: Full support for FRI (Fast Reed-Solomon Interactive) polynomial commitment scheme verification in-circuit
- **ZK support**: Full support for Zero-Knowledge.

## Production Use

**⚠️ This codebase is under active development and hasn't been audited yet.** As such, we do not recommend its use in any production software.

[![Coverage](https://github.com/Plonky3/Plonky3-recursion/actions/workflows/coverage.yml/badge.svg)](https://plonky3.github.io/Plonky3-recursion/coverage/) (_updated weekly_)

## Quick Start

### Unified recursion API

#### Recursive verification

For most use cases, use the **unified API**: a single entry point works for both uni-stark (e.g. Keccak) and batch-stark (e.g. Fibonacci) proofs. Implement [`FriRecursionConfig`] for your config (or a wrapper that holds FRI verifier params), then call [`prove_next_layer`] in a loop.

[`FriRecursionConfig`]: https://docs.rs/p3-recursion/latest/p3_recursion/trait.FriRecursionConfig.html
[`prove_next_layer`]: https://docs.rs/p3-recursion/latest/p3_recursion/fn.prove_next_layer.html

```rust
use p3_recursion::{
    FriRecursionBackend, FriRecursionConfig, ProveNextLayerParams, RecursionInput, RecursionOutput,
    prove_next_layer,
};

// First layer: recurse on a uni-stark proof (e.g. Keccak)
let input = RecursionInput::UniStark {
    proof: &base_proof,
    air: &keccak_air,
    public_inputs: pis.clone(),
    preprocessed_commit: None,
};
let backend = FriRecursionBackend::new(poseidon2_config);
let params = ProveNextLayerParams { table_packing, use_poseidon2_in_circuit: true };
let (verification_circuit, verifier_result) = build_next_layer_circuit(&input, &config, &backend)?;
let output = prove_next_layer(
    &input,
    &verification_circuit,
    &verifier_result,
    &config,
    &backend,
    &params,
)?;

// Next layers: recurse on the previous batch proof
let input = output.into_recursion_input::<BatchOnly>();
let (verification_circuit, verifier_result) = build_next_layer_circuit(&input, &config, &backend)?;
let output = prove_next_layer(
    &input,
    &verification_circuit,
    &verifier_result,
    &config,
    &backend,
    &params,
)?;
```

`RecursionOutput` is `(BatchStarkProof, CircuitProverData)`; use `into_recursion_input::<BatchOnly>()` to chain further layers.

#### Recursive aggregation

This library also supports 2-to-1 recursive aggregation by building circuits verifying possibly different circuits (for instance 
`RecursionInput::UniStark` as left child and `RecursionInput::BatchStark` as right child).

```rust
let input_1 = RecursionInput::UniStark {
    proof: &base_proof_1,
    air: &air_1,
    public_inputs: pis_1.clone(),
    preprocessed_commit: None,
};
let input_2 = RecursionInput::UniStark {
    proof: &base_proof_2,
    air: &air_2,
    public_inputs: pis_2.clone(),
    preprocessed_commit: None,
};

let backend = FriRecursionBackend::new(poseidon2_config);
let params = ProveNextLayerParams { table_packing, use_poseidon2_in_circuit: true };
let (verification_circuit, verifier_result_1, verifier_result_2) = build_aggregation_layer_circuit(&input_1, &input_2, &config, &backend)?;
let output = prove_aggregation_layer(
    &input_1,
    &input_2,
    verification_circuit,
    &verifier_result_1,
    &verifier_result_2,
    &config,
    &backend,
    &params,
)?;
```

### Low-level API

For fine-grained control, you can build the verification circuit and run the prover pipeline yourself via `verify_p3_batch_proof_circuit` (batch) or `verify_circuit` (uni-stark):

```rust
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_recursion::public_inputs::BatchStarkVerifierInputsBuilder;
use p3_circuit::CircuitBuilder;

// Build a verification circuit
let mut circuit_builder = CircuitBuilder::new();
circuit_builder.enable_poseidon2_perm::<Config, _>(trace_generator, poseidon2_perm);

let (verifier_inputs, mmcs_op_ids) = verify_p3_batch_proof_circuit::<
    MyConfig,
    HashTargets<F, DIGEST_ELEMS>,
    InputProofTargets<F, Challenge, RecValMmcs<...>>,
    InnerFri,
    LogUpGadget,
    WIDTH,
    RATE,
    TRACE_D,
>(
    &config,
    &mut circuit_builder,
    &batch_stark_proof,
    &fri_verifier_params,
    common_data,
    &lookup_gadget,
    Poseidon2Config::BabyBearD4Width16,
)?;

// Build and run the circuit
let circuit = circuit_builder.build()?;
let mut runner = circuit.runner();

// Pack public inputs using the builder
let public_inputs = verifier_inputs.pack_public_values(&pis, &batch_proof, common_data);
runner.set_public_inputs(&public_inputs)?;

// Set MMCS private data (Merkle paths)
set_fri_mmcs_private_data(&mut runner, &mmcs_op_ids, &proof.opening_proof)?;

let traces = runner.run()?;
```

### Examples

All examples use the unified API (`prove_next_layer`, `RecursionInput`, `FriRecursionBackend`):

- **`recursive_fibonacci.rs`**: Base layer is a batch-stark circuit (Fibonacci); recursive layers use `into_recursion_input::<BatchOnly>()` and `prove_next_layer`.
  ```bash
  cargo run --profile optimized --example recursive_fibonacci -- --field koala-bear --n 1000 --num-recursive-layers 5
  ```

- **`recursive_keccak.rs`**: Base layer is a uni-stark Keccak proof; layer 1 uses `RecursionInput::UniStark`, then further layers use `into_recursion_input::<BatchOnly>()` and `prove_next_layer`.
  ```bash
  cargo run --profile optimized --example recursive_keccak -- --field koala-bear --n 100 --num-recursive-layers 5
  ```

- **`recursive_aggregation.rs`**: Base layer are dummy batch-stark circuits; recursive layers use `into_recursion_input::<BatchOnly>()` and `prove_next_aggregation` to fold 2 proofs into 1.
  ```bash
  cargo run --profile optimized --example recursive_aggregation -- --field koala-bear --num-recursive-layers 4
  ```

All three examples take `--field <koala-bear|baby-bear|goldilocks>`, `--quintic` (KoalaBear only), and `--hash <poseidon2|poseidon1>` (default `poseidon2`):
```bash
cargo run --profile optimized --example recursive_fibonacci -- --field baby-bear --hash poseidon1 --n 1000
```

## API Overview

### Circuit Builder

The `CircuitBuilder<F>` provides a modular API for building circuits:

**Primitive Operations** (always available):
- `define_const(val)` - Add a constant
- `public_input()` - Allocate a public input
- `mul(a, b)` - Multiply two expressions
- `add(a, b)` / `sub(a, b)` - Arithmetic operations
- `connect(a, b)` - Constrain two expressions to be equal

**Non-primitive Operations** (require explicit enablement):
- `enable_poseidon2_perm()` - Enable Poseidon2 permutation operations
- Operations must be explicitly enabled via `enable_poseidon2_perm()` before use

### Public Inputs

Public inputs must be provided in the **exact order** the circuit allocated them. Use the builder APIs to ensure correctness:

```rust
use p3_recursion::public_inputs::PublicInputBuilder;

let mut builder = PublicInputBuilder::new();
builder
    .add_proof_values(proof_values)
    .add_challenge(alpha)
    .add_challenges(betas);
let public_inputs = builder.build();
```

For batch verification, use `BatchStarkVerifierInputsBuilder::pack_public_values()` which handles the packing automatically.

## Architecture

### Components

1. **Circuit Builder** (`p3_circuit`): Expression graph builder with primitive and non-primitive operations
2. **Circuit Prover** (`p3_circuit_prover`): Generates STARK proofs for circuits
3. **Recursive Verifier** (`p3_recursion::verifier`): Verifies STARK proofs inside circuits
4. **FRI PCS** (`p3_recursion::pcs::fri`): FRI polynomial commitment scheme verification in-circuit

### Recursion Flow

1. **Base Layer**: Prove a computation using Plonky3 STARK
2. **Recursive Layer**: Build a verification circuit that checks the base proof
3. **Prove Recursive Layer**: Prove the verification circuit itself
4. **Repeat**: Continue recursion for additional layers

Each recursive layer verifies the previous layer's proof, creating a chain of proofs.

## Performance Considerations

### Table Packing

The `TablePacking` configuration significantly impacts performance.
Choose lane counts that fit best your circuit size:

```rust
TablePacking::new(2, 5)  // public, alu lanes
```

### FRI Parameters

FRI parameters affect proof size and verification cost:
- `log_blowup`: LDE blowup factor (typically 3)
- `max_log_arity`: Maximum folding arity (typically 4)
- `log_final_poly_len`: Final polynomial size (typically 5)
- `query_pow_bits`: PoW bits for query phase (typically 16)

For intermediate recursive layers, consider relaxed parameters (fewer queries, higher PoW bits).

## Build Profiles and Compile Features

### Build profiles

Two custom profiles are defined in the workspace `Cargo.toml`:

| Profile | Based on | Description |
|---------|----------|-------------|
| `optimized` | `release` | Maximum performance: thin LTO, single codegen unit, `opt-level = 3`. Use for all benchmarks and production runs. |
| `profiling` | `release` | Like `release` but with debug symbols (`debug = true`) for CPU profilers (`perf`, Instruments, `samply`). |

```bash
cargo run --profile optimized --example recursive_fibonacci --features parallel
cargo run --profile profiling  --example recursive_fibonacci --features parallel
```

### Compile features

| Crate | Feature | Description |
|-------|---------|-------------|
| `p3-circuit` | `debugging` | Allocation logging — every witness slot records the operation and scope that created it. |
| `p3-circuit` | `profiling` | Operation-count profiling (implies `debugging`) — tracks `add`/`mul`/`const`/NPO counts globally and per named scope via `OpCounts`. |
| `p3-circuit-prover` | `parallel` | Multi-threaded trace generation via Rayon. Strongly recommended for benchmarks and production. |

See [Debugging](https://Plonky3.github.io/Plonky3-recursion/advanced_topics/debugging.html) in the book for full details.

## Current Limitations

- **Fixed Configurations**: Field extensions are currently not fully parametrizable.

## Documentation

Documentation is still incomplete and will be improved over time.

- **[Plonky3 Recursion Book](https://Plonky3.github.io/Plonky3-recursion/)**: Comprehensive walkthrough of the recursion approach
- **API Documentation**: `cargo doc --open` for full API reference
- **Examples**: See `recursion/examples/` for working code

## Modular Circuit Builder

The `CircuitBuilder<F>` supports both primitive and non-primitive operations. Primitive ops like `Const`, `Public`, `Add` are always available.

Non-primitive operations (e.g. Poseidon2 permutations) must be explicitly enabled on the builder before use. Attempting to use a non-primitive operation that hasn't been enabled will result in a runtime error.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
