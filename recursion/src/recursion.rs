//! Unified recursion API: one entry point to prove the next layer over a uni-stark or batch-stark proof.

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{SymbolicExpression, SymbolicExpressionExt};
use p3_batch_stark::{CommonData, ProverData};
use p3_circuit::ops::{Poseidon2Config, Poseidon2PermCall};
use p3_circuit::symbolic::ColumnsTargets;
use p3_circuit::tables::Traces;
use p3_circuit::types::ExprId;
use p3_circuit::{
    Circuit, CircuitBuilder, CircuitBuilderError, CircuitRunner, NonPrimitiveOpId, NpoTypeId,
};
use p3_circuit_prover::batch_stark_prover::TableProver;
use p3_circuit_prover::common::{NpoAirBuilder, NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::config::StarkField;
use p3_circuit_prover::field_params::ExtractBinomialW;
use p3_circuit_prover::{
    AirVariant, BatchStarkProof, BatchStarkProver, CircuitProverData, ConstraintProfile,
    TablePacking,
};
use p3_commit::Pcs;
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, Field, PrimeField64};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{Lookup, LookupData, LookupProtocol};
use p3_uni_stark::{Proof, StarkGenericConfig, Val};
use tracing::instrument;

use crate::traits::{LookupMetadata, RecursiveAir};
use crate::types::RecursiveLagrangeSelectors;
use crate::verifier::VerificationError;
use crate::{ChallengerPermConfig, Target};

fn proof_shape_err(e: &impl ToString) -> VerificationError {
    VerificationError::InvalidProofShape(e.to_string())
}

/// Fingerprint for the compiled verification [`Circuit`].
///
/// This is used to reject [`AggregationPrepCache`] hits when a new aggregation step builds a different
/// circuit (e.g. verifying proofs from a different recursion depth that reuses a different layout even if
/// `ProveNextLayerParams` and `config` match).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AggregationCircuitFingerprint {
    pub witness_count: u32,
    pub public_flat_len: usize,
    pub private_flat_len: usize,
    pub ops_len: usize,
}

const fn aggregation_circuit_fingerprint<F>(circuit: &Circuit<F>) -> AggregationCircuitFingerprint {
    AggregationCircuitFingerprint {
        witness_count: circuit.witness_count,
        public_flat_len: circuit.public_flat_len,
        private_flat_len: circuit.private_flat_len,
        ops_len: circuit.ops.len(),
    }
}

pub struct AggregationPrepCache<SC: StarkGenericConfig + 'static> {
    pub circuit_fingerprint: AggregationCircuitFingerprint,
    pub circuit_prover_data: Rc<CircuitProverData<SC>>,
    pub prover: BatchStarkProver<SC>,
}

/// Input to one recursion step: either a uni-stark proof or a batch-stark proof (with common data).
pub enum RecursionInput<'a, SC, A>
where
    SC: StarkGenericConfig,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
{
    /// A single-instance STARK proof (e.g. from p3-uni-stark) plus its AIR and public inputs.
    UniStark {
        proof: &'a Proof<SC>,
        air: &'a A,
        public_inputs: Vec<Val<SC>>,
        preprocessed_commit: Option<<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment>,
    },
    /// A batch STARK proof (e.g. from p3-batch-stark / circuit-prover) plus common data and per-table public inputs.
    BatchStark {
        proof: &'a BatchStarkProof<SC>,
        common_data: &'a CommonData<SC>,
        table_public_inputs: Vec<Vec<Val<SC>>>,
        /// **VK-IDENTITY PIN (in-band, lever (a)).** When `Some(commitment)`, the parent
        /// aggregation circuit adds an in-circuit constraint that the child proof's preprocessed
        /// commitment (its verifier-key core — the Merkle cap binding the child verifier circuit's
        /// static op-list) EQUALS this expected commitment. The child's preprocessed commitment is
        /// already allocated as parent-circuit public-input targets (`MerkleCapTargets`, observed in
        /// the transcript + used for the child's preprocessed-trace FRI check), but without this pin
        /// its VALUE is unconstrained — a from-scratch prover could fold a proof of a DIFFERENT
        /// circuit. With the pin, the cap targets are `connect`ed to constants of `commitment`, so a
        /// foreign-circuit child (different preprocessed commitment) makes the parent circuit UNSAT.
        ///
        /// This is the IVC self-verification fixed-point hook: pass the running circuit's OWN fixed
        /// preprocessed commitment to assert "I am folding a proof produced by THE SAME running
        /// circuit." `None` preserves the legacy (unpinned) behaviour. Ignored for the primitive
        /// tables (which carry no preprocessed commitment); applies to the single shared preprocessed
        /// Merkle cap the child's `CommonData` exposes.
        expected_preprocessed_commit:
            Option<<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment>,
    },
    /// A **native** [`p3_batch_stark::BatchProof`] (NOT the circuit-prover wrapper) over a
    /// CALLER-SUPPLIED AIR set `airs`, plus the symbolic `CommonData` and per-table public
    /// inputs. Used to fold a batch proved directly by `p3_batch_stark::prove_batch` over an
    /// arbitrary multi-table AIR set (e.g. dregg's IR-v2 descriptor batch) as a recursion
    /// leaf — the leaf's in-circuit constraint evaluation is `A::eval_folded_circuit` per
    /// instance, NOT the fixed [`crate::verifier::CircuitTablesAir`] reconstruction the
    /// circuit-prover `BatchStark` arm performs.
    NativeBatchStark {
        airs: &'a [A],
        proof: &'a p3_batch_stark::BatchProof<SC>,
        common_data: &'a CommonData<SC>,
        table_public_inputs: Vec<Vec<Val<SC>>>,
    },
}

/// Output of one recursion step: the next-layer batch proof and its prover data (for chaining or verification).
pub struct RecursionOutput<SC>(pub BatchStarkProof<SC>, pub Rc<CircuitProverData<SC>>)
where
    SC: StarkGenericConfig;

impl<SC> RecursionOutput<SC>
where
    SC: StarkGenericConfig,
{
    /// The GENUINE per-table public inputs of this proof, in instance order (the same order the
    /// in-circuit `BatchStark` verifier allocates public-input targets from
    /// `proof.non_primitives[].public_values.len()`).
    ///
    /// **Lever (b): in-circuit public-input THREADING.** The primitive tables (Const / Public / Alu)
    /// carry no public values; each non-primitive table carries its `public_values`. Threading these
    /// as `table_public_inputs` (rather than empty vectors) makes the next aggregation layer's packed
    /// public vector MATCH the public-input targets it allocates — so a child proof whose
    /// non-primitive tables expose public values (e.g. an aggregation proof being RE-folded) is
    /// re-verified with its publics bound IN-CIRCUIT, instead of leaving allocated target slots
    /// unfilled (which over-constrains the witness solver — the `WitnessConflict` the empty-vector
    /// path trips when re-folding a proof that itself contains a fold).
    pub fn genuine_table_public_inputs(&self) -> Vec<Vec<Val<SC>>> {
        let num_primitive =
            p3_circuit_prover::batch_stark_prover::NUM_PRIMITIVE_TABLES;
        let mut tpi: Vec<Vec<Val<SC>>> = vec![vec![]; num_primitive];
        for entry in &self.0.non_primitives {
            tpi.push(entry.public_values.clone());
        }
        debug_assert_eq!(
            tpi.len(),
            self.0.proof.opened_values.instances.len(),
            "table_public_inputs must have one entry per proof instance"
        );
        tpi
    }

    /// Convert this output into a `RecursionInput::BatchStark` for the next recursion layer, THREADING
    /// this proof's genuine per-table public inputs (lever (b)) so they are re-verified in-circuit at
    /// the next layer. The type parameter `A` is only used for the recursion input type; use
    /// `BatchOnly` when chaining batch-to-batch (see [`BatchOnly`]).
    pub fn into_recursion_input<A>(&self) -> RecursionInput<'_, SC, A>
    where
        A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    {
        RecursionInput::BatchStark {
            proof: &self.0,
            common_data: &self.0.stark_common,
            table_public_inputs: self.genuine_table_public_inputs(),
            expected_preprocessed_commit: None,
        }
    }

    /// Like [`into_recursion_input`](Self::into_recursion_input), but PINS the child's VK identity
    /// in-band (lever (a)): the parent aggregation circuit will constrain the child proof's
    /// preprocessed commitment to equal `expected_preprocessed_commit`.
    ///
    /// Pass the running circuit's OWN fixed preprocessed commitment to assert the IVC
    /// self-verification fixed-point ("I fold a proof from the SAME running circuit"). A child whose
    /// preprocessed commitment differs (a proof of a different circuit) makes the parent UNSAT.
    ///
    /// Obtain the expected commitment from a reference running proof via
    /// [`running_preprocessed_commit`](Self::running_preprocessed_commit).
    pub fn into_recursion_input_pinned<A>(
        &self,
        expected_preprocessed_commit: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
    ) -> RecursionInput<'_, SC, A>
    where
        A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    {
        RecursionInput::BatchStark {
            proof: &self.0,
            common_data: &self.0.stark_common,
            table_public_inputs: self.genuine_table_public_inputs(),
            expected_preprocessed_commit: Some(expected_preprocessed_commit),
        }
    }

    /// The child proof's preprocessed commitment (its VK-identity core), if it has preprocessed
    /// columns. This is the value to pin across an IVC fold: extract it ONCE from the running
    /// circuit's reference proof, then pass it to
    /// [`into_recursion_input_pinned`](Self::into_recursion_input_pinned) at every fold step to
    /// assert the running circuit's identity stays constant.
    pub fn running_preprocessed_commit(
        &self,
    ) -> Option<<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment> {
        self.0
            .stark_common
            .preprocessed
            .as_ref()
            .map(|gp| gp.commitment.clone())
    }
}

/// Result of building a verifier circuit: holds enough to pack public inputs and set private data.
pub trait VerifierCircuitResult<SC, A>
where
    SC: StarkGenericConfig,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
{
    /// Pack the public inputs for the verifier circuit from the previous recursion input.
    fn pack_public_inputs(
        &self,
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<Vec<SC::Challenge>, VerificationError>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>;

    /// Pack the private inputs (opened values, FRI siblings, etc.) for the verifier circuit.
    fn pack_private_inputs(
        &self,
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<Vec<SC::Challenge>, VerificationError>
    where
        Val<SC>: PrimeField64,
        SC::Challenge: BasedVectorSpace<Val<SC>> + From<Val<SC>>;

    /// Operation IDs that require private data (e.g. Merkle paths) for the circuit runner.
    fn op_ids(&self) -> &[NonPrimitiveOpId];

    /// Per-instance public-input targets allocated for this verified child.
    ///
    /// `air_public_targets()[i]` is the list of public-input targets for the
    /// child's instance `i` (primitive tables first, then non-primitive tables
    /// in proof order). These are exactly the values bound to the child proof by
    /// the in-circuit verifier, so re-exposing them carries a claim up a layer.
    /// A `UniStark` child returns a single instance.
    fn air_public_targets(&self) -> Vec<Vec<crate::Target>>;
}

/// PCS-specific backend for building verifier circuits and setting private data.
pub trait PcsRecursionBackend<SC, A, const D: usize>
where
    SC: StarkGenericConfig,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
{
    /// Opaque verifier result returned by `build_verifier_circuit`.
    type VerifierResult: VerifierCircuitResult<SC, A>;

    /// Prepare the circuit before building the verifier (e.g. enable challenger permutation and NPOs). Called before `build_verifier_circuit`.
    fn prepare_circuit(
        &self,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<(), VerificationError>;

    /// Build the verifier circuit for the given recursion input; add constraints to `circuit`.
    fn build_verifier_circuit(
        &self,
        prev: &RecursionInput<'_, SC, A>,
        config: &SC,
        circuit: &mut CircuitBuilder<SC::Challenge>,
    ) -> Result<Self::VerifierResult, VerificationError>;

    /// Set PCS-specific private data (e.g. FRI Merkle paths) on the runner.
    fn set_private_data(
        &self,
        config: &SC,
        runner: &mut CircuitRunner<'_, SC::Challenge>,
        op_ids: &[NonPrimitiveOpId],
        prev: &RecursionInput<'_, SC, A>,
    ) -> Result<(), &'static str>;

    /// Challenger permutation config for the in-circuit verifier (e.g. for Fiat–Shamir). Default none.
    fn challenger_perm_config(&self) -> Option<Box<dyn ChallengerPermConfig>> {
        None
    }

    /// Non-primitive preprocessors for this extension degree (e.g. for NPOs that need preprocessing).
    fn non_primitive_preprocessors(&self) -> Vec<Box<dyn NpoPreprocessor<Val<SC>>>> {
        Vec::new()
    }

    /// Non-primitive table provers for the given extension degree.
    /// Default returns empty; backends that use NPOs in the circuit override this.
    fn non_primitive_provers(&self, _ext_degree: usize) -> Vec<Box<dyn TableProver<SC>>> {
        Vec::new()
    }

    /// AIR builders for NPOs from preprocessed data.
    fn non_primitive_air_builders(&self) -> Vec<Box<dyn NpoAirBuilder<SC, D>>> {
        Vec::new()
    }
}

/// Parameters for the shared recursion pipeline (table packing, optional overrides).
#[derive(Clone, Debug)]
pub struct ProveNextLayerParams {
    pub table_packing: TablePacking,
    /// Constraint profile controlling which AIR variants are used for this layer.
    pub constraint_profile: ConstraintProfile,
}

impl Default for ProveNextLayerParams {
    fn default() -> Self {
        Self {
            table_packing: TablePacking::new(1, 4),
            constraint_profile: ConstraintProfile::Standard,
        }
    }
}

/// Marker type for batch-only recursion input. Use with [`RecursionOutput::into_recursion_input`]
/// when chaining batch-to-batch layers (e.g. `output.into_recursion_input::<BatchOnly>()`).
#[derive(Debug)]
pub struct BatchOnly;

impl<F: Field, EF: ExtensionField<F>, LG: LookupProtocol> RecursiveAir<F, EF, LG> for BatchOnly {
    fn width(&self) -> usize {
        0
    }

    fn eval_folded_circuit(
        &self,
        builder: &mut CircuitBuilder<EF>,
        _sels: &RecursiveLagrangeSelectors,
        _alpha: &Target,
        _lookup_metadata: &LookupMetadata<'_, F>,
        _columns: ColumnsTargets<'_>,
        _lookup_gadget: &LG,
    ) -> Target {
        builder.define_const(EF::ZERO)
    }

    fn get_log_num_quotient_chunks(
        &self,
        _preprocessed_width: usize,
        _contexts: &[Lookup<F>],
        _lookup_data: &[LookupData<usize>],
        _is_zk: usize,
        _lookup_gadget: &LG,
    ) -> usize {
        0
    }

    fn uses_main_next_row(&self) -> bool {
        false
    }

    fn uses_preprocessed_next_row(&self) -> bool {
        false
    }
}

/// Preprocessed prover data for a fixed verification circuit shape, produced offline by
/// [`build_next_layer_prep`].
///
/// Pass this to [`prove_next_layer`] via `prep` to skip the overhead of LDEs and Merkle tree
/// construction that would otherwise run on every call. This is safe to reuse across layers
/// because `generate_preprocessed_columns` is purely a function of the circuit's static
/// op-list — it does not depend on runtime witness values.
///
/// Requires that the same config (including ZK seed, if using `HidingFriPcs`) is used for
/// every `prove_next_layer` call that reuses this cache.
pub struct NextLayerPrepCache<SC: StarkGenericConfig + 'static> {
    pub circuit_prover_data: Rc<CircuitProverData<SC>>,
    pub prover: BatchStarkProver<SC>,
}

/// Build a verifier circuit for a recursion layer.
#[instrument(skip_all)]
pub fn build_next_layer_circuit<SC, A, B, const D: usize>(
    prev: &RecursionInput<'_, SC, A>,
    config: &SC,
    backend: &B,
) -> Result<(Circuit<SC::Challenge>, B::VerifierResult), VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    build_next_layer_circuit_with_expose::<SC, A, B, D>(prev, config, backend, None)
}

/// Hook invoked after a single child's verifier constraints are built (and
/// before the circuit is finalized), receiving that child's per-instance
/// `air_public_targets`. Used to re-expose chain claims at a leaf wrap.
pub type NextLayerExposeHook<'a, F> = &'a dyn Fn(&mut CircuitBuilder<F>, &[Vec<crate::Target>]);

/// Like [`build_next_layer_circuit`], but invokes `expose` (if any) on the
/// builder after the child verifier constraints are emitted, so the caller can
/// add an exposed-claim table over the child's `air_public_targets`.
pub fn build_next_layer_circuit_with_expose<SC, A, B, const D: usize>(
    prev: &RecursionInput<'_, SC, A>,
    config: &SC,
    backend: &B,
    expose: Option<NextLayerExposeHook<'_, SC::Challenge>>,
) -> Result<(Circuit<SC::Challenge>, B::VerifierResult), VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let mut circuit_builder = CircuitBuilder::new();
    backend.prepare_circuit(config, &mut circuit_builder)?;

    // Build verifier constraints.
    let verifier_result = backend.build_verifier_circuit(prev, config, &mut circuit_builder)?;

    if let Some(expose) = expose {
        let apt = verifier_result.air_public_targets();
        expose(&mut circuit_builder, &apt);
    }

    let verification_circuit = circuit_builder
        .build()
        .map_err(VerificationError::CircuitBuilder)?;

    Ok((verification_circuit, verifier_result))
}

/// Offline step: commit to preprocessed columns for a fixed verification circuit shape.
///
/// The resulting [`NextLayerPrepCache`] can be reused across many [`prove_next_layer`] calls
/// that share the same circuit shape. Since `generate_preprocessed_columns` depends only on
/// the circuit's static op-list (not on runtime witness values), the commitment is valid for
/// every proof with the same verification circuit structure.
///
/// **Important**: if using `HidingFriPcs` (ZK mode), the same config (including PCS seed)
/// must be used for every `prove_next_layer` call that reuses this cache, because the
/// preprocessed commitment is bound to the PCS randomness.
#[instrument(skip_all)]
pub fn build_next_layer_prep<SC, A, B, const D: usize>(
    verification_circuit: &Circuit<SC::Challenge>,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
) -> Result<NextLayerPrepCache<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let (airs_degrees, primitive_columns, non_primitive_columns) = {
        let preprocessors = backend.non_primitive_preprocessors();
        let air_builders = backend.non_primitive_air_builders();
        get_airs_and_degrees_with_prep::<SC, SC::Challenge, D>(
            verification_circuit,
            &params.table_packing,
            &preprocessors,
            &air_builders,
            params.constraint_profile,
        )
        .map_err(VerificationError::Circuit)?
    };

    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let ext_degrees: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();

    let prover_data = ProverData::from_airs_and_degrees(config, &airs, &ext_degrees);
    let circuit_prover_data = Rc::new(CircuitProverData::new(
        prover_data,
        primitive_columns,
        non_primitive_columns,
    ));

    let mut prover = BatchStarkProver::new(config.clone())
        .with_table_packing(params.table_packing.clone())
        .with_alu_variant(match params.constraint_profile {
            ConstraintProfile::Standard => AirVariant::Baseline,
            ConstraintProfile::RecursionOptimized => AirVariant::Optimized,
        });
    for p in backend.non_primitive_provers(D) {
        prover.register_table_prover(p);
    }

    Ok(NextLayerPrepCache {
        circuit_prover_data,
        prover,
    })
}

/// Prove one recursion layer: run the verifier circuit and prove it with batch STARK.
///
/// Pass a [`NextLayerPrepCache`] produced by [`build_next_layer_prep`] to skip the LDE and
/// Merkle-tree commitment for preprocessed columns on every layer.
#[instrument(skip_all)]
pub fn prove_next_layer<SC, A, B, const D: usize>(
    prev: &RecursionInput<'_, SC, A>,
    verification_circuit: &Circuit<SC::Challenge>,
    verifier_result: &B::VerifierResult,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
    prep: Option<&NextLayerPrepCache<SC>>,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    if let Some(cached) = prep {
        let traces = {
            let public_inputs = verifier_result.pack_public_inputs(prev)?;
            let private_inputs = verifier_result.pack_private_inputs(prev)?;
            let mut runner = verification_circuit.runner();
            runner
                .set_public_inputs(&public_inputs)
                .map_err(VerificationError::Circuit)?;
            runner
                .set_private_inputs(&private_inputs)
                .map_err(VerificationError::Circuit)?;
            backend
                .set_private_data(config, &mut runner, verifier_result.op_ids(), prev)
                .map_err(|e| proof_shape_err(&e))?;
            runner.run().map_err(VerificationError::Circuit)?
        };
        let proof = cached
            .prover
            .prove_all_tables(&traces, &cached.circuit_prover_data)
            .map_err(|e| proof_shape_err(&e.to_string()))?;
        return Ok(RecursionOutput(
            proof,
            Rc::clone(&cached.circuit_prover_data),
        ));
    }

    let (airs_degrees, primitive_columns, non_primitive_columns) = {
        let preprocessors = backend.non_primitive_preprocessors();
        let air_builders = backend.non_primitive_air_builders();
        get_airs_and_degrees_with_prep::<SC, SC::Challenge, D>(
            verification_circuit,
            &params.table_packing,
            &preprocessors,
            &air_builders,
            params.constraint_profile,
        )
        .map_err(VerificationError::Circuit)?
    };

    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let ext_degrees: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();

    let traces = {
        let public_inputs = verifier_result.pack_public_inputs(prev)?;
        let private_inputs = verifier_result.pack_private_inputs(prev)?;
        let mut runner = verification_circuit.runner();
        runner
            .set_public_inputs(&public_inputs)
            .map_err(VerificationError::Circuit)?;
        runner
            .set_private_inputs(&private_inputs)
            .map_err(VerificationError::Circuit)?;

        backend
            .set_private_data(config, &mut runner, verifier_result.op_ids(), prev)
            .map_err(|e| proof_shape_err(&e))?;

        runner.run().map_err(VerificationError::Circuit)?
    };

    let circuit_prover_data = {
        let prover_data = ProverData::from_airs_and_degrees(config, &airs, &ext_degrees);
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns)
    };

    let mut prover = BatchStarkProver::new(config.clone())
        .with_table_packing(params.table_packing.clone())
        .with_alu_variant(match params.constraint_profile {
            ConstraintProfile::Standard => AirVariant::Baseline,
            ConstraintProfile::RecursionOptimized => AirVariant::Optimized,
        });
    for p in backend.non_primitive_provers(D) {
        prover.register_table_prover(p);
    }
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .map_err(|e| proof_shape_err(&e.to_string()))?;

    Ok(RecursionOutput(proof, Rc::new(circuit_prover_data)))
}

/// Convenience wrapper that calls [`build_next_layer_circuit`] then [`prove_next_layer`] without a prep cache.
pub fn build_and_prove_next_layer<SC, A, B, const D: usize>(
    prev: &RecursionInput<'_, SC, A>,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    build_and_prove_next_layer_with_expose::<SC, A, B, D>(prev, config, backend, params, None)
}

/// Like [`build_and_prove_next_layer`], but with an [`NextLayerExposeHook`] that
/// can add an exposed-claim table over the verified child's `air_public_targets`.
pub fn build_and_prove_next_layer_with_expose<SC, A, B, const D: usize>(
    prev: &RecursionInput<'_, SC, A>,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
    expose: Option<NextLayerExposeHook<'_, SC::Challenge>>,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let (verification_circuit, verifier_result) =
        build_next_layer_circuit_with_expose::<SC, A, B, D>(prev, config, backend, expose)?;

    prove_next_layer::<SC, A, B, D>(
        prev,
        &verification_circuit,
        &verifier_result,
        config,
        backend,
        params,
        None,
    )
}

/// The canonical proof shape that `normalize_to_shape` (Pickles `step ∘ wrap`)
/// targets: the FIXED multiset of non-primitive table TYPES every normalized proof
/// must carry, each at a FIXED lane count, plus the global minimum trace height all
/// tables pad up to.
///
/// Two proofs normalized to the SAME `CanonicalShapeSpec` produce a BYTE-IDENTICAL
/// recursive-verifier op-list / [`AggregationCircuitFingerprint`] regardless of their
/// real table content: the HEIGHT axis (`min_trace_height` pads every table to a
/// fixed power-of-two height) and the MANIFEST axis (every proof carries the same
/// ordered non-primitive type set at fixed lanes) are both pinned. That invariance
/// is exactly what lets ONE static VK verify any child of this shape. Both axes are
/// de-risked green in `recursion/tests/normalize_to_shape_spike.rs`
/// (`fingerprint_invariant_under_fixed_shape_padding` for height,
/// `fingerprint_invariant_under_fixed_manifest_padding` for the manifest).
#[derive(Clone, Debug)]
pub struct CanonicalShapeSpec {
    /// log2 of the fixed trace height every table is padded to (`min_trace_height = 1 << log_height`).
    pub log_height: usize,
    /// Fixed primitive public-table lane count.
    pub public_lanes: usize,
    /// Fixed primitive ALU-table lane count.
    pub alu_lanes: usize,
    /// The canonical non-primitive manifest: each `(type, lanes)` is one table TYPE
    /// EVERY normalized proof carries, at its fixed lane count. A child that does not
    /// naturally use a canonical type is padded with a MINIMAL SATISFIABLE instance of
    /// it (e.g. a one-claim `expose_claim`, or a self-balanced minimal
    /// poseidon2/recompose) injected at child-build time.
    pub canonical_npos: Vec<(NpoTypeId, usize)>,
}

impl CanonicalShapeSpec {
    /// Build the [`TablePacking`] that pins this canonical shape: the fixed
    /// `min_trace_height` plus every canonical NPO's fixed lane override. This is the
    /// real, load-bearing canonical-shape construction — it is what makes both the
    /// height and manifest axes invariant.
    pub fn to_table_packing(&self) -> TablePacking {
        let mut packing = TablePacking::new(self.public_lanes, self.alu_lanes)
            .with_min_trace_height(1usize << self.log_height);
        for (op_type, lanes) in &self.canonical_npos {
            packing = packing.with_npo_lanes(op_type.clone(), *lanes);
        }
        packing
    }

    /// The canonical layer params (pinned packing + standard profile).
    pub fn to_params(&self) -> ProveNextLayerParams {
        ProveNextLayerParams {
            table_packing: self.to_table_packing(),
            constraint_profile: ConstraintProfile::Standard,
        }
    }

    /// The canonical depth-1 manifest — the SINGLE source of truth for the ordered
    /// non-primitive type set a depth-1 fixed VK pins: `[poseidon2_perm, recompose,
    /// expose_claim]`, each at `lanes` lanes. These three are exactly the canonical
    /// types proven lookup-balanced as minimal fillers AND fingerprint-invariant
    /// (a uses-all-three child vs a padded-from-none child compile to a byte-identical
    /// recursive-verifier op-list) by the D=4 full-manifest tests in
    /// `recursion/tests/normalize_to_shape_spike.rs`. `inject_canonical_fillers` pads
    /// an absent type to exactly this manifest at child-build time.
    pub fn depth1_default(
        log_height: usize,
        public_lanes: usize,
        alu_lanes: usize,
        poseidon2_config: Poseidon2Config,
        lanes: usize,
    ) -> Self {
        Self {
            log_height,
            public_lanes,
            alu_lanes,
            canonical_npos: alloc::vec![
                (NpoTypeId::poseidon2_perm(poseidon2_config), lanes),
                (NpoTypeId::recompose(), lanes),
                (NpoTypeId::expose_claim(), lanes),
            ],
        }
    }
}

/// Inject a minimal SELF-BALANCED filler for every canonical non-primitive type in
/// `canonical_npos` that is ABSENT from `present`, at child-BUILD time, so the child
/// lands on the full canonical manifest a depth-1 fixed VK re-verifies.
///
/// This is the REAL `normalize_to_shape` padding op: the constructions below are the
/// ones proven lookup-balanced (each filler proves with the lookup debugger ON, and
/// the padded tables are genuinely present in `proof.non_primitives`) AND
/// fingerprint-invariant against a child that uses all three types for real, by the
/// D=4 full-manifest tests in `recursion/tests/normalize_to_shape_spike.rs`.
///
/// Per canonical type (each self-balanced on the buses it touches):
/// * `poseidon2_perm` — one permutation over two `Const` rate inputs with `out_ctl`
///   all false. A permutation CTL-RECEIVES its rate inputs and CTL-SENDS its rate
///   outputs; with no CTL'd output it sends nothing on the hash bus and only receives
///   the consts (balanced by the Const table's send). Self-balanced, no companion.
/// * `recompose` — `decompose_ext_to_base_coeffs(bind)` emits exactly one recompose
///   row whose EF-output SEND is `connect`-bound straight back to `bind`, so it is
///   consumed by `bind`'s existing readers. Self-balanced when `bind` is a genuine
///   NON-CONST extension witness (a const `bind` would const-fold away the row).
/// * `expose_claim` — `expose_as_public_output(&[bind])`: the `WitnessChecks` read is
///   balanced by the `PublicAir` send.
///
/// Contract: `builder` must already have the canonical NPOs ENABLED, and `bind` must
/// be a genuine already-present NON-CONST extension witness of the child (e.g. a
/// public input or a hash output) for the recompose + expose fillers to bind to. A
/// CONST `bind` const-folds the decompose away and emits NO recompose table, leaving
/// the requested canonical entry absent — so `bind` must be a real witness.
///
/// This function OWNS the `set_recompose_coeff_ctl_for_decompose_links` switch for the
/// duration of its recompose fillers: it pins the route per canonical type
/// (`recompose` ⇒ OFF, `recompose/coeff` ⇒ ON) so the emitted table is exactly the
/// requested one regardless of the builder's incoming flag state, and leaves the flag
/// at its constructor default (OFF) on return. A caller that needs the coeff route for
/// its own later ops must set it again after this call.
pub fn inject_canonical_fillers<CF, BF>(
    builder: &mut CircuitBuilder<CF>,
    poseidon2_config: Poseidon2Config,
    canonical_npos: &[(NpoTypeId, usize)],
    present: &BTreeSet<&str>,
    bind: ExprId,
) -> Result<(), CircuitBuilderError>
where
    CF: Field + ExtensionField<BF>,
    BF: PrimeField64,
{
    for (op_type, _) in canonical_npos {
        let name = op_type.as_str();
        if present.contains(name) {
            continue;
        }
        if name.starts_with("poseidon2_perm") {
            let c0 = builder.alloc_const(CF::from_u64(7), "canonical_filler_p2_in0");
            let c1 = builder.alloc_const(CF::from_u64(11), "canonical_filler_p2_in1");
            let mut inputs = alloc::vec![None; poseidon2_config.width_ext()];
            inputs[0] = Some(c0);
            inputs[1] = Some(c1);
            builder.add_poseidon2_perm(&Poseidon2PermCall {
                config: poseidon2_config,
                new_start: true,
                merkle_path: false,
                mmcs_bit: None,
                inputs,
                out_ctl: alloc::vec![false; poseidon2_config.rate_ext()],
                return_all_outputs: false,
                mmcs_index_sum: None,
            })?;
        } else if name == "recompose" {
            // Plain `recompose`: decompose reconnects via the standard recompose table.
            // Force the coeff-CTL route OFF so an incoming `true` flag cannot silently
            // turn this filler into a `recompose/coeff` table (the symmetric footgun of
            // the `recompose/coeff` case below). Restore the default (OFF) after.
            builder.set_recompose_coeff_ctl_for_decompose_links(false);
            let coeffs = builder.decompose_ext_to_base_coeffs::<BF>(bind)?;
            debug_assert!(
                !coeffs.is_empty(),
                "decompose must emit a non-empty coefficient vector"
            );
        } else if name == "recompose/coeff" {
            // `decompose_ext_to_base_coeffs` emits a `recompose/coeff` table ONLY when
            // the coeff-CTL route is enabled; with the default builder state it silently
            // emits a plain `recompose` instead, leaving the requested `recompose/coeff`
            // canonical entry ABSENT. Set the flag OURSELVES so the caller cannot get a
            // silent wrong table, then restore the default (OFF).
            //
            // The authoritative confirmation that the EMITTED op is `recompose/coeff`
            // (not `recompose`) is the proof manifest: a child built through this path
            // carries a `recompose/coeff` entry in `proof.non_primitives` (asserted in
            // `recursion/tests/normalize_to_shape_spike.rs`). There is no public builder
            // accessor to introspect the just-emitted op type in-place; the flag pin is
            // what makes the path deterministic, and `bind` being a genuine non-const
            // witness (the contract above) is what makes a table emit at all.
            builder.set_recompose_coeff_ctl_for_decompose_links(true);
            let coeffs = builder.decompose_ext_to_base_coeffs::<BF>(bind)?;
            builder.set_recompose_coeff_ctl_for_decompose_links(false);
            debug_assert!(
                !coeffs.is_empty(),
                "decompose must emit a non-empty coefficient vector"
            );
        } else if name == "expose_claim" {
            builder.expose_as_public_output(&[bind]);
        }
    }
    Ok(())
}

/// Verify a [`BatchStarkProof`] matches EXACTLY the canonical shape `spec` pins, so the
/// recursion VK this proof is later checked against is the canonical one.
///
/// The in-circuit batch-STARK verifier ([`crate::verifier::verify_p3_batch_proof_circuit`])
/// is DATA-DRIVEN by the proof: it reads `proof.table_packing` (public/ALU lanes, min trace
/// height, Horner packing, per-NPO lanes) and walks the full ordered `proof.non_primitives`
/// manifest, pushing one `CircuitTablesAir::Dynamic` AND one public-input slot per entry.
/// So ANY divergence from the canonical shape — an EXTRA non-primitive table, a MISSING
/// canonical table, a WRONG lane count, or different packing metadata — produces a DIFFERENT
/// verifier op-list and hence a DIFFERENT VK. A name-only "all required types are present"
/// check is therefore NOT sound: a proof with an extra table or wrong lanes passes it yet
/// mismatches the canonical VK.
///
/// This is the fail-closed gate. It pins exactly the spec-derivable, VK-driving axes:
/// 1. `proof.table_packing == spec.to_table_packing()` — the packing the verifier reads.
/// 2. `proof.non_primitives`, as an exact MULTISET of `(op_type, lanes)`, equals
///    `spec.canonical_npos` — no extra, no missing, no wrong-lane entry.
///
/// RESIDUAL (precisely): the verifier op-list ALSO depends on per-entry quantities the
/// current [`CanonicalShapeSpec`] does not encode — each entry's `public_values.len()`
/// (it pushes one `air_public_counts` slot of that width) and `air_variant`, plus the
/// proof-global `alu_variant` / `alu_quintic_trinomial` / `w_binomial`. For the canonical
/// fillers these are fixed by construction (the `poseidon2`/`recompose` fillers expose 0
/// public values, the `expose_claim` filler exposes 1), but a proof that matches the
/// manifest+packing yet carries, e.g., a 5-claim `expose_claim` would pass THIS check and
/// still induce a different VK. Fully closing that gap needs `CanonicalShapeSpec` to also
/// carry the canonical per-entry public-value counts (and air variants), or this check to
/// compare against a stored canonical verifier-circuit fingerprint / VK. The
/// logical-row-count axis is NOT a gap: the spike's fixed-height invariance tests show the
/// verifier op-list is invariant to primitive/NPO logical row counts once the height
/// (`min_trace_height`, pinned by axis 1) is fixed.
pub fn check_canonical_shape<SC>(
    proof: &BatchStarkProof<SC>,
    spec: &CanonicalShapeSpec,
) -> Result<(), VerificationError>
where
    SC: StarkGenericConfig,
{
    // Axis 1: the primitive packing the verifier reads must match the canonical packing.
    // We compare the VK-driving SCALARS explicitly (rather than `TablePacking == `): the
    // stored `public_lanes`/`alu_lanes` are the EFFECTIVE (post prove-time clamp) values
    // the verifier actually uses, and `npo_lanes` is an order-sensitive Vec covered by the
    // per-entry lane check in axis 2 — so a scalar comparison is both order-free and the
    // honest VK pin. The verifier builds `PublicAir::new(rows, public_lanes)` /
    // `AluAir::…(alu_lanes, …, horner)` and pads every table to `min_trace_height`.
    let canonical_packing = spec.to_table_packing();
    let p = &proof.table_packing;
    if p.public_lanes() != canonical_packing.public_lanes()
        || p.alu_lanes() != canonical_packing.alu_lanes()
        || p.min_trace_height() != canonical_packing.min_trace_height()
        || p.horner_packed_steps() != canonical_packing.horner_packed_steps()
    {
        return Err(proof_shape_err(&alloc::format!(
            "prev packing (public_lanes={}, alu_lanes={}, min_trace_height={}, \
             horner_packed_steps={}) does not match the canonical packing (public_lanes={}, \
             alu_lanes={}, min_trace_height={}, horner_packed_steps={}); the verifier \
             op-list (VK) is driven by the packing, so an off-canonical packing would be \
             checked against a different VK",
            p.public_lanes(),
            p.alu_lanes(),
            p.min_trace_height(),
            p.horner_packed_steps(),
            canonical_packing.public_lanes(),
            canonical_packing.alu_lanes(),
            canonical_packing.min_trace_height(),
            canonical_packing.horner_packed_steps(),
        )));
    }

    // Axis 2: the non-primitive manifest must be EXACTLY the canonical multiset of
    // (op_type, lanes). Length-equal + an injective consume ⇒ a bijection, so this rejects
    // any extra table, any missing canonical table, and any wrong-lane entry.
    if proof.non_primitives.len() != spec.canonical_npos.len() {
        return Err(proof_shape_err(&alloc::format!(
            "prev carries {} non-primitive table(s) but the canonical manifest pins {}; \
             run `inject_canonical_fillers` at child-build time so the child lands on the \
             exact canonical manifest before normalizing",
            proof.non_primitives.len(),
            spec.canonical_npos.len()
        )));
    }
    let mut remaining: Vec<(NpoTypeId, usize)> = spec.canonical_npos.clone();
    for entry in &proof.non_primitives {
        match remaining
            .iter()
            .position(|(t, l)| *t == entry.op_type && *l == entry.lanes)
        {
            Some(pos) => {
                remaining.swap_remove(pos);
            }
            None => {
                return Err(proof_shape_err(&alloc::format!(
                    "prev carries non-primitive table `{}` at {} lane(s), which is not in \
                     the canonical manifest at that lane count (extra table / wrong lanes); \
                     a mismatched manifest yields a different verifier VK",
                    entry.op_type, entry.lanes
                )));
            }
        }
    }
    debug_assert!(remaining.is_empty(), "bijection: remaining must be empty");
    Ok(())
}

/// Build + prove ONE normalization layer: verify `prev` with a recursion circuit whose
/// shape is pinned to the canonical `spec`, producing a next-layer proof in canonical
/// shape so a single fixed VK can verify it (the Pickles `step ∘ wrap` normalization).
///
/// The canonical-shape construction is REAL: [`CanonicalShapeSpec::to_params`] pins the
/// fixed height (`min_trace_height`) and the fixed per-type manifest lanes, and the
/// recursion verifier circuit is built + proved against that pinned packing via the
/// existing [`build_and_prove_next_layer`] pipeline.
///
/// REQUIREMENT: `prev` must ALREADY be a proof built in canonical shape — i.e. a child
/// constructed with [`inject_canonical_fillers`] (the child-BUILD-time op that pads every
/// absent canonical NPO type with a minimal self-balanced instance) and proved against
/// `spec.to_table_packing()`. Manifest padding is a child-build-time operation, not a
/// fold-time one: a finished proof's table set is immutable, so this layer (which receives
/// `prev` as an already-proved [`BatchStarkProof`]) CANNOT add a missing table to it.
///
/// This layer is therefore FAIL-CLOSED: it runs the exact-shape precondition
/// [`check_canonical_shape`] against `prev`, which REJECTS any proof that is missing a
/// canonical table, carries an extra non-primitive table, has a wrong lane count, or has
/// off-canonical packing — anything that would not land on the canonical VK. A proof that
/// would silently mismatch the fixed VK is rejected here with a loud `Err`, never folded.
/// See [`check_canonical_shape`] for the precisely-stated residual (per-entry
/// `public_values.len()` / `air_variant` are not yet spec-encoded).
pub fn build_and_prove_normalization_layer<SC, A, B, const D: usize>(
    prev: &RecursionInput<'_, SC, A>,
    config: &SC,
    backend: &B,
    spec: &CanonicalShapeSpec,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    // The real canonical-shape construction: fixed height + fixed manifest lanes.
    let params = spec.to_params();

    // EXACT-SHAPE precondition (fail-closed): `prev` must already carry the canonical
    // packing AND the exact canonical non-primitive manifest. The absent-type filler is
    // injected at child-build time by `inject_canonical_fillers` (an already-proved
    // proof's table set is immutable — see the function doc); here we REJECT any proof
    // that does not land on the canonical shape the fixed VK expects, so a mismatched
    // proof can never be folded against the wrong VK.
    if let RecursionInput::BatchStark { proof, .. } = prev {
        check_canonical_shape::<SC>(proof, spec)?;
    }

    build_and_prove_next_layer::<SC, A, B, D>(prev, config, backend, &params)
}

/// Build a 2-to-1 aggregation layer verifier circuit.
///
/// The two inputs may be different `RecursionInput` variants (e.g. one `UniStark` left
/// and one `BatchStark` right) or identical ones.
#[instrument(skip_all)]
#[allow(clippy::type_complexity)]
#[allow(dead_code, clippy::type_complexity)]
fn build_aggregation_layer_circuit<SC, A1, A2, B, const D: usize>(
    left: &RecursionInput<'_, SC, A1>,
    right: &RecursionInput<'_, SC, A2>,
    config: &SC,
    backend: &B,
) -> Result<
    (
        Circuit<SC::Challenge>,
        (
            <B as PcsRecursionBackend<SC, A1, D>>::VerifierResult, // left
            <B as PcsRecursionBackend<SC, A2, D>>::VerifierResult, // right
        ),
    ),
    VerificationError,
>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A1: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    A2: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A1, D> + PcsRecursionBackend<SC, A2, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    build_aggregation_layer_circuit_with_expose::<SC, A1, A2, B, D>(
        left, right, config, backend, None,
    )
}

/// Hook invoked after both children's verifier constraints are built (and before
/// the aggregation circuit is finalized), receiving the LEFT and RIGHT children's
/// per-instance `air_public_targets`. Used to re-expose + connect-bind chain
/// claims one layer up the fold.
pub type AggExposeHook<'a, F> =
    &'a dyn Fn(&mut CircuitBuilder<F>, &[Vec<crate::Target>], &[Vec<crate::Target>]);

/// Like [`build_aggregation_layer_circuit`], but invokes `expose` (if any) on the
/// builder after both child verifiers are emitted, receiving the left and right
/// `air_public_targets`.
#[allow(clippy::type_complexity)]
fn build_aggregation_layer_circuit_with_expose<SC, A1, A2, B, const D: usize>(
    left: &RecursionInput<'_, SC, A1>,
    right: &RecursionInput<'_, SC, A2>,
    config: &SC,
    backend: &B,
    expose: Option<AggExposeHook<'_, SC::Challenge>>,
) -> Result<
    (
        Circuit<SC::Challenge>,
        (
            <B as PcsRecursionBackend<SC, A1, D>>::VerifierResult, // left
            <B as PcsRecursionBackend<SC, A2, D>>::VerifierResult, // right
        ),
    ),
    VerificationError,
>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A1: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    A2: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A1, D> + PcsRecursionBackend<SC, A2, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let mut circuit_builder = CircuitBuilder::new();

    <B as PcsRecursionBackend<SC, A1, D>>::prepare_circuit(backend, config, &mut circuit_builder)?;
    <B as PcsRecursionBackend<SC, A2, D>>::prepare_circuit(backend, config, &mut circuit_builder)?;

    // Build left verifier constraints.
    let left_result = backend.build_verifier_circuit(left, config, &mut circuit_builder)?;
    // Build right verifier constraints into the same builder.
    let right_result = backend.build_verifier_circuit(right, config, &mut circuit_builder)?;

    if let Some(expose) = expose {
        let left_apt = left_result.air_public_targets();
        let right_apt = right_result.air_public_targets();
        expose(&mut circuit_builder, &left_apt, &right_apt);
    }

    let verification_circuit = circuit_builder
        .build()
        .map_err(VerificationError::CircuitBuilder)?;

    Ok((verification_circuit, (left_result, right_result)))
}

fn run_aggregation_verification_circuit<SC, A1, A2, B, const D: usize>(
    left: &RecursionInput<'_, SC, A1>,
    right: &RecursionInput<'_, SC, A2>,
    left_result: &<B as PcsRecursionBackend<SC, A1, D>>::VerifierResult,
    right_result: &<B as PcsRecursionBackend<SC, A2, D>>::VerifierResult,
    verification_circuit: &Circuit<SC::Challenge>,
    config: &SC,
    backend: &B,
) -> Result<Traces<SC::Challenge>, VerificationError>
where
    SC: StarkGenericConfig,
    A1: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    A2: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A1, D> + PcsRecursionBackend<SC, A2, D>,
    Val<SC>: PrimeField64,
{
    let mut public_inputs = left_result.pack_public_inputs(left)?;
    public_inputs.extend(right_result.pack_public_inputs(right)?);

    let mut private_inputs = left_result.pack_private_inputs(left)?;
    private_inputs.extend(right_result.pack_private_inputs(right)?);

    let mut runner = verification_circuit.runner();
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    <B as PcsRecursionBackend<SC, A1, D>>::set_private_data(
        backend,
        config,
        &mut runner,
        left_result.op_ids(),
        left,
    )
    .map_err(|e| proof_shape_err(&e.to_string()))?;

    <B as PcsRecursionBackend<SC, A2, D>>::set_private_data(
        backend,
        config,
        &mut runner,
        right_result.op_ids(),
        right,
    )
    .map_err(|e| proof_shape_err(&e.to_string()))?;

    runner.run().map_err(VerificationError::Circuit)
}

/// Prove a 2-to-1 aggregation layer: build verifier circuits for both `left` and `right`
/// in a single circuit, run it, and produce one aggregated batch STARK proof.
///
/// When proving multiple pairs that compile to the **same** verification [`Circuit`] fingerprint
/// and use the same `ProveNextLayerParams` / `config`, pass `prep_cache: Some(&mut None)` on the
/// first call; the slot is filled and can be passed again for later pairs to skip
/// [`get_airs_and_degrees_with_prep`]. If the fingerprint changes (different proof structure,
/// etc.), the cache is ignored automatically.
///
/// The two inputs may be different `RecursionInput` variants (e.g. one `UniStark` left
/// and one `BatchStark` right) or identical ones.
#[instrument(skip_all)]
#[allow(clippy::too_many_arguments)]
pub fn prove_aggregation_layer<SC, A1, A2, B, const D: usize>(
    left: &RecursionInput<'_, SC, A1>,
    right: &RecursionInput<'_, SC, A2>,
    left_result: &<B as PcsRecursionBackend<SC, A1, D>>::VerifierResult,
    right_result: &<B as PcsRecursionBackend<SC, A2, D>>::VerifierResult,
    verification_circuit: &Circuit<SC::Challenge>,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
    mut prep_cache: Option<&mut Option<AggregationPrepCache<SC>>>,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A1: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    A2: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A1, D> + PcsRecursionBackend<SC, A2, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let current_fp = aggregation_circuit_fingerprint(verification_circuit);
    if let Some(ref mut cache_slot) = prep_cache
        && let Some(cached) = cache_slot.as_ref()
        && cached.circuit_fingerprint == current_fp
    {
        let traces = run_aggregation_verification_circuit::<SC, A1, A2, B, D>(
            left,
            right,
            left_result,
            right_result,
            verification_circuit,
            config,
            backend,
        )?;
        let proof = cached
            .prover
            .prove_all_tables(&traces, &cached.circuit_prover_data)
            .map_err(|e| proof_shape_err(&e.to_string()))?;
        return Ok(RecursionOutput(
            proof,
            Rc::clone(&cached.circuit_prover_data),
        ));
    }

    let (airs_degrees, primitive_columns, non_primitive_columns) = {
        let preprocessors =
            <B as PcsRecursionBackend<SC, A1, D>>::non_primitive_preprocessors(backend);
        let air_builders =
            <B as PcsRecursionBackend<SC, A1, D>>::non_primitive_air_builders(backend);
        get_airs_and_degrees_with_prep::<SC, SC::Challenge, D>(
            verification_circuit,
            &params.table_packing,
            &preprocessors,
            &air_builders,
            params.constraint_profile,
        )
        .map_err(VerificationError::Circuit)?
    };

    let (airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
    let ext_degrees: Vec<usize> = degrees.iter().map(|&d| d + config.is_zk()).collect();

    let traces = run_aggregation_verification_circuit::<SC, A1, A2, B, D>(
        left,
        right,
        left_result,
        right_result,
        verification_circuit,
        config,
        backend,
    )?;

    let circuit_prover_data = {
        let prover_data = ProverData::from_airs_and_degrees(config, &airs, &ext_degrees);
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns)
    };

    let mut prover = BatchStarkProver::new(config.clone())
        .with_table_packing(params.table_packing.clone())
        .with_alu_variant(match params.constraint_profile {
            ConstraintProfile::Standard => AirVariant::Baseline,
            ConstraintProfile::RecursionOptimized => AirVariant::Optimized,
        });
    for p in <B as PcsRecursionBackend<SC, A1, D>>::non_primitive_provers(backend, D) {
        prover.register_table_prover(p);
    }
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .map_err(|e| proof_shape_err(&e.to_string()))?;

    if let Some(ref mut cache_slot) = prep_cache {
        let circuit_prover_data_rc = Rc::new(circuit_prover_data);
        **cache_slot = Some(AggregationPrepCache {
            circuit_fingerprint: current_fp,
            circuit_prover_data: Rc::clone(&circuit_prover_data_rc),
            prover,
        });
        Ok(RecursionOutput(proof, circuit_prover_data_rc))
    } else {
        Ok(RecursionOutput(proof, Rc::new(circuit_prover_data)))
    }
}

/// Convenience method to build and prove a 2-to-1 aggregation layer.
///
/// The two inputs may be different `RecursionInput` variants (e.g. one `UniStark` left
/// and one `BatchStark` right) or identical ones.
///
/// In production environments, consider using [`prove_aggregation_layer`] directly for better performance.
///
/// # Example
///
/// ```ignore
/// let (verification_circuit, (left_result, right_result)) = build_aggregation_layer_circuit::<SC, A1, A2, B, D>(left, right, config, backend)?;
/// let out = prove_aggregation_layer::<SC, A1, A2, B, D>(..., params, None);
/// ```
pub fn build_and_prove_aggregation_layer<SC, A1, A2, B, const D: usize>(
    left: &RecursionInput<'_, SC, A1>,
    right: &RecursionInput<'_, SC, A2>,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
    prep_cache: Option<&mut Option<AggregationPrepCache<SC>>>,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A1: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    A2: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A1, D> + PcsRecursionBackend<SC, A2, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    build_and_prove_aggregation_layer_with_expose::<SC, A1, A2, B, D>(
        left, right, config, backend, params, prep_cache, None,
    )
}

/// Like [`build_and_prove_aggregation_layer`], but with an [`AggExposeHook`] that
/// can re-expose + connect-bind chain claims one layer up the fold.
#[allow(clippy::too_many_arguments)]
pub fn build_and_prove_aggregation_layer_with_expose<SC, A1, A2, B, const D: usize>(
    left: &RecursionInput<'_, SC, A1>,
    right: &RecursionInput<'_, SC, A2>,
    config: &SC,
    backend: &B,
    params: &ProveNextLayerParams,
    prep_cache: Option<&mut Option<AggregationPrepCache<SC>>>,
    expose: Option<AggExposeHook<'_, SC::Challenge>>,
) -> Result<RecursionOutput<SC>, VerificationError>
where
    SC: StarkGenericConfig + Send + Sync + Clone + 'static,
    A1: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    A2: RecursiveAir<Val<SC>, SC::Challenge, LogUpGadget>,
    B: PcsRecursionBackend<SC, A1, D> + PcsRecursionBackend<SC, A2, D>,
    Val<SC>: PrimeField64 + StarkField,
    SC::Challenge: BasedVectorSpace<Val<SC>>
        + From<Val<SC>>
        + ExtensionField<Val<SC>>
        + ExtractBinomialW<Val<SC>>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let (verification_circuit, (left_result, right_result)) =
        build_aggregation_layer_circuit_with_expose::<SC, A1, A2, B, D>(
            left, right, config, backend, expose,
        )?;

    prove_aggregation_layer::<SC, A1, A2, B, D>(
        left,
        right,
        &left_result,
        &right_result,
        &verification_circuit,
        config,
        backend,
        params,
        prep_cache,
    )
}
