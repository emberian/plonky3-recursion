# Re-assessment: capability-vs-instantiation review of the "stays in our fork" bucket

Status: **analysis only.** Written to the fork's own branch (`update-plonky3-rev`)
for the owner to decide outward action. Nothing pushed; no PR opened. This does
not modify `UPSTREAM-PR-DRAFT.md` — it argues that draft under-scoped the offer.

## The challenge being tested

The prior draft (`UPSTREAM-PR-DRAFT.md`) offered upstream **two** general fixes
(a lowerer `assert_zero` aliasing fix + a `zeta_next` opening-schedule diagnosis)
and bucketed the rest — `expose_claim`, `NativeBatchStark`, the IVC fixed-point /
constant-VK construction — as "application-specific, stays in our fork."

The owner's challenge: *"that dregg-specific stuff is the stuff we needed to do to
have plonky3-recursion work AT ALL, wasn't it?"* — i.e. the draft may have
conflated **"we used it for dregg"** with **"it is meaningful only to dregg."**

The lens applied here, per item: separate the **CAPABILITY** (a general thing any
IVC/recursion user needs that upstream lacks) from the **INSTANTIATION** (our
column layouts, enum variants, `.bin` fixtures, naming, the four specific claims).
A general capability wearing a dregg-flavored shell is **still a general
contribution**, offered as the abstracted capability.

Verdict up front: **the challenge is substantially correct.** Two of the three
bucketed items are general capabilities upstream genuinely lacks, and one of them
(`expose_claim`) is the single biggest tasteful contribution in the fork. The
honest offer is not "one bug-fix" — it is a **multi-PR IVC-enablement offering**.

---

## Verification method

All claims below were checked against `origin/main`
(`b81dc0d`, "chore: bump deps", 2026-06-24), not assumption:

- `git grep` over `origin/main` for each capability under every plausible name.
- Read upstream `recursion/src/recursion.rs` (the `RecursionInput` enum) and
  `recursion/src/verifier/batch_stark.rs` (the batch entry points) in full.
- Diffed the fork's flagged commits (`ccebf66 72ffc56 fc12f23 8d42900 d959ff1`)
  against what upstream already provides.

---

## Item A — `expose_claim` / `expose_as_public_output`  (commits `8d42900`, `d959ff1`)

**What upstream has:** *nothing.* `git grep -ni
'expose|public_output|reveal|as_public_output'` over `origin/main`'s `circuit/`,
`circuit-prover/`, `recursion/` returns **no** mechanism to bind an arbitrary
in-circuit witness to a host-readable **public output** of a recursive proof.
Upstream's `public_values` are verifier **inputs** (values the AIR is checked
against), and `table_public_inputs` flows the same direction. There is no op that
takes an internal witness and *surfaces it outward, provably equal to the genuine
witness.*

**Capability (GENERAL):** a *verified public-output channel for recursive
circuits.* `ExposeClaimAir` is a pure `WitnessChecks`-bus reader: it receives N
chosen witnesses off the bus with reader multiplicity `-1` (keeping the bus
balanced against their creators) and constrains each table public value to equal
the value it read. The host then reads `non_primitives[].public_values` and is
guaranteed they are the *genuine verified witnesses*, not free prover-chosen
scalars. The builder API is `circuit.expose_as_public_output(targets: &[ExprId])`
— it takes **arbitrary `ExprId`s**.

This is a universal IVC need. *Any* folding scheme that wants a light client to
read claims out of an aggregate proof — accumulator state, chain heads, Merkle
roots, step counters, a running hash — needs exactly this primitive. Without it,
plonky3-recursion can verify a fold but cannot **expose** anything bound from
inside it.

**Instantiation (fork-only):** *only the doc examples.* The header narrates
dregg's four claims (`genesis_root`, `final_root`, `num_turns`, `chain_digest`)
and the IVC chain story. The *code* is generic — it reads whatever witnesses the
caller passes. Note commit `d959ff1` already moved the docs toward the general
(full-coeff) design. The dregg-shaped pieces that genuinely stay in the fork are
the *choice* of which four witnesses to expose and the D=4/2/5 plugin
registrations in dregg's orchestration — not this op/AIR.

**Verdict:** the draft was **wrong** to bucket this fork-only. It is a
substantial, general contribution (~1085 LOC: op + AIR + columns + table-prover +
builder method + npo registration). **De-dregg-ification needed:** rewrite the
doc comments to present it as "expose arbitrary witnesses as bound public
outputs" and drop the genesis_root/chain_digest framing; the implementation needs
essentially no change. This is the keystone of an upstream offer, not a leftover.

---

## Item B — `NativeBatchStark` / `verify_p3_native_batch_proof_circuit`  (commits `ccebf66`, `72ffc56`)

**What upstream has:** *most of it, but not the entry point.* Upstream already
exposes the **generic** `verify_batch_circuit<A, …>(config, airs: &[A], …)` —
which takes a caller-supplied AIR slice. But its only public *"verify a proof
object"* entry, `verify_p3_batch_proof_circuit`, is **hardwired** to the
circuit-prover's table model: it reconstructs `CircuitTablesAir` (Const / Public /
Alu primitives + registered non-primitive plugins) from the proof's
`table_packing`/`rows`, and accepts only a
`p3_circuit_prover::batch_stark_prover::BatchStarkProof` (the wrapper). There is
**no** upstream public entry that takes a *bare* `p3_batch_stark::BatchProof` over
a *caller's own AIR set* and folds it.

**Capability (GENERAL, but thin):** "fold an externally-produced batch-STARK over
*your* AIRs as a recursion leaf." The fork's `verify_p3_native_batch_proof_circuit`
is a ~30-line sibling to `verify_p3_batch_proof_circuit`: it allocates
`BatchStarkVerifierInputsBuilder` from the bare proof + caller-supplied
`air_public_counts`, then calls the *existing* generic `verify_batch_circuit` with
the caller's `&[A]`. It is genuinely generic (`A: RecursiveAir`, zero dregg
content). It closes a real usability gap — without it you can only recurse over
proofs shaped exactly like the circuit-prover's internal table model — but the
heavy lifting (`verify_batch_circuit`) is already upstream, so the contribution is
small.

**Instantiation (fork-only):** the `RecursionInput::NativeBatchStark` enum
variant, the three backend routing arms, the `D=4/2/5` wiring, and the
`NativeBatchStark` name are fork plumbing. The *glue function* is the general part.

**The entangled general fix (`72ffc56`):** the "demote a duplicate `Public` to a
bus reader" change (`dup_public_outputs`) is a **general correctness fix**, not
dregg-specific. When a `Public` op's output witness is also asserted `== 0` (or
shares a slot via `connect`/`assert_zero`), both the zero `Const` and the `Public`
op become creators of the same witness, double-sending on the `WitnessChecks` bus
and unbalancing the global LogUp (the fork saw `+779`). The fix enforces
one-creator-per-witness by demoting the duplicate `Public` to a reader. This is
the **same hazard family** as the draft's Item 1 (the `assert_zero` const-aliasing
lowerer fix) — a value-bearing witness colliding with the circuit-wide `ZERO`
slot — handled at a different layer (bus accounting vs. lowering). The draft left
it out as "entangled with Item 1"; it is in fact a complementary general fix to
the same root cause and belongs in the same family of offerings.

**Verdict:** the draft was **partly right, partly wrong.** The enum variant +
naming are fork-shaped instantiation, but (1) the generic "fold a bare batch proof
over caller AIRs" entry is a legitimate small general contribution, and (2)
`72ffc56`'s bus-accounting fix is general. **De-dregg-ification:** offer the glue
fn renamed/doc'd as a generic batch-proof-over-your-AIRs entry (drop the
NativeBatchStark/dregg framing); offer the dup-`Public` fix alongside the draft's
Item 1 as one coherent "`assert_zero`/zero-slot collision" cluster.

---

## Item C — IVC fixed-point / constant-VK pinning  (commit `fc12f23`)

**What upstream has:** *nothing.* `git grep -ni
'expected_preprocessed_commit|vk.identity|fixed.point|child.vk|constant.vk'` over
`origin/main` returns only a stray comment in `examples/recursive_fibonacci.rs`.
No preprocessed-commitment pinning, no VK-identity construction.

**Capability (GENERAL, foundational):** `pin_preprocessed_commit` connects a child
proof's preprocessed-commitment (its verifier-key core — the Merkle cap binding
the child verifier circuit's static op-list) cap targets to **expected constants**
in-circuit. The child's preprocessed commitment is allocated as parent public
inputs and consumed by the child's preprocessed-trace FRI check, but its *value*
is otherwise unconstrained — so a from-scratch prover could fold a proof of a
*different* circuit. Pinning it makes a foreign-circuit child make the parent
UNSAT. This is the **stable anchor VK** that *any* IVC light client requires: it
is what makes "one fixed running circuit verifies its own previous proof" close
into a genuine fixed point rather than an open-ended chain. The function has
**zero** dregg content — it takes cap targets and an expected commitment value.

**Instantiation (fork-only):** dregg's *choice* to pass "the running circuit's own
commitment." The `into_recursion_input_pinned` / `running_preprocessed_commit`
convenience helpers are general (their docs are IVC-flavored but the logic is
not). The `expected_preprocessed_commit` enum field is general plumbing.

**The (b) half is more fork-shaped:** the commit also threads
`table_public_inputs` (`head_root` / `chain_digest` / `num_turns`) across layers.
*Threading public inputs across aggregation layers* is general; the *specific
three publics* are dregg's. So lever (a) (the VK pin) is cleanly general; lever
(b) is a general pattern with a dregg instantiation.

**Verdict:** the draft was **wrong** to bucket the VK pin fork-only. It is the
**foundational IVC primitive** — arguably the most important missing capability
for anyone building an IVC light client on plonky3-recursion — and it is fully
general. **De-dregg-ification:** `pin_preprocessed_commit` needs essentially none;
trim the running-circuit framing on the convenience helpers; present lever (b) as
a generic "re-expose public inputs across layers" hook, not as the three named
roots.

---

## Revised contribution scope

The draft's "two small fixes; the rest stays in our fork" **under-scopes the
offer** by conflating "built for dregg" with "useful only to dregg." Three of the
capabilities are general things upstream genuinely lacks. The honest offer is a
**multi-PR IVC-enablement set**, roughly:

| PR | Source | Capability | Generality | De-dregg work |
|----|--------|-----------|------------|---------------|
| 1 (draft Item 1) | lowerer | `assert_zero`/value↔const aliasing fix | general | none (applies clean) |
| 2 (draft Item 2) | verifier | `zeta_next` opening-schedule gating | general | re-derive vs rewritten `main` |
| **3 (NEW)** | `8d42900`,`d959ff1` | **`expose_as_public_output` — verified public-output channel** | **general, substantial** | doc rewrite only |
| **4 (NEW)** | `fc12f23` (a) | **`pin_preprocessed_commit` — VK-identity pin for IVC fixed-points** | **general, foundational** | minimal |
| **5 (NEW)** | `ccebf66`,`72ffc56` | generic "fold a bare batch proof over caller AIRs" entry + dup-`Public` bus fix | general, thin glue | drop NativeBatchStark naming |

PRs 3 and 4 are the real correction to the prior assessment: they are not
leftovers, they are the machinery that makes plonky3-recursion *usable for IVC at
all* — exactly the owner's point.

### Honest counter-weight (don't over-correct)

Genuinely fork-only, correctly bucketed:
- The `RecursionInput::NativeBatchStark` **variant**, the `D=4/2/5` registrations,
  and the backend routing arms — dregg plumbing around the generic glue fn.
- The **four specific claims** (genesis_root / final_root / num_turns /
  chain_digest) and dregg's choice of what to expose / what VK to pin.
- The `.bin` proof fixtures, dregg-shaped integration tests, the
  `build_and_prove_*_layer_with_expose` orchestration, and lever (b)'s three named
  roots.
- The shared caveat from the draft stands: this work is **AI-developed and
  semi-unaudited**, and Items 3/4/5 all touch soundness-relevant accounting
  (witness-slot creation, bus multiplicities, in-circuit VK binding). Any offer
  must carry the same "review carefully, unaudited proposal" framing.

The de-dregg-ification for PRs 3–5 is **real work** — extracting each general
primitive (the op/AIR, the pin fn, the glue entry) cleanly out of the dregg IVC
orchestration that currently calls them — but it is *abstraction work on a general
capability*, not the manufacture of generality that isn't there.

## Bottom line

The owner is right. The prior draft's biggest miss was reading dregg's *usage* of
these primitives as the primitives' *meaning*. `expose_as_public_output` and
`pin_preprocessed_commit` are general IVC capabilities upstream lacks entirely;
they are the larger, more tasteful contribution. The offer should grow from "one
bug-fix + one diagnosis" to a focused multi-PR IVC-enablement set, with the
public-output channel and the VK pin as its centerpieces.
