# Draft upstream PR — two engine fixes from a downstream fork

Status: **DRAFT — not submitted.** This file is prepared for a human (the fork
owner) to review and decide whether/how to offer it upstream. Nothing here has
been pushed to `Plonky3/Plonky3-recursion` and no PR has been opened.

---

## Honest framing (please read first)

This work was **developed with substantial AI assistance** and is
**semi-unaudited**: it passes our downstream fork's test suite, but it has not
been independently reviewed by a STARK/recursion expert. We are a downstream
user offering two small, general-purpose fixes back **for your consideration,
with no expectation** that you take them. If they are not useful, or the
framing is wrong, please just close — no hard feelings.

We have deliberately kept this **small and easy to evaluate**. The large
majority of our fork is application-specific and stays in our fork; only the two
items below looked like they would help any `Plonky3-recursion` user.

**Please review carefully before trusting either fix.** Both touch
soundness-relevant accounting (witness-slot creation and the recursive opening
schedule). We believe they are correct and our tests agree, but treat them as
unaudited proposals, not vetted patches.

---

## Item 1 (primary) — lowerer: don't alias a `connect(value, const)` / `assert_zero` into the constant's witness slot

### What it is

In the circuit lowerer, declared `connect(a, b)` pairs are fed into a union-find
(`ConnectDsu`) so that connected expressions share a single witness slot. That
aliasing is sound when both sides are value-bearing (they compute the same
value, so sharing a slot is free equality). It is **not** sound when exactly one
side is a constant — the dominant case being `assert_zero(x)`, which lowers to
`connect(x, ExprId::ZERO)`.

Aliasing `x` into the constant's class gives the value-bearing witness the
**constant's shared slot**. Because every `assert_zero` in the circuit routes
through the single circuit-wide `ExprId::ZERO` class, many independent
`assert_zero`s collapse into one shared slot. If any member is genuinely nonzero
at witness-generation time, its creator overwrites the shared constant slot and
the prover raises `WitnessConflict { WitnessId(0) }` — a spurious, hard-to-localize
failure rather than a clean local mismatch. In our fork this manifested as a
~219-member `ZERO`-class collapse.

### The fix

`connect` pairs with **exactly one** constant side are no longer fed to the DSU.
Instead they are deferred and, after `emit_operations`, re-emitted as a
per-target equality **constraint**:

```text
Op::MulAdd { a: x, b: ZERO, c: const, out: x }   // x*0 + c == x  ⇔  x == c
```

This keeps `x` in its own witness slot, keeps the constant the sole writer of
its slot, and makes a mismatch fail **locally** on `x`'s slot. We chose `MulAdd`
over a second `Op::Const` because `ConstAir` treats every `Op::Const` as a slot
creator (which would reintroduce a double-creator), whereas `MulAdd` flows
through the existing ALU bus accounting with no new bus path. The equality is a
genuine AIR constraint, so it is *enforced*, not merely witness-aliased.

### Scope / files

Three files, in `circuit/src/builder/compiler/lowerer/`:

- `state.rs` — partition declared connects; defer value↔const pairs; add
  `emit_deferred_const_connects`.
- `mod.rs` — call `emit_deferred_const_connects` after `emit_operations`,
  before `backfill_connect_mappings`.
- `tests.rs` — three existing tests that asserted the *old* const-aliasing
  behavior, updated to the equality semantics.

A clean, self-contained patch of exactly these three files is included next to
this draft as `fix1-lowerer-assert-zero.patch`.

### Applicability to current `main`

We confirmed this slice **applies cleanly** to current upstream `main`
(`git apply --check` succeeds): the `lowerer/` directory is byte-identical
between our fork base and your current `main`, so this is a clean cherry-pick.

### Testing

- `cargo test -p p3-circuit` — **354 passed, 0 failed** (includes the lowerer
  unit/property suite and the three updated const-connect tests).

### Known limitations

- Verified by our test suite only; not independently audited.
- The fix targets the `assert_zero` / single-constant-side family. We did not
  attempt to re-derive whether any other `connect` shape has an analogous
  aliasing hazard.

---

## Item 2 (secondary — offered as a diagnosis, not a clean patch)

### Recursive batch-STARK verifier assumes `zeta_next` openings for every AIR

The recursive batch-STARK verifier and challenge generator unconditionally
expect a next-row (`zeta_next`) opening for **every** instance — e.g. in current
`main`, `recursion/src/verifier/batch_stark.rs` still checks
`trace_next_targets.len() != air_width` for all AIRs and always opens at
`zeta_next`.

But native `prove_batch` only opens at `zeta_next` when an AIR actually accesses
next-row columns (`main_next_row_columns()` / `preprocessed_next_row_columns()`
non-empty). Single-row / constant tables (e.g. `ConstAir`, `PublicAir`, and in
our fork the Poseidon2-perm and expose tables) override those to return empty.
For such an AIR the prover commits **no** `zeta_next` opening while the recursive
verifier still expects one — a prover/verifier opening-schedule **disagreement**.
In our setting this surfaced as a genuinely-nonzero reduced opening (a claimed
`f(zeta)` paired against a `zeta_next` opening that was never committed),
appearing downstream as a `WitnessConflict` during aggregation.

We note this also requires the dynamic-AIR wrappers to **forward** the wrapped
AIR's `*_next_row_columns()` overrides; the default `BaseAir` impl reports all
columns, so a wrapper that doesn't forward silently re-breaks the gating.

### How we fixed it in our fork (reference only)

We gated `zeta_next` openings on the AIR's declared next-row columns on **both**
sides (mirroring native), added `uses_main_next_row` / `uses_preprocessed_next_row`
to the recursive-AIR trait, forwarded the `*_next_row_columns()` overrides through
the dynamic-AIR wrappers, and made `OpenedValuesTargets` tolerate a `None`
`trace_next`. The red→green witness was `aggregation_different_shapes`:

- `cargo test -p p3-recursion --test aggregation_different_shapes` — **1 passed**.

The full change lives in our fork commit `abc2c2a` (files: `traits/air.rs`,
`verifier/batch_stark.rs`, `types/proof.rs`, `generation.rs`, `recursion.rs`, and
the `batch_stark_prover` dynamic-AIR wrapper).

### Why this is a report, not a patch

Your current `main` has **substantially rewritten** exactly these files
(`verifier/batch_stark.rs`, `generation.rs`, `recursion.rs`, `traits/air.rs`,
`types/proof.rs` all changed heavily since our fork base). Our diff will **not**
cherry-pick cleanly and would need re-derivation against the current structure.
Rather than dump a stale patch, we are flagging the bug + our approach so you can
decide whether to fix it your own way. If it is useful, we are happy to prepare a
fresh patch against `main`.

### Known limitations

- Diagnosis verified by our fork tests only; unaudited.
- We have not checked whether your recent verifier rewrite already addresses this
  for some AIR shapes; from a read of `main` the unconditional `zeta_next` path
  still appears present, but please verify.

---

## What we are NOT offering (stays in our fork)

For honesty about scope: the rest of our fork is application-specific and is not
part of this proposal — a custom claim-exposure AIR + ops, a native-batch
leaf-wrapping recursion input, an IVC fixed-point / constant-VK construction, and
a set of application-shaped integration tests with binary proof fixtures. None of
that belongs upstream. There is also one related witness-accounting fix
(demoting a duplicate `Public` output to a bus reader) that is general in spirit
but entangled with our application feature and interacts subtly with Item 1, so
we left it out to keep this small and clean.

---

## Summary

- **Item 1** is the real proposal: a small (3-file) lowerer correctness fix that
  applies cleanly to `main` and is covered by `p3-circuit`'s suite.
- **Item 2** is a bug report with a reference fix; it needs re-derivation against
  your rewritten verifier.
- Everything is AI-developed and unaudited. Offered humbly, no expectation.
  Please review carefully.
