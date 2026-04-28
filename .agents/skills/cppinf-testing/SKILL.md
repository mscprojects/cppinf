---
name: cppinf-testing
description: Guide cppinf-specific testing work, especially tensor math, BF16 correctness, and offline PyTorch oracle generation. Use when adding, updating, or debugging tests in this repository, and keep `develop` loaded for code changes.
---

# cppinf-testing

## Goal

Create tests that are deterministic, readable, and useful for debugging tensor math and model-loading code in `cppinf`.

## Inputs

- The code being changed.
- Existing repo conventions from `AGENTS.md`.
- Existing tests in `tests/`.
- Offline oracle values when numerical behavior is hard to derive by hand.

## Sources of truth

1. Existing `cppinf` C++ behavior and test style in this repo.
2. Small explicit examples for primitive operations.
3. Offline PyTorch oracle values for numerical kernels and block-level checks.

## Expected output

- Focused tests under `tests/` that mirror `src/`.
- Small readable literals for simple cases.
- Committed golden values for oracle-based numerical tests.
- Short inline comments that show how oracle values were generated.

## Workflow

1. Start with the smallest deterministic test that proves the behavior.
2. Prefer whole-object equality with `operator==` when practical.
3. For tensor math primitives, use explicit hand-written expected values first when the numbers stay readable.
4. When the math becomes tedious or error-prone, generate oracle values offline with pinned PyTorch.
5. Paste the exact generator snippet into the test as a short comment directly above the golden values.
6. Commit only the C++ test data and comments. Do not add a runtime PyTorch dependency to the project.
7. Keep BF16 comparisons deterministic by encoding fixed input literals and asserting on committed golden outputs.

## Decision rules

- Use hand-written expected values when the tensor is tiny and the intended result is easy to audit.
- Use offline PyTorch oracles when the operation is numerically dense, multi-step, or easy to miscompute by hand.
- Prefer deterministic hand-picked inputs for the first golden test of a new operation.
- Add seeded rounded random coverage only after a clear deterministic golden test already exists.
- Keep random oracle inputs bounded and readable. Avoid huge magnitudes unless the test is explicitly about stability or overflow behavior.

## Test placement

- Put tests under the mirrored directory in `tests/`.
- Keep tensor and kernel tests close to the relevant subsystem, for example `tests/ops/` and `tests/tensors/`.
- Reuse local helpers in the test file when that keeps setup obvious.

## Oracle generation

Read `references/oracles.md` when generating or refreshing offline PyTorch goldens.

## Validation

1. Run the relevant focused tests while iterating.
2. Finish with the repository test command.
3. If an oracle-backed test fails, first verify the literal inputs and the pinned generator command before changing tolerances or production code.
