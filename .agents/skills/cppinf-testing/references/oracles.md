# Offline oracle workflow

Use offline PyTorch only to generate committed test data. Do not add PyTorch as a runtime dependency of `cppinf`.

## Generator placement

Put oracle generator code in a checked-in `*_oracle.py` file next to the relevant test, for example
`tests/nn/qwen_attention_oracle.py` next to `tests/nn/qwen_attention_test.cpp`.

- Prefer one focused generator file per tested operation or module.
- Keep the script inputs explicit and deterministic.
- Print or serialize only the data needed to refresh the committed goldens.
- In the C++ test, keep only a short comment that points to the generator file.

## Pinned command

Run the checked-in generator with a pinned `uv` command so the environment is reproducible and cacheable:

```bash
uv run --with torch==2.11.0 python tests/ops/matmul_oracle.py
```

Copy the generated result into the C++ test and keep the generator file checked in with the test.

## Preferred progression

1. Start with tiny deterministic literals.
2. Add or update a nearby `*_oracle.py` script that reproduces the ground truth.
3. Generate the expected output offline with pinned dependencies.
4. Commit the literals, expected values, and the generator script.
5. Add one seeded rounded-random oracle case only if extra coverage is useful.

## Random oracle guidance

Random inputs are useful for supplemental coverage, but keep them deterministic and readable:

- Set a fixed seed.
- Generate values in a bounded range.
- Round them to a small number of decimal places before building tensors.
- Prefer values that still exercise signs, asymmetry, and non-trivial accumulation.

Example:

```bash
uv run --with torch==2.11.0 python tests/ops/matmul_oracle.py
```

## When not to use an oracle

- Do not use PyTorch just to test simple shape checks or error handling.
- Do not replace obvious tiny examples with opaque oracle data.
- Do not use unseeded random inputs in committed tests.
