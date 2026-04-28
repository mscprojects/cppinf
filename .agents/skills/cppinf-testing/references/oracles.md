# Offline oracle workflow

Use offline PyTorch only to generate committed test data. Do not add PyTorch as a runtime dependency of `cppinf`.

## Pinned command

Use a pinned `uv` command so the environment is reproducible and cacheable:

```bash
uv run --with torch==2.11.0 python - <<'PY'
import torch

lhs = torch.tensor([[1.0, -2.5, 3.25], [4.5, 0.5, -1.75]], dtype=torch.bfloat16)
rhs = torch.tensor([[2.0, -1.0], [0.25, 3.5], [-4.0, 1.5]], dtype=torch.bfloat16)

print((lhs @ rhs).float())
PY
```

Copy the printed result into the C++ test and keep the generator snippet as a short comment above the golden values.

## Preferred progression

1. Start with tiny deterministic literals.
2. Generate the expected output offline.
3. Commit the literals and expected values in the C++ test.
4. Add one seeded rounded-random oracle case only if extra coverage is useful.

## Random oracle guidance

Random inputs are useful for supplemental coverage, but keep them deterministic and readable:

- Set a fixed seed.
- Generate values in a bounded range.
- Round them to a small number of decimal places before building tensors.
- Prefer values that still exercise signs, asymmetry, and non-trivial accumulation.

Example:

```bash
uv run --with torch==2.11.0 python - <<'PY'
import torch

torch.manual_seed(1234)
lhs = torch.empty(2, 3).uniform_(-5, 5)
rhs = torch.empty(3, 2).uniform_(-5, 5)

lhs = torch.round(lhs * 100) / 100
rhs = torch.round(rhs * 100) / 100

lhs = lhs.to(torch.bfloat16)
rhs = rhs.to(torch.bfloat16)

print(lhs)
print(rhs)
print((lhs @ rhs).float())
PY
```

## When not to use an oracle

- Do not use PyTorch just to test simple shape checks or error handling.
- Do not replace obvious tiny examples with opaque oracle data.
- Do not use unseeded random inputs in committed tests.
