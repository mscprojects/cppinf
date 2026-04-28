# AGENTS

When working in this repository:

- Use snake_case for function names.
- Use PascalCase for class names.
- Use 4 spaces for indentation.
- Keep lines within 120 characters when formatting.
- Prefer `fmt::format` over string concatenation when building strings.
- Short single-line comments for non-obvious variables and functions are okay.
- For GoogleTest, name fixtures after the file subject, for example `TensorTest`.
- For GoogleTest test names, use Gherkin-style `Given_When_Then`.
- Prefer whole-struct equality assertions with `operator==` over many field-by-field `EXPECT_EQ`s when practical.
