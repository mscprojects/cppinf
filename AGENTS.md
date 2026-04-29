# AGENTS

When working in this repository:

- Use snake_case for function names.
- Use PascalCase for class names.
- Use 4 spaces for indentation.
- Keep lines within 120 characters when formatting.
- Separate independent blocks with a blank line, for example adjacent guard or validation `if` blocks, or adjacent
  teaching-commented inference stages.
- Prefer `auto` or `const auto` when the type remains obvious.
- Prefer `fmt::format` over string concatenation when building strings.
- When adding explanatory code comments, prefer brief teaching-style comments that explain stage purpose and symbolic
  tensor shapes for readers learning how inference works. Write them as one long comment line in source when practical,
  then let clang-format wrap them.
- Short single-line comments for non-obvious variables and functions are okay.
- Add brief contract comments to public functions in headers.
- In prose comments and header docs, use `,` or `.` instead of `;`.
- For GoogleTest, name fixtures after the file subject, for example `TensorTest`.
- For GoogleTest test names, use Gherkin-style `Given_When_Then`.
- Prefer whole-struct equality assertions with `operator==` over many field-by-field `EXPECT_EQ`s when practical.
- Run `just commit` before creating a git commit.
