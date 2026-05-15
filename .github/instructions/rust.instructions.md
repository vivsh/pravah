---
applyTo: "**/*.rs"
---

# Rust

- Always avoid any panicking code. Instead, use `Result` and `Option` types to handle errors gracefully.
- Avoid comments that exist only to separate sections.
- Doc-comments should be terse but adequate. Keep them as if they are written by human
- Avoid unnecessary cloning and allocate memory only when absolutely necessary. Use references and borrowing to minimize memory usage.
- For every test function, include a doc comment that describes the purpose of the test and what it is verifying.
- Try to avoid using unreachable!
- Donot use Box<dyn std::error::Error> as a result error type. Always use thiserror derived structured errors
- A function must not be more than 50 lines - break into smaller files.
- Avoid file modules that are longer than 1000 lines - divide into multiple files and folder module. In a folder module, mod should always be thin and comprise mainly of mod statements and re-exports.
- Donot use blocking code in async functions.
- For sqlx, avoid the query macros. Use FromRow where possible instead of using try_get to fetch information from a row

- Donot make any assumption that breaks any existing public interface. If there is any confusion then ask but never ever work with assumptions.
