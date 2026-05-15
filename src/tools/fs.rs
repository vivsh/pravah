use tokio::fs;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::base::{Tool, ToolError};
use crate::context::Context;

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileOutput {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct WriteFileOutput {
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListDirOutput {
    pub path: String,
    pub entries: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PatchFileOutput {
    pub path: String,
    pub applied: usize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MultiPatchFileOutput {
    pub path: String,
    pub applied: usize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PatchLinesOutput {
    pub path: String,
    pub replaced_lines: usize,
    pub inserted_lines: usize,
}

/// Reads a file and returns its full text content.
#[derive(Deserialize, JsonSchema)]
pub struct ReadFile {
    /// Path to the file to read.
    pub path: String,
}

impl Tool for ReadFile {
    type Output = ReadFileOutput;
    fn name() -> &'static str {
        "read_file"
    }
    fn description() -> &'static str {
        "Read a file and return its full text content."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let resolved = ctx.resolve(&self.path)?;
        let content = fs::read_to_string(&resolved).await?;
        Ok(ReadFileOutput {
            path: self.path,
            content,
        })
    }
}

/// Writes content to a file, creating parent directories as needed.
#[derive(Deserialize, JsonSchema)]
pub struct WriteFile {
    /// Destination path.
    pub path: String,
    /// Content to write.
    pub content: String,
}

impl Tool for WriteFile {
    type Output = WriteFileOutput;
    fn name() -> &'static str {
        "write_file"
    }
    fn description() -> &'static str {
        "Write content to a file, creating parent directories as needed."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let resolved = ctx.resolve(&self.path)?;
        if let Some(parent) = resolved.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        fs::write(&resolved, &self.content).await?;
        Ok(WriteFileOutput { path: self.path })
    }
}

/// Lists entries in a directory (non-recursive).
#[derive(Deserialize, JsonSchema)]
pub struct ListDir {
    /// Path to the directory to list.
    pub path: String,
}

impl Tool for ListDir {
    type Output = ListDirOutput;
    fn name() -> &'static str {
        "list_dir"
    }
    fn description() -> &'static str {
        "List entries in a directory. Subdirectories are suffixed with '/'."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let resolved = ctx.resolve(&self.path)?;
        let mut read_dir = fs::read_dir(&resolved).await?;
        let mut entries = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().await?.is_dir();
            entries.push(if is_dir { format!("{name}/") } else { name });
        }
        Ok(ListDirOutput {
            path: self.path,
            entries,
        })
    }
}

/// A single `(old, new)` replacement within a file.
#[derive(Deserialize, JsonSchema)]
pub struct Replacement {
    /// Exact text to find. Must appear exactly once in the file.
    pub old: String,
    /// Text to substitute in place of `old`.
    pub new: String,
}

/// Replaces one exact text fragment in a file without rewriting the whole file.
///
/// The agent only needs to provide the lines being changed plus enough surrounding
/// context to make `old` unique — far fewer tokens than reading and rewriting the
/// entire file.
#[derive(Deserialize, JsonSchema)]
pub struct PatchFile {
    /// Path to the file to patch.
    pub path: String,
    /// Exact text to find. Must appear exactly once in the file.
    pub old: String,
    /// Replacement text.
    pub new: String,
}

impl Tool for PatchFile {
    type Output = PatchFileOutput;
    fn name() -> &'static str {
        "patch_file"
    }
    fn description() -> &'static str {
        "Replace one exact text fragment in a file. Provide just the changed region plus \
         enough context to locate it uniquely — much cheaper than reading and rewriting the \
         whole file. Fails if `old` appears zero or more than once."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let resolved = ctx.resolve(&self.path)?;
        let source = fs::read_to_string(&resolved).await?;
        let patched = apply_patch(&source, &self.old, &self.new)?;
        fs::write(&resolved, &patched).await?;
        Ok(PatchFileOutput {
            path: self.path,
            applied: 1,
        })
    }
}

/// Applies multiple text replacements to a file in a single round-trip.
///
/// Replacements are applied sequentially. This lets an agent make several
/// independent edits while sending only the changed regions, minimising token usage.
#[derive(Deserialize, JsonSchema)]
pub struct MultiPatchFile {
    /// Path to the file to patch.
    pub path: String,
    /// Ordered list of replacements to apply. Each `old` must be unique at the
    /// point it is applied (i.e. after previous replacements have been made).
    pub replacements: Vec<Replacement>,
}

impl Tool for MultiPatchFile {
    type Output = MultiPatchFileOutput;
    fn name() -> &'static str {
        "multi_patch_file"
    }
    fn description() -> &'static str {
        "Apply multiple text replacements to a file in one call. Cheaper than patching \
         in a loop because only the changed regions need to be sent. Replacements are \
         applied in order; each `old` must be unique at the time it is matched."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let resolved = ctx.resolve(&self.path)?;
        let mut source = fs::read_to_string(&resolved).await?;
        let total = self.replacements.len();
        for r in self.replacements {
            source = apply_patch(&source, &r.old, &r.new)?;
        }
        fs::write(&resolved, &source).await?;
        Ok(MultiPatchFileOutput {
            path: self.path,
            applied: total,
        })
    }
}

/// Finds `old` exactly once in `source` and returns the patched string.
fn apply_patch(source: &str, old: &str, new: &str) -> Result<String, ToolError> {
    let count = source.matches(old).count();
    match count {
        0 => Err(ToolError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("patch target not found in file:\n{old}"),
        ))),
        1 => Ok(source.replacen(old, new, 1)),
        n => Err(ToolError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "patch target matches {n} locations; add more context to make it unique:\n{old}"
            ),
        ))),
    }
}

/// Replaces a contiguous range of lines in a file with new content.
///
/// Line numbers come directly from `file_outline` or `get_symbol` output, making
/// this a low-token alternative to `patch_file` when the agent already knows the
/// line range but the text is large or repetitive.
#[derive(Deserialize, JsonSchema)]
pub struct PatchLines {
    /// Path to the file.
    pub path: String,
    /// First line to replace (1-based, inclusive). Must be at least 1.
    pub start_line: usize,
    /// Last line to replace (1-based, inclusive).
    pub end_line: usize,
    /// Lines to write in place of the replaced range. May be empty to delete the range.
    pub new_lines: Vec<String>,
}

impl Tool for PatchLines {
    type Output = PatchLinesOutput;
    fn name() -> &'static str {
        "patch_lines"
    }
    fn description() -> &'static str {
        "Replace a contiguous range of lines (1-based) in a file with new content. \
         Use line numbers from `file_outline` or `get_symbol` to avoid sending the old \
         text at all — cheapest edit when you already know the line range."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        if self.start_line == 0 || self.start_line > self.end_line {
            return Err(ToolError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid line range: {}–{}", self.start_line, self.end_line),
            )));
        }
        let resolved = ctx.resolve(&self.path)?;
        let source = fs::read_to_string(&resolved).await?;
        let mut lines: Vec<&str> = source.lines().collect();
        let total = lines.len();
        if self.start_line > total || self.end_line > total {
            return Err(ToolError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "line range {}–{} out of bounds (file has {total} lines)",
                    self.start_line, self.end_line
                ),
            )));
        }
        let replaced_lines = self.end_line - self.start_line + 1;
        let inserted_lines = self.new_lines.len();
        let new_refs: Vec<&str> = self.new_lines.iter().map(String::as_str).collect();
        lines.splice((self.start_line - 1)..self.end_line, new_refs);
        let mut out = lines.join("\n");
        if source.ends_with('\n') {
            out.push('\n');
        }
        fs::write(&resolved, &out).await?;
        Ok(PatchLinesOutput {
            path: self.path,
            replaced_lines,
            inserted_lines,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FlowConf;

    fn ctx(dir: &tempfile::TempDir) -> Context {
        Context::new(FlowConf {
            working_dir: Some(dir.path().to_owned()),
            ..Default::default()
        })
    }

    /// `ReadFile` returns the file contents under the `content` key.
    #[tokio::test]
    async fn read_file_returns_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        fs::write(&path, "hello world").await.unwrap();

        let out = ReadFile {
            path: path.to_str().unwrap().into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap();
        assert_eq!(out.content, "hello world");
    }

    /// `ReadFile` returns `ToolError::Io` when the target path (inside working_dir) does not exist.
    #[tokio::test]
    async fn read_file_missing_returns_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.txt");
        let err = ReadFile {
            path: path.to_str().unwrap().into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    /// `WriteFile` creates parent directories and writes the file.
    #[tokio::test]
    async fn write_file_creates_dirs_and_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/dir/out.txt");

        WriteFile {
            path: path.to_str().unwrap().into(),
            content: "hi".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap();

        assert_eq!(fs::read_to_string(&path).await.unwrap(), "hi");
    }

    /// `WriteFile` overwrites an existing file.
    #[tokio::test]
    async fn write_file_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        fs::write(&path, "old").await.unwrap();

        WriteFile {
            path: path.to_str().unwrap().into(),
            content: "new".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap();

        assert_eq!(fs::read_to_string(&path).await.unwrap(), "new");
    }

    /// `ListDir` marks subdirectories with a trailing `/`.
    #[tokio::test]
    async fn list_dir_marks_subdirs() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("file.rs"), "").await.unwrap();
        fs::create_dir(dir.path().join("subdir")).await.unwrap();

        let out = ListDir {
            path: dir.path().to_str().unwrap().into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap();
        let entries = &out.entries;

        assert!(entries.iter().any(|e| e == "file.rs"));
        assert!(entries.iter().any(|e| e == "subdir/"));
    }

    /// `ListDir` returns `ToolError::Io` for a non-existent directory inside working_dir.
    #[tokio::test]
    async fn list_dir_missing_returns_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent_dir");
        let err = ListDir {
            path: path.to_str().unwrap().into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    /// `definition()` auto-derives the tool name and a schema containing all field names.
    #[test]
    fn definition_schema_contains_field_names() {
        let def = ReadFile::definition();
        assert_eq!(def.name, "read_file");
        assert!(
            def.parameters.to_string().contains("path"),
            "schema missing 'path'"
        );

        let def = WriteFile::definition();
        assert_eq!(def.name, "write_file");
        let s = def.parameters.to_string();
        assert!(s.contains("path") && s.contains("content"));
    }

    /// `ReadFile` rejects a path that uses `..` to escape the working directory.
    #[tokio::test]
    async fn read_file_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let escaped = format!("{}/../../../etc/passwd", dir.path().display());
        let err = ReadFile { path: escaped }
            .call(ctx(&dir))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)));
    }

    /// `ReadFile` rejects a symlink inside the working directory when it resolves outside it.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_file_rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("secret.txt");
        fs::write(&outside_file, "secret").await.unwrap();

        let link_path = dir.path().join("outside-link.txt");
        std::os::unix::fs::symlink(&outside_file, &link_path).unwrap();

        let err = ReadFile {
            path: link_path.to_str().unwrap().into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)));
    }

    /// `ReadFile` still follows a symlink when the resolved target stays inside the working directory.
    #[cfg(unix)]
    #[tokio::test]
    async fn read_file_allows_internal_symlink_target() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("data.txt");
        fs::write(&target, "inside").await.unwrap();

        let link_path = dir.path().join("inside-link.txt");
        std::os::unix::fs::symlink(&target, &link_path).unwrap();

        let out = ReadFile {
            path: link_path.to_str().unwrap().into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap();
        assert_eq!(out.content, "inside");
    }

    /// `WriteFile` rejects a path that uses `..` to escape the working directory.
    #[tokio::test]
    async fn write_file_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let escaped = format!("{}/../escape.txt", dir.path().display());
        let err = WriteFile {
            path: escaped,
            content: "x".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)));
    }

    /// `ListDir` rejects an absolute path that is outside the working directory.
    #[tokio::test]
    async fn list_dir_rejects_path_outside_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let err = ListDir {
            path: "/etc".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::PathEscape(_)));
    }

    /// `PatchFile` replaces the unique matching fragment and writes the file.
    #[tokio::test]
    async fn patch_file_replaces_unique_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "fn foo() {}\nfn bar() {}\n")
            .await
            .unwrap();

        let out = PatchFile {
            path: path.to_str().unwrap().into(),
            old: "fn foo() {}".into(),
            new: "fn foo() { /* patched */ }".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap();

        assert_eq!(out.applied, 1);
        let content = fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("fn foo() { /* patched */ }"));
        assert!(content.contains("fn bar() {}"));
    }

    /// `PatchFile` returns an IO error when `old` is not found in the file.
    #[tokio::test]
    async fn patch_file_not_found_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "fn foo() {}").await.unwrap();

        let err = PatchFile {
            path: path.to_str().unwrap().into(),
            old: "fn bar() {}".into(),
            new: "fn bar() { /* x */ }".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    /// `PatchFile` returns an IO error when `old` matches more than once.
    #[tokio::test]
    async fn patch_file_ambiguous_match_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "let x = 1;\nlet x = 1;\n").await.unwrap();

        let err = PatchFile {
            path: path.to_str().unwrap().into(),
            old: "let x = 1;".into(),
            new: "let x = 2;".into(),
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    /// `MultiPatchFile` applies all replacements sequentially in one file round-trip.
    #[tokio::test]
    async fn multi_patch_file_applies_all_replacements() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "fn foo() {}\nfn bar() {}\nfn baz() {}\n")
            .await
            .unwrap();

        let out = MultiPatchFile {
            path: path.to_str().unwrap().into(),
            replacements: vec![
                Replacement {
                    old: "fn foo() {}".into(),
                    new: "fn foo() { 1 }".into(),
                },
                Replacement {
                    old: "fn bar() {}".into(),
                    new: "fn bar() { 2 }".into(),
                },
            ],
        }
        .call(ctx(&dir))
        .await
        .unwrap();

        assert_eq!(out.applied, 2);
        let content = fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("fn foo() { 1 }"));
        assert!(content.contains("fn bar() { 2 }"));
        assert!(content.contains("fn baz() {}"));
    }

    /// `MultiPatchFile` stops and returns an error when a replacement target is not found.
    #[tokio::test]
    async fn multi_patch_file_aborts_on_missing_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "fn foo() {}\n").await.unwrap();

        let err = MultiPatchFile {
            path: path.to_str().unwrap().into(),
            replacements: vec![
                Replacement {
                    old: "fn foo() {}".into(),
                    new: "fn foo() { 1 }".into(),
                },
                Replacement {
                    old: "fn missing() {}".into(),
                    new: "fn missing() { 2 }".into(),
                },
            ],
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }

    /// `PatchLines` replaces the specified line range with new content.
    #[tokio::test]
    async fn patch_lines_replaces_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "line1\nline2\nline3\nline4\n")
            .await
            .unwrap();

        let out = PatchLines {
            path: path.to_str().unwrap().into(),
            start_line: 2,
            end_line: 3,
            new_lines: vec![
                "replaced_a".into(),
                "replaced_b".into(),
                "replaced_c".into(),
            ],
        }
        .call(ctx(&dir))
        .await
        .unwrap();

        assert_eq!(out.replaced_lines, 2);
        assert_eq!(out.inserted_lines, 3);
        let content = fs::read_to_string(&path).await.unwrap();
        assert_eq!(
            content,
            "line1\nreplaced_a\nreplaced_b\nreplaced_c\nline4\n"
        );
    }

    /// `PatchLines` can delete lines by passing an empty `new_lines` list.
    #[tokio::test]
    async fn patch_lines_deletes_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "line1\nline2\nline3\n").await.unwrap();

        PatchLines {
            path: path.to_str().unwrap().into(),
            start_line: 2,
            end_line: 2,
            new_lines: vec![],
        }
        .call(ctx(&dir))
        .await
        .unwrap();

        let content = fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "line1\nline3\n");
    }

    /// `PatchLines` returns an IO error when the range exceeds the file length.
    #[tokio::test]
    async fn patch_lines_out_of_bounds_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src.rs");
        fs::write(&path, "line1\nline2\n").await.unwrap();

        let err = PatchLines {
            path: path.to_str().unwrap().into(),
            start_line: 5,
            end_line: 6,
            new_lines: vec!["x".into()],
        }
        .call(ctx(&dir))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::Io(_)));
    }
}
