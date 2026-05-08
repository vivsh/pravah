use tokio::process::Command;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::base::{Tool, ToolError};
use crate::context::Context;

#[derive(Debug, Serialize)]
pub struct RunCommandOutput {
    pub command: String,
    pub args: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Runs an allowed shell command inside the working directory.
#[derive(Deserialize, JsonSchema)]
pub struct RunCommand {
    /// The executable to run (e.g. `"cargo"`, `"git"`). Must be in the allowlist.
    pub command: String,
    /// Arguments to pass to the command (e.g. `["test", "--", "--nocapture"]`).
    pub args: Vec<String>,
}

impl Tool for RunCommand {
    type Output = RunCommandOutput;
    fn name() -> &'static str {
        "run_command"
    }
    fn description() -> &'static str {
        "Run an allowed command inside the working directory and return its output."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        ctx.check_command(&self.command)?;
        let output = Command::new(&self.command)
            .args(&self.args)
            .current_dir(ctx.working_dir())
            .output()
            .await?;
        Ok(RunCommandOutput {
            command: self.command,
            args: self.args,
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FlowConf;

    fn ctx(dir: &std::path::Path, allowed: &[&str]) -> Context {
        Context::new(FlowConf {
            working_dir: Some(dir.to_path_buf()),
            ..Default::default()
        })
        .with_commands(allowed.iter().map(|s| s.to_string()).collect())
    }

    /// `RunCommand` executes an allowed command and captures stdout.
    #[tokio::test]
    async fn run_command_captures_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let out = RunCommand {
            command: "echo".into(),
            args: vec!["hello".into()],
        }
        .call(ctx(dir.path(), &["echo"]))
        .await
        .unwrap();
        assert!(out.stdout.contains("hello"));
        assert_eq!(out.exit_code, Some(0));
    }

    /// `RunCommand` runs in the working directory (pwd matches).
    #[tokio::test]
    async fn run_command_uses_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let out = RunCommand {
            command: "pwd".into(),
            args: vec![],
        }
        .call(ctx(dir.path(), &["pwd"]))
        .await
        .unwrap();
        let stdout = out.stdout.trim().to_owned();
        // Normalise in case of symlinks (macOS /var → /private/var).
        let stdout_canon = std::path::PathBuf::from(&stdout)
            .canonicalize()
            .unwrap_or(std::path::PathBuf::from(&stdout));
        assert_eq!(stdout_canon, canonical);
    }

    /// `RunCommand` returns `ToolError::ForbiddenCommand` for commands not in the allowlist.
    #[tokio::test]
    async fn run_command_rejects_forbidden_command() {
        let dir = tempfile::tempdir().unwrap();
        let err = RunCommand {
            command: "rm".into(),
            args: vec!["-rf".into(), "/".into()],
        }
        .call(ctx(dir.path(), &["echo"]))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::ForbiddenCommand(c) if c == "rm"));
    }

    /// `RunCommand` with an empty allowlist rejects every command.
    #[tokio::test]
    async fn run_command_empty_allowlist_rejects_all() {
        let dir = tempfile::tempdir().unwrap();
        let err = RunCommand {
            command: "echo".into(),
            args: vec![],
        }
        .call(ctx(dir.path(), &[]))
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::ForbiddenCommand(_)));
    }

    /// A non-zero exit code is captured in the output rather than returned as an error.
    #[tokio::test]
    async fn run_command_captures_nonzero_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        // `false` always exits with code 1.
        let out = RunCommand {
            command: "false".into(),
            args: vec![],
        }
        .call(ctx(dir.path(), &["false"]))
        .await
        .unwrap();
        assert_ne!(out.exit_code, Some(0));
    }
}
