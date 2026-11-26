//! Running commands for agents.
//!
//! The agent crate defines the interface; `hca-sandbox` implements it with
//! an isolated, resource-limited container per project. Nothing here runs
//! commands on the host, so agents can never change the machine HCA runs
//! on.

use std::path::Path;
use std::time::Duration;

use hca_llm::BoxFuture;

/// `uid:gid` for the critic's commands.
///
/// Project files belong to the agent user, uid 1000, so this user can read
/// them and cannot write them. It is not 1001: that is the GitHub Actions
/// runner, and a common second account, so a critic at 1001 would own files
/// those users create.
pub const CRITIC_USER: &str = "10000:10000";

/// What a command produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandOutput {
    /// The exit code, if the command exited normally.
    pub exit_code: Option<i32>,
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Whether the command was stopped for running too long.
    pub timed_out: bool,
}

impl CommandOutput {
    /// Formats the output for the model, keeping the end of long output,
    /// where errors usually are.
    pub fn render(&self, max_chars: usize) -> String {
        let status = match (self.timed_out, self.exit_code) {
            (true, _) => "timed out".to_owned(),
            (false, Some(code)) => format!("exit code {code}"),
            (false, None) => "killed".to_owned(),
        };
        let text = format!("{status}\n--- stdout ---\n{}\n--- stderr ---\n{}", self.stdout, self.stderr);
        tail(&text, max_chars)
    }
}

/// Runs shell commands inside a project's workspace.
pub trait CommandRunner: Send + Sync + std::fmt::Debug {
    /// Runs `command` with `workspace` as its working directory.
    ///
    /// `user`, when set, is the container user (`uid:gid`) the command runs
    /// as. The critic uses a user that can read the project but not write it.
    ///
    /// Returns `Err` only if the command couldn't be started at all; a
    /// failing command is an `Ok` with a non-zero exit code.
    fn run<'a>(
        &'a self,
        workspace: &'a Path,
        command: &'a str,
        timeout: Duration,
        user: Option<&'a str>,
    ) -> BoxFuture<'a, Result<CommandOutput, String>>;
}

fn tail(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_owned();
    }
    let skipped = total - max_chars;
    let kept: String = text.chars().skip(skipped).collect();
    format!("[... {skipped} characters omitted ...]\n{kept}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_output_keeps_its_end() {
        let output = CommandOutput {
            exit_code: Some(1),
            stdout: "ok\n".repeat(1_000),
            stderr: "AssertionError: fib(10) != 55".into(),
            timed_out: false,
        };
        let rendered = output.render(200);
        assert!(rendered.ends_with("AssertionError: fib(10) != 55"));
        assert!(rendered.starts_with("[..."));
    }
}
