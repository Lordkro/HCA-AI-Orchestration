//! Running commands for agents.
//!
//! The agent crate defines the interface; `hca-sandbox` implements it with
//! an isolated, resource-limited container per agent run. Nothing here runs
//! commands on the host, so agents can never change the machine HCA runs
//! on.

use std::path::Path;
use std::time::Duration;

use hca_llm::BoxFuture;

/// `uid:gid` for the critic's commands.
///
/// The sandbox runs these in a container of their own, with the project
/// mounted read-only and a `/tmp` no other role sees, so the critic can't
/// change the project or leave anything behind for the next agent. A
/// distinct uid is a second layer: project files belong to the agent user,
/// uid 1000, so this one can read them but not write them. It is not 1001:
/// that is the GitHub Actions runner, and a common second account, so a
/// critic at 1001 would own files those users create.
pub const CRITIC_USER: &str = "10000:10000";

/// Which agent run a command belongs to, and the user it runs as.
///
/// Commands of one session share processes and `/tmp`, so a server started
/// in one command can be tested by the next. Different sessions never share
/// them, and [`CommandRunner::end_session`] stops everything a session
/// left running.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Session<'a> {
    /// Names the agent run. Empty means one shared default session.
    pub id: &'a str,
    /// The container user (`uid:gid`), when not the default. The critic
    /// uses a user that can read the project but not write it.
    pub user: Option<&'a str>,
}

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
    /// Runs `command` with `workspace` as its working directory, in
    /// `session`.
    ///
    /// Returns `Err` only if the command couldn't be started at all; a
    /// failing command is an `Ok` with a non-zero exit code.
    fn run<'a>(
        &'a self,
        workspace: &'a Path,
        command: &'a str,
        timeout: Duration,
        session: Session<'a>,
    ) -> BoxFuture<'a, Result<CommandOutput, String>>;

    /// Stops everything session `id` still has running in `workspace`:
    /// background processes, its `/tmp`. The orchestrator calls this when an
    /// agent run ends, before its work is handed on, so nothing a run left
    /// behind can change files under review or get in another task's way.
    ///
    /// The default does nothing, for runners that keep no state.
    fn end_session<'a>(&'a self, workspace: &'a Path, id: &'a str) -> BoxFuture<'a, Result<(), String>> {
        let _ = (workspace, id);
        Box::pin(async { Ok(()) })
    }

    /// Frees everything held for `workspace`, for every user. Called when
    /// its project ends. The workspace files are kept.
    ///
    /// The default does nothing.
    fn remove<'a>(&'a self, workspace: &'a Path) -> BoxFuture<'a, Result<(), String>> {
        let _ = workspace;
        Box::pin(async { Ok(()) })
    }
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
