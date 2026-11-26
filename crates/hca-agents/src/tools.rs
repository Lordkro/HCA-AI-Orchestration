//! The tools agents can call, and which role gets which.
//!
//! | Tool            | PM | Research | Spec | Coder | Critic |
//! |-----------------|----|----------|------|-------|--------|
//! | `list_files`    | ✓  | ✓        | ✓    | ✓     | ✓      |
//! | `read_file`     | ✓  | ✓        | ✓    | ✓     | ✓      |
//! | `write_file`    |    | ✓        | ✓    | ✓     |        |
//! | `edit_file`     |    |          | ✓    | ✓     |        |
//! | `run_command`   |    |          |      | ✓ \*  | ✓ \*   |
//! | `submit`        |    | ✓        | ✓    | ✓     |        |
//! | `submit_review` |    |          |      |       | ✓      |
//! | `submit_plan`   | ✓  |          |      |       |        |
//!
//! \* only when a sandboxed [`CommandRunner`] is configured.
//!
//! The critic can't write files, so it can't "fix" what it reviews.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use hca_core::{AgentRole, Limits, ProjectId};
use hca_llm::{Message, ToolCall, ToolSpec};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{CommandRunner, Plan, Review, Session, Submission, SubmitArgs, Verdict, Workspace};

/// Characters of command output shown to the model.
const MAX_COMMAND_OUTPUT: usize = 12_000;

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// Path relative to the project root, e.g. `src/main.py`.
    path: String,
    /// First line to return, counting from 1. Defaults to the first line.
    #[serde(default)]
    start_line: Option<usize>,
    /// Last line to return, inclusive. Defaults to the last line.
    #[serde(default)]
    end_line: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
struct ListArgs {
    /// Folder relative to the project root. Defaults to the root.
    #[serde(default)]
    path: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
struct WriteArgs {
    /// Path relative to the project root.
    path: String,
    /// The complete new contents of the file.
    content: String,
}

#[derive(Deserialize, JsonSchema)]
struct EditArgs {
    /// Path relative to the project root.
    path: String,
    /// Text to replace. Must occur exactly once; copy it from `read_file`.
    old_text: String,
    /// The replacement text.
    new_text: String,
}

#[derive(Deserialize, JsonSchema)]
struct CommandArgs {
    /// A shell command, run from the project root. For example
    /// `python -m pytest -q`.
    command: String,
    /// Seconds before the command is stopped. Defaults to 120, at most 600.
    #[serde(default)]
    timeout_secs: Option<u64>,
}

/// What happened when a tool ran.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutcome {
    /// The result to send back to the model.
    pub message: Message,
    /// Set when the tool was a finishing tool and its input was valid.
    pub submission: Option<Submission>,
    /// The file written, if the tool wrote one.
    pub wrote: Option<String>,
}

/// The tools of one project.
#[derive(Clone, Debug)]
pub struct Toolbox {
    workspace: Workspace,
    runner: Option<Arc<dyn CommandRunner>>,
    limits: Limits,
    /// `uid:gid` passed to `run_command` for the critic only.
    command_user: Option<String>,
    /// The [`Session`] commands run in.
    session: String,
}

impl Toolbox {
    /// Tools working in `workspace`. Without a `runner`, no role gets
    /// `run_command`.
    pub fn new(workspace: Workspace, runner: Option<Arc<dyn CommandRunner>>, limits: Limits) -> Self {
        Self { workspace, runner, limits, command_user: None, session: String::new() }
    }

    /// Runs commands in session `id`, apart from every other session's
    /// processes. See [`CommandRunner::end_session`].
    #[must_use]
    pub fn with_session(mut self, id: impl Into<String>) -> Self {
        self.session = id.into();
        self
    }

    /// Runs the critic's commands as `user` (`uid:gid`).
    ///
    /// Other roles keep the sandbox's default user, which owns the project
    /// files. The critic's user should be able to read those files and write
    /// `/tmp`, and nothing else.
    #[must_use]
    pub fn with_command_user(mut self, user: impl Into<String>) -> Self {
        self.command_user = Some(user.into());
        self
    }

    /// The workspace the tools act on.
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    fn names(&self, role: AgentRole) -> Vec<&'static str> {
        let run = self.runner.is_some();
        let mut names = vec!["list_files", "read_file"];
        match role {
            AgentRole::Pm => names.push("submit_plan"),
            AgentRole::Research => names.extend(["write_file", "submit"]),
            AgentRole::Spec => names.extend(["write_file", "edit_file", "submit"]),
            AgentRole::Coder => {
                names.extend(["write_file", "edit_file"]);
                if run {
                    names.push("run_command");
                }
                names.push("submit");
            }
            AgentRole::Critic => {
                if run {
                    names.push("run_command");
                }
                names.push("submit_review");
            }
        }
        names
    }

    /// The name of the tool that ends `role`'s turn.
    pub const fn finishing_tool(role: AgentRole) -> &'static str {
        match role {
            AgentRole::Pm => "submit_plan",
            AgentRole::Critic => "submit_review",
            AgentRole::Research | AgentRole::Spec | AgentRole::Coder => "submit",
        }
    }

    /// The tool definitions `role` may use.
    pub fn specs(&self, role: AgentRole) -> Vec<ToolSpec> {
        self.names(role).into_iter().map(spec).collect()
    }

    /// Runs one tool call for `role`. Problems (unknown tool, bad
    /// arguments, refused path, failing command) become error results that
    /// the model can read and react to; they never end the run.
    pub async fn execute(&self, role: AgentRole, call: &ToolCall) -> ToolOutcome {
        let mut outcome =
            ToolOutcome { message: Message::tool_result(call, ""), submission: None, wrote: None };
        let result = if self.names(role).contains(&call.name.as_str()) {
            self.dispatch(role, call, &mut outcome).await
        } else {
            Err(format!("the {} role has no tool named {:?}", role, call.name))
        };
        outcome.message = match result {
            Ok(text) => Message::tool_result(call, text),
            Err(error) => {
                outcome.submission = None;
                outcome.wrote = None;
                Message::tool_error(call, error)
            }
        };
        outcome
    }

    async fn dispatch(
        &self,
        role: AgentRole,
        call: &ToolCall,
        outcome: &mut ToolOutcome,
    ) -> Result<String, String> {
        let ws = &self.workspace;
        match call.name.as_str() {
            "list_files" => {
                let args: ListArgs = call.parse()?;
                let entries =
                    ws.list(args.path.as_deref().unwrap_or(".")).await.map_err(|e| e.to_string())?;
                Ok(if entries.is_empty() { "(no files yet)".into() } else { entries.join("\n") })
            }
            "read_file" => {
                let args: ReadArgs = call.parse()?;
                let text = ws.read(&args.path).await.map_err(|e| e.to_string())?;
                if args.start_line.is_none() && args.end_line.is_none() {
                    return Ok(text);
                }
                line_range(&text, args.start_line, args.end_line)
            }
            "write_file" => {
                let args: WriteArgs = call.parse()?;
                ws.write(&args.path, &args.content).await.map_err(|e| e.to_string())?;
                outcome.wrote = Some(args.path.trim().to_owned());
                Ok(format!("wrote {} ({} lines)", args.path, args.content.lines().count()))
            }
            "edit_file" => {
                let args: EditArgs = call.parse()?;
                ws.edit(&args.path, &args.old_text, &args.new_text).await.map_err(|e| e.to_string())?;
                outcome.wrote = Some(args.path.trim().to_owned());
                Ok(format!("edited {}", args.path))
            }
            "run_command" => {
                let args: CommandArgs = call.parse()?;
                let runner = self.runner.as_ref().ok_or("commands are not available")?;
                let timeout = Duration::from_secs(args.timeout_secs.unwrap_or(120).clamp(1, 600));
                let user = if role == AgentRole::Critic { self.command_user.as_deref() } else { None };
                let session = Session { id: &self.session, user };
                let output = runner.run(ws.root(), &args.command, timeout, session).await?;
                Ok(output.render(MAX_COMMAND_OUTPUT))
            }
            "submit" => {
                let args: SubmitArgs = call.parse()?;
                if args.summary.trim().is_empty() {
                    return Err("the summary is empty; describe what you did".into());
                }
                outcome.submission = Some(Submission::Work { summary: args.summary });
                Ok("submitted".into())
            }
            "submit_review" => {
                let review: Review = call.parse()?;
                if review.verdict == Verdict::RequestChanges && review.feedback.trim().is_empty() {
                    return Err(
                        "feedback is required when requesting changes: say what to fix and where".into()
                    );
                }
                outcome.submission = Some(Submission::Review(review));
                Ok("review submitted".into())
            }
            "submit_plan" => {
                let plan: Plan = call.parse()?;
                // Validate now so the PM can fix mistakes in the same run.
                plan.clone().into_tasks(ProjectId::new(), &self.limits, Utc::now())?;
                let count = plan.tasks.len();
                outcome.submission = Some(Submission::Plan(plan));
                Ok(format!("plan with {count} tasks submitted"))
            }
            other => Err(format!("unknown tool {other:?}")),
        }
    }
}

/// Lines `start..=end` (1-based) of `text`, with a header saying which.
fn line_range(text: &str, start: Option<usize>, end: Option<usize>) -> Result<String, String> {
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let start = start.unwrap_or(1).max(1);
    let end = end.unwrap_or(total).min(total);
    if start > total {
        return Err(format!("start_line {start} is past the end: the file has {total} lines"));
    }
    if end < start {
        return Err(format!("end_line {end} is before start_line {start}"));
    }
    Ok(format!("[lines {start}-{end} of {total}]\n{}", lines[start - 1..end].join("\n")))
}

fn spec(name: &'static str) -> ToolSpec {
    match name {
        "list_files" => ToolSpec::of::<ListArgs>(name, "List the project's files and folders, recursively."),
        "read_file" => ToolSpec::of::<ReadArgs>(
            name,
            "Read a text file from the project, optionally only some of its lines.",
        ),
        "write_file" => ToolSpec::of::<WriteArgs>(
            name,
            "Create a file or replace its whole contents. Creates folders as needed.",
        ),
        "edit_file" => ToolSpec::of::<EditArgs>(
            name,
            "Replace one exact piece of text in a file. Prefer this to write_file for small changes.",
        ),
        "run_command" => ToolSpec::of::<CommandArgs>(
            name,
            "Run a shell command in an isolated sandbox at the project root, e.g. to run the tests. \
             Returns the exit code and output.",
        ),
        "submit" => ToolSpec::of::<SubmitArgs>(name, "Finish your task and hand the work to the reviewer."),
        "submit_review" => ToolSpec::of::<Review>(name, "Finish the review with your verdict."),
        "submit_plan" => ToolSpec::of::<Plan>(name, "Finish planning by submitting the task list."),
        other => unreachable!("no spec for tool {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::line_range;

    #[test]
    fn line_ranges_are_one_based_and_inclusive() {
        let text = "a\nb\nc\nd";
        assert_eq!(line_range(text, Some(2), Some(3)).unwrap(), "[lines 2-3 of 4]\nb\nc");
        assert_eq!(line_range(text, Some(3), None).unwrap(), "[lines 3-4 of 4]\nc\nd");
        assert_eq!(line_range(text, None, Some(99)).unwrap(), "[lines 1-4 of 4]\na\nb\nc\nd");
        assert!(line_range(text, Some(9), None).unwrap_err().contains("4 lines"));
        assert!(line_range(text, Some(3), Some(2)).unwrap_err().contains("before"));
    }
}
