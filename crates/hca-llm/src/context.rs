//! Builds a prompt from the pieces an agent needs, within a token budget.
//!
//! Instead of replaying raw chat history, each call gets a fresh, focused
//! prompt: the task, the reviewer's
//! feedback, the latest test output, the file tree and the relevant files.
//! When everything doesn't fit, lower-priority sections are cut or dropped
//! first, and the result says what was left out.

use crate::estimate_tokens;

/// How important a section is when space runs out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Always included, even over budget.
    Required,
    /// Included before normal sections.
    High,
    /// The default.
    Normal,
    /// Included only if there's room left.
    Low,
}

/// Which end of a section to keep when it has to be shortened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Truncate {
    /// Never shorten; include whole or drop.
    Never,
    /// Keep the beginning. Suits files.
    KeepStart,
    /// Keep the end. Suits logs and test output, where errors are last.
    KeepEnd,
}

#[derive(Clone, Debug)]
struct Section {
    title: String,
    body: String,
    code_block: bool,
    priority: Priority,
    truncate: Truncate,
}

/// The assembled prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltContext {
    /// The prompt text, sections in the order they were added.
    pub text: String,
    /// Estimated tokens of [`text`](Self::text).
    pub estimated_tokens: u64,
    /// Titles of sections that were shortened.
    pub truncated: Vec<String>,
    /// Titles of sections that were left out.
    pub dropped: Vec<String>,
}

/// Assembles a prompt within a token budget.
#[derive(Clone, Debug)]
pub struct ContextBuilder {
    budget_tokens: u64,
    sections: Vec<Section>,
}

/// Sections shorter than this after cutting aren't worth including.
const MIN_USEFUL_TOKENS: u64 = 64;

impl ContextBuilder {
    /// A builder for a prompt of at most about `budget_tokens`.
    pub fn new(budget_tokens: u64) -> Self {
        Self { budget_tokens, sections: Vec::new() }
    }

    /// Adds a section that is always included.
    #[must_use]
    pub fn required(self, title: impl Into<String>, body: impl Into<String>) -> Self {
        self.section(title, body, Priority::Required, Truncate::Never)
    }

    /// Adds a prose section.
    #[must_use]
    pub fn section(
        mut self,
        title: impl Into<String>,
        body: impl Into<String>,
        priority: Priority,
        truncate: Truncate,
    ) -> Self {
        self.sections.push(Section {
            title: title.into(),
            body: body.into(),
            code_block: false,
            priority,
            truncate,
        });
        self
    }

    /// Adds a file's contents, shown as a code block under its path.
    #[must_use]
    pub fn file(mut self, path: impl Into<String>, contents: impl Into<String>, priority: Priority) -> Self {
        self.sections.push(Section {
            title: path.into(),
            body: contents.into(),
            code_block: true,
            priority,
            truncate: Truncate::KeepStart,
        });
        self
    }

    /// Assembles the prompt.
    pub fn build(self) -> BuiltContext {
        let mut used = 0u64;
        let mut included: Vec<Option<String>> = vec![None; self.sections.len()];
        let mut truncated = Vec::new();
        let mut dropped = Vec::new();

        let mut order: Vec<usize> = (0..self.sections.len()).collect();
        order.sort_by_key(|&i| self.sections[i].priority);

        for i in order {
            let section = &self.sections[i];
            let full = render(section, &section.body);
            let cost = estimate_tokens(&full);
            let remaining = self.budget_tokens.saturating_sub(used);

            if section.priority == Priority::Required || cost <= remaining {
                used += cost;
                included[i] = Some(full);
                continue;
            }

            let overhead = estimate_tokens(&render(section, ""));
            let room = remaining.saturating_sub(overhead + 16);
            if section.truncate != Truncate::Never && room >= MIN_USEFUL_TOKENS {
                let shortened = cut(&section.body, room, section.truncate);
                let text = render(section, &shortened);
                used += estimate_tokens(&text);
                included[i] = Some(text);
                truncated.push(section.title.clone());
            } else {
                dropped.push(section.title.clone());
            }
        }

        let text: String = included.into_iter().flatten().collect();
        BuiltContext { estimated_tokens: estimate_tokens(&text), text, truncated, dropped }
    }
}

fn render(section: &Section, body: &str) -> String {
    if section.code_block {
        format!("### {}\n```\n{}\n```\n\n", section.title, body)
    } else {
        format!("## {}\n{}\n\n", section.title, body)
    }
}

/// Shortens `body` to about `tokens` characters. The token estimate counts one
/// token per character, so this many characters is the budget for the kept text.
/// The omission note is extra.
fn cut(body: &str, tokens: u64, keep: Truncate) -> String {
    // The inverse of `estimate_tokens`: three bytes per token, cut on a
    // character boundary.
    let keep_bytes = usize::try_from(tokens.saturating_mul(3)).unwrap_or(usize::MAX);
    if body.len() <= keep_bytes {
        return body.to_owned();
    }
    match keep {
        Truncate::KeepEnd => {
            let mut start = body.len() - keep_bytes;
            while !body.is_char_boundary(start) {
                start += 1;
            }
            let omitted = body[..start].chars().count();
            format!("[... {omitted} characters omitted ...]\n{}", &body[start..])
        }
        Truncate::KeepStart | Truncate::Never => {
            let mut end = keep_bytes;
            while !body.is_char_boundary(end) {
                end -= 1;
            }
            let omitted = body[end..].chars().count();
            format!("{}\n[... {omitted} characters omitted ...]", &body[..end])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn everything_fits_in_insertion_order() {
        let built = ContextBuilder::new(10_000)
            .required("Task", "Write fib.py")
            .file("fib.py", "print(1)", Priority::Normal)
            .section("Test output", "1 passed", Priority::High, Truncate::KeepEnd)
            .build();
        assert_eq!((built.dropped.len(), built.truncated.len()), (0, 0));
        let task = built.text.find("## Task").unwrap();
        let file = built.text.find("### fib.py").unwrap();
        let tests = built.text.find("## Test output").unwrap();
        assert!(task < file && file < tests, "sections keep the order they were added in");
    }

    #[test]
    fn required_sections_survive_any_budget() {
        let built = ContextBuilder::new(1).required("Task", "x".repeat(1_000)).build();
        assert!(built.text.contains(&"x".repeat(1_000)));
    }

    #[test]
    fn logs_keep_their_end_and_files_their_start() {
        let log = format!("{}\nFAILED test_fib", "noise ".repeat(2_000));
        let file = format!("def fib():\n{}", "    pass\n".repeat(2_000));
        let built = ContextBuilder::new(800)
            .required("Task", "Fix the failing test")
            .section("Test output", log, Priority::High, Truncate::KeepEnd)
            .file("fib.py", file, Priority::Normal)
            .build();
        assert!(built.text.contains("FAILED test_fib"));
        assert_eq!(built.truncated, vec!["Test output".to_string()]);
        assert_eq!(built.dropped, vec!["fib.py".to_string()], "no room left for the file");
        assert!(built.estimated_tokens <= 800 + 32);
    }

    #[test]
    fn low_priority_goes_first() {
        let built = ContextBuilder::new(200)
            .file("notes.md", "n".repeat(600), Priority::Low)
            .file("main.py", "m".repeat(300), Priority::High)
            .build();
        assert!(built.text.contains("### main.py"));
        assert!(
            built.dropped.contains(&"notes.md".to_string())
                || built.truncated.contains(&"notes.md".to_string())
        );
    }

    #[test]
    fn cutting_respects_multibyte_characters() {
        let cut_text = cut(&"é".repeat(1_000), 10, Truncate::KeepStart);
        assert!(cut_text.starts_with(&"é".repeat(10)));
    }
}
