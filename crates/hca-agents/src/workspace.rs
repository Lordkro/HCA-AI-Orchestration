//! A project folder that agents can read and write, and nothing outside it.
//!
//! Every path an agent gives is checked before use:
//!
//! - relative only: no absolute paths, no `..`;
//! - never inside `.git/`, so an agent can't plant a git hook or read a
//!   token from `.git/config` (both possible in the Python version);
//! - no symlink that leads out of the folder;
//! - size limits on reads and writes.

use std::io;
use std::path::{Component, Path, PathBuf};

/// Why a workspace operation was refused or failed. The messages are
/// written to be shown to the model.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    /// No path was given.
    #[error("a path is required")]
    EmptyPath,
    /// The path is absolute.
    #[error("{0:?} is absolute; use a path relative to the project root")]
    Absolute(String),
    /// The path leaves the workspace.
    #[error("{0:?} is outside the project")]
    Escapes(String),
    /// The path is inside `.git`.
    #[error("{0:?} is inside .git, which agents can't access")]
    GitDir(String),
    /// Nothing exists at the path.
    #[error("{0:?} does not exist")]
    NotFound(String),
    /// The file is too big.
    #[error("{path:?} is {bytes} bytes; the limit is {limit}")]
    TooLarge {
        /// The path.
        path: String,
        /// Its size.
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// The file isn't UTF-8 text.
    #[error("{0:?} is not a text file")]
    NotText(String),
    /// The text to replace wasn't found.
    #[error("the text to replace was not found in {0:?}; read the file and copy it exactly")]
    EditNotFound(String),
    /// The text to replace occurs more than once.
    #[error("the text to replace occurs {count} times in {path:?}; include more surrounding lines so it is unique")]
    EditAmbiguous {
        /// The path.
        path: String,
        /// How often it occurs.
        count: usize,
    },
    /// An I/O error.
    #[error("{path:?}: {source}")]
    Io {
        /// The path.
        path: String,
        /// The error.
        #[source]
        source: io::Error,
    },
}

/// Folders left out of listings: version control and build output.
const SKIPPED_DIRS: [&str; 6] = [".git", "node_modules", "target", "__pycache__", ".venv", "venv"];

/// A project's files.
#[derive(Clone, Debug)]
pub struct Workspace {
    root: PathBuf,
    max_read_bytes: u64,
    max_write_bytes: u64,
    max_list_entries: usize,
}

impl Workspace {
    /// Opens (creating if needed) the folder at `root`.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the folder can't be created or resolved.
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        Ok(Self {
            root: root.as_ref().canonicalize()?,
            max_read_bytes: 256 * 1024,
            max_write_bytes: 1024 * 1024,
            max_list_entries: 500,
        })
    }

    /// The folder's absolute path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Checks a path and returns where it points.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is empty, absolute, contains
    /// `..`, points into `.git`, or leads out of the folder through a
    /// symlink.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, WorkspaceError> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return Err(WorkspaceError::EmptyPath);
        }
        let requested = Path::new(trimmed);
        if requested.is_absolute() {
            return Err(WorkspaceError::Absolute(path.to_owned()));
        }

        let mut relative = PathBuf::new();
        for component in requested.components() {
            match component {
                Component::Normal(part) => relative.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(WorkspaceError::Escapes(path.to_owned()));
                }
            }
        }
        if relative.components().next().is_some_and(|c| c.as_os_str() == ".git") {
            return Err(WorkspaceError::GitDir(path.to_owned()));
        }

        // Follow symlinks on the deepest part of the path that exists and
        // make sure it is still inside the folder.
        let full = self.root.join(&relative);
        let mut existing = full.clone();
        while std::fs::symlink_metadata(&existing).is_err() {
            if !existing.pop() {
                break;
            }
        }
        let real = existing.canonicalize().map_err(|_| WorkspaceError::Escapes(path.to_owned()))?;
        if !real.starts_with(&self.root) {
            return Err(WorkspaceError::Escapes(path.to_owned()));
        }
        Ok(full)
    }

    /// Reads a text file.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is refused, missing, too
    /// large or not UTF-8.
    pub async fn read(&self, path: &str) -> Result<String, WorkspaceError> {
        let full = self.resolve(path)?;
        let metadata = tokio::fs::metadata(&full).await.map_err(|e| missing_or_io(path, e))?;
        if !metadata.is_file() {
            return Err(WorkspaceError::NotFound(path.to_owned()));
        }
        if metadata.len() > self.max_read_bytes {
            return Err(WorkspaceError::TooLarge {
                path: path.to_owned(),
                bytes: metadata.len(),
                limit: self.max_read_bytes,
            });
        }
        let bytes = tokio::fs::read(&full).await.map_err(|e| io_error(path, e))?;
        String::from_utf8(bytes).map_err(|_| WorkspaceError::NotText(path.to_owned()))
    }

    /// Creates or replaces a file, creating folders as needed.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is refused, the content is
    /// too large, or the write fails.
    pub async fn write(&self, path: &str, content: &str) -> Result<(), WorkspaceError> {
        let full = self.resolve(path)?;
        let bytes = u64::try_from(content.len()).unwrap_or(u64::MAX);
        if bytes > self.max_write_bytes {
            return Err(WorkspaceError::TooLarge {
                path: path.to_owned(),
                bytes,
                limit: self.max_write_bytes,
            });
        }
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| io_error(path, e))?;
        }
        tokio::fs::write(&full, content).await.map_err(|e| io_error(path, e))
    }

    /// Replaces the one occurrence of `old` with `new`.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::EditNotFound`] or
    /// [`WorkspaceError::EditAmbiguous`] unless `old` occurs exactly once,
    /// plus the errors of [`read`](Self::read) and [`write`](Self::write).
    pub async fn edit(&self, path: &str, old: &str, new: &str) -> Result<(), WorkspaceError> {
        if old.is_empty() {
            return Err(WorkspaceError::EditNotFound(path.to_owned()));
        }
        let content = self.read(path).await?;
        match content.matches(old).count() {
            0 => Err(WorkspaceError::EditNotFound(path.to_owned())),
            1 => self.write(path, &content.replacen(old, new, 1)).await,
            count => Err(WorkspaceError::EditAmbiguous { path: path.to_owned(), count }),
        }
    }

    /// Lists files under `path` recursively, as paths relative to the
    /// project root (folders end in `/`). Skips `.git` and build folders.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is refused or missing.
    pub async fn list(&self, path: &str) -> Result<Vec<String>, WorkspaceError> {
        let start = if matches!(path.trim(), "" | ".") { self.root.clone() } else { self.resolve(path)? };
        if !start.is_dir() {
            return Err(WorkspaceError::NotFound(path.to_owned()));
        }
        let root = self.root.clone();
        let limit = self.max_list_entries;
        let listed = tokio::task::spawn_blocking(move || walk(&root, &start, limit))
            .await
            .map_err(|e| io_error(path, io::Error::other(e)))?;
        listed.map_err(|e| io_error(path, e))
    }
}

fn walk(root: &Path, start: &Path, limit: usize) -> io::Result<Vec<String>> {
    let mut out = Vec::new();
    let mut stack = vec![start.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)?.collect::<io::Result<_>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        let mut subdirs = Vec::new();
        for entry in entries {
            let name = entry.file_name();
            let file_type = entry.file_type()?;
            let relative =
                entry.path().strip_prefix(root).unwrap_or(&entry.path()).to_string_lossy().replace('\\', "/");
            if file_type.is_dir() {
                if SKIPPED_DIRS.iter().any(|skip| name == *skip) {
                    continue;
                }
                out.push(format!("{relative}/"));
                subdirs.push(entry.path());
            } else {
                out.push(relative);
            }
            if out.len() >= limit {
                out.push(format!("... (stopped at {limit} entries)"));
                return Ok(out);
            }
        }
        stack.extend(subdirs.into_iter().rev());
    }
    out.sort();
    Ok(out)
}

fn missing_or_io(path: &str, error: io::Error) -> WorkspaceError {
    if error.kind() == io::ErrorKind::NotFound {
        WorkspaceError::NotFound(path.to_owned())
    } else {
        io_error(path, error)
    }
}

fn io_error(path: &str, source: io::Error) -> WorkspaceError {
    WorkspaceError::Io { path: path.to_owned(), source }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> (Workspace, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path().join("project")).unwrap();
        (ws, dir)
    }

    #[tokio::test]
    async fn writes_reads_and_edits() {
        let (ws, _dir) = workspace();
        ws.write("src/fib.py", "def fib(n):\n    return n\n").await.unwrap();
        ws.edit("src/fib.py", "return n", "return n if n < 2 else fib(n - 1) + fib(n - 2)").await.unwrap();
        assert!(ws.read("src/fib.py").await.unwrap().contains("fib(n - 1)"));
        assert_eq!(ws.list(".").await.unwrap(), vec!["src/", "src/fib.py"]);
    }

    #[test]
    fn refuses_paths_outside_the_project() {
        let (ws, _dir) = workspace();
        assert!(matches!(ws.resolve("../secret"), Err(WorkspaceError::Escapes(_))));
        assert!(matches!(ws.resolve("src/../../secret"), Err(WorkspaceError::Escapes(_))));
        assert!(matches!(ws.resolve("/etc/passwd"), Err(WorkspaceError::Absolute(_))));
        assert!(matches!(ws.resolve("  "), Err(WorkspaceError::EmptyPath)));
        assert!(ws.resolve("./src/main.py").is_ok());
    }

    #[test]
    fn refuses_git_internals() {
        let (ws, _dir) = workspace();
        assert!(matches!(ws.resolve(".git/hooks/pre-commit"), Err(WorkspaceError::GitDir(_))));
        assert!(matches!(ws.resolve("./.git/config"), Err(WorkspaceError::GitDir(_))));
        assert!(ws.resolve(".gitignore").is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_symlinks_that_lead_out() {
        let (ws, dir) = workspace();
        std::fs::write(dir.path().join("secret.txt"), "token").unwrap();
        std::os::unix::fs::symlink(dir.path(), ws.root().join("escape")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("secret.txt"), ws.root().join("link.txt")).unwrap();

        assert!(matches!(ws.read("escape/secret.txt").await, Err(WorkspaceError::Escapes(_))));
        assert!(matches!(ws.read("link.txt").await, Err(WorkspaceError::Escapes(_))));
        assert!(matches!(ws.write("escape/new.txt", "x").await, Err(WorkspaceError::Escapes(_))));
        assert!(!dir.path().join("new.txt").exists());
    }

    #[tokio::test]
    async fn edits_must_match_exactly_once() {
        let (ws, _dir) = workspace();
        ws.write("a.py", "x = 1\nx = 1\n").await.unwrap();
        assert!(matches!(
            ws.edit("a.py", "x = 1", "x = 2").await,
            Err(WorkspaceError::EditAmbiguous { count: 2, .. })
        ));
        assert!(matches!(ws.edit("a.py", "y = 1", "x = 2").await, Err(WorkspaceError::EditNotFound(_))));
        assert_eq!(ws.read("a.py").await.unwrap(), "x = 1\nx = 1\n", "a refused edit changes nothing");
    }

    #[tokio::test]
    async fn listing_skips_git_and_build_folders() {
        let (ws, _dir) = workspace();
        std::fs::create_dir_all(ws.root().join(".git/objects")).unwrap();
        std::fs::create_dir_all(ws.root().join("node_modules/x")).unwrap();
        ws.write("README.md", "hi").await.unwrap();
        assert_eq!(ws.list(".").await.unwrap(), vec!["README.md"]);
    }

    #[tokio::test]
    async fn reading_reports_missing_files() {
        let (ws, _dir) = workspace();
        assert!(matches!(ws.read("nope.py").await, Err(WorkspaceError::NotFound(_))));
    }
}
