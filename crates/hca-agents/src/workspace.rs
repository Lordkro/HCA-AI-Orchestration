//! A project folder that agents can read and write, and nothing outside it.
//!
//! Every path an agent gives is checked before use:
//!
//! - relative only: no absolute paths, no `..`;
//! - never a path with a `.git` component, so an agent can't plant a git
//!   hook or read a token from `.git/config`;
//! - symlinks are never followed, including directories along the path;
//! - only regular files are read or written, so a fifo can't stall a worker;
//! - size limits on reads and writes.

use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use rustix::fs::{fstat, mkdirat, openat, FileType, Mode, OFlags, CWD};

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
    /// The path contains a `.git` component.
    #[error("{0:?} is inside .git, which agents can't access")]
    GitDir(String),
    /// A symlink is in the path. Symlinks are never followed.
    #[error("{0:?} is a symlink, and symlinks are not followed")]
    Symlink(String),
    /// The path is a directory, fifo, socket, or other non-regular file.
    #[error("{0:?} is not a regular file")]
    NotAFile(String),
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
    /// The returned path is not canonicalized through symlinks. Every
    /// component that already exists is opened with `O_NOFOLLOW`.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is empty, absolute, contains
    /// `..` or a `.git` component, or reaches a symlink.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, WorkspaceError> {
        let parts = relative_parts(path)?;
        reject_symlinks(&self.root, &parts, path)?;
        Ok(self.root.join(PathBuf::from_iter(parts.iter())))
    }

    /// Reads a text file.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is refused, missing, not a
    /// regular file, too large, or not UTF-8.
    pub async fn read(&self, path: &str) -> Result<String, WorkspaceError> {
        let parts = relative_parts(path)?;
        let root = self.root.clone();
        let label = path.to_owned();
        let limit = self.max_read_bytes;
        let bytes = spawn_io(path, move || read_at(&root, &parts, &label, limit)).await?;
        String::from_utf8(bytes).map_err(|_| WorkspaceError::NotText(path.to_owned()))
    }

    /// Creates or replaces a file, creating folders as needed.
    ///
    /// # Errors
    ///
    /// Returns a [`WorkspaceError`] if the path is refused, the content is
    /// too large, the destination is not a regular file, or the write fails.
    pub async fn write(&self, path: &str, content: &str) -> Result<(), WorkspaceError> {
        let parts = relative_parts(path)?;
        let bytes = u64::try_from(content.len()).unwrap_or(u64::MAX);
        if bytes > self.max_write_bytes {
            return Err(WorkspaceError::TooLarge {
                path: path.to_owned(),
                bytes,
                limit: self.max_write_bytes,
            });
        }
        let root = self.root.clone();
        let label = path.to_owned();
        let content = content.to_owned();
        spawn_io(path, move || write_at(&root, &parts, &label, &content)).await
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

fn io_error(path: &str, source: io::Error) -> WorkspaceError {
    WorkspaceError::Io { path: path.to_owned(), source }
}

fn relative_parts(path: &str) -> Result<Vec<OsString>, WorkspaceError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(WorkspaceError::EmptyPath);
    }
    let requested = Path::new(trimmed);
    if requested.is_absolute() {
        return Err(WorkspaceError::Absolute(path.to_owned()));
    }
    let mut parts = Vec::new();
    for component in requested.components() {
        match component {
            Component::Normal(part) => {
                if part == ".git" {
                    return Err(WorkspaceError::GitDir(path.to_owned()));
                }
                parts.push(part.to_os_string());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(WorkspaceError::Escapes(path.to_owned()));
            }
        }
    }
    Ok(parts)
}

fn map_errno(path: &str, errno: rustix::io::Errno) -> WorkspaceError {
    match errno {
        rustix::io::Errno::LOOP => WorkspaceError::Symlink(path.to_owned()),
        rustix::io::Errno::NOENT => WorkspaceError::NotFound(path.to_owned()),
        other => io_error(path, io::Error::from(other)),
    }
}

fn map_io(path: &str, error: io::Error) -> WorkspaceError {
    rustix::io::Errno::from_io_error(&error)
        .map_or_else(|| io_error(path, error), |errno| map_errno(path, errno))
}

// Linux returns ENOTDIR, not ELOOP, when O_NOFOLLOW is combined with
// O_DIRECTORY, so a symlink directory would look like a plain I/O error and
// the caller could not tell it was a symlink. Open with O_NOFOLLOW only, then
// require a directory. O_NONBLOCK keeps a fifo from stalling that open.
const DIR_FLAGS: OFlags =
    OFlags::RDONLY.union(OFlags::NOFOLLOW).union(OFlags::CLOEXEC).union(OFlags::NONBLOCK);

fn open_root(root: &Path) -> Result<rustix::fd::OwnedFd, rustix::io::Errno> {
    openat(CWD, root, OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC, Mode::empty())
}

fn open_dir(parent: &rustix::fd::OwnedFd, name: &OsStr) -> Result<rustix::fd::OwnedFd, rustix::io::Errno> {
    let fd = openat(parent, name, DIR_FLAGS, Mode::empty())?;
    let kind = FileType::from_raw_mode(fstat(&fd)?.st_mode);
    if kind.is_symlink() {
        return Err(rustix::io::Errno::LOOP);
    }
    if !kind.is_dir() {
        return Err(rustix::io::Errno::NOTDIR);
    }
    Ok(fd)
}

fn ensure_dir(
    parent: &rustix::fd::OwnedFd,
    name: &OsStr,
    path: &str,
) -> Result<rustix::fd::OwnedFd, WorkspaceError> {
    match open_dir(parent, name) {
        Ok(fd) => Ok(fd),
        Err(rustix::io::Errno::NOENT) => {
            match mkdirat(parent, name, Mode::from_raw_mode(0o755)) {
                Ok(()) | Err(rustix::io::Errno::EXIST) => {}
                Err(error) => return Err(map_errno(path, error)),
            }
            open_dir(parent, name).map_err(|error| map_errno(path, error))
        }
        Err(error) => Err(map_errno(path, error)),
    }
}

fn descend(
    root: &Path,
    parents: &[OsString],
    path: &str,
    create: bool,
) -> Result<rustix::fd::OwnedFd, WorkspaceError> {
    let mut dir = open_root(root).map_err(|error| map_errno(path, error))?;
    for name in parents {
        dir = if create {
            ensure_dir(&dir, name, path)?
        } else {
            open_dir(&dir, name).map_err(|error| map_errno(path, error))?
        };
    }
    Ok(dir)
}

fn reject_symlinks(root: &Path, parts: &[OsString], path: &str) -> Result<(), WorkspaceError> {
    let Some((last, parents)) = parts.split_last() else {
        return Ok(());
    };
    let mut dir = open_root(root).map_err(|error| map_errno(path, error))?;
    for name in parents {
        match open_dir(&dir, name) {
            Ok(child) => dir = child,
            Err(rustix::io::Errno::NOENT) => return Ok(()),
            Err(error) => return Err(map_errno(path, error)),
        }
    }
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    match openat(&dir, last, flags, Mode::empty()) {
        Ok(_) | Err(rustix::io::Errno::NOENT) => Ok(()),
        Err(error) => Err(map_errno(path, error)),
    }
}

fn require_file(file: &std::fs::File, path: &str) -> Result<u64, WorkspaceError> {
    let stat = fstat(file).map_err(|error| map_errno(path, error))?;
    if !FileType::from_raw_mode(stat.st_mode).is_file() {
        return Err(WorkspaceError::NotAFile(path.to_owned()));
    }
    Ok(u64::try_from(stat.st_size).unwrap_or(u64::MAX))
}

fn read_at(root: &Path, parts: &[OsString], path: &str, limit: u64) -> Result<Vec<u8>, WorkspaceError> {
    let Some((name, parents)) = parts.split_last() else {
        return Err(WorkspaceError::NotAFile(path.to_owned()));
    };
    let dir = descend(root, parents, path, false)?;
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let fd = openat(&dir, name, flags, Mode::empty()).map_err(|error| map_errno(path, error))?;
    let file = std::fs::File::from(fd);
    let size = require_file(&file, path)?;
    if size > limit {
        return Err(WorkspaceError::TooLarge { path: path.to_owned(), bytes: size, limit });
    }
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes).map_err(|error| map_io(path, error))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(WorkspaceError::TooLarge { path: path.to_owned(), bytes: size, limit });
    }
    Ok(bytes)
}

fn write_at(root: &Path, parts: &[OsString], path: &str, content: &str) -> Result<(), WorkspaceError> {
    let Some((name, parents)) = parts.split_last() else {
        return Err(WorkspaceError::NotAFile(path.to_owned()));
    };
    let dir = descend(root, parents, path, true)?;
    let flags = OFlags::RDWR | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let created = openat(&dir, name, flags | OFlags::CREATE | OFlags::EXCL, Mode::from_raw_mode(0o644));
    let fd = match created {
        Ok(fd) => fd,
        Err(rustix::io::Errno::EXIST) => {
            openat(&dir, name, flags, Mode::empty()).map_err(|error| map_errno(path, error))?
        }
        Err(error) => return Err(map_errno(path, error)),
    };
    let mut file = std::fs::File::from(fd);
    require_file(&file, path)?;
    file.set_len(0).map_err(|error| map_io(path, error))?;
    file.write_all(content.as_bytes()).map_err(|error| map_io(path, error))
}

async fn spawn_io<T: Send + 'static>(
    path: &str,
    job: impl FnOnce() -> Result<T, WorkspaceError> + Send + 'static,
) -> Result<T, WorkspaceError> {
    tokio::task::spawn_blocking(job).await.map_err(|error| io_error(path, io::Error::other(error)))?
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
        assert!(matches!(ws.resolve("src/.git/config"), Err(WorkspaceError::GitDir(_))));
        assert!(matches!(ws.resolve("src/.git"), Err(WorkspaceError::GitDir(_))));
        assert!(ws.resolve(".gitignore").is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_symlinks_that_lead_out() {
        let (ws, dir) = workspace();
        std::fs::write(dir.path().join("secret.txt"), "token").unwrap();
        std::os::unix::fs::symlink(dir.path(), ws.root().join("escape")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("secret.txt"), ws.root().join("link.txt")).unwrap();

        assert!(matches!(ws.read("escape/secret.txt").await, Err(WorkspaceError::Symlink(_))));
        assert!(matches!(ws.read("link.txt").await, Err(WorkspaceError::Symlink(_))));
        assert!(matches!(ws.write("escape/new.txt", "x").await, Err(WorkspaceError::Symlink(_))));
        assert!(!dir.path().join("new.txt").exists());

        std::fs::create_dir_all(ws.root().join(".git")).unwrap();
        std::fs::write(ws.root().join(".git/config"), "secret").unwrap();
        std::os::unix::fs::symlink(".git", ws.root().join("g")).unwrap();
        assert!(matches!(ws.read("g/config").await, Err(WorkspaceError::Symlink(_))));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn writing_a_fifo_does_not_block() {
        let (ws, _dir) = workspace();
        let status = std::process::Command::new("mkfifo").arg(ws.root().join("pipe")).status().unwrap();
        assert!(status.success(), "mkfifo");
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), ws.write("pipe", "hello")).await;
        assert!(matches!(result, Ok(Err(WorkspaceError::NotAFile(_)))), "{result:?}");
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
