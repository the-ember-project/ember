"""Symlink-safe file access for the trusted code (the code that holds the keys).

The agent's commands can create files anywhere in its own folders, including symbolic links
("symlinks", files that point at another file). If trusted code opened such a link, it would
read or write whatever the link points at: `/proc/self/environ` (the trusted process's own
secrets), git's settings, or a protected file. So trusted code opens the agent's files only
through these helpers, which walk the path one folder at a time and never follow a link:

- A read of a path that is a link, passes through a linked folder, or isn't a plain file
  (a named pipe, a device) gives None, as if the file were missing.
- A write or append replaces a link or other non-file at the final name with a plain file,
  and refuses a path that passes through a linked folder.
- Anything outside the repository is refused.

This only works on systems with `O_NOFOLLOW` and `dir_fd` support (Linux, macOS), which is
where the agent runs.
"""

import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Open flags that refuse to follow a link at the final name.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class UnsafePath(OSError):
    pass


def _parts(path) -> tuple:
    """The path's parts relative to the repository, without resolving any links."""
    p = Path(os.path.normpath(os.path.abspath(path)))
    try:
        parts = p.relative_to(ROOT).parts
    except ValueError:
        raise UnsafePath(f"{path} is outside the repository") from None
    if not parts:
        raise UnsafePath(f"{path} is the repository itself, not a file")
    return parts


def _open_parent(parts: tuple, create: bool = False, dir_mode=None) -> int:
    """Open the folder holding the last part, one folder at a time, never through a link."""
    fd = os.open(ROOT, os.O_RDONLY | _DIRECTORY)
    try:
        for part in parts[:-1]:
            try:
                nfd = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o777, dir_fd=fd)
                nfd = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
                if dir_mode is not None:
                    os.fchmod(nfd, dir_mode)
            os.close(fd)
            fd = nfd
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_bytes(path):
    """The file's bytes, or None if it is missing, a link, behind a link, or not a plain file."""
    try:
        parts = _parts(path)
        parent = _open_parent(parts)
    except OSError:
        return None
    try:
        # O_NONBLOCK so a named pipe can't make this wait forever; plain files ignore it.
        fd = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError:
        return None
    finally:
        os.close(parent)
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None
        return f.read()


def read_text(path, errors: str = "replace"):
    """The file's text, or None (see read_bytes)."""
    data = read_bytes(path)
    return None if data is None else data.decode("utf-8", errors)


def is_file(path) -> bool:
    """True if the path is a plain file reached without passing through any link."""
    try:
        parts = _parts(path)
        parent = _open_parent(parts)
    except OSError:
        return False
    try:
        return stat.S_ISREG(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False).st_mode)
    except OSError:
        return False
    finally:
        os.close(parent)


def list_dir(path) -> list:
    """(name, kind, size) for each entry of a folder reached without passing through a link.
    kind is "dir", "file", "link" or "other"; links are listed but not followed."""
    p = Path(os.path.normpath(os.path.abspath(path)))
    if p == ROOT:
        fd = os.open(ROOT, os.O_RDONLY | _DIRECTORY)
    else:
        parts = _parts(p)
        parent = _open_parent(parts)
        try:
            fd = os.open(parts[-1], os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
    try:
        entries = []
        for name in sorted(os.listdir(fd)):
            st = os.stat(name, dir_fd=fd, follow_symlinks=False)
            kind = ("dir" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode)
                    else "link" if stat.S_ISLNK(st.st_mode) else "other")
            entries.append((name, kind, st.st_size))
        return entries
    finally:
        os.close(fd)


def open_write(path, append: bool = False, mode=None, dir_mode=None, readable: bool = False):
    """A binary file object for writing (and reading, if readable), created if needed with its
    folders. A link or other non-file at the final name is removed first; a linked folder on
    the way is refused."""
    parts = _parts(path)
    try:
        parent = _open_parent(parts, create=True, dir_mode=dir_mode)
    except OSError as e:
        raise UnsafePath(f"{path}: a folder on the way is missing, a link or unusable ({e})") from None
    try:
        name = parts[-1]
        try:
            if not stat.S_ISREG(os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode):
                os.unlink(name, dir_fd=parent)
        except FileNotFoundError:
            pass
        flags = ((os.O_RDWR if readable else os.O_WRONLY) | os.O_CREAT | _NOFOLLOW
                 | (os.O_APPEND if append else os.O_TRUNC))
        fd = os.open(name, flags, 0o666, dir_fd=parent)
    finally:
        os.close(parent)
    f = os.fdopen(fd, ("a" if append else "w") + ("+b" if readable else "b"))
    if mode is not None:
        try:
            os.fchmod(f.fileno(), mode)
        except PermissionError:
            pass  # a file someone else owns keeps its own permissions
    return f


def write_bytes(path, data: bytes, append: bool = False, mode=None, dir_mode=None) -> None:
    with open_write(path, append=append, mode=mode, dir_mode=dir_mode) as f:
        f.write(data)


def write_text(path, text: str, append: bool = False, mode=None, dir_mode=None) -> None:
    write_bytes(path, text.encode("utf-8"), append=append, mode=mode, dir_mode=dir_mode)


def append_text(path, text: str) -> None:
    write_text(path, text, append=True)
