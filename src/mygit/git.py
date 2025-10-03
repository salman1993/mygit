import sys
import os
import argparse
from typing import Optional
import zlib
from pathlib import Path
from dataclasses import dataclass
import hashlib
import struct
import difflib
import functools
import operator
import time


@dataclass
class Object:
    type: str
    size: int
    content: bytes

    def construct_data(self) -> bytes:
        """Canonical git object representation"""
        header = f"{self.type} {self.size}".encode("utf-8")
        return header + b"\0" + self.content


@dataclass
class IndexEntry:
    ctime_s: int
    ctime_ns: int
    mtime_s: int
    mtime_ns: int
    dev: int
    ino: int
    mode: int  # split into 2 parts: object type, unix permissions
    uid: int
    gid: int
    size: int
    sha1: bytes
    flags: int
    path: str


def compute_digest(obj: Object) -> str:
    """Compute the SHA1 digest for the object"""
    return hashlib.sha1(obj.construct_data()).hexdigest()


def init(repo_dir: str) -> None:
    gitdir = Path(repo_dir).joinpath(".git")
    if gitdir.exists():
        print("Git directory already exists!")
        return

    for d in ["objects", "refs/heads"]:
        gitdir.joinpath(d).mkdir(parents=True)

    for f in ["config", "HEAD"]:
        gitdir.joinpath(f).touch()

    gitdir.joinpath("HEAD").write_text("ref: refs/heads/main")

    print(f"Initialized empty Git repository in {gitdir}")


def check_gitdir(func):
    """Decorator to check if current directory is a git repository."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not Path(".git").exists():
            print("fatal: not a git repository", file=sys.stderr)
            sys.exit(1)
        return func(*args, **kwargs)

    return wrapper


def _obj_path(obj: str) -> Path:
    p = Path(".git").joinpath("objects", obj[:2], obj[2:])
    if not p.exists():
        raise FileNotFoundError(f"Object does not exist at {p}")
    return p


def _zlib_compress(obj: Object, output_path: Path):
    data = obj.construct_data()
    compressed_data = zlib.compress(data)
    output_path.write_bytes(compressed_data)


def _zlib_decompress(input_path: Path) -> Object:
    compressed_data = input_path.read_bytes()
    decompressed = zlib.decompress(compressed_data)
    nul_index = decompressed.index(b"\x00")
    header = decompressed[:nul_index]
    typ, size_str = header.decode().split()
    size = int(size_str)
    data = decompressed[nul_index + 1 :]
    assert size == len(data), "expected size {}, got {} bytes".format(size, len(data))
    return Object(type=typ, size=int(size), content=data)


def _read_tree_entries(content: bytes) -> list[tuple[int, str, bytes]]:
    """Reads tree entries from the tree object content."""
    entries = []
    while content:
        inul = content.index(b"\x00")
        mode, path = content[:inul].decode().split()
        sha1 = content[inul + 1 : inul + 21]  # SHA1 hashes are 20 bytes
        entries.append((int(mode), path, sha1))
        content = content[inul + 21 :]
    return entries


@check_gitdir
def cat_file(t: bool, s: bool, p: bool, obj_hash: str):
    objpath = _obj_path(obj_hash)
    obj = _zlib_decompress(objpath)
    if t:
        print(obj.type)
    elif s:
        print(obj.size)
    elif p:
        if obj.type == "tree":
            tree_entries = _read_tree_entries(obj.content)
            for mode, path, sha1 in tree_entries:
                # we always print blob here cause we don't allow subdirectories to be added
                print(f"{mode:o} blob {sha1.hex()} {path}")
        else:
            print(obj.content.decode())
    else:
        print("Unknown type")
        sys.exit(1)


@check_gitdir
def hash_object(
    typ: str,
    content: Optional[bytes] = None,
    filepath: Optional[str] = None,
    should_write: bool = True,
):
    assert content or filepath, "Either content or filepath must be provided."
    if content is None:
        content = Path(filepath).read_bytes()
    obj = Object(type=typ, size=len(content), content=content)
    digest = compute_digest(obj)
    if should_write:
        dirpath = Path(".git/objects").joinpath(digest[:2])
        dirpath.mkdir(exist_ok=True)
        output_path = dirpath.joinpath(digest[2:])
        _zlib_compress(obj, output_path=output_path)
        # print(f"Saved zlib compressed file at {output_path}")
    return digest


def read_index() -> list[IndexEntry]:
    try:
        data = Path(".git/index").read_bytes()
    except FileNotFoundError:
        return []

    # git index format: https://manpages.ubuntu.com/manpages/plucky/man5/gitformat-index.5.html
    # The first 12 bytes are the header, the last 20 a SHA-1 hash of the index, and the bytes in between are index entries, each 62 bytes plus the length of the path and some padding.
    header = data[:12]
    entries_data = data[12:-20]
    sha1_hash = data[-20:]

    assert hashlib.sha1(header + entries_data).digest() == sha1_hash, (
        "invalid index checksum"
    )

    signature, version, num_entries = struct.unpack("!4sLL", header)
    assert signature == b"DIRC", f"invalid index signature {signature}"
    assert version == 2, f"unknown index version {version}"

    entries = []
    i = 0
    while i + 62 < len(entries_data):
        endi = i + 62
        fields = struct.unpack("!LLLLLLLLLL20sH", entries_data[i:endi])
        path_end = entries_data.index(b"\x00", endi)
        path = entries_data[endi:path_end]
        entries.append(IndexEntry(*(list(fields) + [path.decode()])))

        # 1-8 nul bytes as necessary to pad the entry to a multiple of eight bytes
        #            while keeping the name NUL-terminated.
        entry_len = ((62 + len(path) + 8) // 8) * 8
        i += entry_len

    assert len(entries) == num_entries
    return entries


def write_index(entries: list[IndexEntry]):
    signature = b"DIRC"
    version = 2
    num_entries = len(entries)

    # git index format: https://manpages.ubuntu.com/manpages/plucky/man5/gitformat-index.5.html
    # The first 12 bytes are the header, the last 20 a SHA-1 hash of the index, and the bytes in between are index entries, each 62 bytes plus the length of the path and some padding.
    header = signature + int.to_bytes(version) + int.to_bytes(num_entries)
    header = struct.pack("!4sLL", signature, version, num_entries)
    packed_entries: list[bytes] = []
    for e in entries:
        packed = (
            struct.pack(
                "!LLLLLLLLLL20sH",
                e.ctime_s,
                e.ctime_ns,
                e.mtime_s,
                e.mtime_ns,
                e.dev,
                e.ino,
                e.mode,
                e.uid,
                e.gid,
                e.size,
                e.sha1,
                e.flags,
            )
            + e.path.encode()
        )
        length = ((len(packed) + 8) // 8) * 8
        num_padding = length - len(packed)
        packed += b"\x00" * num_padding
        packed_entries.append(packed)

    entries_data = b"".join(packed_entries)
    sha1_hash = hashlib.sha1(header + entries_data).digest()

    content = header + entries_data + sha1_hash
    Path(".git/index").write_bytes(content)


@check_gitdir
def add(filepaths: list[str]):
    current_entries = read_index()

    # add paths to entries that are not touched by `filepaths`
    entries = [e for e in current_entries if e.path not in filepaths]

    for path in filepaths:
        if not Path(path).exists():
            print(f"File {path} does not exist")
            sys.exit(1)

        if len(Path(path).parts) > 1:
            print(
                "This simplified git CLI does not support adding files in a subdirectory; only root directory files are supported."
            )
            # this requires traversing filepaths bottom-up, i.e, from the deepest directories up to root (depth doesn’t really matter: we just want to see each directory before its parent)
            # https://wyag.thb.lt/#cmd-commit
            sys.exit(1)

        # add or overwrite the file to the index
        sha1 = hash_object(typ="blob", filepath=path, should_write=True)
        st = os.stat(path)
        flags = len(path.encode())
        assert flags < (1 << 12)
        entry = IndexEntry(
            int(st.st_ctime),
            0,
            int(st.st_mtime),
            0,
            st.st_dev,
            st.st_ino,
            st.st_mode,
            st.st_uid,
            st.st_gid,
            st.st_size,
            bytes.fromhex(sha1),
            flags,
            path,
        )
        entries.append(entry)

    entries.sort(key=operator.attrgetter("path"))
    write_index(entries)


@check_gitdir
def ls_files(stage: bool = False, debug: bool = False):
    """Print list of files in index (including mode, SHA-1, and stage number
    if 'stage' is True; extended metadata if 'debug' is True).
    """
    for entry in read_index():
        stage_num = (entry.flags >> 12) & 3

        if stage:
            print(
                "{:06o} {} {}\t{}".format(
                    entry.mode, entry.sha1.hex(), stage_num, entry.path
                )
            )
        else:
            print(entry.path)

        if debug:
            print(f"  ctime: {entry.ctime_s}:{entry.ctime_ns}")
            print(f"  mtime: {entry.mtime_s}:{entry.mtime_ns}")
            print(f"  dev: {entry.dev} ino: {entry.ino}")
            print(f"  uid: {entry.uid}      gid: {entry.gid}")
            print(f"  size: {entry.size}      flags: {entry.flags}")


def _cwd_filepaths() -> list[Path]:
    paths: list[Path] = []
    for p in Path(".").rglob("*"):
        if ".git" not in p.parts:
            paths.append(p)
    return paths


def _latest_commit_filepaths() -> list[Path]:
    commit_hash = get_latest_commit_hash()

    if commit_hash is None:
        return []

    # read the commit object & get the tree sha
    objpath = _obj_path(commit_hash)
    obj = _zlib_decompress(objpath)
    assert obj.type == "commit", "should be commit"
    first_line = obj.content.decode().split("\n")[0]
    typ, tree_hash = first_line.split()
    assert typ == "tree", "type of first line in commit should be tree"

    # then read the tree object & its entries
    objpath = _obj_path(tree_hash)
    obj = _zlib_decompress(objpath)
    tree_entries = _read_tree_entries(obj.content)
    paths = [Path(t[1]) for t in tree_entries]
    return paths


def _compare_cwd_files_to_index() -> tuple[
    list[Path], list[Path], list[Path], list[Path]
]:
    """Returns the deleted, modified, untracked file paths. We don't show newly added files.

    Untracked files: Files that exist in the working directory but are not in the index at all
    Added files: Files that are in the index (staged) but weren't in the most recent commit
    """
    index_entries = read_index()
    index_paths = set([Path(e.path) for e in index_entries])
    cwd_paths = set(_cwd_filepaths())
    latest_commit_paths = set(_latest_commit_filepaths())

    untracked = cwd_paths - index_paths
    deleted = index_paths - cwd_paths
    overlapping = index_paths & cwd_paths
    added = index_paths - latest_commit_paths

    modified: list[Path] = []
    index_path_hash = {e.path: e.sha1 for e in index_entries}
    for p in overlapping:
        curr_hash = hash_object(typ="blob", filepath=str(p), should_write=False)
        prev_hash = index_path_hash[str(p)].hex()
        if curr_hash != prev_hash:
            modified.append(p)

    return (
        sorted(list(deleted)),
        sorted(list(modified)),
        sorted(list(added)),
        sorted(list(untracked)),
    )


@check_gitdir
def status():
    deleted, modified, added, untracked = _compare_cwd_files_to_index()

    for p in deleted:
        print(f"deleted:   {str(p)}")
    for p in added:
        print(f"new:       {str(p)}")
    for p in modified:
        print(f"modified:  {str(p)}")

    if untracked:
        print("\nUntracked files:")
        for p in untracked:
            print(str(p))


@check_gitdir
def diff():
    deleted, modified, added, _ = _compare_cwd_files_to_index()
    entries = {e.path: e for e in read_index()}
    for p in deleted + added + modified:
        curr_content = p.read_text().splitlines()
        index_obj_hash = entries[str(p)].sha1.hex()
        objpath = _obj_path(index_obj_hash)
        obj = _zlib_decompress(objpath)
        index_content = obj.content.decode().splitlines()

        diff_lines = difflib.unified_diff(
            index_content,
            curr_content,
            fromfile=f"{p} - index",
            tofile=f"{p} - current",
        )

        for line in diff_lines:
            print(line)

        print("-" * 60)
        print()


def _write_tree_object():
    """Writes the current index entries as a tree object, as part of the commit"""
    entries = []
    for ie in read_index():
        tree_entry = f"{ie.mode:o} {ie.path}".encode() + b"\0" + ie.sha1
        entries.append(tree_entry)
    content = b"".join(entries)
    digest = hash_object(typ="tree", content=content, should_write=True)
    return digest


def get_latest_commit_hash() -> Optional[str]:
    try:
        return Path(".git/refs/heads/main").read_text()
    except Exception:
        return None


@check_gitdir
def commit(msg: str, author: str):
    tree = _write_tree_object()
    parent = get_latest_commit_hash()

    timestamp = int(time.time())
    utc_offset_seconds = time.localtime().tm_gmtoff
    offset_hours = abs(utc_offset_seconds) // 3600
    offset_minutes = (abs(utc_offset_seconds) // 60) % 60
    author_time = f"{timestamp} {'-' if utc_offset_seconds < 0 else '+'}{offset_hours:02d}{offset_minutes:02d}"

    lines = [f"tree {tree}"]
    if parent:
        lines.append(f"parent {parent}")
    lines.append(f"author {author} {author_time}")
    lines.append(f"committer {author} {author_time}")
    lines.append("")
    lines.append(msg)
    lines.append("")

    content = "\n".join(lines).encode()
    digest = hash_object(typ="commit", content=content, should_write=True)
    Path(".git/refs/heads/main").write_text(digest)
    print("committed to main branch: {:7}".format(digest))
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="A simplified Git CLI")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="<command>",
        required=True,  # Make subcommand required
    )

    # init
    sp = subparsers.add_parser("init", help="Initialize a new repo")
    sp.add_argument("directory", nargs="?", default=".")

    # cat-file
    sp = subparsers.add_parser(
        "cat-file",
        help="Provide content or type and size information for repository objects",
    )
    type_group = sp.add_mutually_exclusive_group(required=True)
    type_group.add_argument("-t", action="store_true", help="Show object type")
    type_group.add_argument("-s", action="store_true", help="Show object size")
    type_group.add_argument("-p", action="store_true", help="Show object content")
    sp.add_argument("object", help="Hash of the object, eg. commit hash")

    # hash-object
    sp = subparsers.add_parser(
        "hash-object", help="Compute object ID and optionally create a blob from a file"
    )
    sp.add_argument("-t", "--type", choices=["commit", "tree", "blob"], default="blob")
    sp.add_argument("-w", "--write", action="store_true")
    sp.add_argument("file", help="Path to the object file")

    # ls-files
    sp = subparsers.add_parser(
        "ls-files",
        help="Show information about files in the index and the working tree",
    )
    sp.add_argument("-s", "--stage", action="store_true")
    sp.add_argument("-d", "--debug", action="store_true")

    # status
    sp = subparsers.add_parser(
        "status",
        help="Show working tree status",
    )

    # diff
    sp = subparsers.add_parser(
        "diff",
        help="Show changes between commits, commit and working tree, etc",
    )

    # add
    sp = subparsers.add_parser("add", help="Add file contents to the index")
    sp.add_argument("filepaths", nargs="+", help="File paths to add")

    # commit
    sp = subparsers.add_parser(
        "commit",
        help="Create a new commit containing the current contents of the index and the given log message describing the changes.",
    )
    sp.add_argument("-m", "--msg", type=str, required=True)
    sp.add_argument("-a", "--author", type=str, default="Salman <salman@example.com>")

    args = parser.parse_args()

    match args.command:
        case "init":
            init(repo_dir=args.directory)
        case "cat-file":
            cat_file(t=args.t, s=args.s, p=args.p, obj_hash=args.object)
        case "hash-object":
            hash_object(typ=args.type, filepath=args.file, should_write=args.write)
        case "ls-files":
            ls_files(stage=args.stage, debug=args.debug)
        case "status":
            status()
        case "diff":
            diff()
        case "add":
            add(args.filepaths)
        case "commit":
            commit(args.msg, args.author)
        case _:
            print("Unknown command")
            sys.exit(1)
