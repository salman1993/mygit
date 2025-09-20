import sys
import argparse
import zlib
from pathlib import Path
from dataclasses import dataclass
import hashlib


@dataclass
class Object:
    type: str
    size: int
    content: bytes

    def construct_data(self) -> bytes:
        """Canonical git object representation"""
        header = f"{self.type} {self.size}".encode("utf-8")
        return header + b"\0" + self.content


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

    print(f"Initialized empty Git repository in {gitdir}")


def _obj_path(obj: str) -> Path:
    p = Path(".git").joinpath("objects", obj[:2], obj[2:])
    if not p.exists():
        raise FileNotFoundError(f"Object does not exist at {p}")
    return p


def _zlib_compress(obj: Object, output_path: Path):
    data = obj.construct_data()
    compressed_data = zlib.compress(data)
    with open(output_path, "wb") as f_out:
        f_out.write(compressed_data)


def _zlib_decompress(input_path: Path) -> Object:
    with open(input_path, "rb") as f_in:
        compressed_data = f_in.read()
    decompressed = zlib.decompress(compressed_data)
    header, body = decompressed.split(b"\0", 2)
    typ, size = header.decode().split(" ")
    return Object(type=typ, size=int(size), content=body)


def cat_file(t: bool, s: bool, p: bool, obj_hash: str):
    objpath = _obj_path(obj_hash)
    obj = _zlib_decompress(objpath)
    if t:
        print(obj.type)
    elif s:
        print(obj.size)
    elif p:
        print(obj.content.decode())
    else:
        print("Unknown type")
        sys.exit(1)


def hash_object(typ: str, filepath: str, should_write: bool):
    with open(filepath, "rb") as f_in:
        content = f_in.read()
    obj = Object(type=typ, size=len(content), content=content)
    digest = compute_digest(obj)
    if should_write:
        dirpath = Path(".git/objects").joinpath(digest[:2])
        dirpath.mkdir(exist_ok=True)
        output_path = dirpath.joinpath(digest[2:])
        _zlib_compress(obj, output_path=output_path)
        print(f"Saved zlib compressed file at {output_path}")


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
    sp.add_argument("directory")

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

    args = parser.parse_args()

    match args.command:
        case "init":
            return init(repo_dir=args.directory)
        case "cat-file":
            return cat_file(t=args.t, s=args.s, p=args.p, obj_hash=args.object)
        case "hash-object":
            return hash_object(
                typ=args.type, filepath=args.file, should_write=args.write
            )
        case _:
            print("Unknown command")
            sys.exit(1)
