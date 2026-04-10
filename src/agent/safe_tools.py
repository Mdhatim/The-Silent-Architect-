from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolAction:
    name: str
    detail: str
    output: str


class SafetyError(Exception):
    pass


def get_output_root(repo_root: Path) -> Path:
    return (repo_root / "output").resolve()


def _ensure_within_output(repo_root: Path, user_path: str) -> Path:
    out_root = get_output_root(repo_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Normalize: allow user to pass "foo.py" or "output/foo.py"
    p = Path(user_path.strip().lstrip("/\\"))
    if p.parts and p.parts[0].lower() == "output":
        p = Path(*p.parts[1:])

    target = (out_root / p).resolve()
    if out_root not in target.parents and target != out_root:
        raise SafetyError("Refusing to write outside output/ folder.")
    return target


def create_file(repo_root: Path, user_path: str, overwrite: bool = False) -> ToolAction:
    target = _ensure_within_output(repo_root, user_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        return ToolAction(
            name="create_file",
            detail=f"File already exists: {target.relative_to(repo_root)}",
            output=str(target.relative_to(repo_root)),
        )

    target.write_text("", encoding="utf-8")
    return ToolAction(
        name="create_file",
        detail=f"Created file: {target.relative_to(repo_root)}",
        output=str(target.relative_to(repo_root)),
    )


def write_text(repo_root: Path, user_path: str, content: str, append: bool = False) -> ToolAction:
    target = _ensure_within_output(repo_root, user_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(target, mode, encoding="utf-8", newline="\n") as f:
        f.write(content)

    return ToolAction(
        name="write_text",
        detail=f"Wrote {'appended' if append else 'saved'} content to: {target.relative_to(repo_root)}",
        output=str(target.relative_to(repo_root)),
    )

