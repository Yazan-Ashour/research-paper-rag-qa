"""Move scripts into folders and rewrite the imports that point at them.

Run from the project root:
    python reorganize.py              dry run: shows what would change, touches nothing
    python reorganize.py --apply      moves the experiment/evaluation/one-off scripts
    python reorganize.py --apply --core   also moves the core modules (more imports change)

Do it on a git branch first:  git checkout -b reorganize
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT_MOVES = {
    "experiments": [
        "compare_chunking", "compare_embeddings", "compare_extractors", "compare_llms",
        "compare_llms_arabic", "compare_llms_crosslingual", "compare_reranker",
        "compare_weights", "grid_search",
    ],
    "evaluation": ["evaluate", "evaluate_arabic"],
    "processing": [
        "download_data", "prepare_arabic", "render_arabic_pdfs", "inspect_arabic_pdfs",
    ],
}
CORE_MOVES = {
    "processing": ["data_prep", "arabic_text", "arabic_corpus", "pdf_extraction"],
    "retrieval": ["vector_store", "reranker", "rag_pipeline", "highlighting"],
    "evaluation": ["reporting"],
}
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules"}
FROM_IMPORT = re.compile(r"^(\s*)from\s+(\w+)(\s+import\b)", re.M)
PLAIN_IMPORT = re.compile(r"^(\s*)import\s+(\w+)(\s+as\s+(\w+))?[ \t]*(?=#|$)", re.M)
COMMA_IMPORT = re.compile(r"^\s*import\s+[^#\n]*,[^#\n]*$", re.M)


def build_mapping(include_core):
    """Return {module_name: package} for every module that will be moved."""
    groups = [SCRIPT_MOVES] + ([CORE_MOVES] if include_core else [])
    return {module: package for group in groups for package, modules in group.items() for module in modules}


def python_files(root):
    """Yield every project .py file, skipping virtual environments and caches."""
    for path in root.rglob("*.py"):
        if not SKIP_DIRS.intersection(path.relative_to(root).parts):
            yield path


def rewrite_imports(text, mapping):
    """Return the text with imports of moved modules pointed at their new package."""
    def from_repl(match):
        indent, module, tail = match.groups()
        if module in mapping:
            return f"{indent}from {mapping[module]}.{module}{tail}"
        return match.group(0)

    def plain_repl(match):
        indent, module, _, alias = match.groups()
        if module in mapping:
            return f"{indent}import {mapping[module]}.{module} as {alias or module}"
        return match.group(0)

    return PLAIN_IMPORT.sub(plain_repl, FROM_IMPORT.sub(from_repl, text))


def comma_warnings(text, mapping):
    """Return 'import a, b' lines that mention a moved module; those need a manual edit."""
    return [
        line.strip() for line in COMMA_IMPORT.findall(text)
        if any(re.search(rf"\b{module}\b", line) for module in mapping)
    ]


def move_file(source, destination):
    """Move a file with git mv when possible so history follows it, otherwise plain move."""
    destination.parent.mkdir(exist_ok=True)
    result = subprocess.run(
        ["git", "mv", str(source), str(destination)], capture_output=True, text=True
    )
    if result.returncode != 0:
        shutil.move(str(source), str(destination))


def main():
    """Plan the moves, report them, and apply them only with --apply."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually move files and edit imports")
    parser.add_argument("--core", action="store_true", help="also move the core modules")
    args = parser.parse_args()

    root = Path.cwd()
    mapping = build_mapping(args.core)
    moves = [(root / f"{m}.py", root / package / f"{m}.py") for m, package in mapping.items()]
    missing = [str(src.name) for src, _ in moves if not src.exists()]
    moves = [(src, dst) for src, dst in moves if src.exists()]

    print(f"{'APPLYING' if args.apply else 'DRY RUN'}: {len(moves)} files to move")
    for source, destination in moves:
        print(f"  {source.name} -> {destination.parent.name}/")
    if missing:
        print(f"  (not found, skipped: {', '.join(missing)})")

    print("\nImport changes:")
    edits = {}
    warnings = []
    for path in python_files(root):
        text = path.read_text(encoding="utf-8")
        new_text = rewrite_imports(text, mapping)
        warnings += [f"{path.name}: {line}" for line in comma_warnings(text, mapping)]
        if new_text != text:
            edits[path] = new_text
            old_lines, new_lines = text.splitlines(), new_text.splitlines()
            for old, new in zip(old_lines, new_lines):
                if old != new:
                    print(f"  {path.name}: {old.strip()}  ->  {new.strip()}")

    if warnings:
        print("\nFIX BY HAND (split these into one import per line, then re-run):")
        for warning in warnings:
            print(f"  {warning}")

    if not args.apply:
        print("\nNothing changed. Re-run with --apply to do it.")
        return

    for path, new_text in edits.items():
        path.write_text(new_text, encoding="utf-8")
    for source, destination in moves:
        move_file(source, destination)
    for package in {destination.parent for _, destination in moves}:
        (package / "__init__.py").touch()

if __name__ == "__main__":
    main()