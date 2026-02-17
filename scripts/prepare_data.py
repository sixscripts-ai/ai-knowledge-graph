import os
import glob
import hashlib


# ---------------------------------------------------------------------------
# Noise filters
# ---------------------------------------------------------------------------

# Directories to skip entirely (case-insensitive path component matching)
SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    "node_modules",
    "site-packages", "dist-info",
    "__pycache__",
    ".git", ".github",
    ".tox", ".mypy_cache", ".pytest_cache",
    "screenshots",       # image-only dirs
    ".agent",            # agent workflow configs, not ICT knowledge
}

# Exact filenames to always skip (case-insensitive)
SKIP_FILENAMES = {
    "license.md", "licence.md",
    "authors.md",
    "changelog.md", "changes.md",
    "contributing.md",
    "code_of_conduct.md",
    "foundry.md",            # Anthropic Foundry docs (not ICT)
    "package.json",          # npm package manifests
    "package-lock.json",
    "install.json",          # jupyterlab install spec
    "metadata.json",         # Python dist metadata
    "pyrightconfig.json",    # editor config
    "tsconfig.json",
    "setuptools.schema.json",
    "distutils.schema.json",
    "_validators.json",      # plotly validators schema
}

# Filename substrings that indicate noise (case-insensitive)
SKIP_FILENAME_CONTAINS = [
    "schema",
    "lock.json",
]

# Path substrings that indicate noise (matched against full resolved path)
SKIP_PATH_CONTAINS = [
    "/venv/",
    "/.venv/",
    "/node_modules/",
    "/site-packages/",
    "/dist-info/",
    "/__pycache__/",
    "/.git/",
]

# Minimum content length to include (skip near-empty files)
MIN_CONTENT_LENGTH = 50


def should_skip_path(filepath: str):
    """Return a reason string if the file should be skipped, else None."""

    resolved = os.path.realpath(filepath)
    basename = os.path.basename(filepath).lower()

    # Check exact filename skip list
    if basename in SKIP_FILENAMES:
        return f"skip-filename: {basename}"

    # Check filename substring patterns
    for pattern in SKIP_FILENAME_CONTAINS:
        if pattern in basename:
            return f"skip-filename-contains: {pattern}"

    # Check path component skip list
    path_lower = resolved.lower()
    for skip in SKIP_PATH_CONTAINS:
        if skip in path_lower:
            return f"skip-path: {skip}"

    # Check directory components
    parts = set(resolved.split(os.sep))
    for part in parts:
        if part.lower() in SKIP_DIRS:
            return f"skip-dir: {part}"

    return None


def collect_files(source_dirs: list[str]) -> list[str]:
    """Collect .md and .json files from source directories, resolving symlinks
    and deduplicating by real path."""

    seen_real_paths: set[str] = set()
    collected: list[str] = []

    for directory in source_dirs:
        directory = os.path.realpath(directory)
        if not os.path.isdir(directory):
            print(f"  WARNING: Directory not found: {directory}")
            continue

        for ext in ("**/*.md", "**/*.json"):
            for filepath in glob.glob(os.path.join(directory, ext), recursive=True):
                real_path = os.path.realpath(filepath)

                # Deduplicate by resolved real path
                if real_path in seen_real_paths:
                    continue
                seen_real_paths.add(real_path)

                # Apply noise filters
                skip_reason = should_skip_path(real_path)
                if skip_reason:
                    continue

                collected.append(real_path)

    return sorted(collected)


def content_hash(text: str) -> str:
    """SHA-256 hash of normalized content for content-level deduplication."""
    # Normalize whitespace for comparison
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def combine_files(source_dirs: list[str], output_file: str) -> None:
    """Combine relevant knowledge files into a single training corpus."""

    print(f"Scanning directories: {source_dirs}")
    files = collect_files(source_dirs)
    print(f"Found {len(files)} unique files after path deduplication and noise filtering.")

    seen_hashes: set[str] = set()
    included = 0
    skipped_short = 0
    skipped_dup_content = 0
    skipped_read_error = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    content = infile.read()
            except Exception as e:
                print(f"  ERROR reading {filepath}: {e}")
                skipped_read_error += 1
                continue

            # Skip near-empty files
            stripped = content.strip()
            if len(stripped) < MIN_CONTENT_LENGTH:
                skipped_short += 1
                continue

            # Content-level deduplication (same text at different paths)
            h = content_hash(stripped)
            if h in seen_hashes:
                skipped_dup_content += 1
                continue
            seen_hashes.add(h)

            # Write with full path for traceability
            outfile.write(f"\n\n--- FILE: {os.path.basename(filepath)} (from: {filepath}) ---\n\n")
            outfile.write(content)
            included += 1

    print(f"\nResults:")
    print(f"  Included:                  {included}")
    print(f"  Skipped (too short):       {skipped_short}")
    print(f"  Skipped (duplicate content):{skipped_dup_content}")
    print(f"  Skipped (read error):      {skipped_read_error}")
    print(f"  Output written to:         {output_file}")


if __name__ == "__main__":
    source_directories = [
        "/Users/villain/Documents/train-ict",
        "/Users/villain/Documents/vex-workspace",
    ]
    output_path = "data/ict_knowledge_combined.txt"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    combine_files(source_directories, output_path)
