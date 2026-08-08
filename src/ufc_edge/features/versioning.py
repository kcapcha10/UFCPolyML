"""Feature versioning and source-hash integrity guard.

Computes a deterministic SHA-256 hash over all Python source files in the
features package, compares it against a committed manifest, and fails loudly
on mismatch to prevent training on stale feature implementations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

FEATURE_VERSION: str = "v1"

_DEFAULT_MANIFEST_FILENAME = "features_version_manifest.json"


@dataclass(frozen=True)
class VersionManifest:
    """Recorded feature version with its source hash and changelog."""

    version: str
    source_hash: str
    changelog: str
    created_at: str | None = None


@dataclass(frozen=True)
class IntegrityResult:
    """Outcome of a version integrity check."""

    ok: bool
    message: str
    expected_hash: str | None = None
    actual_hash: str | None = None


def compute_source_hash(features_dir: Path) -> str:
    """Compute a deterministic SHA-256 hash over all .py files in the directory tree.

    Files are sorted by their path relative to features_dir to ensure
    platform-independent ordering. Each file contributes its relative path
    (as UTF-8) and raw content bytes to the running hash.
    """
    hasher = hashlib.sha256()
    py_files = sorted(features_dir.rglob("*.py"), key=lambda p: p.relative_to(features_dir))

    for py_file in py_files:
        relative_path = py_file.relative_to(features_dir).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(py_file.read_bytes())

    return hasher.hexdigest()


def write_manifest(manifest: VersionManifest, path: Path) -> None:
    """Serialize manifest to JSON at the given path."""
    data = {
        "version": manifest.version,
        "source_hash": manifest.source_hash,
        "changelog": manifest.changelog,
        "created_at": manifest.created_at or datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_manifest(path: Path) -> VersionManifest | None:
    """Read manifest from JSON. Returns None if file does not exist."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return VersionManifest(
        version=data["version"],
        source_hash=data["source_hash"],
        changelog=data["changelog"],
        created_at=data.get("created_at"),
    )


def check_version_integrity(
    features_dir: Path,
    manifest_path: Path,
    expected_version: str | None = None,
) -> IntegrityResult:
    """Verify that current source hash matches the committed manifest.

    Checks manifest existence, optional version match, and source-hash equality.
    Returns an IntegrityResult with diagnostic info on failure.
    """
    manifest = read_manifest(manifest_path)
    if manifest is None:
        return IntegrityResult(
            ok=False,
            message=f"Manifest file not found at {manifest_path}",
        )

    version_to_check = expected_version or FEATURE_VERSION
    if manifest.version != version_to_check:
        return IntegrityResult(
            ok=False,
            message=(
                f"Version mismatch: manifest has '{manifest.version}' "
                f"but expected '{version_to_check}'"
            ),
        )

    actual_hash = compute_source_hash(features_dir)
    if actual_hash != manifest.source_hash:
        return IntegrityResult(
            ok=False,
            message=(
                f"Source hash mismatch for {manifest.version}: "
                f"expected {manifest.source_hash}, got {actual_hash}. "
                f"Feature code changed without a version bump."
            ),
            expected_hash=manifest.source_hash,
            actual_hash=actual_hash,
        )

    return IntegrityResult(
        ok=True,
        message=f"Integrity check passed for {manifest.version}",
        expected_hash=manifest.source_hash,
        actual_hash=actual_hash,
    )
