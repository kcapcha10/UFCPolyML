"""Tests for feature versioning and source-hash integrity guard.

Verifies that the versioning module computes deterministic hashes over Python
source files, detects tampering, and maintains a stable manifest format.
Uses isolated temp directories with fixture .py files to avoid depending on
the concurrent state of the real features package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ufc_edge.features.versioning import (
    FEATURE_VERSION,
    VersionManifest,
    check_version_integrity,
    compute_source_hash,
    read_manifest,
    write_manifest,
)


@pytest.fixture
def feature_source_dir(tmp_path: Path) -> Path:
    """Create a temp directory with known .py fixture files simulating the features package."""
    features_dir = tmp_path / "src" / "ufc_edge" / "features"
    features_dir.mkdir(parents=True)

    (features_dir / "__init__.py").write_text('"""Features package."""\n')
    (features_dir / "versioning.py").write_text("FEATURE_VERSION = 'v1'\n")
    (features_dir / "replay.py").write_text("def replay(): pass\n")

    sub_dir = features_dir / "components"
    sub_dir.mkdir()
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "elo.py").write_text("class EloTracker: pass\n")

    return features_dir


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """Path for a temporary manifest file."""
    return tmp_path / "features_version_manifest.json"


class TestComputeSourceHash:
    """Source hash computation is deterministic and covers all .py files."""

    def test_deterministic_across_runs(self, feature_source_dir: Path) -> None:
        hash_a = compute_source_hash(feature_source_dir)
        hash_b = compute_source_hash(feature_source_dir)
        assert hash_a == hash_b

    def test_returns_hex_sha256(self, feature_source_dir: Path) -> None:
        result = compute_source_hash(feature_source_dir)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_includes_subdirectory_files(self, feature_source_dir: Path) -> None:
        hash_before = compute_source_hash(feature_source_dir)

        # Modify a file in a subdirectory
        (feature_source_dir / "components" / "elo.py").write_text("class EloTracker:\n    x = 1\n")
        hash_after = compute_source_hash(feature_source_dir)

        assert hash_before != hash_after

    def test_tampered_source_changes_hash(self, feature_source_dir: Path) -> None:
        hash_before = compute_source_hash(feature_source_dir)

        (feature_source_dir / "replay.py").write_text("def replay(): return 'TAMPERED'\n")
        hash_after = compute_source_hash(feature_source_dir)

        assert hash_before != hash_after

    def test_ignores_non_python_files(self, feature_source_dir: Path) -> None:
        hash_before = compute_source_hash(feature_source_dir)

        (feature_source_dir / "notes.txt").write_text("not a python file")
        (feature_source_dir / "data.json").write_text("{}")
        hash_after = compute_source_hash(feature_source_dir)

        assert hash_before == hash_after

    def test_order_independent_of_discovery_order(self, feature_source_dir: Path) -> None:
        """Hash uses sorted relative paths, so filesystem ordering doesn't matter."""
        hash_a = compute_source_hash(feature_source_dir)

        # Create files in a different order — hash should still be the same
        another_dir = feature_source_dir.parent / "features_copy"
        another_dir.mkdir(parents=True)

        # Write in reverse alphabetical order
        files = sorted(feature_source_dir.rglob("*.py"))
        for f in reversed(files):
            rel = f.relative_to(feature_source_dir)
            target = another_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.read_text())

        hash_b = compute_source_hash(another_dir)
        assert hash_a == hash_b


class TestManifestReadWrite:
    """Manifest JSON format is stable and round-trips correctly."""

    def test_write_creates_valid_json(self, manifest_path: Path) -> None:
        manifest = VersionManifest(
            version="v1",
            source_hash="a" * 64,
            changelog="Initial feature set",
        )
        write_manifest(manifest, manifest_path)

        data = json.loads(manifest_path.read_text())
        assert data["version"] == "v1"
        assert data["source_hash"] == "a" * 64
        assert data["changelog"] == "Initial feature set"
        assert "created_at" in data

    def test_read_roundtrips_written_manifest(self, manifest_path: Path) -> None:
        manifest = VersionManifest(
            version="v2",
            source_hash="b" * 64,
            changelog="Added Elo features",
        )
        write_manifest(manifest, manifest_path)
        loaded = read_manifest(manifest_path)

        assert loaded.version == manifest.version
        assert loaded.source_hash == manifest.source_hash
        assert loaded.changelog == manifest.changelog

    def test_manifest_format_has_required_fields(self, manifest_path: Path) -> None:
        manifest = VersionManifest(
            version="v1",
            source_hash="c" * 64,
            changelog="Test",
        )
        write_manifest(manifest, manifest_path)

        data = json.loads(manifest_path.read_text())
        required_fields = {"version", "source_hash", "changelog", "created_at"}
        assert required_fields.issubset(set(data.keys()))

    def test_read_missing_manifest_returns_none(self, tmp_path: Path) -> None:
        result = read_manifest(tmp_path / "nonexistent.json")
        assert result is None


class TestCheckVersionIntegrity:
    """Integrity check catches source-hash mismatches and passes on match."""

    def test_passes_when_hash_matches(
        self, feature_source_dir: Path, manifest_path: Path
    ) -> None:
        current_hash = compute_source_hash(feature_source_dir)
        manifest = VersionManifest(
            version="v1",
            source_hash=current_hash,
            changelog="Initial",
        )
        write_manifest(manifest, manifest_path)

        result = check_version_integrity(feature_source_dir, manifest_path)
        assert result.ok is True

    def test_fails_when_source_tampered(
        self, feature_source_dir: Path, manifest_path: Path
    ) -> None:
        current_hash = compute_source_hash(feature_source_dir)
        manifest = VersionManifest(
            version="v1",
            source_hash=current_hash,
            changelog="Initial",
        )
        write_manifest(manifest, manifest_path)

        # Tamper with a source file
        (feature_source_dir / "replay.py").write_text("def replay(): return 'EVIL'\n")

        result = check_version_integrity(feature_source_dir, manifest_path)
        assert result.ok is False
        assert current_hash in result.message
        assert result.expected_hash == current_hash
        assert result.actual_hash != current_hash

    def test_fails_when_manifest_missing(
        self, feature_source_dir: Path, tmp_path: Path
    ) -> None:
        result = check_version_integrity(
            feature_source_dir, tmp_path / "nonexistent.json"
        )
        assert result.ok is False
        assert "manifest" in result.message.lower()

    def test_fails_when_version_mismatch(
        self, feature_source_dir: Path, manifest_path: Path
    ) -> None:
        current_hash = compute_source_hash(feature_source_dir)
        manifest = VersionManifest(
            version="v999",
            source_hash=current_hash,
            changelog="Wrong version",
        )
        write_manifest(manifest, manifest_path)

        result = check_version_integrity(
            feature_source_dir, manifest_path, expected_version="v1"
        )
        assert result.ok is False
        assert "version" in result.message.lower()


class TestFeatureVersion:
    """FEATURE_VERSION constant follows spec format."""

    def test_version_is_v_prefixed_integer(self) -> None:
        assert FEATURE_VERSION.startswith("v")
        assert FEATURE_VERSION[1:].isdigit()

    def test_version_is_v1(self) -> None:
        assert FEATURE_VERSION == "v1"
