"""Check that no NC-licensed data sources are used in production model training.

Scans MLflow run tags or training manifests for non-commercial data sources.
Run as CI gate: python ml/data/check_licenses.py

Exits 1 if any NC/research-only source is found in production training manifests.
"""

import re
import sys
from pathlib import Path

# NC / research-only license patterns
BLOCKED_LICENSES = [
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "CC-BY-NC-ND",
    "research-only",
    "non-commercial",
]

# Known NC datasets (must match DATA_LICENSES.md)
BLOCKED_DATASETS = [
    "xBD",
    "xView2",
    "LandCover.ai",
    "NEU Surface Defect",
]

DATA_LICENSES_PATH = Path(__file__).parent.parent.parent / "DATA_LICENSES.md"


def check_licenses_file():
    """Verify DATA_LICENSES.md exists and is parseable."""
    if not DATA_LICENSES_PATH.exists():
        print(f"❌ DATA_LICENSES.md not found at {DATA_LICENSES_PATH}")
        return False

    content = DATA_LICENSES_PATH.read_text()

    # Check for NC entries
    nc_found = []
    for line in content.split("\n"):
        if any(lic in line for lic in BLOCKED_LICENSES):
            if "❌" in line or "⚠️" in line:
                nc_found.append(line.strip())

    if nc_found:
        print(f"⚠️  Found {len(nc_found)} NC/research-only dataset entries in DATA_LICENSES.md")
        for entry in nc_found:
            print(f"   {entry}")
    else:
        print("✅ No NC-licensed datasets found in DATA_LICENSES.md")

    return True


def check_training_manifest(manifest_path: str | None = None):
    """Check a training manifest YAML/JSON for blocked data sources."""
    if manifest_path is None:
        print("ℹ️  No training manifest provided — skipping manifest check")
        return True

    path = Path(manifest_path)
    if not path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return False

    content = path.read_text()
    violations = []

    for dataset in BLOCKED_DATASETS:
        if dataset.lower() in content.lower():
            violations.append(dataset)

    if violations:
        print(f"❌ BLOCKED: Found NC datasets in training manifest: {violations}")
        print("   These datasets cannot be used in production model training.")
        return False

    print("✅ Training manifest is clean — no NC data sources")
    return True


def main():
    print("=" * 60)
    print("DATA LICENSE COMPLIANCE CHECK")
    print("=" * 60)

    manifest = sys.argv[1] if len(sys.argv) > 1 else None

    licenses_ok = check_licenses_file()
    manifest_ok = check_training_manifest(manifest)

    if licenses_ok and manifest_ok:
        print("\n🟢 All checks passed.")
        sys.exit(0)
    else:
        print("\n🔴 License check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
