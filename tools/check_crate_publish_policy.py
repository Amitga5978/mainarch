#!/usr/bin/env python3
"""Reject Cargo package manifests that can publish without an explicit plan."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any


IGNORED_DIRS = {".git", "target"}


def cargo_manifests(root: Path) -> list[Path]:
    manifests: list[Path] = []
    for path in root.rglob("Cargo.toml"):
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        manifests.append(path)
    return sorted(manifests)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def workspace_package_publish_is_false(manifest: dict[str, Any]) -> bool:
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        return False
    package = workspace.get("package")
    if not isinstance(package, dict):
        return False
    return package.get("publish") is False


def workspace_publish_failures(path: Path, manifest: dict[str, Any]) -> list[str]:
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict) or "members" not in workspace:
        return []
    if workspace_package_publish_is_false(manifest):
        return []
    return [
        f"{path}: workspace with members must set workspace.package.publish = false"
    ]


def nearest_workspace_publish_is_false(
    root: Path,
    manifest_path: Path,
    manifests: dict[Path, dict[str, Any]],
) -> bool:
    current = manifest_path.parent
    while True:
        candidate = current / "Cargo.toml"
        manifest = manifests.get(candidate)
        if manifest is not None and isinstance(manifest.get("workspace"), dict):
            return workspace_package_publish_is_false(manifest)
        if current == root or current == current.parent:
            return False
        current = current.parent


def package_publish_failures(
    path: Path,
    manifest: dict[str, Any],
    inherited_publish_false: bool,
) -> list[str]:
    package = manifest.get("package")
    if not isinstance(package, dict):
        return []

    publish = package.get("publish")
    if publish is False:
        return []
    if isinstance(publish, dict) and publish.get("workspace") is True:
        if inherited_publish_false:
            return []
        return [
            f"{path}: package uses publish.workspace=true but nearest workspace "
            "does not set workspace.package.publish = false"
        ]
    return [
        f"{path}: package must set publish = false or inherit publish.workspace = true "
        "from a workspace with workspace.package.publish = false"
    ]


def check(root: Path) -> int:
    manifest_paths = cargo_manifests(root)
    manifests = {path: load_manifest(path) for path in manifest_paths}
    failures: list[str] = []
    package_count = 0

    for path, manifest in manifests.items():
        failures.extend(workspace_publish_failures(path, manifest))
        if isinstance(manifest.get("package"), dict):
            package_count += 1
        failures.extend(
            package_publish_failures(
                path,
                manifest,
                nearest_workspace_publish_is_false(root, path, manifests),
            )
        )

    if failures:
        print("crate publish policy violation:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "crate publish policy ok: "
        f"scanned {len(manifest_paths)} Cargo.toml files and {package_count} packages"
    )
    return 0


def self_test() -> int:
    root = Path("/repo")
    manifests = {
        root / "Cargo.toml": {
            "workspace": {"members": ["crates/*"], "package": {"publish": False}}
        },
        root / "crates" / "mainarch-core" / "Cargo.toml": {
            "package": {"publish": {"workspace": True}}
        },
        root / "examples" / "plugin" / "Cargo.toml": {
            "workspace": {"resolver": "2"},
            "package": {"publish": False},
        },
    }

    if workspace_publish_failures(root / "Cargo.toml", manifests[root / "Cargo.toml"]):
        print("self-test rejected workspace publish=false", file=sys.stderr)
        return 1

    inherited_ok = nearest_workspace_publish_is_false(
        root, root / "crates" / "mainarch-core" / "Cargo.toml", manifests
    )
    if not inherited_ok:
        print("self-test failed to detect inherited publish=false", file=sys.stderr)
        return 1

    accepted = (
        ({"publish": False}, True),
        ({"publish": {"workspace": True}}, True),
    )
    for package, inherited in accepted:
        failures = package_publish_failures(
            Path("accepted/Cargo.toml"), {"package": package}, inherited
        )
        if failures:
            print(f"self-test rejected accepted package: {failures}", file=sys.stderr)
            return 1

    rejected = (
        ({}, True),
        ({"publish": True}, True),
        ({"publish": ["crates-io"]}, True),
        ({"publish": {"workspace": True}}, False),
    )
    for package, inherited in rejected:
        if not package_publish_failures(
            Path("rejected/Cargo.toml"), {"package": package}, inherited
        ):
            print(f"self-test accepted publishable package: {package}", file=sys.stderr)
            return 1

    if not workspace_publish_failures(
        Path("workspace/Cargo.toml"), {"workspace": {"members": ["crates/*"]}}
    ):
        print("self-test accepted workspace without publish=false", file=sys.stderr)
        return 1

    print("crate publish policy self-test ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    return check(args.root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
