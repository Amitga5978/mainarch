#!/usr/bin/env python3
"""Reject forbidden production runtime dependency families in Cargo manifests."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_FAMILIES = (
    "rocm",
    "hip",
    "hsa",
    "rccl",
    "cuda",
    "pytorch",
    "torch",
    "vllm",
    "sglang",
)


def cargo_manifests(root: Path) -> list[Path]:
    ignored_dirs = {".git", "target"}
    manifests: list[Path] = []
    for path in root.rglob("Cargo.toml"):
        if any(part in ignored_dirs for part in path.relative_to(root).parts):
            continue
        manifests.append(path)
    return sorted(manifests)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def dependency_tables(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    workspace = manifest.get("workspace")
    if isinstance(workspace, dict):
        workspace_deps = workspace.get("dependencies")
        if isinstance(workspace_deps, dict):
            yield "workspace.dependencies", workspace_deps

    for section in ("dependencies", "build-dependencies", "dev-dependencies"):
        deps = manifest.get(section)
        if isinstance(deps, dict):
            yield section, deps

    targets = manifest.get("target")
    if isinstance(targets, dict):
        for target_name, target in targets.items():
            if not isinstance(target, dict):
                continue
            for section in ("dependencies", "build-dependencies", "dev-dependencies"):
                deps = target.get(section)
                if isinstance(deps, dict):
                    yield f"target.{target_name}.{section}", deps


def candidate_values(name: str, spec: Any) -> Iterable[tuple[str, str]]:
    yield "dependency", name
    if isinstance(spec, dict):
        for field in ("package", "git", "registry"):
            value = spec.get(field)
            if isinstance(value, str):
                yield field, value
        features = spec.get("features")
        if isinstance(features, list):
            for index, feature in enumerate(features):
                if isinstance(feature, str):
                    yield f"features[{index}]", feature


def token_match(value: str, family: str) -> bool:
    lowered = value.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    tokens = {token for token in normalized.split("-") if token}

    if family in {"hip", "hsa"}:
        return family in tokens or normalized.startswith(f"{family}-")
    if family == "torch":
        return family in tokens or normalized.startswith("torch-") or "-torch-" in normalized
    return family in lowered


def violations_for_dependency(name: str, spec: Any) -> list[str]:
    violations: list[str] = []
    for field, value in candidate_values(name, spec):
        for family in FORBIDDEN_FAMILIES:
            if token_match(value, family):
                violations.append(f"{field}={value!r} matched forbidden family {family!r}")
    return violations


def check(root: Path) -> int:
    failures: list[str] = []
    manifest_count = 0
    dependency_count = 0

    for manifest_path in cargo_manifests(root):
        manifest_count += 1
        manifest = load_manifest(manifest_path)
        for table_name, deps in dependency_tables(manifest):
            for dep_name, dep_spec in sorted(deps.items()):
                dependency_count += 1
                for violation in violations_for_dependency(dep_name, dep_spec):
                    failures.append(f"{manifest_path}:{table_name}:{dep_name}: {violation}")

    if failures:
        print("runtime dependency policy violation:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "runtime dependency policy ok: "
        f"scanned {manifest_count} Cargo.toml files and {dependency_count} dependency entries"
    )
    return 0


def self_test() -> int:
    allowed = [
        ("sha2", "0.10"),
        ("mainarch-sys", {"path": "../mainarch-sys"}),
        ("shipyard", "0.7"),
        ("clap", {"version": "4", "features": ["derive"]}),
    ]
    blocked = [
        ("rocm-smi", "0.1"),
        ("hip-runtime", "0.1"),
        ("hsa-runtime-sys", "0.1"),
        ("rccl-sys", "0.1"),
        ("cuda-driver-sys", "0.1"),
        ("torch-sys", "0.1"),
        ("ml-serving", {"package": "pytorch-runtime", "version": "0.1"}),
        ("scheduler", {"git": "https://example.invalid/vllm-adapter.git"}),
        ("runtime", {"git": "https://example.invalid/sglang-runtime.git"}),
        ("ml-kernel", {"version": "0.1", "features": ["cuda"]}),
        ("queue", {"version": "0.1", "features": ["hip-runtime"]}),
    ]

    for name, spec in allowed:
        violations = violations_for_dependency(name, spec)
        if violations:
            print(f"self-test rejected allowed dependency {name}: {violations}", file=sys.stderr)
            return 1

    for name, spec in blocked:
        if not violations_for_dependency(name, spec):
            print(f"self-test accepted forbidden dependency {name}", file=sys.stderr)
            return 1

    print("runtime dependency policy self-test ok")
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
