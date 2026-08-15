#!/usr/bin/env python3
"""Guard public model API helper booleans against consistency bypasses."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_API_SOURCE = Path("crates/mainarch-core/src/model_api.rs")
RELEASE_GUARD_IS_HELPERS = {
    "is_admitted",
    "is_accepted",
    "is_rejected",
    "is_static_handoff_ready",
}
RELEASE_GUARD_ASSERT_HELPERS = {
    "assert_admitted",
    "assert_accepted",
    "assert_rejected",
    "assert_static_handoff_ready",
}
RELEASE_GUARD_ASSERT_ONLY_HELPERS = {
    "assert_no_rejection",
    "assert_static_metadata_ready",
}

PUBLIC_GUARD_FUNCTION = re.compile(
    r"pub\s+fn\s+"
    r"(?P<name>(?:is|assert)_[a-z0-9_]+)"
    r"\s*\(\s*&self\s*\)\s*"
    r"(?:->\s*(?P<return>bool|Result\s*<\s*\(\s*\)\s*>))?"
    r"\s*\{",
    re.M,
)


@dataclass(frozen=True)
class BoundaryFunction:
    name: str
    return_type: str
    body: str
    line: int


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def find_matching_brace(text: str, opening: int) -> int:
    depth = 0
    index = opening
    in_line_comment = False
    in_block_comment = 0
    in_string = False
    in_char = False
    raw_hashes: int | None = None
    escaped = False

    while index < len(text):
        ch = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            if ch == "/" and nxt == "*":
                in_block_comment += 1
                index += 2
                continue
            if ch == "*" and nxt == "/":
                in_block_comment -= 1
                index += 2
                continue
            index += 1
            continue

        if in_string:
            if raw_hashes is not None:
                if ch == '"' and text.startswith("#" * raw_hashes, index + 1):
                    index += raw_hashes + 1
                    in_string = False
                    raw_hashes = None
                    continue
                index += 1
                continue
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            index += 1
            continue

        if in_char:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_char = False
            index += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            index += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = 1
            index += 2
            continue
        if ch == "r":
            raw = re.match(r'r(#+)"', text[index:])
            if raw:
                in_string = True
                raw_hashes = len(raw.group(1))
                index += raw.end()
                continue
            if nxt == '"':
                in_string = True
                raw_hashes = 0
                index += 2
                continue
        if ch == '"':
            in_string = True
            escaped = False
            index += 1
            continue
        if ch == "'":
            in_char = True
            escaped = False
            index += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1

    raise ValueError(f"unmatched function brace at byte {opening}")


def boundary_functions(text: str) -> list[BoundaryFunction]:
    functions: list[BoundaryFunction] = []
    for match in PUBLIC_GUARD_FUNCTION.finditer(text):
        name = match.group("name")
        if not (
            name.endswith("_boundary")
            or name in RELEASE_GUARD_IS_HELPERS
            or name in RELEASE_GUARD_ASSERT_HELPERS
            or name in RELEASE_GUARD_ASSERT_ONLY_HELPERS
        ):
            continue
        opening = text.find("{", match.start())
        closing = find_matching_brace(text, opening)
        functions.append(
            BoundaryFunction(
                name=name,
                return_type=(match.group("return") or ""),
                body=text[opening + 1 : closing],
                line=line_number(text, match.start()),
            )
        )
    return functions


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def issues_for_source(text: str) -> list[str]:
    functions = boundary_functions(text)
    by_name = {function.name: function for function in functions}
    issues: list[str] = []

    for function in functions:
        body = normalized(function.body)
        if function.name.startswith("is_"):
            expected_assert = "assert_" + function.name.removeprefix("is_")
            expected_call = f"self.{expected_assert}().is_ok()"
            if normalized(expected_call) not in body:
                issues.append(
                    f"line {function.line}: {function.name} must delegate to {expected_assert}().is_ok()"
                )
            if "_boundary_issues().is_empty()" in body:
                issues.append(
                    f"line {function.line}: {function.name} must not bypass its assertion helper"
                )
            if expected_assert not in by_name:
                issues.append(
                    f"line {function.line}: {function.name} is missing paired {expected_assert}"
                )
        elif function.name.startswith("assert_"):
            expected_is = "is_" + function.name.removeprefix("assert_")
            if "self.assert_consistent()" not in body:
                issues.append(
                    f"line {function.line}: {function.name} must compose self.assert_consistent()"
                )
            if "consistency:" not in function.body:
                issues.append(
                    f"line {function.line}: {function.name} must prefix consistency failures with 'consistency:'"
                )
            if function.name.endswith("_boundary") and "_boundary_issues()" not in body:
                issues.append(
                    f"line {function.line}: {function.name} must extend boundary-specific issues"
                )
            if (
                function.name not in RELEASE_GUARD_ASSERT_ONLY_HELPERS
                and expected_is not in by_name
            ):
                issues.append(
                    f"line {function.line}: {function.name} is missing paired {expected_is}"
                )

    return issues


def check(root: Path) -> int:
    source = root / MODEL_API_SOURCE
    if not source.exists():
        print(f"model API helper guard failed: missing {source}", file=sys.stderr)
        return 1

    text = source.read_text(encoding="utf-8")
    issues = issues_for_source(text)
    if issues:
        print("model API helper guard failed:", file=sys.stderr)
        for issue in issues:
            print(f"  - {MODEL_API_SOURCE}:{issue}", file=sys.stderr)
        return 1

    count = len(boundary_functions(text))
    print(f"model API helper guard ok: checked {count} public guard helpers")
    return 0


def self_test() -> int:
    accepted = """
impl Receipt {
    fn non_execution_boundary_issues(&self) -> Vec<String> { vec![] }
    pub fn is_non_executing_boundary(&self) -> bool {
        self.assert_non_executing_boundary().is_ok()
    }
    pub fn assert_non_executing_boundary(&self) -> Result<()> {
        let mut issues = Vec::new();
        if let Err(err) = self.assert_consistent() {
            issues.push(format!("consistency: {err}"));
        }
        issues.extend(self.non_execution_boundary_issues());
        if issues.is_empty() { Ok(()) } else { Err(anyhow!("blocked {issues:?}")) }
    }
}
"""
    if issues_for_source(accepted):
        print("self-test rejected accepted helper pair", file=sys.stderr)
        return 1

    accepted_release_guard = """
impl Receipt {
    pub fn is_static_handoff_ready(&self) -> bool {
        self.assert_static_handoff_ready().is_ok()
    }
    pub fn assert_static_handoff_ready(&self) -> Result<()> {
        if let Err(err) = self.assert_consistent() {
            return Err(anyhow!("not ready: consistency: {err}"));
        }
        Ok(())
    }
}
"""
    if issues_for_source(accepted_release_guard):
        print("self-test rejected accepted release guard pair", file=sys.stderr)
        return 1

    accepted_assert_only_guard = """
impl Receipt {
    pub fn assert_no_rejection(&self) -> Result<()> {
        if let Err(err) = self.assert_consistent() {
            return Err(anyhow!("has rejection issue(s): consistency: {err}"));
        }
        Ok(())
    }
}
"""
    if issues_for_source(accepted_assert_only_guard):
        print("self-test rejected accepted assert-only release guard", file=sys.stderr)
        return 1

    accepted_static_metadata_guard = """
impl Receipt {
    pub fn assert_static_metadata_ready(&self) -> Result<()> {
        if let Err(err) = self.assert_consistent() {
            return Err(anyhow!("not static metadata ready: consistency: {err}"));
        }
        Ok(())
    }
}
"""
    if issues_for_source(accepted_static_metadata_guard):
        print(
            "self-test rejected accepted static-metadata release guard",
            file=sys.stderr,
        )
        return 1

    missing_assert_only_prefix = accepted_assert_only_guard.replace(
        "consistency: {err}",
        "{err}",
    )
    missing_assert_only_prefix_issues = issues_for_source(missing_assert_only_prefix)
    if not any(
        "consistency failures" in issue for issue in missing_assert_only_prefix_issues
    ):
        print(
            "self-test accepted assert-only release guard without consistency prefix",
            file=sys.stderr,
        )
        return 1

    direct_boolean = accepted.replace(
        "self.assert_non_executing_boundary().is_ok()",
        "self.non_execution_boundary_issues().is_empty()",
    )
    direct_issues = issues_for_source(direct_boolean)
    if not any("must delegate" in issue for issue in direct_issues):
        print("self-test accepted direct boundary issue boolean", file=sys.stderr)
        return 1

    missing_consistency = accepted.replace(
        """if let Err(err) = self.assert_consistent() {
            issues.push(format!("consistency: {err}"));
        }
        """,
        "",
    )
    missing_consistency_issues = issues_for_source(missing_consistency)
    if not any("assert_consistent" in issue for issue in missing_consistency_issues):
        print("self-test accepted assertion without consistency guard", file=sys.stderr)
        return 1

    missing_prefix = accepted.replace('format!("consistency: {err}")', "err.to_string()")
    missing_prefix_issues = issues_for_source(missing_prefix)
    if not any("consistency failures" in issue for issue in missing_prefix_issues):
        print("self-test accepted assertion without consistency prefix", file=sys.stderr)
        return 1

    print("model API helper guard self-test ok")
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
