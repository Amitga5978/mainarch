# Security Policy

`mainarch` talks directly to AMD GPU kernel interfaces. Security reports should
be handled privately until a fix or mitigation is ready.

## Reporting

Use GitHub's private vulnerability reporting or security advisory flow for this
repository when it is available. If private reporting is unavailable, open a
minimal public issue asking for a private coordination channel and do not include
exploit details, crash dumps, tokens, machine identifiers, or private hardware
logs.

Good reports include:

- affected commit or release
- host kernel, GPU model, and container/runtime environment
- command line, environment variables, and required hardware state
- observed behavior and expected behavior
- whether `/dev/kfd`, `/dev/dri`, peer mappings, queues, or doorbells are
  involved
- minimal reproduction steps or a reduced proof of concept

## Scope

Security-sensitive areas include:

- raw `amdkfd` and `amdgpu` ioctl bindings
- GPU virtual memory allocation, mapping, and peer mapping
- AQL queue creation, packet construction, doorbell writes, and completion
  signals
- code-object loading and kernel descriptor interpretation
- checkpoint parsing, memory-mapped weight loading, and rank-shard conversion
- CLI paths that expose device state, benchmark output, or hardware diagnostics

Out of scope for private security handling:

- performance-only regressions without a safety or confidentiality impact
- reports that require already-compromised host root privileges with no new
  privilege boundary crossed
- unsupported forks or modified kernels that cannot be reproduced from the
  report

## Handling Expectations

- Do not publish exploit details until maintainers have had a reasonable chance
  to investigate and coordinate a fix.
- Keep reports narrowly scoped and include evidence that distinguishes a security
  issue from a correctness, performance, or documentation issue.
- Do not introduce ROCm, HIP, HSA runtime, RCCL, PyTorch, CUDA, or
  Python-serving runtime dependencies as a mitigation for production paths
  unless maintainers explicitly accept that architectural change.
- Treat logs from GPU hosts as potentially sensitive. Redact tokens, local paths,
  usernames, hardware serials, hostnames, and cluster identifiers.

## Supported Versions

The project is pre-1.0. Security fixes are targeted at the current default
branch unless a maintainer explicitly marks a release branch as supported.
