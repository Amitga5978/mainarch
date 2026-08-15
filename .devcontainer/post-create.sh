#!/usr/bin/env bash
# Runs once when the devcontainer is created. Nothing here is required for the
# build; it just tells you which lane you are in.
set -u

echo
rustc --version
just --version 2>/dev/null || true
echo

if [ -e /dev/kfd ] && [ -d /sys/class/kfd/kfd/topology/nodes ]; then
  echo "AMD kernel driver visible: /dev/kfd is present."
  ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null | head -4
  echo "GPU lane available. 'just demo' will run the live KFD/AQL path."
else
  echo "No /dev/kfd in this container: running the CPU-only lane."
  echo "Everything in 'just tour' still works; the live GPU proofs are skipped."
  echo "On an AMD GPU host, reopen using the '.devcontainer/gpu' configuration."
fi

echo
echo "mainarch dev harness ready (no ROCm anywhere in this image)."
echo "Next:  just demo"
echo
