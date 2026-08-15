# mainarch dev image. A clean Rust toolchain that talks to AMD silicon at the
# KERNEL ABI. Deliberately NO ROCm / HIP / HSA runtime: this project replaces
# that stack rather than building on it. The only AMD-specific things needed are
#   * the kernel UAPI headers (linux/kfd_ioctl.h) shipped by linux-libc-dev
#   * libdrm's amdgpu_drm.h for the DRM render-node ioctls
#   * LLVM/clang, whose upstream amdgpu backend can emit gfx950 ISA on its own
# The GPU, when present, is reached purely via /dev/kfd + /dev/dri passthrough
# (see .devcontainer/gpu/devcontainer.json). The image itself builds and runs
# fine on a machine with no AMD GPU at all, and every CPU-only path still works.
ARG BASE=docker.io/library/ubuntu:24.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# --- system: build toolchain, kernel/driver ABI headers, debug + dev ergonomics
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang lld llvm libclang-dev \
      cmake ninja-build pkg-config \
      linux-libc-dev libdrm-dev libelf-dev libnuma-dev zlib1g-dev \
      mold \
      git curl ca-certificates gnupg openssh-client \
      gdb strace ltrace \
      vim tmux htop jq ripgrep bsdmainutils file less \
      python3 \
 && rm -rf /var/lib/apt/lists/*

# --- Rust toolchain (system-wide so root and any user share it)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# NB: rustup-init takes one component per -c flag. clippy + rustfmt already come
# from `--profile default`; only rust-analyzer + rust-src need adding.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile default \
        -c rust-analyzer -c rust-src \
 && rustc --version && cargo --version

# --- iteration harness: `just` task runner, nextest, watch mode
# Installed from crates.io rather than a prebuilt release, so this works on any
# builder without needing BuildKit cache mounts or a static-musl downloader.
RUN cargo install --locked just cargo-nextest cargo-watch \
 && just --version && cargo-nextest --version && cargo-watch --version

# mold keeps incremental links snappy on a workspace this size
ENV RUSTFLAGS="-C link-arg=-fuse-ld=mold"

WORKDIR /workspaces
LABEL org.opencontainers.image.title="mainarch-dev" \
      org.opencontainers.image.description="Rust + raw AMD kernel ABI dev harness (no ROCm)"
