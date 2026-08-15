# Baseline: upstream rccl-tests (reference only)

`mainarch-collectives` mirrors the **rccl-tests methodology**, the same size
sweep, correctness check, and algbw/busbw convention, so the numbers are
directly comparable. This directory holds reference runs captured from the
*existing* ROCm stack to compare against.

Building upstream rccl-tests requires RCCL + HIP, which the mainarch
devcontainer deliberately does **not** ship (not depending on ROCm is the whole
point). Build it in a separate ROCm container and keep the two stacks apart.

## Building an rccl-tests image

```bash
cat > Dockerfile.rccl-tests <<'DOCKER'
FROM rocm/dev-ubuntu-24.04:latest
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential rccl rccl-dev && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ROCm/rccl-tests /opt/rccl-tests \
 && make -C /opt/rccl-tests MPI=0
ENV PATH=/opt/rccl-tests/build:$PATH
ENTRYPOINT ["all_reduce_perf"]
DOCKER

docker build -t local/rccl-tests -f Dockerfile.rccl-tests .
```

Package names and the ROCm base tag move between releases; adjust to whatever
your host's ROCm version provides.

## Capturing a paired comparison

```bash
MAINARCH_RCCL_IMAGE=local/rccl-tests bench/compare-allreduce.sh
```

This runs both stacks over the same 8-GPU ladder and writes
`baseline/allreduce-comparison.txt`.

**Pick the RCCL configuration honestly.** `bench/compare-allreduce.sh` defaults
to `RCCL_ARGS="-t 8 -g 1"`, meaning 8 threads with one GPU each, which is
RCCL's *fastest*
layout and therefore the right comparator. `-g 8` puts one host thread in charge
of all 8 GPUs, which serializes RCCL's launch path and flatters mainarch. If you
quote a `-g 8` number, say so.

The checked-in `allreduce-comparison.txt` was captured with `-g 8` and carries
that caveat in its header; treat it as the single-threaded reference point, not
the headline.

## Direct sweep

```bash
mainarch rccl-test all-reduce --backend gpu --ranks 8    # mainarch, on hardware
./build/all_reduce_perf -b 8 -e 128M -f 2 -t 8 -g 1      # RCCL, its best config
```
