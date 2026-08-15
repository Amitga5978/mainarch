#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/kfd_xgmi_preflight.sh [options]

Validate the kernel's raw AMDKFD sysfs view of a single-node XGMI hive without
ROCm user-space tools. This is a preflight for mainarch raw KFD/XGMI collectives
and RCCL baseline runs.

Options:
  --expect-gpus N        Expected physical GPU count. Default: 8
  --topology-dir DIR     KFD topology directory.
                         Default: /sys/class/kfd/kfd/topology/nodes
  -h, --help             Show this help.

Checks:
  - GPU nodes have gfx_target_version > 0.
  - Physical GPUs are deduplicated by domain:location_id.
  - All GPU nodes share one non-zero hive_id.
  - Every GPU node has num_sdma_xgmi_engines > 0.
  - Every GPU node has expect-gpus - 1 XGMI peer links with type 11.
USAGE
}

expect_gpus=8
topology_dir="/sys/class/kfd/kfd/topology/nodes"

while (($#)); do
  case "$1" in
    --expect-gpus)
      expect_gpus="$2"
      shift 2
      ;;
    --topology-dir)
      topology_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$expect_gpus" =~ ^[0-9]+$ ]] || ((expect_gpus < 1)); then
  echo "invalid --expect-gpus: $expect_gpus" >&2
  exit 2
fi

if [[ ! -d "$topology_dir" ]]; then
  echo "FAIL: KFD topology directory not found: $topology_dir" >&2
  exit 1
fi

prop() {
  local file="$1"
  local key="$2"
  awk -v key="$key" '$1 == key { print $2; found=1; exit } END { exit(found ? 0 : 1) }' "$file" 2>/dev/null || true
}

declare -a gpu_nodes=()
declare -A node_name=()
declare -A node_gpu_id=()
declare -A node_gfx=()
declare -A node_hive=()
declare -A node_domain=()
declare -A node_location=()
declare -A node_render=()
declare -A node_sdma_xgmi=()
declare -A physical_seen=()
declare -A hive_seen=()
declare -A gpu_node_seen=()

mapfile -t node_dirs < <(find "$topology_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -n)

for node in "${node_dirs[@]}"; do
  props="$topology_dir/$node/properties"
  [[ -r "$props" ]] || continue
  gfx="$(prop "$props" gfx_target_version)"
  [[ -n "$gfx" && "$gfx" != "0" ]] || continue

  gpu_nodes+=("$node")
  gpu_node_seen["$node"]=1
  node_name["$node"]="$(cat "$topology_dir/$node/name" 2>/dev/null || true)"
  node_gpu_id["$node"]="$(cat "$topology_dir/$node/gpu_id" 2>/dev/null || true)"
  node_gfx["$node"]="$gfx"
  node_hive["$node"]="$(prop "$props" hive_id)"
  node_domain["$node"]="$(prop "$props" domain)"
  node_location["$node"]="$(prop "$props" location_id)"
  node_render["$node"]="$(prop "$props" drm_render_minor)"
  node_sdma_xgmi["$node"]="$(prop "$props" num_sdma_xgmi_engines)"

  physical_seen["${node_domain[$node]}:${node_location[$node]}"]=1
  hive_seen["${node_hive[$node]}"]=1
done

declare -A node_xgmi_peer_links=()
declare -A node_xgmi_total_links=()
total_xgmi_peer_links=0
total_xgmi_links=0

for node in "${gpu_nodes[@]}"; do
  peer_count=0
  type11_count=0
  for link_props in "$topology_dir/$node"/io_links/*/properties; do
    [[ -r "$link_props" ]] || continue
    link_type="$(prop "$link_props" type)"
    [[ "$link_type" == "11" ]] || continue
    type11_count=$((type11_count + 1))
    node_to="$(prop "$link_props" node_to)"
    if [[ -n "${gpu_node_seen[$node_to]+x}" ]]; then
      peer_count=$((peer_count + 1))
    fi
  done
  node_xgmi_total_links["$node"]="$type11_count"
  node_xgmi_peer_links["$node"]="$peer_count"
  total_xgmi_links=$((total_xgmi_links + type11_count))
  total_xgmi_peer_links=$((total_xgmi_peer_links + peer_count))
done

physical_count="${#physical_seen[@]}"
hive_count="${#hive_seen[@]}"
expected_peer_links_per_gpu=$((expect_gpus - 1))
expected_total_peer_links=$((expect_gpus * expected_peer_links_per_gpu))
rc=0

echo "mainarch KFD XGMI topology preflight"
echo "  topology:       $topology_dir"
echo "  expected GPUs:  $expect_gpus"
echo "  gpu nodes:      ${#gpu_nodes[@]}"
echo "  physical GPUs:  $physical_count"
echo "  hive count:     $hive_count"
echo "  XGMI peer links: $total_xgmi_peer_links/$expected_total_peer_links"
echo
printf '%-5s %-8s %-8s %-8s %-12s %-7s %-9s %-12s %-9s\n' \
  node gfx gpu_id domain location hive render sdma_xgmi xgmi_peer
for node in "${gpu_nodes[@]}"; do
  printf '%-5s %-8s %-8s %-8s %-12s %-7s %-9s %-12s %-9s\n' \
    "$node" \
    "${node_gfx[$node]:-}" \
    "${node_gpu_id[$node]:-}" \
    "${node_domain[$node]:-}" \
    "${node_location[$node]:-}" \
    "${node_hive[$node]:-}" \
    "${node_render[$node]:-}" \
    "${node_sdma_xgmi[$node]:-}" \
    "${node_xgmi_peer_links[$node]:-0}"
done
echo

if ((physical_count != expect_gpus)); then
  echo "FAIL: expected $expect_gpus physical GPUs, found $physical_count" >&2
  rc=1
else
  echo "PASS: physical GPU count matches"
fi

if ((hive_count != 1)); then
  echo "FAIL: expected one XGMI hive, found $hive_count" >&2
  rc=1
else
  only_hive="${!hive_seen[*]}"
  if [[ "$only_hive" == "0" || -z "$only_hive" ]]; then
    echo "FAIL: GPU hive_id is zero or missing" >&2
    rc=1
  else
    echo "PASS: all GPU nodes share non-zero hive_id $only_hive"
  fi
fi

for node in "${gpu_nodes[@]}"; do
  sdma="${node_sdma_xgmi[$node]:-0}"
  if [[ ! "$sdma" =~ ^[0-9]+$ ]] || ((sdma == 0)); then
    echo "FAIL: node $node has no SDMA XGMI engines" >&2
    rc=1
  fi
  peer_links="${node_xgmi_peer_links[$node]:-0}"
  if ((peer_links != expected_peer_links_per_gpu)); then
    echo "FAIL: node $node expected $expected_peer_links_per_gpu XGMI peer links, found $peer_links" >&2
    rc=1
  fi
done

if ((rc == 0)); then
  echo "PASS: all GPU nodes have SDMA XGMI engines"
  echo "PASS: every GPU node has $expected_peer_links_per_gpu XGMI peer links"
  echo "PASS: KFD reports a full $expect_gpus-GPU XGMI hive"
fi

exit "$rc"
