//! Safe device + topology layer.
//!
//! Enumerates AMD GPUs straight from the `amdkfd` sysfs topology and opens the
//! kernel device — no ROCm. This is the seam where the rest of `mainarch` stops
//! caring about ioctl encodings and starts working with [`GpuNode`]s and an
//! open [`Kfd`] handle.

use anyhow::{anyhow, Context, Result};
use std::cmp::min;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::fs;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

pub use mainarch_sys::gfx_name;

pub mod attn;
pub mod codeobject;
pub mod gemm;
pub mod gpu;
pub mod layer;
pub mod model;
pub mod model_api;
pub mod multigpu;
pub mod olmo2;
pub mod weights;
pub use codeobject::{
    CodeObject, CodeObjectInfo, CodeObjectKernelSetValidation, Kernel, KernelInfo,
    MAINARCH_KERNELS_GFX950,
};
pub use gpu::{
    aql_kernarg_stress_selftest as gpu_aql_kernarg_stress_selftest,
    check_allreduce as gpu_check_allreduce,
    paged_kv_attention_selftest as gpu_paged_kv_attention_selftest,
    paged_kv_gather_selftest as gpu_paged_kv_gather_selftest,
    paged_kv_probe_selftest as gpu_paged_kv_probe_selftest,
    paged_kv_qk_selftest as gpu_paged_kv_qk_selftest,
    paged_kv_softmax_selftest as gpu_paged_kv_softmax_selftest,
    paged_mla_dot_selftest as gpu_paged_mla_dot_selftest,
    paged_mla_fp8_gqa_output_tile_selftest as gpu_paged_mla_fp8_gqa_output_tile_selftest,
    paged_mla_fp8_kimi_hot_decode_gate as gpu_paged_mla_fp8_kimi_hot_decode_gate,
    paged_mla_fp8_kimi_long_persistent_gate as gpu_paged_mla_fp8_kimi_long_persistent_gate,
    paged_mla_fp8_splitk_e2e_selftest as gpu_paged_mla_fp8_splitk_e2e_selftest,
    paged_mla_fp8_splitk_full_latent_sweep_gate as gpu_paged_mla_fp8_splitk_full_latent_sweep_gate,
    paged_mla_fp8_splitk_ragged_full_latent_gate as gpu_paged_mla_fp8_splitk_ragged_full_latent_gate,
    paged_mla_fp8_splitk_stage1_selftest as gpu_paged_mla_fp8_splitk_stage1_selftest,
    paged_mla_fp8_splitk_stage2_selftest as gpu_paged_mla_fp8_splitk_stage2_selftest,
    paged_mla_gqa_output_tile_selftest as gpu_paged_mla_gqa_output_tile_selftest,
    paged_mla_latent_selftest as gpu_paged_mla_latent_selftest,
    paged_mla_output_tile_selftest as gpu_paged_mla_output_tile_selftest,
    paged_mla_softmax_selftest as gpu_paged_mla_softmax_selftest,
    paged_split_kv_combine_n_selftest as gpu_paged_split_kv_combine_n_selftest,
    paged_split_kv_combine_selftest as gpu_paged_split_kv_combine_selftest,
    qwen_candidate_replay_chain_selftest as gpu_qwen_candidate_replay_chain_selftest,
    qwen_candidate_resident_replay_selftest as gpu_qwen_candidate_resident_replay_selftest,
    replay_chain_selftest as gpu_replay_chain_selftest,
    replay_dense_residual_chain_selftest as gpu_dense_replay_chain_selftest,
    replay_model_chain_selftest as gpu_model_replay_chain_selftest,
    replay_selftest as gpu_replay_selftest, selftest as gpu_selftest,
    split_kv_combine_n_heads128_coop_selftest as gpu_split_kv_combine_n_heads128_coop_selftest,
    split_kv_combine_n_heads128_lanes_selftest as gpu_split_kv_combine_n_heads128_lanes_selftest,
    split_kv_combine_n_heads128_selftest as gpu_split_kv_combine_n_heads128_selftest,
    split_kv_combine_n_heads_selftest as gpu_split_kv_combine_n_heads_selftest,
    split_kv_combine_n_selftest as gpu_split_kv_combine_n_selftest,
    split_kv_combine_selftest as gpu_split_kv_combine_selftest, AqlLastDispatchCompletion,
    GpuDevice, KernargBuilder,
};
pub use multigpu::AllReduce;

const KFD_TOPOLOGY: &str = "/sys/class/kfd/kfd/topology/nodes";

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TopologyLinkType {
    Undefined,
    HyperTransport,
    PciExpress,
    Amba,
    Mipi,
    Xgmi,
    RapidIo,
    Infiniband,
    RdmaOther,
    EthernetRdma,
    Other(u32),
}

impl TopologyLinkType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Undefined,
            1 => Self::HyperTransport,
            2 => Self::PciExpress,
            3 => Self::Amba,
            4 => Self::Mipi,
            11 => Self::Xgmi,
            8 => Self::RapidIo,
            9 => Self::Infiniband,
            14 => Self::EthernetRdma,
            15 => Self::RdmaOther,
            _ => Self::Other(raw),
        }
    }

    pub fn is_xgmi_like(self) -> bool {
        matches!(self, Self::Xgmi)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Undefined => "unknown",
            Self::HyperTransport => "ht",
            Self::PciExpress => "pcie",
            Self::Amba => "amba",
            Self::Mipi => "mipi",
            Self::Xgmi => "xgmi",
            Self::RapidIo => "rapidio",
            Self::Infiniband => "infiniband",
            Self::RdmaOther => "rdma",
            Self::EthernetRdma => "eth-rdma",
            Self::Other(_) => "other",
        }
    }
}

impl fmt::Display for TopologyLinkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone)]
pub struct TopologyLink {
    pub from_node: u32,
    pub to_node: u32,
    pub raw_type: u32,
    pub kind: TopologyLinkType,
    pub weight: u32,
    pub min_latency: u32,
    pub max_latency: u32,
    pub min_bandwidth: u32,
    pub max_bandwidth: u32,
    pub recommended_transfer_size: u32,
    pub flags: u32,
}

impl TopologyLink {
    pub fn effective_bandwidth(&self) -> u32 {
        self.max_bandwidth.max(self.min_bandwidth)
    }
}

#[derive(Debug, Clone)]
pub struct TopologyGraph {
    pub nodes: Vec<GpuNode>,
    pub links: Vec<TopologyLink>,
}

impl TopologyGraph {
    pub fn node_ids(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.node_id).collect()
    }

    pub fn links_for_node(&self, node_id: u32) -> Vec<&TopologyLink> {
        self.links
            .iter()
            .filter(|link| link.from_node == node_id || link.to_node == node_id)
            .collect()
    }

    pub fn peer_nodes(&self, node_id: u32) -> Vec<u32> {
        let mut peers = BTreeSet::new();
        for link in self.links_for_node(node_id) {
            if link.from_node == node_id {
                peers.insert(link.to_node);
            } else {
                peers.insert(link.from_node);
            }
        }
        peers.into_iter().collect()
    }

    pub fn link_between(&self, from: u32, to: u32) -> Option<&TopologyLink> {
        self.links.iter().find(|link| {
            (link.from_node == from && link.to_node == to)
                || (link.from_node == to && link.to_node == from)
        })
    }

    /// A stable, topology-aware peer order. First prefer XGMI links, then by
    /// effective bandwidth, then by node id.
    pub fn xgmi_first_order(&self, start_node: u32, target_nodes: usize) -> Vec<u32> {
        let mut peers = self.peer_nodes(start_node);
        peers.sort_by_cached_key(|n| {
            let link = self.link_between(start_node, *n);
            let is_xgmi = link.is_some_and(TopologyLink::is_xgmi_like);
            let bw = link.map_or(0, TopologyLink::effective_bandwidth);
            // keep deterministic descending sort by using tuple trick
            (if is_xgmi { 0_u8 } else { 1_u8 }, u32::MAX - bw, *n)
        });

        peers.truncate(target_nodes.min(peers.len()));
        peers
    }

    pub fn count_gpu_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl TopologyLink {
    fn is_xgmi_like(link: &TopologyLink) -> bool {
        link.kind.is_xgmi_like()
    }
}

#[derive(Debug, Clone)]
pub struct GpuNode {
    pub node_id: u32,
    pub name: String,
    pub gfx_target_version: u64,
    pub simd_count: u32,
    pub vram_bytes: u128,
}

impl GpuNode {
    /// Architecture string, e.g. `"gfx950"`.
    pub fn gfx(&self) -> String {
        gfx_name(self.gfx_target_version)
    }
    pub fn vram_gib(&self) -> f64 {
        self.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// An open handle to `/dev/kfd`, the kernel compute device.
pub struct Kfd {
    fd: OwnedFd,
    /// Per-device acquired VMs: `gpu_id -> render node fd`. The fd must stay
    /// open for the life of the process VM, so we own it here.
    vms: Mutex<HashMap<u32, OwnedFd>>,
}

impl Kfd {
    /// Open the kernel compute device.
    pub fn open() -> Result<Self> {
        let f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(mainarch_sys::KFD_DEVICE)
            .with_context(|| {
                format!(
                    "opening {}: is the device passed into the container and are you in the 'render' group?",
                    mainarch_sys::KFD_DEVICE
                )
            })?;
        Ok(Self {
            fd: OwnedFd::from(f),
            vms: Mutex::new(HashMap::new()),
        })
    }

    /// Bind this process to a device's GPU VM (`AMDKFD_IOC_ACQUIRE_VM`).
    ///
    /// Mandatory before any memory or queue ioctl on the device. Idempotent
    /// per device within this handle; returns the kernel `gpu_id` for the
    /// topology node (the hashed id every KFD ioctl wants — *not* the node
    /// index).
    pub fn ensure_vm(&self, node_id: u32) -> Result<u32> {
        let gpu_id = gpu_id_for_node(node_id)?;
        let mut vms = self.vms.lock().expect("kfd vm lock poisoned");
        if let std::collections::hash_map::Entry::Vacant(e) = vms.entry(gpu_id) {
            let render = render_fd_for_node(node_id)?;
            let mut args = mainarch_sys::AcquireVmArgs {
                drm_fd: render.as_raw_fd() as u32,
                gpu_id,
            };
            unsafe {
                mainarch_sys::ioctl_acquire_vm(self.fd.as_raw_fd(), &mut args)
                    .with_context(|| format!("AMDKFD_IOC_ACQUIRE_VM gpu_id={gpu_id}"))?;
            }
            e.insert(render);
        }
        Ok(gpu_id)
    }

    /// Raw fd of the render node whose VM was acquired for `gpu_id`. Valid for
    /// the lifetime of this `Kfd`.
    fn render_raw_fd(&self, gpu_id: u32) -> Result<i32> {
        let vms = self.vms.lock().expect("kfd vm lock poisoned");
        vms.get(&gpu_id)
            .map(|fd| fd.as_raw_fd())
            .ok_or_else(|| anyhow!("no acquired VM for gpu_id {gpu_id}: call ensure_vm first"))
    }

    /// Allocate a host-visible, GPU-mapped buffer on `node_id`.
    ///
    /// CPU and GPU share the VA. Set `executable` for buffers that hold GPU
    /// code (the loaded code object). Memory is GTT (host RAM mapped into the
    /// GPU VM), which is coherent enough for the host to observe kernel results
    /// after a system-scope release fence.
    pub fn alloc_host_visible(
        &self,
        node_id: u32,
        bytes: usize,
        executable: bool,
    ) -> Result<DeviceBuffer> {
        let gpu_id = self.ensure_vm(node_id)?;
        let mut flags = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_GTT
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        if executable {
            flags |= mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
        }
        let inner = KfdAllocatedBuffer::new(self, gpu_id, bytes, flags)?;
        Ok(DeviceBuffer { inner })
    }

    /// Allocate device-local VRAM (HBM) on `node_id`. On large-BAR parts the
    /// allocation is also CPU-mappable (PUBLIC), so the same VA serves host
    /// fills/reads and GPU compute while data physically lives in HBM — the key
    /// to interconnect-bound bandwidth. Falls back to a GPU-only VRAM buffer if
    /// the public mapping is unavailable.
    pub fn alloc_vram(&self, node_id: u32, bytes: usize) -> Result<DeviceBuffer> {
        let gpu_id = self.ensure_vm(node_id)?;
        let gpu_only = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        match self.alloc_public_coherent_vram(node_id, bytes) {
            Ok(buffer) => Ok(buffer),
            Err(_) => {
                let inner = KfdAllocatedBuffer::new(self, gpu_id, bytes, gpu_only)?;
                Ok(DeviceBuffer { inner })
            }
        }
    }

    /// Allocate public coherent VRAM (HBM) on `node_id` without falling back to
    /// GPU-only VRAM. This is required for small cross-GPU synchronization
    /// words such as direct ready flags: if the public/coherent allocation is
    /// unavailable, using a silent fallback would invalidate the acquire/release
    /// proof while still letting the program continue.
    pub fn alloc_public_coherent_vram(&self, node_id: u32, bytes: usize) -> Result<DeviceBuffer> {
        let gpu_id = self.ensure_vm(node_id)?;
        let flags = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        let inner = KfdAllocatedBuffer::new(self, gpu_id, bytes, flags)?;
        Ok(DeviceBuffer { inner })
    }

    /// Map an existing allocation (by `DeviceBuffer`) into a peer device's VM
    /// for XGMI access. `node_id` is the peer topology node.
    pub fn map_buffer_to_peer(&self, buffer: &DeviceBuffer, node_id: u32) -> Result<u32> {
        let gpu_id = self.ensure_vm(node_id)?;
        self.map_memory_to_gpus(buffer.inner.handle, std::slice::from_ref(&gpu_id))?;
        register_gpu_allocation_mapping(buffer.va(), gpu_id)?;
        Ok(gpu_id)
    }

    /// `AMDKFD_IOC_GET_VERSION` — the simplest proof we are talking to the
    /// driver.
    pub fn driver_version(&self) -> Result<(u32, u32)> {
        let v = unsafe { mainarch_sys::ioctl_get_version(self.fd.as_raw_fd()) }
            .context("AMDKFD_IOC_GET_VERSION ioctl")?;
        Ok((v.major_version, v.minor_version))
    }

    pub fn create_aql_queue(&self, node_id: u32) -> Result<KfdQueue> {
        KfdQueue::new(self, node_id)
    }

    /// Allocate GPU-visible memory on a device. `gpu_id` is the kernel id
    /// returned by [`Kfd::ensure_vm`].
    pub fn allocate_memory_with_offset(
        &self,
        gpu_id: u32,
        bytes: usize,
        flags: u32,
        va_addr: u64,
        mmap_offset: u64,
    ) -> Result<(u64, u64)> {
        let mut args = mainarch_sys::AllocMemoryOfGpuArgs {
            va_addr,
            size: bytes as u64,
            handle: 0,
            mmap_offset,
            gpu_id,
            flags,
        };
        unsafe {
            mainarch_sys::ioctl_alloc_memory_of_gpu(self.fd.as_raw_fd(), &mut args)
                .context("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU")?;
        }

        if args.handle == 0 {
            return Err(anyhow!("alloc memory ioctl returned empty handle"));
        }

        Ok((args.handle, args.mmap_offset))
    }

    pub fn free_memory(&self, handle: u64) -> Result<()> {
        let mut args = mainarch_sys::FreeMemoryOfGpuArgs { handle };
        unsafe {
            mainarch_sys::ioctl_free_memory_of_gpu(self.fd.as_raw_fd(), &mut args)
                .context("AMDKFD_IOC_FREE_MEMORY_OF_GPU")?;
        }
        Ok(())
    }

    /// Map an allocation into one or more device VMs (kernel `gpu_id`s).
    pub fn map_memory_to_gpus(&self, handle: u64, gpu_ids: &[u32]) -> Result<()> {
        if gpu_ids.is_empty() {
            return Ok(());
        }
        let mut ids = gpu_ids.to_vec();
        let mut args = mainarch_sys::MapMemoryToGpuArgs {
            handle,
            device_ids_array_ptr: ids.as_mut_ptr() as usize as u64,
            n_devices: ids.len() as u32,
            n_success: 0,
        };

        unsafe {
            mainarch_sys::ioctl_map_memory_to_gpu(self.fd.as_raw_fd(), &mut args)
                .context("AMDKFD_IOC_MAP_MEMORY_TO_GPU")?;
        }

        if args.n_success as usize != gpu_ids.len() {
            return Err(anyhow!(
                "partial GPU memory mapping: mapped {} of {} devices",
                args.n_success,
                gpu_ids.len(),
            ));
        }
        Ok(())
    }

    pub fn unmap_memory_from_gpus(&self, handle: u64, gpu_ids: &[u32]) -> Result<()> {
        if gpu_ids.is_empty() {
            return Ok(());
        }
        let mut ids = gpu_ids.to_vec();
        let mut args = mainarch_sys::UnmapMemoryFromGpuArgs {
            handle,
            device_ids_array_ptr: ids.as_mut_ptr() as usize as u64,
            n_devices: ids.len() as u32,
            n_success: 0,
        };

        unsafe {
            mainarch_sys::ioctl_unmap_memory_from_gpu(self.fd.as_raw_fd(), &mut args)
                .context("AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU")?;
        }
        Ok(())
    }
}

pub struct KfdQueue {
    kfd_fd: i32,
    queue_id: u32,
    gpu_id: u32,
    doorbell_offset: u64,
    /// CPU/GPU VA of the AQL ring (host-visible).
    ring_va: u64,
    /// Number of 64-byte AQL packet slots in the ring.
    ring_slots: u64,
    /// Mailbox the CP reads for the producer write index.
    write_ptr_va: u64,
    /// Mailbox the CP writes with its consumed read index.
    read_ptr_va: u64,
    /// mmap'd doorbell for this queue (a u64 within the doorbell aperture).
    doorbell: *mut u64,
    /// Base + length of the doorbell aperture mapping, for munmap on drop.
    doorbell_map_base: *mut c_void,
    doorbell_map_len: usize,
    /// Monotonic count of packets ever enqueued.
    write_index: u64,
    /// Cached once at queue creation to avoid environment lookups in the hot path.
    packet_guard: bool,
    _ring: KfdAllocatedBuffer,
    _write_ptr: KfdAllocatedBuffer,
    _read_ptr: KfdAllocatedBuffer,
    _eop: Option<KfdAllocatedBuffer>,
    _ctx_save_restore: Option<KfdAllocatedBuffer>,
}

/// Host snapshot of the live KFD AQL queue geometry.
#[derive(Debug, Clone, Copy)]
pub struct KfdQueueSnapshot {
    pub queue_id: u32,
    pub gpu_id: u32,
    pub ring_va: u64,
    pub ring_slots: u64,
    pub packet_bytes: u64,
    pub write_ptr_va: u64,
    pub read_ptr_va: u64,
    pub doorbell_offset: u64,
    pub host_write_index: u64,
    pub producer_index: u64,
    pub consumer_index: u64,
}

/// Host-side AQL reservation-token plan for a contiguous packet batch.
///
/// This does not mutate the queue. It is the arithmetic contract that a real
/// producer reservation must satisfy before packet payload writes, release
/// header stores, and a final doorbell with `doorbell_packet_id`.
#[derive(Debug, Clone, Copy)]
pub struct KfdQueueBatchReservationPlan {
    pub base_packet_id: u64,
    pub packet_count: u64,
    pub last_packet_id: u64,
    pub desired_write_index: u64,
    pub read_index: u64,
    pub inflight_packets: u64,
    pub capacity_ok: bool,
    pub first_slot_index: u64,
    pub first_slot_offset: u64,
    pub first_slot_va: u64,
    pub last_slot_index: u64,
    pub last_slot_offset: u64,
    pub last_slot_va: u64,
    pub slots_distinct: bool,
    pub slots_aligned64: bool,
    pub first_slot_formula_ok: bool,
    pub last_slot_formula_ok: bool,
    pub doorbell_packet_id: u64,
    pub doorbell_matches_last_packet: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReservationInput {
    pub operands_probe_version: u64,
    pub packet_id: u64,
    pub read_index: u64,
    pub packet_id_matches_host_snapshot: u64,
    pub read_index_matches_host_snapshot: u64,
    pub inflight_packets: u64,
    pub capacity_ok: u64,
    pub slot_index: u64,
    pub slot_offset: u64,
    pub slot_va: u64,
    pub slot_va_aligned64: u64,
    pub desired_write_index: u64,
    pub packet_count: u64,
    pub doorbell_packet_id: u64,
    pub doorbell_matches_last_packet: u64,
    pub publish_low32: u64,
    pub header_release_width_bits: u64,
    pub live_low32: u64,
    pub valid_header_not_stored: u64,
    pub fetch_add_not_performed: u64,
    pub doorbell_not_written: u64,
    pub capacity_formula_ok: u64,
    pub slot_formula_ok: u64,
    pub metadata_ready_dependency: u64,
    pub non_consuming_contract: u64,
    pub observed_ready: u64,
    pub expected_capacity_ok: bool,
    pub expected_slot_aligned64: bool,
    pub expected_doorbell_matches_last_packet: bool,
    pub expected_valid_header_not_stored: bool,
    pub expected_metadata_ready_dependency: bool,
    pub expected_slot_formula_ok: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReservationProof {
    pub operands_probe_version: u64,
    pub packet_id: u64,
    pub read_index: u64,
    pub packet_id_matches_host_snapshot: u64,
    pub read_index_matches_host_snapshot: u64,
    pub inflight_packets: u64,
    pub capacity_ok: u64,
    pub slot_index: u64,
    pub slot_offset: u64,
    pub slot_va: u64,
    pub slot_va_aligned64: u64,
    pub desired_write_index: u64,
    pub packet_count: u64,
    pub doorbell_packet_id: u64,
    pub doorbell_matches_last_packet: u64,
    pub publish_low32: u64,
    pub header_release_width_bits: u64,
    pub live_low32: u64,
    pub valid_header_not_stored: u64,
    pub fetch_add_not_performed: u64,
    pub doorbell_not_written: u64,
    pub capacity_formula_ok: u64,
    pub slot_formula_ok: u64,
    pub metadata_ready_dependency: u64,
    pub non_consuming_contract: u64,
    pub observed_ready: u64,
    pub expected_capacity_ok: u64,
    pub expected_slot_aligned64: u64,
    pub expected_doorbell_matches_last_packet: u64,
    pub expected_valid_header_not_stored: u64,
    pub expected_metadata_ready_dependency: u64,
    pub expected_slot_formula_ok: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReservationValidation {
    pub printed_ready: bool,
    pub expected_capacity_ok: bool,
    pub expected_slot_aligned64: bool,
    pub expected_doorbell_matches_last_packet: bool,
    pub expected_valid_header_not_stored: bool,
    pub expected_metadata_ready_dependency: bool,
    pub expected_slot_formula_ok: bool,
    pub no_fetch_add: bool,
    pub no_doorbell: bool,
    pub non_consuming_contract: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlReservationInput {
    pub fn proof(self) -> KfdQueueLiveAqlReservationProof {
        let expected_capacity_ok = self.expected_capacity_ok as u64;
        let expected_slot_aligned64 = self.expected_slot_aligned64 as u64;
        let expected_doorbell_matches_last_packet =
            self.expected_doorbell_matches_last_packet as u64;
        let expected_valid_header_not_stored = self.expected_valid_header_not_stored as u64;
        let expected_metadata_ready_dependency = self.expected_metadata_ready_dependency as u64;
        let expected_slot_formula_ok = self.expected_slot_formula_ok as u64;
        let ready = (self.expected_capacity_ok
            && self.expected_slot_aligned64
            && self.expected_doorbell_matches_last_packet
            && self.expected_valid_header_not_stored
            && self.expected_metadata_ready_dependency
            && self.expected_slot_formula_ok) as u64;

        KfdQueueLiveAqlReservationProof {
            operands_probe_version: self.operands_probe_version,
            packet_id: self.packet_id,
            read_index: self.read_index,
            packet_id_matches_host_snapshot: self.packet_id_matches_host_snapshot,
            read_index_matches_host_snapshot: self.read_index_matches_host_snapshot,
            inflight_packets: self.inflight_packets,
            capacity_ok: self.capacity_ok,
            slot_index: self.slot_index,
            slot_offset: self.slot_offset,
            slot_va: self.slot_va,
            slot_va_aligned64: self.slot_va_aligned64,
            desired_write_index: self.desired_write_index,
            packet_count: self.packet_count,
            doorbell_packet_id: self.doorbell_packet_id,
            doorbell_matches_last_packet: self.doorbell_matches_last_packet,
            publish_low32: self.publish_low32,
            header_release_width_bits: self.header_release_width_bits,
            live_low32: self.live_low32,
            valid_header_not_stored: self.valid_header_not_stored,
            fetch_add_not_performed: self.fetch_add_not_performed,
            doorbell_not_written: self.doorbell_not_written,
            capacity_formula_ok: self.capacity_formula_ok,
            slot_formula_ok: self.slot_formula_ok,
            metadata_ready_dependency: self.metadata_ready_dependency,
            non_consuming_contract: self.non_consuming_contract,
            observed_ready: self.observed_ready,
            expected_capacity_ok,
            expected_slot_aligned64,
            expected_doorbell_matches_last_packet,
            expected_valid_header_not_stored,
            expected_metadata_ready_dependency,
            expected_slot_formula_ok,
            ready,
        }
    }
}

impl KfdQueueLiveAqlReservationProof {
    pub fn validate_ready(self) -> KfdQueueLiveAqlReservationValidation {
        let printed_ready = self.observed_ready == 1;
        let expected_capacity_ok = self.expected_capacity_ok == 1;
        let expected_slot_aligned64 = self.expected_slot_aligned64 == 1;
        let expected_doorbell_matches_last_packet = self.expected_doorbell_matches_last_packet == 1;
        let expected_valid_header_not_stored = self.expected_valid_header_not_stored == 1;
        let expected_metadata_ready_dependency = self.expected_metadata_ready_dependency == 1;
        let expected_slot_formula_ok = self.expected_slot_formula_ok == 1;
        let no_fetch_add = self.fetch_add_not_performed == 1;
        let no_doorbell = self.doorbell_not_written == 1;
        let non_consuming_contract = self.non_consuming_contract == 1;
        let ready = self.ready == 1;
        let passed = printed_ready
            && expected_capacity_ok
            && expected_slot_aligned64
            && expected_doorbell_matches_last_packet
            && expected_valid_header_not_stored
            && expected_metadata_ready_dependency
            && expected_slot_formula_ok
            && no_fetch_add
            && no_doorbell
            && non_consuming_contract
            && ready;

        KfdQueueLiveAqlReservationValidation {
            printed_ready,
            expected_capacity_ok,
            expected_slot_aligned64,
            expected_doorbell_matches_last_packet,
            expected_valid_header_not_stored,
            expected_metadata_ready_dependency,
            expected_slot_formula_ok,
            no_fetch_add,
            no_doorbell,
            non_consuming_contract,
            ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReserveBeforeStageInput {
    pub probe_version: u64,
    pub staged_packet_id: u64,
    pub reserved_packet_id: u64,
    pub staged_slot_va: u64,
    pub reserved_slot_va: u64,
    pub staged_slot_offset: u64,
    pub reserved_slot_offset: u64,
    pub same_packet_id: u64,
    pub same_slot: u64,
    pub old_payload_write_ready: u64,
    pub old_payload_publishable: u64,
    pub must_restage_after_reserve: u64,
    pub publish_blocked_until_restage: u64,
    pub old_slot_still_invalid: u64,
    pub reservation_ready_dependency: u64,
    pub valid_header_not_stored: u64,
    pub publish_low32: u64,
    pub live_low32: u64,
    pub slot_progress_observed: u64,
    pub desired_write_index: u64,
    pub doorbell_packet_id: u64,
    pub capacity_ok: u64,
    pub slot_formula_ok: u64,
    pub fetch_add_not_performed: u64,
    pub reserved_slot_not_written: u64,
    pub header_not_published: u64,
    pub doorbell_not_written: u64,
    pub reserve_first_contract: u64,
    pub reserved_slot_stage_required: u64,
    pub non_consuming_contract: u64,
    pub sequence_ready: u64,
    pub observed_ready: u64,
    pub expected_same_packet_id: bool,
    pub expected_same_slot: bool,
    pub expected_old_payload_publishable: bool,
    pub expected_must_restage_after_reserve: bool,
    pub expected_reservation_ready_dependency: bool,
    pub expected_capacity_ok: bool,
    pub expected_slot_formula_ok: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReserveBeforeStageProof {
    pub probe_version: u64,
    pub staged_packet_id: u64,
    pub reserved_packet_id: u64,
    pub staged_slot_va: u64,
    pub reserved_slot_va: u64,
    pub staged_slot_offset: u64,
    pub reserved_slot_offset: u64,
    pub same_packet_id: u64,
    pub same_slot: u64,
    pub old_payload_write_ready: u64,
    pub old_payload_publishable: u64,
    pub must_restage_after_reserve: u64,
    pub publish_blocked_until_restage: u64,
    pub old_slot_still_invalid: u64,
    pub reservation_ready_dependency: u64,
    pub valid_header_not_stored: u64,
    pub publish_low32: u64,
    pub live_low32: u64,
    pub slot_progress_observed: u64,
    pub desired_write_index: u64,
    pub doorbell_packet_id: u64,
    pub capacity_ok: u64,
    pub slot_formula_ok: u64,
    pub fetch_add_not_performed: u64,
    pub reserved_slot_not_written: u64,
    pub header_not_published: u64,
    pub doorbell_not_written: u64,
    pub reserve_first_contract: u64,
    pub reserved_slot_stage_required: u64,
    pub non_consuming_contract: u64,
    pub sequence_ready: u64,
    pub observed_ready: u64,
    pub expected_same_packet_id: u64,
    pub expected_same_slot: u64,
    pub expected_old_payload_publishable: u64,
    pub expected_must_restage_after_reserve: u64,
    pub expected_reservation_ready_dependency: u64,
    pub expected_capacity_ok: u64,
    pub expected_slot_formula_ok: u64,
    pub same_packet_id_matches_expected: u64,
    pub same_slot_matches_expected: u64,
    pub old_payload_publishable_matches_expected: u64,
    pub must_restage_matches_expected: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReserveBeforeStageValidation {
    pub printed_ready: bool,
    pub same_packet_id_matches_expected: bool,
    pub same_slot_matches_expected: bool,
    pub old_payload_publishable_matches_expected: bool,
    pub must_restage_matches_expected: bool,
    pub reservation_ready_dependency: bool,
    pub expected_reservation_ready_dependency: bool,
    pub expected_capacity_ok: bool,
    pub expected_slot_formula_ok: bool,
    pub publish_blocked_until_restage: bool,
    pub old_slot_still_invalid: bool,
    pub valid_header_not_stored: bool,
    pub no_fetch_add: bool,
    pub reserved_slot_not_written: bool,
    pub header_not_published: bool,
    pub no_doorbell: bool,
    pub reserve_first_contract: bool,
    pub reserved_slot_stage_required: bool,
    pub non_consuming_contract: bool,
    pub sequence_ready: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlReserveBeforeStageInput {
    pub fn proof(self) -> KfdQueueLiveAqlReserveBeforeStageProof {
        let expected_same_packet_id = self.expected_same_packet_id as u64;
        let expected_same_slot = self.expected_same_slot as u64;
        let expected_old_payload_publishable = self.expected_old_payload_publishable as u64;
        let expected_must_restage_after_reserve = self.expected_must_restage_after_reserve as u64;
        let expected_reservation_ready_dependency =
            self.expected_reservation_ready_dependency as u64;
        let expected_capacity_ok = self.expected_capacity_ok as u64;
        let expected_slot_formula_ok = self.expected_slot_formula_ok as u64;
        let same_packet_id_matches_expected =
            (self.same_packet_id == expected_same_packet_id) as u64;
        let same_slot_matches_expected = (self.same_slot == expected_same_slot) as u64;
        let old_payload_publishable_matches_expected =
            (self.old_payload_publishable == expected_old_payload_publishable) as u64;
        let must_restage_matches_expected =
            (self.must_restage_after_reserve == expected_must_restage_after_reserve) as u64;
        let ready = (same_packet_id_matches_expected == 1
            && same_slot_matches_expected == 1
            && old_payload_publishable_matches_expected == 1
            && must_restage_matches_expected == 1
            && self.expected_reservation_ready_dependency
            && self.expected_capacity_ok
            && self.expected_slot_formula_ok
            && self.reservation_ready_dependency == 1
            && self.capacity_ok == 1
            && self.slot_formula_ok == 1
            && self.publish_blocked_until_restage == 1
            && self.old_slot_still_invalid == 1
            && self.valid_header_not_stored == 1
            && self.fetch_add_not_performed == 1
            && self.reserved_slot_not_written == 1
            && self.header_not_published == 1
            && self.doorbell_not_written == 1
            && self.reserve_first_contract == 1
            && self.reserved_slot_stage_required == 1
            && self.non_consuming_contract == 1
            && self.sequence_ready == 1) as u64;

        KfdQueueLiveAqlReserveBeforeStageProof {
            probe_version: self.probe_version,
            staged_packet_id: self.staged_packet_id,
            reserved_packet_id: self.reserved_packet_id,
            staged_slot_va: self.staged_slot_va,
            reserved_slot_va: self.reserved_slot_va,
            staged_slot_offset: self.staged_slot_offset,
            reserved_slot_offset: self.reserved_slot_offset,
            same_packet_id: self.same_packet_id,
            same_slot: self.same_slot,
            old_payload_write_ready: self.old_payload_write_ready,
            old_payload_publishable: self.old_payload_publishable,
            must_restage_after_reserve: self.must_restage_after_reserve,
            publish_blocked_until_restage: self.publish_blocked_until_restage,
            old_slot_still_invalid: self.old_slot_still_invalid,
            reservation_ready_dependency: self.reservation_ready_dependency,
            valid_header_not_stored: self.valid_header_not_stored,
            publish_low32: self.publish_low32,
            live_low32: self.live_low32,
            slot_progress_observed: self.slot_progress_observed,
            desired_write_index: self.desired_write_index,
            doorbell_packet_id: self.doorbell_packet_id,
            capacity_ok: self.capacity_ok,
            slot_formula_ok: self.slot_formula_ok,
            fetch_add_not_performed: self.fetch_add_not_performed,
            reserved_slot_not_written: self.reserved_slot_not_written,
            header_not_published: self.header_not_published,
            doorbell_not_written: self.doorbell_not_written,
            reserve_first_contract: self.reserve_first_contract,
            reserved_slot_stage_required: self.reserved_slot_stage_required,
            non_consuming_contract: self.non_consuming_contract,
            sequence_ready: self.sequence_ready,
            observed_ready: self.observed_ready,
            expected_same_packet_id,
            expected_same_slot,
            expected_old_payload_publishable,
            expected_must_restage_after_reserve,
            expected_reservation_ready_dependency,
            expected_capacity_ok,
            expected_slot_formula_ok,
            same_packet_id_matches_expected,
            same_slot_matches_expected,
            old_payload_publishable_matches_expected,
            must_restage_matches_expected,
            ready,
        }
    }
}

impl KfdQueueLiveAqlReserveBeforeStageProof {
    pub fn validate_ready(self) -> KfdQueueLiveAqlReserveBeforeStageValidation {
        let printed_ready = self.observed_ready == 1;
        let same_packet_id_matches_expected = self.same_packet_id_matches_expected == 1;
        let same_slot_matches_expected = self.same_slot_matches_expected == 1;
        let old_payload_publishable_matches_expected =
            self.old_payload_publishable_matches_expected == 1;
        let must_restage_matches_expected = self.must_restage_matches_expected == 1;
        let reservation_ready_dependency = self.reservation_ready_dependency == 1;
        let expected_reservation_ready_dependency = self.expected_reservation_ready_dependency == 1;
        let expected_capacity_ok = self.expected_capacity_ok == 1;
        let expected_slot_formula_ok = self.expected_slot_formula_ok == 1;
        let publish_blocked_until_restage = self.publish_blocked_until_restage == 1;
        let old_slot_still_invalid = self.old_slot_still_invalid == 1;
        let valid_header_not_stored = self.valid_header_not_stored == 1;
        let no_fetch_add = self.fetch_add_not_performed == 1;
        let reserved_slot_not_written = self.reserved_slot_not_written == 1;
        let header_not_published = self.header_not_published == 1;
        let no_doorbell = self.doorbell_not_written == 1;
        let reserve_first_contract = self.reserve_first_contract == 1;
        let reserved_slot_stage_required = self.reserved_slot_stage_required == 1;
        let non_consuming_contract = self.non_consuming_contract == 1;
        let sequence_ready = self.sequence_ready == 1;
        let ready = self.ready == 1;
        let passed = printed_ready
            && same_packet_id_matches_expected
            && same_slot_matches_expected
            && old_payload_publishable_matches_expected
            && must_restage_matches_expected
            && reservation_ready_dependency
            && expected_reservation_ready_dependency
            && expected_capacity_ok
            && expected_slot_formula_ok
            && publish_blocked_until_restage
            && old_slot_still_invalid
            && valid_header_not_stored
            && no_fetch_add
            && reserved_slot_not_written
            && header_not_published
            && no_doorbell
            && reserve_first_contract
            && reserved_slot_stage_required
            && non_consuming_contract
            && sequence_ready
            && ready;

        KfdQueueLiveAqlReserveBeforeStageValidation {
            printed_ready,
            same_packet_id_matches_expected,
            same_slot_matches_expected,
            old_payload_publishable_matches_expected,
            must_restage_matches_expected,
            reservation_ready_dependency,
            expected_reservation_ready_dependency,
            expected_capacity_ok,
            expected_slot_formula_ok,
            publish_blocked_until_restage,
            old_slot_still_invalid,
            valid_header_not_stored,
            no_fetch_add,
            reserved_slot_not_written,
            header_not_published,
            no_doorbell,
            reserve_first_contract,
            reserved_slot_stage_required,
            non_consuming_contract,
            sequence_ready,
            ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReserveFirstRestageInput {
    pub probe_version: u64,
    pub target_packet_id: u64,
    pub target_slot_va: u64,
    pub target_slot_offset: u64,
    pub reservation_packet_id: u64,
    pub reservation_slot_va: u64,
    pub reservation_slot_offset: u64,
    pub target_matches_reservation: u64,
    pub old_packet_id: u64,
    pub old_slot_va: u64,
    pub old_slot_bypassed: u64,
    pub payload_inputs_ready: u64,
    pub publish_low32: u64,
    pub live_low32: u64,
    pub valid_header_store_pending: u64,
    pub reserved_slot_write_pending: u64,
    pub write_index_fetch_add_pending: u64,
    pub doorbell_pending: u64,
    pub release_header_after_payload_contract: u64,
    pub reserve_before_payload_contract: u64,
    pub doorbell_after_header_contract: u64,
    pub no_live_queue_mutation_contract: u64,
    pub observed_plan_ready: u64,
    pub capacity_ok: u64,
    pub slot_formula_ok: u64,
    pub desired_write_index: u64,
    pub doorbell_packet_id: u64,
    pub packet_bytes: u64,
    pub ring_slots: u64,
    pub slot_mask: u64,
    pub publish_blocked_before_restage: u64,
    pub observed_ready: u64,
    pub expected_must_restage: bool,
    pub expected_target_matches_reservation: bool,
    pub expected_payload_inputs_ready: bool,
    pub expected_capacity_ok: bool,
    pub expected_slot_formula_ok: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReserveFirstRestageProof {
    pub probe_version: u64,
    pub target_packet_id: u64,
    pub target_slot_va: u64,
    pub target_slot_offset: u64,
    pub reservation_packet_id: u64,
    pub reservation_slot_va: u64,
    pub reservation_slot_offset: u64,
    pub target_matches_reservation: u64,
    pub old_packet_id: u64,
    pub old_slot_va: u64,
    pub old_slot_bypassed: u64,
    pub payload_inputs_ready: u64,
    pub publish_low32: u64,
    pub live_low32: u64,
    pub valid_header_store_pending: u64,
    pub reserved_slot_write_pending: u64,
    pub write_index_fetch_add_pending: u64,
    pub doorbell_pending: u64,
    pub release_header_after_payload_contract: u64,
    pub reserve_before_payload_contract: u64,
    pub doorbell_after_header_contract: u64,
    pub no_live_queue_mutation_contract: u64,
    pub observed_plan_ready: u64,
    pub capacity_ok: u64,
    pub slot_formula_ok: u64,
    pub desired_write_index: u64,
    pub doorbell_packet_id: u64,
    pub packet_bytes: u64,
    pub ring_slots: u64,
    pub slot_mask: u64,
    pub publish_blocked_before_restage: u64,
    pub observed_ready: u64,
    pub expected_must_restage: u64,
    pub expected_target_matches_reservation: u64,
    pub expected_payload_inputs_ready: u64,
    pub expected_capacity_ok: u64,
    pub expected_slot_formula_ok: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlReserveFirstRestageValidation {
    pub printed_plan_ready: bool,
    pub printed_ready: bool,
    pub expected_must_restage: bool,
    pub expected_target_matches_reservation: bool,
    pub expected_payload_inputs_ready: bool,
    pub expected_capacity_ok: bool,
    pub expected_slot_formula_ok: bool,
    pub payload_inputs_ready: bool,
    pub valid_header_store_pending: bool,
    pub reserved_slot_write_pending: bool,
    pub write_index_fetch_add_pending: bool,
    pub doorbell_pending: bool,
    pub release_header_after_payload_contract: bool,
    pub reserve_before_payload_contract: bool,
    pub doorbell_after_header_contract: bool,
    pub publish_blocked_before_restage: bool,
    pub no_live_queue_mutation_contract: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlReserveFirstRestageInput {
    pub fn proof(self) -> KfdQueueLiveAqlReserveFirstRestageProof {
        let expected_must_restage = self.expected_must_restage as u64;
        let expected_target_matches_reservation = self.expected_target_matches_reservation as u64;
        let expected_payload_inputs_ready = self.expected_payload_inputs_ready as u64;
        let expected_capacity_ok = self.expected_capacity_ok as u64;
        let expected_slot_formula_ok = self.expected_slot_formula_ok as u64;
        let payload_inputs_ready = self.payload_inputs_ready == 1;
        let valid_header_store_pending = self.valid_header_store_pending == 1;
        let reserved_slot_write_pending = self.reserved_slot_write_pending == 1;
        let write_index_fetch_add_pending = self.write_index_fetch_add_pending == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let release_header_after_payload_contract = self.release_header_after_payload_contract == 1;
        let reserve_before_payload_contract = self.reserve_before_payload_contract == 1;
        let doorbell_after_header_contract = self.doorbell_after_header_contract == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let publish_blocked_before_restage = self.publish_blocked_before_restage == 1;
        let ready = (self.expected_must_restage
            && self.expected_target_matches_reservation
            && self.expected_payload_inputs_ready
            && self.expected_capacity_ok
            && self.expected_slot_formula_ok
            && payload_inputs_ready
            && valid_header_store_pending
            && reserved_slot_write_pending
            && write_index_fetch_add_pending
            && doorbell_pending
            && release_header_after_payload_contract
            && reserve_before_payload_contract
            && doorbell_after_header_contract
            && no_live_queue_mutation_contract
            && publish_blocked_before_restage) as u64;

        KfdQueueLiveAqlReserveFirstRestageProof {
            probe_version: self.probe_version,
            target_packet_id: self.target_packet_id,
            target_slot_va: self.target_slot_va,
            target_slot_offset: self.target_slot_offset,
            reservation_packet_id: self.reservation_packet_id,
            reservation_slot_va: self.reservation_slot_va,
            reservation_slot_offset: self.reservation_slot_offset,
            target_matches_reservation: self.target_matches_reservation,
            old_packet_id: self.old_packet_id,
            old_slot_va: self.old_slot_va,
            old_slot_bypassed: self.old_slot_bypassed,
            payload_inputs_ready: self.payload_inputs_ready,
            publish_low32: self.publish_low32,
            live_low32: self.live_low32,
            valid_header_store_pending: self.valid_header_store_pending,
            reserved_slot_write_pending: self.reserved_slot_write_pending,
            write_index_fetch_add_pending: self.write_index_fetch_add_pending,
            doorbell_pending: self.doorbell_pending,
            release_header_after_payload_contract: self.release_header_after_payload_contract,
            reserve_before_payload_contract: self.reserve_before_payload_contract,
            doorbell_after_header_contract: self.doorbell_after_header_contract,
            no_live_queue_mutation_contract: self.no_live_queue_mutation_contract,
            observed_plan_ready: self.observed_plan_ready,
            capacity_ok: self.capacity_ok,
            slot_formula_ok: self.slot_formula_ok,
            desired_write_index: self.desired_write_index,
            doorbell_packet_id: self.doorbell_packet_id,
            packet_bytes: self.packet_bytes,
            ring_slots: self.ring_slots,
            slot_mask: self.slot_mask,
            publish_blocked_before_restage: self.publish_blocked_before_restage,
            observed_ready: self.observed_ready,
            expected_must_restage,
            expected_target_matches_reservation,
            expected_payload_inputs_ready,
            expected_capacity_ok,
            expected_slot_formula_ok,
            ready,
        }
    }
}

impl KfdQueueLiveAqlReserveFirstRestageProof {
    pub fn validate_ready(self) -> KfdQueueLiveAqlReserveFirstRestageValidation {
        let printed_plan_ready = self.observed_plan_ready == 1;
        let printed_ready = self.observed_ready == 1;
        let expected_must_restage = self.expected_must_restage == 1;
        let expected_target_matches_reservation = self.expected_target_matches_reservation == 1;
        let expected_payload_inputs_ready = self.expected_payload_inputs_ready == 1;
        let expected_capacity_ok = self.expected_capacity_ok == 1;
        let expected_slot_formula_ok = self.expected_slot_formula_ok == 1;
        let payload_inputs_ready = self.payload_inputs_ready == 1;
        let valid_header_store_pending = self.valid_header_store_pending == 1;
        let reserved_slot_write_pending = self.reserved_slot_write_pending == 1;
        let write_index_fetch_add_pending = self.write_index_fetch_add_pending == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let release_header_after_payload_contract = self.release_header_after_payload_contract == 1;
        let reserve_before_payload_contract = self.reserve_before_payload_contract == 1;
        let doorbell_after_header_contract = self.doorbell_after_header_contract == 1;
        let publish_blocked_before_restage = self.publish_blocked_before_restage == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let ready = self.ready == 1;
        let passed = printed_plan_ready
            && printed_ready
            && expected_must_restage
            && expected_target_matches_reservation
            && expected_payload_inputs_ready
            && expected_capacity_ok
            && expected_slot_formula_ok
            && payload_inputs_ready
            && valid_header_store_pending
            && reserved_slot_write_pending
            && write_index_fetch_add_pending
            && doorbell_pending
            && release_header_after_payload_contract
            && reserve_before_payload_contract
            && doorbell_after_header_contract
            && publish_blocked_before_restage
            && no_live_queue_mutation_contract
            && ready;

        KfdQueueLiveAqlReserveFirstRestageValidation {
            printed_plan_ready,
            printed_ready,
            expected_must_restage,
            expected_target_matches_reservation,
            expected_payload_inputs_ready,
            expected_capacity_ok,
            expected_slot_formula_ok,
            payload_inputs_ready,
            valid_header_store_pending,
            reserved_slot_write_pending,
            write_index_fetch_add_pending,
            doorbell_pending,
            release_header_after_payload_contract,
            reserve_before_payload_contract,
            doorbell_after_header_contract,
            publish_blocked_before_restage,
            no_live_queue_mutation_contract,
            ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlBatchReservationPlanInput {
    pub probe_version: u64,
    pub base_packet_id: u64,
    pub packet_count: u64,
    pub last_packet_id: u64,
    pub desired_write_index: u64,
    pub read_index: u64,
    pub inflight_packets: u64,
    pub capacity_ok: u64,
    pub slot0_va: u64,
    pub slot1_va: u64,
    pub slot0_offset: u64,
    pub slot1_offset: u64,
    pub slot0_index: u64,
    pub slot1_index: u64,
    pub slots_distinct: u64,
    pub slots_aligned64: u64,
    pub slot0_formula_ok: u64,
    pub slot1_formula_ok: u64,
    pub doorbell_packet_id: u64,
    pub doorbell_matches_last_packet: u64,
    pub single_doorbell_contract: u64,
    pub reserve_before_payload_contract: u64,
    pub payloads_before_headers_contract: u64,
    pub headers_before_doorbell_contract: u64,
    pub release_header_store_contract: u64,
    pub write_index_fetch_add_pending: u64,
    pub payload_writes_pending: u64,
    pub valid_headers_pending: u64,
    pub doorbell_pending: u64,
    pub no_live_queue_mutation_contract: u64,
    pub first_slot_matches_single_reservation: u64,
    pub observed_ready: u64,
    pub expected_restage_or_payload_ready: bool,
    pub expected_capacity_ok: bool,
    pub expected_slots_distinct: bool,
    pub expected_slots_aligned64: bool,
    pub expected_slot0_formula_ok: bool,
    pub expected_slot1_formula_ok: bool,
    pub expected_doorbell_matches_last_packet: bool,
    pub expected_first_slot_matches_single_reservation: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlBatchReservationPlanProof {
    pub probe_version: u64,
    pub base_packet_id: u64,
    pub packet_count: u64,
    pub last_packet_id: u64,
    pub desired_write_index: u64,
    pub read_index: u64,
    pub inflight_packets: u64,
    pub capacity_ok: u64,
    pub slot0_va: u64,
    pub slot1_va: u64,
    pub slot0_offset: u64,
    pub slot1_offset: u64,
    pub slot0_index: u64,
    pub slot1_index: u64,
    pub slots_distinct: u64,
    pub slots_aligned64: u64,
    pub slot0_formula_ok: u64,
    pub slot1_formula_ok: u64,
    pub doorbell_packet_id: u64,
    pub doorbell_matches_last_packet: u64,
    pub single_doorbell_contract: u64,
    pub reserve_before_payload_contract: u64,
    pub payloads_before_headers_contract: u64,
    pub headers_before_doorbell_contract: u64,
    pub release_header_store_contract: u64,
    pub write_index_fetch_add_pending: u64,
    pub payload_writes_pending: u64,
    pub valid_headers_pending: u64,
    pub doorbell_pending: u64,
    pub no_live_queue_mutation_contract: u64,
    pub first_slot_matches_single_reservation: u64,
    pub observed_ready: u64,
    pub expected_restage_or_payload_ready: u64,
    pub expected_capacity_ok: u64,
    pub expected_slots_distinct: u64,
    pub expected_slots_aligned64: u64,
    pub expected_slot0_formula_ok: u64,
    pub expected_slot1_formula_ok: u64,
    pub expected_doorbell_matches_last_packet: u64,
    pub expected_first_slot_matches_single_reservation: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlBatchReservationPlanValidation {
    pub printed_plan_ready: bool,
    pub expected_restage_or_payload_ready: bool,
    pub expected_capacity_ok: bool,
    pub expected_slots_distinct: bool,
    pub expected_slots_aligned64: bool,
    pub expected_slot0_formula_ok: bool,
    pub expected_slot1_formula_ok: bool,
    pub expected_doorbell_matches_last_packet: bool,
    pub expected_first_slot_matches_single_reservation: bool,
    pub single_doorbell_contract: bool,
    pub reserve_before_payload_contract: bool,
    pub payloads_before_headers_contract: bool,
    pub headers_before_doorbell_contract: bool,
    pub release_header_store_contract: bool,
    pub write_index_fetch_add_pending: bool,
    pub payload_writes_pending: bool,
    pub valid_headers_pending: bool,
    pub doorbell_pending: bool,
    pub no_live_queue_mutation_contract: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlBatchReservationPlanInput {
    pub fn proof(self) -> KfdQueueLiveAqlBatchReservationPlanProof {
        let expected_restage_or_payload_ready = self.expected_restage_or_payload_ready as u64;
        let expected_capacity_ok = self.expected_capacity_ok as u64;
        let expected_slots_distinct = self.expected_slots_distinct as u64;
        let expected_slots_aligned64 = self.expected_slots_aligned64 as u64;
        let expected_slot0_formula_ok = self.expected_slot0_formula_ok as u64;
        let expected_slot1_formula_ok = self.expected_slot1_formula_ok as u64;
        let expected_doorbell_matches_last_packet =
            self.expected_doorbell_matches_last_packet as u64;
        let expected_first_slot_matches_single_reservation =
            self.expected_first_slot_matches_single_reservation as u64;
        let single_doorbell_contract = self.single_doorbell_contract == 1;
        let reserve_before_payload_contract = self.reserve_before_payload_contract == 1;
        let payloads_before_headers_contract = self.payloads_before_headers_contract == 1;
        let headers_before_doorbell_contract = self.headers_before_doorbell_contract == 1;
        let release_header_store_contract = self.release_header_store_contract == 1;
        let write_index_fetch_add_pending = self.write_index_fetch_add_pending == 1;
        let payload_writes_pending = self.payload_writes_pending == 1;
        let valid_headers_pending = self.valid_headers_pending == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let ready = (self.expected_restage_or_payload_ready
            && self.expected_capacity_ok
            && self.expected_slots_distinct
            && self.expected_slots_aligned64
            && self.expected_slot0_formula_ok
            && self.expected_slot1_formula_ok
            && self.expected_doorbell_matches_last_packet
            && self.expected_first_slot_matches_single_reservation
            && single_doorbell_contract
            && reserve_before_payload_contract
            && payloads_before_headers_contract
            && headers_before_doorbell_contract
            && release_header_store_contract
            && write_index_fetch_add_pending
            && payload_writes_pending
            && valid_headers_pending
            && doorbell_pending
            && no_live_queue_mutation_contract) as u64;

        KfdQueueLiveAqlBatchReservationPlanProof {
            probe_version: self.probe_version,
            base_packet_id: self.base_packet_id,
            packet_count: self.packet_count,
            last_packet_id: self.last_packet_id,
            desired_write_index: self.desired_write_index,
            read_index: self.read_index,
            inflight_packets: self.inflight_packets,
            capacity_ok: self.capacity_ok,
            slot0_va: self.slot0_va,
            slot1_va: self.slot1_va,
            slot0_offset: self.slot0_offset,
            slot1_offset: self.slot1_offset,
            slot0_index: self.slot0_index,
            slot1_index: self.slot1_index,
            slots_distinct: self.slots_distinct,
            slots_aligned64: self.slots_aligned64,
            slot0_formula_ok: self.slot0_formula_ok,
            slot1_formula_ok: self.slot1_formula_ok,
            doorbell_packet_id: self.doorbell_packet_id,
            doorbell_matches_last_packet: self.doorbell_matches_last_packet,
            single_doorbell_contract: self.single_doorbell_contract,
            reserve_before_payload_contract: self.reserve_before_payload_contract,
            payloads_before_headers_contract: self.payloads_before_headers_contract,
            headers_before_doorbell_contract: self.headers_before_doorbell_contract,
            release_header_store_contract: self.release_header_store_contract,
            write_index_fetch_add_pending: self.write_index_fetch_add_pending,
            payload_writes_pending: self.payload_writes_pending,
            valid_headers_pending: self.valid_headers_pending,
            doorbell_pending: self.doorbell_pending,
            no_live_queue_mutation_contract: self.no_live_queue_mutation_contract,
            first_slot_matches_single_reservation: self.first_slot_matches_single_reservation,
            observed_ready: self.observed_ready,
            expected_restage_or_payload_ready,
            expected_capacity_ok,
            expected_slots_distinct,
            expected_slots_aligned64,
            expected_slot0_formula_ok,
            expected_slot1_formula_ok,
            expected_doorbell_matches_last_packet,
            expected_first_slot_matches_single_reservation,
            ready,
        }
    }
}

impl KfdQueueLiveAqlBatchReservationPlanProof {
    pub fn validate_ready(self) -> KfdQueueLiveAqlBatchReservationPlanValidation {
        let printed_plan_ready = self.observed_ready == 1;
        let expected_restage_or_payload_ready = self.expected_restage_or_payload_ready == 1;
        let expected_capacity_ok = self.expected_capacity_ok == 1;
        let expected_slots_distinct = self.expected_slots_distinct == 1;
        let expected_slots_aligned64 = self.expected_slots_aligned64 == 1;
        let expected_slot0_formula_ok = self.expected_slot0_formula_ok == 1;
        let expected_slot1_formula_ok = self.expected_slot1_formula_ok == 1;
        let expected_doorbell_matches_last_packet = self.expected_doorbell_matches_last_packet == 1;
        let expected_first_slot_matches_single_reservation =
            self.expected_first_slot_matches_single_reservation == 1;
        let single_doorbell_contract = self.single_doorbell_contract == 1;
        let reserve_before_payload_contract = self.reserve_before_payload_contract == 1;
        let payloads_before_headers_contract = self.payloads_before_headers_contract == 1;
        let headers_before_doorbell_contract = self.headers_before_doorbell_contract == 1;
        let release_header_store_contract = self.release_header_store_contract == 1;
        let write_index_fetch_add_pending = self.write_index_fetch_add_pending == 1;
        let payload_writes_pending = self.payload_writes_pending == 1;
        let valid_headers_pending = self.valid_headers_pending == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let ready = self.ready == 1;
        let passed = printed_plan_ready
            && expected_restage_or_payload_ready
            && expected_capacity_ok
            && expected_slots_distinct
            && expected_slots_aligned64
            && expected_slot0_formula_ok
            && expected_slot1_formula_ok
            && expected_doorbell_matches_last_packet
            && expected_first_slot_matches_single_reservation
            && single_doorbell_contract
            && reserve_before_payload_contract
            && payloads_before_headers_contract
            && headers_before_doorbell_contract
            && release_header_store_contract
            && write_index_fetch_add_pending
            && payload_writes_pending
            && valid_headers_pending
            && doorbell_pending
            && no_live_queue_mutation_contract
            && ready;

        KfdQueueLiveAqlBatchReservationPlanValidation {
            printed_plan_ready,
            expected_restage_or_payload_ready,
            expected_capacity_ok,
            expected_slots_distinct,
            expected_slots_aligned64,
            expected_slot0_formula_ok,
            expected_slot1_formula_ok,
            expected_doorbell_matches_last_packet,
            expected_first_slot_matches_single_reservation,
            single_doorbell_contract,
            reserve_before_payload_contract,
            payloads_before_headers_contract,
            headers_before_doorbell_contract,
            release_header_store_contract,
            write_index_fetch_add_pending,
            payload_writes_pending,
            valid_headers_pending,
            doorbell_pending,
            no_live_queue_mutation_contract,
            ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlMaterializedPacketPlanInput {
    pub probe_version: u64,
    pub packet0_packet_id: u64,
    pub packet1_packet_id: u64,
    pub packet0_slot_va: u64,
    pub packet1_slot_va: u64,
    pub packet0_word0: u64,
    pub packet0_word4_kernel_object: u64,
    pub packet0_word5_kernarg_va: u64,
    pub packet1_word0: u64,
    pub packet1_word4_kernel_object: u64,
    pub packet1_word5_kernarg_va: u64,
    pub packet0_words_match_host_template: u64,
    pub packet1_words_match_host_template: u64,
    pub payload_words_match_host_template: u64,
    pub header_words_match_host_template: u64,
    pub target_slots_match_batch_plan: u64,
    pub packet0_slot_offset: u64,
    pub packet1_slot_offset: u64,
    pub packet_bytes: u64,
    pub packet_count: u64,
    pub batch_plan_ready: u64,
    pub reserve_first_restage_ready: u64,
    pub payloads_before_headers_contract: u64,
    pub release_header_store_contract: u64,
    pub doorbell_pending: u64,
    pub no_live_queue_mutation_contract: u64,
    pub packet_plan_ready: u64,
    pub publish_low32: u64,
    pub packet0_low32: u64,
    pub aql_packet_image_ready: u64,
    pub expected_batch_ready: bool,
    pub expected_reserve_restage_plan_ready: bool,
    pub expected_aql_packet_image_ready: bool,
    pub expected_target_slots_match_batch_plan: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlMaterializedPacketPlanProof {
    pub probe_version: u64,
    pub packet0_packet_id: u64,
    pub packet1_packet_id: u64,
    pub packet0_slot_va: u64,
    pub packet1_slot_va: u64,
    pub packet0_word0: u64,
    pub packet0_word4_kernel_object: u64,
    pub packet0_word5_kernarg_va: u64,
    pub packet1_word0: u64,
    pub packet1_word4_kernel_object: u64,
    pub packet1_word5_kernarg_va: u64,
    pub packet0_words_match_host_template: u64,
    pub packet1_words_match_host_template: u64,
    pub payload_words_match_host_template: u64,
    pub header_words_match_host_template: u64,
    pub target_slots_match_batch_plan: u64,
    pub packet0_slot_offset: u64,
    pub packet1_slot_offset: u64,
    pub packet_bytes: u64,
    pub packet_count: u64,
    pub batch_plan_ready: u64,
    pub reserve_first_restage_ready: u64,
    pub payloads_before_headers_contract: u64,
    pub release_header_store_contract: u64,
    pub doorbell_pending: u64,
    pub no_live_queue_mutation_contract: u64,
    pub packet_plan_ready: u64,
    pub publish_low32: u64,
    pub packet0_low32: u64,
    pub aql_packet_image_ready: u64,
    pub expected_batch_ready: u64,
    pub expected_reserve_restage_plan_ready: u64,
    pub expected_aql_packet_image_ready: u64,
    pub expected_target_slots_match_batch_plan: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlMaterializedPacketPlanValidation {
    pub printed_packet_plan_ready: bool,
    pub expected_batch_ready: bool,
    pub expected_reserve_restage_plan_ready: bool,
    pub expected_aql_packet_image_ready: bool,
    pub expected_target_slots_match_batch_plan: bool,
    pub packet0_words_match_host_template: bool,
    pub packet1_words_match_host_template: bool,
    pub payload_words_match_host_template: bool,
    pub header_words_match_host_template: bool,
    pub target_slots_match_batch_plan: bool,
    pub batch_plan_ready: bool,
    pub reserve_first_restage_ready: bool,
    pub payloads_before_headers_contract: bool,
    pub release_header_store_contract: bool,
    pub doorbell_pending: bool,
    pub no_live_queue_mutation_contract: bool,
    pub aql_packet_image_ready: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlMaterializedPacketPlanInput {
    pub fn proof(self) -> KfdQueueLiveAqlMaterializedPacketPlanProof {
        let expected_batch_ready = self.expected_batch_ready as u64;
        let expected_reserve_restage_plan_ready = self.expected_reserve_restage_plan_ready as u64;
        let expected_aql_packet_image_ready = self.expected_aql_packet_image_ready as u64;
        let expected_target_slots_match_batch_plan =
            self.expected_target_slots_match_batch_plan as u64;
        let packet0_words_match_host_template = self.packet0_words_match_host_template == 1;
        let packet1_words_match_host_template = self.packet1_words_match_host_template == 1;
        let payload_words_match_host_template = self.payload_words_match_host_template == 1;
        let header_words_match_host_template = self.header_words_match_host_template == 1;
        let target_slots_match_batch_plan = self.target_slots_match_batch_plan == 1;
        let batch_plan_ready = self.batch_plan_ready == 1;
        let reserve_first_restage_ready = self.reserve_first_restage_ready == 1;
        let payloads_before_headers_contract = self.payloads_before_headers_contract == 1;
        let release_header_store_contract = self.release_header_store_contract == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let aql_packet_image_ready = self.aql_packet_image_ready == 1;
        let ready = (self.expected_batch_ready
            && self.expected_reserve_restage_plan_ready
            && self.expected_aql_packet_image_ready
            && self.expected_target_slots_match_batch_plan
            && packet0_words_match_host_template
            && packet1_words_match_host_template
            && payload_words_match_host_template
            && header_words_match_host_template
            && target_slots_match_batch_plan
            && batch_plan_ready
            && reserve_first_restage_ready
            && payloads_before_headers_contract
            && release_header_store_contract
            && doorbell_pending
            && no_live_queue_mutation_contract
            && aql_packet_image_ready) as u64;

        KfdQueueLiveAqlMaterializedPacketPlanProof {
            probe_version: self.probe_version,
            packet0_packet_id: self.packet0_packet_id,
            packet1_packet_id: self.packet1_packet_id,
            packet0_slot_va: self.packet0_slot_va,
            packet1_slot_va: self.packet1_slot_va,
            packet0_word0: self.packet0_word0,
            packet0_word4_kernel_object: self.packet0_word4_kernel_object,
            packet0_word5_kernarg_va: self.packet0_word5_kernarg_va,
            packet1_word0: self.packet1_word0,
            packet1_word4_kernel_object: self.packet1_word4_kernel_object,
            packet1_word5_kernarg_va: self.packet1_word5_kernarg_va,
            packet0_words_match_host_template: self.packet0_words_match_host_template,
            packet1_words_match_host_template: self.packet1_words_match_host_template,
            payload_words_match_host_template: self.payload_words_match_host_template,
            header_words_match_host_template: self.header_words_match_host_template,
            target_slots_match_batch_plan: self.target_slots_match_batch_plan,
            packet0_slot_offset: self.packet0_slot_offset,
            packet1_slot_offset: self.packet1_slot_offset,
            packet_bytes: self.packet_bytes,
            packet_count: self.packet_count,
            batch_plan_ready: self.batch_plan_ready,
            reserve_first_restage_ready: self.reserve_first_restage_ready,
            payloads_before_headers_contract: self.payloads_before_headers_contract,
            release_header_store_contract: self.release_header_store_contract,
            doorbell_pending: self.doorbell_pending,
            no_live_queue_mutation_contract: self.no_live_queue_mutation_contract,
            packet_plan_ready: self.packet_plan_ready,
            publish_low32: self.publish_low32,
            packet0_low32: self.packet0_low32,
            aql_packet_image_ready: self.aql_packet_image_ready,
            expected_batch_ready,
            expected_reserve_restage_plan_ready,
            expected_aql_packet_image_ready,
            expected_target_slots_match_batch_plan,
            ready,
        }
    }
}

impl KfdQueueLiveAqlMaterializedPacketPlanProof {
    pub fn validate_ready(self) -> KfdQueueLiveAqlMaterializedPacketPlanValidation {
        let printed_packet_plan_ready = self.packet_plan_ready == 1;
        let expected_batch_ready = self.expected_batch_ready == 1;
        let expected_reserve_restage_plan_ready = self.expected_reserve_restage_plan_ready == 1;
        let expected_aql_packet_image_ready = self.expected_aql_packet_image_ready == 1;
        let expected_target_slots_match_batch_plan =
            self.expected_target_slots_match_batch_plan == 1;
        let packet0_words_match_host_template = self.packet0_words_match_host_template == 1;
        let packet1_words_match_host_template = self.packet1_words_match_host_template == 1;
        let payload_words_match_host_template = self.payload_words_match_host_template == 1;
        let header_words_match_host_template = self.header_words_match_host_template == 1;
        let target_slots_match_batch_plan = self.target_slots_match_batch_plan == 1;
        let batch_plan_ready = self.batch_plan_ready == 1;
        let reserve_first_restage_ready = self.reserve_first_restage_ready == 1;
        let payloads_before_headers_contract = self.payloads_before_headers_contract == 1;
        let release_header_store_contract = self.release_header_store_contract == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let aql_packet_image_ready = self.aql_packet_image_ready == 1;
        let ready = self.ready == 1;
        let passed = printed_packet_plan_ready
            && expected_batch_ready
            && expected_reserve_restage_plan_ready
            && expected_aql_packet_image_ready
            && expected_target_slots_match_batch_plan
            && packet0_words_match_host_template
            && packet1_words_match_host_template
            && payload_words_match_host_template
            && header_words_match_host_template
            && target_slots_match_batch_plan
            && batch_plan_ready
            && reserve_first_restage_ready
            && payloads_before_headers_contract
            && release_header_store_contract
            && doorbell_pending
            && no_live_queue_mutation_contract
            && aql_packet_image_ready
            && ready;

        KfdQueueLiveAqlMaterializedPacketPlanValidation {
            printed_packet_plan_ready,
            expected_batch_ready,
            expected_reserve_restage_plan_ready,
            expected_aql_packet_image_ready,
            expected_target_slots_match_batch_plan,
            packet0_words_match_host_template,
            packet1_words_match_host_template,
            payload_words_match_host_template,
            header_words_match_host_template,
            target_slots_match_batch_plan,
            batch_plan_ready,
            reserve_first_restage_ready,
            payloads_before_headers_contract,
            release_header_store_contract,
            doorbell_pending,
            no_live_queue_mutation_contract,
            aql_packet_image_ready,
            ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlShadowPacketStoreInput {
    pub device_va: u64,
    pub requested_iterations: u64,
    pub executed_iterations: u64,
    pub observed_present: u64,
    pub packet0_word0: u64,
    pub packet1_word0: u64,
    pub words_match_host_template: u64,
    pub payload_words_match_host_template: u64,
    pub header_words_match_host_template: u64,
    pub materialized_source_ready: u64,
    pub payloads_before_headers_contract: u64,
    pub low32_release_headers_last_contract: u64,
    pub doorbell_pending: u64,
    pub no_live_queue_mutation_contract: u64,
    pub region_bytes: u64,
    pub packet_count: u64,
    pub store_ready: u64,
    pub batch_plan_ready: u64,
    pub materialized_ready: bool,
    pub host_present: bool,
    pub host_shadow_words_match: bool,
    pub host_sequence0_match: bool,
    pub host_sentinel_match: bool,
    pub host_sequence1_match: bool,
    pub host_batch_ready_match: bool,
    pub host_poll_ready: bool,
    pub host_poll_header_match: bool,
    pub host_poll_sequence_match: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlShadowPacketStoreProof {
    pub device_va: u64,
    pub requested_iterations: u64,
    pub executed_iterations: u64,
    pub observed_present: u64,
    pub packet0_word0: u64,
    pub packet1_word0: u64,
    pub words_match_host_template: u64,
    pub payload_words_match_host_template: u64,
    pub header_words_match_host_template: u64,
    pub materialized_source_ready: u64,
    pub payloads_before_headers_contract: u64,
    pub low32_release_headers_last_contract: u64,
    pub doorbell_pending: u64,
    pub no_live_queue_mutation_contract: u64,
    pub region_bytes: u64,
    pub packet_count: u64,
    pub store_ready: u64,
    pub batch_plan_ready: u64,
    pub materialized_ready: u64,
    pub host_present: u64,
    pub host_shadow_words_match: u64,
    pub host_sequence0_match: u64,
    pub host_sentinel_match: u64,
    pub host_sequence1_match: u64,
    pub host_batch_ready_match: u64,
    pub host_poll_ready: u64,
    pub host_poll_header_match: u64,
    pub host_poll_sequence_match: u64,
    pub handoff_ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlShadowPacketStoreValidation {
    pub printed_store_ready: bool,
    pub printed_present: bool,
    pub words_match_host_template: bool,
    pub payload_words_match_host_template: bool,
    pub header_words_match_host_template: bool,
    pub materialized_source_ready: bool,
    pub payloads_before_headers_contract: bool,
    pub low32_release_headers_last_contract: bool,
    pub doorbell_pending: bool,
    pub batch_plan_ready: bool,
    pub host_memory_ready: bool,
    pub host_poll_ready: bool,
    pub no_live_queue_mutation_contract: bool,
    pub handoff_ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlShadowPacketStoreInput {
    pub fn proof(self) -> KfdQueueLiveAqlShadowPacketStoreProof {
        let materialized_ready = self.materialized_ready as u64;
        let host_present = self.host_present as u64;
        let host_shadow_words_match = self.host_shadow_words_match as u64;
        let host_sequence0_match = self.host_sequence0_match as u64;
        let host_sentinel_match = self.host_sentinel_match as u64;
        let host_sequence1_match = self.host_sequence1_match as u64;
        let host_batch_ready_match = self.host_batch_ready_match as u64;
        let host_poll_ready = self.host_poll_ready as u64;
        let host_poll_header_match = self.host_poll_header_match as u64;
        let host_poll_sequence_match = self.host_poll_sequence_match as u64;
        let printed_present = self.observed_present == 1;
        let words_match_host_template = self.words_match_host_template == 1;
        let payload_words_match_host_template = self.payload_words_match_host_template == 1;
        let header_words_match_host_template = self.header_words_match_host_template == 1;
        let materialized_source_ready = self.materialized_source_ready == 1;
        let payloads_before_headers_contract = self.payloads_before_headers_contract == 1;
        let low32_release_headers_last_contract = self.low32_release_headers_last_contract == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let batch_plan_ready = self.batch_plan_ready == 1;
        let handoff_ready = (printed_present
            && words_match_host_template
            && payload_words_match_host_template
            && header_words_match_host_template
            && materialized_source_ready
            && payloads_before_headers_contract
            && low32_release_headers_last_contract
            && doorbell_pending
            && no_live_queue_mutation_contract
            && batch_plan_ready
            && self.materialized_ready
            && self.host_present
            && self.host_shadow_words_match
            && self.host_sequence0_match
            && self.host_sentinel_match
            && self.host_sequence1_match
            && self.host_batch_ready_match
            && self.host_poll_ready
            && self.host_poll_header_match
            && self.host_poll_sequence_match) as u64;

        KfdQueueLiveAqlShadowPacketStoreProof {
            device_va: self.device_va,
            requested_iterations: self.requested_iterations,
            executed_iterations: self.executed_iterations,
            observed_present: self.observed_present,
            packet0_word0: self.packet0_word0,
            packet1_word0: self.packet1_word0,
            words_match_host_template: self.words_match_host_template,
            payload_words_match_host_template: self.payload_words_match_host_template,
            header_words_match_host_template: self.header_words_match_host_template,
            materialized_source_ready: self.materialized_source_ready,
            payloads_before_headers_contract: self.payloads_before_headers_contract,
            low32_release_headers_last_contract: self.low32_release_headers_last_contract,
            doorbell_pending: self.doorbell_pending,
            no_live_queue_mutation_contract: self.no_live_queue_mutation_contract,
            region_bytes: self.region_bytes,
            packet_count: self.packet_count,
            store_ready: self.store_ready,
            batch_plan_ready: self.batch_plan_ready,
            materialized_ready,
            host_present,
            host_shadow_words_match,
            host_sequence0_match,
            host_sentinel_match,
            host_sequence1_match,
            host_batch_ready_match,
            host_poll_ready,
            host_poll_header_match,
            host_poll_sequence_match,
            handoff_ready,
        }
    }
}

impl KfdQueueLiveAqlShadowPacketStoreProof {
    pub fn validate_handoff_ready(self) -> KfdQueueLiveAqlShadowPacketStoreValidation {
        let printed_store_ready = self.store_ready == 1;
        let printed_present = self.observed_present == 1;
        let words_match_host_template = self.words_match_host_template == 1;
        let payload_words_match_host_template = self.payload_words_match_host_template == 1;
        let header_words_match_host_template = self.header_words_match_host_template == 1;
        let materialized_source_ready = self.materialized_source_ready == 1;
        let payloads_before_headers_contract = self.payloads_before_headers_contract == 1;
        let low32_release_headers_last_contract = self.low32_release_headers_last_contract == 1;
        let doorbell_pending = self.doorbell_pending == 1;
        let batch_plan_ready = self.batch_plan_ready == 1;
        let host_memory_ready = self.materialized_ready == 1
            && self.host_present == 1
            && self.host_shadow_words_match == 1
            && self.host_sequence0_match == 1
            && self.host_sentinel_match == 1
            && self.host_sequence1_match == 1
            && self.host_batch_ready_match == 1;
        let host_poll_ready = self.host_poll_ready == 1
            && self.host_poll_header_match == 1
            && self.host_poll_sequence_match == 1;
        let no_live_queue_mutation_contract = self.no_live_queue_mutation_contract == 1;
        let handoff_ready = self.handoff_ready == 1;
        let passed = printed_store_ready
            && printed_present
            && words_match_host_template
            && payload_words_match_host_template
            && header_words_match_host_template
            && materialized_source_ready
            && payloads_before_headers_contract
            && low32_release_headers_last_contract
            && doorbell_pending
            && batch_plan_ready
            && host_memory_ready
            && host_poll_ready
            && no_live_queue_mutation_contract
            && handoff_ready;

        KfdQueueLiveAqlShadowPacketStoreValidation {
            printed_store_ready,
            printed_present,
            words_match_host_template,
            payload_words_match_host_template,
            header_words_match_host_template,
            materialized_source_ready,
            payloads_before_headers_contract,
            low32_release_headers_last_contract,
            doorbell_pending,
            batch_plan_ready,
            host_memory_ready,
            host_poll_ready,
            no_live_queue_mutation_contract,
            handoff_ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlHostPollInput {
    pub expected_low32_header: u32,
    pub header0: u32,
    pub header1: u32,
    pub sequence: u64,
    pub expected_sequence: u64,
    pub sentinel: u64,
    pub expected_sentinel: u64,
    pub spins: u64,
    pub elapsed_us: f64,
    pub timeout_ms: f64,
    pub ready_before_device_wait: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlHostPollProof {
    pub expected_low32_header: u32,
    pub header0: u32,
    pub header1: u32,
    pub sequence: u64,
    pub expected_sequence: u64,
    pub sentinel: u64,
    pub spins: u64,
    pub elapsed_us: f64,
    pub timeout_ms: f64,
    pub header0_match: u64,
    pub header1_match: u64,
    pub sequence_match: u64,
    pub sentinel_match: u64,
    pub ready_before_device_wait: u64,
    pub fetch_add_performed: bool,
    pub doorbell_written: bool,
    pub live_queue_mutated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlHostPollValidation {
    pub header0_match: bool,
    pub header1_match: bool,
    pub sequence_match: bool,
    pub sentinel_match: bool,
    pub ready_before_device_wait: bool,
    pub no_fetch_add: bool,
    pub no_doorbell: bool,
    pub no_live_queue_mutation: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlHostPollInput {
    pub fn proof(self) -> KfdQueueLiveAqlHostPollProof {
        let header0_match = (self.header0 == self.expected_low32_header) as u64;
        let header1_match = (self.header1 == self.expected_low32_header) as u64;
        let sequence_match = (self.sequence == self.expected_sequence) as u64;
        let sentinel_match = (self.sentinel == self.expected_sentinel) as u64;
        let ready_before_device_wait = self.ready_before_device_wait as u64;

        KfdQueueLiveAqlHostPollProof {
            expected_low32_header: self.expected_low32_header,
            header0: self.header0,
            header1: self.header1,
            sequence: self.sequence,
            expected_sequence: self.expected_sequence,
            sentinel: self.sentinel,
            spins: self.spins,
            elapsed_us: self.elapsed_us,
            timeout_ms: self.timeout_ms,
            header0_match,
            header1_match,
            sequence_match,
            sentinel_match,
            ready_before_device_wait,
            fetch_add_performed: false,
            doorbell_written: false,
            live_queue_mutated: false,
        }
    }
}

impl KfdQueueLiveAqlHostPollProof {
    pub fn validate_acquire_only(self) -> KfdQueueLiveAqlHostPollValidation {
        let header0_match = self.header0_match == 1;
        let header1_match = self.header1_match == 1;
        let sequence_match = self.sequence_match == 1;
        let sentinel_match = self.sentinel_match == 1;
        let ready_before_device_wait = self.ready_before_device_wait == 1;
        let no_fetch_add = !self.fetch_add_performed;
        let no_doorbell = !self.doorbell_written;
        let no_live_queue_mutation = !self.live_queue_mutated;
        let passed = header0_match
            && header1_match
            && sequence_match
            && sentinel_match
            && ready_before_device_wait
            && no_fetch_add
            && no_doorbell
            && no_live_queue_mutation;

        KfdQueueLiveAqlHostPollValidation {
            header0_match,
            header1_match,
            sequence_match,
            sentinel_match,
            ready_before_device_wait,
            no_fetch_add,
            no_doorbell,
            no_live_queue_mutation,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlAdmissionGuardInput {
    pub shadow_words_match: bool,
    pub header_acquire_match: bool,
    pub sequence_match: bool,
    pub reservation_ready: bool,
    pub restage_ready: bool,
    pub batch_ready: bool,
    pub materialized_ready: bool,
    pub shadow_store_ready: bool,
    pub host_poll_ready: bool,
    pub host_poll_validated: bool,
    pub no_live_mutation_lane0: bool,
    pub no_live_mutation_lane1: bool,
    pub no_live_mutation_lane2: bool,
    pub no_live_mutation_lane3: bool,
    pub no_live_mutation_lane4: bool,
    pub no_live_mutation_lane5: bool,
    pub no_live_mutation_lane6: bool,
    pub no_live_mutation_lane7: bool,
    pub submit_enabled: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlAdmissionGuardProof {
    pub shadow_words_match: u64,
    pub header_acquire_match: u64,
    pub sequence_match: u64,
    pub reservation_ready: u64,
    pub restage_ready: u64,
    pub batch_ready: u64,
    pub materialized_ready: u64,
    pub shadow_store_ready: u64,
    pub host_poll_ready: u64,
    pub host_poll_validated: u64,
    pub prereqs_ready: u64,
    pub no_live_mutation_contract: u64,
    pub token_ready: u64,
    pub submit_enabled: u64,
    pub submit_allowed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlAdmissionGuardValidation {
    pub shadow_words_match: bool,
    pub header_acquire_match: bool,
    pub sequence_match: bool,
    pub host_poll_ready: bool,
    pub host_poll_validated: bool,
    pub prereqs_ready: bool,
    pub no_live_mutation_contract: bool,
    pub token_ready: bool,
    pub submit_disabled: bool,
    pub submit_not_allowed: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlAdmissionGuardInput {
    pub fn proof(self) -> KfdQueueLiveAqlAdmissionGuardProof {
        let shadow_words_match = self.shadow_words_match as u64;
        let header_acquire_match = self.header_acquire_match as u64;
        let sequence_match = self.sequence_match as u64;
        let reservation_ready = self.reservation_ready as u64;
        let restage_ready = self.restage_ready as u64;
        let batch_ready = self.batch_ready as u64;
        let materialized_ready = self.materialized_ready as u64;
        let shadow_store_ready = self.shadow_store_ready as u64;
        let host_poll_ready = self.host_poll_ready as u64;
        let host_poll_validated = self.host_poll_validated as u64;
        let prereqs_ready = (self.reservation_ready
            && self.restage_ready
            && self.batch_ready
            && self.materialized_ready
            && self.shadow_store_ready
            && self.host_poll_ready
            && self.host_poll_validated
            && self.shadow_words_match
            && self.header_acquire_match
            && self.sequence_match) as u64;
        let no_live_mutation_contract = (self.no_live_mutation_lane0
            && self.no_live_mutation_lane1
            && self.no_live_mutation_lane2
            && self.no_live_mutation_lane3
            && self.no_live_mutation_lane4
            && self.no_live_mutation_lane5
            && self.no_live_mutation_lane6
            && self.no_live_mutation_lane7) as u64;
        let token_ready = (prereqs_ready == 1 && no_live_mutation_contract == 1) as u64;
        let submit_allowed = (token_ready == 1 && self.submit_enabled == 1) as u64;

        KfdQueueLiveAqlAdmissionGuardProof {
            shadow_words_match,
            header_acquire_match,
            sequence_match,
            reservation_ready,
            restage_ready,
            batch_ready,
            materialized_ready,
            shadow_store_ready,
            host_poll_ready,
            host_poll_validated,
            prereqs_ready,
            no_live_mutation_contract,
            token_ready,
            submit_enabled: self.submit_enabled,
            submit_allowed,
        }
    }
}

impl KfdQueueLiveAqlAdmissionGuardProof {
    pub fn validate_non_submitting(self) -> KfdQueueLiveAqlAdmissionGuardValidation {
        let shadow_words_match = self.shadow_words_match == 1;
        let header_acquire_match = self.header_acquire_match == 1;
        let sequence_match = self.sequence_match == 1;
        let host_poll_ready = self.host_poll_ready == 1;
        let host_poll_validated = self.host_poll_validated == 1;
        let prereqs_ready = self.prereqs_ready == 1;
        let no_live_mutation_contract = self.no_live_mutation_contract == 1;
        let token_ready = self.token_ready == 1;
        let submit_disabled = self.submit_enabled == 0;
        let submit_not_allowed = self.submit_allowed == 0;
        let passed = shadow_words_match
            && header_acquire_match
            && sequence_match
            && host_poll_ready
            && host_poll_validated
            && prereqs_ready
            && no_live_mutation_contract
            && token_ready
            && submit_disabled
            && submit_not_allowed;

        KfdQueueLiveAqlAdmissionGuardValidation {
            shadow_words_match,
            header_acquire_match,
            sequence_match,
            host_poll_ready,
            host_poll_validated,
            prereqs_ready,
            no_live_mutation_contract,
            token_ready,
            submit_disabled,
            submit_not_allowed,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlSlotPreflightInput {
    pub offline_template_header_low32: u64,
    pub packet_template_header_low32: u64,
    pub expected_publish_low32: u64,
    pub packet_template_kernel_object: u64,
    pub packet_template_kernarg_address: u64,
    pub admission_token_ready: bool,
    pub admission_validated: bool,
    pub admission_submit_enabled: u64,
    pub admission_submit_allowed: u64,
    pub admission_no_live_mutation: bool,
    pub queue_write_index_not_mutated: bool,
    pub queue_read_index_not_mutated: bool,
    pub batch_plan_no_fetch_add: bool,
    pub batch_plan_no_doorbell: bool,
    pub first_slot_matches_reservation: bool,
    pub reservation_ready: bool,
    pub live_write_allowed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlSlotPreflightProof {
    pub offline_template_header_invalid: u64,
    pub packet_template_ready: u64,
    pub admission_token_ready: u64,
    pub admission_validated: u64,
    pub future_write_blocked: u64,
    pub no_ownership_transfer: u64,
    pub first_slot_matches_reservation: u64,
    pub reservation_ready: u64,
    pub ready: u64,
    pub live_write_allowed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlSlotPreflightValidation {
    pub offline_template_header_invalid: bool,
    pub packet_template_ready: bool,
    pub admission_token_ready: bool,
    pub admission_validated: bool,
    pub future_write_blocked: bool,
    pub no_ownership_transfer: bool,
    pub first_slot_matches_reservation: bool,
    pub reservation_ready: bool,
    pub ready: bool,
    pub live_write_disabled: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlSlotPreflightInput {
    pub fn proof(self) -> KfdQueueLiveAqlSlotPreflightProof {
        let offline_template_header_invalid = (self.offline_template_header_low32 == 0) as u64;
        let packet_template_ready =
            (self.packet_template_header_low32 == self.expected_publish_low32
                && self.packet_template_kernel_object != 0
                && self.packet_template_kernarg_address != 0
                && (self.packet_template_kernarg_address & 0xf) == 0) as u64;
        let admission_token_ready = self.admission_token_ready as u64;
        let admission_validated = self.admission_validated as u64;
        let future_write_blocked =
            (self.admission_submit_enabled == 0 && self.admission_submit_allowed == 0) as u64;
        let no_ownership_transfer = (self.admission_no_live_mutation
            && self.queue_write_index_not_mutated
            && self.queue_read_index_not_mutated
            && self.batch_plan_no_fetch_add
            && self.batch_plan_no_doorbell) as u64;
        let first_slot_matches_reservation = self.first_slot_matches_reservation as u64;
        let reservation_ready = self.reservation_ready as u64;
        let ready = (admission_token_ready == 1
            && admission_validated == 1
            && offline_template_header_invalid == 1
            && packet_template_ready == 1
            && future_write_blocked == 1
            && no_ownership_transfer == 1
            && first_slot_matches_reservation == 1
            && reservation_ready == 1) as u64;

        KfdQueueLiveAqlSlotPreflightProof {
            offline_template_header_invalid,
            packet_template_ready,
            admission_token_ready,
            admission_validated,
            future_write_blocked,
            no_ownership_transfer,
            first_slot_matches_reservation,
            reservation_ready,
            ready,
            live_write_allowed: self.live_write_allowed,
        }
    }
}

impl KfdQueueLiveAqlSlotPreflightProof {
    pub fn validate_disabled_live_write(self) -> KfdQueueLiveAqlSlotPreflightValidation {
        let offline_template_header_invalid = self.offline_template_header_invalid == 1;
        let packet_template_ready = self.packet_template_ready == 1;
        let admission_token_ready = self.admission_token_ready == 1;
        let admission_validated = self.admission_validated == 1;
        let future_write_blocked = self.future_write_blocked == 1;
        let no_ownership_transfer = self.no_ownership_transfer == 1;
        let first_slot_matches_reservation = self.first_slot_matches_reservation == 1;
        let reservation_ready = self.reservation_ready == 1;
        let ready = self.ready == 1;
        let live_write_disabled = self.live_write_allowed == 0;
        let passed = offline_template_header_invalid
            && packet_template_ready
            && admission_token_ready
            && admission_validated
            && future_write_blocked
            && no_ownership_transfer
            && first_slot_matches_reservation
            && reservation_ready
            && ready
            && live_write_disabled;

        KfdQueueLiveAqlSlotPreflightValidation {
            offline_template_header_invalid,
            packet_template_ready,
            admission_token_ready,
            admission_validated,
            future_write_blocked,
            no_ownership_transfer,
            first_slot_matches_reservation,
            reservation_ready,
            ready,
            live_write_disabled,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlHeaderProbeInput {
    pub slot0_va: u64,
    pub slot1_va: u64,
    pub slot0_offset: u64,
    pub slot1_offset: u64,
    pub slot0_low32: u64,
    pub slot1_low32: u64,
    pub slot0_type: u64,
    pub slot1_type: u64,
    pub expected_publish_low32: u64,
    pub targets_match_batch_plan: bool,
    pub read_only_contract: bool,
    pub fetch_add_not_performed: bool,
    pub doorbell_not_written: bool,
    pub live_slot_not_written: bool,
    pub future_copy_blocked: bool,
    pub live_slot_preflight_ready: bool,
    pub live_slot_preflight_validated: bool,
    pub batch_ready: bool,
    pub reservation_ready: bool,
    pub live_write_allowed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlHeaderProbeProof {
    pub slot0_va: u64,
    pub slot1_va: u64,
    pub slot0_offset: u64,
    pub slot1_offset: u64,
    pub slot0_low32: u64,
    pub slot1_low32: u64,
    pub slot0_type: u64,
    pub slot1_type: u64,
    pub slot0_not_target_publish: u64,
    pub slot1_not_target_publish: u64,
    pub targets_match_batch_plan: u64,
    pub read_only_contract: u64,
    pub fetch_add_not_performed: u64,
    pub doorbell_not_written: u64,
    pub live_slot_not_written: u64,
    pub future_copy_blocked: u64,
    pub live_slot_preflight_ready: u64,
    pub live_slot_preflight_validated: u64,
    pub ready: u64,
    pub expected_publish_low32: u64,
    pub live_write_allowed: u64,
    pub no_mutation_contract: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlHeaderProbeValidation {
    pub slot0_low32_fits: bool,
    pub slot1_low32_fits: bool,
    pub slot0_type_matches: bool,
    pub slot1_type_matches: bool,
    pub live_slot_preflight_ready: bool,
    pub live_slot_preflight_validated: bool,
    pub no_mutation_contract: bool,
    pub live_write_disabled: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlHeaderProbeInput {
    pub fn proof(self) -> KfdQueueLiveAqlHeaderProbeProof {
        let slot0_not_target_publish = (self.slot0_low32 != self.expected_publish_low32) as u64;
        let slot1_not_target_publish = (self.slot1_low32 != self.expected_publish_low32) as u64;
        let read_only_contract = self.read_only_contract as u64;
        let fetch_add_not_performed = self.fetch_add_not_performed as u64;
        let doorbell_not_written = self.doorbell_not_written as u64;
        let live_slot_not_written = self.live_slot_not_written as u64;
        let future_copy_blocked = self.future_copy_blocked as u64;
        let live_slot_preflight_ready = self.live_slot_preflight_ready as u64;
        let live_slot_preflight_validated = self.live_slot_preflight_validated as u64;
        let no_mutation_contract = (self.read_only_contract
            && self.fetch_add_not_performed
            && self.doorbell_not_written
            && self.live_slot_not_written) as u64;
        let targets_match_batch_plan = self.targets_match_batch_plan as u64;
        let ready = (self.live_slot_preflight_ready
            && self.live_slot_preflight_validated
            && self.targets_match_batch_plan
            && self.batch_ready
            && self.reservation_ready
            && no_mutation_contract == 1
            && self.future_copy_blocked
            && self.live_write_allowed == 0) as u64;

        KfdQueueLiveAqlHeaderProbeProof {
            slot0_va: self.slot0_va,
            slot1_va: self.slot1_va,
            slot0_offset: self.slot0_offset,
            slot1_offset: self.slot1_offset,
            slot0_low32: self.slot0_low32,
            slot1_low32: self.slot1_low32,
            slot0_type: self.slot0_type,
            slot1_type: self.slot1_type,
            slot0_not_target_publish,
            slot1_not_target_publish,
            targets_match_batch_plan,
            read_only_contract,
            fetch_add_not_performed,
            doorbell_not_written,
            live_slot_not_written,
            future_copy_blocked,
            live_slot_preflight_ready,
            live_slot_preflight_validated,
            ready,
            expected_publish_low32: self.expected_publish_low32,
            live_write_allowed: self.live_write_allowed,
            no_mutation_contract,
        }
    }
}

impl KfdQueueLiveAqlHeaderProbeProof {
    pub fn validate_read_only_no_mutation(self) -> KfdQueueLiveAqlHeaderProbeValidation {
        let slot0_low32_fits = self.slot0_low32 <= u64::from(u32::MAX);
        let slot1_low32_fits = self.slot1_low32 <= u64::from(u32::MAX);
        let slot0_type_matches = self.slot0_type == (self.slot0_low32 & 0xff);
        let slot1_type_matches = self.slot1_type == (self.slot1_low32 & 0xff);
        let live_slot_preflight_ready = self.live_slot_preflight_ready == 1;
        let live_slot_preflight_validated = self.live_slot_preflight_validated == 1;
        let no_mutation_contract = self.no_mutation_contract == 1
            && self.read_only_contract == 1
            && self.fetch_add_not_performed == 1
            && self.doorbell_not_written == 1
            && self.live_slot_not_written == 1;
        let live_write_disabled = self.live_write_allowed == 0;
        let ready = self.ready == 1;
        let passed = slot0_low32_fits
            && slot1_low32_fits
            && slot0_type_matches
            && slot1_type_matches
            && live_slot_preflight_ready
            && live_slot_preflight_validated
            && no_mutation_contract
            && live_write_disabled
            && ready;

        KfdQueueLiveAqlHeaderProbeValidation {
            slot0_low32_fits,
            slot1_low32_fits,
            slot0_type_matches,
            slot1_type_matches,
            live_slot_preflight_ready,
            live_slot_preflight_validated,
            no_mutation_contract,
            live_write_disabled,
            ready,
            passed,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlCopyDecisionInput {
    pub slot0_header_low32: u64,
    pub slot1_header_low32: u64,
    pub publish_header_low32: u64,
    pub header_probe_ready: bool,
    pub header_probe_validated: bool,
    pub no_live_write_observed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlCopyDecisionProof {
    pub slot0_reason: u64,
    pub slot1_reason: u64,
    pub any_header_block: u64,
    pub requires_cleanup: u64,
    pub header_probe_ready: u64,
    pub header_probe_validated: u64,
    pub header_reset_allowed: u64,
    pub copy_allowed: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueLiveAqlCopyDecisionValidation {
    pub any_header_block_matches_reasons: bool,
    pub requires_cleanup_matches_header_block: bool,
    pub header_probe_ready: bool,
    pub header_probe_validated: bool,
    pub header_reset_disabled: bool,
    pub copy_disabled: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueLiveAqlCopyDecisionInput {
    pub fn proof(self) -> KfdQueueLiveAqlCopyDecisionProof {
        let slot0_reason = Self::slot_reason(self.slot0_header_low32, self.publish_header_low32);
        let slot1_reason = Self::slot_reason(self.slot1_header_low32, self.publish_header_low32);
        let any_header_block = (slot0_reason != 0 || slot1_reason != 0) as u64;
        let requires_cleanup = any_header_block;
        let header_probe_ready = self.header_probe_ready as u64;
        let header_probe_validated = self.header_probe_validated as u64;
        let header_reset_allowed = 0;
        let copy_allowed = 0;
        let ready = (self.header_probe_ready
            && self.header_probe_validated
            && header_reset_allowed == 0
            && copy_allowed == 0
            && self.no_live_write_observed) as u64;

        KfdQueueLiveAqlCopyDecisionProof {
            slot0_reason,
            slot1_reason,
            any_header_block,
            requires_cleanup,
            header_probe_ready,
            header_probe_validated,
            header_reset_allowed,
            copy_allowed,
            ready,
        }
    }

    fn slot_reason(slot_header_low32: u64, publish_header_low32: u64) -> u64 {
        if slot_header_low32 == publish_header_low32 {
            1
        } else if slot_header_low32 != 0 {
            2
        } else {
            0
        }
    }
}

impl KfdQueueLiveAqlCopyDecisionProof {
    pub fn validate_disabled_not_copied(self) -> KfdQueueLiveAqlCopyDecisionValidation {
        let any_header_block_matches_reasons =
            self.any_header_block == ((self.slot0_reason != 0 || self.slot1_reason != 0) as u64);
        let requires_cleanup_matches_header_block = self.requires_cleanup == self.any_header_block;
        let header_probe_ready = self.header_probe_ready == 1;
        let header_probe_validated = self.header_probe_validated == 1;
        let header_reset_disabled = self.header_reset_allowed == 0;
        let copy_disabled = self.copy_allowed == 0;
        let ready = self.ready == 1;
        let passed = any_header_block_matches_reasons
            && requires_cleanup_matches_header_block
            && header_probe_ready
            && header_probe_validated
            && header_reset_disabled
            && copy_disabled
            && ready;

        KfdQueueLiveAqlCopyDecisionValidation {
            any_header_block_matches_reasons,
            requires_cleanup_matches_header_block,
            header_probe_ready,
            header_probe_validated,
            header_reset_disabled,
            copy_disabled,
            ready,
            passed,
        }
    }
}

/// Inputs for the disabled live-AQL cleanup preflight decision.
///
/// The helper keeps logical packet IDs separate from physical ring slots. It
/// decides whether the queue read index has advanced beyond the packet ID that
/// previously occupied the same physical slot, but it does not enable any live
/// queue mutation. Consumption of an AQL packet is not kernel completion.
#[derive(Debug, Clone, Copy)]
pub struct KfdQueueCleanupPreflightInput {
    pub host_snapshot_read_index: u64,
    pub gpu_read_index: u64,
    pub gpu_reference_read_index: u64,
    pub ring_slots: u64,
    pub slot0_target_packet_id: u64,
    pub slot1_target_packet_id: u64,
    pub slot0_header_blocked: bool,
    pub slot1_header_blocked: bool,
    pub observed_block: bool,
    pub requires_cleanup: bool,
    pub copy_decision_ready: bool,
    pub copy_decision_validated: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueCleanupSlotDecision {
    pub target_packet_id: u64,
    pub blocked_packet_id: u64,
    pub blocked_id_known: bool,
    pub read_index_passed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueCleanupPreflightDecision {
    pub host_snapshot_read_index: u64,
    pub gpu_read_index: u64,
    pub gpu_read_index_matches_reference: bool,
    pub gpu_read_index_matches_host_snapshot: bool,
    pub gpu_read_index_not_behind_host_snapshot: bool,
    pub ring_slots: u64,
    pub slot0: KfdQueueCleanupSlotDecision,
    pub slot1: KfdQueueCleanupSlotDecision,
    pub observed_block: bool,
    pub requires_cleanup: bool,
    pub copy_decision_ready: bool,
    pub copy_decision_validated: bool,
    pub any_reset_eligible: bool,
    pub reset_allowed: bool,
    pub copy_allowed: bool,
    pub ready: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueCleanupPreflightProof {
    pub host_snapshot_read_index: u64,
    pub gpu_read_index: u64,
    pub gpu_read_index_matches_reference: u64,
    pub gpu_read_index_matches_host_snapshot: u64,
    pub gpu_read_index_not_behind_host_snapshot: u64,
    pub ring_slots: u64,
    pub slot0_target_packet_id: u64,
    pub slot1_target_packet_id: u64,
    pub slot0_blocked_packet_id: u64,
    pub slot1_blocked_packet_id: u64,
    pub slot0_blocked_id_known: u64,
    pub slot1_blocked_id_known: u64,
    pub slot0_read_index_passed: u64,
    pub slot1_read_index_passed: u64,
    pub observed_block: u64,
    pub requires_cleanup: u64,
    pub copy_decision_ready: u64,
    pub copy_decision_validated: u64,
    pub any_reset_eligible: u64,
    pub reset_allowed: u64,
    pub copy_allowed: u64,
    pub ready: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct KfdQueueCleanupPreflightValidation {
    pub gpu_read_index_matches_reference: bool,
    pub gpu_read_index_matches_host_snapshot: bool,
    pub gpu_read_index_not_behind_host_snapshot: bool,
    pub observed_block_matches_requires_cleanup: bool,
    pub copy_decision_ready: bool,
    pub copy_decision_validated: bool,
    pub reset_disabled: bool,
    pub copy_disabled: bool,
    pub ready: bool,
    pub passed: bool,
}

impl KfdQueueCleanupPreflightDecision {
    pub fn proof(self) -> KfdQueueCleanupPreflightProof {
        KfdQueueCleanupPreflightProof {
            host_snapshot_read_index: self.host_snapshot_read_index,
            gpu_read_index: self.gpu_read_index,
            gpu_read_index_matches_reference: self.gpu_read_index_matches_reference as u64,
            gpu_read_index_matches_host_snapshot: self.gpu_read_index_matches_host_snapshot as u64,
            gpu_read_index_not_behind_host_snapshot: self.gpu_read_index_not_behind_host_snapshot
                as u64,
            ring_slots: self.ring_slots,
            slot0_target_packet_id: self.slot0.target_packet_id,
            slot1_target_packet_id: self.slot1.target_packet_id,
            slot0_blocked_packet_id: self.slot0.blocked_packet_id,
            slot1_blocked_packet_id: self.slot1.blocked_packet_id,
            slot0_blocked_id_known: self.slot0.blocked_id_known as u64,
            slot1_blocked_id_known: self.slot1.blocked_id_known as u64,
            slot0_read_index_passed: self.slot0.read_index_passed as u64,
            slot1_read_index_passed: self.slot1.read_index_passed as u64,
            observed_block: self.observed_block as u64,
            requires_cleanup: self.requires_cleanup as u64,
            copy_decision_ready: self.copy_decision_ready as u64,
            copy_decision_validated: self.copy_decision_validated as u64,
            any_reset_eligible: self.any_reset_eligible as u64,
            reset_allowed: self.reset_allowed as u64,
            copy_allowed: self.copy_allowed as u64,
            ready: self.ready as u64,
        }
    }
}

impl KfdQueueCleanupPreflightProof {
    pub fn validate_disabled_observed_cleanup(self) -> KfdQueueCleanupPreflightValidation {
        let gpu_read_index_matches_reference = self.gpu_read_index_matches_reference == 1;
        let gpu_read_index_matches_host_snapshot = self.gpu_read_index_matches_host_snapshot == 1;
        let gpu_read_index_not_behind_host_snapshot =
            self.gpu_read_index_not_behind_host_snapshot == 1;
        let observed_block_matches_requires_cleanup = self.observed_block == self.requires_cleanup;
        let copy_decision_ready = self.copy_decision_ready == 1;
        let copy_decision_validated = self.copy_decision_validated == 1;
        let reset_disabled = self.reset_allowed == 0;
        let copy_disabled = self.copy_allowed == 0;
        let ready = self.ready == 1;
        let passed = gpu_read_index_matches_reference
            && gpu_read_index_matches_host_snapshot
            && gpu_read_index_not_behind_host_snapshot
            && observed_block_matches_requires_cleanup
            && copy_decision_ready
            && copy_decision_validated
            && reset_disabled
            && copy_disabled
            && ready;

        KfdQueueCleanupPreflightValidation {
            gpu_read_index_matches_reference,
            gpu_read_index_matches_host_snapshot,
            gpu_read_index_not_behind_host_snapshot,
            observed_block_matches_requires_cleanup,
            copy_decision_ready,
            copy_decision_validated,
            reset_disabled,
            copy_disabled,
            ready,
            passed,
        }
    }
}

impl KfdQueueCleanupPreflightInput {
    pub fn decide(self) -> KfdQueueCleanupPreflightDecision {
        let slot0 = KfdQueueCleanupSlotDecision::from_target(
            self.slot0_target_packet_id,
            self.ring_slots,
            self.slot0_header_blocked,
            self.gpu_read_index,
        );
        let slot1 = KfdQueueCleanupSlotDecision::from_target(
            self.slot1_target_packet_id,
            self.ring_slots,
            self.slot1_header_blocked,
            self.gpu_read_index,
        );
        let any_reset_eligible = slot0.read_index_passed || slot1.read_index_passed;
        let reset_allowed = false;
        let copy_allowed = false;
        let gpu_read_index_not_behind_host_snapshot =
            self.gpu_read_index >= self.host_snapshot_read_index;
        let ready = self.copy_decision_ready
            && self.copy_decision_validated
            && gpu_read_index_not_behind_host_snapshot
            && self.observed_block == self.requires_cleanup
            && !reset_allowed
            && !copy_allowed;

        KfdQueueCleanupPreflightDecision {
            host_snapshot_read_index: self.host_snapshot_read_index,
            gpu_read_index: self.gpu_read_index,
            gpu_read_index_matches_reference: self.gpu_read_index == self.gpu_reference_read_index,
            gpu_read_index_matches_host_snapshot: self.gpu_read_index
                == self.host_snapshot_read_index,
            gpu_read_index_not_behind_host_snapshot,
            ring_slots: self.ring_slots,
            slot0,
            slot1,
            observed_block: self.observed_block,
            requires_cleanup: self.requires_cleanup,
            copy_decision_ready: self.copy_decision_ready,
            copy_decision_validated: self.copy_decision_validated,
            any_reset_eligible,
            reset_allowed,
            copy_allowed,
            ready,
        }
    }
}

impl KfdQueueCleanupSlotDecision {
    fn from_target(
        target_packet_id: u64,
        ring_slots: u64,
        header_blocked: bool,
        gpu_read_index: u64,
    ) -> Self {
        let blocked_packet_id = if target_packet_id >= ring_slots {
            target_packet_id - ring_slots
        } else {
            u64::MAX
        };
        let blocked_id_known = header_blocked && blocked_packet_id != u64::MAX;
        let read_index_passed = blocked_id_known && gpu_read_index > blocked_packet_id;

        Self {
            target_packet_id,
            blocked_packet_id,
            blocked_id_known,
            read_index_passed,
        }
    }
}

impl KfdQueueSnapshot {
    pub fn batch_reservation_plan_from_indices(
        &self,
        base_packet_id: u64,
        read_index: u64,
        packet_count: u64,
    ) -> Result<KfdQueueBatchReservationPlan> {
        if packet_count == 0 {
            return Err(anyhow!(
                "mainarch AQL reservation plan: packet_count must be nonzero"
            ));
        }
        if self.packet_bytes != AQL_PACKET_BYTES {
            return Err(anyhow!(
                "mainarch AQL reservation plan: packet_bytes mismatch \
                (queue_id={}, gpu_id={}, expected={}, observed={})",
                self.queue_id,
                self.gpu_id,
                AQL_PACKET_BYTES,
                self.packet_bytes
            ));
        }
        if self.ring_slots == 0 || !self.ring_slots.is_power_of_two() {
            return Err(anyhow!(
                "mainarch AQL reservation plan: ring_slots must be nonzero power-of-two \
                (queue_id={}, gpu_id={}, ring_slots={})",
                self.queue_id,
                self.gpu_id,
                self.ring_slots
            ));
        }
        if packet_count > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL reservation plan: packet_count exceeds ring slots \
                (queue_id={}, gpu_id={}, packet_count={}, ring_slots={})",
                self.queue_id,
                self.gpu_id,
                packet_count,
                self.ring_slots
            ));
        }
        if read_index > base_packet_id {
            return Err(anyhow!(
                "mainarch AQL reservation plan: read index advanced beyond base packet id \
                (queue_id={}, gpu_id={}, base_packet_id={}, read_index={})",
                self.queue_id,
                self.gpu_id,
                base_packet_id,
                read_index
            ));
        }
        let inflight_packets = base_packet_id - read_index;
        let last_packet_id = base_packet_id
            .checked_add(packet_count - 1)
            .ok_or_else(|| {
                anyhow!(
                    "mainarch AQL reservation plan: last packet id overflow \
                    (queue_id={}, gpu_id={}, base_packet_id={}, packet_count={})",
                    self.queue_id,
                    self.gpu_id,
                    base_packet_id,
                    packet_count
                )
            })?;
        let desired_write_index = base_packet_id.checked_add(packet_count).ok_or_else(|| {
            anyhow!(
                "mainarch AQL reservation plan: desired write index overflow \
                (queue_id={}, gpu_id={}, base_packet_id={}, packet_count={})",
                self.queue_id,
                self.gpu_id,
                base_packet_id,
                packet_count
            )
        })?;
        let capacity_ok = inflight_packets
            .checked_add(packet_count)
            .map_or(false, |needed| needed <= self.ring_slots);
        let slot_mask = self.ring_slots - 1;
        let first_slot_index = base_packet_id & slot_mask;
        let last_slot_index = last_packet_id & slot_mask;
        let first_slot_offset = first_slot_index * self.packet_bytes;
        let last_slot_offset = last_slot_index * self.packet_bytes;
        let first_slot_va = self.ring_va + first_slot_offset;
        let last_slot_va = self.ring_va + last_slot_offset;
        let first_slot_formula_ok =
            first_slot_va == self.ring_va + ((base_packet_id & slot_mask) * self.packet_bytes);
        let last_slot_formula_ok =
            last_slot_va == self.ring_va + ((last_packet_id & slot_mask) * self.packet_bytes);
        Ok(KfdQueueBatchReservationPlan {
            base_packet_id,
            packet_count,
            last_packet_id,
            desired_write_index,
            read_index,
            inflight_packets,
            capacity_ok,
            first_slot_index,
            first_slot_offset,
            first_slot_va,
            last_slot_index,
            last_slot_offset,
            last_slot_va,
            slots_distinct: first_slot_va != last_slot_va,
            slots_aligned64: (first_slot_va & 0x3f) == 0 && (last_slot_va & 0x3f) == 0,
            first_slot_formula_ok,
            last_slot_formula_ok,
            doorbell_packet_id: last_packet_id,
            doorbell_matches_last_packet: last_packet_id + 1 == desired_write_index,
        })
    }
}

/// EOP (end-of-pipe) buffer size used by ROCr for gfx9-class devices.
const GFX9_EOP_BUFFER_SIZE: usize = 4096;

/// AQL packet size in bytes (`hsa_kernel_dispatch_packet_t`).
pub const AQL_PACKET_BYTES: u64 = 64;

/// Compute AQL ring slots for decode chains that intentionally keep multiple
/// token steps in flight before the host waits.
const COMPUTE_AQL_RING_SLOTS: usize = 256;

impl KfdQueue {
    pub fn snapshot(&self) -> KfdQueueSnapshot {
        let producer_index = unsafe { std::ptr::read_volatile(self.write_ptr_va as *const u64) };
        let consumer_index = unsafe { std::ptr::read_volatile(self.read_ptr_va as *const u64) };
        KfdQueueSnapshot {
            queue_id: self.queue_id,
            gpu_id: self.gpu_id,
            ring_va: self.ring_va,
            ring_slots: self.ring_slots,
            packet_bytes: AQL_PACKET_BYTES,
            write_ptr_va: self.write_ptr_va,
            read_ptr_va: self.read_ptr_va,
            doorbell_offset: self.doorbell_offset,
            host_write_index: self.write_index,
            producer_index,
            consumer_index,
        }
    }

    fn new(kfd: &Kfd, node_id: u32) -> Result<Self> {
        let debug = std::env::var_os("MAINARCH_KFD_QUEUE_DEBUG").is_some();
        let packet_guard = debug
            || std::env::var_os("MAINARCH_AQL_PACKET_GUARD").is_some()
            || std::env::var_os("MAINARCH_AQL_TRACE").is_some()
            || std::env::var_os("MAINARCH_AQL_SERIALIZE").is_some()
            || std::env::var_os("AMD_SERIALIZE_KERNEL").is_some();
        let page = page_size();
        let gpu_id = kfd.ensure_vm(node_id)?;
        let props = node_properties(node_id)?;
        let queue_sizes = compute_queue_sizes(&props, page as u64)?;
        let ring_bytes = (COMPUTE_AQL_RING_SLOTS * AQL_PACKET_BYTES as usize)
            .max(mainarch_sys::KFD_IOC_QUEUE_MIN_RING_SIZE as usize)
            .max(page)
            .next_power_of_two();
        let ring_size = u32::try_from(ring_bytes).context("queue ring size exceeds u32")?;
        if debug {
            eprintln!(
                "mainarch: queue-create node={node_id} gpu_id={gpu_id} ring_size={ring_size} \
                page={page} ctx_save_restore_size={} ctl_stack_size={} total_cwsr_buffer_size={}",
                queue_sizes.ctx_save_restore_size,
                queue_sizes.ctl_stack_size,
                queue_sizes.total_cwsr_buffer_size,
            );
        }

        let allow_userptr_aql_queue =
            std::env::var_os("MAINARCH_ALLOW_USERPTR_AQL_QUEUE").is_some();
        let force_aql_queue_mem = std::env::var_os("MAINARCH_FORCE_AQL_QUEUE_MEM").is_some();
        let userptr_rwx = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
        let userptr_rw = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
        let gtt_queue = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_GTT
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        // ROCr uses the double-map AQL queue workaround only on old gfx7/gfx8
        // agents. On gfx950/MI355X, forcing KFD AQL_QUEUE_MEM makes the second
        // raw KFD queue fail to map with EINVAL; plain GTT is the production
        // host-dispatched ring path.
        let gtt_ring = gtt_queue;
        let gtt_ring_aql = gtt_queue | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM;
        let vram_eop = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        let ring_candidates: Vec<u32> = if force_aql_queue_mem && allow_userptr_aql_queue {
            vec![gtt_ring_aql, userptr_rwx]
        } else if force_aql_queue_mem {
            vec![gtt_ring_aql]
        } else if allow_userptr_aql_queue {
            vec![gtt_ring, gtt_ring_aql, userptr_rwx]
        } else {
            vec![gtt_ring, gtt_ring_aql]
        };
        let queue_control_candidates: Vec<u32> = if allow_userptr_aql_queue {
            vec![gtt_queue, userptr_rw]
        } else {
            vec![gtt_queue]
        };
        let cwsr_candidates: Vec<u32> = if allow_userptr_aql_queue {
            vec![gtt_queue, userptr_rwx]
        } else {
            vec![gtt_queue]
        };
        if debug && allow_userptr_aql_queue {
            eprintln!(
                "mainarch: MAINARCH_ALLOW_USERPTR_AQL_QUEUE enabled; USERPTR queue-control fallback is debug-only"
            );
        }
        if debug && force_aql_queue_mem {
            eprintln!(
                "mainarch: MAINARCH_FORCE_AQL_QUEUE_MEM enabled; forcing KFD AQL_QUEUE_MEM ring allocation"
            );
        }

        let try_buffer = |label: &str,
                          len: usize,
                          candidates: &[u32]|
         -> Result<KfdAllocatedBuffer> {
            let mut last_err = None;
            for &flags in candidates {
                match KfdAllocatedBuffer::new(kfd, gpu_id, len, flags) {
                    Ok(buf) => {
                        if debug {
                            eprintln!(
                                "mainarch: queue-alloc {label} ok flags=0x{flags:08x} len={len} va=0x{:x}",
                                buf.ptr()
                            );
                        }
                        return Ok(buf);
                    }
                    Err(e) => {
                        if debug {
                            eprintln!(
                                "mainarch: queue-alloc {label} failed flags=0x{flags:08x} len={len}: {e:#}"
                            );
                        }
                        last_err = Some(e);
                    }
                }
            }
            Err(last_err
                .unwrap_or_else(|| anyhow!("no allocation flag candidates"))
                .context(format!(
                    "queue-alloc {label} failed for all candidate flags"
                )))
        };

        let ring = try_buffer("ring", ring_bytes, &ring_candidates).with_context(|| {
            if allow_userptr_aql_queue {
                "queue ring allocation failed after trying the debug USERPTR fallback; both GTT AQL_QUEUE_MEM and USERPTR candidates failed"
            } else if force_aql_queue_mem {
                "queue ring allocation failed with forced GTT AQL_QUEUE_MEM"
            } else {
                "queue ring allocation failed with plain GTT and GTT AQL_QUEUE_MEM; MAINARCH_FORCE_AQL_QUEUE_MEM=1 restores the old forced AQL_QUEUE_MEM behavior, and MAINARCH_ALLOW_USERPTR_AQL_QUEUE=1 enables a debug-only USERPTR ring fallback for localization"
            }
        })?;
        let write_ptr = try_buffer("write_ptr", page, &queue_control_candidates)?;
        let read_ptr = try_buffer("read_ptr", page, &queue_control_candidates)?;
        let eop = try_buffer("eop", GFX9_EOP_BUFFER_SIZE, &[vram_eop, gtt_queue]).ok();
        let cwsr_bytes = usize::try_from(queue_sizes.total_cwsr_buffer_size)
            .context("queue cwsr allocation size exceeds usize")?;
        let cwsr = try_buffer("cwsr", cwsr_bytes, &cwsr_candidates).ok();

        unsafe {
            std::ptr::write_bytes(ring.host_ptr(), 0, ring.len());
            std::ptr::write_volatile(write_ptr.host_ptr() as *mut u64, 0);
            std::ptr::write_volatile(read_ptr.host_ptr() as *mut u64, 0);
        }

        let mut args = mainarch_sys::CreateQueueArgsCompat {
            ring_base_address: ring.ptr(),
            write_pointer_address: write_ptr.ptr(),
            read_pointer_address: read_ptr.ptr(),
            doorbell_offset: 0,
            ring_size,
            gpu_id,
            queue_type: mainarch_sys::KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: mainarch_sys::KFD_IOC_QUEUE_MAX_PERCENTAGE,
            queue_priority: 7,
            queue_id: 0,
            eop_buffer_address: eop.as_ref().map_or(0, KfdAllocatedBuffer::ptr),
            eop_buffer_size: if eop.is_some() {
                GFX9_EOP_BUFFER_SIZE as u64
            } else {
                0
            },
            ctx_save_restore_address: cwsr.as_ref().map_or(0, KfdAllocatedBuffer::ptr),
            ctx_save_restore_size: if cwsr.is_some() {
                queue_sizes.ctx_save_restore_size
            } else {
                0
            },
            ctl_stack_size: if cwsr.is_some() {
                queue_sizes.ctl_stack_size
            } else {
                0
            },
        };

        let fd = kfd.fd.as_raw_fd();
        // Kernel 6.8 ships the 88-byte layout (no sdma_engine_id) as the
        // native struct, so the "compat" encoding is primary here; the newer
        // 96-byte layout is the fallback for newer kernels.
        let create = |a: &mut mainarch_sys::CreateQueueArgsCompat| -> Result<()> {
            let compat_res = unsafe { mainarch_sys::ioctl_create_queue_compat(fd, a) };
            match compat_res {
                Ok(()) => Ok(()),
                Err(compat_err) => {
                    let mut new_args = mainarch_sys::CreateQueueArgs {
                        ring_base_address: a.ring_base_address,
                        write_pointer_address: a.write_pointer_address,
                        read_pointer_address: a.read_pointer_address,
                        doorbell_offset: a.doorbell_offset,
                        ring_size: a.ring_size,
                        gpu_id: a.gpu_id,
                        queue_type: a.queue_type,
                        queue_percentage: a.queue_percentage,
                        queue_priority: a.queue_priority,
                        queue_id: a.queue_id,
                        eop_buffer_address: a.eop_buffer_address,
                        eop_buffer_size: a.eop_buffer_size,
                        ctx_save_restore_address: a.ctx_save_restore_address,
                        ctx_save_restore_size: a.ctx_save_restore_size,
                        ctl_stack_size: a.ctl_stack_size,
                        sdma_engine_id: 0,
                        pad: 0,
                    };
                    match unsafe { mainarch_sys::ioctl_create_queue(fd, &mut new_args) } {
                        Ok(()) => {
                            a.doorbell_offset = new_args.doorbell_offset;
                            a.queue_id = new_args.queue_id;
                            a.write_pointer_address = new_args.write_pointer_address;
                            a.read_pointer_address = new_args.read_pointer_address;
                            Ok(())
                        }
                        Err(new_err) => Err(anyhow!(
                            "AMDKFD_IOC_CREATE_QUEUE failed (compat: {compat_err}, new: {new_err})"
                        )),
                    }
                }
            }
        };

        let mut created = create(&mut args);
        if created.is_err() && cwsr.is_some() {
            if debug {
                eprintln!(
                    "mainarch: queue-create with cwsr failed ({created:?}), retrying without"
                );
            }
            args.ctx_save_restore_address = 0;
            args.ctx_save_restore_size = 0;
            args.ctl_stack_size = 0;
            created = create(&mut args);
        }
        created?;

        if debug {
            eprintln!(
                "mainarch: queue-create ok queue_id={} doorbell_offset=0x{:x}",
                args.queue_id, args.doorbell_offset
            );
        }

        // Map the doorbell aperture. The KFD-returned offset encodes type +
        // gpu_id in the high bits; this queue's doorbell sits at the in-page
        // byte offset in the low bits (0 for the first queue). The gfx9 KFD
        // doorbell handler requires the VMA length to equal the device's
        // doorbell "process slice" exactly, so we try the known slice sizes
        // (page, 8 KiB) until one maps.
        let (doorbell, db_map_base, db_map_len) = {
            let mut found = None;
            for &slice in &[page, 0x2000usize, 0x4000usize] {
                let page_off = (args.doorbell_offset as usize) & (slice - 1);
                let base = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        slice,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        kfd.fd.as_raw_fd(),
                        (args.doorbell_offset & !((slice as u64) - 1)) as libc::off_t,
                    )
                };
                if base != libc::MAP_FAILED {
                    if debug {
                        eprintln!(
                            "mainarch: doorbell mmap ok slice=0x{slice:x} in_page_off=0x{page_off:x}"
                        );
                    }
                    found = Some((
                        unsafe { (base as *mut u8).add(page_off) as *mut u64 },
                        base,
                        slice,
                    ));
                    break;
                } else if debug {
                    eprintln!(
                        "mainarch: doorbell mmap slice=0x{slice:x} failed: {}",
                        std::io::Error::last_os_error()
                    );
                }
            }
            found.ok_or_else(|| {
                anyhow!(
                    "mmap doorbell aperture failed for all slice sizes: {}",
                    std::io::Error::last_os_error()
                )
            })?
        };

        Ok(Self {
            kfd_fd: kfd.fd.as_raw_fd(),
            queue_id: args.queue_id,
            gpu_id,
            doorbell_offset: args.doorbell_offset,
            ring_va: ring.ptr(),
            ring_slots: ring_bytes as u64 / AQL_PACKET_BYTES,
            write_ptr_va: write_ptr.ptr(),
            read_ptr_va: read_ptr.ptr(),
            doorbell,
            doorbell_map_base: db_map_base,
            doorbell_map_len: db_map_len,
            write_index: 0,
            packet_guard,
            _ring: ring,
            _write_ptr: write_ptr,
            _read_ptr: read_ptr,
            _eop: eop,
            _ctx_save_restore: cwsr,
        })
    }

    pub fn queue_id(&self) -> u32 {
        self.queue_id
    }

    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    pub fn doorbell_offset(&self) -> u64 {
        self.doorbell_offset
    }

    /// CP-reported consumed read index (for diagnostics).
    pub fn read_index(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.read_ptr_va as *const u64) }
    }

    /// Producer write index.
    pub fn write_index(&self) -> u64 {
        self.write_index
    }

    fn store_write_index_release(&self, next: u64) {
        // The CP observes this mailbox to discover newly published AQL
        // packets. Make the producer index itself a release operation rather
        // than relying on a post-store fence after a volatile write; the packet
        // header/body stores must become visible before the fresh producer
        // index can be consumed.
        let write_ptr = unsafe { &*(self.write_ptr_va as *const std::sync::atomic::AtomicU64) };
        write_ptr.store(next, std::sync::atomic::Ordering::Release);
    }

    fn ring_doorbell_release(&self, packet_idx: u64) {
        // The doorbell is MMIO and stays volatile, but keep an explicit release
        // edge immediately before the kick so packet/header/mailbox publication
        // cannot be reordered after the CP notification.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { std::ptr::write_volatile(self.doorbell, packet_idx) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }

    /// Enqueue one kernel-dispatch AQL packet and ring the doorbell.
    ///
    /// Caller supplies the loaded `kernel_object` VA, a kernarg buffer VA whose
    /// contents are already populated, and the grid/workgroup geometry. The
    /// packet uses system-scope acquire/release fences so a GPU write is
    /// visible to the host once the dispatch retires (which the caller observes
    /// by polling memory or, later, a completion signal).
    ///
    /// # Safety
    /// `kernel_object` and `kernarg_va` must reference live GPU-mapped memory
    /// that outlives the dispatch.
    pub unsafe fn dispatch_kernel(&mut self, d: &AqlDispatch) -> Result<()> {
        validate_aql_dispatch(d, self.gpu_id)?;
        let debug = std::env::var_os("MAINARCH_KFD_QUEUE_DEBUG").is_some();
        let idx = self.write_index;
        let observed_wptr = std::ptr::read_volatile(self.write_ptr_va as *const u64);
        if observed_wptr != idx {
            return Err(anyhow!(
                "mainarch AQL guard: queue producer mailbox mismatch before dispatch \
                (queue_id={}, gpu_id={}, expected_wptr={idx}, observed_wptr={observed_wptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let rptr = std::ptr::read_volatile(self.read_ptr_va as *const u64);
        if rptr > idx {
            return Err(anyhow!(
                "mainarch AQL guard: queue consumer read pointer advanced beyond producer \
                before dispatch (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let in_flight = idx - rptr;
        if in_flight >= self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL guard: queue ring is full before dispatch \
                (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, in_flight={in_flight}, \
                slots={}, ring_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_slots,
                self.ring_va,
            ));
        }
        let slot = (idx % self.ring_slots) as usize;
        let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize)
            as *mut AqlKernelDispatchPacket;

        // header (type KERNEL_DISPATCH=2, barrier, system acquire+release).
        let header: u16 = MAINARCH_AQL_KERNEL_HEADER;
        let setup: u16 = d.dims as u16;

        let body = AqlKernelDispatchPacket {
            header: 0, // written last
            setup,
            workgroup_size_x: d.wg_x,
            workgroup_size_y: d.wg_y,
            workgroup_size_z: d.wg_z,
            reserved0: 0,
            grid_size_x: d.grid_x,
            grid_size_y: d.grid_y,
            grid_size_z: d.grid_z,
            private_segment_size: d.private_segment_size,
            group_segment_size: d.group_segment_size,
            kernel_object: d.kernel_object,
            kernarg_address: d.kernarg_va,
            reserved2: 0,
            completion_signal: d.completion_signal,
        };
        if self.packet_guard {
            validate_aql_packet_semantics(idx, slot, header, &body, d.kernarg_size)?;
        }
        // Write the body first, then publish the header with a release fence.
        std::ptr::write_volatile(pkt, body);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
            validate_written_aql_body(idx, slot, &body, &observed)?;
        }
        let header_atomic =
            &*(std::ptr::addr_of!((*pkt).header) as *const std::sync::atomic::AtomicU16);
        header_atomic.store(header, std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            let observed_header = header_atomic.load(std::sync::atomic::Ordering::Acquire);
            if observed_header != header {
                return Err(anyhow!(
                    "mainarch AQL guard: packet header readback mismatch before doorbell (idx={idx}, slot={slot}, expected=0x{header:04x}, observed=0x{observed_header:04x})"
                ));
            }
        }
        if aql_cpc_trace_enabled() {
            let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
            trace_aql_cpc_packet(
                "dispatch",
                self.queue_id,
                self.gpu_id,
                idx,
                slot,
                self.ring_va,
                self.ring_slots,
                self.write_ptr_va,
                self.read_ptr_va,
                self.doorbell_offset,
                observed_wptr,
                rptr,
                &observed,
                d.kernarg_size,
            );
        }
        if aql_cpc_snapshot_enabled() {
            let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
            append_aql_cpc_snapshot(
                "dispatch",
                self.queue_id,
                self.gpu_id,
                idx,
                slot,
                self.ring_va,
                self.ring_slots,
                self.write_ptr_va,
                self.read_ptr_va,
                self.doorbell_offset,
                observed_wptr,
                rptr,
                &observed,
                d.kernarg_size,
            )?;
        }
        // CPC faults are often launch/control-plane faults. Keep packet header
        // publication strictly before the producer mailbox and doorbell update;
        // otherwise the command processor can observe a fresh write pointer and
        // fetch a not-yet-visible packet body/header on weakly ordered paths.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        // Advance the producer write index, then kick the CP. The gfx9 AQL
        // doorbell is rung with the 0-based index of the packet just written
        // (matching ROCr) — NOT the packet count. Ringing with the count only
        // happens to work for the first packet (the CP scans up to the WPTR
        // mailbox); every subsequent dispatch needs the exact index or the CP
        // never advances past it.
        let next = idx + 1;
        self.store_write_index_release(next);
        if self.packet_guard {
            let observed_next = std::ptr::read_volatile(self.write_ptr_va as *const u64);
            if observed_next != next {
                return Err(anyhow!(
                    "mainarch AQL guard: producer mailbox readback mismatch before doorbell \
                    (queue_id={}, gpu_id={}, idx={idx}, expected_wptr={next}, \
                    observed_wptr={observed_next}, ring_va=0x{:016x}, write_ptr_va=0x{:016x})",
                    self.queue_id,
                    self.gpu_id,
                    self.ring_va,
                    self.write_ptr_va,
                ));
            }
        }
        self.ring_doorbell_release(idx);
        self.write_index = next;

        if debug {
            eprintln!(
                "mainarch: dispatch idx={idx} slot={slot} rptr={rptr} kobj=0x{:x} kernarg=0x{:x} \
                grid=({},{},{}) wg=({},{},{}) doorbell<-{idx}",
                d.kernel_object, d.kernarg_va, d.grid_x, d.grid_y, d.grid_z, d.wg_x, d.wg_y, d.wg_z,
            );
        }
        Ok(())
    }

    /// Enqueue one prebuilt 64-byte kernel-dispatch packet and ring the
    /// doorbell. This is the live-replay boundary: callers provide exact packet
    /// bytes, while this method owns queue slot math, body-first/header-last
    /// publication, producer mailbox update, and doorbell write.
    ///
    /// # Safety
    /// Packet contents must reference live GPU-mapped memory that outlives the
    /// dispatch. This bypasses `AqlDispatch` packet construction but still
    /// validates packet semantics and registered pointer spans before publish.
    pub unsafe fn dispatch_kernel_packet_bytes_for_replay(
        &mut self,
        packet: &[u8; AQL_PACKET_BYTES as usize],
        kernarg_size: u32,
    ) -> Result<()> {
        let observed_packet =
            std::ptr::read_unaligned(packet.as_ptr().cast::<AqlKernelDispatchPacket>());
        let header = observed_packet.header;
        let body_for_guard = AqlKernelDispatchPacket {
            header: 0,
            ..observed_packet
        };
        validate_aql_packet_semantics(self.write_index, 0, header, &body_for_guard, kernarg_size)?;
        let dispatch = AqlDispatch {
            kernel_object: observed_packet.kernel_object,
            kernarg_va: observed_packet.kernarg_address,
            kernarg_size,
            dims: (observed_packet.setup & 0xff) as u8,
            grid_x: observed_packet.grid_size_x,
            grid_y: observed_packet.grid_size_y,
            grid_z: observed_packet.grid_size_z,
            wg_x: observed_packet.workgroup_size_x,
            wg_y: observed_packet.workgroup_size_y,
            wg_z: observed_packet.workgroup_size_z,
            private_segment_size: observed_packet.private_segment_size,
            group_segment_size: observed_packet.group_segment_size,
            completion_signal: observed_packet.completion_signal,
        };
        validate_aql_dispatch(&dispatch, self.gpu_id)?;

        let idx = self.write_index;
        let observed_wptr = std::ptr::read_volatile(self.write_ptr_va as *const u64);
        if observed_wptr != idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue producer mailbox mismatch before replay dispatch \
                (queue_id={}, gpu_id={}, expected_wptr={idx}, observed_wptr={observed_wptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let rptr = std::ptr::read_volatile(self.read_ptr_va as *const u64);
        if rptr > idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue consumer read pointer advanced beyond producer \
                before replay dispatch (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let in_flight = idx - rptr;
        if in_flight >= self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue ring is full before replay dispatch \
                (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, in_flight={in_flight}, \
                slots={}, ring_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_slots,
                self.ring_va,
            ));
        }

        let slot = (idx % self.ring_slots) as usize;
        let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
        std::ptr::write_bytes(pkt, 0, AQL_PACKET_BYTES as usize);
        // Publish words 1..15 first; header stays zero until the release store
        // below. The setup word is part of the body at bytes 2..4.
        std::ptr::copy_nonoverlapping(
            packet.as_ptr().add(2),
            pkt.add(2),
            AQL_PACKET_BYTES as usize - 2,
        );
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
            validate_written_aql_body(idx, slot, &body_for_guard, &observed)?;
        }
        let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
        let header_atomic =
            &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
        header_atomic.store(header, std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            let observed_header = header_atomic.load(std::sync::atomic::Ordering::Acquire);
            if observed_header != header {
                return Err(anyhow!(
                    "mainarch AQL replay guard: packet header readback mismatch before doorbell (idx={idx}, slot={slot}, expected=0x{header:04x}, observed=0x{observed_header:04x})"
                ));
            }
        }
        if aql_cpc_trace_enabled() {
            let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
            trace_aql_cpc_packet(
                "replay-dispatch",
                self.queue_id,
                self.gpu_id,
                idx,
                slot,
                self.ring_va,
                self.ring_slots,
                self.write_ptr_va,
                self.read_ptr_va,
                self.doorbell_offset,
                observed_wptr,
                rptr,
                &observed,
                kernarg_size,
            );
        }
        if aql_cpc_snapshot_enabled() {
            let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
            append_aql_cpc_snapshot(
                "replay-dispatch",
                self.queue_id,
                self.gpu_id,
                idx,
                slot,
                self.ring_va,
                self.ring_slots,
                self.write_ptr_va,
                self.read_ptr_va,
                self.doorbell_offset,
                observed_wptr,
                rptr,
                &observed,
                kernarg_size,
            )?;
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let next = idx + 1;
        self.store_write_index_release(next);
        if self.packet_guard {
            let observed_next = std::ptr::read_volatile(self.write_ptr_va as *const u64);
            if observed_next != next {
                return Err(anyhow!(
                    "mainarch AQL replay guard: producer mailbox readback mismatch before doorbell \
                    (queue_id={}, gpu_id={}, idx={idx}, expected_wptr={next}, \
                    observed_wptr={observed_next}, ring_va=0x{:016x}, write_ptr_va=0x{:016x})",
                    self.queue_id,
                    self.gpu_id,
                    self.ring_va,
                    self.write_ptr_va,
                ));
            }
        }
        self.ring_doorbell_release(idx);
        self.write_index = next;
        Ok(())
    }

    /// Enqueue multiple prebuilt kernel-dispatch packets and ring the doorbell
    /// once with the last packet id. This is the raw AQL equivalent of graph
    /// replay for one queue-ordered chain.
    ///
    /// # Safety
    /// Every packet must reference live GPU-mapped memory that outlives the
    /// dispatch chain. The final packet should normally carry the completion
    /// signal the caller waits on.
    pub unsafe fn dispatch_kernel_packet_chain_for_replay(
        &mut self,
        packets: &[(&[u8; AQL_PACKET_BYTES as usize], u32)],
    ) -> Result<()> {
        if packets.is_empty() {
            return Err(anyhow!("mainarch AQL replay guard: empty packet chain"));
        }
        if packets.len() as u64 > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: packet chain length {} exceeds ring slots {}",
                packets.len(),
                self.ring_slots
            ));
        }

        let idx = self.write_index;
        let observed_wptr = std::ptr::read_volatile(self.write_ptr_va as *const u64);
        if observed_wptr != idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue producer mailbox mismatch before replay chain \
                (queue_id={}, gpu_id={}, expected_wptr={idx}, observed_wptr={observed_wptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let rptr = std::ptr::read_volatile(self.read_ptr_va as *const u64);
        if rptr > idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue consumer read pointer advanced beyond producer \
                before replay chain (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let in_flight = idx - rptr;
        if in_flight + packets.len() as u64 > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue ring lacks capacity before replay chain \
                (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, in_flight={in_flight}, \
                chain_len={}, slots={}, ring_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                packets.len(),
                self.ring_slots,
                self.ring_va,
            ));
        }

        let mut parsed = Vec::with_capacity(packets.len());
        for (chain_pos, (packet, kernarg_size)) in packets.iter().enumerate() {
            let observed_packet =
                std::ptr::read_unaligned(packet.as_ptr().cast::<AqlKernelDispatchPacket>());
            let header = observed_packet.header;
            let body_for_guard = AqlKernelDispatchPacket {
                header: 0,
                ..observed_packet
            };
            let slot = ((idx + chain_pos as u64) % self.ring_slots) as usize;
            validate_aql_packet_semantics(
                idx + chain_pos as u64,
                slot,
                header,
                &body_for_guard,
                *kernarg_size,
            )?;
            let dispatch = AqlDispatch {
                kernel_object: observed_packet.kernel_object,
                kernarg_va: observed_packet.kernarg_address,
                kernarg_size: *kernarg_size,
                dims: (observed_packet.setup & 0xff) as u8,
                grid_x: observed_packet.grid_size_x,
                grid_y: observed_packet.grid_size_y,
                grid_z: observed_packet.grid_size_z,
                wg_x: observed_packet.workgroup_size_x,
                wg_y: observed_packet.workgroup_size_y,
                wg_z: observed_packet.workgroup_size_z,
                private_segment_size: observed_packet.private_segment_size,
                group_segment_size: observed_packet.group_segment_size,
                completion_signal: observed_packet.completion_signal,
            };
            validate_aql_dispatch(&dispatch, self.gpu_id)?;
            parsed.push((header, body_for_guard));
        }

        for (chain_pos, (packet, _)) in packets.iter().enumerate() {
            let slot = ((idx + chain_pos as u64) % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            std::ptr::write_bytes(pkt, 0, AQL_PACKET_BYTES as usize);
            std::ptr::copy_nonoverlapping(
                packet.as_ptr().add(2),
                pkt.add(2),
                AQL_PACKET_BYTES as usize - 2,
            );
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            for (chain_pos, (_, body_for_guard)) in parsed.iter().enumerate() {
                let slot = ((idx + chain_pos as u64) % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                validate_written_aql_body(idx + chain_pos as u64, slot, body_for_guard, &observed)?;
            }
        }

        // Publish non-first headers first, then packet 0 last. If the packet
        // processor ever observes packet 0's valid header early, the rest of
        // the chain is already visible.
        for chain_pos in 1..packets.len() {
            let slot = ((idx + chain_pos as u64) % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(parsed[chain_pos].0, std::sync::atomic::Ordering::Release);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        {
            let slot = (idx % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(parsed[0].0, std::sync::atomic::Ordering::Release);
        }
        if self.packet_guard {
            for (chain_pos, (header, _)) in parsed.iter().enumerate() {
                let slot = ((idx + chain_pos as u64) % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
                let header_atomic = &*(std::ptr::addr_of!((*pkt_struct).header)
                    as *const std::sync::atomic::AtomicU16);
                let observed_header = header_atomic.load(std::sync::atomic::Ordering::Acquire);
                if observed_header != *header {
                    return Err(anyhow!(
                        "mainarch AQL replay guard: chain packet header readback mismatch before doorbell (idx={}, slot={}, expected=0x{:04x}, observed=0x{observed_header:04x})",
                        idx + chain_pos as u64,
                        slot,
                        header
                    ));
                }
            }
        }
        if aql_cpc_snapshot_enabled() {
            for (chain_pos, (_, kernarg_size)) in packets.iter().enumerate() {
                let packet_idx = idx + chain_pos as u64;
                let slot = (packet_idx % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                append_aql_cpc_snapshot(
                    "replay-chain",
                    self.queue_id,
                    self.gpu_id,
                    packet_idx,
                    slot,
                    self.ring_va,
                    self.ring_slots,
                    self.write_ptr_va,
                    self.read_ptr_va,
                    self.doorbell_offset,
                    observed_wptr,
                    rptr,
                    &observed,
                    *kernarg_size,
                )?;
            }
        }
        if aql_cpc_trace_enabled() {
            for (chain_pos, (_, kernarg_size)) in packets.iter().enumerate() {
                let packet_idx = idx + chain_pos as u64;
                let slot = (packet_idx % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                trace_aql_cpc_packet(
                    "replay-chain",
                    self.queue_id,
                    self.gpu_id,
                    packet_idx,
                    slot,
                    self.ring_va,
                    self.ring_slots,
                    self.write_ptr_va,
                    self.read_ptr_va,
                    self.doorbell_offset,
                    observed_wptr,
                    rptr,
                    &observed,
                    *kernarg_size,
                );
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let next = idx + packets.len() as u64;
        self.store_write_index_release(next);
        if self.packet_guard {
            let observed_next = std::ptr::read_volatile(self.write_ptr_va as *const u64);
            if observed_next != next {
                return Err(anyhow!(
                    "mainarch AQL replay guard: producer mailbox readback mismatch before replay chain doorbell \
                    (queue_id={}, gpu_id={}, idx={idx}, expected_wptr={next}, \
                    observed_wptr={observed_next}, ring_va=0x{:016x}, write_ptr_va=0x{:016x})",
                    self.queue_id,
                    self.gpu_id,
                    self.ring_va,
                    self.write_ptr_va,
                ));
            }
        }
        let doorbell_idx = next - 1;
        self.ring_doorbell_release(doorbell_idx);
        self.write_index = next;
        Ok(())
    }

    /// Enqueue an already-validated prebuilt kernel-dispatch packet chain and
    /// ring the doorbell once with the last packet id. This is the hot replay
    /// publisher for chains whose packets were built and validated during the
    /// prepare phase.
    ///
    /// This deliberately skips per-dispatch semantic parsing, VA validation,
    /// packet body readback, and CPC trace/snapshot generation. Use
    /// `dispatch_kernel_packet_chain_for_replay` when debugging those guards.
    ///
    /// # Safety
    /// Every packet must be a valid kernel-dispatch packet, must reference live
    /// GPU-mapped memory that outlives the dispatch chain, and the final packet
    /// should normally carry the completion signal the caller waits on.
    pub unsafe fn dispatch_trusted_kernel_packet_chain_for_replay(
        &mut self,
        packets: &[[u8; AQL_PACKET_BYTES as usize]],
    ) -> Result<()> {
        if packets.is_empty() {
            return Err(anyhow!(
                "mainarch AQL trusted replay guard: empty packet chain"
            ));
        }
        if packets.len() as u64 > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL trusted replay guard: packet chain length {} exceeds ring slots {}",
                packets.len(),
                self.ring_slots
            ));
        }

        let idx = self.write_index;
        let observed_wptr = std::ptr::read_volatile(self.write_ptr_va as *const u64);
        if observed_wptr != idx {
            return Err(anyhow!(
                "mainarch AQL trusted replay guard: queue producer mailbox mismatch before replay chain \
                (queue_id={}, gpu_id={}, expected_wptr={idx}, observed_wptr={observed_wptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let rptr = std::ptr::read_volatile(self.read_ptr_va as *const u64);
        if rptr > idx {
            return Err(anyhow!(
                "mainarch AQL trusted replay guard: queue consumer read pointer advanced beyond producer \
                before replay chain (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let packet_count = packets.len() as u64;
        let reservation_plan = KfdQueueSnapshot {
            queue_id: self.queue_id,
            gpu_id: self.gpu_id,
            ring_va: self.ring_va,
            ring_slots: self.ring_slots,
            packet_bytes: AQL_PACKET_BYTES,
            write_ptr_va: self.write_ptr_va,
            read_ptr_va: self.read_ptr_va,
            doorbell_offset: self.doorbell_offset,
            host_write_index: self.write_index,
            producer_index: observed_wptr,
            consumer_index: rptr,
        }
        .batch_reservation_plan_from_indices(idx, rptr, packet_count)?;
        if !reservation_plan.capacity_ok {
            return Err(anyhow!(
                "mainarch AQL trusted replay guard: queue ring lacks capacity before replay chain \
                (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, in_flight={}, \
                chain_len={}, slots={}, ring_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                reservation_plan.inflight_packets,
                packet_count,
                self.ring_slots,
                self.ring_va,
            ));
        }

        for (chain_pos, packet) in packets.iter().enumerate() {
            let packet_idx = reservation_plan.base_packet_id + chain_pos as u64;
            let slot = (packet_idx & (self.ring_slots - 1)) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            std::ptr::write_bytes(pkt, 0, AQL_PACKET_BYTES as usize);
            std::ptr::copy_nonoverlapping(
                packet.as_ptr().add(2),
                pkt.add(2),
                AQL_PACKET_BYTES as usize - 2,
            );
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

        // Publish non-first headers first, then packet 0 last. If the packet
        // processor observes packet 0's valid header early, the rest of the
        // chain is already visible.
        for chain_pos in 1..packets.len() {
            let packet_idx = reservation_plan.base_packet_id + chain_pos as u64;
            let slot = (packet_idx & (self.ring_slots - 1)) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header = u16::from_le_bytes([packets[chain_pos][0], packets[chain_pos][1]]);
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(header, std::sync::atomic::Ordering::Release);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        {
            let slot = reservation_plan.first_slot_index as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header = u16::from_le_bytes([packets[0][0], packets[0][1]]);
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(header, std::sync::atomic::Ordering::Release);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let next = reservation_plan.desired_write_index;
        self.store_write_index_release(next);
        let doorbell_idx = reservation_plan.doorbell_packet_id;
        self.ring_doorbell_release(doorbell_idx);
        self.write_index = next;
        Ok(())
    }

    /// Enqueue N repetitions of one prebuilt kernel-dispatch packet and ring
    /// the doorbell once. This is the single-kernel hot-loop specialization for
    /// fixed decode replay schedules where materializing a packet-reference
    /// chain would be avoidable host work.
    ///
    /// # Safety
    /// The packet must reference live GPU-mapped memory that outlives the
    /// dispatch chain. If the caller waits on a completion signal, that signal
    /// must be initialized for one completion per repeated dispatch.
    pub unsafe fn dispatch_repeated_kernel_packet_for_replay(
        &mut self,
        packet: &[u8; AQL_PACKET_BYTES as usize],
        kernarg_size: u32,
        repeats: u32,
    ) -> Result<()> {
        if repeats == 0 {
            return Err(anyhow!(
                "mainarch AQL replay guard: empty repeated packet chain"
            ));
        }
        let packet_count = u64::from(repeats);
        if packet_count > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: repeated packet chain length {packet_count} exceeds ring slots {}",
                self.ring_slots
            ));
        }

        let idx = self.write_index;
        let observed_wptr = std::ptr::read_volatile(self.write_ptr_va as *const u64);
        if observed_wptr != idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue producer mailbox mismatch before repeated replay packet \
                (queue_id={}, gpu_id={}, expected_wptr={idx}, observed_wptr={observed_wptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let rptr = std::ptr::read_volatile(self.read_ptr_va as *const u64);
        if rptr > idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue consumer read pointer advanced beyond producer \
                before repeated replay packet (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let in_flight = idx - rptr;
        if in_flight + packet_count > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue ring lacks capacity before repeated replay packet \
                (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, in_flight={in_flight}, \
                chain_len={packet_count}, slots={}, ring_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_slots,
                self.ring_va,
            ));
        }

        let observed_packet =
            std::ptr::read_unaligned(packet.as_ptr().cast::<AqlKernelDispatchPacket>());
        let header = observed_packet.header;
        let body_for_guard = AqlKernelDispatchPacket {
            header: 0,
            ..observed_packet
        };
        let first_slot = (idx % self.ring_slots) as usize;
        validate_aql_packet_semantics(idx, first_slot, header, &body_for_guard, kernarg_size)?;
        let dispatch = AqlDispatch {
            kernel_object: observed_packet.kernel_object,
            kernarg_va: observed_packet.kernarg_address,
            kernarg_size,
            dims: (observed_packet.setup & 0xff) as u8,
            grid_x: observed_packet.grid_size_x,
            grid_y: observed_packet.grid_size_y,
            grid_z: observed_packet.grid_size_z,
            wg_x: observed_packet.workgroup_size_x,
            wg_y: observed_packet.workgroup_size_y,
            wg_z: observed_packet.workgroup_size_z,
            private_segment_size: observed_packet.private_segment_size,
            group_segment_size: observed_packet.group_segment_size,
            completion_signal: observed_packet.completion_signal,
        };
        validate_aql_dispatch(&dispatch, self.gpu_id)?;

        for chain_pos in 0..packet_count {
            let slot = ((idx + chain_pos) % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            std::ptr::write_bytes(pkt, 0, AQL_PACKET_BYTES as usize);
            std::ptr::copy_nonoverlapping(
                packet.as_ptr().add(2),
                pkt.add(2),
                AQL_PACKET_BYTES as usize - 2,
            );
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            for chain_pos in 0..packet_count {
                let slot = ((idx + chain_pos) % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                validate_written_aql_body(idx + chain_pos, slot, &body_for_guard, &observed)?;
            }
        }

        // Publish non-first headers first, then packet 0 last. If the packet
        // processor ever observes packet 0's valid header early, the rest of
        // the repeated chain is already visible.
        for chain_pos in 1..packet_count {
            let slot = ((idx + chain_pos) % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(header, std::sync::atomic::Ordering::Release);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        {
            let slot = (idx % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(header, std::sync::atomic::Ordering::Release);
        }
        if self.packet_guard {
            for chain_pos in 0..packet_count {
                let slot = ((idx + chain_pos) % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
                let header_atomic = &*(std::ptr::addr_of!((*pkt_struct).header)
                    as *const std::sync::atomic::AtomicU16);
                let observed_header = header_atomic.load(std::sync::atomic::Ordering::Acquire);
                if observed_header != header {
                    return Err(anyhow!(
                        "mainarch AQL replay guard: repeated packet header readback mismatch before doorbell (idx={}, slot={}, expected=0x{header:04x}, observed=0x{observed_header:04x})",
                        idx + chain_pos,
                        slot
                    ));
                }
            }
        }
        if aql_cpc_snapshot_enabled() {
            for chain_pos in 0..packet_count {
                let packet_idx = idx + chain_pos;
                let slot = (packet_idx % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                append_aql_cpc_snapshot(
                    "replay-repeat",
                    self.queue_id,
                    self.gpu_id,
                    packet_idx,
                    slot,
                    self.ring_va,
                    self.ring_slots,
                    self.write_ptr_va,
                    self.read_ptr_va,
                    self.doorbell_offset,
                    observed_wptr,
                    rptr,
                    &observed,
                    kernarg_size,
                )?;
            }
        }
        if aql_cpc_trace_enabled() {
            for chain_pos in 0..packet_count {
                let packet_idx = idx + chain_pos;
                let slot = (packet_idx % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                trace_aql_cpc_packet(
                    "replay-repeat",
                    self.queue_id,
                    self.gpu_id,
                    packet_idx,
                    slot,
                    self.ring_va,
                    self.ring_slots,
                    self.write_ptr_va,
                    self.read_ptr_va,
                    self.doorbell_offset,
                    observed_wptr,
                    rptr,
                    &observed,
                    kernarg_size,
                );
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let next = idx + packet_count;
        self.store_write_index_release(next);
        if self.packet_guard {
            let observed_next = std::ptr::read_volatile(self.write_ptr_va as *const u64);
            if observed_next != next {
                return Err(anyhow!(
                    "mainarch AQL replay guard: producer mailbox readback mismatch before repeated replay packet doorbell \
                    (queue_id={}, gpu_id={}, idx={idx}, expected_wptr={next}, \
                    observed_wptr={observed_next}, ring_va=0x{:016x}, write_ptr_va=0x{:016x})",
                    self.queue_id,
                    self.gpu_id,
                    self.ring_va,
                    self.write_ptr_va,
                ));
            }
        }
        let doorbell_idx = next - 1;
        self.ring_doorbell_release(doorbell_idx);
        self.write_index = next;
        Ok(())
    }

    /// Enqueue N repetitions of two prebuilt kernel-dispatch packets and ring
    /// the doorbell once. This is a hot-path specialization for fixed decode
    /// replay schedules where materializing a packet-reference chain would be
    /// avoidable host work.
    ///
    /// # Safety
    /// Both packets must reference live GPU-mapped memory that outlives the
    /// dispatch chain. The second packet in each pair should normally carry
    /// the completion signal when the caller waits for one completion per
    /// repeated pair.
    pub unsafe fn dispatch_repeated_kernel_packet_pair_for_replay(
        &mut self,
        first_packet: &[u8; AQL_PACKET_BYTES as usize],
        first_kernarg_size: u32,
        second_packet: &[u8; AQL_PACKET_BYTES as usize],
        second_kernarg_size: u32,
        repeats: u32,
    ) -> Result<()> {
        if repeats == 0 {
            return Err(anyhow!(
                "mainarch AQL replay guard: empty repeated packet pair chain"
            ));
        }
        let packet_count = u64::from(repeats).checked_mul(2).ok_or_else(|| {
            anyhow!("mainarch AQL replay guard: repeated packet pair length overflow")
        })?;
        if packet_count > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: repeated packet pair chain length {packet_count} exceeds ring slots {}",
                self.ring_slots
            ));
        }

        let idx = self.write_index;
        let observed_wptr = std::ptr::read_volatile(self.write_ptr_va as *const u64);
        if observed_wptr != idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue producer mailbox mismatch before repeated replay pair \
                (queue_id={}, gpu_id={}, expected_wptr={idx}, observed_wptr={observed_wptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let rptr = std::ptr::read_volatile(self.read_ptr_va as *const u64);
        if rptr > idx {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue consumer read pointer advanced beyond producer \
                before repeated replay pair (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, \
                ring_va=0x{:016x}, write_ptr_va=0x{:016x}, read_ptr_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_va,
                self.write_ptr_va,
                self.read_ptr_va,
            ));
        }
        let in_flight = idx - rptr;
        if in_flight + packet_count > self.ring_slots {
            return Err(anyhow!(
                "mainarch AQL replay guard: queue ring lacks capacity before repeated replay pair \
                (queue_id={}, gpu_id={}, wptr={idx}, rptr={rptr}, in_flight={in_flight}, \
                chain_len={packet_count}, slots={}, ring_va=0x{:016x})",
                self.queue_id,
                self.gpu_id,
                self.ring_slots,
                self.ring_va,
            ));
        }

        let first_observed =
            std::ptr::read_unaligned(first_packet.as_ptr().cast::<AqlKernelDispatchPacket>());
        let first_header = first_observed.header;
        let first_body_for_guard = AqlKernelDispatchPacket {
            header: 0,
            ..first_observed
        };
        let first_dispatch = AqlDispatch {
            kernel_object: first_observed.kernel_object,
            kernarg_va: first_observed.kernarg_address,
            kernarg_size: first_kernarg_size,
            dims: (first_observed.setup & 0xff) as u8,
            grid_x: first_observed.grid_size_x,
            grid_y: first_observed.grid_size_y,
            grid_z: first_observed.grid_size_z,
            wg_x: first_observed.workgroup_size_x,
            wg_y: first_observed.workgroup_size_y,
            wg_z: first_observed.workgroup_size_z,
            private_segment_size: first_observed.private_segment_size,
            group_segment_size: first_observed.group_segment_size,
            completion_signal: first_observed.completion_signal,
        };
        validate_aql_dispatch(&first_dispatch, self.gpu_id)?;

        let second_observed =
            std::ptr::read_unaligned(second_packet.as_ptr().cast::<AqlKernelDispatchPacket>());
        let second_header = second_observed.header;
        let second_body_for_guard = AqlKernelDispatchPacket {
            header: 0,
            ..second_observed
        };
        let second_dispatch = AqlDispatch {
            kernel_object: second_observed.kernel_object,
            kernarg_va: second_observed.kernarg_address,
            kernarg_size: second_kernarg_size,
            dims: (second_observed.setup & 0xff) as u8,
            grid_x: second_observed.grid_size_x,
            grid_y: second_observed.grid_size_y,
            grid_z: second_observed.grid_size_z,
            wg_x: second_observed.workgroup_size_x,
            wg_y: second_observed.workgroup_size_y,
            wg_z: second_observed.workgroup_size_z,
            private_segment_size: second_observed.private_segment_size,
            group_segment_size: second_observed.group_segment_size,
            completion_signal: second_observed.completion_signal,
        };
        validate_aql_dispatch(&second_dispatch, self.gpu_id)?;

        for chain_pos in 0..packet_count {
            let use_second = (chain_pos & 1) != 0;
            let packet = if use_second {
                second_packet
            } else {
                first_packet
            };
            let kernarg_size = if use_second {
                second_kernarg_size
            } else {
                first_kernarg_size
            };
            let header = if use_second {
                second_header
            } else {
                first_header
            };
            let body_for_guard = if use_second {
                &second_body_for_guard
            } else {
                &first_body_for_guard
            };
            let slot = ((idx + chain_pos) % self.ring_slots) as usize;
            validate_aql_packet_semantics(
                idx + chain_pos,
                slot,
                header,
                body_for_guard,
                kernarg_size,
            )?;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            std::ptr::write_bytes(pkt, 0, AQL_PACKET_BYTES as usize);
            std::ptr::copy_nonoverlapping(
                packet.as_ptr().add(2),
                pkt.add(2),
                AQL_PACKET_BYTES as usize - 2,
            );
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        if self.packet_guard {
            for chain_pos in 0..packet_count {
                let body_for_guard = if (chain_pos & 1) != 0 {
                    &second_body_for_guard
                } else {
                    &first_body_for_guard
                };
                let slot = ((idx + chain_pos) % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                validate_written_aql_body(idx + chain_pos, slot, body_for_guard, &observed)?;
            }
        }

        for chain_pos in 1..packet_count {
            let slot = ((idx + chain_pos) % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            let header = if (chain_pos & 1) != 0 {
                second_header
            } else {
                first_header
            };
            header_atomic.store(header, std::sync::atomic::Ordering::Release);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        {
            let slot = (idx % self.ring_slots) as usize;
            let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
            let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
            let header_atomic =
                &*(std::ptr::addr_of!((*pkt_struct).header) as *const std::sync::atomic::AtomicU16);
            header_atomic.store(first_header, std::sync::atomic::Ordering::Release);
        }
        if self.packet_guard {
            for chain_pos in 0..packet_count {
                let slot = ((idx + chain_pos) % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let pkt_struct = pkt as *mut AqlKernelDispatchPacket;
                let header_atomic = &*(std::ptr::addr_of!((*pkt_struct).header)
                    as *const std::sync::atomic::AtomicU16);
                let observed_header = header_atomic.load(std::sync::atomic::Ordering::Acquire);
                let expected_header = if (chain_pos & 1) != 0 {
                    second_header
                } else {
                    first_header
                };
                if observed_header != expected_header {
                    return Err(anyhow!(
                        "mainarch AQL replay guard: repeated pair header readback mismatch before doorbell (idx={}, slot={}, expected=0x{expected_header:04x}, observed=0x{observed_header:04x})",
                        idx + chain_pos,
                        slot
                    ));
                }
            }
        }
        if aql_cpc_snapshot_enabled() {
            for chain_pos in 0..packet_count {
                let packet_idx = idx + chain_pos;
                let slot = (packet_idx % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                let kernarg_size = if (chain_pos & 1) != 0 {
                    second_kernarg_size
                } else {
                    first_kernarg_size
                };
                append_aql_cpc_snapshot(
                    "replay-pair",
                    self.queue_id,
                    self.gpu_id,
                    packet_idx,
                    slot,
                    self.ring_va,
                    self.ring_slots,
                    self.write_ptr_va,
                    self.read_ptr_va,
                    self.doorbell_offset,
                    observed_wptr,
                    rptr,
                    &observed,
                    kernarg_size,
                )?;
            }
        }
        if aql_cpc_trace_enabled() {
            for chain_pos in 0..packet_count {
                let packet_idx = idx + chain_pos;
                let slot = (packet_idx % self.ring_slots) as usize;
                let pkt = (self.ring_va as *mut u8).add(slot * AQL_PACKET_BYTES as usize);
                let observed = std::ptr::read_volatile(pkt as *const AqlKernelDispatchPacket);
                let kernarg_size = if (chain_pos & 1) != 0 {
                    second_kernarg_size
                } else {
                    first_kernarg_size
                };
                trace_aql_cpc_packet(
                    "replay-pair",
                    self.queue_id,
                    self.gpu_id,
                    packet_idx,
                    slot,
                    self.ring_va,
                    self.ring_slots,
                    self.write_ptr_va,
                    self.read_ptr_va,
                    self.doorbell_offset,
                    observed_wptr,
                    rptr,
                    &observed,
                    kernarg_size,
                );
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let next = idx + packet_count;
        self.store_write_index_release(next);
        if self.packet_guard {
            let observed_next = std::ptr::read_volatile(self.write_ptr_va as *const u64);
            if observed_next != next {
                return Err(anyhow!(
                    "mainarch AQL replay guard: producer mailbox readback mismatch before repeated pair doorbell \
                    (queue_id={}, gpu_id={}, idx={idx}, expected_wptr={next}, \
                    observed_wptr={observed_next}, ring_va=0x{:016x}, write_ptr_va=0x{:016x})",
                    self.queue_id,
                    self.gpu_id,
                    self.ring_va,
                    self.write_ptr_va,
                ));
            }
        }
        let doorbell_idx = next - 1;
        self.ring_doorbell_release(doorbell_idx);
        self.write_index = next;
        Ok(())
    }
}

/// Geometry + code references for a single AQL kernel dispatch.
#[derive(Debug, Clone, Copy)]
pub struct AqlDispatch {
    pub kernel_object: u64,
    pub kernarg_va: u64,
    pub kernarg_size: u32,
    pub dims: u8,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub wg_x: u16,
    pub wg_y: u16,
    pub wg_z: u16,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    /// VA of an `amd_signal_t` the CP decrements on completion, or 0 for none.
    pub completion_signal: u64,
}

const AMDGPU_KERNEL_DESCRIPTOR_BYTES: u64 = 64;
const AMD_SIGNAL_BYTES: u64 = 64;
const MAX_WORKGROUP_SIZE: u64 = 1024;
const MAX_KERNARG_BYTES: u32 = 4096;
const KERNEL_DESCRIPTOR_ALIGN: u64 = 64;
const KERNARG_ALIGN: u64 = 16;
const AMD_SIGNAL_ALIGN: u64 = 8;
const AQL_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const AQL_FENCE_SCOPE_SYSTEM: u16 = 2;
const MAINARCH_AQL_KERNEL_HEADER: u16 = AQL_PACKET_TYPE_KERNEL_DISPATCH
    | (1 << 8)
    | (AQL_FENCE_SCOPE_SYSTEM << 9)
    | (AQL_FENCE_SCOPE_SYSTEM << 11);

fn validate_aql_dispatch(d: &AqlDispatch, target_gpu_id: u32) -> Result<()> {
    validate_aql_geometry(d)?;
    if d.kernarg_size == 0 {
        return Err(anyhow!(
            "invalid AQL dispatch: kernarg_size is zero for kernel_object=0x{:016x}",
            d.kernel_object
        ));
    }
    if gpu_va_guard_disabled() {
        return Ok(());
    }
    validate_registered_gpu_span(
        "kernel_object",
        d.kernel_object,
        AMDGPU_KERNEL_DESCRIPTOR_BYTES,
        false,
        target_gpu_id,
    )?;
    validate_registered_gpu_span(
        "kernarg",
        d.kernarg_va,
        d.kernarg_size as u64,
        true,
        target_gpu_id,
    )?;
    if d.completion_signal != 0 {
        validate_registered_gpu_span(
            "completion_signal",
            d.completion_signal,
            AMD_SIGNAL_BYTES,
            true,
            target_gpu_id,
        )?;
    }
    Ok(())
}

fn gpu_va_guard_disabled() -> bool {
    std::env::var("MAINARCH_DISABLE_GPU_VA_GUARD")
        .map(|v| {
            let v = v.trim();
            !(v.eq_ignore_ascii_case("0")
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no"))
        })
        .unwrap_or(false)
}

fn validate_aql_geometry(d: &AqlDispatch) -> Result<()> {
    if d.dims == 0 || d.dims > 3 {
        return Err(anyhow!(
            "invalid AQL dispatch: dims={} (expected 1..=3)",
            d.dims
        ));
    }
    let grids = [d.grid_x, d.grid_y, d.grid_z];
    let wgs = [d.wg_x as u32, d.wg_y as u32, d.wg_z as u32];
    let mut wg_total = 1u64;
    for axis in 0..3 {
        if axis < d.dims as usize {
            if grids[axis] == 0 {
                return Err(anyhow!("invalid AQL dispatch: grid axis {axis} is zero"));
            }
            if wgs[axis] == 0 {
                return Err(anyhow!(
                    "invalid AQL dispatch: workgroup axis {axis} is zero"
                ));
            }
            wg_total = wg_total
                .checked_mul(wgs[axis] as u64)
                .ok_or_else(|| anyhow!("invalid AQL dispatch: workgroup size overflows"))?;
        } else if grids[axis] != 1 || wgs[axis] != 1 {
            return Err(anyhow!(
                "invalid AQL dispatch: inactive axis {axis} has grid={} wg={} (expected 1)",
                grids[axis],
                wgs[axis]
            ));
        }
    }
    if wg_total > MAX_WORKGROUP_SIZE {
        return Err(anyhow!(
            "invalid AQL dispatch: workgroup has {wg_total} threads (max {MAX_WORKGROUP_SIZE})"
        ));
    }
    Ok(())
}

fn validate_aql_packet_semantics(
    idx: u64,
    slot: usize,
    header: u16,
    body: &AqlKernelDispatchPacket,
    kernarg_size: u32,
) -> Result<()> {
    if header != MAINARCH_AQL_KERNEL_HEADER {
        return Err(anyhow!(
            "mainarch AQL guard: invalid packet header before doorbell (idx={idx}, slot={slot}, expected=0x{:04x}, observed=0x{header:04x})",
            MAINARCH_AQL_KERNEL_HEADER
        ));
    }
    if body.header != 0 {
        return Err(anyhow!(
            "mainarch AQL guard: packet body header must remain unpublished before doorbell (idx={idx}, slot={slot}, observed=0x{:04x})",
            body.header
        ));
    }
    if body.reserved0 != 0 || body.reserved2 != 0 {
        return Err(anyhow!(
            "mainarch AQL guard: reserved packet fields are nonzero before doorbell (idx={idx}, slot={slot}, reserved0=0x{:04x}, reserved2=0x{:016x})",
            body.reserved0,
            body.reserved2
        ));
    }
    if kernarg_size == 0 || kernarg_size > MAX_KERNARG_BYTES {
        return Err(anyhow!(
            "mainarch AQL guard: invalid kernarg size before doorbell (idx={idx}, slot={slot}, kernarg_size={kernarg_size}, max={MAX_KERNARG_BYTES})"
        ));
    }
    if body.kernel_object & (KERNEL_DESCRIPTOR_ALIGN - 1) != 0 {
        return Err(anyhow!(
            "mainarch AQL guard: kernel descriptor is not {KERNEL_DESCRIPTOR_ALIGN}-byte aligned before doorbell (idx={idx}, slot={slot}, kernel_object=0x{:016x})",
            body.kernel_object
        ));
    }
    if body.kernarg_address & (KERNARG_ALIGN - 1) != 0 {
        return Err(anyhow!(
            "mainarch AQL guard: kernarg address is not {KERNARG_ALIGN}-byte aligned before doorbell (idx={idx}, slot={slot}, kernarg=0x{:016x})",
            body.kernarg_address
        ));
    }
    if body.completion_signal != 0 && body.completion_signal & (AMD_SIGNAL_ALIGN - 1) != 0 {
        return Err(anyhow!(
            "mainarch AQL guard: completion signal is not {AMD_SIGNAL_ALIGN}-byte aligned before doorbell (idx={idx}, slot={slot}, signal=0x{:016x})",
            body.completion_signal
        ));
    }
    Ok(())
}

fn validate_registered_gpu_span(
    label: &str,
    va: u64,
    bytes: u64,
    require_host_visible: bool,
    target_gpu_id: u32,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    if va == 0 {
        return Err(anyhow!("invalid AQL dispatch: {label} VA is null"));
    }
    let Some(record) = find_live_gpu_allocation(va) else {
        return Err(anyhow!(
            "invalid AQL dispatch: {label}=0x{va:016x} is not inside a live KFD allocation; {}",
            gpu_va_debug_context(va)
        ));
    };
    let offset = va - record.va;
    let Some(end) = offset.checked_add(bytes) else {
        return Err(anyhow!(
            "invalid AQL dispatch: {label} span overflows: va=0x{va:016x} bytes={bytes}"
        ));
    };
    if end > record.len as u64 {
        return Err(anyhow!(
            "invalid AQL dispatch: {label} span out of bounds: va=0x{va:016x} bytes={bytes} allocation=0x{:016x}..0x{:016x} owner_gpu_id={} mapped_gpu_ids={:?} flags=0x{:08x}",
            record.va,
            record.va.saturating_add(record.len as u64),
            record.gpu_id,
            record.mapped_gpu_ids(),
            record.flags
        ));
    }
    if !record.is_mapped_to_gpu(target_gpu_id) {
        return Err(anyhow!(
            "invalid AQL dispatch: {label}=0x{va:016x} is live but not mapped to target gpu_id={target_gpu_id}; allocation=0x{:016x}..0x{:016x} owner_gpu_id={} mapped_gpu_ids={:?}",
            record.va,
            record.va.saturating_add(record.len as u64),
            record.gpu_id,
            record.mapped_gpu_ids()
        ));
    }
    if require_host_visible && !allocation_flags_are_host_visible(record.flags) {
        return Err(anyhow!(
            "invalid AQL dispatch: {label}=0x{va:016x} is not in host-visible KFD memory; allocation=0x{:016x}..0x{:016x} owner_gpu_id={} mapped_gpu_ids={:?} flags=0x{:08x}",
            record.va,
            record.va.saturating_add(record.len as u64),
            record.gpu_id,
            record.mapped_gpu_ids(),
            record.flags
        ));
    }
    Ok(())
}

fn allocation_flags_are_host_visible(flags: u32) -> bool {
    (flags & mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0
        || (flags & mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_GTT) != 0
        || ((flags & mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_VRAM) != 0
            && (flags & mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC) != 0)
}

fn validate_written_aql_body(
    idx: u64,
    slot: usize,
    expected: &AqlKernelDispatchPacket,
    observed: &AqlKernelDispatchPacket,
) -> Result<()> {
    macro_rules! check_field {
        ($field:ident) => {
            if observed.$field != expected.$field {
                return Err(anyhow!(
                    "mainarch AQL guard: packet body readback mismatch before header publish (idx={}, slot={}, field={}, expected={:?}, observed={:?})",
                    idx,
                    slot,
                    stringify!($field),
                    expected.$field,
                    observed.$field
                ));
            }
        };
    }

    check_field!(header);
    check_field!(setup);
    check_field!(workgroup_size_x);
    check_field!(workgroup_size_y);
    check_field!(workgroup_size_z);
    check_field!(reserved0);
    check_field!(grid_size_x);
    check_field!(grid_size_y);
    check_field!(grid_size_z);
    check_field!(private_segment_size);
    check_field!(group_segment_size);
    check_field!(kernel_object);
    check_field!(kernarg_address);
    check_field!(reserved2);
    check_field!(completion_signal);
    Ok(())
}

fn aql_cpc_trace_enabled() -> bool {
    std::env::var_os("MAINARCH_AQL_CPC_TRACE").is_some()
        || std::env::var_os("MAINARCH_AQL_TRACE").is_some()
}

fn aql_cpc_snapshot_enabled() -> bool {
    std::env::var_os("MAINARCH_AQL_CPC_SNAPSHOT").is_some()
}

#[allow(clippy::too_many_arguments)]
fn append_aql_cpc_snapshot(
    label: &str,
    queue_id: u32,
    gpu_id: u32,
    idx: u64,
    slot: usize,
    ring_va: u64,
    ring_slots: u64,
    write_ptr_va: u64,
    read_ptr_va: u64,
    doorbell_offset: u64,
    observed_wptr: u64,
    observed_rptr: u64,
    packet: &AqlKernelDispatchPacket,
    kernarg_size: u32,
) -> Result<()> {
    use std::io::Write as _;

    let Some(path) = std::env::var_os("MAINARCH_AQL_CPC_SNAPSHOT") else {
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "create MAINARCH_AQL_CPC_SNAPSHOT parent {}",
                parent.display()
            )
        })?;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open MAINARCH_AQL_CPC_SNAPSHOT {}", path.display()))?;
    let packet_va = ring_va + slot as u64 * AQL_PACKET_BYTES;
    let words = aql_packet_raw_words(packet);
    let unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    writeln!(
        file,
        "BEGIN_AQL_CPC_SNAPSHOT ts_unix_ns={unix_ns} pid={} label={label} queue_id={queue_id} gpu_id={gpu_id} idx={idx} slot={slot}",
        std::process::id()
    )?;
    writeln!(
        file,
        "queue ring_va=0x{ring_va:016x} ring_slots={ring_slots} write_ptr_va=0x{write_ptr_va:016x} read_ptr_va=0x{read_ptr_va:016x} observed_wptr={observed_wptr} observed_rptr={observed_rptr} doorbell_offset=0x{doorbell_offset:016x}"
    )?;
    writeln!(
        file,
        "packet packet_va=0x{packet_va:016x} header=0x{:04x} setup=0x{:04x} grid=({},{},{}) wg=({},{},{}) group={} private={} kernel_object=0x{:016x} kernarg=0x{:016x} kernarg_size={} completion_signal=0x{:016x}",
        packet.header,
        packet.setup,
        packet.grid_size_x,
        packet.grid_size_y,
        packet.grid_size_z,
        packet.workgroup_size_x,
        packet.workgroup_size_y,
        packet.workgroup_size_z,
        packet.group_segment_size,
        packet.private_segment_size,
        packet.kernel_object,
        packet.kernarg_address,
        kernarg_size,
        packet.completion_signal,
    )?;
    writeln!(
        file,
        "raw_u64=[0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x}]",
        words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]
    )?;
    writeln!(
        file,
        "span {}",
        gpu_span_trace_context("ring_packet", packet_va, AQL_PACKET_BYTES)
    )?;
    writeln!(
        file,
        "span {}",
        gpu_span_trace_context("write_ptr", write_ptr_va, 8)
    )?;
    writeln!(
        file,
        "span {}",
        gpu_span_trace_context("read_ptr", read_ptr_va, 8)
    )?;
    writeln!(
        file,
        "span {}",
        gpu_span_trace_context(
            "kernel_object",
            packet.kernel_object,
            AMDGPU_KERNEL_DESCRIPTOR_BYTES
        )
    )?;
    writeln!(
        file,
        "span {}",
        gpu_span_trace_context("kernarg", packet.kernarg_address, u64::from(kernarg_size))
    )?;
    if packet.completion_signal == 0 {
        writeln!(file, "span completion_signal=none")?;
    } else {
        writeln!(
            file,
            "span {}",
            gpu_span_trace_context(
                "completion_signal",
                packet.completion_signal,
                AMD_SIGNAL_BYTES
            )
        )?;
    }
    writeln!(file, "END_AQL_CPC_SNAPSHOT")?;
    file.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn trace_aql_cpc_packet(
    label: &str,
    queue_id: u32,
    gpu_id: u32,
    idx: u64,
    slot: usize,
    ring_va: u64,
    ring_slots: u64,
    write_ptr_va: u64,
    read_ptr_va: u64,
    doorbell_offset: u64,
    observed_wptr: u64,
    observed_rptr: u64,
    packet: &AqlKernelDispatchPacket,
    kernarg_size: u32,
) {
    let packet_va = ring_va + slot as u64 * AQL_PACKET_BYTES;
    let words = aql_packet_raw_words(packet);
    eprintln!(
        "mainarch: AQL_CPC_TRACE {label} queue_id={queue_id} gpu_id={gpu_id} \
         idx={idx} slot={slot} packet_va=0x{packet_va:016x} ring_va=0x{ring_va:016x} \
         ring_slots={ring_slots} write_ptr_va=0x{write_ptr_va:016x} \
         read_ptr_va=0x{read_ptr_va:016x} observed_wptr={observed_wptr} \
         observed_rptr={observed_rptr} doorbell_offset=0x{doorbell_offset:016x} \
         header=0x{:04x} setup=0x{:04x} grid=({},{},{}) wg=({},{},{}) \
         group={} private={} kernel_object=0x{:016x} kernarg=0x{:016x} \
         kernarg_size={} completion_signal=0x{:016x} \
         raw_u64=[0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x},0x{:016x}]",
        packet.header,
        packet.setup,
        packet.grid_size_x,
        packet.grid_size_y,
        packet.grid_size_z,
        packet.workgroup_size_x,
        packet.workgroup_size_y,
        packet.workgroup_size_z,
        packet.group_segment_size,
        packet.private_segment_size,
        packet.kernel_object,
        packet.kernarg_address,
        kernarg_size,
        packet.completion_signal,
        words[0],
        words[1],
        words[2],
        words[3],
        words[4],
        words[5],
        words[6],
        words[7],
    );
    eprintln!(
        "mainarch: AQL_CPC_TRACE {label} idx={idx} {}",
        gpu_span_trace_context(
            "kernel_object",
            packet.kernel_object,
            AMDGPU_KERNEL_DESCRIPTOR_BYTES
        )
    );
    eprintln!(
        "mainarch: AQL_CPC_TRACE {label} idx={idx} {}",
        gpu_span_trace_context("kernarg", packet.kernarg_address, u64::from(kernarg_size))
    );
    if packet.completion_signal == 0 {
        eprintln!("mainarch: AQL_CPC_TRACE {label} idx={idx} completion_signal=none");
    } else {
        eprintln!(
            "mainarch: AQL_CPC_TRACE {label} idx={idx} {}",
            gpu_span_trace_context(
                "completion_signal",
                packet.completion_signal,
                AMD_SIGNAL_BYTES
            )
        );
    }
}

fn aql_packet_raw_words(packet: &AqlKernelDispatchPacket) -> [u64; 8] {
    let mut bytes = [0u8; AQL_PACKET_BYTES as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(
            (packet as *const AqlKernelDispatchPacket).cast::<u8>(),
            bytes.as_mut_ptr(),
            bytes.len(),
        );
    }
    let mut words = [0u64; 8];
    for i in 0..words.len() {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        words[i] = u64::from_le_bytes(word);
    }
    words
}

fn gpu_span_trace_context(label: &str, va: u64, bytes: u64) -> String {
    if bytes == 0 {
        return format!("{label}=0x{va:016x} bytes=0");
    }
    if va == 0 {
        return format!("{label}=0x0000000000000000 bytes={bytes} null");
    }
    let Some(record) = find_live_gpu_allocation(va) else {
        return format!(
            "{label}=0x{va:016x} bytes={bytes} live_allocation=missing {}",
            gpu_va_debug_context(va)
        );
    };
    let allocation_end = record.va.saturating_add(record.len as u64);
    let offset = va.saturating_sub(record.va);
    let requested_end = va.saturating_add(bytes);
    let remaining = allocation_end.saturating_sub(va);
    format!(
        "{label}=0x{va:016x} bytes={bytes} requested_end=0x{requested_end:016x} \
         allocation=0x{:016x}..0x{:016x} offset={offset} remaining={remaining} \
         owner_gpu_id={} mapped_gpu_ids={:?} flags=0x{:08x} host_visible={}",
        record.va,
        allocation_end,
        record.gpu_id,
        record.mapped_gpu_ids(),
        record.flags,
        allocation_flags_are_host_visible(record.flags),
    )
}

/// `hsa_kernel_dispatch_packet_t` (64 bytes, little-endian).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AqlKernelDispatchPacket {
    header: u16,
    setup: u16,
    workgroup_size_x: u16,
    workgroup_size_y: u16,
    workgroup_size_z: u16,
    reserved0: u16,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    reserved2: u64,
    completion_signal: u64,
}

const _: () = assert!(core::mem::size_of::<AqlKernelDispatchPacket>() == 64);

/// Build the exact final 64-byte AQL kernel-dispatch packet bytes for shadow
/// replay validation. This mirrors the eager packet body plus published header,
/// while allowing the caller to substitute the kernarg VA that a replay template
/// would use.
pub fn build_aql_kernel_dispatch_packet_bytes_for_shadow(
    d: &AqlDispatch,
    kernarg_va: u64,
) -> Result<[u8; AQL_PACKET_BYTES as usize]> {
    validate_aql_geometry(d)?;
    if kernarg_va == 0 {
        return Err(anyhow!("AQL shadow packet kernarg VA is null"));
    }
    if kernarg_va & (KERNARG_ALIGN - 1) != 0 {
        return Err(anyhow!(
            "AQL shadow packet kernarg VA 0x{kernarg_va:016x} is not {KERNARG_ALIGN}-byte aligned"
        ));
    }
    if d.kernel_object & (KERNEL_DESCRIPTOR_ALIGN - 1) != 0 {
        return Err(anyhow!(
            "AQL shadow packet kernel_object 0x{:016x} is not {KERNEL_DESCRIPTOR_ALIGN}-byte aligned",
            d.kernel_object
        ));
    }
    if d.completion_signal != 0 && d.completion_signal & (AMD_SIGNAL_ALIGN - 1) != 0 {
        return Err(anyhow!(
            "AQL shadow packet completion_signal 0x{:016x} is not {AMD_SIGNAL_ALIGN}-byte aligned",
            d.completion_signal
        ));
    }

    let pkt = AqlKernelDispatchPacket {
        header: MAINARCH_AQL_KERNEL_HEADER,
        setup: d.dims as u16,
        workgroup_size_x: d.wg_x,
        workgroup_size_y: d.wg_y,
        workgroup_size_z: d.wg_z,
        reserved0: 0,
        grid_size_x: d.grid_x,
        grid_size_y: d.grid_y,
        grid_size_z: d.grid_z,
        private_segment_size: d.private_segment_size,
        group_segment_size: d.group_segment_size,
        kernel_object: d.kernel_object,
        kernarg_address: kernarg_va,
        reserved2: 0,
        completion_signal: d.completion_signal,
    };
    let mut bytes = [0u8; AQL_PACKET_BYTES as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&pkt as *const AqlKernelDispatchPacket).cast::<u8>(),
            bytes.as_mut_ptr(),
            bytes.len(),
        );
    }
    Ok(bytes)
}

/// A public, host-visible GPU allocation: CPU and GPU observe the same VA.
///
/// Backed by either a userptr registration or a host-visible BO; in both cases
/// `va()` is valid for CPU loads/stores and as a GPU-side pointer.
pub struct DeviceBuffer {
    inner: KfdAllocatedBuffer,
}

impl DeviceBuffer {
    /// GPU (and CPU) virtual address of the allocation.
    pub fn va(&self) -> u64 {
        self.inner.ptr()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Mutable byte view of the allocation (host-visible).
    ///
    /// # Safety
    /// The GPU may concurrently read or write this memory; the caller is
    /// responsible for ordering (e.g. only touch it before dispatch / after the
    /// dispatch is observed complete).
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.inner.host_ptr(), self.inner.len())
    }

    /// Typed mutable view (host-visible). See [`DeviceBuffer::as_mut_slice`].
    ///
    /// # Safety
    /// Same ordering caveats; `T` must be valid for any bit pattern.
    pub unsafe fn as_mut_slice_of<T: Copy>(&mut self) -> &mut [T] {
        let n = self.inner.len() / core::mem::size_of::<T>();
        std::slice::from_raw_parts_mut(self.inner.host_ptr() as *mut T, n)
    }

    /// Read a `u32` at byte `offset` (host-visible).
    pub fn read_u32(&self, offset: usize) -> u32 {
        unsafe { std::ptr::read_volatile(self.inner.host_ptr().add(offset) as *const u32) }
    }

    pub fn allocation_flags(&self) -> u32 {
        self.inner.flags()
    }

    pub fn is_public_coherent_vram(&self) -> bool {
        let flags = self.allocation_flags();
        let required = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        flags & required == required
    }
}

const MAX_TRACKED_GPU_MAPPINGS: usize = 16;

#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuAllocationRecord {
    pub va: u64,
    pub len: usize,
    pub gpu_id: u32,
    pub flags: u32,
    mapped_gpu_ids: [u32; MAX_TRACKED_GPU_MAPPINGS],
    mapped_gpu_count: u8,
}

impl GpuAllocationRecord {
    fn new(va: u64, len: usize, gpu_id: u32, flags: u32) -> Self {
        let mut mapped_gpu_ids = [0; MAX_TRACKED_GPU_MAPPINGS];
        mapped_gpu_ids[0] = gpu_id;
        Self {
            va,
            len,
            gpu_id,
            flags,
            mapped_gpu_ids,
            mapped_gpu_count: 1,
        }
    }

    pub(crate) fn is_mapped_to_gpu(&self, gpu_id: u32) -> bool {
        self.mapped_gpu_ids()
            .iter()
            .any(|&mapped_gpu_id| mapped_gpu_id == gpu_id)
    }

    pub(crate) fn mapped_gpu_ids(&self) -> &[u32] {
        &self.mapped_gpu_ids[..self.mapped_gpu_count as usize]
    }
}

static GPU_ALLOCATION_REGISTRY: OnceLock<Mutex<Vec<GpuAllocationRecord>>> = OnceLock::new();

fn gpu_allocation_registry() -> &'static Mutex<Vec<GpuAllocationRecord>> {
    GPU_ALLOCATION_REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

fn register_gpu_allocation(record: GpuAllocationRecord) {
    gpu_allocation_registry()
        .lock()
        .expect("gpu allocation registry poisoned")
        .push(record);
}

fn register_gpu_allocation_mapping(va: u64, gpu_id: u32) -> Result<()> {
    let mut records = gpu_allocation_registry()
        .lock()
        .expect("gpu allocation registry poisoned");
    let Some(record) = records.iter_mut().find(|record| record.va == va) else {
        return Err(anyhow!(
            "mainarch GPU allocation registry: cannot mark unknown allocation 0x{va:016x} as mapped to gpu_id={gpu_id}"
        ));
    };
    if record.is_mapped_to_gpu(gpu_id) {
        return Ok(());
    }
    let count = record.mapped_gpu_count as usize;
    if count >= MAX_TRACKED_GPU_MAPPINGS {
        return Err(anyhow!(
            "mainarch GPU allocation registry: allocation 0x{va:016x} exceeded {MAX_TRACKED_GPU_MAPPINGS} tracked GPU mappings"
        ));
    }
    record.mapped_gpu_ids[count] = gpu_id;
    record.mapped_gpu_count += 1;
    Ok(())
}

fn unregister_gpu_allocation(va: u64) -> Option<Vec<u32>> {
    let mut records = gpu_allocation_registry()
        .lock()
        .expect("gpu allocation registry poisoned");
    if let Some(index) = records.iter().position(|record| record.va == va) {
        let record = records.swap_remove(index);
        return Some(record.mapped_gpu_ids().to_vec());
    }
    None
}

pub(crate) fn find_live_gpu_allocation(va: u64) -> Option<GpuAllocationRecord> {
    let records = gpu_allocation_registry()
        .lock()
        .expect("gpu allocation registry poisoned");
    records.iter().copied().find(|record| {
        let end = record.va.saturating_add(record.len as u64);
        va >= record.va && va < end
    })
}

pub(crate) fn gpu_va_debug_context(va: u64) -> String {
    let records = gpu_allocation_registry()
        .lock()
        .expect("gpu allocation registry poisoned");
    if records.is_empty() {
        return "live_allocations=0".to_string();
    }

    let mut nearest: Vec<(u64, GpuAllocationRecord)> = records
        .iter()
        .copied()
        .map(|record| (gpu_va_distance(va, record), record))
        .collect();
    nearest.sort_by_key(|(distance, record)| (*distance, record.va));

    let mut out = format!("live_allocations={} nearest_allocations=", records.len());
    for (i, (_, record)) in nearest.iter().take(4).enumerate() {
        if i != 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "0x{:016x}..0x{:016x}/owner_gpu_id={}/mapped_gpu_ids={:?}/flags=0x{:08x}",
            record.va,
            record.va.saturating_add(record.len as u64),
            record.gpu_id,
            record.mapped_gpu_ids(),
            record.flags
        ));
    }
    out
}

fn gpu_va_distance(va: u64, record: GpuAllocationRecord) -> u64 {
    let end = record.va.saturating_add(record.len as u64);
    if va < record.va {
        record.va - va
    } else if va >= end {
        va.saturating_sub(end).saturating_add(1)
    } else {
        0
    }
}

impl Drop for KfdQueue {
    fn drop(&mut self) {
        if !self.doorbell_map_base.is_null() {
            unsafe {
                let _ = libc::munmap(self.doorbell_map_base, self.doorbell_map_len);
            }
        }
        let mut args = mainarch_sys::DestroyQueueArgs {
            queue_id: self.queue_id,
            pad: 0,
        };

        // best-effort cleanup; ignore failures on drop-path
        unsafe {
            let _ = mainarch_sys::ioctl_destroy_queue(self.kfd_fd, &mut args);
        }
    }
}

/// SDMA (copy-engine) user queue driven straight through the KFD ABI — no ROCm.
/// The dedicated DMA engines saturate XGMI links better than compute-kernel
/// stores (which top out below the link ceiling), so this is the path to peak
/// peer-to-peer bandwidth (the mechanism RCCL uses for large transfers).
///
/// Linear-copy (`SDMA_OP_COPY`/LINEAR) packets are written to a dword ring;
/// completion is observed by polling the read pointer reaching the write pointer.
pub struct SdmaQueue {
    kfd_fd: i32,
    queue_id: u32,
    ring_host: *mut u32,
    ring_dwords: usize,
    wptr_host: *mut u64,
    rptr_host: *const u64,
    doorbell: *mut u64,
    doorbell_map_base: *mut c_void,
    doorbell_map_len: usize,
    /// Monotonic write pointer, in dwords.
    wptr: u64,
    /// Completion target (read-pointer value to wait for), in doorbell units.
    target: u64,
    /// Doorbell/wptr unit toggle (debug): false = dwords (default), true = bytes.
    db_bytes: bool,
    _ring: KfdAllocatedBuffer,
    _wptr: KfdAllocatedBuffer,
    _rptr: KfdAllocatedBuffer,
}

const SDMA_OP_COPY: u32 = 1;
const SDMA_SUBOP_COPY_LINEAR: u32 = 0;
const SDMA_OP_FENCE: u32 = 5;
const SDMA_OP_POLL_REGMEM: u32 = 8;
/// Bytes per linear-copy packet — kept well within the count field, large bursts.
/// Tunable for experiments (the gfx9 linear-copy COUNT field is wide; bigger
/// descriptors mean fewer packets and can drive the engines harder).
fn sdma_copy_chunk() -> usize {
    std::env::var("MAINARCH_SDMA_CHUNK_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&m| m >= 1)
        .map(|m| m * 1024 * 1024)
        .unwrap_or(2 * 1024 * 1024)
}

impl SdmaQueue {
    /// SDMA queue on a general copy engine.
    pub fn new(kfd: &Kfd, node_id: u32) -> Result<Self> {
        Self::new_typed(kfd, node_id, mainarch_sys::KFD_IOC_QUEUE_TYPE_SDMA)
    }
    /// SDMA queue on an XGMI-dedicated copy engine (for peer-to-peer fan-out).
    pub fn new_xgmi(kfd: &Kfd, node_id: u32) -> Result<Self> {
        Self::new_typed(kfd, node_id, mainarch_sys::KFD_IOC_QUEUE_TYPE_SDMA_XGMI)
    }
    pub fn new_typed(kfd: &Kfd, node_id: u32, queue_type: u32) -> Result<Self> {
        let debug = std::env::var_os("MAINARCH_SDMA_DEBUG").is_some();
        let page = page_size();
        let gpu_id = kfd.ensure_vm(node_id)?;
        let ring_bytes = (1usize << 20).max(page); // 1 MiB dword packet ring
        let userptr_rw = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
        let gtt_rw = mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_GTT
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        let try_buf = |len: usize| -> Result<KfdAllocatedBuffer> {
            KfdAllocatedBuffer::new(kfd, gpu_id, len, userptr_rw)
                .or_else(|_| KfdAllocatedBuffer::new(kfd, gpu_id, len, gtt_rw))
        };
        let ring = try_buf(ring_bytes)?;
        let wptr_buf = try_buf(page)?;
        let rptr_buf = try_buf(page)?;
        unsafe {
            std::ptr::write_volatile(wptr_buf.host_ptr() as *mut u64, 0);
            std::ptr::write_volatile(rptr_buf.host_ptr() as *mut u64, 0);
        }
        let fd = kfd.fd.as_raw_fd();
        let mut args = mainarch_sys::CreateQueueArgsCompat {
            ring_base_address: ring.ptr(),
            write_pointer_address: wptr_buf.ptr(),
            read_pointer_address: rptr_buf.ptr(),
            doorbell_offset: 0,
            ring_size: ring_bytes as u32,
            gpu_id,
            queue_type,
            queue_percentage: mainarch_sys::KFD_IOC_QUEUE_MAX_PERCENTAGE,
            queue_priority: 7,
            queue_id: 0,
            eop_buffer_address: 0,
            eop_buffer_size: 0,
            ctx_save_restore_address: 0,
            ctx_save_restore_size: 0,
            ctl_stack_size: 0,
        };
        if let Err(compat_err) = unsafe { mainarch_sys::ioctl_create_queue_compat(fd, &mut args) } {
            let mut na = mainarch_sys::CreateQueueArgs {
                ring_base_address: args.ring_base_address,
                write_pointer_address: args.write_pointer_address,
                read_pointer_address: args.read_pointer_address,
                doorbell_offset: 0,
                ring_size: args.ring_size,
                gpu_id,
                queue_type: args.queue_type,
                queue_percentage: args.queue_percentage,
                queue_priority: args.queue_priority,
                queue_id: 0,
                eop_buffer_address: 0,
                eop_buffer_size: 0,
                ctx_save_restore_address: 0,
                ctx_save_restore_size: 0,
                ctl_stack_size: 0,
                sdma_engine_id: 0,
                pad: 0,
            };
            unsafe { mainarch_sys::ioctl_create_queue(fd, &mut na) }.map_err(|new_err| {
                anyhow!("SDMA create_queue failed (compat: {compat_err}, new: {new_err})")
            })?;
            args.doorbell_offset = na.doorbell_offset;
            args.queue_id = na.queue_id;
        }
        if debug {
            eprintln!(
                "mainarch: SDMA queue_id={} doorbell_offset=0x{:x} ring_va=0x{:x}",
                args.queue_id,
                args.doorbell_offset,
                ring.ptr()
            );
        }
        let (doorbell, db_base, db_len) = {
            let mut found = None;
            for &slice in &[page, 0x2000usize, 0x4000usize] {
                let page_off = (args.doorbell_offset as usize) & (slice - 1);
                let base = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        slice,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        fd,
                        (args.doorbell_offset & !((slice as u64) - 1)) as libc::off_t,
                    )
                };
                if base != libc::MAP_FAILED {
                    found = Some((
                        unsafe { (base as *mut u8).add(page_off) as *mut u64 },
                        base,
                        slice,
                    ));
                    break;
                }
            }
            found.ok_or_else(|| anyhow!("SDMA doorbell mmap failed"))?
        };
        Ok(Self {
            kfd_fd: fd,
            queue_id: args.queue_id,
            ring_host: ring.host_ptr() as *mut u32,
            ring_dwords: ring_bytes / 4,
            wptr_host: wptr_buf.host_ptr() as *mut u64,
            rptr_host: rptr_buf.host_ptr() as *const u64,
            doorbell,
            doorbell_map_base: db_base,
            doorbell_map_len: db_len,
            wptr: 0,
            target: 0,
            // gfx9 SDMA wptr/doorbell/rptr are byte units (verified: dword units
            // only advanced one packet). MAINARCH_SDMA_DWORDS forces the old path.
            db_bytes: std::env::var_os("MAINARCH_SDMA_DWORDS").is_none(),
            _ring: ring,
            _wptr: wptr_buf,
            _rptr: rptr_buf,
        })
    }

    /// Copy `bytes` from `src_va` to `dst_va` via SDMA linear-copy packets and
    /// block until the engine drains (read ptr reaches write ptr) or `timeout`.
    /// `dst_va` may be a peer-mapped VA for XGMI peer-to-peer.
    pub fn copy(
        &mut self,
        src_va: u64,
        dst_va: u64,
        bytes: usize,
        timeout: std::time::Duration,
    ) -> Result<()> {
        self.copy_async(src_va, dst_va, bytes);
        self.wait(timeout)
    }

    /// Write one packet's dwords into the ring (no doorbell), advancing wptr.
    fn push_dwords(&mut self, dws: &[u32]) {
        for &w in dws {
            let slot = (self.wptr % self.ring_dwords as u64) as usize;
            unsafe { std::ptr::write_volatile(self.ring_host.add(slot), w) };
            self.wptr += 1;
        }
    }

    /// Push linear-copy packets for `bytes` into the ring WITHOUT ringing the
    /// doorbell — for building a multi-packet program. Pair with `commit`.
    pub fn push_copy(&mut self, src_va: u64, dst_va: u64, bytes: usize) {
        let chunk = sdma_copy_chunk();
        let (mut off, mut remaining) = (0usize, bytes);
        while remaining > 0 {
            let n = remaining.min(chunk);
            let s = src_va + off as u64;
            let d = dst_va + off as u64;
            self.push_dwords(&[
                SDMA_OP_COPY | (SDMA_SUBOP_COPY_LINEAR << 8),
                (n as u32) - 1,
                0,
                (s & 0xffff_ffff) as u32,
                (s >> 32) as u32,
                (d & 0xffff_ffff) as u32,
                (d >> 32) as u32,
            ]);
            remaining -= n;
            off += n;
        }
    }

    /// Push an SDMA FENCE: when the engine reaches this packet it writes the
    /// 32-bit `value` to `addr_va` (a dword-aligned, GPU-visible VA). Lets the
    /// SDMA engine signal a memory semaphore a kernel can poll device-side.
    pub fn push_fence(&mut self, addr_va: u64, value: u32) {
        self.push_dwords(&[
            SDMA_OP_FENCE,
            (addr_va & 0xffff_fffc) as u32,
            (addr_va >> 32) as u32,
            value,
        ]);
    }

    /// Push an SDMA POLL_REGMEM (memory poll): stall the engine until
    /// `(mem[addr_va] & mask) == value`. Lets the SDMA engine wait device-side
    /// on a kernel-set release semaphore — no host round-trip. Has a finite
    /// retry budget (it proceeds rather than wedging if never released).
    pub fn push_poll_eq(&mut self, addr_va: u64, value: u32, mask: u32) {
        const FUNC_EQ: u32 = 3; // WAIT_REG_MEM compare: ==
        let header = SDMA_OP_POLL_REGMEM | (FUNC_EQ << 28) | (1u32 << 31); // mem_poll=1
                                                                           // dw5: retry_count(16:27) | interval(0:15). Short interval = low wake
                                                                           // latency after release; max retries keep the total wait long (~ms) so it
                                                                           // never gives up before the kernel signals. Tunable for experiments.
        let interval = std::env::var("MAINARCH_SDMA_POLL_INTERVAL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0x40)
            & 0xffff;
        let dw5 = (0xfffu32 << 16) | interval;
        self.push_dwords(&[
            header,
            (addr_va & 0xffff_fffc) as u32,
            (addr_va >> 32) as u32,
            value,
            mask,
            dw5,
        ]);
    }

    /// Ring the doorbell for everything pushed since the last commit, and set
    /// the completion target. Pair with `wait`.
    pub fn commit(&mut self) {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        let db_val = if self.db_bytes {
            self.wptr * 4
        } else {
            self.wptr
        };
        self.target = db_val;
        unsafe {
            std::ptr::write_volatile(self.wptr_host, db_val);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(self.doorbell, db_val);
        }
    }

    /// Enqueue a copy and ring the doorbell WITHOUT waiting, so multiple queues
    /// (engines) can run concurrently. Call `wait` to drain.
    pub fn copy_async(&mut self, src_va: u64, dst_va: u64, bytes: usize) {
        self.push_copy(src_va, dst_va, bytes);
        self.commit();
    }

    /// Block until the engine has drained all enqueued copies (read ptr reaches
    /// the write ptr) or `timeout` elapses.
    pub fn wait(&self, timeout: std::time::Duration) -> Result<()> {
        let start = std::time::Instant::now();
        loop {
            let rptr = unsafe { std::ptr::read_volatile(self.rptr_host) };
            if rptr >= self.target {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(anyhow!(
                    "SDMA copy timeout (rptr={rptr} target={})",
                    self.target
                ));
            }
            std::hint::spin_loop();
        }
    }
}

impl Drop for SdmaQueue {
    fn drop(&mut self) {
        if !self.doorbell_map_base.is_null() {
            unsafe {
                let _ = libc::munmap(self.doorbell_map_base, self.doorbell_map_len);
            }
        }
        let mut args = mainarch_sys::DestroyQueueArgs {
            queue_id: self.queue_id,
            pad: 0,
        };
        unsafe {
            let _ = mainarch_sys::ioctl_destroy_queue(self.kfd_fd, &mut args);
        }
    }
}

/// A GPU-visible allocation with a process VA region we own.
///
/// Two shapes, mirroring libhsakmt's fmm:
/// - **userptr**: anonymous RW pages registered with KFD; CPU VA == GPU VA.
/// - **BO (GTT/VRAM)**: we reserve a VA range with `PROT_NONE`, hand it to the
///   alloc ioctl as the GPU VA, then (for host-visible heaps) CPU-map the BO
///   over the reservation via the render node at the returned mmap offset.
#[derive(Debug)]
struct KfdAllocatedBuffer {
    region: CpuMappedAllocation,
    kfd_fd: i32,
    handle: u64,
    gpu_id: u32,
    flags: u32,
}

impl KfdAllocatedBuffer {
    fn new(kfd: &Kfd, gpu_id: u32, len: usize, flags: u32) -> Result<Self> {
        let page = page_size();
        let len = len.max(1).div_ceil(page) * page;
        let use_userptr = flags & mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_USERPTR != 0;
        let host_visible = flags
            & (mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_GTT
                | mainarch_sys::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
            != 0;

        let region = if use_userptr {
            CpuMappedAllocation::new_rw(len)?
        } else {
            CpuMappedAllocation::new_reservation(len)?
        };
        let va = region.ptr();

        let (handle, mmap_offset) = if use_userptr {
            // For userptrs mmap_offset carries the CPU address into the kernel.
            kfd.allocate_memory_with_offset(gpu_id, len, flags, va, va)?
        } else {
            kfd.allocate_memory_with_offset(gpu_id, len, flags, va, 0)?
        };

        let mut buf = Self {
            region,
            kfd_fd: kfd.fd.as_raw_fd(),
            handle,
            gpu_id,
            flags,
        };

        if !use_userptr && host_visible {
            let render_fd = kfd.render_raw_fd(gpu_id)?;
            let mapped = unsafe {
                libc::mmap(
                    va as *mut c_void,
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    render_fd,
                    mmap_offset as libc::off_t,
                )
            };
            if mapped == libc::MAP_FAILED {
                // buf's Drop frees the handle and the reservation.
                return Err(anyhow!(
                    "CPU-mapping BO over reserved VA failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }

        kfd.map_memory_to_gpus(handle, std::slice::from_ref(&gpu_id))?;
        buf.handle = handle;
        register_gpu_allocation(GpuAllocationRecord::new(
            buf.ptr(),
            buf.len(),
            gpu_id,
            flags,
        ));
        Ok(buf)
    }

    fn ptr(&self) -> u64 {
        self.region.ptr()
    }

    fn len(&self) -> usize {
        self.region.len
    }

    fn flags(&self) -> u32 {
        self.flags
    }

    /// CPU-accessible only for userptr/host-visible allocations; for those the
    /// CPU and GPU observe the same VA.
    fn host_ptr(&self) -> *mut u8 {
        self.region.ptr() as *mut u8
    }
}

impl Drop for KfdAllocatedBuffer {
    fn drop(&mut self) {
        let mut mapped_gpu_ids =
            unregister_gpu_allocation(self.ptr()).unwrap_or_else(|| vec![self.gpu_id]);
        if !mapped_gpu_ids.iter().any(|&gpu_id| gpu_id == self.gpu_id) {
            mapped_gpu_ids.push(self.gpu_id);
        }
        if self.handle != 0 && self.kfd_fd >= 0 {
            let mut unmap_args = mainarch_sys::UnmapMemoryFromGpuArgs {
                handle: self.handle,
                device_ids_array_ptr: mapped_gpu_ids.as_mut_ptr() as usize as u64,
                n_devices: mapped_gpu_ids.len() as u32,
                n_success: 0,
            };
            let _ =
                unsafe { mainarch_sys::ioctl_unmap_memory_from_gpu(self.kfd_fd, &mut unmap_args) };
            let mut free_args = mainarch_sys::FreeMemoryOfGpuArgs {
                handle: self.handle,
            };
            let _ = unsafe { mainarch_sys::ioctl_free_memory_of_gpu(self.kfd_fd, &mut free_args) };
            self.handle = 0;
        }
        // self.region munmaps the VA range after the BO is gone.
    }
}

/// Kernel `gpu_id` for a topology node — the hashed device id every KFD ioctl
/// expects (distinct from the sysfs node index).
fn gpu_id_for_node(node_id: u32) -> Result<u32> {
    let path = Path::new(KFD_TOPOLOGY)
        .join(node_id.to_string())
        .join("gpu_id");
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let id: u32 = text
        .trim()
        .parse()
        .with_context(|| format!("parsing gpu_id from {}", path.display()))?;
    if id == 0 {
        return Err(anyhow!("node {node_id} is not a GPU (gpu_id 0)"));
    }
    Ok(id)
}

/// Open the render node for a KFD GPU node.
fn render_fd_for_node(node_id: u32) -> Result<OwnedFd> {
    let props = node_properties(node_id)?;
    let render_minor = props
        .get("drm_render_minor")
        .copied()
        .context("GPU node is missing drm_render_minor in topology properties")?;

    if render_minor == 0 {
        return Err(anyhow!(
            "GPU node {node_id} advertises invalid render minor 0"
        ));
    }

    let path = format!("/dev/dri/renderD{render_minor}");
    let fd = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path.clone())
        .with_context(|| format!("opening render fd {path}"))?;

    Ok(OwnedFd::from(fd))
}

#[derive(Debug)]
struct CpuMappedAllocation {
    ptr: *mut c_void,
    len: usize,
}

impl CpuMappedAllocation {
    fn new_with_prot(len: usize, prot: libc::c_int, extra_flags: libc::c_int) -> Result<Self> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                prot,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | extra_flags,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(anyhow!(
                "anonymous mmap of {len} bytes failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { ptr, len })
    }

    /// Readable/writable anonymous memory (userptr backing).
    fn new_rw(len: usize) -> Result<Self> {
        Self::new_with_prot(len, libc::PROT_READ | libc::PROT_WRITE, 0)
    }

    /// Address-space-only reservation used as a GPU VA range; CPU access (if
    /// any) is established later with `MAP_FIXED` over this range.
    fn new_reservation(len: usize) -> Result<Self> {
        Self::new_with_prot(len, libc::PROT_NONE, libc::MAP_NORESERVE)
    }

    fn ptr(&self) -> u64 {
        self.ptr as u64
    }
}

#[derive(Debug)]
struct QueueCreateSizes {
    ctx_save_restore_size: u32,
    ctl_stack_size: u32,
    total_cwsr_buffer_size: u64,
}

fn node_properties(gpu_node_id: u32) -> Result<HashMap<String, u64>> {
    let path = Path::new(KFD_TOPOLOGY)
        .join(gpu_node_id.to_string())
        .join("properties");
    let props = read_props(&path);
    if props.is_empty() {
        return Err(anyhow!(
            "missing or unreadable topology properties for gpu node {gpu_node_id}"
        ));
    }
    Ok(props)
}

fn compute_queue_sizes(props: &HashMap<String, u64>, page_size: u64) -> Result<QueueCreateSizes> {
    let simd_count = props.get("simd_count").copied().unwrap_or(0);
    let simd_per_cu = props.get("simd_per_cu").copied().unwrap_or(0).max(1);
    let num_xcc = props.get("num_xcc").copied().unwrap_or(1).max(1);
    let gfx_target_version = props.get("gfx_target_version").copied().unwrap_or(0);
    let array_count = props.get("array_count").copied().unwrap_or(0);
    let simd_arrays_per_engine = props
        .get("simd_arrays_per_engine")
        .copied()
        .unwrap_or(1)
        .max(1);

    if simd_count == 0 {
        return Err(anyhow!("cannot compute queue sizes: simd_count is zero"));
    }

    let lds_size_in_kb = props.get("lds_size_in_kb").copied().unwrap_or(0);
    let cu_num = simd_count / simd_per_cu / num_xcc;
    let wave_num = if gfx_target_version < 100100 {
        min(
            cu_num.saturating_mul(40),
            array_count
                .saturating_div(simd_arrays_per_engine)
                .saturating_mul(512),
        )
    } else {
        cu_num.saturating_mul(32)
    };

    let vgpr_size_per_cu: u64 = match gfx_target_version {
        90402 | 90010 | 90008 | 90500 => 0x80000,
        110000 | 110001 | 110501 | 120000 | 120001 => 0x60000,
        _ => 0x40000,
    };

    let lds_size_per_cu = if gfx_target_version == 90500 {
        lds_size_in_kb.saturating_mul(1024)
    } else {
        0x10000u64
    };
    let wg_context_data_size_per_cu = vgpr_size_per_cu
        .saturating_add(0x4000u64)
        .saturating_add(lds_size_per_cu)
        .saturating_add(0x1000u64);
    let wg_data_size = align_up(
        cu_num
            .saturating_mul(wg_context_data_size_per_cu)
            .saturating_mul(1),
        page_size,
    );

    let wave_stack_bytes_per_wave = if gfx_target_version >= 100100 { 12 } else { 8 };
    let mut ctl_stack_size = align_up(
        40u64.saturating_add(
            wave_num
                .saturating_mul(wave_stack_bytes_per_wave)
                .saturating_add(8),
        ),
        page_size,
    );
    if (gfx_target_version / 10000) * 10000 == 100000 {
        ctl_stack_size = min(ctl_stack_size, 0x7000);
    }

    let debug_memory_size = align_up(wave_num.saturating_mul(32), 64);
    let ctx_save_restore_size = ctl_stack_size.saturating_add(wg_data_size);
    let num_xcc_u64 = num_xcc.max(1);
    let total_cwsr_buffer_size = align_up(
        ctx_save_restore_size
            .saturating_add(debug_memory_size)
            .saturating_mul(num_xcc_u64),
        page_size,
    );

    Ok(QueueCreateSizes {
        ctx_save_restore_size: u32::try_from(ctx_save_restore_size)
            .context("computed ctx_save_restore_size does not fit u32")?,
        ctl_stack_size: u32::try_from(ctl_stack_size)
            .context("computed ctl_stack_size does not fit u32")?,
        total_cwsr_buffer_size,
    })
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

impl Drop for CpuMappedAllocation {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            // best effort; nothing to do if this fails.
            let _ = unsafe { libc::munmap(self.ptr, self.len) };
        }
    }
}

fn page_size() -> usize {
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps <= 0 {
        4096
    } else {
        ps as usize
    }
}

/// Enumerate GPU nodes from the kfd sysfs topology (CPU-only nodes are skipped).
pub fn enumerate_gpus() -> Result<Vec<GpuNode>> {
    let base = Path::new(KFD_TOPOLOGY);
    let mut out = Vec::new();
    if !base.exists() {
        return Ok(out);
    }
    let mut ids: Vec<u32> = fs::read_dir(base)
        .with_context(|| format!("reading {KFD_TOPOLOGY}"))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_string_lossy().parse::<u32>().ok())
        .collect();
    ids.sort_unstable();

    for id in ids {
        let dir = base.join(id.to_string());
        let props = read_props(&dir.join("properties"));
        let simd_count = props.get("simd_count").copied().unwrap_or(0) as u32;
        if simd_count == 0 {
            continue; // CPU / non-GPU node
        }
        let name = fs::read_to_string(dir.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        out.push(GpuNode {
            node_id: id,
            name: if name.is_empty() {
                format!("node{id}")
            } else {
                name
            },
            gfx_target_version: props.get("gfx_target_version").copied().unwrap_or(0),
            simd_count,
            vram_bytes: vram_bytes(&dir),
        });
    }
    Ok(out)
}

pub fn enumerate_topology() -> Result<TopologyGraph> {
    let nodes = enumerate_gpus()?;
    let gpu_ids: HashSet<u32> = nodes.iter().map(|n| n.node_id).collect();
    let mut links = Vec::new();

    let base = Path::new(KFD_TOPOLOGY);
    for node in &nodes {
        let io_dir = base.join(node.node_id.to_string()).join("io_links");
        if !io_dir.exists() {
            continue;
        }

        let mut io_links = collect_links_in_dir(&io_dir)?;
        io_links.retain(|l| gpu_ids.contains(&l.from_node) && gpu_ids.contains(&l.to_node));
        links.append(&mut io_links);
    }

    links.sort_unstable_by(|a, b| {
        a.from_node
            .cmp(&b.from_node)
            .then(a.to_node.cmp(&b.to_node))
            .then(a.raw_type.cmp(&b.raw_type))
    });

    Ok(TopologyGraph { nodes, links })
}

fn collect_links_in_dir(dir: &Path) -> Result<Vec<TopologyLink>> {
    let mut links = Vec::new();
    for child in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = child.with_context(|| format!("reading link entry in {}", dir.display()))?;
        let props = read_props(&entry.path().join("properties"));
        let from_node = props.get("node_from").copied().unwrap_or(0);
        let to_node = props.get("node_to").copied().unwrap_or(0);
        if from_node == to_node {
            continue;
        }
        let raw_type = props.get("type").copied().unwrap_or(0) as u32;
        links.push(TopologyLink {
            from_node: from_node as u32,
            to_node: to_node as u32,
            raw_type,
            kind: TopologyLinkType::from_raw(raw_type),
            weight: *props.get("weight").unwrap_or(&0) as u32,
            min_latency: *props.get("min_latency").unwrap_or(&0) as u32,
            max_latency: *props.get("max_latency").unwrap_or(&0) as u32,
            min_bandwidth: *props.get("min_bandwidth").unwrap_or(&0) as u32,
            max_bandwidth: *props.get("max_bandwidth").unwrap_or(&0) as u32,
            recommended_transfer_size: *props.get("recommended_transfer_size").unwrap_or(&0) as u32,
            flags: *props.get("flags").unwrap_or(&0) as u32,
        });
    }
    Ok(links)
}

/// Parse a sysfs `properties` file (one `key value` token pair per line).
fn read_props(path: &Path) -> HashMap<String, u64> {
    let mut m = HashMap::new();
    if let Ok(text) = fs::read_to_string(path) {
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                if let Ok(n) = v.parse::<u64>() {
                    m.insert(k.to_string(), n);
                }
            }
        }
    }
    m
}

/// Sum framebuffer (VRAM) memory banks from `<node>/mem_banks/*/properties`.
/// `heap_type` 1 = FB_PUBLIC, 2 = FB_PRIVATE.
fn vram_bytes(node_dir: &Path) -> u128 {
    let mut total: u128 = 0;
    if let Ok(entries) = fs::read_dir(node_dir.join("mem_banks")) {
        for e in entries.flatten() {
            let props = read_props(&e.path().join("properties"));
            match props.get("heap_type").copied() {
                Some(1) | Some(2) => {
                    total += props.get("size_in_bytes").copied().unwrap_or(0) as u128
                }
                _ => {}
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_vram_aggregates_and_link_type_names() {
        assert_eq!(TopologyLinkType::from_raw(11), TopologyLinkType::Xgmi);
        assert_eq!(TopologyLinkType::from_raw(2), TopologyLinkType::PciExpress);
        assert_eq!(TopologyLinkType::from_raw(0), TopologyLinkType::Undefined);
        assert_eq!(
            TopologyLinkType::from_raw(999),
            TopologyLinkType::Other(999)
        );
        assert!(TopologyLinkType::Xgmi.is_xgmi_like());
        assert!(!TopologyLinkType::PciExpress.is_xgmi_like());
    }

    #[test]
    fn io_link_name_lookup() {
        assert_eq!(TopologyLinkType::Xgmi.name(), "xgmi");
        assert_eq!(TopologyLinkType::Other(999).name(), "other");
    }

    #[test]
    fn topology_enumeration_is_optional() {
        let gpus = enumerate_gpus().expect("enumerating gpus should not hard-fail");
        assert!(gpus.len() < usize::MAX); // non-panicking path on systems with no kfd nodes

        let topo = enumerate_topology().expect("topology enumeration should not hard-fail");
        assert_eq!(topo.nodes.len(), gpus.len());
    }

    #[test]
    fn mapped_allocation_roundtrip() {
        let alloc = CpuMappedAllocation::new_rw(4096).expect("mmap allocation should work");
        assert!(alloc.ptr() > 0);

        let reservation =
            CpuMappedAllocation::new_reservation(1 << 20).expect("VA reservation should work");
        assert!(reservation.ptr() > 0);
    }

    fn live_aql_reservation_input() -> KfdQueueLiveAqlReservationInput {
        KfdQueueLiveAqlReservationInput {
            operands_probe_version: 1,
            packet_id: 18,
            read_index: 15,
            packet_id_matches_host_snapshot: 1,
            read_index_matches_host_snapshot: 1,
            inflight_packets: 3,
            capacity_ok: 1,
            slot_index: 18,
            slot_offset: 1152,
            slot_va: 0x1000,
            slot_va_aligned64: 1,
            desired_write_index: 19,
            packet_count: 1,
            doorbell_packet_id: 18,
            doorbell_matches_last_packet: 1,
            publish_low32: 0x0001_1502,
            header_release_width_bits: 32,
            live_low32: 0,
            valid_header_not_stored: 1,
            fetch_add_not_performed: 1,
            doorbell_not_written: 1,
            capacity_formula_ok: 1,
            slot_formula_ok: 1,
            metadata_ready_dependency: 1,
            non_consuming_contract: 1,
            observed_ready: 1,
            expected_capacity_ok: true,
            expected_slot_aligned64: true,
            expected_doorbell_matches_last_packet: true,
            expected_valid_header_not_stored: true,
            expected_metadata_ready_dependency: true,
            expected_slot_formula_ok: true,
        }
    }

    #[test]
    fn live_aql_reservation_proof_preserves_ready_gate() {
        let proof = live_aql_reservation_input().proof();

        assert_eq!(proof.packet_id, 18);
        assert_eq!(proof.read_index, 15);
        assert_eq!(proof.packet_id_matches_host_snapshot, 1);
        assert_eq!(proof.read_index_matches_host_snapshot, 1);
        assert_eq!(proof.capacity_ok, 1);
        assert_eq!(proof.slot_va_aligned64, 1);
        assert_eq!(proof.doorbell_matches_last_packet, 1);
        assert_eq!(proof.valid_header_not_stored, 1);
        assert_eq!(proof.metadata_ready_dependency, 1);
        assert_eq!(proof.slot_formula_ok, 1);
        assert_eq!(proof.fetch_add_not_performed, 1);
        assert_eq!(proof.doorbell_not_written, 1);
        assert_eq!(proof.non_consuming_contract, 1);
        assert_eq!(proof.observed_ready, 1);
        assert_eq!(proof.ready, 1);
        assert!(proof.validate_ready().passed);
    }

    #[test]
    fn live_aql_reservation_validation_rejects_metadata_or_slot_miss() {
        let mut metadata_miss = live_aql_reservation_input();
        metadata_miss.expected_metadata_ready_dependency = false;
        let metadata_miss_validation = metadata_miss.proof().validate_ready();
        assert!(!metadata_miss_validation.expected_metadata_ready_dependency);
        assert!(!metadata_miss_validation.ready);
        assert!(!metadata_miss_validation.passed);

        let mut slot_miss = live_aql_reservation_input();
        slot_miss.expected_slot_formula_ok = false;
        let slot_miss_validation = slot_miss.proof().validate_ready();
        assert!(!slot_miss_validation.expected_slot_formula_ok);
        assert!(!slot_miss_validation.ready);
        assert!(!slot_miss_validation.passed);
    }

    fn live_aql_reserve_before_stage_input() -> KfdQueueLiveAqlReserveBeforeStageInput {
        KfdQueueLiveAqlReserveBeforeStageInput {
            probe_version: 1,
            staged_packet_id: 17,
            reserved_packet_id: 18,
            staged_slot_va: 0x0fc0,
            reserved_slot_va: 0x1000,
            staged_slot_offset: 1088,
            reserved_slot_offset: 1152,
            same_packet_id: 0,
            same_slot: 0,
            old_payload_write_ready: 1,
            old_payload_publishable: 0,
            must_restage_after_reserve: 1,
            publish_blocked_until_restage: 1,
            old_slot_still_invalid: 1,
            reservation_ready_dependency: 1,
            valid_header_not_stored: 1,
            publish_low32: 0x0001_1502,
            live_low32: 0,
            slot_progress_observed: 0,
            desired_write_index: 19,
            doorbell_packet_id: 18,
            capacity_ok: 1,
            slot_formula_ok: 1,
            fetch_add_not_performed: 1,
            reserved_slot_not_written: 1,
            header_not_published: 1,
            doorbell_not_written: 1,
            reserve_first_contract: 1,
            reserved_slot_stage_required: 1,
            non_consuming_contract: 1,
            sequence_ready: 1,
            observed_ready: 1,
            expected_same_packet_id: false,
            expected_same_slot: false,
            expected_old_payload_publishable: false,
            expected_must_restage_after_reserve: true,
            expected_reservation_ready_dependency: true,
            expected_capacity_ok: true,
            expected_slot_formula_ok: true,
        }
    }

    #[test]
    fn live_aql_reserve_before_stage_proof_preserves_ready_gate() {
        let proof = live_aql_reserve_before_stage_input().proof();

        assert_eq!(proof.probe_version, 1);
        assert_eq!(proof.staged_packet_id, 17);
        assert_eq!(proof.reserved_packet_id, 18);
        assert_eq!(proof.staged_slot_va, 0x0fc0);
        assert_eq!(proof.reserved_slot_va, 0x1000);
        assert_eq!(proof.same_packet_id, 0);
        assert_eq!(proof.same_slot, 0);
        assert_eq!(proof.old_payload_write_ready, 1);
        assert_eq!(proof.old_payload_publishable, 0);
        assert_eq!(proof.must_restage_after_reserve, 1);
        assert_eq!(proof.publish_blocked_until_restage, 1);
        assert_eq!(proof.old_slot_still_invalid, 1);
        assert_eq!(proof.reservation_ready_dependency, 1);
        assert_eq!(proof.valid_header_not_stored, 1);
        assert_eq!(proof.fetch_add_not_performed, 1);
        assert_eq!(proof.reserved_slot_not_written, 1);
        assert_eq!(proof.header_not_published, 1);
        assert_eq!(proof.doorbell_not_written, 1);
        assert_eq!(proof.reserve_first_contract, 1);
        assert_eq!(proof.reserved_slot_stage_required, 1);
        assert_eq!(proof.non_consuming_contract, 1);
        assert_eq!(proof.sequence_ready, 1);
        assert_eq!(proof.same_packet_id_matches_expected, 1);
        assert_eq!(proof.same_slot_matches_expected, 1);
        assert_eq!(proof.old_payload_publishable_matches_expected, 1);
        assert_eq!(proof.must_restage_matches_expected, 1);
        assert_eq!(proof.ready, 1);
        assert!(proof.validate_ready().passed);
    }

    #[test]
    fn live_aql_reserve_before_stage_validation_rejects_mismatch_or_mutation() {
        let mut mismatch = live_aql_reserve_before_stage_input();
        mismatch.same_slot = 1;
        let mismatch_validation = mismatch.proof().validate_ready();
        assert!(!mismatch_validation.same_slot_matches_expected);
        assert!(!mismatch_validation.ready);
        assert!(!mismatch_validation.passed);

        let mut mutation = live_aql_reserve_before_stage_input();
        mutation.fetch_add_not_performed = 0;
        let mutation_validation = mutation.proof().validate_ready();
        assert!(!mutation_validation.no_fetch_add);
        assert!(!mutation_validation.ready);
        assert!(!mutation_validation.passed);
    }

    fn live_aql_reserve_first_restage_input() -> KfdQueueLiveAqlReserveFirstRestageInput {
        KfdQueueLiveAqlReserveFirstRestageInput {
            probe_version: 1,
            target_packet_id: 18,
            target_slot_va: 0x1000,
            target_slot_offset: 1152,
            reservation_packet_id: 18,
            reservation_slot_va: 0x1000,
            reservation_slot_offset: 1152,
            target_matches_reservation: 1,
            old_packet_id: 17,
            old_slot_va: 0x0fc0,
            old_slot_bypassed: 1,
            payload_inputs_ready: 1,
            publish_low32: 0x0001_1502,
            live_low32: 0,
            valid_header_store_pending: 1,
            reserved_slot_write_pending: 1,
            write_index_fetch_add_pending: 1,
            doorbell_pending: 1,
            release_header_after_payload_contract: 1,
            reserve_before_payload_contract: 1,
            doorbell_after_header_contract: 1,
            no_live_queue_mutation_contract: 1,
            observed_plan_ready: 1,
            capacity_ok: 1,
            slot_formula_ok: 1,
            desired_write_index: 19,
            doorbell_packet_id: 18,
            packet_bytes: 64,
            ring_slots: 1024,
            slot_mask: 1023,
            publish_blocked_before_restage: 1,
            observed_ready: 1,
            expected_must_restage: true,
            expected_target_matches_reservation: true,
            expected_payload_inputs_ready: true,
            expected_capacity_ok: true,
            expected_slot_formula_ok: true,
        }
    }

    #[test]
    fn live_aql_reserve_first_restage_proof_preserves_ready_gate() {
        let proof = live_aql_reserve_first_restage_input().proof();

        assert_eq!(proof.probe_version, 1);
        assert_eq!(proof.target_packet_id, 18);
        assert_eq!(proof.target_slot_va, 0x1000);
        assert_eq!(proof.reservation_packet_id, 18);
        assert_eq!(proof.reservation_slot_va, 0x1000);
        assert_eq!(proof.target_matches_reservation, 1);
        assert_eq!(proof.old_packet_id, 17);
        assert_eq!(proof.old_slot_va, 0x0fc0);
        assert_eq!(proof.old_slot_bypassed, 1);
        assert_eq!(proof.payload_inputs_ready, 1);
        assert_eq!(proof.publish_low32, 0x0001_1502);
        assert_eq!(proof.live_low32, 0);
        assert_eq!(proof.valid_header_store_pending, 1);
        assert_eq!(proof.reserved_slot_write_pending, 1);
        assert_eq!(proof.write_index_fetch_add_pending, 1);
        assert_eq!(proof.doorbell_pending, 1);
        assert_eq!(proof.release_header_after_payload_contract, 1);
        assert_eq!(proof.reserve_before_payload_contract, 1);
        assert_eq!(proof.doorbell_after_header_contract, 1);
        assert_eq!(proof.no_live_queue_mutation_contract, 1);
        assert_eq!(proof.observed_plan_ready, 1);
        assert_eq!(proof.capacity_ok, 1);
        assert_eq!(proof.slot_formula_ok, 1);
        assert_eq!(proof.packet_bytes, 64);
        assert_eq!(proof.ring_slots, 1024);
        assert_eq!(proof.slot_mask, 1023);
        assert_eq!(proof.publish_blocked_before_restage, 1);
        assert_eq!(proof.observed_ready, 1);
        assert_eq!(proof.expected_must_restage, 1);
        assert_eq!(proof.expected_target_matches_reservation, 1);
        assert_eq!(proof.expected_payload_inputs_ready, 1);
        assert_eq!(proof.expected_capacity_ok, 1);
        assert_eq!(proof.expected_slot_formula_ok, 1);
        assert_eq!(proof.ready, 1);
        assert!(proof.validate_ready().passed);
    }

    #[test]
    fn live_aql_reserve_first_restage_validation_rejects_payload_or_slot_miss() {
        let mut payload_miss = live_aql_reserve_first_restage_input();
        payload_miss.expected_payload_inputs_ready = false;
        let payload_miss_validation = payload_miss.proof().validate_ready();
        assert!(!payload_miss_validation.expected_payload_inputs_ready);
        assert!(!payload_miss_validation.ready);
        assert!(!payload_miss_validation.passed);

        let mut slot_miss = live_aql_reserve_first_restage_input();
        slot_miss.expected_slot_formula_ok = false;
        let slot_miss_validation = slot_miss.proof().validate_ready();
        assert!(!slot_miss_validation.expected_slot_formula_ok);
        assert!(!slot_miss_validation.ready);
        assert!(!slot_miss_validation.passed);

        let mut contract_miss = live_aql_reserve_first_restage_input();
        contract_miss.publish_blocked_before_restage = 0;
        let contract_miss_validation = contract_miss.proof().validate_ready();
        assert!(!contract_miss_validation.publish_blocked_before_restage);
        assert!(!contract_miss_validation.ready);
        assert!(!contract_miss_validation.passed);
    }

    fn live_aql_batch_reservation_plan_input() -> KfdQueueLiveAqlBatchReservationPlanInput {
        KfdQueueLiveAqlBatchReservationPlanInput {
            probe_version: 1,
            base_packet_id: 18,
            packet_count: 2,
            last_packet_id: 19,
            desired_write_index: 20,
            read_index: 15,
            inflight_packets: 3,
            capacity_ok: 1,
            slot0_va: 0x1000,
            slot1_va: 0x1040,
            slot0_offset: 1152,
            slot1_offset: 1216,
            slot0_index: 18,
            slot1_index: 19,
            slots_distinct: 1,
            slots_aligned64: 1,
            slot0_formula_ok: 1,
            slot1_formula_ok: 1,
            doorbell_packet_id: 19,
            doorbell_matches_last_packet: 1,
            single_doorbell_contract: 1,
            reserve_before_payload_contract: 1,
            payloads_before_headers_contract: 1,
            headers_before_doorbell_contract: 1,
            release_header_store_contract: 1,
            write_index_fetch_add_pending: 1,
            payload_writes_pending: 1,
            valid_headers_pending: 1,
            doorbell_pending: 1,
            no_live_queue_mutation_contract: 1,
            first_slot_matches_single_reservation: 1,
            observed_ready: 1,
            expected_restage_or_payload_ready: true,
            expected_capacity_ok: true,
            expected_slots_distinct: true,
            expected_slots_aligned64: true,
            expected_slot0_formula_ok: true,
            expected_slot1_formula_ok: true,
            expected_doorbell_matches_last_packet: true,
            expected_first_slot_matches_single_reservation: true,
        }
    }

    #[test]
    fn live_aql_batch_reservation_plan_proof_preserves_ready_gate() {
        let proof = live_aql_batch_reservation_plan_input().proof();

        assert_eq!(proof.probe_version, 1);
        assert_eq!(proof.base_packet_id, 18);
        assert_eq!(proof.packet_count, 2);
        assert_eq!(proof.last_packet_id, 19);
        assert_eq!(proof.desired_write_index, 20);
        assert_eq!(proof.capacity_ok, 1);
        assert_eq!(proof.slots_distinct, 1);
        assert_eq!(proof.slots_aligned64, 1);
        assert_eq!(proof.slot0_formula_ok, 1);
        assert_eq!(proof.slot1_formula_ok, 1);
        assert_eq!(proof.doorbell_matches_last_packet, 1);
        assert_eq!(proof.single_doorbell_contract, 1);
        assert_eq!(proof.reserve_before_payload_contract, 1);
        assert_eq!(proof.payloads_before_headers_contract, 1);
        assert_eq!(proof.headers_before_doorbell_contract, 1);
        assert_eq!(proof.release_header_store_contract, 1);
        assert_eq!(proof.write_index_fetch_add_pending, 1);
        assert_eq!(proof.payload_writes_pending, 1);
        assert_eq!(proof.valid_headers_pending, 1);
        assert_eq!(proof.doorbell_pending, 1);
        assert_eq!(proof.no_live_queue_mutation_contract, 1);
        assert_eq!(proof.first_slot_matches_single_reservation, 1);
        assert_eq!(proof.expected_restage_or_payload_ready, 1);
        assert_eq!(proof.expected_capacity_ok, 1);
        assert_eq!(proof.expected_slots_distinct, 1);
        assert_eq!(proof.expected_slots_aligned64, 1);
        assert_eq!(proof.expected_doorbell_matches_last_packet, 1);
        assert_eq!(proof.expected_first_slot_matches_single_reservation, 1);
        assert_eq!(proof.ready, 1);
        assert!(proof.validate_ready().passed);
    }

    #[test]
    fn live_aql_batch_reservation_plan_validation_rejects_capacity_or_slot_miss() {
        let mut capacity_miss = live_aql_batch_reservation_plan_input();
        capacity_miss.expected_capacity_ok = false;
        let capacity_miss_validation = capacity_miss.proof().validate_ready();
        assert!(!capacity_miss_validation.expected_capacity_ok);
        assert!(!capacity_miss_validation.ready);
        assert!(!capacity_miss_validation.passed);

        let mut slot_miss = live_aql_batch_reservation_plan_input();
        slot_miss.expected_first_slot_matches_single_reservation = false;
        let slot_miss_validation = slot_miss.proof().validate_ready();
        assert!(!slot_miss_validation.expected_first_slot_matches_single_reservation);
        assert!(!slot_miss_validation.ready);
        assert!(!slot_miss_validation.passed);

        let mut contract_miss = live_aql_batch_reservation_plan_input();
        contract_miss.headers_before_doorbell_contract = 0;
        let contract_miss_validation = contract_miss.proof().validate_ready();
        assert!(!contract_miss_validation.headers_before_doorbell_contract);
        assert!(!contract_miss_validation.ready);
        assert!(!contract_miss_validation.passed);

        let mut pending_miss = live_aql_batch_reservation_plan_input();
        pending_miss.doorbell_pending = 0;
        let pending_miss_validation = pending_miss.proof().validate_ready();
        assert!(!pending_miss_validation.doorbell_pending);
        assert!(!pending_miss_validation.ready);
        assert!(!pending_miss_validation.passed);
    }

    fn live_aql_materialized_packet_plan_input() -> KfdQueueLiveAqlMaterializedPacketPlanInput {
        KfdQueueLiveAqlMaterializedPacketPlanInput {
            probe_version: 1,
            packet0_packet_id: 18,
            packet1_packet_id: 19,
            packet0_slot_va: 0x1000,
            packet1_slot_va: 0x1040,
            packet0_word0: 0x0001_1502,
            packet0_word4_kernel_object: 0x2000,
            packet0_word5_kernarg_va: 0x3000,
            packet1_word0: 0x0001_1502,
            packet1_word4_kernel_object: 0x2000,
            packet1_word5_kernarg_va: 0x3000,
            packet0_words_match_host_template: 1,
            packet1_words_match_host_template: 1,
            payload_words_match_host_template: 1,
            header_words_match_host_template: 1,
            target_slots_match_batch_plan: 1,
            packet0_slot_offset: 1152,
            packet1_slot_offset: 1216,
            packet_bytes: 64,
            packet_count: 2,
            batch_plan_ready: 1,
            reserve_first_restage_ready: 1,
            payloads_before_headers_contract: 1,
            release_header_store_contract: 1,
            doorbell_pending: 1,
            no_live_queue_mutation_contract: 1,
            packet_plan_ready: 1,
            publish_low32: 0x0001_1502,
            packet0_low32: 0x0001_1502,
            aql_packet_image_ready: 1,
            expected_batch_ready: true,
            expected_reserve_restage_plan_ready: true,
            expected_aql_packet_image_ready: true,
            expected_target_slots_match_batch_plan: true,
        }
    }

    #[test]
    fn live_aql_materialized_packet_plan_proof_preserves_ready_gate() {
        let proof = live_aql_materialized_packet_plan_input().proof();

        assert_eq!(proof.probe_version, 1);
        assert_eq!(proof.packet0_packet_id, 18);
        assert_eq!(proof.packet1_packet_id, 19);
        assert_eq!(proof.packet0_slot_va, 0x1000);
        assert_eq!(proof.packet1_slot_va, 0x1040);
        assert_eq!(proof.packet0_words_match_host_template, 1);
        assert_eq!(proof.packet1_words_match_host_template, 1);
        assert_eq!(proof.payload_words_match_host_template, 1);
        assert_eq!(proof.header_words_match_host_template, 1);
        assert_eq!(proof.target_slots_match_batch_plan, 1);
        assert_eq!(proof.batch_plan_ready, 1);
        assert_eq!(proof.reserve_first_restage_ready, 1);
        assert_eq!(proof.payloads_before_headers_contract, 1);
        assert_eq!(proof.release_header_store_contract, 1);
        assert_eq!(proof.doorbell_pending, 1);
        assert_eq!(proof.no_live_queue_mutation_contract, 1);
        assert_eq!(proof.aql_packet_image_ready, 1);
        assert_eq!(proof.expected_batch_ready, 1);
        assert_eq!(proof.expected_reserve_restage_plan_ready, 1);
        assert_eq!(proof.expected_aql_packet_image_ready, 1);
        assert_eq!(proof.expected_target_slots_match_batch_plan, 1);
        assert_eq!(proof.ready, 1);
        assert!(proof.validate_ready().passed);
    }

    #[test]
    fn live_aql_materialized_packet_plan_validation_rejects_missing_source_or_slot_match() {
        let mut source_miss = live_aql_materialized_packet_plan_input();
        source_miss.expected_aql_packet_image_ready = false;
        let source_miss_validation = source_miss.proof().validate_ready();
        assert!(!source_miss_validation.expected_aql_packet_image_ready);
        assert!(!source_miss_validation.ready);
        assert!(!source_miss_validation.passed);

        let mut slot_miss = live_aql_materialized_packet_plan_input();
        slot_miss.expected_target_slots_match_batch_plan = false;
        let slot_miss_validation = slot_miss.proof().validate_ready();
        assert!(!slot_miss_validation.expected_target_slots_match_batch_plan);
        assert!(!slot_miss_validation.ready);
        assert!(!slot_miss_validation.passed);

        let mut packet_miss = live_aql_materialized_packet_plan_input();
        packet_miss.packet1_words_match_host_template = 0;
        let packet_miss_validation = packet_miss.proof().validate_ready();
        assert!(!packet_miss_validation.packet1_words_match_host_template);
        assert!(!packet_miss_validation.ready);
        assert!(!packet_miss_validation.passed);

        let mut mutation_miss = live_aql_materialized_packet_plan_input();
        mutation_miss.no_live_queue_mutation_contract = 0;
        let mutation_miss_validation = mutation_miss.proof().validate_ready();
        assert!(!mutation_miss_validation.no_live_queue_mutation_contract);
        assert!(!mutation_miss_validation.ready);
        assert!(!mutation_miss_validation.passed);
    }

    fn live_aql_shadow_packet_store_input() -> KfdQueueLiveAqlShadowPacketStoreInput {
        KfdQueueLiveAqlShadowPacketStoreInput {
            device_va: 0x1000,
            requested_iterations: 64,
            executed_iterations: 64,
            observed_present: 1,
            packet0_word0: 0x0001_1502,
            packet1_word0: 0x0001_1502,
            words_match_host_template: 1,
            payload_words_match_host_template: 1,
            header_words_match_host_template: 1,
            materialized_source_ready: 1,
            payloads_before_headers_contract: 1,
            low32_release_headers_last_contract: 1,
            doorbell_pending: 1,
            no_live_queue_mutation_contract: 1,
            region_bytes: 128,
            packet_count: 2,
            store_ready: 1,
            batch_plan_ready: 1,
            materialized_ready: true,
            host_present: true,
            host_shadow_words_match: true,
            host_sequence0_match: true,
            host_sentinel_match: true,
            host_sequence1_match: true,
            host_batch_ready_match: true,
            host_poll_ready: true,
            host_poll_header_match: true,
            host_poll_sequence_match: true,
        }
    }

    #[test]
    fn live_aql_shadow_packet_store_proof_preserves_ready_handoff() {
        let proof = live_aql_shadow_packet_store_input().proof();

        assert_eq!(proof.device_va, 0x1000);
        assert_eq!(proof.requested_iterations, 64);
        assert_eq!(proof.executed_iterations, 64);
        assert_eq!(proof.observed_present, 1);
        assert_eq!(proof.words_match_host_template, 1);
        assert_eq!(proof.payload_words_match_host_template, 1);
        assert_eq!(proof.header_words_match_host_template, 1);
        assert_eq!(proof.materialized_source_ready, 1);
        assert_eq!(proof.payloads_before_headers_contract, 1);
        assert_eq!(proof.low32_release_headers_last_contract, 1);
        assert_eq!(proof.doorbell_pending, 1);
        assert_eq!(proof.no_live_queue_mutation_contract, 1);
        assert_eq!(proof.store_ready, 1);
        assert_eq!(proof.batch_plan_ready, 1);
        assert_eq!(proof.handoff_ready, 1);
        assert!(proof.validate_handoff_ready().passed);
    }

    #[test]
    fn live_aql_shadow_packet_store_validation_rejects_host_or_poll_miss() {
        let mut host_miss = live_aql_shadow_packet_store_input();
        host_miss.host_sequence1_match = false;
        let host_miss_validation = host_miss.proof().validate_handoff_ready();
        assert!(!host_miss_validation.host_memory_ready);
        assert!(!host_miss_validation.handoff_ready);
        assert!(!host_miss_validation.passed);

        let mut poll_miss = live_aql_shadow_packet_store_input();
        poll_miss.host_poll_header_match = false;
        let poll_miss_validation = poll_miss.proof().validate_handoff_ready();
        assert!(!poll_miss_validation.host_poll_ready);
        assert!(!poll_miss_validation.handoff_ready);
        assert!(!poll_miss_validation.passed);

        let mut source_miss = live_aql_shadow_packet_store_input();
        source_miss.materialized_source_ready = 0;
        let source_miss_validation = source_miss.proof().validate_handoff_ready();
        assert!(!source_miss_validation.materialized_source_ready);
        assert!(!source_miss_validation.handoff_ready);
        assert!(!source_miss_validation.passed);

        let mut contract_miss = live_aql_shadow_packet_store_input();
        contract_miss.low32_release_headers_last_contract = 0;
        let contract_miss_validation = contract_miss.proof().validate_handoff_ready();
        assert!(!contract_miss_validation.low32_release_headers_last_contract);
        assert!(!contract_miss_validation.handoff_ready);
        assert!(!contract_miss_validation.passed);
    }

    fn live_aql_host_poll_input() -> KfdQueueLiveAqlHostPollInput {
        KfdQueueLiveAqlHostPollInput {
            expected_low32_header: 0x15,
            header0: 0x15,
            header1: 0x15,
            sequence: 2,
            expected_sequence: 2,
            sentinel: 0x514d415f53484457,
            expected_sentinel: 0x514d415f53484457,
            spins: 4,
            elapsed_us: 3.25,
            timeout_ms: 20.0,
            ready_before_device_wait: true,
        }
    }

    #[test]
    fn live_aql_host_poll_proof_preserves_acquire_only_ready_poll() {
        let proof = live_aql_host_poll_input().proof();

        assert_eq!(proof.expected_low32_header, 0x15);
        assert_eq!(proof.header0, 0x15);
        assert_eq!(proof.header1, 0x15);
        assert_eq!(proof.sequence, 2);
        assert_eq!(proof.expected_sequence, 2);
        assert_eq!(proof.sentinel, 0x514d415f53484457);
        assert_eq!(proof.spins, 4);
        assert_eq!(proof.elapsed_us, 3.25);
        assert_eq!(proof.timeout_ms, 20.0);
        assert_eq!(proof.header0_match, 1);
        assert_eq!(proof.header1_match, 1);
        assert_eq!(proof.sequence_match, 1);
        assert_eq!(proof.sentinel_match, 1);
        assert_eq!(proof.ready_before_device_wait, 1);
        assert!(!proof.fetch_add_performed);
        assert!(!proof.doorbell_written);
        assert!(!proof.live_queue_mutated);
        assert!(proof.validate_acquire_only().passed);
    }

    #[test]
    fn live_aql_host_poll_validation_rejects_mismatched_header_and_not_ready() {
        let mut bad_header = live_aql_host_poll_input();
        bad_header.header1 = 0;
        let bad_header_validation = bad_header.proof().validate_acquire_only();
        assert!(!bad_header_validation.header1_match);
        assert!(!bad_header_validation.passed);

        let mut not_ready = live_aql_host_poll_input();
        not_ready.ready_before_device_wait = false;
        let not_ready_validation = not_ready.proof().validate_acquire_only();
        assert!(!not_ready_validation.ready_before_device_wait);
        assert!(!not_ready_validation.passed);
    }

    fn live_aql_admission_guard_input() -> KfdQueueLiveAqlAdmissionGuardInput {
        KfdQueueLiveAqlAdmissionGuardInput {
            shadow_words_match: true,
            header_acquire_match: true,
            sequence_match: true,
            reservation_ready: true,
            restage_ready: true,
            batch_ready: true,
            materialized_ready: true,
            shadow_store_ready: true,
            host_poll_ready: true,
            host_poll_validated: true,
            no_live_mutation_lane0: true,
            no_live_mutation_lane1: true,
            no_live_mutation_lane2: true,
            no_live_mutation_lane3: true,
            no_live_mutation_lane4: true,
            no_live_mutation_lane5: true,
            no_live_mutation_lane6: true,
            no_live_mutation_lane7: true,
            submit_enabled: 0,
        }
    }

    #[test]
    fn live_aql_admission_guard_proof_preserves_non_submitting_token() {
        let proof = live_aql_admission_guard_input().proof();

        assert_eq!(proof.shadow_words_match, 1);
        assert_eq!(proof.header_acquire_match, 1);
        assert_eq!(proof.sequence_match, 1);
        assert_eq!(proof.reservation_ready, 1);
        assert_eq!(proof.restage_ready, 1);
        assert_eq!(proof.batch_ready, 1);
        assert_eq!(proof.materialized_ready, 1);
        assert_eq!(proof.shadow_store_ready, 1);
        assert_eq!(proof.host_poll_ready, 1);
        assert_eq!(proof.host_poll_validated, 1);
        assert_eq!(proof.prereqs_ready, 1);
        assert_eq!(proof.no_live_mutation_contract, 1);
        assert_eq!(proof.token_ready, 1);
        assert_eq!(proof.submit_enabled, 0);
        assert_eq!(proof.submit_allowed, 0);
        assert!(proof.validate_non_submitting().passed);
    }

    #[test]
    fn live_aql_admission_guard_validation_rejects_submit_and_mutation() {
        let mut submit_enabled = live_aql_admission_guard_input();
        submit_enabled.submit_enabled = 1;
        let submit_validation = submit_enabled.proof().validate_non_submitting();
        assert!(!submit_validation.submit_disabled);
        assert!(!submit_validation.submit_not_allowed);
        assert!(!submit_validation.passed);

        let mut mutation = live_aql_admission_guard_input();
        mutation.no_live_mutation_lane5 = false;
        let mutation_validation = mutation.proof().validate_non_submitting();
        assert!(!mutation_validation.no_live_mutation_contract);
        assert!(!mutation_validation.token_ready);
        assert!(!mutation_validation.passed);

        let mut unvalidated_host_poll = live_aql_admission_guard_input();
        unvalidated_host_poll.host_poll_validated = false;
        let host_poll_validation = unvalidated_host_poll.proof().validate_non_submitting();
        assert!(!host_poll_validation.host_poll_validated);
        assert!(!host_poll_validation.prereqs_ready);
        assert!(!host_poll_validation.token_ready);
        assert!(!host_poll_validation.passed);
    }

    fn live_aql_slot_preflight_input() -> KfdQueueLiveAqlSlotPreflightInput {
        KfdQueueLiveAqlSlotPreflightInput {
            offline_template_header_low32: 0,
            packet_template_header_low32: 0x0001_1502,
            expected_publish_low32: 0x0001_1502,
            packet_template_kernel_object: 0x1000,
            packet_template_kernarg_address: 0x2000,
            admission_token_ready: true,
            admission_validated: true,
            admission_submit_enabled: 0,
            admission_submit_allowed: 0,
            admission_no_live_mutation: true,
            queue_write_index_not_mutated: true,
            queue_read_index_not_mutated: true,
            batch_plan_no_fetch_add: true,
            batch_plan_no_doorbell: true,
            first_slot_matches_reservation: true,
            reservation_ready: true,
            live_write_allowed: 0,
        }
    }

    #[test]
    fn live_aql_slot_preflight_proof_preserves_disabled_write_contract() {
        let proof = live_aql_slot_preflight_input().proof();

        assert_eq!(proof.offline_template_header_invalid, 1);
        assert_eq!(proof.packet_template_ready, 1);
        assert_eq!(proof.admission_token_ready, 1);
        assert_eq!(proof.admission_validated, 1);
        assert_eq!(proof.future_write_blocked, 1);
        assert_eq!(proof.no_ownership_transfer, 1);
        assert_eq!(proof.first_slot_matches_reservation, 1);
        assert_eq!(proof.reservation_ready, 1);
        assert_eq!(proof.ready, 1);
        assert_eq!(proof.live_write_allowed, 0);
        assert!(proof.validate_disabled_live_write().passed);
    }

    #[test]
    fn live_aql_slot_preflight_validation_rejects_live_write_and_submit() {
        let mut live_write = live_aql_slot_preflight_input();
        live_write.live_write_allowed = 1;
        let live_write_validation = live_write.proof().validate_disabled_live_write();
        assert!(!live_write_validation.live_write_disabled);
        assert!(!live_write_validation.passed);

        let mut submit_allowed = live_aql_slot_preflight_input();
        submit_allowed.admission_submit_allowed = 1;
        let submit_validation = submit_allowed.proof().validate_disabled_live_write();
        assert!(!submit_validation.future_write_blocked);
        assert!(!submit_validation.ready);
        assert!(!submit_validation.passed);

        let mut unvalidated_admission = live_aql_slot_preflight_input();
        unvalidated_admission.admission_validated = false;
        let admission_validation = unvalidated_admission.proof().validate_disabled_live_write();
        assert!(!admission_validation.admission_validated);
        assert!(!admission_validation.ready);
        assert!(!admission_validation.passed);
    }

    fn live_aql_header_probe_input() -> KfdQueueLiveAqlHeaderProbeInput {
        KfdQueueLiveAqlHeaderProbeInput {
            slot0_va: 0x1000,
            slot1_va: 0x1040,
            slot0_offset: 0,
            slot1_offset: 64,
            slot0_low32: 0x0001_1502,
            slot1_low32: 0,
            slot0_type: 0x02,
            slot1_type: 0,
            expected_publish_low32: 0x0001_1502,
            targets_match_batch_plan: true,
            read_only_contract: true,
            fetch_add_not_performed: true,
            doorbell_not_written: true,
            live_slot_not_written: true,
            future_copy_blocked: true,
            live_slot_preflight_ready: true,
            live_slot_preflight_validated: true,
            batch_ready: true,
            reservation_ready: true,
            live_write_allowed: 0,
        }
    }

    #[test]
    fn live_aql_header_probe_proof_preserves_read_only_contract() {
        let proof = live_aql_header_probe_input().proof();

        assert_eq!(proof.slot0_low32, 0x0001_1502);
        assert_eq!(proof.slot1_low32, 0);
        assert_eq!(proof.slot0_type, 0x02);
        assert_eq!(proof.slot1_type, 0);
        assert_eq!(proof.slot0_not_target_publish, 0);
        assert_eq!(proof.slot1_not_target_publish, 1);
        assert_eq!(proof.targets_match_batch_plan, 1);
        assert_eq!(proof.read_only_contract, 1);
        assert_eq!(proof.fetch_add_not_performed, 1);
        assert_eq!(proof.doorbell_not_written, 1);
        assert_eq!(proof.live_slot_not_written, 1);
        assert_eq!(proof.future_copy_blocked, 1);
        assert_eq!(proof.live_slot_preflight_ready, 1);
        assert_eq!(proof.live_slot_preflight_validated, 1);
        assert_eq!(proof.live_write_allowed, 0);
        assert_eq!(proof.no_mutation_contract, 1);
        assert_eq!(proof.ready, 1);
        assert!(proof.validate_read_only_no_mutation().passed);
    }

    #[test]
    fn live_aql_header_probe_validation_rejects_type_mismatch_and_live_write() {
        let mut bad_type = live_aql_header_probe_input();
        bad_type.slot0_type = 0xff;
        let bad_type_validation = bad_type.proof().validate_read_only_no_mutation();
        assert!(!bad_type_validation.slot0_type_matches);
        assert!(!bad_type_validation.passed);

        let mut live_write = live_aql_header_probe_input();
        live_write.live_write_allowed = 1;
        let live_write_validation = live_write.proof().validate_read_only_no_mutation();
        assert!(!live_write_validation.live_write_disabled);
        assert!(!live_write_validation.ready);
        assert!(!live_write_validation.passed);

        let mut unvalidated_preflight = live_aql_header_probe_input();
        unvalidated_preflight.live_slot_preflight_validated = false;
        let preflight_validation = unvalidated_preflight
            .proof()
            .validate_read_only_no_mutation();
        assert!(!preflight_validation.live_slot_preflight_validated);
        assert!(!preflight_validation.ready);
        assert!(!preflight_validation.passed);
    }

    fn live_aql_copy_decision_input() -> KfdQueueLiveAqlCopyDecisionInput {
        KfdQueueLiveAqlCopyDecisionInput {
            slot0_header_low32: 0x0001_1502,
            slot1_header_low32: 0,
            publish_header_low32: 0x0001_1502,
            header_probe_ready: true,
            header_probe_validated: true,
            no_live_write_observed: true,
        }
    }

    #[test]
    fn live_aql_copy_decision_maps_slot_headers_to_reasons() {
        let proof = live_aql_copy_decision_input().proof();

        assert_eq!(proof.slot0_reason, 1);
        assert_eq!(proof.slot1_reason, 0);
        assert_eq!(proof.any_header_block, 1);
        assert_eq!(proof.requires_cleanup, 1);
        assert_eq!(proof.header_probe_ready, 1);
        assert_eq!(proof.header_probe_validated, 1);
        assert_eq!(proof.header_reset_allowed, 0);
        assert_eq!(proof.copy_allowed, 0);
        assert_eq!(proof.ready, 1);

        let mut non_publish = live_aql_copy_decision_input();
        non_publish.slot0_header_low32 = 0x0000_0001;
        let non_publish_proof = non_publish.proof();
        assert_eq!(non_publish_proof.slot0_reason, 2);
        assert_eq!(non_publish_proof.any_header_block, 1);
    }

    #[test]
    fn live_aql_copy_decision_validation_rejects_mutation_and_not_ready() {
        let validation = live_aql_copy_decision_input()
            .proof()
            .validate_disabled_not_copied();

        assert!(validation.any_header_block_matches_reasons);
        assert!(validation.requires_cleanup_matches_header_block);
        assert!(validation.header_reset_disabled);
        assert!(validation.copy_disabled);
        assert!(validation.ready);
        assert!(validation.passed);

        let mut live_write = live_aql_copy_decision_input();
        live_write.no_live_write_observed = false;
        let live_write_validation = live_write.proof().validate_disabled_not_copied();
        assert!(!live_write_validation.ready);
        assert!(!live_write_validation.passed);

        let mut unvalidated_header_probe = live_aql_copy_decision_input();
        unvalidated_header_probe.header_probe_validated = false;
        let header_probe_validation = unvalidated_header_probe
            .proof()
            .validate_disabled_not_copied();
        assert!(!header_probe_validation.header_probe_validated);
        assert!(!header_probe_validation.ready);
        assert!(!header_probe_validation.passed);

        let mut mutation_enabled = live_aql_copy_decision_input().proof();
        mutation_enabled.copy_allowed = 1;
        let mutation_validation = mutation_enabled.validate_disabled_not_copied();
        assert!(!mutation_validation.copy_disabled);
        assert!(!mutation_validation.passed);
    }

    fn cleanup_preflight_input() -> KfdQueueCleanupPreflightInput {
        KfdQueueCleanupPreflightInput {
            host_snapshot_read_index: 15,
            gpu_read_index: 15,
            gpu_reference_read_index: 15,
            ring_slots: 256,
            slot0_target_packet_id: 18,
            slot1_target_packet_id: 19,
            slot0_header_blocked: true,
            slot1_header_blocked: true,
            observed_block: true,
            requires_cleanup: true,
            copy_decision_ready: true,
            copy_decision_validated: true,
        }
    }

    #[test]
    fn cleanup_preflight_keeps_blocked_id_unknown_before_ring_wrap() {
        let decision = cleanup_preflight_input().decide();

        assert_eq!(decision.slot0.target_packet_id, 18);
        assert_eq!(decision.slot1.target_packet_id, 19);
        assert_eq!(decision.slot0.blocked_packet_id, u64::MAX);
        assert_eq!(decision.slot1.blocked_packet_id, u64::MAX);
        assert!(!decision.slot0.blocked_id_known);
        assert!(!decision.slot1.blocked_id_known);
        assert!(!decision.slot0.read_index_passed);
        assert!(!decision.slot1.read_index_passed);
        assert!(decision.ready);
    }

    #[test]
    fn cleanup_preflight_derives_prior_packet_id_after_ring_wrap() {
        let mut input = cleanup_preflight_input();
        input.host_snapshot_read_index = 43;
        input.gpu_read_index = 44;
        input.gpu_reference_read_index = 44;
        input.slot0_target_packet_id = 300;
        input.slot1_target_packet_id = 301;

        let decision = input.decide();

        assert_eq!(decision.slot0.blocked_packet_id, 44);
        assert_eq!(decision.slot1.blocked_packet_id, 45);
        assert!(decision.slot0.blocked_id_known);
        assert!(decision.slot1.blocked_id_known);
        assert!(!decision.slot0.read_index_passed);
        assert!(!decision.slot1.read_index_passed);
    }

    #[test]
    fn cleanup_preflight_requires_strict_read_index_pass() {
        let mut input = cleanup_preflight_input();
        input.host_snapshot_read_index = 45;
        input.gpu_read_index = 45;
        input.gpu_reference_read_index = 45;
        input.slot0_target_packet_id = 300;
        input.slot1_target_packet_id = 301;

        let decision = input.decide();

        assert_eq!(decision.slot0.blocked_packet_id, 44);
        assert_eq!(decision.slot1.blocked_packet_id, 45);
        assert!(decision.slot0.read_index_passed);
        assert!(!decision.slot1.read_index_passed);
        assert!(decision.any_reset_eligible);
        assert!(!decision.reset_allowed);
        assert!(!decision.copy_allowed);
    }

    #[test]
    fn cleanup_preflight_requires_gpu_read_index_not_behind_host_snapshot() {
        let mut input = cleanup_preflight_input();
        input.host_snapshot_read_index = 16;
        input.gpu_read_index = 15;
        input.gpu_reference_read_index = 15;

        let decision = input.decide();

        assert!(!decision.gpu_read_index_matches_host_snapshot);
        assert!(!decision.gpu_read_index_not_behind_host_snapshot);
        assert!(!decision.ready);
        assert!(!decision.reset_allowed);
        assert!(!decision.copy_allowed);
    }

    #[test]
    fn cleanup_preflight_proof_snapshot_preserves_log_scalars() {
        let proof = cleanup_preflight_input().decide().proof();

        assert_eq!(proof.gpu_read_index, 15);
        assert_eq!(proof.host_snapshot_read_index, 15);
        assert_eq!(proof.gpu_read_index_matches_reference, 1);
        assert_eq!(proof.gpu_read_index_matches_host_snapshot, 1);
        assert_eq!(proof.gpu_read_index_not_behind_host_snapshot, 1);
        assert_eq!(proof.ring_slots, 256);
        assert_eq!(proof.slot0_target_packet_id, 18);
        assert_eq!(proof.slot1_target_packet_id, 19);
        assert_eq!(proof.slot0_blocked_packet_id, u64::MAX);
        assert_eq!(proof.slot1_blocked_packet_id, u64::MAX);
        assert_eq!(proof.slot0_blocked_id_known, 0);
        assert_eq!(proof.slot1_blocked_id_known, 0);
        assert_eq!(proof.slot0_read_index_passed, 0);
        assert_eq!(proof.slot1_read_index_passed, 0);
        assert_eq!(proof.observed_block, 1);
        assert_eq!(proof.requires_cleanup, 1);
        assert_eq!(proof.copy_decision_ready, 1);
        assert_eq!(proof.copy_decision_validated, 1);
        assert_eq!(proof.any_reset_eligible, 0);
        assert_eq!(proof.reset_allowed, 0);
        assert_eq!(proof.copy_allowed, 0);
        assert_eq!(proof.ready, 1);
    }

    #[test]
    fn cleanup_preflight_validation_covers_disabled_cleanup_contract() {
        let validation = cleanup_preflight_input()
            .decide()
            .proof()
            .validate_disabled_observed_cleanup();

        assert!(validation.gpu_read_index_matches_reference);
        assert!(validation.gpu_read_index_matches_host_snapshot);
        assert!(validation.gpu_read_index_not_behind_host_snapshot);
        assert!(validation.observed_block_matches_requires_cleanup);
        assert!(validation.copy_decision_ready);
        assert!(validation.copy_decision_validated);
        assert!(validation.reset_disabled);
        assert!(validation.copy_disabled);
        assert!(validation.ready);
        assert!(validation.passed);

        let mut stale_input = cleanup_preflight_input();
        stale_input.host_snapshot_read_index = 16;
        let stale_validation = stale_input
            .decide()
            .proof()
            .validate_disabled_observed_cleanup();
        assert!(!stale_validation.gpu_read_index_matches_host_snapshot);
        assert!(!stale_validation.gpu_read_index_not_behind_host_snapshot);
        assert!(!stale_validation.passed);

        let mut unvalidated_copy_decision = cleanup_preflight_input();
        unvalidated_copy_decision.copy_decision_validated = false;
        let copy_decision_validation = unvalidated_copy_decision
            .decide()
            .proof()
            .validate_disabled_observed_cleanup();
        assert!(!copy_decision_validation.copy_decision_validated);
        assert!(!copy_decision_validation.ready);
        assert!(!copy_decision_validation.passed);

        let mut mutation_enabled = cleanup_preflight_input().decide().proof();
        mutation_enabled.reset_allowed = 1;
        let mutation_validation = mutation_enabled.validate_disabled_observed_cleanup();
        assert!(!mutation_validation.reset_disabled);
        assert!(!mutation_validation.passed);
    }
}
