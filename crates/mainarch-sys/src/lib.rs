//! `mainarch-sys` — the bottom of the stack.
//!
//! Direct bindings to the AMD kernel compute driver (`amdkfd`, exposed at
//! `/dev/kfd`) using the kernel UAPI ioctl protocol. **No ROCm, no libhsakmt,
//! no HSA runtime** — this is the literal kernel ABI those libraries wrap.
//!
//! For v1 we keep the ABI surface deliberately explicit: direct structs and
//! ioctl arguments instead of runtime indirection. The kernel UAPI is canonical;
//! these structures mirror `<linux/kfd_ioctl.h>` so call semantics are stable and
//! auditable.

#![allow(clippy::missing_safety_doc)]

use std::os::fd::RawFd;

/// Path to the AMD kernel compute device.
pub const KFD_DEVICE: &str = "/dev/kfd";

// ---- Linux ioctl number encoding, mirrors <asm-generic/ioctl.h> ----
const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;

const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

const fn ioc(dir: u32, typ: u32, nr: u32, size: usize) -> u32 {
    (dir << IOC_DIRSHIFT)
        | (typ << IOC_TYPESHIFT)
        | (nr << IOC_NRSHIFT)
        | ((size as u32) << IOC_SIZESHIFT)
}

/// `'K'` — `AMDKFD_IOCTL_BASE`.
const KFD_IOCTL_BASE: u32 = b'K' as u32;

const fn ior<T>(nr: u32) -> u32 {
    ioc(IOC_READ, KFD_IOCTL_BASE, nr, core::mem::size_of::<T>())
}
const fn iow<T>(nr: u32) -> u32 {
    ioc(IOC_WRITE, KFD_IOCTL_BASE, nr, core::mem::size_of::<T>())
}
const fn iowr<T>(nr: u32) -> u32 {
    ioc(
        IOC_READ | IOC_WRITE,
        KFD_IOCTL_BASE,
        nr,
        core::mem::size_of::<T>(),
    )
}

pub const KFD_IOC_QUEUE_TYPE_COMPUTE: u32 = 0x0;
pub const KFD_IOC_QUEUE_TYPE_SDMA: u32 = 0x1;
pub const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;
pub const KFD_IOC_QUEUE_TYPE_SDMA_XGMI: u32 = 0x3;
pub const KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID: u32 = 0x4;

pub const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL_MIN_PERCENTAGE: u32 = 100;
pub const KFD_IOC_QUEUE_MAX_PERCENTAGE: u32 = 100;
pub const KFD_IOC_QUEUE_MAX_PRIORITY: u32 = 15;
pub const KFD_IOC_QUEUE_MIN_RING_SIZE: u32 = 1024;

pub const KFD_IOC_ALLOC_MEM_FLAGS_VRAM: u32 = 1 << 0;
pub const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;
pub const KFD_IOC_ALLOC_MEM_FLAGS_USERPTR: u32 = 1 << 2;
pub const KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL: u32 = 1 << 3;
pub const KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP: u32 = 1 << 4;
pub const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;
pub const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
pub const KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC: u32 = 1 << 29;
pub const KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE: u32 = 1 << 28;
pub const KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM: u32 = 1 << 27;
pub const KFD_IOC_ALLOC_MEM_FLAGS_COHERENT: u32 = 1 << 26;
pub const KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED: u32 = 1 << 25;
pub const KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT: u32 = 1 << 24;

/// `struct kfd_ioctl_get_version_args`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GetVersionArgs {
    pub major_version: u32,
    pub minor_version: u32,
}

/// `AMDKFD_IOC_GET_VERSION` request number (`AMDKFD_IOR(0x01, ...)`).
pub fn ioc_get_version() -> libc::c_ulong {
    ior::<GetVersionArgs>(0x01) as libc::c_ulong
}

pub fn ioc_create_queue() -> libc::c_ulong {
    iowr::<CreateQueueArgs>(0x02) as libc::c_ulong
}

pub fn ioc_create_queue_compat() -> libc::c_ulong {
    iowr::<CreateQueueArgsCompat>(0x02) as libc::c_ulong
}

pub fn ioc_destroy_queue() -> libc::c_ulong {
    iowr::<DestroyQueueArgs>(0x03) as libc::c_ulong
}

pub fn ioc_acquire_vm() -> libc::c_ulong {
    iow::<AcquireVmArgs>(0x15) as libc::c_ulong
}

pub fn ioc_alloc_memory_of_gpu() -> libc::c_ulong {
    iowr::<AllocMemoryOfGpuArgs>(0x16) as libc::c_ulong
}

pub fn ioc_free_memory_of_gpu() -> libc::c_ulong {
    iow::<FreeMemoryOfGpuArgs>(0x17) as libc::c_ulong
}

pub fn ioc_map_memory_to_gpu() -> libc::c_ulong {
    iowr::<MapMemoryToGpuArgs>(0x18) as libc::c_ulong
}

pub fn ioc_unmap_memory_from_gpu() -> libc::c_ulong {
    iowr::<UnmapMemoryFromGpuArgs>(0x19) as libc::c_ulong
}

/// Issue `AMDKFD_IOC_GET_VERSION` against an open fd to `/dev/kfd`.
///
/// The simplest possible proof that we are speaking the kernel ABI directly.
pub unsafe fn ioctl_get_version(fd: RawFd) -> std::io::Result<GetVersionArgs> {
    let mut args = GetVersionArgs::default();
    let rc = libc::ioctl(fd, ioc_get_version(), &mut args as *mut GetVersionArgs);
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(args)
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CreateQueueArgs {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub doorbell_offset: u64,

    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
    pub queue_id: u32,

    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
    pub sdma_engine_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CreateQueueArgsCompat {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub doorbell_offset: u64,

    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
    pub queue_id: u32,

    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DestroyQueueArgs {
    pub queue_id: u32,
    pub pad: u32,
}

/// `struct kfd_ioctl_acquire_vm_args`. Binds a DRM render-node fd to the
/// process's KFD VM for one device. Mandatory before any memory ioctl.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct AcquireVmArgs {
    pub drm_fd: u32,
    pub gpu_id: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct AllocMemoryOfGpuArgs {
    pub va_addr: u64,
    pub size: u64,
    pub handle: u64,
    pub mmap_offset: u64,
    pub gpu_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct FreeMemoryOfGpuArgs {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct MapMemoryToGpuArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UnmapMemoryFromGpuArgs {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

/// Create a compute queue on a KFD node.
///
/// Mirrors `AMDKFD_IOC_CREATE_QUEUE` in `kfd_ioctl.h`.
pub unsafe fn ioctl_create_queue(fd: RawFd, args: &mut CreateQueueArgs) -> std::io::Result<()> {
    let rc = libc::ioctl(fd, ioc_create_queue(), args as *mut CreateQueueArgs);
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Create queue with older kernel ABI layout.
pub unsafe fn ioctl_create_queue_compat(
    fd: RawFd,
    args: &mut CreateQueueArgsCompat,
) -> std::io::Result<()> {
    let rc = libc::ioctl(
        fd,
        ioc_create_queue_compat(),
        args as *mut CreateQueueArgsCompat,
    );
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Destroy a KFD compute queue by ID.
pub unsafe fn ioctl_destroy_queue(fd: RawFd, args: &mut DestroyQueueArgs) -> std::io::Result<()> {
    let rc = libc::ioctl(fd, ioc_destroy_queue(), args as *mut DestroyQueueArgs);
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Acquire the per-device GPU VM for this process (`AMDKFD_IOC_ACQUIRE_VM`).
pub unsafe fn ioctl_acquire_vm(fd: RawFd, args: &mut AcquireVmArgs) -> std::io::Result<()> {
    let rc = libc::ioctl(fd, ioc_acquire_vm(), args as *mut AcquireVmArgs);
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Allocate SVM-like GPU-visible memory.
pub unsafe fn ioctl_alloc_memory_of_gpu(
    fd: RawFd,
    args: &mut AllocMemoryOfGpuArgs,
) -> std::io::Result<()> {
    let rc = libc::ioctl(
        fd,
        ioc_alloc_memory_of_gpu(),
        args as *mut AllocMemoryOfGpuArgs,
    );
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Release memory previously allocated with `ioctl_alloc_memory_of_gpu`.
pub unsafe fn ioctl_free_memory_of_gpu(
    fd: RawFd,
    args: &mut FreeMemoryOfGpuArgs,
) -> std::io::Result<()> {
    let rc = libc::ioctl(
        fd,
        ioc_free_memory_of_gpu(),
        args as *mut FreeMemoryOfGpuArgs,
    );
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Map memory to one or more GPUs.
pub unsafe fn ioctl_map_memory_to_gpu(
    fd: RawFd,
    args: &mut MapMemoryToGpuArgs,
) -> std::io::Result<()> {
    let rc = libc::ioctl(fd, ioc_map_memory_to_gpu(), args as *mut MapMemoryToGpuArgs);
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Unmap memory from one or more GPUs.
pub unsafe fn ioctl_unmap_memory_from_gpu(
    fd: RawFd,
    args: &mut UnmapMemoryFromGpuArgs,
) -> std::io::Result<()> {
    let rc = libc::ioctl(
        fd,
        ioc_unmap_memory_from_gpu(),
        args as *mut UnmapMemoryFromGpuArgs,
    );
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Decode a kfd `gfx_target_version` integer (e.g. `90500`) into `"gfx950"`.
pub fn gfx_name(gfx_target_version: u64) -> String {
    if gfx_target_version == 0 {
        return "gfx?".to_string();
    }
    let major = gfx_target_version / 10000;
    let minor = (gfx_target_version / 100) % 100;
    let step = gfx_target_version % 100;
    format!("gfx{major}{minor:x}{step:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx_decode() {
        assert_eq!(gfx_name(90500), "gfx950"); // MI355X
        assert_eq!(gfx_name(90402), "gfx942"); // MI300X
        assert_eq!(gfx_name(90010), "gfx90a"); // MI200
        assert_eq!(gfx_name(110000), "gfx1100");
    }

    #[test]
    fn get_version_ioctl_number() {
        // dir=READ(2), type='K'(0x4b), nr=0x01, size=8  ->  0x80084b01
        assert_eq!(ioc_get_version(), 0x8008_4b01);
    }

    #[test]
    fn kfd_queue_ioctl_numbers() {
        assert_eq!(ioc_create_queue(), 0xc0604b02);
        assert_eq!(ioc_create_queue_compat(), 0xc0584b02);
        assert_eq!(ioc_destroy_queue(), 0xc0084b03);
        assert_eq!(ioc_acquire_vm(), 0x40084b15);
        assert_eq!(ioc_alloc_memory_of_gpu(), 0xc0284b16);
        assert_eq!(ioc_free_memory_of_gpu(), 0x40084b17);
        assert_eq!(ioc_map_memory_to_gpu(), 0xc0184b18);
        assert_eq!(ioc_unmap_memory_from_gpu(), 0xc0184b19);
    }
}
