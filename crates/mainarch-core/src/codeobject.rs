//! Minimal AMDHSA code-object loader.
//!
//! Parses a gfx9-class ELF code object (as produced by the LLVM AMDGPU
//! backend), loads its `PT_LOAD` segments into a host-visible, executable
//! [`DeviceBuffer`], and resolves each kernel's descriptor (`<name>.kd`) so it
//! can be handed to an AQL kernel-dispatch packet. No HSA/ROCr runtime is
//! involved — this is the loader those runtimes wrap.

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

use crate::{DeviceBuffer, Kfd};

/// The gfx950 code object built from `kernels/mainarch_kernels.cl`.
pub const MAINARCH_KERNELS_GFX950: &[u8] =
    include_bytes!("../artifacts/mainarch_kernels.gfx950.co");

/// A loaded kernel: its descriptor VA (the AQL `kernel_object`) plus segment
/// sizes the dispatch packet needs.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    /// Code-object virtual address of the `<name>.kd` kernel descriptor.
    pub kernel_descriptor_vaddr: u64,
    /// VA of the `<name>.kd` kernel descriptor — the AQL `kernel_object`.
    pub kernel_object: u64,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub kernarg_size: u32,
    pub kernarg_segment_align: u32,
    pub wavefront_size: u32,
    pub max_flat_workgroup_size: u32,
}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// CPU-readable metadata for a kernel in an AMDHSA code object.
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub name: String,
    /// Code-object virtual address of the `<name>.kd` kernel descriptor.
    pub kernel_descriptor_vaddr: u64,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub kernarg_size: u32,
    pub kernarg_segment_align: u32,
    pub wavefront_size: u32,
    pub max_flat_workgroup_size: u32,
}

/// A code object resident in GPU memory, with its kernels resolved by name.
pub struct CodeObject {
    _image: DeviceBuffer,
    base: u64,
    kernels: HashMap<String, Kernel>,
}

impl CodeObject {
    /// Load `bytes` (an AMDHSA ELF) onto `node_id`.
    pub fn load(kfd: &Kfd, node_id: u32, bytes: &[u8]) -> Result<Self> {
        let elf = Elf::parse(bytes)?;

        let image_size = elf
            .loads
            .iter()
            .map(|s| s.vaddr + s.memsz)
            .max()
            .ok_or_else(|| anyhow!("code object has no PT_LOAD segments"))?
            as usize;

        let mut image = kfd
            .alloc_host_visible(node_id, image_size, true)
            .context("allocating executable code-object buffer")?;
        let base = image.va();

        // Place each loadable segment at base + vaddr; the file-backed prefix is
        // copied, the remainder (bss) stays zero (fresh allocation is zeroed by
        // the kernel).
        {
            let dst = unsafe { image.as_mut_slice() };
            for seg in &elf.loads {
                let off = seg.vaddr as usize;
                let fz = seg.filesz as usize;
                dst[off..off + fz]
                    .copy_from_slice(&bytes[seg.offset as usize..seg.offset as usize + fz]);
            }
        }

        let metadata = kernel_metadata_from_note(bytes, &elf);
        let mut kernels = HashMap::new();
        for sym in &elf.kd_symbols {
            let kd_file = elf
                .va_to_file(sym.value)
                .ok_or_else(|| anyhow!("kernel descriptor {} VA not in any segment", sym.name))?;
            // amdhsa kernel descriptor (64 bytes): group_seg @0, priv_seg @4,
            // kernarg_size @8.
            let group_segment_size = read_u32(bytes, kd_file);
            let private_segment_size = read_u32(bytes, kd_file + 4);
            let kernarg_size = read_u32(bytes, kd_file + 8);
            let name = sym
                .name
                .strip_suffix(".kd")
                .unwrap_or(&sym.name)
                .to_string();
            let metadata = metadata.get(&name).cloned().unwrap_or_default();
            kernels.insert(
                name.clone(),
                Kernel {
                    name,
                    kernel_descriptor_vaddr: sym.value,
                    kernel_object: base + sym.value,
                    group_segment_size,
                    private_segment_size,
                    kernarg_size,
                    kernarg_segment_align: metadata.kernarg_segment_align,
                    wavefront_size: metadata.wavefront_size,
                    max_flat_workgroup_size: metadata.max_flat_workgroup_size,
                },
            );
        }

        if kernels.is_empty() {
            return Err(anyhow!("code object exposed no kernel descriptors"));
        }

        if std::env::var_os("MAINARCH_KFD_QUEUE_DEBUG").is_some() {
            eprintln!(
                "mainarch: code-object loaded base=0x{base:x} size=0x{image_size:x} kernels={:?}",
                kernels.keys().collect::<Vec<_>>()
            );
        }

        Ok(Self {
            _image: image,
            base,
            kernels,
        })
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        self.kernels
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("kernel '{name}' not found in code object"))
    }

    pub fn base(&self) -> u64 {
        self.base
    }
}

/// CPU-only summary of an AMDHSA code object. No GPU is required.
#[derive(Debug, Clone)]
pub struct CodeObjectInfo {
    /// gfx target triple, e.g. `amdgcn-amd-amdhsa--gfx950`.
    pub target: String,
    /// SHA-256 hex digest of the raw code-object bytes.
    pub sha256: String,
    /// Raw code-object size in bytes.
    pub size: usize,
    /// Kernels found in the code object.
    pub kernels: Vec<KernelInfo>,
}

/// CPU-only validation receipt for a required set of code-object kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeObjectKernelSetValidation {
    pub target: String,
    pub sha256: String,
    pub required_kernels: Vec<String>,
    pub present_count: usize,
    pub missing_kernels: Vec<String>,
}

impl CodeObjectKernelSetValidation {
    pub fn is_complete(&self) -> bool {
        self.missing_kernels.is_empty()
    }

    pub fn assert_complete(&self) -> Result<()> {
        if self.is_complete() {
            return Ok(());
        }
        Err(anyhow!(
            "code object {} is missing {} required kernel(s): {}",
            self.target,
            self.missing_kernels.len(),
            self.missing_kernels.join(", ")
        ))
    }
}

impl CodeObjectInfo {
    /// Parse `bytes` and return a printable summary without touching a GPU.
    pub fn inspect(bytes: &[u8]) -> Result<Self> {
        let elf = Elf::parse(bytes)?;
        let target = gfx_target_from_note(bytes, &elf).unwrap_or_else(|| "unknown".to_string());
        let sha256 = hex_digest(bytes);

        let metadata = kernel_metadata_from_note(bytes, &elf);
        let mut kernels = Vec::new();
        for sym in &elf.kd_symbols {
            let kd_file = elf
                .va_to_file(sym.value)
                .ok_or_else(|| anyhow!("kernel descriptor {} VA not in any segment", sym.name))?;
            let group_segment_size = read_u32(bytes, kd_file);
            let private_segment_size = read_u32(bytes, kd_file + 4);
            let kernarg_size = read_u32(bytes, kd_file + 8);
            let name = sym
                .name
                .strip_suffix(".kd")
                .unwrap_or(&sym.name)
                .to_string();
            let metadata = metadata.get(&name).cloned().unwrap_or_default();
            kernels.push(KernelInfo {
                name,
                kernel_descriptor_vaddr: sym.value,
                group_segment_size,
                private_segment_size,
                kernarg_size,
                kernarg_segment_align: metadata.kernarg_segment_align,
                wavefront_size: metadata.wavefront_size,
                max_flat_workgroup_size: metadata.max_flat_workgroup_size,
            });
        }

        kernels.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Self {
            target,
            sha256,
            size: bytes.len(),
            kernels,
        })
    }

    pub fn kernel_for(&self, name: &str) -> Option<&KernelInfo> {
        self.kernels.iter().find(|kernel| kernel.name == name)
    }

    pub fn contains_kernel(&self, name: &str) -> bool {
        self.kernel_for(name).is_some()
    }

    pub fn validate_required_kernels(&self, names: &[&str]) -> CodeObjectKernelSetValidation {
        let mut seen = BTreeSet::new();
        let required_kernels = names
            .iter()
            .filter_map(|name| {
                let name = (*name).to_string();
                if seen.insert(name.clone()) {
                    Some(name)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let missing_kernels = required_kernels
            .iter()
            .filter(|name| !self.contains_kernel(name))
            .cloned()
            .collect::<Vec<_>>();
        CodeObjectKernelSetValidation {
            target: self.target.clone(),
            sha256: self.sha256.clone(),
            present_count: required_kernels.len() - missing_kernels.len(),
            required_kernels,
            missing_kernels,
        }
    }
}

fn gfx_target_from_note(bytes: &[u8], elf: &Elf) -> Option<String> {
    // The AMDGPU note lives in the `.note` section and contains a
    // MessagePack-style descriptor. Locate the `amdhsa.target` key and read
    // the following fixstr value.
    let desc = amdgpu_metadata_note(bytes, elf).or_else(|| note_section_bytes(bytes, elf))?;

    // fixstr of "amdhsa.target" is 13 bytes -> 0xa0 + 13 = 0xad.
    let key = b"\xadamdhsa.target";
    let pos = desc.windows(key.len()).position(|w| w == key)?;
    let mut p = pos + key.len();
    let marker = *desc.get(p)?;
    if (0xa0..=0xbf).contains(&marker) {
        let len = (marker - 0xa0) as usize;
        p += 1;
        let value = desc.get(p..p + len)?;
        return String::from_utf8(value.to_vec()).ok();
    }
    None
}

#[derive(Debug, Clone, Default)]
struct KernelMetadata {
    kernarg_segment_align: u32,
    wavefront_size: u32,
    max_flat_workgroup_size: u32,
}

fn kernel_metadata_from_note(bytes: &[u8], elf: &Elf) -> HashMap<String, KernelMetadata> {
    amdgpu_metadata_note(bytes, elf)
        .and_then(parse_amdhsa_kernel_metadata)
        .unwrap_or_default()
}

fn note_section_bytes<'a>(bytes: &'a [u8], elf: &Elf) -> Option<&'a [u8]> {
    let note = elf.sections.iter().find(|s| s.name == ".note")?;
    let note_seg = elf
        .loads
        .iter()
        .find(|s| note.vaddr >= s.vaddr && note.vaddr < s.vaddr + s.filesz)?;
    let file_off = (note_seg.offset + (note.vaddr - note_seg.vaddr)) as usize;
    bytes.get(file_off..file_off + note.size as usize)
}

fn amdgpu_metadata_note<'a>(bytes: &'a [u8], elf: &Elf) -> Option<&'a [u8]> {
    let note = note_section_bytes(bytes, elf)?;
    let mut pos = 0usize;
    while pos + 12 <= note.len() {
        let namesz = read_u32(note, pos) as usize;
        let descsz = read_u32(note, pos + 4) as usize;
        let note_type = read_u32(note, pos + 8);
        let name_off = pos + 12;
        let name_end = name_off.checked_add(namesz)?;
        let desc_off = align4(name_end);
        let desc_end = desc_off.checked_add(descsz)?;
        if desc_end > note.len() || name_end > note.len() {
            break;
        }
        let name = &note[name_off..name_end];
        if note_type == 32 && name.starts_with(b"AMDGPU") {
            return note.get(desc_off..desc_end);
        }
        pos = align4(desc_end);
    }
    None
}

fn align4(v: usize) -> usize {
    (v + 3) & !3
}

fn parse_amdhsa_kernel_metadata(desc: &[u8]) -> Option<HashMap<String, KernelMetadata>> {
    let mut reader = MsgPackReader::new(desc);
    let marker = reader.read_marker()?;
    let len = reader.map_len(marker)?;
    let mut out = HashMap::new();
    for _ in 0..len {
        let key = reader.read_str()?;
        let value_marker = reader.read_marker()?;
        if key == "amdhsa.kernels" {
            let kernels_len = reader.array_len(value_marker)?;
            for _ in 0..kernels_len {
                if let Some((name, metadata)) = reader.read_kernel_metadata() {
                    out.insert(name, metadata);
                }
            }
        } else {
            reader.skip_with_marker(value_marker)?;
        }
    }
    Some(out)
}

struct MsgPackReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MsgPackReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_marker(&mut self) -> Option<u8> {
        let marker = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(marker)
    }

    fn read_kernel_metadata(&mut self) -> Option<(String, KernelMetadata)> {
        let marker = self.read_marker()?;
        let len = self.map_len(marker)?;
        let mut name = None;
        let mut metadata = KernelMetadata::default();
        for _ in 0..len {
            let key = self.read_str()?;
            let value_marker = self.read_marker()?;
            match key.as_str() {
                ".name" => name = self.read_str_with_marker(value_marker),
                ".kernarg_segment_align" => {
                    metadata.kernarg_segment_align =
                        self.read_u64_with_marker(value_marker)? as u32;
                }
                ".wavefront_size" => {
                    metadata.wavefront_size = self.read_u64_with_marker(value_marker)? as u32;
                }
                ".max_flat_workgroup_size" => {
                    metadata.max_flat_workgroup_size =
                        self.read_u64_with_marker(value_marker)? as u32;
                }
                _ => self.skip_with_marker(value_marker)?,
            }
        }
        name.map(|name| (name, metadata))
    }

    fn read_str(&mut self) -> Option<String> {
        let marker = self.read_marker()?;
        self.read_str_with_marker(marker)
    }

    fn read_str_with_marker(&mut self, marker: u8) -> Option<String> {
        let len = match marker {
            0xa0..=0xbf => (marker & 0x1f) as usize,
            0xd9 => self.read_u8()? as usize,
            0xda => self.read_u16()? as usize,
            0xdb => self.read_u32()? as usize,
            _ => return None,
        };
        let end = self.pos.checked_add(len)?;
        let bytes = self.data.get(self.pos..end)?;
        self.pos = end;
        String::from_utf8(bytes.to_vec()).ok()
    }

    fn read_u64_with_marker(&mut self, marker: u8) -> Option<u64> {
        match marker {
            0x00..=0x7f => Some(marker as u64),
            0xcc => Some(self.read_u8()? as u64),
            0xcd => Some(self.read_u16()? as u64),
            0xce => Some(self.read_u32()? as u64),
            0xcf => self.read_u64(),
            0xd0 => Some(self.read_i8()? as u64),
            0xd1 => Some(self.read_i16()? as u64),
            0xd2 => Some(self.read_i32()? as u64),
            0xd3 => Some(self.read_i64()? as u64),
            _ => None,
        }
    }

    fn map_len(&mut self, marker: u8) -> Option<usize> {
        match marker {
            0x80..=0x8f => Some((marker & 0x0f) as usize),
            0xde => Some(self.read_u16()? as usize),
            0xdf => Some(self.read_u32()? as usize),
            _ => None,
        }
    }

    fn array_len(&mut self, marker: u8) -> Option<usize> {
        match marker {
            0x90..=0x9f => Some((marker & 0x0f) as usize),
            0xdc => Some(self.read_u16()? as usize),
            0xdd => Some(self.read_u32()? as usize),
            _ => None,
        }
    }

    fn skip(&mut self) -> Option<()> {
        let marker = self.read_marker()?;
        self.skip_with_marker(marker)
    }

    fn skip_with_marker(&mut self, marker: u8) -> Option<()> {
        match marker {
            0x00..=0x7f | 0xc0 | 0xc2 | 0xc3 | 0xe0..=0xff => Some(()),
            0xcc | 0xd0 => self.skip_bytes(1),
            0xcd | 0xd1 => self.skip_bytes(2),
            0xce | 0xd2 | 0xca => self.skip_bytes(4),
            0xcf | 0xd3 | 0xcb => self.skip_bytes(8),
            0xa0..=0xbf => self.skip_bytes((marker & 0x1f) as usize),
            0xd9 | 0xc4 => {
                let len = self.read_u8()? as usize;
                self.skip_bytes(len)
            }
            0xda | 0xc5 => {
                let len = self.read_u16()? as usize;
                self.skip_bytes(len)
            }
            0xdb | 0xc6 => {
                let len = self.read_u32()? as usize;
                self.skip_bytes(len)
            }
            0x90..=0x9f => {
                for _ in 0..(marker & 0x0f) {
                    self.skip()?;
                }
                Some(())
            }
            0xdc => {
                let len = self.read_u16()?;
                for _ in 0..len {
                    self.skip()?;
                }
                Some(())
            }
            0xdd => {
                let len = self.read_u32()?;
                for _ in 0..len {
                    self.skip()?;
                }
                Some(())
            }
            0x80..=0x8f => {
                for _ in 0..(marker & 0x0f) {
                    self.skip()?;
                    self.skip()?;
                }
                Some(())
            }
            0xde => {
                let len = self.read_u16()?;
                for _ in 0..len {
                    self.skip()?;
                    self.skip()?;
                }
                Some(())
            }
            0xdf => {
                let len = self.read_u32()?;
                for _ in 0..len {
                    self.skip()?;
                    self.skip()?;
                }
                Some(())
            }
            _ => None,
        }
    }

    fn skip_bytes(&mut self, len: usize) -> Option<()> {
        self.pos = self.pos.checked_add(len)?;
        (self.pos <= self.data.len()).then_some(())
    }

    fn read_u8(&mut self) -> Option<u8> {
        let value = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(value)
    }

    fn read_u16(&mut self) -> Option<u16> {
        let end = self.pos.checked_add(2)?;
        let bytes: [u8; 2] = self.data.get(self.pos..end)?.try_into().ok()?;
        self.pos = end;
        Some(u16::from_be_bytes(bytes))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let end = self.pos.checked_add(4)?;
        let bytes: [u8; 4] = self.data.get(self.pos..end)?.try_into().ok()?;
        self.pos = end;
        Some(u32::from_be_bytes(bytes))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let end = self.pos.checked_add(8)?;
        let bytes: [u8; 8] = self.data.get(self.pos..end)?.try_into().ok()?;
        self.pos = end;
        Some(u64::from_be_bytes(bytes))
    }

    fn read_i8(&mut self) -> Option<i8> {
        Some(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Option<i16> {
        Some(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> Option<i32> {
        Some(self.read_u32()? as i32)
    }

    fn read_i64(&mut self) -> Option<i64> {
        Some(self.read_u64()? as i64)
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

struct LoadSeg {
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
}

struct Section {
    name: String,
    vaddr: u64,
    size: u64,
}

struct KdSym {
    name: String,
    value: u64,
}

struct Elf {
    loads: Vec<LoadSeg>,
    sections: Vec<Section>,
    kd_symbols: Vec<KdSym>,
}

impl Elf {
    fn parse(d: &[u8]) -> Result<Self> {
        if d.len() < 64 || &d[0..4] != b"\x7fELF" {
            return Err(anyhow!("not an ELF file"));
        }
        if d[4] != 2 || d[5] != 1 {
            return Err(anyhow!("expected 64-bit little-endian ELF"));
        }

        let e_phoff = read_u64(d, 0x20);
        let e_phentsize = read_u16(d, 0x36) as usize;
        let e_phnum = read_u16(d, 0x38) as usize;
        let e_shoff = read_u64(d, 0x28);
        let e_shentsize = read_u16(d, 0x3a) as usize;
        let e_shnum = read_u16(d, 0x3c) as usize;

        let mut loads = Vec::new();
        for i in 0..e_phnum {
            let o = e_phoff as usize + i * e_phentsize;
            let p_type = read_u32(d, o);
            if p_type != 1 {
                continue; // PT_LOAD only
            }
            loads.push(LoadSeg {
                offset: read_u64(d, o + 8),
                vaddr: read_u64(d, o + 16),
                filesz: read_u64(d, o + 32),
                memsz: read_u64(d, o + 40),
            });
        }

        // Find .dynsym / .dynstr via the section table and keep sections for
        // metadata extraction.
        let mut dynsym: Option<(u64, u64, u64)> = None; // (off, size, entsize)
        let mut dynstr_off: Option<u64> = None;
        let e_shstrndx = read_u16(d, 0x3e) as usize;
        let shstr_off = read_u64(d, e_shoff as usize + e_shstrndx * e_shentsize + 24);
        let mut sections = Vec::new();
        for i in 0..e_shnum {
            let o = e_shoff as usize + i * e_shentsize;
            let name_off = read_u32(d, o) as usize;
            let name = cstr(d, shstr_off as usize + name_off);
            let sh_addr = read_u64(d, o + 16);
            let sh_off = read_u64(d, o + 24);
            let sh_size = read_u64(d, o + 32);
            let sh_entsize = read_u64(d, o + 56);
            sections.push(Section {
                name: name.clone(),
                vaddr: sh_addr,
                size: sh_size,
            });
            match name.as_str() {
                ".dynsym" => dynsym = Some((sh_off, sh_size, sh_entsize)),
                ".dynstr" => dynstr_off = Some(sh_off),
                _ => {}
            }
        }

        let (sym_off, sym_size, sym_ent) =
            dynsym.ok_or_else(|| anyhow!("code object has no .dynsym"))?;
        let str_off = dynstr_off.ok_or_else(|| anyhow!("code object has no .dynstr"))?;
        if sym_ent == 0 {
            return Err(anyhow!(".dynsym has zero entry size"));
        }

        let mut kd_symbols = Vec::new();
        let count = sym_size / sym_ent;
        for i in 0..count {
            let o = (sym_off + i * sym_ent) as usize;
            let st_name = read_u32(d, o) as usize;
            let st_value = read_u64(d, o + 8);
            let name = cstr(d, str_off as usize + st_name);
            if name.ends_with(".kd") {
                kd_symbols.push(KdSym {
                    name,
                    value: st_value,
                });
            }
        }

        Ok(Self {
            loads,
            sections,
            kd_symbols,
        })
    }

    fn va_to_file(&self, va: u64) -> Option<usize> {
        for s in &self.loads {
            if va >= s.vaddr && va < s.vaddr + s.filesz {
                return Some((s.offset + (va - s.vaddr)) as usize);
            }
        }
        None
    }
}

fn read_u16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}
fn read_u32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}
fn read_u64(d: &[u8], o: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[o..o + 8]);
    u64::from_le_bytes(b)
}
fn cstr(d: &[u8], o: usize) -> String {
    let end = d[o..]
        .iter()
        .position(|&c| c == 0)
        .map_or(d.len(), |p| o + p);
    String::from_utf8_lossy(&d[o..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_object_info_is_cpu_readable() {
        let info = CodeObjectInfo::inspect(MAINARCH_KERNELS_GFX950).expect("inspect should parse");
        assert!(info.target.ends_with("gfx950"), "target={}", info.target);
        assert_eq!(info.sha256.len(), 64);
        assert_eq!(info.size, MAINARCH_KERNELS_GFX950.len());
        assert!(!info.kernels.is_empty());

        // Spot-check a few well-known kernels.
        let names: Vec<_> = info.kernels.iter().map(|k| k.name.as_str()).collect();
        for want in ["poke", "reduce_sum_f32", "accumulate_f32", "scale_f32"] {
            assert!(names.contains(&want), "missing {want} in {names:?}");
        }
        assert!(info.contains_kernel("gemv_f16"));
        let gemv = info.kernel_for("gemv_f16").unwrap();
        assert_eq!(gemv.name, "gemv_f16");
        assert!(gemv.kernarg_size > 0);

        let validation = info.validate_required_kernels(&[
            "gemv_f16",
            "rmsnorm_f16",
            "decode_step_embed_rmsnorm_token_f16",
            "gemv_f16",
        ]);
        assert!(validation.is_complete());
        assert_eq!(
            validation.required_kernels,
            vec![
                "gemv_f16".to_string(),
                "rmsnorm_f16".to_string(),
                "decode_step_embed_rmsnorm_token_f16".to_string(),
            ]
        );
        assert_eq!(validation.present_count, 3);
        validation.assert_complete().unwrap();

        let missing = info.validate_required_kernels(&["gemv_f16", "not_a_mainarch_kernel"]);
        assert!(!missing.is_complete());
        assert_eq!(missing.present_count, 1);
        assert_eq!(
            missing.missing_kernels,
            vec!["not_a_mainarch_kernel".to_string()]
        );
        let err = missing.assert_complete().unwrap_err().to_string();
        assert!(err.contains("missing 1 required kernel(s)"));
        assert!(err.contains("not_a_mainarch_kernel"));

        // Descriptor sizes look sane.
        for k in &info.kernels {
            assert!(
                (8..=4096).contains(&k.kernarg_size),
                "kernarg {} for {}",
                k.kernarg_size,
                k.name
            );
        }
    }

    #[test]
    fn embedded_object_parses_and_exposes_kernels() {
        let elf = Elf::parse(MAINARCH_KERNELS_GFX950).expect("artifact should parse");
        assert!(!elf.loads.is_empty());
        let names: Vec<_> = elf.kd_symbols.iter().map(|s| s.name.as_str()).collect();
        for want in [
            "poke.kd",
            "reduce_sum_f32.kd",
            "accumulate_f32.kd",
            "scale_f32.kd",
        ] {
            assert!(names.contains(&want), "missing {want} in {names:?}");
        }
        // Descriptor fields are readable and kernarg sizes are sane.
        for s in &elf.kd_symbols {
            let f = elf.va_to_file(s.value).expect("kd VA maps to a segment");
            let kernarg = read_u32(MAINARCH_KERNELS_GFX950, f + 8);
            assert!(
                (8..=4096).contains(&kernarg),
                "kernarg {kernarg} for {}",
                s.name
            );
        }
    }
}
