"""Hardware Telemetry, Dynamic Tiering, and Autonomous Runtime Auto-Tuning.

Standalone, portable hardware discovery and LLM inference auto-configurator.
Compatible with Python 3.10+ standard library. psutil is an optional accelerator.
No external framework dependencies, no Ogun-specific imports, no side effects at import.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Guarded optional psutil import
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

# Windows subprocess creation flag to suppress console windows
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Calibrated VRAM overhead constant:
# In llama.cpp / Ollama, binary weights and context structures leave nominal overhead
# close to 0.0 GB when accounting for standard fp16 KV cache at 8192 ctx.
CALIBRATED_VRAM_OVERHEAD_GB: float = 0.0


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuInfo:
    """Telemetry information for a detected graphics compute device."""
    vendor: str              # "nvidia" | "amd" | "intel" | "apple" | "none"
    name: str                # Device name (e.g. "NVIDIA GeForce RTX 5070")
    vram_total_gb: float     # Total physical VRAM in gigabytes
    vram_free_gb: float      # Free / unallocated VRAM in gigabytes
    unified_memory: bool     # True if system RAM is unified (Apple Silicon)
    driver: str              # Driver version string
    compute_capability: str  # CUDA compute capability or API support
    index: int               # Device index


@dataclass(frozen=True)
class HardwareInfo:
    """Full snapshot of host system hardware capabilities and telemetry."""
    os: str                  # "windows" | "linux" | "darwin"
    arch: str                # "x86_64" | "amd64" | "arm64" | "aarch64"
    cpu_name: str            # CPU brand string
    cpu_physical: int        # Number of physical CPU cores
    cpu_logical: int         # Number of logical execution threads
    isa: Dict[str, bool]     # Instruction set extensions: avx, avx2, avx512, neon
    ram_total_gb: float      # Total physical system RAM in gigabytes
    ram_available_gb: float  # Currently available system RAM in gigabytes
    gpus: List[GpuInfo]      # All detected GPUs, sorted best-first
    gpu: Optional[GpuInfo]   # Primary compute GPU or None
    probe_seconds: float     # Execution time spent probing hardware
    notes: List[str]         # Diagnostic telemetry notes and fallback notices


@dataclass(frozen=True)
class ModelSpec:
    """Architecture and memory profile specifications for an LLM."""
    tag: str                 # Model identifier tag (e.g. "qwen2.5-coder:14b")
    blocks: int              # Number of transformer layer blocks (e.g. 48)
    head_count_kv: int       # Number of key-value attention heads (e.g. 8)
    head_dim: int            # Dimension per attention head (e.g. 128)
    hidden: int              # Hidden dimension / embedding length (e.g. 5120)
    file_gb: float           # Model weight file size in gigabytes (e.g. 9.0)
    native_ctx: int          # Native trained context length (e.g. 32768)
    quant: str               # Quantization string (e.g. "Q4_K_M")
    source: str              # "ollama" | "catalogue" | "estimate"


@dataclass(frozen=True)
class ExecutionProfile:
    """Calculated inference runtime execution profile."""
    tier: str                # "high" | "mid" | "low"
    mode: str                # "gpu" | "hybrid" | "cpu"
    model_tag: str           # Target model tag
    quant: str               # Target quantization format
    num_gpu: Optional[int]   # Assigned GPU layers (None = auto/omit, 99 = all, 0 = CPU)
    gpu_units: int           # Number of units placed on GPU
    total_units: int         # Total units (blocks + 1)
    num_ctx: int             # Configured context window size
    num_thread: int          # Configured compute threads
    reserve_gb: float        # VRAM reserved for host OS / 3D apps
    est_vram_gb: float       # Estimated GPU VRAM consumption
    est_ram_gb: float        # Estimated System RAM consumption
    fits: bool               # True if configuration fits without memory exhaustion
    warnings: List[str]      # Health / performance warnings
    source: str              # "auto" | "override"


# -----------------------------------------------------------------------------
# Hardware Discovery Implementation
# -----------------------------------------------------------------------------

def _run_subproc(cmd: List[str], timeout: float = 3.0) -> Tuple[int, str]:
    """Execute a subprocess command with safety flags and timeout."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
        return proc.returncode, proc.stdout or ""
    except Exception as e:
        return -1, str(e)


def _probe_cpu_name(sys_os: str) -> str:
    """Probe CPU brand/name string without heavy external tools."""
    if sys_os == "windows":
        try:
            import winreg  # type: ignore
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as k:
                val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
    elif sys_os == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sys_os == "darwin":
        rc, out = _run_subproc(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=1.5)
        if rc == 0 and out.strip():
            return out.strip()

    return platform.processor() or "Unknown CPU"


def _probe_cpu_cores(sys_os: str, logical: int, timeout: float, notes: List[str]) -> Tuple[int, int]:
    """Probe physical and logical CPU cores."""
    phys = None
    if psutil is not None:
        try:
            p = psutil.cpu_count(logical=False)
            if p and p > 0:
                phys = p
        except Exception:
            pass

    if phys is None:
        if sys_os == "windows":
            # Try PowerShell CIM Win32_Processor
            rc, out = _run_subproc(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum | Select-Object -ExpandProperty Sum",
                ],
                timeout=min(timeout, 2.5),
            )
            if rc == 0 and out.strip().isdigit():
                phys = int(out.strip())
            else:
                # Try wmic as secondary fallback
                rc_w, out_w = _run_subproc(["wmic", "cpu", "get", "NumberOfCores"], timeout=min(timeout, 2.0))
                if rc_w == 0:
                    nums = [int(line.strip()) for line in out_w.splitlines() if line.strip().isdigit()]
                    if nums:
                        phys = sum(nums)
        elif sys_os == "linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                    cores = set()
                    cur_phys = 0
                    for line in f:
                        if line.startswith("physical id"):
                            cur_phys = int(line.split(":", 1)[1].strip())
                        elif line.startswith("core id"):
                            cur_core = int(line.split(":", 1)[1].strip())
                            cores.add((cur_phys, cur_core))
                    if cores:
                        phys = len(cores)
            except Exception:
                pass
        elif sys_os == "darwin":
            rc, out = _run_subproc(["sysctl", "-n", "hw.physicalcpu"], timeout=1.5)
            if rc == 0 and out.strip().isdigit():
                phys = int(out.strip())

    if phys is None or phys <= 0:
        phys = max(1, logical // 2)
        notes.append("Physical CPU cores could not be directly queried; estimated as logical // 2.")

    return phys, logical


def _probe_isa(sys_os: str, arch: str, notes: List[str]) -> Dict[str, bool]:
    """Detect instruction set extensions (AVX, AVX2, AVX512, NEON)."""
    isa = {"avx": False, "avx2": False, "avx512": False, "neon": False}
    if sys_os == "windows":
        try:
            import ctypes
            k = ctypes.windll.kernel32  # type: ignore
            # Windows API IsProcessorFeaturePresent:
            # 34: PF_ARM_NEON_INSTRUCTIONS_AVAILABLE
            # 39: PF_AVX_INSTRUCTIONS_AVAILABLE
            # 40: PF_AVX2_INSTRUCTIONS_AVAILABLE
            # 41: PF_AVX512F_INSTRUCTIONS_AVAILABLE
            isa["avx"] = bool(k.IsProcessorFeaturePresent(39))
            isa["avx2"] = bool(k.IsProcessorFeaturePresent(40))
            isa["avx512"] = bool(k.IsProcessorFeaturePresent(41))
            isa["neon"] = bool(k.IsProcessorFeaturePresent(34)) or (arch in ("arm64", "aarch64"))
        except Exception as e:
            notes.append(f"Windows ISA feature probe failed: {e}")
    elif sys_os == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                flags_text = ""
                for line in f:
                    if line.startswith("flags") or line.startswith("Features"):
                        flags_text = line.split(":", 1)[1].strip()
                        break
                tokens = set(flags_text.split())
                isa["avx"] = "avx" in tokens
                isa["avx2"] = "avx2" in tokens
                isa["avx512"] = any(t.startswith("avx512") for t in tokens)
                isa["neon"] = ("neon" in tokens) or ("asimd" in tokens)
        except Exception as e:
            notes.append(f"Linux ISA probe failed: {e}")
    elif sys_os == "darwin":
        if arch in ("arm64", "aarch64"):
            isa["neon"] = True
        else:
            rc, out = _run_subproc(["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"], timeout=1.5)
            if rc == 0:
                tokens = set(out.lower().split())
                isa["avx"] = ("avx" in tokens) or ("avx1.0" in tokens)
                isa["avx2"] = "avx2" in tokens
                isa["avx512"] = any("avx512" in t for t in tokens)
    return isa


def _probe_ram(sys_os: str, notes: List[str]) -> Tuple[float, float]:
    """Probe total and available physical RAM in gigabytes."""
    total_gb = 0.0
    avail_gb = 0.0

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            total_gb = round(vm.total / (1024**3), 2)
            avail_gb = round(vm.available / (1024**3), 2)
            return total_gb, avail_gb
        except Exception as e:
            notes.append(f"psutil RAM probe failed: {e}")

    # Fallbacks without psutil
    if sys_os == "windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore
                total_gb = round(stat.ullTotalPhys / (1024**3), 2)
                avail_gb = round(stat.ullAvailPhys / (1024**3), 2)
                return total_gb, avail_gb
        except Exception as e:
            notes.append(f"ctypes GlobalMemoryStatusEx failed: {e}")
    elif sys_os == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
                mem: Dict[str, int] = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].strip().split()[0])
                if "MemTotal" in mem:
                    total_gb = round(mem["MemTotal"] / (1024**2), 2)
                if "MemAvailable" in mem:
                    avail_gb = round(mem["MemAvailable"] / (1024**2), 2)
                elif "MemFree" in mem:
                    avail_gb = round(mem["MemFree"] / (1024**2), 2)
                return total_gb, avail_gb
        except Exception as e:
            notes.append(f"/proc/meminfo probe failed: {e}")
    elif sys_os == "darwin":
        try:
            rc, out = _run_subproc(["sysctl", "-n", "hw.memsize"], timeout=1.5)
            if rc == 0 and out.strip().isdigit():
                total_gb = round(int(out.strip()) / (1024**3), 2)
            rc_vm, out_vm = _run_subproc(["vm_stat"], timeout=1.5)
            if rc_vm == 0:
                page_size = 4096
                free_pages = 0
                for line in out_vm.splitlines():
                    if "page size of" in line:
                        p_str = line.split("page size of")[1].split()[0]
                        if p_str.isdigit():
                            page_size = int(p_str)
                    elif "Pages free:" in line or "Pages inactive:" in line:
                        parts = line.split(":")
                        if len(parts) == 2:
                            val_str = parts[1].strip().rstrip(".")
                            if val_str.isdigit():
                                free_pages += int(val_str)
                avail_gb = round((free_pages * page_size) / (1024**3), 2)
            else:
                avail_gb = round(total_gb * 0.7, 2)
            return total_gb, avail_gb
        except Exception as e:
            notes.append(f"Darwin RAM probe failed: {e}")

    # Extreme fallback
    if total_gb <= 0.0:
        total_gb = 16.0
        avail_gb = 8.0
        notes.append("System RAM detection failed; defaulted to 16.0 GB total / 8.0 GB available.")

    return total_gb, avail_gb


def _find_nvidia_smi() -> Optional[str]:
    """Find nvidia-smi executable on PATH or standard Windows paths."""
    candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for path in candidates:
        if path == "nvidia-smi":
            # Check PATH
            from shutil import which
            found = which(path)
            if found:
                return found
        else:
            if Path(path).exists():
                return path
    return None


def _probe_nvidia_gpus(timeout: float, notes: List[str]) -> List[GpuInfo]:
    """Probe NVIDIA GPUs via nvidia-smi with graceful fallbacks."""
    smi = _find_nvidia_smi()
    if not smi:
        return []

    cmd = [
        smi,
        "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    rc, out = _run_subproc(cmd, timeout=min(timeout, 3.0))
    if rc != 0:
        # Fallback without compute_cap for legacy drivers
        cmd_fallback = [
            smi,
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        rc, out = _run_subproc(cmd_fallback, timeout=min(timeout, 2.5))
        if rc != 0:
            notes.append(f"nvidia-smi execution failed (code {rc}).")
            return []

    gpus: List[GpuInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                idx = int(parts[0])
                name = parts[1]
                # Reject [N/A] or non-numeric tokens
                if parts[2].upper() in ("[N/A]", "N/A", "NONE", "") or parts[3].upper() in ("[N/A]", "N/A", "NONE", ""):
                    notes.append(f"nvidia-smi reported [N/A] memory values on line '{line}'.")
                    continue
                tot_mib = float(parts[2])
                free_mib = float(parts[3])
                import math
                if not (math.isfinite(tot_mib) and math.isfinite(free_mib)):
                    notes.append(f"nvidia-smi reported non-finite memory values ({tot_mib}, {free_mib}).")
                    continue
                if tot_mib <= 0 or free_mib < 0 or free_mib > tot_mib:
                    notes.append(f"nvidia-smi reported invalid memory bounds (total={tot_mib}, free={free_mib}).")
                    continue
                driver = parts[4]
                cc = parts[5] if len(parts) >= 6 else ""
                gpus.append(
                    GpuInfo(
                        vendor="nvidia",
                        name=name,
                        vram_total_gb=round(tot_mib / 1024.0, 2),
                        vram_free_gb=round(free_mib / 1024.0, 2),
                        unified_memory=False,
                        driver=driver,
                        compute_capability=cc,
                        index=idx,
                    )
                )
            except Exception as e:
                notes.append(f"Failed parsing nvidia-smi line '{line}': {e}")

    return gpus


def _probe_amd_gpus(sys_os: str, timeout: float, notes: List[str]) -> List[GpuInfo]:
    """Probe AMD GPUs on Windows or Linux."""
    gpus: List[GpuInfo] = []
    if sys_os == "linux":
        rc, out = _run_subproc(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=min(timeout, 2.5))
        if rc == 0:
            try:
                data = json.loads(out)
                for idx_str, info in data.items():
                    tot = float(info.get("VRAM Total Memory (B)", 0)) / (1024**3)
                    used = float(info.get("VRAM Total Used Memory (B)", 0)) / (1024**3)
                    free = max(0.0, tot - used)
                    gpus.append(
                        GpuInfo(
                            vendor="amd",
                            name=f"AMD ROCm GPU {idx_str}",
                            vram_total_gb=round(tot, 2),
                            vram_free_gb=round(free, 2),
                            unified_memory=False,
                            driver="ROCm",
                            compute_capability="",
                            index=int(idx_str) if idx_str.isdigit() else 0,
                        )
                    )
            except Exception as e:
                notes.append(f"ROCm JSON parsing error: {e}")
    elif sys_os == "windows":
        try:
            import winreg  # type: ignore
            base_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_key) as k:
                subkeys_count, _, _ = winreg.QueryInfoKey(k)
                for i in range(subkeys_count):
                    try:
                        sub = winreg.EnumKey(k, i)
                        with winreg.OpenKey(k, sub) as sk:
                            try:
                                desc, _ = winreg.QueryValueEx(sk, "DriverDesc")
                                if isinstance(desc, str) and any(w in desc.upper() for w in ("AMD", "RADEON")):
                                    try:
                                        mem_bytes, _ = winreg.QueryValueEx(sk, "HardwareInformation.qwMemorySize")
                                        if isinstance(mem_bytes, int) and mem_bytes > 0:
                                            tot_gb = round(mem_bytes / (1024**3), 2)
                                            free_gb = max(0.0, round(tot_gb - 1.5, 2))
                                            notes.append("AMD Windows: free VRAM estimated as total - 1.5 GB.")
                                            gpus.append(
                                                GpuInfo(
                                                    vendor="amd",
                                                    name=desc,
                                                    vram_total_gb=tot_gb,
                                                    vram_free_gb=free_gb,
                                                    unified_memory=False,
                                                    driver="",
                                                    compute_capability="",
                                                    index=len(gpus),
                                                )
                                            )
                                    except (FileNotFoundError, OSError):
                                        pass
                            except (FileNotFoundError, OSError):
                                pass
                    except OSError:
                        continue
        except Exception as e:
            notes.append(f"Windows AMD registry query error: {e}")

    return gpus


def _probe_intel_gpus(sys_os: str, notes: List[str]) -> List[GpuInfo]:
    """Probe Intel GPUs (Arc dedicated vs iGPU)."""
    gpus: List[GpuInfo] = []
    if sys_os == "windows":
        try:
            import winreg  # type: ignore
            base_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_key) as k:
                subkeys_count, _, _ = winreg.QueryInfoKey(k)
                for i in range(subkeys_count):
                    try:
                        sub = winreg.EnumKey(k, i)
                        with winreg.OpenKey(k, sub) as sk:
                            try:
                                desc, _ = winreg.QueryValueEx(sk, "DriverDesc")
                                if isinstance(desc, str) and "INTEL" in desc.upper():
                                    try:
                                        mem_bytes, _ = winreg.QueryValueEx(sk, "HardwareInformation.qwMemorySize")
                                        tot_gb = round(mem_bytes / (1024**3), 2) if isinstance(mem_bytes, int) else 0.0
                                    except (FileNotFoundError, OSError):
                                        tot_gb = 0.0

                                    if tot_gb >= 6.0:
                                        free_gb = max(0.0, round(tot_gb - 1.5, 2))
                                        gpus.append(
                                            GpuInfo(
                                                vendor="intel",
                                                name=desc,
                                                vram_total_gb=tot_gb,
                                                vram_free_gb=free_gb,
                                                unified_memory=False,
                                                driver="",
                                                compute_capability="",
                                                index=len(gpus),
                                            )
                                        )
                                    else:
                                        notes.append(f"Intel GPU '{desc}' ({tot_gb} GB) detected but below 6 GB threshold; CPU-only.")
                            except (FileNotFoundError, OSError):
                                pass
                    except OSError:
                        continue
        except Exception as e:
            notes.append(f"Windows Intel GPU check error: {e}")

    return gpus


def _probe_apple_silicon(sys_os: str, arch: str, cpu_name: str, ram_total_gb: float, ram_available_gb: float) -> Optional[GpuInfo]:
    """Probe Apple Silicon Unified Memory GPU."""
    if sys_os == "darwin" and arch in ("arm64", "aarch64"):
        budget_pct = 0.65 if ram_total_gb <= 16.0 else 0.75
        vram_budget = round(ram_total_gb * budget_pct, 2)
        rc, out = _run_subproc(["sysctl", "-n", "iogpu.wired_limit_mb"], timeout=1.5)
        if rc == 0 and out.strip().isdigit():
            mb = int(out.strip())
            if mb > 0:
                vram_budget = round(mb / 1024.0, 2)

        free_ratio = ram_available_gb / max(1.0, ram_total_gb)
        vram_free = round(vram_budget * free_ratio, 2)
        return GpuInfo(
            vendor="apple",
            name=f"Apple Silicon ({cpu_name})",
            vram_total_gb=vram_budget,
            vram_free_gb=vram_free,
            unified_memory=True,
            driver="Metal",
            compute_capability="metal",
            index=0,
        )
    return None


def probe_hardware(timeout: float = 4.0) -> HardwareInfo:
    """Discover host hardware configuration safely without raising exceptions."""
    start_t = time.perf_counter()
    notes: List[str] = []

    try:
        sys_os = platform.system().lower()
    except Exception as e:
        sys_os = "windows" if sys.platform == "win32" else "linux"
        notes.append(f"OS detection failed: {e}")

    try:
        sys_arch = platform.machine().lower()
    except Exception as e:
        sys_arch = "x86_64"
        notes.append(f"Arch detection failed: {e}")

    # 1. CPU Name
    try:
        cpu_name = _probe_cpu_name(sys_os)
    except Exception as e:
        cpu_name = "Unknown CPU"
        notes.append(f"CPU name probe failed: {e}")

    # 2. CPU Cores
    try:
        logical = os.cpu_count() or 1
        physical, logical = _probe_cpu_cores(sys_os, logical, timeout, notes)
    except Exception as e:
        logical = 1
        physical = 1
        notes.append(f"CPU core count probe failed: {e}")

    # 3. Instruction Set Support
    try:
        isa = _probe_isa(sys_os, sys_arch, notes)
    except Exception as e:
        isa = {"avx": False, "avx2": False, "avx512": False, "neon": False}
        notes.append(f"ISA detection failed: {e}")

    # 4. System RAM
    try:
        ram_total, ram_avail = _probe_ram(sys_os, notes)
    except Exception as e:
        ram_total = 16.0
        ram_avail = 8.0
        notes.append(f"RAM probe failed: {e}")

    # 5. GPU Detection (NVIDIA -> AMD -> Intel -> Apple)
    gpus: List[GpuInfo] = []
    try:
        gpus.extend(_probe_nvidia_gpus(timeout, notes))
    except Exception as e:
        notes.append(f"NVIDIA GPU probe failed: {e}")

    if not gpus:
        try:
            gpus.extend(_probe_amd_gpus(sys_os, timeout, notes))
        except Exception as e:
            notes.append(f"AMD GPU probe failed: {e}")

    if not gpus:
        try:
            gpus.extend(_probe_intel_gpus(sys_os, notes))
        except Exception as e:
            notes.append(f"Intel GPU probe failed: {e}")

    if not gpus:
        try:
            apple_gpu = _probe_apple_silicon(sys_os, sys_arch, cpu_name, ram_total, ram_avail)
            if apple_gpu:
                gpus.append(apple_gpu)
        except Exception as e:
            notes.append(f"Apple Silicon probe failed: {e}")

    # Sort GPUs: best first (by free VRAM, then total VRAM)
    gpus.sort(key=lambda g: (g.vram_free_gb, g.vram_total_gb), reverse=True)
    primary_gpu = gpus[0] if gpus else None

    if primary_gpu is None:
        notes.append("No supported compute GPU detected; inference will run on CPU.")

    elapsed = round(time.perf_counter() - start_t, 3)

    return HardwareInfo(
        os=sys_os,
        arch=sys_arch,
        cpu_name=cpu_name,
        cpu_physical=physical,
        cpu_logical=logical,
        isa=isa,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpus=gpus,
        gpu=primary_gpu,
        probe_seconds=elapsed,
        notes=notes,
    )


# -----------------------------------------------------------------------------
# Model Catalogue & Specification Helpers
# -----------------------------------------------------------------------------

def catalogue() -> List[ModelSpec]:
    """Return the built-in reference catalogue for the Qwen 2.5 Coder family."""
    return [
        ModelSpec(
            tag="qwen2.5-coder:1.5b",
            blocks=28,
            head_count_kv=2,
            head_dim=128,
            hidden=1536,
            file_gb=1.0,
            native_ctx=32768,
            quant="Q4_K_M",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:3b",
            blocks=36,
            head_count_kv=2,
            head_dim=128,
            hidden=2048,
            file_gb=1.9,
            native_ctx=32768,
            quant="Q4_K_M",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:7b",
            blocks=28,
            head_count_kv=4,
            head_dim=128,
            hidden=3584,
            file_gb=4.7,
            native_ctx=32768,
            quant="Q4_K_M",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:7b-q6",
            blocks=28,
            head_count_kv=4,
            head_dim=128,
            hidden=3584,
            file_gb=6.3,
            native_ctx=32768,
            quant="Q6_K",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:14b",
            blocks=48,
            head_count_kv=8,
            head_dim=128,
            hidden=5120,
            file_gb=9.0,
            native_ctx=32768,
            quant="Q4_K_M",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:14b-q6",
            blocks=48,
            head_count_kv=8,
            head_dim=128,
            hidden=5120,
            file_gb=12.1,
            native_ctx=32768,
            quant="Q6_K",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:32b",
            blocks=64,
            head_count_kv=8,
            head_dim=128,
            hidden=5120,
            file_gb=19.9,
            native_ctx=32768,
            quant="Q4_K_M",
            source="catalogue",
        ),
        ModelSpec(
            tag="qwen2.5-coder:32b-q6",
            blocks=64,
            head_count_kv=8,
            head_dim=128,
            hidden=5120,
            file_gb=27.0,
            native_ctx=32768,
            quant="Q6_K",
            source="catalogue",
        ),
    ]


def model_spec_from_show(tag: str, show_json: Dict[str, Any], size_bytes: Optional[int] = None) -> ModelSpec:
    """Construct a ModelSpec from Ollama's /api/show response or fallback to catalogue."""
    cat_map = {m.tag: m for m in catalogue()}
    base_cat = cat_map.get(tag)

    # Check model_info dict from GGUF metadata
    info = show_json.get("model_info", {})
    details = show_json.get("details", {})

    # Extract block count (<arch>.block_count)
    blocks = None
    head_count_kv = None
    head_dim = 128
    have_key_length = False
    head_count_all = None
    hidden = None
    native_ctx = None

    for k, v in info.items():
        if k.endswith(".block_count") and isinstance(v, int):
            blocks = v
        elif k.endswith(".attention.head_count_kv") and isinstance(v, int):
            head_count_kv = v
        elif k.endswith(".attention.head_count") and isinstance(v, int):
            head_count_all = v
            if head_count_kv is None:
                head_count_kv = v
        elif k.endswith(".embedding_length") and isinstance(v, int):
            hidden = v
        elif k.endswith(".context_length") and isinstance(v, int):
            native_ctx = v
        elif (k.endswith(".attention.key_length") or k.endswith(".attention.value_length")) and isinstance(v, int):
            head_dim = v
            have_key_length = True

    if not have_key_length and hidden and head_count_all:
        head_dim = max(1, hidden // head_count_all)       # e.g. 5120 / 40 heads = 128

    if base_cat is not None:
        blocks = blocks or base_cat.blocks
        head_count_kv = head_count_kv or base_cat.head_count_kv
        hidden = hidden or base_cat.hidden
        native_ctx = native_ctx or base_cat.native_ctx
        head_dim = head_dim or base_cat.head_dim

    blocks = blocks or 48
    head_count_kv = head_count_kv or 8
    hidden = hidden or 5120
    native_ctx = native_ctx or 32768

    quant = details.get("quantization_level") or (base_cat.quant if base_cat else "Q4_K_M")

    if size_bytes and size_bytes > 0:
        file_gb = round(size_bytes / (1024**3), 2)
    elif base_cat is not None:
        file_gb = base_cat.file_gb
    else:
        file_gb = 9.0

    return ModelSpec(
        tag=tag,
        blocks=blocks,
        head_count_kv=head_count_kv,
        head_dim=head_dim,
        hidden=hidden,
        file_gb=file_gb,
        native_ctx=native_ctx,
        quant=quant,
        source="ollama" if info else ("catalogue" if base_cat else "estimate"),
    )


# -----------------------------------------------------------------------------
# Tier Determination & Auto-Configuration
# -----------------------------------------------------------------------------

def choose_tier(hw: HardwareInfo) -> str:
    """Determine host hardware performance tier.
    
    Precedence rules:
      - High:  VRAM >= 16 GB or System RAM >= 64 GB
      - Mid:   (8 GB <= VRAM < 16 GB) or System RAM >= 32 GB
      - Low:   All other configurations
    """
    vram = round(hw.gpu.vram_total_gb, 0) if hw.gpu else 0.0
    ram = round(hw.ram_total_gb, 0)

    if vram >= 16.0 or ram >= 64.0:
        return "high"
    elif (8.0 <= vram < 16.0) or ram >= 32.0:
        return "mid"
    return "low"


def recommend_model(hw: HardwareInfo) -> ModelSpec:
    """Select the optimal model recommendation based on hardware capability."""
    tier = choose_tier(hw)
    cat = {m.tag: m for m in catalogue()}
    vram_free = hw.gpu.vram_free_gb if hw.gpu else 0.0
    vram_tot = round(hw.gpu.vram_total_gb, 0) if hw.gpu else 0.0

    # CPU-only hosts: memory bandwidth caps token generation (~7-10 tok/s on 7B vs ~3 tok/s on 14B/32B)
    if hw.gpu is None or hw.gpu.vendor == "none" or vram_tot <= 0.0:
        if hw.ram_total_gb <= 8.0 or hw.ram_available_gb < 6.0:
            return cat["qwen2.5-coder:3b"]
        return cat["qwen2.5-coder:7b"]

    if tier == "high":
        if vram_tot >= 24.0 and vram_free >= 20.0:
            return cat.get("qwen2.5-coder:32b-q6") or cat["qwen2.5-coder:32b"]
        elif vram_tot >= 16.0 or hw.ram_total_gb >= 64.0:
            return cat["qwen2.5-coder:32b"]
        return cat["qwen2.5-coder:14b"]
    elif tier == "mid":
        if vram_tot >= 8.0 and vram_free >= 7.0:
            return cat["qwen2.5-coder:14b"]
        elif hw.ram_total_gb >= 32.0:
            return cat["qwen2.5-coder:14b"]
        return cat.get("qwen2.5-coder:7b-q6") or cat["qwen2.5-coder:7b"]
    else:  # low
        if hw.ram_total_gb <= 8.0 or hw.ram_available_gb < 6.0:
            return cat["qwen2.5-coder:3b"]
        return cat["qwen2.5-coder:7b"]


def _estimate_kv_cache_gb(
    blocks: int,
    head_count_kv: int,
    head_dim: int,
    num_ctx: int,
    kv_cache_type: str = "fp16",
) -> float:
    """Calculate total KV cache memory in gigabytes for fp16 (2B), q8_0 (1B), or q4_0 (0.5B)."""
    kt = (kv_cache_type or "fp16").strip().lower()
    if kt in ("q4_0", "q4", "int4"):
        bytes_per_elem = 0.5
    elif kt in ("q8_0", "q8", "int8"):
        bytes_per_elem = 1.0
    else:
        bytes_per_elem = 2.0
    bytes_per_tok = 2 * blocks * head_count_kv * head_dim * bytes_per_elem
    return (bytes_per_tok * num_ctx) / (1024**3)


def auto_configure(
    hw: HardwareInfo,
    model: Optional[ModelSpec] = None,
    installed: Optional[List[ModelSpec]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    target_ctx: Optional[int] = None,
) -> ExecutionProfile:
    """Compute deterministic execution profile without performing I/O.
    
    Pure function implementing layer offloading mathematics, memory headroom
    guards, and context/thread auto-tuning.
    """
    overrides = overrides or {}
    tier = choose_tier(hw)
    cat = catalogue()
    cat_by_tag = {m.tag: m for m in cat}

    # 1. Candidate model selection
    if model is None:
        tag_req = overrides.get("model") or overrides.get("configured_model") or "qwen2.5-coder:14b"
        model = cat_by_tag.get(tag_req) or ModelSpec(
            tag=tag_req,
            blocks=48,
            head_count_kv=8,
            head_dim=128,
            hidden=5120,
            file_gb=9.0,
            native_ctx=32768,
            quant="Q4_K_M",
            source="estimate",
        )

    # 2. Safety margin taken from FREE VRAM. Free VRAM already excludes the desktop and every running program, so a
    # large reserve double-counts them: 2.0 GB left the 14b model at 40/49 layers on a 12 GB GPU with 10.7 GB free
    # (~16 tok/s instead of ~54 all-GPU), and builder calls hit their timeout. Ollama alone fits the whole model there.
    if "reserve_gb" in overrides:
        reserve_gb = max(0.0, float(overrides["reserve_gb"]))
    else:
        reserve_gb = 0.5

    # 3. Context window tuning (default 8192, 4096 <= num_ctx <= 16384, step 1024)
    force_cpu = bool(overrides.get("force_cpu"))
    has_gpu = (hw.gpu is not None) and (hw.gpu.vendor != "none") and not force_cpu

    if "ctx" in overrides:
        num_ctx = int(overrides["ctx"])
    elif target_ctx is not None:
        num_ctx = target_ctx
    elif not has_gpu:
        # CPU-only mode: keep context compact (4096) to avoid quadratic prompt prefill slowdown in system RAM
        num_ctx = 4096
    else:
        num_ctx = 8192
        # Scale DOWN to 4096 if RAM headroom < 3 GB after weights on CPU
        ram_headroom = hw.ram_available_gb - (model.file_gb + 2.0)
        if ram_headroom < 3.0 and hw.ram_total_gb <= 16.0:
            num_ctx = 4096
        # Scale UP to 12288/16384 only when generous headroom exists
        elif hw.gpu and not hw.gpu.unified_memory:
            free_excess = hw.gpu.vram_free_gb - (reserve_gb + model.file_gb)
            if free_excess >= 4.0 and model.native_ctx >= 16384:
                num_ctx = 16384
            elif free_excess >= 2.0 and model.native_ctx >= 12288:
                num_ctx = 12288
        elif hw.ram_available_gb >= 32.0 and model.native_ctx >= 16384:
            num_ctx = 16384

    # Enforce constraints: multiple of 1024, clamp [4096, min(16384, native_ctx)]
    num_ctx = max(4096, min(16384, min(model.native_ctx, (num_ctx // 1024) * 1024)))

    # 4. Total units and KV cache calculation (supports quantized KV cache q8_0 / q4_0)
    kv_cache_type = str(overrides.get("kv_cache_type") or os.environ.get("OLLAMA_KV_CACHE_TYPE") or "fp16").strip().lower()
    if overrides.get("compact_ctx") and "ctx" not in overrides and target_ctx is None:
        if hw.gpu and not hw.gpu.unified_memory and hw.gpu.vram_free_gb < 8.5:
            num_ctx = 6144
    total_units = model.blocks + 1
    kv_total_gb = _estimate_kv_cache_gb(model.blocks, model.head_count_kv, model.head_dim, num_ctx, kv_cache_type=kv_cache_type)

    # 5. Layer offloading math

    if not has_gpu:
        vram_budget = 0.0
    elif hw.gpu.unified_memory:  # type: ignore
        # Apple Silicon unified memory pool
        vram_budget = max(0.0, hw.gpu.vram_free_gb - reserve_gb)  # type: ignore
    else:
        vram_budget = max(0.0, hw.gpu.vram_free_gb - reserve_gb)  # type: ignore

    # Calculate safe assigned GPU layers
    assigned = 0
    if has_gpu and vram_budget > 0.0:
        for n in range(total_units, -1, -1):
            if n == 0:
                assigned = 0
                break
            weight_part = (model.file_gb / total_units) * n
            kv_part = kv_total_gb * min(1.0, n / model.blocks)
            v_est = weight_part + kv_part + CALIBRATED_VRAM_OVERHEAD_GB
            if v_est <= vram_budget:
                assigned = n
                break

    # 6. Apply layer overrides if specified
    source = "override" if any(k != "configured_model" for k in overrides) else "auto"
    if force_cpu:
        assigned = 0
    elif "gpu_layers" in overrides:
        raw_layers = overrides["gpu_layers"]
        if raw_layers is None or (isinstance(raw_layers, str) and raw_layers.lower() in ("all", "auto", "max")):
            assigned = total_units
        else:
            try:
                l_val = int(raw_layers)
                if l_val >= total_units or l_val in (99, -1):
                    assigned = total_units
                elif l_val <= 0:
                    assigned = 0
                else:
                    assigned = min(total_units, l_val)
            except (ValueError, TypeError):
                pass

    # 7. Determine mode and num_gpu value
    if assigned == total_units and has_gpu and not force_cpu:
        mode = "gpu"
        num_gpu: Optional[int] = 99  # Never blocks; send 99 for complete offload
    elif assigned == 0 or not has_gpu or force_cpu:
        mode = "cpu"
        num_gpu = 0
    else:
        mode = "hybrid"
        num_gpu = assigned

    # 8. Compute memory usage estimates
    if mode == "cpu":
        est_vram_gb = 0.0
    else:
        weight_v = (model.file_gb / total_units) * assigned
        kv_v = kv_total_gb * min(1.0, assigned / model.blocks)
        est_vram_gb = round(weight_v + kv_v + CALIBRATED_VRAM_OVERHEAD_GB, 2)

    cpu_units = total_units - assigned
    if cpu_units > 0:
        weight_r = (model.file_gb / total_units) * cpu_units
        kv_r = kv_total_gb * ((model.blocks - min(assigned, model.blocks)) / model.blocks)
        est_ram_gb = round(weight_r + kv_r + 1.5, 2)
    else:
        est_ram_gb = 1.5  # Base runtime process memory

    # 9. Verify if execution fits within host limits
    if hw.gpu and hw.gpu.unified_memory:
        # Unified memory pool (Apple): total allocation cannot exceed available RAM minus 2 GB
        total_unified_est = est_vram_gb + max(0.0, est_ram_gb - 1.5)
        fits = total_unified_est <= max(0.0, hw.ram_available_gb - 2.0)
    else:
        fits_vram = (est_vram_gb <= (hw.gpu.vram_free_gb if hw.gpu else 0.0))
        fits_ram = (est_ram_gb <= max(0.0, hw.ram_available_gb - 2.0))
        fits = fits_vram and fits_ram

    # 10. Thread allocation
    # Note on Intel hybrid architecture: cpu_physical includes efficiency cores.
    if "threads" in overrides:
        num_thread = max(1, min(32, int(overrides["threads"])))
    else:
        phys = max(1, min(hw.cpu_physical, 32))
        num_thread = min(phys, 8) if mode == "gpu" else phys

    # 11. Warnings generation and installed model fallback
    warnings: List[str] = []
    if mode == "cpu":
        warnings.append("Inference running on CPU (< 5 tok/s estimated). Token generation will be slow.")
    elif mode == "hybrid":
        warnings.append(f"Hybrid GPU/CPU execution ({assigned}/{total_units} units on GPU; 5-15 tok/s estimated).")

    if tier == "high" and mode == "cpu":
        warnings.append("High tier machine running on CPU; token throughput will be severely constrained.")

    # Model replacement fallback: if candidate does not fit and wasn't explicitly forced via CLI/env
    if not fits and "model" not in overrides and installed:
        smaller_installed = sorted(
            [m for m in installed if m.file_gb < model.file_gb],
            key=lambda m: m.file_gb,
            reverse=True,
        )
        for inst in smaller_installed:
            cand_prof = auto_configure(hw, model=inst, installed=None, overrides=overrides, target_ctx=target_ctx)
            if cand_prof.fits:
                cand_warnings = [
                    f"Configured model '{model.tag}' does not fit available memory; automatically selected installed model '{inst.tag}' for this run."
                ]
                cand_warnings.extend(cand_prof.warnings)
                return ExecutionProfile(
                    tier=cand_prof.tier,
                    mode=cand_prof.mode,
                    model_tag=cand_prof.model_tag,
                    quant=cand_prof.quant,
                    num_gpu=cand_prof.num_gpu,
                    gpu_units=cand_prof.gpu_units,
                    total_units=cand_prof.total_units,
                    num_ctx=cand_prof.num_ctx,
                    num_thread=cand_prof.num_thread,
                    reserve_gb=cand_prof.reserve_gb,
                    est_vram_gb=cand_prof.est_vram_gb,
                    est_ram_gb=cand_prof.est_ram_gb,
                    fits=cand_prof.fits,
                    warnings=cand_warnings,
                    source=cand_prof.source,
                )

    # CPU-only speed optimization: if running on CPU with a large model (> 5.5 GB, e.g. 14B/32B)
    # and the user did not explicitly force --model, prefer an installed <= 5.5 GB model (e.g. 7B/3B)
    # for 2-3x faster CPU token generation.
    if mode == "cpu" and model.file_gb > 5.5 and "model" not in overrides and installed:
        fast_cpu_installed = sorted(
            [m for m in installed if m.file_gb <= 5.5],
            key=lambda m: m.file_gb,
            reverse=True,
        )
        for inst in fast_cpu_installed:
            cand_prof = auto_configure(hw, model=inst, installed=None, overrides=overrides, target_ctx=target_ctx)
            if cand_prof.fits:
                cand_warnings = [
                    f"CPU-only mode: automatically selected faster installed model '{inst.tag}' instead of '{model.tag}' for 2-3x higher CPU throughput."
                ]
                cand_warnings.extend(cand_prof.warnings)
                return ExecutionProfile(
                    tier=cand_prof.tier,
                    mode=cand_prof.mode,
                    model_tag=cand_prof.model_tag,
                    quant=cand_prof.quant,
                    num_gpu=cand_prof.num_gpu,
                    gpu_units=cand_prof.gpu_units,
                    total_units=cand_prof.total_units,
                    num_ctx=cand_prof.num_ctx,
                    num_thread=cand_prof.num_thread,
                    reserve_gb=cand_prof.reserve_gb,
                    est_vram_gb=cand_prof.est_vram_gb,
                    est_ram_gb=cand_prof.est_ram_gb,
                    fits=cand_prof.fits,
                    warnings=cand_warnings,
                    source=cand_prof.source,
                )

    if not fits:
        rec_m = recommend_model(hw)
        warnings.append(
            f"WARNING: Model '{model.tag}' exceeds safe host memory (est VRAM {est_vram_gb} GB, est RAM {est_ram_gb} GB)! "
            f"Recommended: install a smaller model via 'ollama pull {rec_m.tag}'."
        )

    if overrides:
        if "gpu_layers" in overrides and est_vram_gb > vram_budget and has_gpu:
            warnings.append(
                f"Unsafe override: gpu_layers={overrides['gpu_layers']} estimated at {est_vram_gb} GB VRAM exceeds budget {vram_budget:.1f} GB."
            )

    return ExecutionProfile(
        tier=tier,
        mode=mode,
        model_tag=model.tag,
        quant=model.quant,
        num_gpu=num_gpu,
        gpu_units=assigned,
        total_units=total_units,
        num_ctx=num_ctx,
        num_thread=num_thread,
        reserve_gb=reserve_gb,
        est_vram_gb=est_vram_gb,
        est_ram_gb=est_ram_gb,
        fits=fits,
        warnings=warnings,
        source=source,
    )


# -----------------------------------------------------------------------------
# Diagnostic Banner & Serialization
# -----------------------------------------------------------------------------

def format_banner(
    hw: HardwareInfo,
    profile: ExecutionProfile,
    notes: Optional[List[str]] = None,
    profile_path: Optional[str] = None,
    installed_tags: Optional[set[str]] = None,
) -> str:
    """Format a clean, pure-ASCII startup diagnostic banner."""
    isa_features: List[str] = []
    if hw.isa.get("avx2"):
        isa_features.append("AVX2")
    elif hw.isa.get("avx"):
        isa_features.append("AVX")
    if hw.isa.get("avx512"):
        isa_features.append("AVX-512")
    if hw.isa.get("neon"):
        isa_features.append("NEON")
    isa_str = ", ".join(isa_features) if isa_features else "Standard ISA"

    lines = [
        "======================================================================",
        "                     OGUN HARDWARE PROFILE & RUNTIME",
        "======================================================================",
        f"  CPU:        {hw.cpu_name} ({hw.cpu_physical}C / {hw.cpu_logical}T, {isa_str})",
        f"  System RAM: {hw.ram_total_gb:.1f} GB Total | {hw.ram_available_gb:.1f} GB Available",
    ]

    if hw.gpu:
        import math
        tot_valid = math.isfinite(hw.gpu.vram_total_gb) and hw.gpu.vram_total_gb > 0
        free_valid = (
            math.isfinite(hw.gpu.vram_free_gb)
            and 0 <= hw.gpu.vram_free_gb <= hw.gpu.vram_total_gb
        )
        if tot_valid and free_valid:
            mem_str = f"({hw.gpu.vram_total_gb:.1f} GB VRAM, {hw.gpu.vram_free_gb:.1f} GB Free)"
        elif tot_valid:
            mem_str = f"({hw.gpu.vram_total_gb:.1f} GB VRAM)"
        else:
            mem_str = ""
        gpu_line = f"  GPU:        {hw.gpu.name}"
        if mem_str:
            gpu_line += f" {mem_str}"
        lines.append(gpu_line)
        drv_info = f"Driver {hw.gpu.driver}" if hw.gpu.driver else "Driver Active"
        if hw.gpu.compute_capability:
            drv_info += f" (Compute {hw.gpu.compute_capability})"
        lines.append(f"  GPU Driver: {drv_info}")
    else:
        lines.append("  GPU:        None (CPU-only execution)")

    lines.append("----------------------------------------------------------------------")
    rec_spec = recommend_model(hw)
    rec_inst = ("yes" if rec_spec.tag in installed_tags else "no") if installed_tags is not None else ("yes" if rec_spec.tag == "qwen2.5-coder:14b" else "no")
    model_inst = ("yes" if profile.model_tag in installed_tags else "no") if installed_tags is not None else "yes"

    lines.append(f"  Tier:       {profile.tier.upper()}")
    fits_str = "yes" if profile.fits else "no (oom risk)"
    lines.append(f"  Model:      {profile.model_tag} [{profile.quant}] (fits: {fits_str}, installed: {model_inst})")
    lines.append(f"  Recommended: {rec_spec.tag} [{rec_spec.quant}] (installed: {rec_inst})")
    lines.append(
        f"  Mode:       {profile.mode.upper()} ({profile.gpu_units}/{profile.total_units} units on GPU, "
        f"{profile.total_units - profile.gpu_units} in RAM)"
    )
    lines.append(
        f"  Options:    num_gpu: {profile.num_gpu} | num_ctx: {profile.num_ctx} | num_thread: {profile.num_thread}"
    )
    lines.append(
        f"  Memory Est: VRAM: {profile.est_vram_gb:.1f} GB (Reserve: {profile.reserve_gb:.1f} GB) | "
        f"RAM: {profile.est_ram_gb:.1f} GB"
    )

    path_display = profile_path or "runtime/hardware_profile.json"
    lines.append(
        f"  Status:     {profile.source.upper()} [Probe: {hw.probe_seconds:.2f}s] -> {path_display}"
    )

    all_warnings = list(profile.warnings)
    if notes:
        all_warnings.extend(notes)

    if all_warnings:
        lines.append("----------------------------------------------------------------------")
        for w in all_warnings[:3]:
            lines.append(f"  Notice:     {w}")

    lines.append("======================================================================")
    banner = "\n".join(lines)
    # Strict ASCII verification
    return banner.encode("ascii", errors="replace").decode("ascii")


def profile_to_dict(profile: ExecutionProfile) -> Dict[str, Any]:
    """Serialize ExecutionProfile dataclass to dictionary."""
    return asdict(profile)


def profile_from_dict(d: Dict[str, Any]) -> ExecutionProfile:
    """Deserialize ExecutionProfile from dictionary."""
    return ExecutionProfile(
        tier=str(d.get("tier", "mid")),
        mode=str(d.get("mode", "hybrid")),
        model_tag=str(d.get("model_tag", "qwen2.5-coder:14b")),
        quant=str(d.get("quant", "Q4_K_M")),
        num_gpu=d.get("num_gpu"),
        gpu_units=int(d.get("gpu_units", 0)),
        total_units=int(d.get("total_units", 49)),
        num_ctx=int(d.get("num_ctx", 8192)),
        num_thread=int(d.get("num_thread", 4)),
        reserve_gb=float(d.get("reserve_gb", 1.5)),
        est_vram_gb=float(d.get("est_vram_gb", 0.0)),
        est_ram_gb=float(d.get("est_ram_gb", 1.5)),
        fits=bool(d.get("fits", True)),
        warnings=list(d.get("warnings", [])),
        source=str(d.get("source", "auto")),
    )


def hardware_to_dict(hw: HardwareInfo) -> Dict[str, Any]:
    """Serialize HardwareInfo dataclass to dictionary."""
    return asdict(hw)


def fingerprint(hw: HardwareInfo) -> str:
    """Compute a stable hardware identity hash excluding volatile free memory."""
    parts = [
        hw.os,
        hw.arch,
        hw.cpu_name.strip(),
        str(hw.cpu_physical),
        str(hw.cpu_logical),
        str(round(hw.ram_total_gb, 1)),
    ]
    for g in hw.gpus:
        parts.extend([g.vendor, g.name, str(round(g.vram_total_gb, 1)), g.driver])
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_profile(
    path: Path | str,
    hw: HardwareInfo,
    profile: ExecutionProfile,
    overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    """Atomically write hardware profile JSON to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint(hw),
        "hardware": hardware_to_dict(hw),
        "profile": profile_to_dict(profile),
        "overrides": overrides or {},
    }
    tmp_path = p.with_suffix(f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
        return True
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False


def load_profile(path: Path | str) -> Optional[Dict[str, Any]]:
    """Load profile JSON tolerating UTF-8 BOM, backing up corrupt files."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and "profile" in data:
            return data
        return None
    except Exception:
        # Corrupt file: backup and return None
        try:
            corrupt_name = f"{p.name}.corrupt-{int(time.time())}"
            backup_path = p.parent / corrupt_name
            os.replace(p, backup_path)
        except Exception:
            pass
        return None


# -----------------------------------------------------------------------------
# Standalone CLI Entry Point
# -----------------------------------------------------------------------------

def adjust_for_resident_models(hw: HardwareInfo, resident_vram_gb: float, resident_ram_gb: float = 0.0) -> HardwareInfo:
    """Count the memory that models Ollama has ALREADY loaded as available for planning.

    The driver's "free VRAM" excludes the model(s) Ollama currently holds. Loading (or reloading) the model replaces them,
    so a re-probe taken while our own model is loaded would see (total - model - other apps) and wrongly conclude there is
    no room for it (it measured 1.9 GB free and chose CPU-only when 6+ GB were really available). Add that memory back,
    never beyond the card's total VRAM / the machine's total RAM. Pure; a no-op without a discrete GPU or for unified memory.
    """
    try:
        import math
        raw_gpu = float(resident_vram_gb or 0.0)
        gpu_extra = max(0.0, raw_gpu) if math.isfinite(raw_gpu) else 0.0
        raw_ram = float(resident_ram_gb or 0.0)
        ram_extra = max(0.0, raw_ram) if math.isfinite(raw_ram) else 0.0
        new_hw = hw
        notes = list(hw.notes)
        g = hw.gpu
        if g is not None and not g.unified_memory and gpu_extra > 0.05:
            new_free = round(min(g.vram_total_gb, g.vram_free_gb + gpu_extra), 2)
            if new_free > g.vram_free_gb:
                ng = _dc_replace(g, vram_free_gb=new_free)
                gpus = [ng if (x.index == g.index and x.name == g.name) else x for x in hw.gpus]
                notes.append(f"Counted {new_free - g.vram_free_gb:.1f} GB of VRAM held by models Ollama has loaded as available "
                             "(reloading replaces them).")
                new_hw = _dc_replace(new_hw, gpus=gpus, gpu=ng)
        if ram_extra > 0.05:
            new_ram = round(min(hw.ram_total_gb, hw.ram_available_gb + ram_extra), 2)
            if new_ram > hw.ram_available_gb:
                notes.append(f"Counted {new_ram - hw.ram_available_gb:.1f} GB of RAM held by models Ollama has loaded as available.")
                new_hw = _dc_replace(new_hw, ram_available_gb=new_ram)
        return _dc_replace(new_hw, notes=notes) if notes != list(hw.notes) else new_hw
    except Exception:
        return hw


def query_ollama(host: str = "http://127.0.0.1:11434", timeout: float = 1.5, family: Optional[str] = None,
                 max_models: int = 8) -> Dict[str, Any]:
    """Ask a running Ollama what is installed and what is currently loaded. Never raises.

    Returns {"reachable": bool, "installed": [ModelSpec], "resident": [names], "resident_vram_gb": float,
    "resident_ram_gb": float}. `family` (e.g. "qwen2.5-coder") keeps only installed models of that family as automatic
    fallbacks (a vision or chat model is no substitute for a coding model). Real per-model architecture comes from
    /api/show, so the memory math uses each model's own layers and KV heads instead of assuming one architecture.
    """
    import urllib.request

    out: Dict[str, Any] = {"reachable": False, "installed": [], "resident": [], "resident_vram_gb": 0.0, "resident_ram_gb": 0.0}
    base = (host or "").rstrip("/")
    if not base:
        return out

    def _call(path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            data = json.dumps(payload).encode("utf-8") if payload is not None else None
            req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET",
                                         headers={"Content-Type": "application/json"} if data is not None else {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    tags = _call("/api/tags")
    if tags is None:
        return out
    out["reachable"] = True
    installed: List[ModelSpec] = []
    for m in (tags.get("models") or [])[:max_models]:
        try:
            tag = m.get("name") or m.get("model")
            if not tag or (family and str(tag).split(":")[0] != family):
                continue
            show = _call("/api/show", {"model": tag}) or {}
            installed.append(model_spec_from_show(tag, show, size_bytes=m.get("size")))
        except Exception:
            continue
    out["installed"] = installed
    ps = _call("/api/ps") or {}
    for m in ps.get("models") or []:
        try:
            size = float(m.get("size") or 0)
            in_vram = float(m.get("size_vram") or 0)
            out["resident"].append(m.get("name") or m.get("model") or "?")
            out["resident_vram_gb"] += in_vram / (1024 ** 3)
            out["resident_ram_gb"] += max(0.0, size - in_vram) / (1024 ** 3)
        except Exception:
            continue
    out["resident_vram_gb"] = round(out["resident_vram_gb"], 2)
    out["resident_ram_gb"] = round(out["resident_ram_gb"], 2)
    return out


def main():
    """CLI diagnostic tool: probe host hardware and display banner or JSON."""
    as_json = "--json" in sys.argv
    hw = probe_hardware()
    profile = auto_configure(hw)

    if as_json:
        data = {
            "fingerprint": fingerprint(hw),
            "hardware": hardware_to_dict(hw),
            "profile": profile_to_dict(profile),
        }
        print(json.dumps(data, indent=2))
    else:
        banner = format_banner(hw, profile)
        print(banner)


if __name__ == "__main__":
    main()
