"""
Device utility functions for MLIP models.

This module provides device selection utilities without importing
heavy dependencies like lightning/pytorch-lightning.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _visible_physical_indices() -> list[int] | None:
    """Physical GPU indices, in torch-ordinal order, from CUDA_VISIBLE_DEVICES.

    Returns None when the variable is unset (torch ordinal == nvidia-smi index),
    and an empty list when it names devices by UUID or MIG slice, which cannot
    be matched to nvidia-smi indices.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    entries = [e.strip() for e in value.split(",") if e.strip()]
    if all(e.isdigit() for e in entries):
        return [int(e) for e in entries]
    return []


def get_best_device(device_preference: str = "auto") -> str:
    """
    Determine the best device for computation based on available hardware.

    Args:
        device_preference: Device preference ("auto", "cpu", "cuda", "mps")

    Returns:
        Best available device string
    """
    if device_preference != "auto":
        if device_preference.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif device_preference == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return device_preference

    # Auto device selection based on available hardware

    # nvidia-smi sees a GPU even when this torch build cannot use it -- most
    # often a driver older than the torch build's CUDA needs (CUDA 13 builds
    # need driver >= 580). Choosing it then fails at model load, so ask torch.
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return "mps"
        import shutil

        if shutil.which("nvidia-smi"):
            logger.warning(
                "An NVIDIA GPU is present but torch cannot use it (is the driver "
                "older than this torch build's CUDA %s needs?); using CPU.",
                torch.version.cuda,
            )
        return "cpu"

    # 1. Try nvidia-smi first (most reliable for physical GPU detection)
    try:
        import subprocess

        # Query nvidia-smi for real-time memory usage
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_memory = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) >= 3:
                    try:
                        gpu_idx = int(parts[0])
                        # Handle case where memory reporting is not supported (e.g. [N/A])
                        try:
                            memory_used = (
                                int(parts[1].replace(" MiB", "").replace(" ", ""))
                                * 1024**2
                            )
                            memory_total = (
                                int(parts[2].replace(" MiB", "").replace(" ", ""))
                                * 1024**2
                            )
                            memory_free = memory_total - memory_used
                        except (ValueError, IndexError):
                            # Memory info not available, assume it's usable and give high priority
                            # This handles cases like MIG slices or restricted access where GPU is present but stats hidden
                            memory_free = float("inf")

                        gpu_memory.append((gpu_idx, memory_free))
                    except ValueError:
                        continue

            # nvidia-smi numbers every physical GPU; torch numbers only the
            # visible ones. Translate, or leave the choice to torch below when
            # the mapping is unknown (UUIDs or MIG slices in CUDA_VISIBLE_DEVICES).
            visible = _visible_physical_indices()
            if visible is not None:
                gpu_memory = [
                    (visible.index(idx), free)
                    for idx, free in gpu_memory
                    if idx in visible
                ]
            gpu_memory = [g for g in gpu_memory if g[0] < torch.cuda.device_count()]

            if gpu_memory:
                # Sort by free memory (descending) and select the GPU with most free VRAM
                gpu_memory.sort(key=lambda x: x[1], reverse=True)
                best_gpu = gpu_memory[0][0]

                # Format memory string for logging
                mem_str = "unknown"
                if gpu_memory[0][1] != float("inf"):
                    mem_str = f"{gpu_memory[0][1] / 1024**3:.1f} GB"

                logger.info(
                    f"Selected GPU {best_gpu} with {mem_str} free VRAM (using nvidia-smi)"
                )
                return f"cuda:{best_gpu}"
    except Exception as e:
        logger.debug(f"Could not use nvidia-smi for device selection: {e}")

    # 2. Fallback to PyTorch detection
    if torch.cuda.is_available():
        # Fallback: Get GPU with most free VRAM using PyTorch (less accurate but works)
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_free = memory_total - memory_reserved
            gpu_memory.append((i, memory_free))

        # Sort by free memory (descending) and select the GPU with most free VRAM
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        best_gpu = gpu_memory[0][0]

        logger.info(
            f"Selected GPU {best_gpu} with {gpu_memory[0][1] / 1024**3:.1f} GB free VRAM (using PyTorch)"
        )
        return f"cuda:{best_gpu}"

    elif torch.backends.mps.is_available():
        logger.info("Selected MPS (Apple Silicon) device")
        return "mps"

    else:
        logger.info("Selected CPU device")
        return "cpu"
