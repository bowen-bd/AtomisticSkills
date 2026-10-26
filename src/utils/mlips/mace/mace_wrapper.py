"""
MACE model wrapper for MLIP Agent
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
from ase.calculators.calculator import Calculator

from src.utils.mlips.base import MLIPModel

logger = logging.getLogger(__name__)

# How a calculator may be accelerated. CuEquivariance ("cueq") swaps MACE's
# e3nn tensor-product kernels for fused CUDA ones, which needs both its Python
# front end and its compiled ops package, and only helps on a GPU.
#
# "auto" is deliberately forgiving. A host with no GPU, no cuequivariance
# installed (the CPU-only environments, or a run that did not sync the cu130
# extra), or a card that cuequivariance has no kernels for must still be able
# to run MACE, so an accelerator that cannot be used is a log line and a slower
# path, never an exception out of load(). Asking for "cueq" is a request, not a
# guarantee, for the same reason: it is honoured where it works.
ACCELERATORS = ("auto", "cueq", "none")
_CUEQ_MODULES = ("cuequivariance_torch", "cuequivariance_ops_torch")


def cueq_installed() -> bool:
    """Whether both halves of CuEquivariance can be imported."""
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in _CUEQ_MODULES)


def _is_cuda(device: str) -> bool:
    """Whether an ASE/``torch`` device string names a CUDA device."""
    return str(device).split(":")[0] == "cuda"


# Available MACE checkpoints and models
AVAILABLE_MACE_MODELS = {
    # MACE-MH (Multi-Head) models - MACE-MH-1 is the latest recommended default
    "MACE": "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model",
    "MACE-MP": "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model",
    "MACE-MH-1": "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model",
    "MACE-MH-0": "https://huggingface.co/mace-foundations/mace-mh-0/resolve/main/mace-mh-0.model",
    # MACE-MP (Materials Project) models
    "MACE-MP-small": "small",
    "MACE-MP-medium": "medium",
    "MACE-MP-large": "large",
    # MACE-MP second generation (recommended for fine-tuning/stability)
    "MACE-MP-small-0b": "small-0b",
    "MACE-MP-medium-0b": "medium-0b",
    "MACE-MP-small-0b2": "small-0b2",
    "MACE-MP-medium-0b2": "medium-0b2",
    "MACE-MP-large-0b2": "large-0b2",
    "MACE-MP-medium-0b3": "medium-0b3",
    # MACE-MPA (trained on enlarged datasets)
    "MACE-MPA-0": "medium-mpa-0",
    # MACE-OMAT (trained on OMAT dataset)
    "MACE-OMAT-0-small": "small-omat-0",
    "MACE-OMAT-0-medium": "medium-omat-0",
    # MACE-OFF (Organic Transferable Force Fields)
    "MACE-OFF23-small": "MACE-OFF23-small",
    "MACE-OFF23-medium": "MACE-OFF23-medium",
    "MACE-OFF23-large": "MACE-OFF23-large",
    # MACE-MATPES (Fine-tuned on MATPES dataset)
    "MACE-MATPES-PBE-0": "mace-matpes-pbe-0",
    "MACE-MATPES-R2SCAN-0": "mace-matpes-r2scan-0",
    # MACE-OMOL (Organic Molecules with charges/spins)
    "MACE-OMOL-extra-large": "extra_large",
    # ANI-CC (ANI Coupled-Cluster accuracy)
    "MACE-ANI-CC": "mace-anicc",
}


# Try to import MACE components with clear error messages
MACE_AVAILABLE = False

try:
    import mace  # noqa: F401 — availability check only; imported for ImportError side-effect

    MACE_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "MACE is not installed in this Python environment. MACE runs in the 'mlip' "
        "environment; start the command through the launcher, e.g.\n"
        "  venv/run mlip python <script>.py ...\n"
        f"Original error: {e}"
    ) from e


class MACEWrapper(MLIPModel):
    """
    MACE model wrapper implementing the MLIPModel interface.
    """

    def __init__(
        self,
        model_name: str = "MACE",
        model_version: str = "latest",
        device: str = "auto",
        head: Optional[str] = None,
        accelerator: str = "auto",
    ):
        """
        Initialize MACE wrapper.

        Args:
            model_name: Name of the model ("MACE", "MACE-MP", etc.)
            model_version: Version of the model
            device: Device to run the model on ("auto", "cpu", "cuda")
            head: Optional head name for multi-head models (e.g. "omat_pbe")
            accelerator: One of ``ACCELERATORS``. "auto" uses CuEquivariance
                where it can help and quietly runs the plain e3nn path
                elsewhere; "cueq" asks for it but still degrades; "none" pins
                the plain path for reproducible timings.
        """
        if accelerator not in ACCELERATORS:
            raise ValueError(
                f"Unknown accelerator {accelerator!r}; expected one of "
                f"{', '.join(ACCELERATORS)}."
            )
        super().__init__(model_name, model_version)
        self.device = device
        self.head = head
        self.accelerator = accelerator
        # What actually ended up on the calculator; set by _build_calculator.
        self.accelerator_effective = "none"
        self._accelerator_warned = False
        # Set once the accelerated attempt has failed on this device, so a
        # per-structure rebuild does not retry a conversion that cannot work.
        self._accelerator_failed = False
        self.model_class = None
        self.calculator_class = None
        self.model = None
        self.trainer = None
        self.model_path = None

        # Note: MACE_AVAILABLE is False when package is not installed
        # This is expected in testing environments

    def _get_available_models(self) -> List[str]:
        """Get list of available MACE model names (all checkpoints)."""
        return sorted(list(AVAILABLE_MACE_MODELS.keys()))

    def load(self, model_path: Optional[str] = None) -> None:
        """
        Load a MACE model.

        Args:
            model_path: Path to model checkpoint. If None, loads default pretrained model.
                       If provided, loads checkpoint from the specified path.
        """
        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path is None:
            # Load pretrained model
            # Store the model name for calculator creation
            if Path(self.model_name).exists():
                self.model_path = self.model_name
                self.load_checkpoint(self.model_name)
                logger.info(f"Loaded model from path: {self.model_name}")
            else:
                self.model = "MACE"
                self.is_loaded = True
                logger.info(f"Loaded pretrained {self.model_name} model")
        else:
            # Load from checkpoint file
            self.model_path = model_path
            self.load_checkpoint(model_path)
            logger.info(f"Loaded model from checkpoint: {model_path}")

    def _accelerator_kwargs(self) -> Dict[str, Any]:
        """
        Calculator kwargs for the accelerated path, or empty when it is out.

        Decides once per call and explains the decision the first time it is
        a "no", so a user who expected the speedup can tell whether the host
        lacks a GPU, the package, or a compatible card.
        """
        if self.accelerator == "none" or self._accelerator_failed:
            return {}
        if not _is_cuda(self.device):
            if not self._accelerator_warned:
                logger.info(
                    "CuEquivariance needs a CUDA device; running %s on %s "
                    "without it.",
                    self.model_name,
                    self.device,
                )
                self._accelerator_warned = True
            return {}
        if not cueq_installed():
            if not self._accelerator_warned:
                logger.info(
                    "CuEquivariance is not installed; running %s without it. "
                    "It ships with the mlip cu126/cu130 extras.",
                    self.model_name,
                )
                self._accelerator_warned = True
            return {}
        return {"enable_cueq": True}

    def describe_accelerator(self) -> str:
        """
        One line on what the calculator will attempt, and why.

        Cheap and side-effect free: it answers from the device, the installed
        packages and the requested mode, without building a calculator or
        touching the model. Whether cueq then holds up on the card is only
        known when the calculator is built, and _build_calculator logs it.
        """
        if self.accelerator == "none":
            return "none (disabled by request)"
        if not _is_cuda(self.device):
            return f"none (device is {self.device}, and cueq needs CUDA)"
        if not cueq_installed():
            return "none (cuequivariance is not installed)"
        return "cueq (falls back to the plain path if this card has no kernels)"

    def _build_calculator(self, factory, /, **kwargs):
        """
        Build a calculator, preferring the accelerated path.

        MACE raises rather than degrading when cueqvariance is missing or has
        no kernels for the card in use, and doing that here would take MACE
        away from a user whose GPU merely is not one cuequivariance supports.
        So the accelerated attempt is the only thing in the try: if it fails,
        rebuild on the plain path and keep the reason. The plain path is not
        guarded, so a real error (a bad checkpoint, an unreadable model) still
        surfaces instead of being absorbed.
        """
        extra = self._accelerator_kwargs()
        if extra:
            try:
                calculator = factory(**kwargs, **extra)
            except Exception as exc:
                logger.warning(
                    "CuEquivariance could not run %s on %s (%s: %s); "
                    "continuing on the plain e3nn path, which is slower.",
                    self.model_name,
                    self.device,
                    type(exc).__name__,
                    exc,
                )
                self.accelerator_effective = "none"
                self._accelerator_failed = True
            else:
                self.accelerator_effective = "cueq"
                logger.info(
                    "CuEquivariance acceleration enabled for %s.", self.model_name
                )
                return calculator
        else:
            self.accelerator_effective = "none"
        return factory(**kwargs)

    def create_calculator(self) -> Calculator:
        """
        Create an ASE calculator from the model.

        Returns:
            ASE Calculator object.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Custom model path (e.g. fine-tuned)
        if self.model_path and Path(self.model_path).exists():
            from mace.calculators import MACECalculator

            return self._build_calculator(
                MACECalculator,
                model_paths=str(self.model_path),
                device=self.device,
                head=self.head,
            )

        # 2. Check if model_name is a path (fallback)
        if Path(self.model_name).exists():
            from mace.calculators import MACECalculator

            return self._build_calculator(
                MACECalculator,
                model_paths=str(self.model_name),
                device=self.device,
                head=self.head,
            )

        # 3. Standard Pretrained mapping
        model_id = AVAILABLE_MACE_MODELS.get(self.model_name)

        if model_id is None:
            available_models = self._get_available_models()
            raise ValueError(
                f"Unknown model name: '{self.model_name}'. "
                f"Available models: {', '.join(available_models)}. "
                f"Please use one of the available model names."
            )

        # Determine which calculator function to use based on model type
        model_name_upper = self.model_name.upper()

        # Special case for MACE-MH models (direct URL or explicit name)
        if (
            "MH-" in model_name_upper
            or model_name_upper in ["MACE", "MACE-MP"]
            or model_id.startswith("http")
        ):
            # Use direct MACECalculator for URLs or MACE-MH models
            from mace.calculators import MACECalculator

            # Download if it's a URL
            if model_id.startswith("http"):
                from mace.calculators.foundations_models import (
                    download_mace_mp_checkpoint,
                )

                model_path = download_mace_mp_checkpoint(model_id)
            else:
                model_path = model_id

            # Default head for MH-1 and MH-0 is omat_pbe if not specified
            head = self.head
            if head is None and (
                "MH-1" in model_name_upper
                or "MH-0" in model_name_upper
                or model_name_upper in ["MACE", "MACE-MP"]
            ):
                head = "omat_pbe"

            return self._build_calculator(
                MACECalculator, model_paths=model_path, device=self.device, head=head
            )

        if "OFF" in model_name_upper:
            # Use MACE-OFF calculator
            from mace.calculators import mace_off

            off_model_mapping = {
                "MACE-OFF23-SMALL": "small",
                "MACE-OFF23-MEDIUM": "medium",
                "MACE-OFF23-LARGE": "large",
            }
            off_model_size = off_model_mapping.get(model_name_upper, "medium")
            return self._build_calculator(
                mace_off, model=off_model_size, device=self.device
            )

        elif "ANI-CC" in model_name_upper:
            # Use MACE-ANI-CC
            from mace.calculators import mace_anicc

            # mace_anicc takes a fixed argument list, so it cannot be handed
            # enable_cueq the way the other helpers can.
            return mace_anicc(device=self.device)

        elif "OMOL" in model_name_upper:
            # Use MACE-OMOL
            from mace.calculators import mace_omol

            return self._build_calculator(mace_omol, model=model_id, device=self.device)

        else:
            # Use MACE-MP calculator for materials models (including MPA, MATPES, OMAT)
            from mace.calculators import mace_mp

            return self._build_calculator(mace_mp, model=model_id, device=self.device)

    def predict_atomic_features(self, structure_data: Any) -> Dict[str, Any]:
        """
        Predict atomic latent features (descriptors) for a structure using MACE.
        """
        if not self.is_loaded:
            return {"error": "Model not loaded. Please call load() first."}

        atoms = self.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms

        try:
            calc = self.create_calculator()
            # MACE get_descriptors returns (num_atoms, num_features)
            # invariants_only=True returns the invariant part of the descriptors
            descriptors = calc.get_descriptors(atoms, invariants_only=True)

            return {
                "atomic_features": descriptors.tolist(),
                "feature_dim": descriptors.shape[1],
                "num_atoms": descriptors.shape[0],
            }
        except Exception as e:
            import traceback
            import sys

            print(f"Failed to predict atomic features: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {"error": f"Failed to predict atomic features: {str(e)}"}

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save a model checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint.
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # For MACE, we save the calculator state
            checkpoint = {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "is_fine_tuned": self.is_fine_tuned,
                "device": self.device,
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        try:
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Check if checkpoint is a dict (standard checkpoint)
            if isinstance(checkpoint, dict):
                # Load metadata
                self.model_name = checkpoint.get("model_name", self.model_name)
                self.model_version = checkpoint.get("model_version", self.model_version)
                self.is_fine_tuned = checkpoint.get("is_fine_tuned", False)
                self.device = checkpoint.get("device", self.device)
            else:
                # Assume it's a compiled model or other format (e.g. TorchScript)
                # We can't extract metadata easily, but we trust the user provided a valid model
                logger.info(
                    f"Loaded checkpoint is not a dictionary (type: {type(checkpoint)}). Assuming compiled model or direct model object."
                )
                # Keep existing metadata

            self.is_loaded = True

            logger.info(f"Model checkpoint loaded from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    @property
    def supports_charge_spin(self) -> bool:
        """
        Return True for models trained with joint_embedding on charge/spin.

        Supported: MACE-OMOL, MACE-MH (omol head).
        The calculator reads charge and spin from atoms.info via the default
        info_keys mapping:  {"total_charge": "charge", "total_spin": "spin"}.
        i.e. set atoms.info["charge"] and atoms.info["spin"].
        """
        name_upper = self.model_name.upper()
        return "OMOL" in name_upper or "MACE-MH" in name_upper

    def get_model_capabilities(self) -> Dict[str, bool]:
        """
        Get model capabilities.

        Returns:
            Dictionary of capabilities.
        """
        return {
            "energy": True,
            "forces": True,
            "stress": True,
            "optimization": True,
            "relaxation": True,
            "charge_spin": self.supports_charge_spin,
        }

    def get_supported_elements(self) -> List[str]:
        """
        Get list of chemical elements supported by the loaded model.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # MACE-MP models typically support 89 elements
        try:
            temp_calc = self.create_calculator()
            if hasattr(temp_calc, "z_table") and temp_calc.z_table is not None:
                from ase.data import chemical_symbols

                return [chemical_symbols[int(z)] for z in temp_calc.z_table.zs]
        except Exception as e:
            # If we couldn't properly load the calculator to extract the z_table, we must not guess.
            raise RuntimeError(
                f"Could not determine supported elements for {self.model_name}. "
                f"Failed to extract z_table from the MACECalculator. Internal error: {str(e)}"
            )

    def _get_nvalchemi_model(self):
        """Return a NValchemi-compatible MACEWrapper, or None if unavailable."""
        from src.utils.mlips.nvalchemi.mace_nv import get_nvalchemi_mace_model

        return get_nvalchemi_mace_model(self)
