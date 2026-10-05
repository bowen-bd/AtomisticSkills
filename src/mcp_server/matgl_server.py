import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Silence Wandb and suppress warnings to prevent protocol pollution
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

# Add project root to sys.path before importing project-local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.mcp_utils import setup_mcp_stdout, run_fastmcp_server

# Setup stdout redirection for MCP
mcp_pipe_binary = setup_mcp_stdout()

from mcp.server.fastmcp import FastMCP
from src.utils.serialization_utils import recursive_tolist
from src.utils.research_utils import get_current_research_dir

# Suppress all warnings to prevent protocol pollution
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("MatGL-Server")

# Initialize FastMCP server
mcp = FastMCP("MatGL")

# Global variables to hold state
wrapper: Optional[Any] = None


@mcp.tool()
def load_model(
    model_name: str = "CHGNet-PES-MatPES-PBE-1M-2026.9", device: str = "auto"
) -> str:
    """
    Load a MatGL model.

    Supported models include:
    - CHGNet PES: 'CHGNet-PES-MatPES-PBE-1M-2026.9' (default), 'CHGNet-PES-MatPES-r2SCAN-1M-2026.9';
      earlier checkpoints 'CHGNet-PES-MatPES-PBE-2025.2.10', 'CHGNet-PES-MatPES-r2SCAN-2025.2.10'
    - M3GNet PES: 'M3GNet-PES-MatPES-PBE-2025.2', 'M3GNet-PES-MatPES-r2SCAN-2025.2', 'M3GNet-PES-ANI-1x-Subset'
    - TensorNet PES: 'TensorNet-PES-MatPES-PBE-2025.2', 'TensorNet-PES-MatPES-r2SCAN-2025.2', 'TensorNet-PES-ANI-1x-Subset'
    - QET PES: 'QET-PES-MatPES-PBE-2025.2', 'QET-PES-MatPES-r2SCAN-2025.2', 'QET-PES-MatQ'
    - SO3Net PES: 'SO3Net-PES-ANI-1x-Subset'
    - Formation energy: 'MEGNet-Eform-MP-2018.6.1', 'M3GNet-Eform-MP-2018.6.1'

    Args:
        model_name: Name of the model to load.
        device: Device to use ("auto", "cpu", "cuda").

    Returns:
        Confirmation message.

    CRITICAL: This tool must be called before using any other tool (except predict_bandgap) to load the model into memory.
    """
    global wrapper
    try:
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(model_name=model_name, device=device)
        wrapper.load()
        return f"Successfully loaded MatGL model: {model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


@mcp.tool()
def predict_structure(
    structure_data: Union[Dict[str, Any], str, List[Union[Dict[str, Any], str]]],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Predict energy and forces for a structure or a batch of structures.

    Args:
        structure_data: Single structure or batch (directory path, list of dicts/paths).

    Returns:
        Dictionary containing:
        - "energy": Total potential energy (eV)
        - "forces": Atomic forces (eV/A)
        - "stress": (Optional) Stress tensor in eV/Å³
        - "charges": (Optional) Atomic charges if supported (e.g., CHGNet)
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}

    return recursive_tolist(wrapper.static_calculation(structure_data))


@mcp.tool()
def predict_atomic_features(
    structure_data: Union[Dict[str, Any], str], output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict atomic latent features (descriptors) for a structure.
    Automatically saves features to the current research directory.

    Args:
        structure_data: Single structure or batch (directory path, list of dicts/paths).
        output_path: Optional custom path for saving features. If not provided, auto-generates
                     based on structure filename (e.g., 'solid.cif' -> 'solid_features.json').

    Returns:
        Dict: {
            'atomic_features': [[...], ...],
            'feature_dim': int,
            'num_atoms': int,
            'saved_path': str  # Path to saved JSON file
        }
    """
    global wrapper
    import json
    from src.utils.research_utils import get_current_research_dir

    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}

    # Get features
    result = wrapper.predict_atomic_features(structure_data)

    if "error" in result:
        return result

    # Save to research directory
    try:
        # Use custom path if provided, otherwise auto-generate
        if output_path:
            save_path = Path(output_path)
        else:
            research_dir = get_current_research_dir()

            # Generate filename based on structure path or use generic name
            if isinstance(structure_data, str):
                struct_path = Path(structure_data)
                feature_filename = f"{struct_path.stem}_features.json"
            else:
                feature_filename = "atomic_features.json"

            save_path = research_dir / feature_filename

        # Save features
        with open(save_path, "w") as f:
            json.dump(recursive_tolist(result), f)

        # Add saved path to result
        result["saved_path"] = str(save_path)

        return result

    except Exception as e:
        # If saving fails, still return the features but with error info
        result["save_error"] = f"Failed to save features: {str(e)}"
        return result


# Local variable to cache bandgap predictor separate from global PES wrapper
_bandgap_wrapper: Optional[Any] = None


@mcp.tool()
def predict_bandgap(
    structure_data: Union[Dict[str, Any], str],
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict the bandgap for a structure using MEGNet-BandGap-mfi.
    Uses an isolated model instance to avoid conflicts with PES calculations.

    Args:
        structure_data: Single structure or batch (directory path, list of dicts/paths).
        task_name: DFT functional for the bandgap prediction.
            Supported: "PBE" (default), "GLLB-SC", "HSE", "SCAN".

    Returns:
        Dictionary containing "bandgap" in eV.
    """
    global _bandgap_wrapper
    try:
        if _bandgap_wrapper is None or _bandgap_wrapper.task_name != task_name:
            from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

            _bandgap_wrapper = MatGLWrapper(
                model_name="MEGNet-MP-2019.4.1-BandGap-mfi",
                device="auto",
                task_name=task_name,
            )
            _bandgap_wrapper.load()

        return _bandgap_wrapper.static_calculation(structure_data)
    except Exception as e:
        return {"error": f"Bandgap prediction failed: {str(e)}"}


@mcp.tool()
def get_info() -> Dict[str, Any]:
    """
    Get information about the current loaded model.
    """
    global wrapper
    if wrapper is None:
        return {"status": "no_model_loaded"}
    return wrapper.get_model_info()


@mcp.tool()
def relax_structure(
    structure_data: Union[Dict[str, Any], str, List[Union[Dict[str, Any], str]]],
    fmax: float = 0.02,
    steps: int = 500,
    optimizer: str = "FIRE",
    relax_cell: bool = True,
    output_dir: Optional[str] = None,
    extract_batch_results: bool = True,
    max_batch_atoms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Relax one or multiple structures using the loaded MatGL model.

    Args:
        structure_data: Single structure or batch (directory path, list of dicts/paths).
        fmax: Force convergence criterion (eV/Ang).
        steps: Maximum number of optimization steps.
        optimizer: Optimizer to use ("FIRE", "BFGS", "LBFGS").
        relax_cell: Whether to relax the unit cell.
        output_dir: Directory to save results. For batch mode, each structure gets a subdirectory.
        extract_batch_results: Whether to extract full trajectory / logs for all structures in batch mode.
        max_batch_atoms: Override the atom budget for the NValchemi inflight live batch.
            When None (default) the budget is estimated from free VRAM and model size.
            Set a smaller value (e.g. 300–500) on shared GPUs to avoid OOM.

    Returns:
        For single: Dict with energy, trajectory_path, cif_path, json_path
        For batch: Dict with mode="batch", total_structures, successful, failed, results list
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}

    try:
        # Simply delegate to base wrapper's unified relax_structure method
        return recursive_tolist(
            wrapper.relax_structure(
                structure_data=structure_data,
                fmax=fmax,
                steps=steps,
                optimizer=optimizer,
                relax_cell=relax_cell,
                output_dir=output_dir,
                extract_batch_results=extract_batch_results,
                max_batch_atoms=max_batch_atoms,
            )
        )
    finally:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@mcp.tool()
def run_md(
    structure_data: Union[Dict[str, Any], str, List[Union[Dict[str, Any], str]]],
    temperature: float = 300,
    steps: int = 1000,
    timestep: float = 1.0,
    ensemble: str = "nvt",
    log_interval: int = 10,
    pressure: float = 0.0,
    pressure_mask: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    monitor: bool = False,
    monitor_type: Optional[Union[str, List[str]]] = None,
    monitor_params: Optional[Dict[str, Any]] = None,
    supercell_min_length: Optional[float] = None,
    extract_batch_results: bool = True,
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulation using MatCalc.

    Args:
        structure_data: Single structure or batch (directory path, list of dicts/paths).
        temperature: Temperature in Kelvin.
        steps: Number of steps.
        timestep: Timestep in fs.
        ensemble: Ensemble "nve", "nvt" (Nose-Hoover), or "npt" (NPT).
                  Also supports variants like "nvt_langevin", "nvt_andersen", "npt_berendsen", "npt_mtk".
        log_interval: Interval for logging to trajectory and logfile.
        pressure: Target pressure in bar (for NPT).
        pressure_mask: Mask for anisotropic NPT (e.g., [1, 0, 0] for 1D).
        output_dir: Directory to save results.
        monitor: Whether to enable automatic MD monitoring.
        monitor_type: Type of monitor ("melting", "explosion", "overshoot", "volume") or list of types.
        monitor_params: Optional dictionary of parameters for the monitors (e.g., {"upper_limit_ratio": 4.0}).
        supercell_min_length: Minimum length (Å) for each lattice vector. Automatically expands supercell.
        extract_batch_results: Whether to extract full trajectory / logs for all structures in batch mode.
    Returns:
        Dictionary with MD results (trajectory_path, final_structure).
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}

    # Setup Directory
    if not output_dir:
        output_dir = str(get_current_research_dir() / "matgl" / "md")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare configuration for worker
    try:
        # Run MD via wrapper's unified run_md (supports monitoring callbacks)
        result = wrapper.run_md(
            structure_data=structure_data,
            temperature=temperature,
            steps=steps,
            timestep=timestep,
            ensemble=ensemble,
            log_interval=log_interval,
            pressure=pressure,
            pressure_mask=pressure_mask,
            output_dir=output_dir,
            monitor=monitor,
            monitor_type=monitor_type,
            monitor_params=monitor_params,
            supercell_min_length=supercell_min_length,
            extract_batch_results=extract_batch_results,
        )

        return recursive_tolist(result)

    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {
            "error": f"MD execution failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
    finally:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_fastmcp_server(mcp, mcp_pipe_binary)
