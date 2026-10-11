#!/usr/bin/env python3
"""Train a property predictor on a pretrained MatGL CHGNet backbone.

matgl >= 4 is built on PyTorch Geometric only: graphs are PyG ``Data`` objects,
batched with ``Batch.from_data_list``. Positions and periodic image shifts are
attached to each graph before batching, because each graph needs its own
lattice to turn ``pbc_offset`` into a Cartesian shift.

Usage:
    venv/run mlip python train_matgl_property.py --data_path data.json \
        --model_name CHGNet-PES-MatPES-PBE-1M-2026.9 --target_property bulk_modulus

Requirements:
    - Environment: mlip (run with: venv/run mlip python ...)
"""

import argparse
import tempfile
import logging
from pathlib import Path
import json

import torch
from torch_geometric.data import Batch

import matgl
from matgl.ext.pymatgen import Structure2Graph

# Ase/Pymatgen
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a property predictor on a pretrained MatGL CHGNet backbone."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSON or XYZ file containing structures and properties.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CHGNet-PES-MatPES-PBE-1M-2026.9",
        help="Pretrained MatGL CHGNet model to start from (e.g. CHGNet-PES-MatPES-r2SCAN-1M-2026.9).",
    )
    parser.add_argument(
        "--target_property",
        type=str,
        default="target_property",
        help="Key name for the target property in the data file.",
    )
    parser.add_argument(
        "--property_type",
        type=str,
        choices=["intensive", "extensive"],
        default="intensive",
        help="Type of property.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the pre-trained graph backbone.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use.",
    )
    return parser.parse_args()


def prepare_data(data_path, target_property):
    structures = []
    labels = []

    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)

        for item in data:
            struct = Structure.from_dict(item["structure"])
            structures.append(struct)

            if target_property in item:
                labels.append(float(item[target_property]))
            elif "property" in item:
                labels.append(float(item["property"]))
            else:
                raise ValueError(
                    f"Could not find property {target_property} in JSON item."
                )

    elif data_path.endswith(".xyz") or data_path.endswith(".extxyz"):
        atoms_list = read(data_path, index=":")
        converter = AseAtomsAdaptor()
        for atoms in atoms_list:
            if target_property not in atoms.info:
                raise ValueError(
                    f"Property key '{target_property}' not found in XYZ file info."
                )

            labels.append(float(atoms.info[target_property]))
            structures.append(converter.get_structure(atoms))
    else:
        raise ValueError("Unsupported data format. Please provide .json or .xyz files.")

    return structures, labels


def structure_to_graph(
    structure: Structure, converter: Structure2Graph
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a structure into a PyG graph ready for batching.

    Args:
        structure: Pymatgen structure.
        converter: MatGL graph converter built with the backbone's element list.

    Returns:
        Tuple (graph with ``pos`` and ``pbc_offshift`` attached, state attributes).
    """
    g, lattice, state_attr = converter.get_graph(structure)
    lattice = torch.as_tensor(lattice, dtype=matgl.float_th).reshape(3, 3)
    g.pos = torch.tensor(structure.cart_coords, dtype=matgl.float_th)
    g.pbc_offshift = g.pbc_offset.to(matgl.float_th) @ lattice
    return g, torch.as_tensor(state_attr, dtype=matgl.float_th)


def collate_graphs(batch: list) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """
    Collate (graph, label, state_attr) items into a batched graph.

    Args:
        batch: List of items returned by the dataset.

    Returns:
        Tuple (PyG Batch, 1-D labels of shape [B], stacked state attributes).
    """
    graphs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    state_attrs = torch.stack([item[2] for item in batch])
    return Batch.from_data_list(graphs), labels, state_attrs


def predict_batch(
    model: torch.nn.Module,
    graph: Batch,
    state_attrs: torch.Tensor,
    per_atom: bool,
) -> torch.Tensor:
    """
    Predict one value per structure in a batched graph.

    Args:
        model: MatGL model (e.g. CHGNet) returning one value per graph.
        graph: Batched PyG graph.
        state_attrs: Stacked state attributes of the batch.
        per_atom: Divide each structure's prediction by its own atom count, which
            turns an extensive model output into an intensive target.

    Returns:
        1-D tensor of shape [num_graphs].
    """
    preds = model(graph, state_attr=state_attrs).reshape(-1)
    if per_atom:
        counts = torch.bincount(graph.batch, minlength=graph.num_graphs)
        preds = preds / counts.to(preds.dtype)
    return preds


def evaluate(
    model: torch.nn.Module, loader, device: str, per_atom: bool
) -> tuple[float, float]:
    """
    Mean squared and mean absolute error over every structure in a loader.

    Errors are summed per structure and divided by the number of structures, so
    batches of different sizes are weighted correctly.

    Args:
        model: MatGL model.
        loader: DataLoader yielding ``collate_graphs`` batches.
        device: Torch device string.
        per_atom: Passed to ``predict_batch``.

    Returns:
        Tuple (MSE, MAE).
    """
    sq_sum, abs_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for graph, labels, state_attrs in loader:
            preds = predict_batch(
                model, graph.to(device), state_attrs.to(device), per_atom
            )
            err = preds - labels.to(device).reshape(-1)
            sq_sum += float((err**2).sum())
            abs_sum += float(err.abs().sum())
            n += err.numel()
    return sq_sum / n, abs_sum / n


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading data from {args.data_path}")
    structures, labels = prepare_data(args.data_path, args.target_property)
    logger.info(f"Loaded {len(structures)} structures.")

    # Load the backbone first: its element list defines the node-type indices,
    # so graphs must be built with it rather than with the dataset's elements.
    logger.info(f"Loading pretrained model: {args.model_name}")
    # Accept the legacy names skills and papers use (e.g. *-v2025.1-PES);
    # matgl 4 renamed every pretrained model. A load failure is an error.
    from src.utils.mlips.matgl.matgl_wrapper import AVAILABLE_MATGL_MODELS

    potential = matgl.load_model(
        AVAILABLE_MATGL_MODELS.get(args.model_name, args.model_name)
    )
    model = getattr(potential, "model", potential)
    elem_list = list(model.element_types)
    converter = Structure2Graph(
        element_types=elem_list, cutoff=getattr(model, "cutoff", 5.0)
    )

    output_dir = args.output_dir or tempfile.mkdtemp(prefix="matgl_property_")
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    class InMemoryPropertyDataset(torch.utils.data.Dataset):
        def __init__(self, structures, labels, converter):
            self.structures = structures
            self.labels = labels
            self.converter = converter

        def __len__(self):
            return len(self.structures)

        def __getitem__(self, idx):
            g, state_attr = structure_to_graph(self.structures[idx], self.converter)
            return g, self.labels[idx], state_attr

    dataset = InMemoryPropertyDataset(structures, labels, converter)

    # Split dataset manually
    num_samples = len(dataset)
    num_val = int(0.1 * num_samples)
    num_test = int(0.1 * num_samples)
    num_train = num_samples - num_val - num_test

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(42),
    )

    from torch.utils.data import DataLoader

    batch_size = min(args.batch_size, len(train_data))
    train_loader = DataLoader(
        train_data,
        collate_fn=collate_graphs,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=collate_graphs,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )
    # test_data (10%) is held out and not used during training.

    # We do NOT override model.is_intensive here because changing an extensive
    # pretrained model to intensive breaks its forward method (missing 'readout' attr).
    # Instead, we will divide the output by the number of atoms dynamically.
    is_intensive_target = args.property_type == "intensive"
    per_atom = is_intensive_target and not getattr(model, "is_intensive", True)

    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm

    # Move model to device natively
    model = model.to(device)

    if args.freeze_backbone:
        logger.info("Freezing GNN backbone (only training the final MLP readout).")
        for name, param in model.named_parameters():
            # Usually the readout layers in MatGL start with 'final_layer' or 'readout' or 'fc'
            if (
                "readout" not in name.lower()
                and "final" not in name.lower()
                and "fc" not in name.lower()
            ):
                param.requires_grad = False
            else:
                logger.info(f"  Training parameter: {name}")
    else:
        logger.info("Training full model (unfrozen).")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    criterion = nn.MSELoss()

    logger.info("Starting manual PyTorch Training...")
    best_val_mae = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_sq_sum, n_train = 0.0, 0
        for graph, labels, state_attrs in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"
        ):
            graph = graph.to(device)
            state_attrs = state_attrs.to(device)
            labels = labels.to(device).reshape(-1)

            optimizer.zero_grad()
            preds = predict_batch(model, graph, state_attrs, per_atom)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_sq_sum += float(loss) * labels.numel()
            n_train += labels.numel()

        model.eval()
        val_loss, val_mae = evaluate(model, val_loader, device, per_atom)

        logger.info(
            f"Epoch {epoch+1}: Train MSE={train_sq_sum / n_train:.4f}, Val MSE={val_loss:.4f}, Val MAE={val_mae:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            logger.info(
                f"New best model found (Val MAE: {best_val_mae:.4f}). Saving..."
            )
            model.save(str(output_path / "matgl_model"))

    logger.info("MatGL Property Predictor training finished.")
    logger.info(f"Best model saved to {output_path / 'matgl_model'}")


if __name__ == "__main__":
    main()
