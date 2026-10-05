#!/usr/bin/env python3
"""Train a property predictor on a MatGL backbone (M3GNet or MEGNet).

matgl >= 4 is built on PyTorch Geometric only: graphs are PyG ``Data`` objects,
batched with ``Batch.from_data_list``. Positions and periodic image shifts are
attached to each graph before batching, because each graph needs its own
lattice to turn ``pbc_offset`` into a Cartesian shift.

Usage:
    venv/run mlip python train_matgl_property.py --data_path data.json \
        --model_name M3GNet-PES-MatPES-PBE-2025.2 --target_property bulk_modulus

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
from matgl.models import M3GNet
from matgl.ext.pymatgen import Structure2Graph, get_element_list

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
        description="Train a MEGNet property predictor using MatGL."
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
        default="M3GNet-PES-MatPES-PBE-2025.2",
        help="Pretrained MatGL model to start from (see matgl.get_available_pretrained_models()).",
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
    try:
        # Accept the legacy names skills and papers use (e.g. *-v2025.1-PES);
        # matgl 4 renamed every pretrained model.
        from src.utils.mlips.matgl.matgl_wrapper import AVAILABLE_MATGL_MODELS

        potential = matgl.load_model(
            AVAILABLE_MATGL_MODELS.get(args.model_name, args.model_name)
        )
        model = getattr(potential, "model", potential)
    except Exception as e:
        logger.warning(
            f"Failed to load pretrained model '{args.model_name}', falling back to fresh initialization: {e}"
        )
        model = M3GNet(
            element_types=get_element_list(structures),
            is_intensive=(args.property_type == "intensive"),
        )
    elem_list = list(model.element_types)
    converter = Structure2Graph(
        element_types=elem_list, cutoff=getattr(model, "cutoff", 5.0)
    )

    def structure_to_graph(structure):
        """Return (PyG graph with pos and pbc_offshift, state attributes)."""
        g, lattice, state_attr = converter.get_graph(structure)
        lattice = torch.as_tensor(lattice, dtype=matgl.float_th).reshape(3, 3)
        g.pos = torch.tensor(structure.cart_coords, dtype=matgl.float_th)
        g.pbc_offshift = g.pbc_offset.to(matgl.float_th) @ lattice
        return g, torch.as_tensor(state_attr, dtype=matgl.float_th)

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
            g, state_attr = structure_to_graph(self.structures[idx])
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

    def my_collate(batch):
        graphs = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        state_attrs = torch.stack([item[2] for item in batch])
        return Batch.from_data_list(graphs), labels, state_attrs

    batch_size = min(args.batch_size, len(train_data))
    train_loader = DataLoader(
        train_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        collate_fn=my_collate,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    # We do NOT override model.is_intensive here because changing an extensive
    # pretrained model to intensive breaks its forward method (missing 'readout' attr).
    # Instead, we will divide the output by the number of atoms dynamically.
    is_intensive_target = args.property_type == "intensive"

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
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            graph, labels, state_attrs = batch[0], batch[1], batch[2]
            graph = graph.to(device)
            state_attrs = state_attrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(graph, state_attr=state_attrs)
            if is_intensive_target and not getattr(model, "is_intensive", True):
                preds = preds / torch.bincount(
                    graph.batch, minlength=graph.num_graphs
                ).unsqueeze(-1)
            loss = criterion(preds.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                graph, labels, state_attrs = batch[0], batch[1], batch[2]
                graph = graph.to(device)
                state_attrs = state_attrs.to(device)
                labels = labels.to(device)

                preds = model(graph, state_attr=state_attrs)
                if is_intensive_target and not getattr(model, "is_intensive", True):
                    preds = preds / torch.bincount(
                        graph.batch, minlength=graph.num_graphs
                    ).unsqueeze(-1)
                val_loss += criterion(preds.squeeze(), labels.squeeze()).item()
                val_mae += torch.nn.functional.l1_loss(
                    preds.squeeze(), labels.squeeze()
                ).item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        logger.info(
            f"Epoch {epoch+1}: Train MSE={train_loss/len(train_loader):.4f}, Val MSE={val_loss:.4f}, Val MAE={val_mae:.4f}"
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
