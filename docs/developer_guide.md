# Developer Guide

## Architecture Overview

**AtomisticSkills** follows a **modular architecture** designed to isolate dependencies across three `uv` projects (`cpu`, `mlip`, `fairchem`) and expose functionality through the Model Context Protocol (MCP) using the unified `venv/run` launcher:

```
┌─────────────────────────────────────────────────────────┐
│                    AI Coding Agent                      │
│        (Antigravity, Claude Code, Cursor, Codex)        │
└────────────────────┬────────────────────────────────────┘
                     │ MCP Protocol / CLI Fallback
          ┌──────────┴──────────┬───────────┬─────────────┐
          │                     │           │             │
     ┌────▼────┐         ┌──────▼──┐   ┌───▼───┐   ┌─────▼─────┐
     │  MACE   │         │ MatGL   │   │ Fair  │   │   Base    │
     │ Server  │         │ Server  │   │ Chem  │   │  Server   │
     └────┬────┘         └────┬────┘   └───┬───┘   └─────┬─────┘
          │                   │            │             │
┌─────────▼───────────────────▼────────────▼─────────────▼────────┐
│                      venv/run Launcher                          │
│     (Native uv execution with container image fallback)         │
└─────────┬───────────────────┬──────────────────────────┬────────┘
          │                   │                          │
     ┌────▼────────┐    ┌─────▼──────┐             ┌─────▼──────┐
     │  venv/mlip  │    │venv/fairchem│            │  venv/cpu  │
     │ (MACE,MatGL)│    │ (FairChem) │             │ (No Torch) │
     └─────────────┘    └────────────┘             └────────────┘
```

---

## Core Components

### 1. MCP Servers (`src/mcp_server/`)

Each MCP server is a standalone Python module that exposes tools via the FastMCP framework:

| Server | Runtime / venv | Primary Functionality |
|--------|----------------|----------------------|
| [base_server.py](../src/mcp_server/base_server.py) | `cpu` | Materials Project queries, structure utilities, literature |
| [atomate2_server.py](../src/mcp_server/atomate2_server.py) | `cpu` | Remote DFT workflows, calculation status monitoring |
| [drugdisc_server.py](../src/mcp_server/drugdisc_server.py) | `cpu` | Molecular descriptors, standardization, PDBQT conversion |
| [smol_server.py](../src/mcp_server/smol_server.py) | `cpu` | Cluster expansion training and Monte Carlo simulations |
| [mace_server.py](../src/mcp_server/mace_server.py) | `mlip` | MACE foundation models (relax, MD, features) |
| [matgl_server.py](../src/mcp_server/matgl_server.py) | `mlip` | MatGL models (CHGNet, TensorNet, M3GNet), bandgap prediction |
| [fairchem_server.py](../src/mcp_server/fairchem_server.py) | `fairchem` | FairChem models (UMA, eSEN) |
| [mattergen_server.py](../src/mcp_server/mattergen_server.py) | `generative` image / conda | MatterGen generative crystal design |
| [adit_server.py](../src/mcp_server/adit_server.py) | `generative` image / conda | ADiT all-atom diffusion transformer |
| [diffcsp_server.py](../src/mcp_server/diffcsp_server.py) | `generative` image / conda | DiffCSP++ crystal structure generation |

### 2. Utility Modules (`src/utils/`)

Supporting libraries shared across MCP servers:

| Module | Purpose |
|--------|---------|
| `mlips/` | Unified MLIP wrappers (MACE, MatGL, FairChem) with common predict / relax / MD / fine-tune interface |
| `generative_models/` | Generative model wrappers (ADiT, DiffCSP++, MatterGen) for structure generation |
| `dft/` | VASP input generation and output parsing via Pymatgen |
| `drugdisc_utils.py` | RDKit-based molecular descriptors, standardization, and PDBQT conversion |
| `structure_utils.py` | Convert between ASE Atoms, Pymatgen Structure, and dict formats |
| `structure_viz.py` | Crystal structure visualization and rendering |
| `disordered_material/` | Order-disorder sampling for partial-occupancy structures |
| `mlips/md_utils.py` | MD monitors (explosion, volume, melting, equilibration detection) |
| `config_utils.py` | Global configuration and API key management |

### 3. Skills (`skills/`)

Skills are **modular, self-contained capabilities** that combine multiple tools and scripts to accomplish complex research tasks. Each skill lives in its own directory with a standardized structure:

```
skills/<skill-name>/
├── SKILL.md              # Instructions and documentation
├── scripts/              # Python helper scripts
├── examples/             # Reference input/output files with README.md
└── resources/            # Configuration files, templates, reference data
```

**How Skills Work**:
1. The agent reads `SKILL.md` to understand the task
2. Follows step-by-step instructions
3. Executes scripts via `venv/run <venv>` in the declared environment (`metadata.venv`)
4. Uses resources and examples as templates

> [!TIP]
> To create a new skill, follow the guidelines in [skill-standards.md](../.agents/rules/skill-standards.md).

---

## Development Workflow

### Adding a New MCP Tool

1. **Choose the appropriate server** based on dependencies
2. **Define the tool function** with type hints and docstrings:
   ```python
   @mcp.tool()
   def my_new_tool(
       structure_data: dict,
       parameter1: float = 1.0,
       parameter2: str = "default",
   ) -> dict:
       """
       Brief description of what this tool does.

       Args:
           structure_data: Structure in dictionary format.
           parameter1: Description of parameter.
           parameter2: Another parameter.

       Returns:
           Dictionary with results.
       """
       # Implementation logic
       return results
   ```

3. **Test the tool**:
   ```bash
   # Test via the shell CLI fallback:
   venv/run cpu python -m src.mcp_server.cli base my_new_tool parameter1=2.0

   # Or run the server over stdio for agent connection:
   venv/run --server base
   ```

### Implementing a New Skill

1. Create the skill directory: `skills/<skill-name>/`
2. Write `SKILL.md` following [skill-standards.md](../.agents/rules/skill-standards.md), declaring `metadata.venv: [cpu]` (or `mlip`, `fairchem`)
3. Add helper scripts to `scripts/`
4. Provide examples and resources as needed
5. Test the skill commands through `venv/run`

### Running Tests

The project uses `pytest` executed through the launcher:

```bash
# Run core CPU test suites
venv/run cpu python -m pytest tests/test_launcher.py tests/test_skill_runtime.py tests/test_uv_projects.py tests/test_images_and_manifests.py tests/test_tool_cli.py -q

# Test server-specific suites in their respective environments
venv/run cpu python -m pytest tests/base/ tests/atomate2/ tests/drugdisc/ tests/smol/
venv/run mlip python -m pytest tests/mace/ tests/matgl/
venv/run fairchem python -m pytest tests/fairchem/
```

---

## Important Technical Details

### Environment Isolation
- Three `uv` projects (`cpu`, `mlip`, `fairchem`) isolate incompatible dependencies (e.g., `e3nn` version pins in MACE vs FairChem).
- The repository root is installed as an editable package into each virtualenv, ensuring `import src` works uniformly.
- Commands execute via `venv/run <venv>[+<extra>]`, which resolves the virtual environment automatically without manual activation.

### Stdout/Stderr Handling
All MCP servers use centralized output redirection (see `src/utils/mcp_utils.py`) to prevent:
- Print statements from polluting MCP responses
- Training logs from breaking the JSON protocol
- Debugging output from interfering with tool calls

For a complete guide on how AtomisticSkills mitigates this execution noise issue across heterogeneous infrastructures, see the [MCP STDIO Redirection Guide](mcp_stdio_redirection.md).

> [!WARNING]
> Never use raw `print()` in MCP tool functions expecting the user to read them cleanly. Use proper `logging`. Legacy outputs are automatically routed to standard error by the server.

### Research Directory Management
Every research task should:
1. Call `create_research_dir(research_topic)` to establish a timestamped directory under `ATOMISTIC_WORKSPACE`
2. Save all results (structures, plots, logs) to this directory
3. Document findings in the research directory

---

## Troubleshooting

### MCP Server Not Loading
- Run `venv/run --doctor` to verify runtime health and prerequisites.
- For first-time server start, the environment may sync in the background; reconnect once ready.
- If using containers, ensure Docker/Podman/Apptainer is running.

### Import Errors in Scripts
- Execute scripts via `venv/run <venv>` rather than a bare `python` command.
- Verify that `metadata.venv` specifies the appropriate project.
- Use absolute imports: `from src.utils.mlips.loader import load_wrapper`.

### Fine-Tuning Fails
- Verify stress units are in eV/Å³ (see [stress-units.md](../.agents/rules/stress-units.md) for details)
- Check training data format matches expected structure
- For small datasets (<500 structures), use `freeze_backbone=True`

### MD Simulation Explodes
- Enable MD monitors: `monitor=True, monitor_type=["explosion", "volume"]`
- Reduce timestep (default: 1fs, try 0.5fs)
- Check initial structure is relaxed with `fmax < 0.05 eV/Å`
