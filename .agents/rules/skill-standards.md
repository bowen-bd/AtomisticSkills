---
trigger: model_decision
description: Rules to implement a skill under `skills/`
---

# Skill Standards

All modular capabilities in this project should be implemented as "Skills" within the `skills/` directory. This rule ensures consistency, discoverability, and reusability for the agent.

## Directory Structure

Each skill must reside in its own subdirectory with the following structure:
```
skills/<skill-name>/
├── SKILL.md                  # Required: Main documentation
├── scripts/                  # Optional: Helper scripts
│   ├── script1.py
│   └── script2.py
├── examples/                 # Optional: Reference input/output files
│   └── example-name/
│       ├── README.md         # Required for each example
│       ├── example_input.cif
│       └── example_output.json
└── resources/                # Optional: Config files, templates, data
    ├── config_template.yaml
    └── reference_data.json
```

## SKILL.md Format

The `SKILL.md` file must follow this standardized structure:

### 1. YAML Frontmatter
```yaml
---
name: skill-name-in-kebab-case
description: Concise one-sentence summary of the skill's purpose and outcome.
metadata:
  category: [category-name]
  venv: [cpu] # or mlip, fairchem, or a research stack (adit, diffcsp, mattergen, msms, reactot, scd)
---
```

**Only the six keys of the [Agent Skills](https://agentskills.io) spec are allowed at the top level** — `name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`. Claude Code tolerates extra keys, but claude.ai skill uploads and the Skills API reject them with a hard error, which would break importing this repository into Claude Science. Project-specific fields therefore go inside the free-form `metadata` map.

**Guidelines:**
- `name`: Use lowercase letters, numbers, and hyphens only (kebab-case)
- `description`: Should be clear enough for the agent to decide if this skill is relevant to a user query. **The description must state what the skill is used for, NOT how the skill works.** Avoid mid-sentence colons (`: `) in unquoted values — they break YAML parsing.
- `metadata.category`: Always a YAML list, even for a single value (e.g. `[materials]` or `[materials, chemistry]`). Must be drawn from:
  - `materials`: Materials science simulation and analysis skills (prefix: `mat-`)
  - `chemistry`: Chemistry and molecular simulation skills (prefix: `chem-`)
  - `machine-learning`: MLIP training, model selection, and ML workflows (prefix: `ml-`)
  - `drug-discovery`: Drug design, docking, and molecular property prediction (prefix: `drug-`)
  - `general`: General-purpose research utilities (prefix: `general-`)
- `metadata.venv`: A YAML list declaring the uv environments used: the shared projects (`[cpu]`, `[mlip]`, `[fairchem]`) or a research stack's own project under `venv/` (e.g. `[msms]`). A new stack that cannot share an environment gets its own pinned uv project rather than a conda environment.

### 2. Title and Goal Section
Begin with a level-1 header matching the skill name, followed by a `## Goal` section:

```markdown
# Skill Name

<!-- mcp-tools-note -->
> [!NOTE]
> Steps written `server.tool` are MCP tool calls: `server.tool` is the `tool`
> tool of the `server` server (`mcp__<server>__<tool>`, or
> `mcp__plugin_atomistic-skills_<server>__<tool>` when installed as a plugin).
> Without a connected server, run the same tools from the shell. Tools named in
> one command share a process, so a model loaded by `load_model` stays loaded:
>
> ```bash
> ${CLAUDE_SKILL_DIR}/../../venv/run <venv> python -m src.mcp_server.cli <server> <tool> key=value
> ```

## Goal
Clearly state what this skill achieves. Use precise technical language and, when applicable, include mathematical notation (e.g., "To determine the thermodynamic melting temperature ($T_m$) of a bulk material").
```

### 3. Instructions Section
Provide numbered, step-by-step instructions. Each step should:
- **State the objective clearly** (e.g., "Background Research", "Phase Preparation")
- **Provide specific commands** with the launcher invocation
- **Include all necessary parameters** with explanations
- **Link to related skills** when appropriate

**Format for code blocks:**
Skill commands use the self-locating launcher form:
````markdown
```bash
${CLAUDE_SKILL_DIR}/../../venv/run <venv>[+<extra>] python ${CLAUDE_SKILL_DIR}/scripts/<script>.py [arguments]
```
````
Or when executed from the repository root:
````markdown
```bash
venv/run <venv>[+<extra>] python skills/<skill-name>/scripts/<script>.py [arguments]
```
````

**Format for MCP tool calls:**
Write steps in `server.tool` notation (e.g., `matgl.relax_structure`, `base.search_materials_project_by_formula`):
````markdown
```bash
server.tool_name(
    parameter1=value,  # Comment explaining the parameter
    parameter2=value,  # Use realistic values, not placeholders
    output_dir="descriptive_name"
)
```
````

**Key principles:**
- Always specify the environment (`cpu`, `mlip`, `fairchem`) and any extras via `venv/run`
- Use absolute paths or `${CLAUDE_SKILL_DIR}` for script references
- Provide inline comments explaining non-obvious parameters
- Use realistic example values instead of generic placeholders
- Cross-reference other skills using relative links (e.g., `[mat-diffusion-analysis](../mat-diffusion-analysis/SKILL.md)`)

### 4. Examples Section
Provide concrete, runnable examples that demonstrate typical usage.

**CRITICAL RULE:** Each distinct example should be placed in its own dedicated subdirectory within `examples/` (e.g., `examples/my-example/`) and MUST contain its own `README.md` file. This README should comprehensively document the example's goal, step-by-step instructions, and expected outputs.

**LITERATURE VALIDATION RULE:** Whenever possible, choose an example system with known, published literature reported values. In the example's `README.md`, you MUST compare the skill's execution results to the reported literature values to validate the skill's correctness, and explicitly include the literature reference citation.

> [!WARNING]
> **Artifact Retention**: Example folders are purely for structural reference and lightweight logging. NEVER commit or retain large execution artifacts such as PyTorch model checkpoints (`.pth`, `.model`), checkpoint snapshots (`.pt`), or uncompressed trajectory aggregations (`.xyz`) inside these example subdirectories.

**3D STRUCTURE RENDERING RULE:** To ensure compatibility with the documentation website's automatic 3D structure viewer (3Dmol.js), any `.cif` or `.xyz` files you want rendered MUST be written as standard markdown links (e.g., `[my_structure.cif](my_structure.cif)`). Do NOT write them merely as backticked code snippets (` `my_structure.cif` `) within tables or lists, as they will not be rendered. It is recommended to add a dedicated "## 3D Structures" section at the end of the `README.md` for these links.


```markdown
## Examples

Creating a solid-liquid interface for Aluminum:
```bash
venv/run cpu python skills/mat-melting-point/scripts/create_interface.py Al_solid.cif Al_liquid.cif --axis 0 --output Al_interface.cif
```
```

### 5. Constraints Section
Document important limitations, safety rules, and requirements:

```markdown
## Constraints
- **Box Dimensions**: The lattice parameters perpendicular to the stacking axis must be identical.
- **Ensemble**: The final production run must be in the **NVE** ensemble.
- **Environments**: Scripts specify their required uv environment in frontmatter `metadata.venv` (`cpu`, `mlip`, `fairchem`, or a research stack) and execute via `venv/run <venv>[+<extra>]`.
- **System Size**: Recommended for supercells with >50 atoms to reduce noise.
```

### 6. References Section
Include basic literature citations (with DOIs if available) for the methods, algorithms, or software packages the skill relies on to ensure scientific reproducibility.

```markdown
## References
- Author et al., "Paper Title", *Journal Name*, Year. [DOI](https://doi.org/...)
```

## Script Documentation Standards

All Python scripts in the `scripts/` directory must include:

### 1. Module-Level Docstring
```python
"""
Brief description of what this script does.

Usage:
    python script_name.py input.cif --option value

Requirements:
    - Environment: <venv> (cpu, mlip, fairchem, or a research stack)
    - Required packages: ase, pymatgen, etc.
"""
```

### 2. Machine Path and Job Submission Safety
- **No hard-coded machine paths**: Scripts must never hard-code host machine paths (e.g. `/home/user/...`). Use arguments, relative paths, or resolve relative to `ATOMISTIC_WORKSPACE`.
- **No job submission by default**: Scripts must not submit jobs or call remote job runners (such as `jobflow-remote` or remote clusters) by default. They should write input files or flow manifests locally by default and submit only when explicit submission flags (e.g. `--submit`) are passed.

### 3. Argument Parser with Help Text
```python
parser = argparse.ArgumentParser(
    description="Clear description of the script's purpose"
)
parser.add_argument("input", help="Description of input file/parameter")
parser.add_argument("--option", default=default_value, help="What this option controls")
```

### 4. Type Hints and Docstrings for Functions
```python
def process_structure(atoms: Atoms, threshold: float = 0.5) -> dict:
    """
    Brief description of what the function does.

    Args:
        atoms: ASE Atoms object to process
        threshold: Cutoff value for some criterion

    Returns:
        Dictionary containing results with keys: 'metric1', 'metric2'
    """
```

## Environment Management Standards

### 1. Explicit Specification
Every skill must declare its environment in its `SKILL.md` frontmatter:
- `metadata.venv: [<venv>]` for standard uv environments (`cpu`, `mlip`, or `fairchem`). Commands use `${CLAUDE_SKILL_DIR}/../../venv/run <venv>[+<extra>] ...` (or `venv/run ...`).
- `metadata.conda_env: <name>` only where a build needs conda (`mat-lammps-md`, which compiles LAMMPS against each MLIP). Commands use `conda run -n <name> ...`.

### 2. Environment Mapping
Refer to `mcp-environments.md` for standard environment mapping:
- `cpu`: General materials tools, chemistry, drug discovery, MP API, parsing, analysis (no torch). MCP servers: `base`, `atomate2`, `drugdisc`, `smol`.
- `mlip`: MACE and MatGL (CHGNet, TensorNet, M3GNet) foundation models and training (GPU). MCP servers: `mace`, `matgl`.
- `fairchem`: FairChem (UMA, eSEN) models and training (GPU). MCP servers: `fairchem`.

### 3. Documentation Consistency
The required environment must be consistent across:
- The `metadata.venv` or `metadata.conda_env` frontmatter in `SKILL.md`.
- The `venv/run <venv>` invocation in command examples.
- The `Requirements` section in the script's module-level docstring.
- The `Constraints` section of `SKILL.md`.

## Best Practices

- **Scope & Progressive Disclosure**: Target one well-defined task. Focus `SKILL.md` on workflow logic; move implementation to `scripts/`, datasets to `resources/`, and examples to `examples/`.
- **References**: Always use relative paths from `SKILL.md` when linking to scripts or other skills.
- **Environments**: Always declare `metadata.venv` in frontmatter, invoke commands via `venv/run <venv>`, and specify requirements in the Constraints section.
- **Integration**: Prioritize existing MCP tools over custom code. If writing custom MLIP scripts, ALWAYS use `src.utils.mlips.loader.load_wrapper`.
- **Validation & Documentation**: Embed verification steps, document expected outcomes, and use concrete, reproducible parameters in examples.
- **Parameter Persistence**: Skill scripts that accept input kwargs and hyperparameters **must** save all input parameters to a **separate `input_configs.yaml`** file in the same output directory where results are written. This file must capture both user-specified values **and** the default values of any parameters that were not explicitly provided. Do **not** embed configs inside JSON output files (e.g. as a `"config"` key). This ensures results remain clean and can be fully interpreted and reproduced in the future without re-inspecting the source code or command history.

## Skill Naming Conventions

- **Purpose over Method**: Skill names should be informative of the *function or purpose* of the skill, NOT the specific computational method being used (e.g., `mat-solid-free-energy` is preferred over `mat-frenkel-ladd`).
- Use **kebab-case** for skill directory names (lowercase with hyphens)
- **Every skill name must start with a category prefix** matching its `metadata.category` field:
  - `mat-` for `materials` skills (e.g., `mat-melting-point`, `mat-diffusion-analysis`, `mat-phonon`)
  - `ml-` for `machine-learning` skills (e.g., `ml-foundation-potentials`, `ml-mlip-training`, `ml-cluster-expansion`)
  - `drug-` for `drug-discovery` skills (e.g., `drug-docking-vina`, `drug-admet-prediction`)
  - `general-` for `general` skills (e.g., `general-arxiv-search`)
- The part after the prefix should be **descriptive and concise**:
  - Good: `mat-melting-point`, `ml-mlip-speed`, `drug-protein-prep`
  - Avoid: `mat-mp`, `ml-train`, `drug-d`
- Use **noun forms** for result-oriented skills: `mat-phase-diagram`, `mat-surface-energy`
- Use **action/process names** for workflow skills: `mat-diffusion-analysis`, `ml-mlip-training`
- **Private Skills**: To create a proprietary or private skill that should not be tracked by version control, prefix the entire name with `private-` (e.g., `private-mat-proprietary-workflow`). The repository's `.gitignore` is configured to ignore all directories matching `skills/private-*/`, ensuring they remain local while still being automatically discovered by the agent.

## Example Skill Structure

See [`skills/melting-point/`](../../skills/melting-point/) for a comprehensive reference implementation demonstrating workflows, tool integration, validation, and environment handling.

### 7. Author Information

At the very end of every `SKILL.md` file, include a footer separated by a horizontal rule (`---`).

**CRITICAL NOTE:** The author and contact cannot be an AI agent. It must be the human author who initiated and pushed this change.

**GitHub contact (preferred):**
```markdown
---

**Author:** Name
**Contact:** [GitHub @username](https://github.com/username)
```

**Email contact:**
```markdown
---

**Author:** Name
**Contact:** [name@example.com](mailto:name@example.com)
```
