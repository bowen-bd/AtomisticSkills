---
name: general-atomisticskills-rules
description: Working rules for atomistic research with AtomisticSkills -- how to scope a request, set up a research directory and a plan before simulating, run skills and MCP tools, stay within GPU memory, and report results. Read it before any multi-step simulation, screening, fine-tuning or benchmarking task.
metadata:
  category: [general]
  venv: []
---

# AtomisticSkills Working Rules

## Goal

Run research tasks the way the AtomisticSkills project expects: scoped, planned,
reproducible and safe for the machine. These are the project's agent rules in
skill form, because a plugin install does not load the repository's own agent
instructions.

## 1. Classify the request first

- **Direct property query**: one value or one structure, e.g. "lattice constant
  of Ni", "structure of aspirin". Answer with the matching tool or skill
  directly; no research directory, no plan.
- **Computational research task**: new data, simulation, screening,
  benchmarking or fine-tuning. Follow the research protocol in section 2.
- **Literature synthesis**: pros and cons, state of the art. Answer in the
  chat from the literature, then offer to start a computational project that
  verifies the claims for a specific system.

## 2. Research protocol

1. **Research directory.** Call `base.create_research_dir` with a short
   snake_case topic (e.g. `alloy_melting_point`). It creates
   `research/<date>_<topic>/` in the current project and makes it the default
   output location of every tool and script. Keep all structures, logs, models
   and figures of the task there, never inside the skill or plugin directory.
2. **Plan before simulating.** Write `research_plan.md` in the research
   directory with:
   - literature takeaways (`general-query-literature-database` skill);
   - a short methodology abstract;
   - the skills you will use, and the capabilities no skill covers;
   - a chronological action plan with concrete parameters: model and
     checkpoint, functional, ensemble, temperature, timestep, steps, `fmax`,
     k-points, and so on.

   Before planning a fine-tuning step, check `base.search_model_registry` for an
   existing model for the chemical system.
3. **Get approval.** Show the plan to the user and wait for approval or comments
   before running any simulation.
4. **Reuse before writing.** Prefer existing skills and MCP tools. Write a new
   script only for missing functionality, and put it in the research directory.

## 3. Running skills and MCP tools

- Run skill commands exactly as the skill writes them
  (`${CLAUDE_SKILL_DIR}/../../venv/run <env> ...`). The launcher selects and, on
  first use, creates the environment. Do not activate environments or call
  `python` directly.
- MCP tools are written `server.tool`. Through the plugin, the tool is named
  `mcp__plugin_atomistic-skills_<server>__<tool>`. When no server is connected,
  use the shell fallback given in the skill's note.
- When a tool or command fails, read the error and fix its cause: the input, the
  model name, or the environment (see the `general-atomisticskills-setup`
  skill). Do not quietly replace a failing tool with a hand-written substitute;
  tell the user what failed and why.
- Ask the user before installing, upgrading or removing packages, and before
  submitting jobs to a cluster or a workflow manager (atomate2, jobflow).

## 4. Compute safety

- Before an expensive GPU run (training, long or large MD, batched inference
  over many structures), estimate peak memory and compare it with the free GPU
  memory (`nvidia-smi`). Account for atoms × neighbors × feature width, and
  roughly ×3 for training. On machines whose CPU and GPU share memory (NVIDIA
  GB10, Grace Hopper), running out can freeze the whole machine, not only the
  job.
- Start with a small batch or system, confirm memory is stable, then scale up.
  If a design fits only near the memory limit, change the design.
- Match the model to the cost of the task (`ml-foundation-potentials`). Pass
  many small structures as one list so the batched GPU path is used.

## 5. Units and reporting

- Use ASE units throughout: eV, Å, fs, eV/Å for forces and eV/Å³ for stress
  (1 eV/Å³ = 160.2 GPa). Conversions are in `general-property-units`.
- For every reported number, give the model or checkpoint, the parameters and
  the path of the raw output that produced it.
- Look at every figure you generate before presenting it.
- MLIP energies inherit their training functional (PBE, r2SCAN). Do not mix
  energies from different functionals without corrections
  (`mat-dft-mixing-functionals`).
