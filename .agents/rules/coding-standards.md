---
trigger: always_on
---

# AtomisticSkills Agent Project Coding Standards

These rules are universally applied across all aspects of this project. **You MUST ALWAYS FOLLOW them WITHOUT EXCEPTION.** Keep all processes clean, modular, and reliant on existing infrastructure.

## 1. Global Agent Guidelines
1. **Language**: All code and comments must be in **English**.
2. **Temporary Files**: All temporary log, validation, testing, and summary files must be created under `<project_root>/.agents/test` to maintain a clean project structure.
3. **Error Handling**: Avoid using `try...except` blocks unless absolutely necessary; ensure the code output expectation is static and predictable.
4. **Cleanup**: After implementation changes, clean up temporary test code for previous deprecated functions.
5. **Reusability**: Before implementing a new function, search whether the same function already exists in dependency packages. Do not reinvent the wheel.
6. **Imports**: Use **absolute imports** for Python projects.
7. **MCP Stability**: If there's any bug with MCP tool calling, pause any ongoing research and **debug the MCP tool**. Do not write custom scripts to bypass MCP tool failures. Debug and ensure the MCP tool is stable for production use.
8. **MCP Edits**: After modifying MCP files, ask the user to refresh to make the changes in MCP servers effective.
9. **Package Management**: Before installing, upgrading, downgrading, or removing any package, **ask for permission**.
10. **URL Validation**: After generating any URLs, always test whether the URL is valid by verifying it directly (e.g., curling).

## 2. Environment and Dependency Management
- **Environment Isolation**: You must use the appropriate environment (`cpu`, `mlip`, `fairchem`) via the `venv/run` launcher depending on the task (see `.agents/rules/mcp-environments.md`). The `cpu` environment serves as the generic fallback for standard processing without torch. For remaining standalone conda environments (e.g., generative models), use the declared environment in `metadata.conda_env`.
- **Dependency Versions**: Dependencies are rigorously managed per uv project in `venv/{cpu,mlip,fairchem}/pyproject.toml` and committed `uv.lock`. Never arbitrarily force or install package versions globally. Optional extras (`openmm`, `docking`, `pymol`, `void`, `transport`) handle system-dependent packages.
- **Failures**: NEVER implement fallback functions when package imports fail. Debug the root cause and fix the environment configuration.


## 3. Documentation and Testing
- **Structure**: Create independent Python packages for each major functional module.
- **Docs**: Provide detailed docstrings (including type hints) for functions and classes. Modules require a `README.md`, and skills strictly follow `@.agents/rules/skill-standards.md`.
- **Testing**: MCP tool functionalities must include `pytest`-based unit tests placed in the `tests/` directory.

## 4. Security and Performance
- **Secrets**: All API keys must be passed through environment variables or local `.env` files. Never hardcode credentials.
- **Efficiency**: Use streaming processing for files with large memory footprints. Monitor memory usage and optimize iterative loops to prevent OOM errors.
- **Debugging Blocks**: Pause immediately if you encounter an "improper format stop reason" error to analyze the trigger condition.
