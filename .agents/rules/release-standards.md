---
trigger: model_decision
description: Rules for preparing a release tag for this repository
---

# Release Standards

Follow these steps when preparing a release tag.

## Version Numbering

Use [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`

- **PATCH** (`v1.0.x`): bug fixes, documentation updates, no new skills or tools
- **MINOR** (`v1.x.0`): new skills, new MCP tools, or new workflows added
- **MAJOR** (`v2.0.0`): breaking changes to the framework, tool API, or skill interface

## Pre-Tag Checklist

0. Bump the version and propagate it to every published manifest. `VERSION` at
   the repository root is the single source; `.claude-plugin/plugin.json`,
   `.claude-plugin/marketplace.json` and `server.json` are derived from it.
   The tag must match `VERSION` exactly.
   ```bash
   conda run -n base-agent python tools/sync_version.py --set 1.x.y
   conda run -n base-agent python tools/sync_version.py --check
   ```
   > CI runs `--check` on every push, so a tag can never ship with the four
   > files disagreeing. Before `VERSION` existed, `server.json` had drifted to
   > `1.0.0` while the repository was tagged `v1.3.4`.

   If `docker/images.json` changed in this release, re-render the plugin's MCP
   wiring as well, since the image tags follow the version:
   ```bash
   conda run -n base-agent python docker/render.py plugin-mcp
   ```

1. Rebuild the doc site to get accurate public counts:
   ```bash
   conda run -n base-agent python site/build_skills.py
   ```
2. Read counts from `site/skills_index.js` and git (source of truth):
   ```bash
   # skills, tools, servers (= conda-env dirs) — from rebuilt index
   python3 -c "
   import re
   content = open('site/skills_index.js').read()
   for key in ['skills', 'tools', 'servers']:
       m = re.search(r'\"' + key + r'\"\s*:\s*(\d+)', content)
       print(key, m.group(1))
   "

   # workflows — NOT in skills_index.js; count directly from disk
   ls .agents/workflows/*.md | wc -l
   ```

   > **Note**: Workflows live as plain `*.md` files in `.agents/workflows/` (e.g. `drug-hit-finding-htvs.md`), **not** as `WORKFLOW.md`. Always use `ls .agents/workflows/*.md | wc -l` for the current count and `git ls-tree -r <tag> --name-only | grep "workflows/.*\.md" | wc -l` for the previous-tag count.
3. Ensure all pre-commit hooks pass on HEAD.
4. Confirm the MCP server does not expose built-in agent tools (e.g. `task_boundary`, `notify_user`).

## Tag Message Format

The annotated tag message must follow this structure exactly:

```
## What's Changed

### Repository Stats

| Component    | vPREV  | vNEW   | Added |
|--------------|--------|--------|-------|
| Skills       | <n>    | <n>    | +N or — |
| Workflows    | <n>    | <n>    | +N or — |
| MCP Tools    | <n>    | <n>    | +N or — |
| Tool Servers | <n>    | <n>    | +N or — |

### New Skills   ← omit section if none

| Skill | Description | Author |
|-------|-------------|--------|
| `<skill-id>` | One-sentence description from SKILL.md frontmatter. | @github-handle |

### New Workflows   ← omit section if none

| Workflow | Description | Author |
|----------|-------------|--------|
| `<workflow-id>` | One-sentence description from WORKFLOW.md frontmatter. | @github-handle |

### New MCP Tools   ← omit section if none

| Tool | Server | Description | Author |
|------|--------|-------------|--------|
| `<tool_name>` | `<server_id>` | One-sentence description from docstring. | @github-handle |

### Other Highlights
- **<Area>**: concise bullet per notable fix or improvement
```

### Field rules

- **Stats table**: use counts from `site/skills_index.js` (public skills only). Write `—` when a count did not change.
- **New Skills table**: one row per skill whose `SKILL.md` was *added* (not just modified) since the previous tag. Use the `description:` frontmatter value, truncated to one sentence. Author is the git committer of the adding commit (`git show --format="%an" <sha>`).
- **New Workflows table**: one row per workflow `*.md` file *added* to `.agents/workflows/` (not just modified) since the previous tag. Use the `description:` frontmatter value, truncated to one sentence. Author is the git committer of the adding commit. Detect with: `git log <prev>..HEAD --diff-filter=A --name-only --pretty="" -- ".agents/workflows/*.md"`
- **New MCP Tools table**: one row per `@mcp.tool()`-decorated function *added* since the previous tag. Exclude built-in agent tools (tools that belong to the harness, not the MCP server). Server is the `*_server.py` basename without `_server`. Author is the git committer of the adding commit.
- **Other Highlights**: group by area (Skills, Sorption, Drug discovery, Docs & CI, etc.). One bullet per logical change, not per commit.

## Creating the Tag and GitHub Release

1. Create the annotated tag using `--cleanup=verbatim` to preserve Markdown headings (which start with `#`):
   ```bash
   git tag --cleanup=verbatim -a v1.x.y -m "$(cat <<'EOF'
   <paste message above>
   EOF
   )"
   git push origin v1.x.y
   ```

   > **Note**: Always use an annotated tag (`-a`) so the message is stored in the tag object. Specifying `--cleanup=verbatim` is required so that git does not strip the Markdown headers (`#`) as comment lines.

2. Create the release wrapper on GitHub through the GitHub CLI (`gh`), piping the tag's contents as the release body:
   ```bash
   git tag -l --format='%(contents)' v1.x.y | gh release create v1.x.y -t "v1.x.y" -F -
   ```

3. Validate that the release is online by checking the release URL using `curl` (ensuring it returns a `200 OK` status):
   ```bash
   curl -I -L -s https://github.com/learningmatter-mit/AtomisticSkills/releases/tag/v1.x.y | head -n 10
   ```
