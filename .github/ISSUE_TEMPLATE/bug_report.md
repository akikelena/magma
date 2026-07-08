---
name: Bug report
about: Something in MAGMA doesn't work as documented
title: ''
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what's wrong.

**To reproduce**
Command(s) run, or a minimal code snippet, that shows the problem. Include the config
(`config/*.yaml` overrides, CLI flags) if relevant.

**Expected behavior**
What you expected to happen instead.

**Environment**
- MAGMA commit/version:
- Python version:
- Optional dependency groups installed (`crewai`, `ml`, `observability`, `web`):
- Neo4j/Weaviate versions, if using the real backends:

**Logs / traceback**
Paste the relevant output. If it's from `run_workflow`/`run_doc_watch`, `--json` output plus
stderr warnings is usually enough.

**Additional context**
Anything else that might be relevant (a specific vendor's documentation, a particular template,
etc.).
