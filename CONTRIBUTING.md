# Contributing to MAGMA

## Setup

```bash
git clone <this repo>
cd magma
poetry install                                    # core + dev tools (pytest, flake8, mypy, black, bandit)
poetry install --with crewai,ml,observability,web  # add the optional agent backends
cp .env.example .env                               # only needed for live Neo4j/Weaviate/LLM runs; the test suite doesn't require it
```

`crewai`/`ml`/`observability`/`web` are `optional = true` groups in `pyproject.toml` -- `poetry
install` with no `--with` flag skips them, matching the "every agent runs and is tested without
them installed" design the whole codebase follows (lazy imports at each call site).

Without Poetry:

```bash
pip install -e . requests prometheus-client
pip install pytest pytest-cov flake8
```

## Running the test suite

```bash
pytest magma/tests/unit magma/tests/integration
```

The full suite runs with zero external services or API keys: every agent defaults to an
in-memory graph/vector store and `MockLLMProvider`. Tests for the optional backends (neo4j,
weaviate-client, crewai, requests) skip themselves via `pytest.importorskip(...)` when that
package isn't installed rather than failing -- CI doesn't install the `ml`/`crewai` groups, so
those tests only run if you've installed them locally.

## Linting

```bash
flake8 magma
```

`.flake8` sets `max-line-length = 200` and ignores `E127`/`E306`, since this codebase
deliberately favors compact, dense definitions (dataclasses, asserts, argument lists) over
one-item-per-line formatting -- real issues (unused imports, undefined names, etc.) still fail.
There's no `black --check` gate in CI: running black across the repo would reformat nearly
every file to a much more spread-out style than what's here now, which is a deliberate choice,
not an oversight. If you use black locally, expect its output to disagree with the existing
style; that's fine for your own editing, just don't commit a repo-wide reformat.

## Making changes

- Keep PRs scoped to one logical change; a PR that mixes a bug fix with an unrelated feature is
  harder to review and to revert independently if something breaks.
- Add or update tests for anything you change in `magma/`. A change without a test covering it
  didn't happen as far as CI is concerned.
- Match the existing comment style: senior-engineer-terse, explaining *why* a non-obvious
  decision was made, not narrating *what* the code below it already says.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, ...) with a scope in parentheses when it helps
  (`fix(phase1): ...`).
- New optional dependencies belong in the right `[tool.poetry.group.*]` in `pyproject.toml`,
  imported lazily inside the function/class that needs them (see `HttpDocumentFetcher`,
  `AnthropicProvider`, `Neo4jGraphStore` for the pattern) so the core test suite keeps running
  without them installed.

## Architecture

See `docs/architecture/agent_cards.md` for the modality, pre/postconditions, mechanism, and
failure behavior of every agent, and the README's section headers for how the pieces (Docker,
CrewAI orchestration, the human review queue, documentation watching) fit together.

## Reporting bugs / requesting features

Open a GitHub issue using the provided templates. For security vulnerabilities, see
`SECURITY.md` instead of opening a public issue.
